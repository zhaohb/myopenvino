// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_deconv_node.h"
#include <legacy/ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <legacy/ie_layers_internal.hpp>
#include "ie_parallel.hpp"
#include "utils/general_utils.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNDeconvolutionNode::MKLDNNDeconvolutionNode(const InferenceEngine::CNNLayerPtr& layer,
                                                 const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(layer, eng, cache) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_desc(0));
    });
}

void MKLDNNDeconvolutionNode::getSupportedDescriptors() {
    if (!descs_fwd.empty() && !descs_bwd.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32 && precision != InferenceEngine::Precision::BF16)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32 && precision != InferenceEngine::Precision::BF16)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    if (inputDataType == memory::data_type::bf16 || outputDataType == memory::data_type::bf16)
       inputDataType = outputDataType = memory::data_type::bf16;

    if (getParentEdges().empty() || getParentEdges().size() > 3)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    auto * deconvLayer = dynamic_cast<DeconvolutionLayer*>(getCnnLayer().get());
    if (deconvLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert deconvolution layer.";
    if (getParentEdges().size() == 1 && deconvLayer->_weights == nullptr) {
        THROW_IE_EXCEPTION << "Weights are empty for layer: " << deconvLayer->name
                           << " used in MKLDNN node: " << getName() << "\n"
                           << "Use the second argumemt of InferenceEngine::Core::ReadNetwork"
                           << " to load them from .bin part of the IR";
    }
    withGroups = (deconvLayer->_group > 1);
    isDW = withGroups && deconvLayer->_group == deconvLayer->_out_depth &&
            deconvLayer->_group == deconvLayer->input()->getDims()[1];

    bool withBiases = (deconvLayer->_biases != nullptr && deconvLayer->_biases->size() != 0) || getParentEdges().size() == 3;
    if (withBiases) {
        Blob::Ptr biases;

        if (getParentEdges().size() == 3) {
            auto biasLayer = getParentEdgesAtPort(2)[0]->getParent()->getCnnLayer();
            if (biasLayer->type != "Const")
                THROW_IE_EXCEPTION << "Deconvolution layer with name '" << getName() << "' doesn't support non-constant biases";
            biases = biasLayer->blobs["custom"];
        } else {
            biases = deconvLayer->_biases;
        }

        //  WA: we add bias as depthwise post op
        setBiasAsPostOp(biases);
    }

    /* Original layout format for deconv weights is iohw (from Caffe).
     * We specify oihw, but mean iohw, because there are no more
     * suitable format in MKLDNN.
     */
    SizeVector weightDims;
    if (withGroups) {
        weightDims = {
                deconvLayer->_group,
                deconvLayer->input()->getTensorDesc().getDims()[1] / deconvLayer->_group,
                deconvLayer->_out_depth / deconvLayer->_group,
        };
        groupNum = deconvLayer->_group;
    } else {
        weightDims = {
                deconvLayer->input()->getTensorDesc().getDims()[1],
                deconvLayer->_out_depth
        };
    }
    for (int i = 1; i <= deconvLayer->_kernel.size(); i++) {
        weightDims.push_back(deconvLayer->_kernel[deconvLayer->_kernel.size() - i]);
    }

    if (getParentEdges().size() == 1)
        internalBlobs.push_back(createInternalBlob(weightDims, true));

    invertVectorCopyUtoI(deconvLayer->_stride, stride);
    for (int i = 1; i <= deconvLayer->_dilation.size(); i++) {
        dilation.push_back(static_cast<int>(deconvLayer->_dilation[deconvLayer->_dilation.size() - i]) - 1);
    }
    auto allPads = getPaddings(*deconvLayer);
    invertVectorCopyUtoI(allPads.begin, paddingL);
    invertVectorCopyUtoI(allPads.end, paddingR);

    weightsDims = MKLDNNDims(weightDims);

    for (int i = 0; i < paddingR.size(); i++) {
        int with_group = (withGroups) ? 1 : 0;
        int krn = weightsDims[with_group + 2 + i];
        int src = getChildEdgeAt(0)->getDims()[2 + i];
        int dst = getParentEdgeAt(0)->getDims()[2 + i];

        krn = (krn - 1)*(dilation[i] + 1) + 1;
        int calc_dst = (src - krn + paddingL[i]) / stride[i] + 1;
        paddingR[i] = (dst - calc_dst) * stride[i];
    }

    for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
        MKLDNNMemoryDesc in_candidate(getParentEdgeAt(0)->getDims(), inputDataType, format);
        MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType, format);
        createDescriptor({in_candidate}, {out_candidate});
    }
}

void MKLDNNDeconvolutionNode::setBiasAsPostOp(const InferenceEngine::Blob::Ptr& biases) {
    mkldnn::post_ops ops;
    auto depthwiseSize = static_cast<ptrdiff_t>(rnd_up(biases->size(), 16));

    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
    PostOpsIntBlobMemory[0]->Create({depthwiseSize}, memory::data_type::f32, memory::format_tag::x);
    PostOpsIntBlobMemory[0]->FillZero();
    std::vector<float> weights(depthwiseSize, 1.0f);
    std::fill(weights.begin() + biases->size(), weights.end(), 0.0f);
    PostOpsIntBlobMemory[0]->SetData(memory::data_type::f32, memory::format_tag::x, weights.data(),
            weights.size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
    PostOpsIntBlobMemory[1]->Create({depthwiseSize}, memory::data_type::f32, memory::format_tag::x);
    PostOpsIntBlobMemory[1]->FillZero();
    auto biases_ptr = biases->buffer().as<float*>();
    std::vector<float> bias(depthwiseSize, 0.0f);
    std::copy(biases_ptr, biases_ptr + biases->size(), bias.begin());
    PostOpsIntBlobMemory[1]->SetData(memory::data_type::f32, memory::format_tag::x, bias.data(),
             bias.size() * MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

    ops.append_depthwise(algorithm::depthwise_scale_shift,
                         (const float *) PostOpsIntBlobMemory[0]->GetData(),
                         (const float *) PostOpsIntBlobMemory[1]->GetData());

    attr.set_post_ops(ops);
}

void MKLDNNDeconvolutionNode::filterSupportedPrimitiveDescriptors() {
    MKLDNNNode::filterSupportedPrimitiveDescriptors();
    filterSupportedDescriptors();
}

void MKLDNNDeconvolutionNode::filterSupportedDescriptors() {
    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty()) {
        if (inputMemoryFormatsFilter.size() > 1 || outputMemoryFormatsFilter.size() > 1) {
            THROW_IE_EXCEPTION << "Incorrect number of input or output memory formats for Deconvolution node";
        }
        auto itd = descs.begin();
        while (itd != descs.end()) {
            bool isSuitableDesc = true;
            if (!inputMemoryFormatsFilter.empty()) {
                auto src_tdesc = MKLDNNMemoryDesc(std::shared_ptr<mkldnn::convolution_backward_data::desc>(*itd)->data.diff_src_desc);
                isSuitableDesc &= src_tdesc.isSame(inputMemoryFormatsFilter[0]);
            }
            if (!outputMemoryFormatsFilter.empty()) {
                auto dst_tdesc = MKLDNNMemoryDesc(std::shared_ptr<mkldnn::convolution_backward_data::desc>(*itd)->data.diff_dst_desc);
                isSuitableDesc &= dst_tdesc.isSame(outputMemoryFormatsFilter[0]);
            }
            if (!isSuitableDesc) {
                itd = descs.erase(itd);
            } else {
                itd++;
            }
        }
    }
}

bool MKLDNNDeconvolutionNode::created() const {
    return getType() == Deconvolution;
}

void MKLDNNDeconvolutionNode::createPrimitive() {
    if (prim)
        return;

    auto prim_desc = createPrimitiveDescriptor<convolution_backward_data::primitive_desc,
            convolution_backward_data::desc, convolution_forward::primitive_desc>(attr);

    prim.reset(new convolution_backward_data(prim_desc));

    auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    primArgs = {{DNNL_ARG_DIFF_DST, src}, {DNNL_ARG_WEIGHTS, getWeights()}, {DNNL_ARG_DIFF_SRC, dst}};
}

void MKLDNNDeconvolutionNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                               const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNMemoryDesc out_candidate(outputDesc[0]);

    // grouping and autoblicking is not compatible
    if ((withGroups && !isDW) && (in_candidate.blocksExtended() || out_candidate.blocksExtended()))
        return;

    MKLDNNMemoryDesc wgh_candidate{weightsDims, in_candidate.getDataType(), memory::format_tag::any};
    for (auto alg : {algorithm::convolution_winograd, algorithm::convolution_direct}) {
        auto convert = [] (const std::vector<ptrdiff_t>& orig_dims) {
            return memory::dims(orig_dims.begin(), orig_dims.end());
        };

        std::shared_ptr<mkldnn::convolution_forward::desc> conv_desc;
        conv_desc.reset(new convolution_forward::desc(prop_kind::forward_inference, alg,
                                                      out_candidate, wgh_candidate, in_candidate,
                                                      convert(stride),
                                                      convert(dilation),
                                                      convert(paddingL),
                                                      convert(paddingR)));

        std::shared_ptr<mkldnn::convolution_backward_data::desc> deconv_desc;
        deconv_desc.reset(new convolution_backward_data::desc(alg, out_candidate, wgh_candidate,
                                                    in_candidate,
                                                    convert(stride),
                                                    convert(dilation),
                                                    convert(paddingL),
                                                    convert(paddingR)));
        descs_fwd.push_back(conv_desc);
        descs_bwd.push_back(deconv_desc);

        auto fwd_conv_pd = std::make_shared<convolution_forward::primitive_desc>(*conv_desc, getEngine(), true);
        if (fwd_conv_pd->get(true) == nullptr)
            continue;

        descs.emplace_back(deconv_desc, fwd_conv_pd);
    }
}

MKLDNNMemoryDesc MKLDNNDeconvolutionNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = idx > 0 ? MKLDNNMemoryDesc(primitive_desc_it.weights_desc(idx - 1))
                                               : MKLDNNMemoryDesc(primitive_desc_it.diff_dst_desc(idx));

    if (desc.getLayout() == InferenceEngine::Layout::ANY) {
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    } else {
        if (getParentEdgeAt(idx)->getDims().ToSizeVector().size() != *std::max_element(desc.getBlockingDesc().getOrder().begin(),
                                                                                       desc.getBlockingDesc().getOrder().end()) + 1) {
            auto old_dims = getParentEdgeAt(idx)->getDims().ToSizeVector();
            auto new_dims = weightsDims.ToSizeVector();

            auto td = InferenceEngine::TensorDesc(desc.getPrecision(),
                                                  new_dims,
                                                  desc.getBlockingDesc());
            if (new_dims.size() == desc.getBlockingDesc().getBlockDims().size()) {
                td.setLayout(BLOCKED);
            }
            return MKLDNNMemoryDesc(td);
        } else {
            return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                                getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                                desc.getBlockingDesc()));
        }
    }
}

MKLDNNMemoryDesc MKLDNNDeconvolutionNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.diff_src_desc(idx));
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
}

const mkldnn::memory& MKLDNNDeconvolutionNode::getWeights() const {
    return getParentEdges().size() > 1 ? getParentEdgeAt(1)->getMemory().GetPrimitive() : internalBlobMemory[0]->GetPrimitive();
}

InferenceEngine::Precision MKLDNNDeconvolutionNode::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == MKLDNNEdge::Status::Validated) {
            inputPrecisions.emplace_back(MKLDNNExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }

    return MKLDNNExtensionUtils::getMaxPrecision(inputPrecisions);
}

REG_MKLDNN_PRIM_FOR(MKLDNNDeconvolutionNode, Deconvolution);
