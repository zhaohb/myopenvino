// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_pooling_node.h"

#include "mkldnn_quantize_node.h"
#include "mkldnn_conv_node.h"
#include "mkldnn_concat_node.h"
#include <legacy/ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <legacy/ie_layers_internal.hpp>
#include <utils/general_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNPoolingNode::MKLDNNPoolingNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

std::vector<memory::format_tag> MKLDNNPoolingNode::getAvailableFormatsForDims(const MKLDNNDims &dims) const {
    if (dims.ndims() == 0)
        return {memory::format_tag::x};
    else if (dims.ndims() == 1)
        return {memory::format_tag::x};
    else if (dims.ndims() == 2)
        return {memory::format_tag::nc};
    else if (dims.ndims() == 3)
        return {memory::format_tag::tnc, memory::format_tag::ntc};
    else if (dims.ndims() == 4)
        return {memory::format_tag::nChw8c, memory::format_tag::nChw16c, memory::format_tag::nhwc, memory::format_tag::nchw};
    else if (dims.ndims() == 5)
        return {memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c, memory::format_tag::ndhwc, memory::format_tag::ncdhw};
    return {memory::format_tag::any};
}

void MKLDNNPoolingNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    auto * poolingLayer = dynamic_cast<PoolingLayer*>(getCnnLayer().get());
    if (poolingLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert pooling layer.";

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    type = poolingLayer->_type;
    exclude_pad = poolingLayer->_exclude_pad;

    inputPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
    outputPrecision = getCnnLayer()->outData[0]->getPrecision();
    // Dirty WA to support stat based quantization approach
    if (this->getCnnLayer()->precision != Precision::I8
        && inputPrecision != Precision::BF16) {
        if (type == PoolingLayer::MAX) {
            // MKLDNN supports only equal precisions for input and output
            outputPrecision = inputPrecision;
        } else if (type == PoolingLayer::AVG) {
            outputPrecision = Precision::FP32;
        }
    }
    if (inputPrecision == Precision::BF16) {
        outputPrecision = inputPrecision;
    }

    if (!fusedWith.empty()) {
        auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
        if (lastFusedLayer) {
            outputPrecision = lastFusedLayer->outData[0]->getPrecision();
        }
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    invertVectorCopyUtoI(poolingLayer->_stride, stride);
    invertVectorCopyUtoI(poolingLayer->_kernel, kernel);
    auto allPads = getPaddings(*poolingLayer);
    invertVectorCopyUtoI(allPads.begin, data_pad_begin);
    invertVectorCopyUtoI(allPads.end, data_pad_end);
    effective_pad_begin = data_pad_begin;
    effective_pad_end.resize(data_pad_end.size());

    auto parentDims = getParentEdgeAt(0)->getDims();
    auto childDims = getChildEdgeAt(0)->getDims();
    if ((parentDims.ndims() < 4) || (parentDims.ndims() > 5))
        THROW_IE_EXCEPTION << "Pooling layer. Unsupported mode. Only 4D and 5D blobs are supported as input.";

    for (int i = 0; i < effective_pad_end.size(); i++) {
        int krn = kernel[i];
        int src = getParentEdgeAt(0)->getDims()[2 + i];
        int dst = getChildEdgeAt(0)->getDims()[2 + i];

        int calc_dst = (src - krn + data_pad_begin[i]) / stride[i] + 1;
        effective_pad_end[i] = (dst - calc_dst) * stride[i];
    }
    if (inputPrecision == Precision::I8 || inputPrecision == Precision::U8) {
        // i8 layers supports only ndhwc and nhwc layouts
        MKLDNNMemoryDesc in_candidate{parentDims, inputDataType, parentDims.ndims() == 5 ? memory::format_tag::ndhwc : memory::format_tag::nhwc};
        MKLDNNMemoryDesc out_candidate{childDims, outputDataType, parentDims.ndims() == 5 ? memory::format_tag::ndhwc : memory::format_tag::nhwc};
        createDescriptor({ in_candidate }, { out_candidate });
    } else if ((parentDims.ndims() == 4 || parentDims.ndims() == 5) && parentDims[1] == 1) {
        // WA. We should force planar layout since it provides better performance
        MKLDNNMemoryDesc in_candidate{parentDims, inputDataType, parentDims.ndims() == 5 ? memory::format_tag::ncdhw : memory::format_tag::nchw};
        MKLDNNMemoryDesc out_candidate{childDims, outputDataType, parentDims.ndims() == 5 ? memory::format_tag::ncdhw : memory::format_tag::nchw};
        createDescriptor({ in_candidate }, { out_candidate });
    } else {
        if (inputDataType != memory::data_type::bf16) {
            inputDataType = memory::data_type::f32;
            outputDataType = memory::data_type::f32;
        }
        // It doesn't support any format
        for (auto format : getAvailableFormatsForDims(parentDims)) {
            MKLDNNMemoryDesc in_candidate{parentDims, inputDataType, format};
            MKLDNNMemoryDesc out_candidate{childDims, outputDataType, format};
            createDescriptor({in_candidate}, {out_candidate});
        }
    }
}

void MKLDNNPoolingNode::createPrimitive() {
    if (prim)
        return;

    mkldnn::primitive_attr attr;
    setPostOps(attr, true);

    auto prim_desc = createPrimitiveDescriptor<pooling_forward::primitive_desc, pooling_forward::desc>(attr);

    prim.reset(new pooling_forward(prim_desc));

    auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}};
}

bool MKLDNNPoolingNode::created() const {
    return getType() == Pooling;
}

void MKLDNNPoolingNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                         const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNMemoryDesc out_candidate(outputDesc[0]);

    algorithm alg;
    if (type == PoolingLayer::PoolType::AVG) {
        bool not_zero_l = false;
        for (auto lr : data_pad_begin) {
            if (lr) {
                not_zero_l = true;
                break;
            }
        }
        bool not_zero_r = false;
        for (auto pr : data_pad_end) {
            if (pr) {
                not_zero_r = true;
                break;
            }
        }
        if (!exclude_pad && (not_zero_l || not_zero_r))
            alg = algorithm::pooling_avg_include_padding;
        else
            alg = algorithm::pooling_avg_exclude_padding;
    } else if (type == PoolingLayer::PoolType::MAX) {
        alg = algorithm::pooling_max;
    } else {
        // TODO: Handle rest of the possible: STOCH, ROI, SPACIAL_PYRAMID
        THROW_IE_EXCEPTION << "Unsupported pooling type";
    }

    auto convert = [] (std::vector<ptrdiff_t> orig_dims) {
        return memory::dims(orig_dims.begin(), orig_dims.end());
    };
    std::shared_ptr<pooling_forward::desc> desc_ptr(
            new pooling_forward::desc(prop_kind::forward_scoring, alg,
                                      in_candidate, out_candidate,
                                      convert(stride),
                                      convert(kernel),
                                      convert(effective_pad_begin),
                                      convert(effective_pad_end)));

    if (alg == algorithm::pooling_avg_include_padding) {
        // In case of AVG including paddings the norm coeff should be calculated
        // with tacking into account original pads. So we need to restore
        // original values for end paddings.
        //
        // WA. Because mkldnn uses different formula to calculate AVG norm coeff
        //     in compare with Caffe. In mkldnn coeff is always 1/(KH*KW)
        for (int i = 0; i < data_pad_end.size(); i++) {
            if (data_pad_end[i] != effective_pad_end[i])
            desc_ptr->data.padding[1][i] = static_cast<ptrdiff_t>(data_pad_end[i]);
        }
    }

    descs.emplace_back(desc_ptr);
}

void MKLDNNPoolingNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    mkldnn::primitive_attr attr;
    setPostOps(attr);

    for (auto& desc : descs) {
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
        while (static_cast<bool>(itpd)) {
            InferenceEngine::LayerConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(getSrcMemDesc(itpd, i));
                config.inConfs.push_back(dataConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(getDstMemDesc(itpd, i));
                config.outConfs.push_back(dataConfig);
            }
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}

void MKLDNNPoolingNode::initDescriptor(const InferenceEngine::LayerConfig &config) {
    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }
    std::vector<InferenceEngine::TensorDesc> inDescs;
    for (const auto& inConf : config.inConfs)
        inDescs.push_back(inConf.desc);
    std::vector<InferenceEngine::TensorDesc> outDescs;
    for (const auto& outConf : config.outConfs)
        outDescs.push_back(outConf.desc);
    createDescriptor({inDescs}, {outDescs});

    mkldnn::primitive_attr attr;
    setPostOps(attr);

    InferenceEngine::LayerConfig rightConfig = selectedPD->getConfig();
    size_t selected_count = 0;
    for (size_t j = 0; j < descs.size(); j++) {
        const auto &desc = descs[j];
        primitive_desc_iterator itpd;

        itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);

        while (itpd) {
            InferenceEngine::LayerConfig cfg;
            cfg.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                dataConfig.constant = false;
                dataConfig.desc = getSrcMemDesc(itpd, i);
                cfg.inConfs.push_back(dataConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getDstMemDesc(itpd, i);
                cfg.outConfs.push_back(dataConfig);
            }
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
            if (selected_count == selectedPrimitiveDescriptorIndex) {
                if (impl_type != selectedPD->getImplementationType()) {
                    THROW_IE_EXCEPTION << "Cannot get the original layer configuration!";
                }
                rightConfig = cfg;
            }
            if (j == descs.size() - 1) {
                if (impl_type == selectedPD->getImplementationType()) {
                    rightConfig = config;
                }
            }
            selected_count++;
            if (!itpd.next_impl())
                break;
        }
    }

    if (descs.empty()) {
        const auto& selectedConfig = selectedPD->getConfig();
        if (selectedConfig.inConfs.size() != config.inConfs.size() || selectedConfig.outConfs.size() != config.outConfs.size())
            return;

        for (size_t i = 0; i < selectedConfig.inConfs.size(); i++) {
            if (selectedConfig.inConfs[i].desc.getLayout() != InferenceEngine::Layout::ANY &&
                !MKLDNNExtensionUtils::initTensorsAreEqual(selectedConfig.inConfs[i].desc, config.inConfs[i].desc))
                THROW_IE_EXCEPTION << "Incorrect descriptor for node: " << getName();
        }

        for (size_t i = 0; i < selectedConfig.outConfs.size(); i++) {
            if (selectedConfig.outConfs[i].desc.getLayout() != InferenceEngine::Layout::ANY &&
                !MKLDNNExtensionUtils::initTensorsAreEqual(selectedConfig.outConfs[i].desc, config.outConfs[i].desc))
                THROW_IE_EXCEPTION << "Incorrect descriptor for node: " << getName();
        }
        rightConfig = config;
    }

    selectedPD->getConfig() = rightConfig;
}

void MKLDNNPoolingNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights) {
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode *>(node.get());
        if (quantizeNode) {
            quantizeNode->appendPostOps(ops);
            continue;
        }

        THROW_IE_EXCEPTION << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

REG_MKLDNN_PRIM_FOR(MKLDNNPoolingNode, Pooling);
