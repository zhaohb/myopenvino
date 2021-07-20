// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_split_node.h"
#include "common/cpu_memcpy.h"
#include <legacy/ie_layers.h>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_parallel.hpp>

#define THROW_ERROR THROW_IE_EXCEPTION << "Split layer with name '" << getName() <<"' "

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

static TensorDesc makePlainTensorDesc(const Precision& precision, const SizeVector& srcDims) {
    SizeVector order(srcDims.size());
    std::iota(order.begin(), order.end(), 0);
    return TensorDesc(precision, srcDims, {srcDims, order});
}

static TensorDesc makePerChannelTensorDesc(const Precision& precision, const SizeVector& srcDims) {
    constexpr size_t channelsPos = 1lu;
    SizeVector order(srcDims.size());
    std::iota(order.begin(), order.end(), 0);
    SizeVector blkDims = srcDims;
    if (srcDims.size() > 2) {
        auto moveElementBack = [](SizeVector& vector, size_t indx) {
            auto itr = vector.begin() + indx;
            std::rotate(itr, itr + 1, vector.end());
        };

        moveElementBack(order, channelsPos);
        moveElementBack(blkDims, channelsPos);
    }

    return TensorDesc(precision, srcDims, {blkDims, order});
}

static TensorDesc makeChannelBlockedTensorDesc(const Precision& precision, const SizeVector& srcDims, size_t blockSize) {
    if (srcDims.size() < 2) {
        THROW_IE_EXCEPTION << "Can't create blocked tensor descriptor!";
    }

    constexpr size_t channelsPos = 1lu;
    SizeVector order(srcDims.size());
    std::iota(order.begin(), order.end(), 0);
    order.push_back(channelsPos);

    SizeVector blkDims = srcDims;
    blkDims[1] = blkDims[1] / blockSize + (blkDims[1] % blockSize ? 1 : 0);
    blkDims.push_back(blockSize);

    return TensorDesc(precision, srcDims, {blkDims, order});
}

MKLDNNSplitNode::MKLDNNSplitNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {}

void MKLDNNSplitNode::getSupportedDescriptors() {
    auto splitLayer = dynamic_cast<SplitLayer*>(getCnnLayer().get());

    if (splitLayer == nullptr)
        THROW_ERROR << "can not convert from CNN layer.";

    if (getParentEdges().size() != 1)
        THROW_ERROR << "has incorrect number of input nodes.";
    if (getChildEdges().empty())
        THROW_ERROR << "has incorrect number of output nodes.";

    axis = splitLayer->_axis;
    if (axis >= getParentEdgeAt(0)->getDims().ndims())
        THROW_ERROR << "has invalid value of axis parameter.";
}

void MKLDNNSplitNode::initSupportedPrimitiveDescriptors() {
    using TensorDescFactory = std::function<TensorDesc(const Precision&, const SizeVector&)>;
    constexpr size_t channelsPos = 1lu;

    if (!supportedPrimitiveDescriptors.empty())
        return;

    if (getCnnLayer()->insData.empty()) {
        THROW_ERROR << "has an empty input in the CNN layer";
    }

    auto inpData = getCnnLayer()->insData[0].lock();
    if (!inpData) {
        THROW_ERROR << "input data is empty";
    }

    auto srcDims = getParentEdgeAt(0)->getDims();
    auto axis_size = 0;
    auto dstFirstDims = getChildEdgeAt(0)->getDims();
    for (size_t i = 0; i < outDims.size(); i++) {
        auto o_Dims = outDims[i];
        if (dstFirstDims.ndims() != o_Dims.ndims()) {
            THROW_ERROR << "only supports output blobs with equal number of dimensions";
        }

        axis_size += o_Dims[axis];
        for (size_t j = 0; j < dstFirstDims.ndims(); j++) {
            if (j == axis)
                continue;
            if (o_Dims[j] != dstFirstDims[j])
                THROW_ERROR << "has incorrect output dimensions";
        }
    }
    dstFirstDims[axis] = axis_size;
    if (dstFirstDims.size() != srcDims.size())
        THROW_ERROR << "sizes of input blob and sum of output blobs are not equal.";


    InferenceEngine::Precision inpPrecision = inpData->getPrecision();
    auto outPrecision = inpPrecision; // the split layer doesn't convert precisions

    // make primitive descriptor factory function for different configurations
    bool dynBatchSupport = true;
    if (axis < 1) {
        dynBatchSupport = false;
    }
    auto makePdInfo = [dynBatchSupport](TensorDescFactory getTensorDesc, const Precision& precision,  const MKLDNNDims& srcDims,
                                        const std::vector<MKLDNNDims>& outDims, impl_desc_type type) -> PrimitiveDescInfo {
        InferenceEngine::LayerConfig config;

        config.dynBatchSupport = dynBatchSupport;
        config.inConfs.resize(1);
        config.inConfs[0].inPlace = -1;
        config.inConfs[0].constant = false;
        config.inConfs[0].desc = getTensorDesc(precision, srcDims.ToSizeVector());
        config.outConfs.resize(outDims.size());

        std::vector<memory::format_tag> outFormats;

        for (size_t i = 0; i < outDims.size(); i++) {
            auto o_Dims = outDims[i];

            config.outConfs[i].inPlace = -1;
            config.outConfs[i].constant = false;
            config.outConfs[i].desc = getTensorDesc(precision, o_Dims.ToSizeVector());
            outFormats.push_back(MKLDNNMemoryDesc(config.outConfs[i].desc).getFormat());
        }
        return {config, type, outFormats};
    };

    //Set plain format
    supportedPrimitiveDescriptors.push_back(makePdInfo(&makePlainTensorDesc, inpPrecision, srcDims, outDims, impl_desc_type::ref));

    //Set per channel format.
    supportedPrimitiveDescriptors.push_back(makePdInfo(&makePerChannelTensorDesc, inpPrecision, srcDims, outDims, impl_desc_type::ref));

    //Support channel blocked format
    std::vector<size_t> blockedPdIndexes;
    if (srcDims.ndims() > channelsPos) {
        for (size_t sizeS : {8lu, 16lu}) {
            SizeVector blkDims = srcDims.ToSizeVector();
            if (blkDims[channelsPos] % sizeS)
                continue;

            bool blocked = true;
            for (size_t i = 0; i < outDims.size(); i++) {
                if (outDims[i].ToSizeVector()[channelsPos] % sizeS) {
                    blocked = false;
                    break;
                }
            }
            if (blocked) {
                using std::placeholders::_1;
                using std::placeholders::_2;
                supportedPrimitiveDescriptors.push_back(makePdInfo(std::bind(&makeChannelBlockedTensorDesc, _1, _2, sizeS),
                                                                   inpPrecision, srcDims, outDims, impl_desc_type::ref));
                blockedPdIndexes.push_back(supportedPrimitiveDescriptors.size() - 1);
            }
        }
    }

    // Optimized inplace case
    std::vector<size_t> pdIndexesToReuse(1, 0); // at least the first plain layout can be optimized inplace.
    if (axis < 2) {
        pdIndexesToReuse.insert(pdIndexesToReuse.end(), blockedPdIndexes.begin(), blockedPdIndexes.end());
    }

    for (auto refPdIndex : pdIndexesToReuse) {
        const auto& refConfig = supportedPrimitiveDescriptors[refPdIndex].getConfig();
        auto config = refConfig;

        const auto& order = refConfig.inConfs[0].desc.getBlockingDesc().getOrder();
        const auto& blkDims = refConfig.inConfs[0].desc.getBlockingDesc().getBlockDims();
        auto numOfDim = blkDims.size();

        std::vector<memory::format_tag> outFormats;
        SizeVector offsets(numOfDim, 0lu);
        SizeVector strides(numOfDim);
        strides.back() = 1lu;
        size_t offset = (std::numeric_limits<size_t>::max)();

        for (size_t i = 2; i <= numOfDim; i++) {
            if (numOfDim - i < axis) {
                strides[numOfDim - i] = (std::numeric_limits<size_t>::max)();
            } else {
                strides[numOfDim - i] = strides[numOfDim - i + 1] * blkDims[numOfDim - i + 1];
            }
        }

        config.inConfs[0].desc = TensorDesc(inpPrecision, srcDims.ToSizeVector(), {blkDims, order, offset, offsets, strides});

        for (size_t i = 0; i < outDims.size(); i++) {
            const auto& outBlkDims = refConfig.outConfs[i].desc.getBlockingDesc().getBlockDims();
            const auto& dims = refConfig.outConfs[i].desc.getDims();

            config.outConfs[i].inPlace = 0;
            config.outConfs[i].desc = TensorDesc(outPrecision, dims, {outBlkDims, order, offset, offsets, strides});
            outFormats.emplace_back(MKLDNNMemoryDesc(config.outConfs[i].desc).getFormat());
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, outFormats);
    }

    // Special nspc -> ncsp case when splitting channels
    if (axis == 1 && (dstFirstDims.ndims() == 4 || dstFirstDims.ndims() == 5)) {
        auto plain = makePdInfo(&makePlainTensorDesc, inpPrecision, srcDims, outDims, impl_desc_type::ref);
        auto perChannel = makePdInfo(&makePerChannelTensorDesc, inpPrecision, srcDims, outDims, impl_desc_type::ref);

        plain.getConfig().inConfs[0].desc = perChannel.getConfig().inConfs[0].desc;

        supportedPrimitiveDescriptors.push_back(plain);
    }
}

void MKLDNNSplitNode::createPrimitive() {
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_ERROR << "Input memory has not been allocated.";
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        if (!getChildEdgeAt(i)->getMemoryPtr() || !getChildEdgeAt(i)->getMemory().GetPrimitivePtr())
            THROW_ERROR << "Destination memory has not been allocated.";
    }
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << "Preferable primitive descriptor is not set.";

    canUseOptimizedNspc2Ncsp = true;
    if (axis != 1)
        canUseOptimizedNspc2Ncsp = false;

    if (getParentEdgeAt(0)->getBlob()->getTensorDesc().getLayout() != NHWC &&
        getParentEdgeAt(0)->getBlob()->getTensorDesc().getLayout() != NDHWC)
        canUseOptimizedNspc2Ncsp = false;

    for (size_t i = 0; i < getChildEdges().size(); i++) {
        if (getChildEdgeAt(i)->getBlob()->getTensorDesc().getLayout() != NCHW &&
            getChildEdgeAt(i)->getBlob()->getTensorDesc().getLayout() != NCDHW)
            canUseOptimizedNspc2Ncsp = false;
    }

    if (!isOptimized()) {
        initializeDstMemPtrs();
        if (!canUseOptimizedNspc2Ncsp)
            prepareOptimizedParams();
    }
}

void MKLDNNSplitNode::execute(mkldnn::stream strm) {
    if (isOptimized())
        return;

    if (dstMemPtrs.empty())
        THROW_ERROR << "Output data pointers have not been initialized.";

    int MB = batchToProcess();

    if (canUseOptimizedNspc2Ncsp) {
        optimizedNspc2Ncsp(MB);
        return;
    }

    uint8_t* srcData = reinterpret_cast<uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    size_t batch = this->getParentEdgeAt(0)->getDims()[0];

    if (batch != MB)
        optimizedParams.countStrides = optimizedParams.countStrides / batch * MB;

    parallel_for2d(dstMemPtrs.size(), optimizedParams.countStrides, [&](size_t i, size_t j) {
        uint8_t* dstData = dstMemPtrs[i];

        cpu_memcpy(&dstData[j * optimizedParams.dataSize[i]],
                   &srcData[optimizedParams.srcDataOffsets[i] + j * optimizedParams.srcDataStride],
                   optimizedParams.dataSize[i]);
    });
}

bool MKLDNNSplitNode::created() const {
    return getType() == Split;
}

bool MKLDNNSplitNode::isOptimized() {
    return getSelectedPrimitiveDescriptor() && getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].inPlace >= 0;
}

void MKLDNNSplitNode::initOptimalPrimitiveDescriptor() {
    if (!isOptimized()) {
        MKLDNNNode::initOptimalPrimitiveDescriptor();
        return;
    }

    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_ERROR << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (isInitConfig(config))
        return;

    for (size_t i = 0; i < config.inConfs.size(); i++) {
        if (config.inConfs[i].desc.getLayout() == InferenceEngine::Layout::ANY ||
            !isUninitTensorDesc(config.inConfs[i].desc))
            continue;

        int num = getParentEdgeAt(i)->getOutputNum();
        if (getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()) {
            if (num >= 0) {
                if (isUninitTensorDesc(getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].desc) &&
                        getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].inPlace >= 0)
                    getParentEdgeAt(i)->getParent()->initOptimalPrimitiveDescriptor();
                if (!isUninitTensorDesc(getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].desc) &&
                    MKLDNNExtensionUtils::initTensorsAreEqual(
                            getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].desc,
                            config.inConfs[i].desc)) {
                    config.inConfs[i].desc = getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].desc;
                    continue;
                }
            }
        }
        config.inConfs[i].desc = InferenceEngine::TensorDesc(config.inConfs[i].desc.getPrecision(),
                                                              config.inConfs[i].desc.getDims(), {
                                                                      config.inConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                      config.inConfs[i].desc.getBlockingDesc().getOrder()
                                                              });
    }
    const auto& cnnLayer = getCnnLayer();
    if (!cnnLayer)
        THROW_ERROR << "cannot be created without CNNLayer!";
    if (config.outConfs.size() != outDims.size())
        THROW_ERROR << "has invalid config";
    size_t offset = 0;
    for (size_t i = 0; i < cnnLayer->outData.size(); i++) {
        config.outConfs[i].desc = InferenceEngine::TensorDesc(config.outConfs[i].desc.getPrecision(),
                                                              config.outConfs[i].desc.getDims(), {
                                                                      config.outConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                      config.outConfs[i].desc.getBlockingDesc().getOrder(),
                                                                      config.inConfs[0].desc.getBlockingDesc().getOffsetPadding() + offset,
                                                                      config.inConfs[0].desc.getBlockingDesc().getOffsetPaddingToData(),
                                                                      config.inConfs[0].desc.getBlockingDesc().getStrides()
                                                              });
        size_t axisSize = 1;
        for (size_t j = axis; j < config.outConfs[i].desc.getBlockingDesc().getBlockDims().size(); j++) {
            axisSize *= config.outConfs[i].desc.getBlockingDesc().getBlockDims()[j];
        }
        offset += axisSize;
    }
    initDescriptor(config);
}

void MKLDNNSplitNode::selectOptimalPrimitiveDescriptor() {
    if (implPriorities.size() > 0 && implPriorities[0] == impl_desc_type::ref) {
        selectPrimitiveDescriptorByIndex(0);
        return;
    }

    //check the descriptors and select the ones that have the same data format as the input

    std::vector<size_t> canSelectPrimitive;
    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
        auto parentEdge = getParentEdgeAt(0);
        auto parentPtr = parentEdge->getParent();
        auto parent_spd = parentPtr->getSelectedPrimitiveDescriptor();

        if (parent_spd != nullptr && !parent_spd->getConfig().outConfs.empty()) {
            int inNum = parentEdge->getInputNum();
            if (inNum < 0 || inNum >= parent_spd->getConfig().outConfs.size()) {
                inNum = 0;
            }
            if (MKLDNNExtensionUtils::initTensorsAreEqual(
                    supportedPrimitiveDescriptors[i].getConfig().inConfs[0].desc,
                    parent_spd->getConfig().outConfs[inNum].desc)) {
                canSelectPrimitive.push_back(i);
            }
        }
    }
    if (canSelectPrimitive.size() == 1) {
        selectPrimitiveDescriptorByIndex(static_cast<int>(canSelectPrimitive[0]));
        return;
    }
    // if there are more then one PD with similar data layouts - select the optimized one
    for (auto indx : canSelectPrimitive) {
        if (supportedPrimitiveDescriptors[indx].getImplementationType() == impl_desc_type::unknown) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(indx));
            return;
        }
    }

    // if there are no inPlace, but more than one suitable configurations, select the one that matches the output layout
    for (auto indx : canSelectPrimitive) {
        bool outputDescFullMatch = true;
        for (size_t i = 0; i < getChildEdges().size(); ++i) {
            auto childEdge = getChildEdgeAt(i);
            auto childPtr = childEdge->getChild();
            auto& vecChildSpd = childPtr->getSupportedPrimitiveDescriptors();
            const auto& outputDesc = supportedPrimitiveDescriptors[indx].getConfig().outConfs[i].desc;

            if (!vecChildSpd.empty()) {
                int inNum = childEdge->getOutputNum();
                if (inNum < 0) {
                    inNum = 0;
                }
                bool hasMatchDesc = false;
                for (auto& childSpd : vecChildSpd) {
                    if (inNum >= childSpd.getConfig().inConfs.size()) {
                        inNum = 0;
                    }
                    if (MKLDNNExtensionUtils::initTensorsAreEqual(outputDesc, childSpd.getConfig().inConfs[inNum].desc)) {
                        hasMatchDesc = true;
                        break;
                    }
                }
                if (!hasMatchDesc) {
                    outputDescFullMatch = false;
                    break;
                }
            }
        }
        if (outputDescFullMatch) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(indx));
            return;
        }
    }
    if (!canSelectPrimitive.empty()) {
        selectPrimitiveDescriptorByIndex(static_cast<int>(canSelectPrimitive.front()));
        return;
    }

    // if there are no matching data layouts, select first optimized implementation
    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
        if (supportedPrimitiveDescriptors[i].getImplementationType() == impl_desc_type::unknown) {
            selectPrimitiveDescriptorByIndex(static_cast<int>(i));
            return;
        }
    }

    selectPrimitiveDescriptorByIndex(0);
}

void MKLDNNSplitNode::setDynamicBatchLim(int lim) {
    if (axis == 0)
        THROW_ERROR << "Dynamic batch is not supported by split layer with axis == 0 parameter";

    dynBatchLim = lim;
}

void MKLDNNSplitNode::prepareOptimizedParams() {
    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        THROW_IE_EXCEPTION << "CPU Split node with name '" << getName() << "' doesn't have primitive descriptors.";
    const auto& inpTensorDesc = selectedPrimitiveDescriptor->getConfig().inConfs[0].desc;
    const auto outputPortsCount = outDims.size();

    //find axis order position
    const auto& order = inpTensorDesc.getBlockingDesc().getOrder();
    unsigned axisOrderPos = std::numeric_limits<unsigned>::max();
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] == axis) {
            axisOrderPos = i;
            break;
        }
    }
    if (std::numeric_limits<unsigned>::max() == axisOrderPos) {
        THROW_ERROR << "Can't find the axis in the input tensor order list";
    }

    uint8_t srcDataSize = inpTensorDesc.getPrecision().size();
    const auto& srcDims = inpTensorDesc.getBlockingDesc().getBlockDims();
    const auto nDims = srcDims.size();

    optimizedParams.countStrides = 1;
    for (int i = 0; i < axisOrderPos; i++)
        optimizedParams.countStrides *= srcDims[i];

    optimizedParams.srcDataStride = 0;
    optimizedParams.dataSize.resize(outputPortsCount);

    for (size_t i = 0; i < outputPortsCount; i++) {
        auto outputEdge = this->getChildEdgesAtPort(i).front();
        optimizedParams.dataSize[i] = srcDataSize;

        for (size_t j = axisOrderPos; j < nDims; j++)
            optimizedParams.dataSize[i] *= outputEdge->getDesc().getBlockingDesc().getBlockDims()[j];

        optimizedParams.srcDataStride += optimizedParams.dataSize[i];
    }

    optimizedParams.srcDataOffsets.resize(outputPortsCount);
    optimizedParams.srcDataOffsets[0] = 0;
    for (size_t i = 1; i < outputPortsCount; i++) {
        optimizedParams.srcDataOffsets[i] = optimizedParams.srcDataOffsets[i - 1] + optimizedParams.dataSize[i - 1];
    }
}
void MKLDNNSplitNode::optimizedNspc2Ncsp(size_t MB) {
    auto parentEdge = getParentEdgeAt(0);
    const int ndims = parentEdge->getDims().ndims();
    const size_t IC = parentEdge->getDims()[1];
    const size_t D = ndims == 5 ? parentEdge->getDims()[ndims - 3] : 1;
    const size_t H = parentEdge->getDims()[ndims - 2];
    const size_t W = parentEdge->getDims()[ndims - 1];

    auto srcBlob = parentEdge->getBlob();
    auto srcData = srcBlob->cbuffer().as<const uint8_t*>();
    const auto dataSize = srcBlob->getTensorDesc().getPrecision().size();

    const size_t DHW = D*H*W;
    const size_t strideIB = DHW * IC * dataSize;
    const size_t strideIW = IC*dataSize;
    const size_t strideOC = DHW * dataSize;

    for (size_t i = 0, sIdx = 0; i < outDims.size(); i++) {
        auto dstData = dstMemPtrs[i];

        size_t innerSize = 1;
        auto dims = outDims[i].ToSizeVector();

        for (size_t j = axis; j < dims.size(); j++) {
            innerSize *= dims[j];
        }
        auto srcPtr = srcData + srcBlob->getTensorDesc().offset(sIdx) * dataSize;

        const size_t OC = dims[1];
        const size_t strideOB = OC * strideOC;

        parallel_for2d(MB, DHW, [&](size_t b, size_t j) {
            auto localSrcPtr = srcPtr + b*strideIB + j*strideIW;
            auto localDstPtr = dstData + b*strideOB + j*dataSize;
            for (size_t c = 0; c < OC; c++) {
                cpu_memcpy(localDstPtr, localSrcPtr, dataSize);
                localSrcPtr += dataSize;
                localDstPtr += strideOC;
            }
        });

        sIdx += innerSize;
    }
}

void MKLDNNSplitNode::initializeDstMemPtrs() {
    dstMemPtrs.clear();

    for (size_t i = 0; i < outDims.size(); ++i) {
        auto outputEdges = this->getChildEdgesAtPort(i);
        if (uint8_t* dstData = reinterpret_cast<uint8_t*>(outputEdges.front()->getMemoryPtr()->GetPtr())) {
            dstMemPtrs.push_back(dstData);
        } else {
            THROW_ERROR << "can't get child edge indx " << i << "data.";
        }
    }
}

REG_MKLDNN_PRIM_FOR(MKLDNNSplitNode, Split);
