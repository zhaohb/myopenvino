// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "mkldnn_memory_node.hpp"
#include "common/cpu_memcpy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

std::mutex MKLDNNMemoryNodeVirtualEdge::holderMutex;

MKLDNNMemoryOutputNode::MKLDNNMemoryOutputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) , MKLDNNMemoryNode(layer) {
    if (created()) {
        holder = MKLDNNMemoryNodeVirtualEdge::registerOutput(this);
    }
}

MKLDNNMemoryOutputNode::~MKLDNNMemoryOutputNode() {
    MKLDNNMemoryNodeVirtualEdge::remove(this, holder);
}

void MKLDNNMemoryOutputNode::getSupportedDescriptors() {}

void MKLDNNMemoryOutputNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims()));
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, memory::format_tag::any);
}

void MKLDNNMemoryOutputNode::execute(mkldnn::stream strm)  {
    auto& srcMemory = getParentEdgeAt(0)->getMemory();

    auto inputMemoryNode = dynamic_cast<MKLDNNMemoryInputNode*>(inputNode);
    IE_ASSERT(inputMemoryNode != nullptr);
    inputMemoryNode->storeState(srcMemory);
}

MKLDNNMemoryInputNode::MKLDNNMemoryInputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNInputNode(layer, eng, cache), MKLDNNMemoryNode(layer), dataStore(new MKLDNNMemory{eng}) {
    if (created()) {
        holder = MKLDNNMemoryNodeVirtualEdge::registerInput(this);
    }
}

void MKLDNNMemoryInputNode::createPrimitive() {
    MKLDNNInputNode::createPrimitive();

    auto mem_desc = getChildEdgeAt(0)->getMemoryPtr()->GetDescriptor();
    dataStore->Create(mem_desc);

    // default memory state is zero filled
    dataStore->FillZero();
}

/**
 * Copy data from one tensor into other.
 * As is. Assume that data is dense tensor with same layout.
 * @param dst destination memory object
 * @param src source memory object
 */
inline
static void simple_copy(MKLDNNMemory& dst, const MKLDNNMemory& src) {
    auto srcPtr = static_cast<uint8_t*>(src.GetPtr());
    auto dstPtr = static_cast<uint8_t*>(dst.GetPtr());
    auto srcSizeInByte = src.GetSize();
    auto dstSizeInByte = dst.GetSize();

    IE_ASSERT(srcSizeInByte == dstSizeInByte) << "Memory objects are not compatible. Has different sizes.";

    cpu_memcpy(dstPtr, srcPtr, srcSizeInByte);
}

MKLDNNMemoryInputNode::~MKLDNNMemoryInputNode() {
    MKLDNNMemoryNodeVirtualEdge::remove(this, holder);
}

MKLDNNMemoryPtr MKLDNNMemoryInputNode::getStore() {
    return dataStore;
}

void MKLDNNMemoryInputNode::storeState(const MKLDNNMemory &new_state) {
    // TODO: Should be next one call:
    //           dataStore.SetData(new_state, false);
    //       But because of performance reason we use simple manual copy
    simple_copy(*dataStore, new_state);
}

void MKLDNNMemoryInputNode::execute(mkldnn::stream strm) {
    auto dst_mem = getChildEdgeAt(0)->getMemory();
    // TODO: Should be simple call of:
    //           dst_mem.SetData(dataStore, false);
    //       But because of performance reason we use simple manual copy
    simple_copy(dst_mem, *dataStore);
}

MKLDNNMemoryNodeVirtualEdge::Holder* MKLDNNMemoryNodeVirtualEdge::registerInput(MKLDNNMemoryInputNode * node) {
    std::lock_guard<std::mutex> lock{MKLDNNMemoryNodeVirtualEdge::holderMutex};
    // in case of output already registered
    auto& holder = MKLDNNMemoryNodeVirtualEdge::getExisted();
    auto sibling = MKLDNNMemoryNodeVirtualEdge::getByName(holder, node->getId());
    if (sibling != nullptr) {
        auto outputNode = dynamic_cast<MKLDNNMemoryOutputNode*>(sibling);
        IE_ASSERT(outputNode != nullptr);
        outputNode->setInputNode(node);
    } else {
        holder[node->getId()] = node;
    }
    return &holder;
}

MKLDNNMemoryNodeVirtualEdge::Holder* MKLDNNMemoryNodeVirtualEdge::registerOutput(MKLDNNMemoryOutputNode * node) {
    std::lock_guard<std::mutex> lock{MKLDNNMemoryNodeVirtualEdge::holderMutex};
    // in case of output layer
    auto& holder = MKLDNNMemoryNodeVirtualEdge::getExisted();
    auto sibling = MKLDNNMemoryNodeVirtualEdge::getByName(holder, node->getId());
    if (sibling != nullptr) {
        auto inputNode = dynamic_cast<MKLDNNMemoryInputNode*>(sibling);
        IE_ASSERT(inputNode != nullptr);
        node->setInputNode(inputNode);
    } else {
        holder[node->getId()] = node;
    }
    return &holder;
}

void MKLDNNMemoryNodeVirtualEdge::remove(MKLDNNMemoryNode * node, Holder* holder) {
    std::lock_guard<std::mutex> lock{MKLDNNMemoryNodeVirtualEdge::holderMutex};
    if (nullptr != holder) {
        InferenceEngine::details::erase_if(*holder, [&](const Holder::value_type & it){
            return it.second == node;
        });
    }
}

REG_MKLDNN_PRIM_FOR(MKLDNNMemoryInputNode, MemoryInput);
REG_MKLDNN_PRIM_FOR(MKLDNNMemoryOutputNode, MemoryOutput);
