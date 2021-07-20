// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNPoolingNode : public MKLDNNNode {
public:
    MKLDNNPoolingNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNPoolingNode() override = default;

    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    std::vector<mkldnn::memory::format_tag> getAvailableFormatsForDims(const MKLDNNDims &dims) const override;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initDescriptor(const InferenceEngine::LayerConfig &config) override;
    void createPrimitive() override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

private:
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false);

    InferenceEngine::PoolingLayer::PoolType type = InferenceEngine::PoolingLayer::MAX;
    bool exclude_pad = false;
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> kernel;

    /// Effective padding. Used to define correct output shape by MKLDNN
    /// reshape formula: (iw - kernel + pad_l + pad_r) / strides[i - 2] + 1
    /// should be passed into pooling desc constructor.
    std::vector<ptrdiff_t> effective_pad_begin;
    std::vector<ptrdiff_t> effective_pad_end;

    /// Effective pad value. Describe how much zero element added to input
    /// data tensor. May be less than "Effective padding" values.
    /// If pooling window is out of this padding, the region of averaging
    /// is decreased.
    std::vector<ptrdiff_t> data_pad_begin;
    std::vector<ptrdiff_t> data_pad_end;

    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP32;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32;

    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;
};

}  // namespace MKLDNNPlugin

