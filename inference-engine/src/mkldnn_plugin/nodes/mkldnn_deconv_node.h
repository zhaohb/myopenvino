// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <memory>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNDeconvolutionNode : public MKLDNNNode {
public:
    MKLDNNDeconvolutionNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNDeconvolutionNode() override = default;

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void createPrimitive() override;
    void filterSupportedPrimitiveDescriptors() override;
    void filterSupportedDescriptors();
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    size_t descInputNumbers(MKLDNNDescriptor desc) override {
        return static_cast<size_t>(getParentEdges().size());
    }

    MKLDNNMemoryDesc getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    MKLDNNMemoryDesc getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

    InferenceEngine::Precision getRuntimePrecision() const override;

private:
    bool withGroups = false;
    bool isDW = false;
    size_t groupNum = 1;
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingR;
    MKLDNNDims weightsDims;
    std::vector<std::shared_ptr<mkldnn::convolution_forward::desc>> descs_fwd;
    std::vector<std::shared_ptr<mkldnn::convolution_backward_data::desc>> descs_bwd;

    mkldnn::primitive_attr attr;
    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;
    void setBiasAsPostOp(const InferenceEngine::Blob::Ptr& biases);

    const mkldnn::memory& getWeights() const;
};

}  // namespace MKLDNNPlugin

