// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNGemmNode : public MKLDNNNode {
public:
    MKLDNNGemmNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNGemmNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    int getMaxBatch() override;

    InferenceEngine::Precision getRuntimePrecision() const override;

private:
    float alpha = 1.0f;
    float beta = 1.0f;
    bool transposeA = false;
    bool transposeB = false;

    int xAxis = 0;
    int yAxis = 0;

    bool isThreeInputs = false;

    std::vector<int> aOffsets;
    std::vector<int> bOffsets;
    std::vector<int> cOffsets;

    template<typename T0, typename T1> void process_data();
};

}  // namespace MKLDNNPlugin

