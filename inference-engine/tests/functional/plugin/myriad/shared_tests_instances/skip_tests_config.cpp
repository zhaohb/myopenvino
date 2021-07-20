// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        // Not supported activation types
        ".*ActivationLayerTest\\.CompareWithRefs/Tanh.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Exp.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Log.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Sigmoid.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Relu.*netPRC=FP32.*",
        // TODO: Issue: 26268
        ".*ConcatLayerTest.*axis=0.*",
        // TODO: Issue 31197
        R"(.*(IEClassBasicTestP).*smoke_registerPluginsXMLUnicodePath.*)",
        // TODO: Issue: 34348
        R"(.*IEClassGetAvailableDevices.*)",
        // TODO: Issue: 40473
        R"(.*TopKLayerTest.*mode=min.*sort=index.*)",
        // TODO: Issue: 40961
        R"(.*(ConstantResultSubgraphTest).*)",
        // TODO: Issue: 42828
        R"(.*DSR_NonMaxSuppression.*NBoxes=(5|20|200).*)",
        // TODO: Issue: 42721
        R"(.*(DSR_GatherND).*)",
        // TODO: Issue 26090
        ".*DSR_GatherStaticDataDynamicIdx.*f32.*1.3.200.304.*",
        // TODO: Issue 47315
        ".*ProposalLayerTest.*",
        // TODO: Issue 48183
        R"(.*CTCGreedyDecoderSeqLen.*?\(1.1.1\).*)",
    };
}
