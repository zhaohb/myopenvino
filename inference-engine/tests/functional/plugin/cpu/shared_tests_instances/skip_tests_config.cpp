// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include <ie_system_conf.h>
#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> retVector{
        // TODO: Issue 26264
        R"(.*(MaxPool|AvgPool).*S\(1\.2\).*Rounding=ceil.*)",
        // TODO: Issue 31841
        R"(.*(QuantGroupConvBackpropData3D).*)",
        // TODO: Issue 31843
        R"(.*(QuantConvBackpropData3D).*)",
        R"(.*(QuantConvBackpropData2D).*(QG=Perchannel).*)",
        R"(.*(QuantGroupConvBackpropData2D).*(QG=Perchannel).*)",
        // TODO: Issue 33886
        R"(.*(QuantGroupConv2D).*)",
        R"(.*(QuantGroupConv3D).*)",
        // TODO: failed to downgrade to opset v0 in interpreter backend
        R"(.*Gather.*axis=-1.*)",
        // TODO: Issue: 34518
        R"(.*RangeLayerTest.*)",
        R"(.*(RangeAddSubgraphTest).*Start=1.2.*Stop=(5.2|-5.2).*Step=(0.1|-0.1).*netPRC=FP16.*)",
        R"(.*(RangeNumpyAddSubgraphTest).*netPRC=FP16.*)",
        // TODO: Issue: 34083
#if (defined(_WIN32) || defined(_WIN64))
        R"(.*(CoreThreadingTestsWithIterations).*(smoke_LoadNetworkAccuracy).*)",
#endif
        // TODO: Issue: 43793
        R"(.*(PreprocessTest).*(SetScalePreProcessSetBlob).*)",
        R"(.*(PreprocessTest).*(SetScalePreProcessGetBlob).*)",
        R"(.*(PreprocessTest).*(SetMeanValuePreProcessSetBlob).*)",
        R"(.*(PreprocessTest).*(SetMeanImagePreProcessSetBlob).*)",
        R"(.*(PreprocessTest).*(ReverseInputChannelsPreProcessGetBlob).*)",
        // TODO: Issue: 34348
        R"(.*IEClassGetAvailableDevices.*)",
        // TODO: Issue: 25533
        R"(.*ConvertLikeLayerTest.*)",
        // TODO: Issue: 34055
        R"(.*ShapeOfLayerTest.*)",
        R"(.*ReluShapeOfSubgraphTest.*)",
        // TODO: Issue: 34805
        R"(.*ActivationLayerTest.*Ceiling.*)",
        // TODO: Issue: 32032
        R"(.*ActivationParamLayerTest.*)",
        // TODO: Issue: 38841
        R"(.*TopKLayerTest.*k=10.*mode=min.*sort=index.*)",
        R"(.*TopKLayerTest.*k=5.*sort=(none|index).*)",
        // TODO: Issue: 43314
        R"(.*Broadcast.*mode=BIDIRECTIONAL.*inNPrec=BOOL.*)",
        // TODO: Issue 43417 sporadic issue, looks like an issue in test, reproducible only on Windows platform
        R"(.*decomposition1_batch=5_hidden_size=10_input_size=30_.*tanh.relu.*_clip=0_linear_before_reset=1.*_targetDevice=CPU_.*)",
        // Skip platforms that do not support BF16 (i.e. sse, avx, avx2)
        R"(.*BF16.*(jit_avx(?!5)|jit_sse).*)",
    };

    if (!InferenceEngine::with_cpu_x86_avx512_core()) {
        // on platforms which do not support bfloat16, we are disabling bf16 tests since there are no bf16 primitives,
        // tests are useless on such platforms
       retVector.emplace_back(R"(.*BF16.*)");
    }

    return retVector;
}
