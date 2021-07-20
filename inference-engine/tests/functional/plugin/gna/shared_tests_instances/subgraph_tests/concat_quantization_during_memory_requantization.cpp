// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <vector>
#include "subgraph_tests/concat_quantization_during_memory_requantization.hpp"
#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32
    };

    std::map<std::string, std::string> config = {
            {"GNA_COMPACT_MODE", "NO"}
    };

    std::vector<size_t> inputSizes = {
        128,
        64,
        32
    };

    std::vector<size_t> hiddenSizes = {
        128,
        64,
        32
    };

    INSTANTIATE_TEST_CASE_P(smoke_concat_quant_memory_requant, ConcatQuantDuringMemoryRequantTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(inputSizes),
            ::testing::ValuesIn(hiddenSizes),
            ::testing::Values(config)),
        ConcatQuantDuringMemoryRequantTest::getTestCaseName);
}  // namespace
