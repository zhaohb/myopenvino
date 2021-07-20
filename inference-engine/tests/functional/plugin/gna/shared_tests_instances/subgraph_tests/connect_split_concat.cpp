// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "subgraph_tests/multiple_connect_split_concat.hpp"
#include "subgraph_tests/connect_split_concat_concat.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16
};

std::map<std::string, std::string> additional_config = {
        {"GNA_COMPACT_MODE", "NO"}
};

INSTANTIATE_TEST_CASE_P(multiple_connect_split_concat, MultipleConnectSplitConcatTest,
        ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::Values(additional_config)),
        MultipleConnectSplitConcatTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(connect_split_concat_concat, SplitConcatConcatTest,
        ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::Values(additional_config)),
        MultipleConnectSplitConcatTest::getTestCaseName);
}  // namespace
