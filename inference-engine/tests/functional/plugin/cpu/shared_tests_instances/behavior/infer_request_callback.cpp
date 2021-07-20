// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multi-device/multi_device_config.hpp"

#include "behavior/infer_request_callback.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
        {},
        {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
        {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "0"}, {InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "1"}}
};

const std::vector<std::map<std::string, std::string>> multiConfigs = {
        {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_CPU}}
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CallbackTests,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::ValuesIn(configs)),
        CallbackTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, CallbackTests,
        ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                ::testing::ValuesIn(multiConfigs)),
        CallbackTests::getTestCaseName);
}  // namespace
