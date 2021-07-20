// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multi-device/multi_device_config.hpp"

#include "behavior/infer_request_input.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16,
            InferenceEngine::Precision::U8,
            InferenceEngine::Precision::U16,
            InferenceEngine::Precision::I16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}}
    };

    const std::vector<std::map<std::string, std::string>> multiConfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferRequestInputTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                    ::testing::ValuesIn(configs)),
                            InferRequestInputTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, InferRequestInputTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiConfigs)),
                            InferRequestInputTests::getTestCaseName);

}  // namespace
