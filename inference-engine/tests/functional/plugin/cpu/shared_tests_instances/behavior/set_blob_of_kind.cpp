// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/set_blob_of_kind.hpp"
#include "common_test_utils/test_constants.hpp"
#include "multi-device/multi_device_config.hpp"

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine;

const std::vector<FuncTestUtils::BlobKind> blobKinds = {
    FuncTestUtils::BlobKind::Simple,
    FuncTestUtils::BlobKind::Compound,
    FuncTestUtils::BlobKind::BatchOfSimple
};

const SetBlobOfKindConfig cpuConfig{}; //nothing special
const SetBlobOfKindConfig multiConfig{{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_CPU}};
const SetBlobOfKindConfig heteroConfig{{ "TARGET_FALLBACK", CommonTestUtils::DEVICE_CPU }};

INSTANTIATE_TEST_CASE_P(smoke_SetBlobOfKindCPU, SetBlobOfKindTest,
    ::testing::Combine(::testing::ValuesIn(blobKinds),
                       ::testing::Values(CommonTestUtils::DEVICE_CPU),
                       ::testing::Values(cpuConfig)),
    SetBlobOfKindTest::getTestCaseName);


INSTANTIATE_TEST_CASE_P(smoke_SetBlobOfKindMULTI, SetBlobOfKindTest,
    ::testing::Combine(::testing::ValuesIn(blobKinds),
                       ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                       ::testing::Values(multiConfig)),
    SetBlobOfKindTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_SetBlobOfKindHETERO, SetBlobOfKindTest,
    ::testing::Combine(::testing::ValuesIn(blobKinds),
                       ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                       ::testing::Values(heteroConfig)),
    SetBlobOfKindTest::getTestCaseName);
