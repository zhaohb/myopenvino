// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/parameter_result.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

INSTANTIATE_TEST_CASE_P(smoke_Check, ParameterResultSubgraphTest,
                        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                        ParameterResultSubgraphTest::getTestCaseName);

}  // namespace
