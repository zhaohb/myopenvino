// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "shared_test_classes/subgraph/connect_split_concat_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(SplitConcatConcatTest, CompareWithRefs) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
