// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/mvn.hpp"

namespace LayerTestsDefinitions {

TEST_P(MvnLayerTest, CompareWithRefs) {
    Run();
};

TEST_P(Mvn6LayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions