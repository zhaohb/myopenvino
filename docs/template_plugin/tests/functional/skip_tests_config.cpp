// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        ".*ExclusiveAsyncRequests.*",
        ".*reusableCPUStreamsExecutor.*",
        R"(.*SplitLayerTest.*numSplits\=30.*)",
        // CVS-44774
        ".*PreprocessTest.*",
    };
}