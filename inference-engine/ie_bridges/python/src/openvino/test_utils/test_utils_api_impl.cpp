// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils_api_impl.hpp"

#include <string>

#include <common_test_utils/ngraph_test_utils.hpp>

std::pair<bool, std::string> InferenceEnginePython::CompareNetworks(InferenceEnginePython::IENetwork lhs,
                                                                    InferenceEnginePython::IENetwork rhs) {
    return compare_functions(lhs.actual->getFunction(), rhs.actual->getFunction(), true, true, false, true);
}
