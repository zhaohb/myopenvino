// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_clamp_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsClampParams_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_clampTensors),
        ::testing::ValuesIn(s_clampParams))
);
