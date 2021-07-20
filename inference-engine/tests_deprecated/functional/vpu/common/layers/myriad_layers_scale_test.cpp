// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_scale_test.hpp"

INSTANTIATE_TEST_CASE_P(
        accuracy, myriadLayersTestsScale_smoke,
        ::testing::Combine(
            ::testing::ValuesIn(s_inputScaleTensors),
            ::testing::ValuesIn(s_inputBiasScale)));
