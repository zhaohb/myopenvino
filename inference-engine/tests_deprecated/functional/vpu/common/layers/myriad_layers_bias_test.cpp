// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_bias_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsBias_smoke,
        ::testing::ValuesIn(s_biasDims)
);
