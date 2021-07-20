// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_exp_topkrois_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsExpTopKROIs_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_ExpTopKROIsInputRoisNum),
        ::testing::ValuesIn(s_ExpTopKROIsMaxRoisNum))
);
