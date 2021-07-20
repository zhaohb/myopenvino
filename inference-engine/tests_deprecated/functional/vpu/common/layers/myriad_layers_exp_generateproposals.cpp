// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_exp_generateproposals_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsExpGenerateProposals_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_ExpGenerateProposalsLayerScores),
        ::testing::ValuesIn(s_ExpGenerateProposalsLayerImInfo),
        ::testing::ValuesIn(s_ExpGenerateProposalsLayerParam))
);
