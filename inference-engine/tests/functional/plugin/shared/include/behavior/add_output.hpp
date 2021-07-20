// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "common_test_utils/test_common.hpp"
#include <ie_core.hpp>

typedef std::tuple<
        InferenceEngine::CNNNetwork, // CNNNetwork to work with
        std::vector<std::string>,    // Target layers to add as outputs
        std::string>                 // Target device name
        addOutputsParams;

class AddOutputsTest : public CommonTestUtils::TestsCommon,
                       public testing::WithParamInterface<addOutputsParams> {
protected:
    InferenceEngine::CNNNetwork net;
    std::vector<std::string> outputsToAdd;
    std::string deviceName;

    void SetUp();
public:
    static std::string getTestCaseName(const testing::TestParamInfo<addOutputsParams> &obj);
};
