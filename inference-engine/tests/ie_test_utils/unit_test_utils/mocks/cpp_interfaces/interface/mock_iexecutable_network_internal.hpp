// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <gmock/gmock.h>

#include "ie_input_info.hpp"
#include "ie_icnn_network.hpp"
#include "ie_iexecutable_network.hpp"
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>

#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinfer_request_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_infer_request_internal.hpp"

using namespace InferenceEngine;

class MockIExecutableNetworkInternal : public IExecutableNetworkInternal {
public:
    MOCK_CONST_METHOD0(GetOutputsInfo, ConstOutputsDataMap(void));
    MOCK_CONST_METHOD0(GetInputsInfo, ConstInputsDataMap(void));
    MOCK_METHOD0(CreateInferRequest, IInferRequest::Ptr(void));
    MOCK_METHOD1(Export, void(const std::string &));
    void Export(std::ostream &) override {};
    MOCK_METHOD0(QueryState, std::vector<IVariableStateInternal::Ptr>(void));
    MOCK_METHOD0(GetExecGraphInfo, CNNNetwork(void));

    MOCK_METHOD1(SetConfig, void(const std::map<std::string, Parameter> &config));
    MOCK_CONST_METHOD1(GetConfig, Parameter(const std::string &name));
    MOCK_CONST_METHOD1(GetMetric, Parameter(const std::string &name));
    MOCK_CONST_METHOD0(GetContext, RemoteContext::Ptr(void));
};
