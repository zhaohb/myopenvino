// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <map>
#include <string>
#include <vector>

#include <cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp>
#include <cpp_interfaces/interface/ie_ivariable_state_internal.hpp>

class MockIAsyncInferRequestInternal : public InferenceEngine::IAsyncInferRequestInternal {
public:
    MOCK_METHOD0(StartAsync, void());
    MOCK_METHOD1(Wait, InferenceEngine::StatusCode(int64_t));
    MOCK_METHOD1(GetUserData, void(void **));
    MOCK_METHOD1(SetUserData, void(void *));
    MOCK_METHOD0(Infer, void());
    MOCK_CONST_METHOD0(GetPerformanceCounts, std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>());
    MOCK_METHOD2(SetBlob, void(const std::string&, const InferenceEngine::Blob::Ptr &));
    MOCK_METHOD1(GetBlob, InferenceEngine::Blob::Ptr(const std::string&));
    MOCK_METHOD3(SetBlob, void(const std::string&, const InferenceEngine::Blob::Ptr &, const InferenceEngine::PreProcessInfo&));
    MOCK_CONST_METHOD1(GetPreProcess, const InferenceEngine::PreProcessInfo&(const std::string&));
    MOCK_METHOD1(SetCompletionCallback, void(InferenceEngine::IInferRequest::CompletionCallback));
    MOCK_METHOD1(SetBatch, void(int));
    MOCK_METHOD0(QueryState, std::vector<IVariableStateInternal::Ptr>());
    MOCK_METHOD0(Cancel, void());
};
