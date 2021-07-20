// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mat_mul.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<ShapeRelatedParams> shapeRelatedParams = {
        { { {1, 4, 5, 6}, false }, { {1, 4, 6, 4}, false } },
        { { {4, 5, 6}, false }, { {6, 3}, false } },
        { { {9, 9, 9}, false }, { {9, 9}, false } }
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_CASE_P(smoke_MatMul, MatMulTest,
        ::testing::Combine(
                ::testing::ValuesIn(shapeRelatedParams),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        MatMulTest::getTestCaseName);

} // namespace

