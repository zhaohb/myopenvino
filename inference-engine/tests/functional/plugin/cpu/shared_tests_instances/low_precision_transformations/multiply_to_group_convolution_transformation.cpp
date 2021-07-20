// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/multiply_to_group_convolution_transformation.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<element::Type> precisions = {
    element::f32
};

const std::vector< ngraph::Shape > inputShapes = {
    Shape{ 1ul, 4ul, 16ul, 16ul }, Shape{ 1ul, 4ul, 16ul, 16ul, 16ul }
};

const std::vector<builder::subgraph::FakeQuantizeOnData> fqOnData = {
    { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
    { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { 10.f }, { 25.5f } },
    { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 25.5f }, { -12.8f }, { 12.7f } }
};

INSTANTIATE_TEST_CASE_P(smoke_LPT, MultiplyToGroupConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(fqOnData)),
    MultiplyToGroupConvolutionTransformation::getTestCaseName);
}  // namespace
