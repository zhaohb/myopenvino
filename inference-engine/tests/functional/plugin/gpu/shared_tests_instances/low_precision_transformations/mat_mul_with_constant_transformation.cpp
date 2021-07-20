// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include "low_precision_transformations/mat_mul_with_constant_transformation.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> precisions = { ngraph::element::f32 };

//transpose_a = false, transpose_b = true
std::vector<MatMulWithConstantTransformationTestValues> testValues = {
    {
        { 2, 3, 4 },
        { 256ul, {{1, 1, 1}, {1, 1, 1}, {1, 3, 1}, {1, 3, 1}}, {0.f}, {255.f}, {0.f, 0.f, 0.f}, {255.f, 25.5f, 255.f} },
        { std::vector<float>(4 * 2, 2.f), ngraph::element::f32, ngraph::Shape{ 2, 4 } },
        { 256ul, {{1}, {1}, {2, 1}, {2, 1}}, {-128.f}, {127.f}, {-128.f, -12.8f}, {127.f, 12.7f} },
        { {}, {}, {} },
    },
    {
        { 2, 3, 4 },
        { 256ul, {{1, 1, 1}, {1, 1, 1}, {1, 3, 1}, {1, 3, 1}}, {0.f}, {255.f}, {0.f, 0.f, 0.f}, {255.f, 25.5f, 255.f} },
        { std::vector<float>(4 * 2, 2.f), ngraph::element::i8, ngraph::Shape{ 2, 4 } },
        {},
        { ngraph::element::f32, {}, {0.1f} },
    },
    {
        { 1, 3, 4 },
        { 256ul, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, {-10.5f}, {4.5f}, {-10.5f}, {4.5f} },
        { std::vector<float>(4 * 2, 2.f), ngraph::element::f32, ngraph::Shape{ 2, 4 } },
        { 256ul, {{1}, {1}, {2, 1}, {2, 1}}, {-128.f}, {127.f}, {-128.f, -12.8f}, {127.f, 12.7f} },
        { {}, {}, {} },
    },
    {
        { 1, 1, 3, 4 },
        { 256ul, {{1, 1, 1}, {1, 1, 1}, {1, 3, 1}, {1, 3, 1}}, {0.f}, {255.f}, {0.f, 0.f, 0.f}, {255.f, 25.5f, 255.f} },
        { std::vector<float>(4 * 2, 2.f), ngraph::element::f32, ngraph::Shape{ 2, 4 } },
        { 256ul, {{1}, {1}, {2, 1}, {2, 1}}, {-128.f}, {127.f}, {-128.f, -12.8f}, {127.f, 12.7f} },
        { {}, {}, {} },
    },
    {
        { 1, 1, 3, 4 },
        { 256ul, {{1, 1, 1}, {1, 1, 1}, {1, 3, 1}, {1, 3, 1}}, {0.f}, {255.f}, {0.f, 0.f, 0.f}, {255.f, 25.5f, 255.f} },
        { std::vector<float>(4 * 2, 2.f), ngraph::element::i8, ngraph::Shape{ 2, 4 } },
        {},
        { ngraph::element::f32, {}, {{0.1f, 0.01}, ngraph::element::f32, ngraph::Shape{ 2, 1 }} },
    },
    {
        { 1, 3, 4 },
        { 256ul, {{1}, {1}, {1}, {1}}, {0.f}, {255.f}, {0.f}, {25.5f} },
        { std::vector<float>(4 * 4, 2.f), ngraph::element::f32, ngraph::Shape{ 4, 4 } },
        { 256ul, {{1}, {1}, {1}, {1}}, {-128.f}, {127.f}, {-128.f}, {127.f} },
        { {}, {}, {} },
    },
    {
        { 2, 3 },
        { 256ul, {{1}, {1}, {2, 1}, {2, 1}}, {-10.f}, {5.f}, {-10.f, -5.f}, {5.f, 5.f} },
        { std::vector<float>{1, 2, 3, 4, 5, 6}, ngraph::element::f32, ngraph::Shape{ 2, 3 } },
        { 256ul, {{1}, {1}, {1}, {1}}, {-128.f}, {127.f}, {-12.8f}, {12.7f} },
        { {}, {}, {} },
    },
    {
        { 2, 3 },
        { 256ul, {{1}, {1}, {2, 1}, {2, 1}}, {-10.f}, {5.f}, {-10.f, -5.f}, {5.f, 5.f} },
        { std::vector<float>{1, 2, 3, 4, 5, 6}, ngraph::element::i8, ngraph::Shape{ 2, 3 } },
        {},
        { ngraph::element::f32, {}, {0.1f} },
    }
};

INSTANTIATE_TEST_CASE_P(smoke_LPT, MatMulWithConstantTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(testValues)),
    MatMulWithConstantTransformation::getTestCaseName);
}  // namespace
