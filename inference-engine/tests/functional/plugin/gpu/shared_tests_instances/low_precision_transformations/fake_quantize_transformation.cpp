// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fake_quantize_transformation.hpp"
#include "common_test_utils/test_constants.hpp"
#include "lpt_ngraph_functions/fake_quantize_function.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::pass::low_precision;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    // can not be passed to plugin
    // nGraph: I8 -> FP32 Convert is not supported
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams(),
    // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8(),
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<ngraph::builder::subgraph::FakeQuantizeOnData> fakeQuantizeOnDataValues = {
    { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
    { 256ul, { 1ul }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
    // nGraph: I8->FP32 Convert is not supported
    // { 256ul, {}, { -1.28f} , { 1.27f }, { -1.28f} , { 1.27f } },
    // { 256ul, { 1ul }, { -1.28f} , { 1.27f } }
};

INSTANTIATE_TEST_CASE_P(smoke_LPT, FakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::Shape({ 1, 32, 72, 48 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(fakeQuantizeOnDataValues)),
    FakeQuantizeTransformation::getTestCaseName);
}  // namespace
