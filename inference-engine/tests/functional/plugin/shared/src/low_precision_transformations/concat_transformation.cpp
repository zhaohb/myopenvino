// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/concat_function.hpp"

namespace LayerTestsDefinitions {

std::string ConcatTransformation::getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
    ngraph::element::Type precision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ConcatTransformationTestValues testValues;
    std::tie(precision, inputShapes, targetDevice, testValues) = obj.param;

    const auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, inputShapes, targetDevice, params) << testValues.fqOnData1 << testValues.fqOnData2;
    return result.str();
}

InferenceEngine::Blob::Ptr ConcatTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::SizeVector inputShape;
    ngraph::element::Type netPrecision;
    std::string targetDevice;
    ConcatTransformationTestValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, testValues) = this->GetParam();

    const auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();

    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(
        params.precisionsOnActivations[0],
        info.getTensorDesc(),
        k);
}

void ConcatTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    ngraph::element::Type precision;
    ConcatTransformationTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::ConcatFunction::getOriginal(
        precision,
        inputShape,
        testValues.fqOnData1,
        testValues.fqOnData2);

    validate();
}

void ConcatTransformation::validate() {
    ngraph::element::Type precision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ConcatTransformationTestValues testValues;
    std::tie(precision, inputShapes, targetDevice, testValues) = GetParam();

    const auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));

    const auto output = transformed->get_output_op(0);
    const auto scaleShift = output->get_input_node_shared_ptr(0);
    const std::string typeName = scaleShift->get_type_name();
    ASSERT_EQ("ScaleShiftIE", typeName);
}

TEST_P(ConcatTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
