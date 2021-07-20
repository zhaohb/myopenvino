// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/mat_mul_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <queue>
#include <ie_core.hpp>

#include "ngraph/op/op.hpp"
#include <transformations/init_node_info.hpp>
#include "low_precision_transformations/mat_mul_transformation.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/mat_mul_function.hpp"

namespace LayerTestsDefinitions {

std::string MatMulTransformation::getTestCaseName(testing::TestParamInfo<MatMulTransformationParams> obj) {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    MatMulTransformationTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.inputShape1 << "_" <<
        testValues.fqOnData1 << "_" <<
        testValues.inputShape2 << "_" <<
        testValues.fqOnData2;

    return result.str();
}

InferenceEngine::Blob::Ptr MatMulTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    if ((info.name() != "input1") && (info.name() != "input2")) {
        THROW_IE_EXCEPTION << "unexpected layer name " << info.name();
    }

    size_t low;
    size_t high;
    if (info.name() == "input1") {
        low = 1ul;
        high = 5ul;
    } else if (info.name() == "input2") {
        low = 5ul;
        high = 10ul;
    } else {
        THROW_IE_EXCEPTION << "unexpected input name " << info.name();
    }

    return FuncTestUtils::createAndFillBlobConsistently(info.getTensorDesc(), high - low, low, 1ul);
}

void MatMulTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    MatMulTransformationTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    function = ngraph::builder::subgraph::MatMulFunction::getOriginal(
        precision,
        testValues.inputShape1,
        testValues.fqOnData1,
        testValues.inputShape2,
        testValues.fqOnData2);

    ngraph::pass::InitNodeInfo().run_on_function(function);
    validate();
}

void MatMulTransformation::validate() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    MatMulTransformationTestValues testValues;
    std::tie(precision, inputShape, targetDevice, testValues) = this->GetParam();

    const auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams();
    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));

    const auto output = transformed->get_output_op(0);
    const auto scaleShift = output->get_input_node_shared_ptr(0);
    const std::string typeName = scaleShift->get_type_name();
    ASSERT_EQ("ScaleShiftIE", typeName);
}

void MatMulTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<3>(GetParam());
    const auto actualType = getRuntimePrecision(params.expectedKernelName);

    EXPECT_EQ(actualType, params.expectedRuntimePrecision);
}

TEST_P(MatMulTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
