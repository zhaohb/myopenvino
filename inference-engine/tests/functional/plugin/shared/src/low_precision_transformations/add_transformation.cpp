// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/add_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/add_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace LayerTestsDefinitions {

std::string AddTransformation::getTestCaseName(testing::TestParamInfo< AddTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    AddTestValues param;
    std::tie(netPrecision, inputShapes, targetDevice, param) = obj.param;

    if (!param.precisionOnActivations.empty()) {
        params.precisionsOnActivations = param.precisionOnActivations;
    }

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params) <<
        (param.broadcast ? "_broadcast" : "");
    if (!param.fakeQuantize1.empty()) {
        result << "_on_branch1_" <<
            param.fakeQuantize1.inputLowValues[0] << "_" <<
            param.fakeQuantize1.inputHighValues[0] << "_" <<
            param.fakeQuantize1.outputLowValues[0] << "_" <<
            param.fakeQuantize1.outputHighValues[0];
    }
    if (!param.fakeQuantize2.empty()) {
        result << "_on_branch2_" <<
            param.fakeQuantize2.inputLowValues[0] << "_" <<
            param.fakeQuantize2.inputHighValues[0] << "_" <<
            param.fakeQuantize2.outputLowValues[0] << "_" <<
            param.fakeQuantize2.outputHighValues[0];
    }
    return result.str();
}

void AddTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    AddTestValues param;
    std::tie(precision, inputShape, targetDevice, param) = this->GetParam();

    function = ngraph::builder::subgraph::AddFunction::getOriginal(
        precision, inputShape, param.broadcast,
        param.fakeQuantize1, param.fakeQuantize2);

    ngraph::pass::InitNodeInfo().run_on_function(function);
    validate();
}

void AddTransformation::validate() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    AddTestValues param;
    std::tie(precision, inputShape, targetDevice, param) = this->GetParam();

    const auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));

    const auto output = transformed->get_output_op(0);
    if ((!param.fakeQuantize1.empty()) && (!param.fakeQuantize2.empty())) {
        const auto scaleShift = output->get_input_node_shared_ptr(0);
        const std::string typeName = scaleShift->get_type_name();
        ASSERT_EQ("ScaleShiftIE", typeName);
    }
}

TEST_P(AddTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
