// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/interpolate_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/interpolate_function.hpp"

namespace LayerTestsDefinitions {

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& values) {
    os << "{";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ",";
        }
    }
    os << "}";
    return os;
}

std::string InterpolateTransformation::getTestCaseName(testing::TestParamInfo<InterpolateTransformationParams> obj) {
    ngraph::element::Type precision;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    std::string targetDevice;
    interpAttributes attributes;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    std::tie(precision, shapes, targetDevice, attributes) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, shapes.first, targetDevice, params) << "_" <<
        shapes.second << "_" <<
        attributes.align_corners << "_" <<
        attributes.antialias << "_" <<
        attributes.axes << "_" <<
        attributes.mode << "_" <<
        attributes.pads_begin << "_" <<
        attributes.pads_end;
    return result.str();
}

void InterpolateTransformation::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);
    ngraph::element::Type precision;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    interpAttributes attributes;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    std::tie(precision, shapes, targetDevice, attributes) = this->GetParam();

    ngraph::op::InterpolateAttrs interpAttrs;
    interpAttrs.axes = attributes.axes;
    interpAttrs.mode = attributes.mode;
    interpAttrs.align_corners = attributes.align_corners;
    interpAttrs.antialias = attributes.antialias;
    interpAttrs.pads_begin = attributes.pads_begin;
    interpAttrs.pads_end = attributes.pads_end;

    function = ngraph::builder::subgraph::InterpolateFunction::getOriginal(precision, shapes.first, shapes.second, interpAttrs);

    validate();
}

void InterpolateTransformation::validate() {
    ngraph::element::Type precision;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    std::string targetDevice;
    interpAttributes attributes;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    std::tie(precision, shapes, targetDevice, attributes) = this->GetParam();

    const auto transformed = transformNGraph(params, getLowPrecisionTransformationsNGraph(params));

    const auto output = transformed->get_output_op(0);
    const auto scaleShift = output->get_input_node_shared_ptr(0);
    const std::string typeName = scaleShift->get_type_name();
    if (attributes.mode == "nearest") {
        ASSERT_EQ("ScaleShiftIE", typeName);
    } else {
        ASSERT_TRUE("Interp" == typeName || "Interpolate" == typeName);
    }
}

TEST_P(InterpolateTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
