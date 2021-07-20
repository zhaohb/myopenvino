// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/clamp.hpp"
#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ClampTransformation::ClampTransformation(const Params& params) : LayerTransformation(params) {}

void ClampTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(pass,
               context,
               make_op_pattern<opset1::Clamp>({ make_op_label<opset1::Multiply>() }));
}

bool ClampTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) const {
    auto subWithTheSameValues = [](std::shared_ptr<ngraph::opset1::Subtract> sub) {
        if (sub == nullptr) {
            return false;
        }

        auto constant = as_type_ptr<ngraph::opset1::Constant>(sub->get_input_node_shared_ptr(1));
        if (constant == nullptr) {
            const auto convert = sub->get_input_node_shared_ptr(1);
            if (!is_type<ngraph::opset1::Convert>(convert)) {
                return false;
            }
            constant = as_type_ptr<ngraph::opset1::Constant>(convert->get_input_node_shared_ptr(0));
        }

        if (constant == nullptr) {
            return false;
        }

        return NetworkHelper::isScalarLike(constant);
    };

    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    std::shared_ptr<Node> clamp = NetworkHelper::separateInStandaloneBranch(m.get_match_root());
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(clamp);

    const bool moveSubtract = subWithTheSameValues(dequantization.subtract);
    // issue #43136
    if (!moveSubtract && (dequantization.subtract != nullptr)) {
        return false;
    }

    const auto newClamp = as_type_ptr<opset1::Clamp>(moveDequantizationAfter(context, clamp, dequantization, false, moveSubtract));

    std::shared_ptr<ngraph::opset1::Clamp> replacement;
    {
        double min = newClamp->get_min();
        double max = newClamp->get_max();

        if (dequantization.multiply != nullptr) {
            double scale = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1))->cast_vector<double>()[0];
            if (scale < 0.0) {
                std::swap(min, max);
            }
            min /= scale;
            max /= scale;
        }

        if (dequantization.subtract != nullptr && moveSubtract) {
            double shift = as_type_ptr<opset1::Constant>(dequantization.subtractConstant)->cast_vector<double>()[0];
            min += shift;
            max += shift;
        }

        replacement = std::make_shared<ngraph::opset1::Clamp>(newClamp->get_input_node_shared_ptr(0), min, max);
    }
    replace_node(newClamp, replacement);

    element::Type outputClampType = dequantization.multiply ?
        dequantization.multiply->get_output_element_type(0) :
        dequantization.subtract->get_output_element_type(0);
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(replacement, outputClampType);
    return true;
}

bool ClampTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(op);
    if (dequantization.multiply == nullptr) {
        return false;
    }

    return NetworkHelper::isScalarLike(dequantization.multiplyConstant);
}

bool ClampTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
