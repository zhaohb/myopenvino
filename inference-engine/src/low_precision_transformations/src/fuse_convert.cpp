﻿// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fuse_convert.hpp"

#include <memory>
#include <string>
#include <vector>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void FuseConvertTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Multiply>({ make_op_label<opset1::Convert>(), make_op_label<opset1::Constant>() }));

    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Subtract>({ make_op_label<opset1::Convert>(), make_op_label<opset1::Constant>() }));

    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Add>({ make_op_label<opset1::Convert>(), make_op_label<opset1::Constant>() }));
}

std::shared_ptr<Node> removeConvertIfPossibleForSubtract(
    const std::shared_ptr<opset1::Convert>& convert,
    const std::shared_ptr<opset1::Subtract>& subtract) {
    std::shared_ptr<Node> newSubtract;

    const element::Type precisionBeforeConvert = convert->input(0).get_element_type();
    if (NetworkHelper::checkConstantValuePrecision(precisionBeforeConvert, subtract->get_input_node_shared_ptr(1))) {
        newSubtract = std::make_shared<ngraph::op::TypeRelaxed<opset1::Subtract>>(
            std::vector<ngraph::element::Type>{ element::f32, element::f32 }, std::vector<ngraph::element::Type>{},
            ngraph::op::TemporaryReplaceOutputType(convert->get_input_source_output(0), element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(subtract->get_input_node_shared_ptr(1), element::f32).get());
        NetworkHelper::setOutDataPrecisionForTypeRelaxed(newSubtract, subtract->get_output_element_type(0));
        replace_node(subtract, newSubtract);
    }

    return newSubtract;
}

bool FuseConvertTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    const auto op = m.get_match_root();
    if (!canBeTransformed(context, op)) {
        return false;
    }

    const auto convert = as_type_ptr<opset1::Convert>(op->get_input_node_shared_ptr(0));
    std::shared_ptr<Node> parent = convert->get_input_node_shared_ptr(0);

    if (is_type<opset1::Constant>(parent)) {
        auto convertedConstant = fold<opset1::Convert>(parent, convert->get_convert_element_type());
        NetworkHelper::copyInfo(parent, convertedConstant);
        replace_node(convert, convertedConstant);
    } else {
        std::shared_ptr<Node> newOp;
        if (is_type<opset1::Subtract>(op)) {
            auto subtract = as_type_ptr<opset1::Subtract>(op);
            newOp = removeConvertIfPossibleForSubtract(convert, subtract);
        } else if (is_type<opset1::Multiply>(op)) {
            newOp = std::make_shared<ngraph::op::TypeRelaxed<opset1::Multiply>>(
                    std::vector<ngraph::element::Type>{ element::f32, element::f32 }, std::vector<ngraph::element::Type>{},
                    ngraph::op::TemporaryReplaceOutputType(convert->get_input_source_output(0), element::f32).get(),
                    ngraph::op::TemporaryReplaceOutputType(op->get_input_node_shared_ptr(1), element::f32).get());
            NetworkHelper::setOutDataPrecisionForTypeRelaxed(newOp, op->get_output_element_type(0));
            replace_node(op, newOp);
        } else if (is_type<opset1::Add>(op)) {
            newOp = std::make_shared<ngraph::op::TypeRelaxed<opset1::Add>>(
                    std::vector<ngraph::element::Type>{ element::f32, element::f32 }, std::vector<ngraph::element::Type>{},
                    ngraph::op::TemporaryReplaceOutputType(convert->get_input_source_output(0), element::f32).get(),
                    ngraph::op::TemporaryReplaceOutputType(op->get_input_node_shared_ptr(1), element::f32).get());
            NetworkHelper::setOutDataPrecisionForTypeRelaxed(newOp, op->get_output_element_type(0));
            replace_node(op, newOp);
        }

        if (newOp != nullptr) {
            ngraph::copy_runtime_info({ convert, op }, newOp);
            newOp->set_friendly_name(op->get_friendly_name());
        }
    }

    return true;
}

bool FuseConvertTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    const auto convert = as_type_ptr<opset1::Convert>(op->get_input_node_shared_ptr(0));
    // issue #40395
    if (convert == nullptr) {
        return false;
    }

    const auto destType = convert->get_destination_type();
    if ((destType != element::f16) && (destType != element::f32)) {
        return false;
    }

    return true;
}

bool FuseConvertTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
