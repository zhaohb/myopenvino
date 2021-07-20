// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MultiplyToGroupConvolutionFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::Shape& inputShape,
        const ngraph::element::Type& precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const bool haveMultiplyWithNoConstBeforeDequantization);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fqOnData);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::Shape& inputShape,
        const ngraph::element::Type& precision,
        const std::shared_ptr<ngraph::opset1::Constant>& weights,
        const std::shared_ptr<ngraph::opset1::Constant>& biases,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
