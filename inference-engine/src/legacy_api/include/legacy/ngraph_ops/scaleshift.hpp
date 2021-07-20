// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(ScaleShiftIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"ScaleShiftIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    ScaleShiftIE(const Output<Node>& data_batch,
                 const Output<Node>& weights,
                 const Output<Node>& bias,
                 const element::Type output_type = element::undefined);
    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    element::Type output_type;
};

}  // namespace op
}  // namespace ngraph
