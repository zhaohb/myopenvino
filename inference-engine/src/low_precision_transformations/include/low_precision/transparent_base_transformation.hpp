// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API TransparentBaseTransformation : public LayerTransformation {
public:
    TransparentBaseTransformation(const Params& params) : LayerTransformation(params) {}
    ~TransparentBaseTransformation() override {};
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
