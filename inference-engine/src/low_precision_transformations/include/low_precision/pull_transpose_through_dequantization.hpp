// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API PullTransposeThroughDequantization;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

class ngraph::pass::low_precision::PullTransposeThroughDequantization : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PullTransposeThroughDequantization(const std::vector<ngraph::element::Type>& inputPrecisions = {});
};
