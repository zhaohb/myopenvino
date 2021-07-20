// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>
#include <legacy/ngraph_ops/gather_ie.hpp>

#include "ngraph/op/gather.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/op/unsqueeze.hpp"


namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertGatherToGatherIEMatcher);

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *     This transformation converts opset1::Gather to legacy GatherIE
 *     GatherIE takes axes as value and if indices input has empty shape (scalar)
 *     we unsqueeze indices input and squeeze GatherIE output.
 */

class ngraph::pass::ConvertGatherToGatherIEMatcher : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGatherToGatherIEMatcher();
};
