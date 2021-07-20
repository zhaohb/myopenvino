//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "backend_visibility.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/pass.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph
{
    namespace pass
    {
        NodeVector explicit_broadcast(std::shared_ptr<Node>& node);
        class ImplicitBroadcastElimination;
    }
}

class BACKEND_API ngraph::pass::ImplicitBroadcastElimination : public ngraph::pass::NodePass
{
public:
    bool run_on_node(std::shared_ptr<ngraph::Node> node) override;
};

NGRAPH_SUPPRESS_DEPRECATED_END
