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

#include <memory>

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API Unsqueeze : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                Unsqueeze() = default;
                Unsqueeze(const Output<Node>& data, const Output<Node>& axes);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool evaluate_lower(const HostTensorVector& output_values) const override;
                bool evaluate_upper(const HostTensorVector& output_values) const override;

                bool constant_fold(OutputVector& output_values,
                                   const OutputVector& inputs_values) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }
        using v0::Unsqueeze;
    }
}
