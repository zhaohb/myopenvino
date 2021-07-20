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

#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

op::util::ArithmeticReductionKeepDims::ArithmeticReductionKeepDims(
    const ngraph::Output<ngraph::Node>& arg,
    const ngraph::Output<ngraph::Node>& reduction_axes,
    bool keep_dims)
    : ArithmeticReduction(arg, reduction_axes)
    , m_keep_dims{keep_dims}
{
}

bool ngraph::op::util::ArithmeticReductionKeepDims::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_util_ArithmeticReductionKeepDims_visit_attributes);
    visitor.on_attribute("keep_dims", m_keep_dims);
    return true;
}

void op::util::ArithmeticReductionKeepDims::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_util_ArithmeticReductionKeepDims_validate_and_infer_types);
    if (m_keep_dims)
    {
        auto input_shape = get_input_partial_shape(0);
        auto input_rank = input_shape.rank();
        PartialShape result_shape{PartialShape::dynamic()};

        if (input_rank.is_static())
            result_shape = PartialShape::dynamic(input_rank);

        const auto& axes = get_constant_from_source(input_value(1));
        if (input_rank.is_static() && axes)
        {
            AxisSet reduction_axes;
            auto reduction_axes_val = axes->cast_vector<int64_t>();
            for (auto axis : reduction_axes_val)
            {
                try
                {
                    axis = normalize_axis(this, axis, input_rank);
                }
                catch (const ngraph_error&)
                {
                    NODE_VALIDATION_CHECK(this,
                                          false,
                                          "Reduction axis (",
                                          axis,
                                          ") is out of bounds ",
                                          "(argument shape: ",
                                          input_shape,
                                          ", reduction axes: ",
                                          reduction_axes,
                                          ")");
                }
                reduction_axes.insert(axis);
            }

            std::vector<Dimension> dims;
            for (size_t i = 0; i < input_rank.get_length(); i++)
            {
                if (reduction_axes.count(i) == 0)
                {
                    dims.push_back(input_shape[i]);
                }
                else
                {
                    dims.emplace_back(Dimension{1});
                }
            }
            result_shape = PartialShape(dims);
        }
        set_input_is_relevant_to_shape(1);
        set_output_type(0, get_input_element_type(0), result_shape);
    }
    else
    {
        ArithmeticReduction::validate_and_infer_types();
    }
}
