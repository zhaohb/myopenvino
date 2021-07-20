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
#include <numeric>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/shuffle_channels.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

constexpr NodeTypeInfo op::ShuffleChannels::type_info;

op::ShuffleChannels::ShuffleChannels(const Output<Node>& data,
                                     const int64_t axis,
                                     const int64_t group)
    : Op({data})
    , m_axis(axis)
    , m_group{group}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::ShuffleChannels::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_ShuffleChannels_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("group", m_group);
    return true;
}

size_t op::ShuffleChannels::get_zero_based_axis() const
{
    if (m_axis >= 0)
    {
        return m_axis;
    }
    else
    {
        if (!get_input_partial_shape(0).rank().is_dynamic())
        {
            return m_axis + get_input_partial_shape(0).rank().get_length();
        }
        else
        {
            throw ngraph_error("Cannot request zero-based axis with a input of unknown rank");
        }
    }
}

void op::ShuffleChannels::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_ShuffleChannels_validate_and_infer_types);
    const auto& data_type = get_input_element_type(0);
    if (get_input_partial_shape(0).is_static())
    {
        const auto shape = get_input_shape(0);

        NODE_VALIDATION_CHECK(
            this, shape.size() >= 1, "The input tensor's shape is expected to be at least 1D.");
        size_t axis_zb = get_zero_based_axis();

        NODE_VALIDATION_CHECK(this,
                              axis_zb < shape.size(),
                              "The 'axis' parameter for ShuffleChannels has to point to one of the "
                              "input tensor's shape dimensions.");

        NODE_VALIDATION_CHECK(
            this, m_group >= 1, "The 'group' parameter must be greater or equal to 1.");

        const auto channel_dim_size = shape.at(axis_zb);
        NODE_VALIDATION_CHECK(
            this,
            channel_dim_size % m_group == 0,
            "The channel dimension size has to be a multiple of the groups parameter value.");
        set_output_size(1);
        set_output_type(0, data_type, shape);
    }
    else
    {
        set_output_type(0, data_type, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::ShuffleChannels::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_ShuffleChannels_clone_with_new_inputs);
    if (new_args.size() != 1)
    {
        throw ngraph_error("Expected 1 element in new_args for the ShuffleChannels op but got " +
                           std::to_string(new_args.size()));
    }

    return make_shared<ShuffleChannels>(new_args.at(0), m_axis, m_group);
}

Shape op::ShuffleChannels::get_pre_shuffle_shape(const Shape& data_shape) const
{
    const Shape& ds = data_shape;

    // in general the resulting shape should contain the following values:
    // [0]: ds[0] * ds[1] * ... * ds[m_axis-1] (or 1 if m_axis == 0)
    // [1]: m_group
    // [2]: ds[axis] / m_group
    // [3]: ds[axis+1] * ds[axis+2] * ... * ds[ds.size()-1] (or 1 if m_axis points to the last elem
    //                                                       of ds)
    Shape res(4, 1);

    size_t axis_zb = get_zero_based_axis();
    for (size_t i = 0; i < axis_zb; ++i)
    {
        res[0] *= ds[i];
    }

    res[1] = m_group;
    res[2] = ds[axis_zb] / m_group;

    for (size_t i = axis_zb + 1; i < ds.size(); ++i)
    {
        res[3] *= ds[i];
    }

    return res;
}

bool op::ShuffleChannels::evaluate_shuffle_channels(const HostTensorVector& outputs,
                                                    const HostTensorVector& inputs) const
{
    const auto arg = inputs[0]->get_data_ptr<const char>();
    auto out = outputs[0]->get_data_ptr<char>();
    Shape data_shape = inputs[0]->get_shape();
    const Shape& ds = data_shape;
    size_t elem_size = inputs[0]->get_element_type().size();

    Shape reshaped_out_shape(4, 1);
    size_t axis_zb = m_axis >= 0 ? m_axis : m_axis + data_shape.size();
    for (size_t i = 0; i < axis_zb; ++i)
    {
        reshaped_out_shape[0] *= ds[i];
    }

    reshaped_out_shape[1] = m_group;
    reshaped_out_shape[2] = ds[axis_zb] / m_group;

    for (size_t i = axis_zb + 1; i < ds.size(); ++i)
    {
        reshaped_out_shape[3] *= ds[i];
    }

    // first reshape from data_shape to reshaped_out_shape is skipped since it doesn't affect
    // out
    // data

    Shape transpose_axes_order = {0, 2, 1, 3};
    Shape transposed_shape(transpose_axes_order.size());

    for (size_t i = 0; i < transpose_axes_order.size(); ++i)
    {
        transposed_shape[i] = data_shape.at(transpose_axes_order.at(i));
    }
    auto axis_vector = AxisVector{begin(transpose_axes_order), end(transpose_axes_order)};
    runtime::opt_kernel::reshape(
        arg, out, reshaped_out_shape, axis_vector, transposed_shape, elem_size);

    // last reshape from transposed_shape to data_shape is skipped since it doesn't affect out
    // data
    return true;
}
bool op::ShuffleChannels::evaluate(const HostTensorVector& outputs,
                                   const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_ShuffleChannels_evaluate);
    return evaluate_shuffle_channels(outputs, inputs);
}
