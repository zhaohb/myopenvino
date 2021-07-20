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

#include "ngraph/op/scatter_update.hpp"
#include "itt.hpp"
#include "ngraph/runtime/reference/scatter_update.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::ScatterUpdate::type_info;

op::v3::ScatterUpdate::ScatterUpdate(const Output<Node>& data,
                                     const Output<Node>& indices,
                                     const Output<Node>& updates,
                                     const Output<Node>& axis)
    : util::ScatterBase(data, indices, updates, axis)
{
}

shared_ptr<Node> op::v3::ScatterUpdate::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v3_ScatterUpdate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v3::ScatterUpdate>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

namespace scatter_update
{
    template <element::Type_t ET>
    std::vector<int64_t> get_indices(const HostTensorPtr& in)
    {
        auto data_ptr = in->get_data_ptr<ET>();
        return std::vector<int64_t>(data_ptr, data_ptr + in->get_element_count());
    }
}

#define GET_INDICES(a, ...)                                                                        \
    case element::Type_t::a:                                                                       \
    {                                                                                              \
        NGRAPH_OP_SCOPE(OV_PP_CAT3(get_scatter_update_indices, _, a));                             \
        indices_casted_vector = scatter_update::get_indices<element::Type_t::a>(__VA_ARGS__);      \
    }                                                                                              \
    break;

bool op::v3::ScatterUpdate::evaluate_scatter_update(const HostTensorVector& outputs,
                                                    const HostTensorVector& inputs) const
{
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];
    const auto& axis = inputs[3];
    const auto& out = outputs[0];

    const auto elem_size = data->get_element_type().size();
    out->set_shape(data->get_shape());

    NGRAPH_CHECK(axis->get_element_type().is_integral_number(),
                 "axis element type is not integral data type");

    int64_t axis_val = host_tensor_2_vector<int64_t>(axis)[0];
    if (axis_val < 0)
    {
        axis_val =
            ngraph::normalize_axis(this, axis_val, static_cast<int64_t>(data->get_shape().size()));
    }

    std::vector<int64_t> indices_casted_vector;
    switch (indices->get_element_type())
    {
        GET_INDICES(i8, indices);
        GET_INDICES(i16, indices);
        GET_INDICES(i32, indices);
        GET_INDICES(i64, indices);
        GET_INDICES(u8, indices);
        GET_INDICES(u16, indices);
        GET_INDICES(u32, indices);
        GET_INDICES(u64, indices);
    default: return false;
    }

    runtime::reference::scatter_update(data->get_data_ptr<char>(),
                                       indices_casted_vector.data(),
                                       updates->get_data_ptr<char>(),
                                       axis_val,
                                       out->get_data_ptr<char>(),
                                       elem_size,
                                       data->get_shape(),
                                       indices->get_shape(),
                                       updates->get_shape());

    return true;
}

bool op::v3::ScatterUpdate::evaluate(const HostTensorVector& outputs,
                                     const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v3_ScatterUpdate_evaluate);
    return evaluate_scatter_update(outputs, inputs);
}
