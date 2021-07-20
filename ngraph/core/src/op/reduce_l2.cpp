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

#include "ngraph/op/reduce_l2.hpp"
#include "itt.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/reduce_l2.hpp"
#include "ngraph/shape_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v4::ReduceL2::type_info;

op::v4::ReduceL2::ReduceL2(const Output<Node>& arg,
                           const Output<Node>& reduction_axes,
                           bool keep_dims)
    : ArithmeticReductionKeepDims(arg, reduction_axes, keep_dims)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v4::ReduceL2::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}

shared_ptr<Node> op::v4::ReduceL2::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v4_ReduceL2_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v4::ReduceL2>(new_args.at(0), new_args.at(1), get_keep_dims());
}

namespace reduce_l2
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg,
                  const HostTensorPtr& out,
                  const AxisSet& axes,
                  bool keep_dims)
    {
        out->set_shape(reduce(arg->get_shape(), axes, keep_dims));
        runtime::reference::reduce_l2(
            arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), arg->get_shape(), axes, keep_dims);
        return true;
    }

    bool evaluate_reduce_l2(const HostTensorPtr& arg,
                            const HostTensorPtr& out,
                            const AxisSet& axes,
                            bool keep_dims)
    {
        bool rc = true;
        switch (arg->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_reduce_l2, bf16, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_reduce_l2, f16, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_reduce_l2, f32, arg, out, axes, keep_dims);
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v4::ReduceL2::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v4_ReduceL2_evaluate);
    return reduce_l2::evaluate_reduce_l2(
        inputs[0], outputs[0], get_reduction_axes(), get_keep_dims());
}
