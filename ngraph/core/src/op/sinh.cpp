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

#include "itt.hpp"

#include "ngraph/op/cosh.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sinh.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/sinh.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Sinh::type_info;

op::Sinh::Sinh(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Sinh::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Sinh_visit_attributes);
    return true;
}

shared_ptr<Node> op::Sinh::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Sinh_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Sinh>(new_args.at(0));
}

namespace sinhop
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::sinh<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_sinh(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_sinh, boolean, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_sinh, i32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_sinh, i64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_sinh, u32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_sinh, u64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_sinh, f16, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_sinh, f32, arg0, out, count);
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::Sinh::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Sinh_evaluate);
    return sinhop::evaluate_sinh(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}
