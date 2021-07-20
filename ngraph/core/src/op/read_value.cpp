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

#include "ngraph/op/read_value.hpp"
#include "itt.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::ReadValueBase, "ReadValueBase", 0);
NGRAPH_RTTI_DEFINITION(op::v3::ReadValue, "ReadValue", 3);
NGRAPH_RTTI_DEFINITION(op::v6::ReadValue, "ReadValue", 6);

op::v3::ReadValue::ReadValue(const Output<Node>& init_value, const std::string& variable_id)
    : ReadValueBase({init_value})
    , m_variable_id(variable_id)
{
    constructor_validate_and_infer_types();
}

void op::v3::ReadValue::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v3_ReadValue_validate_and_infer_types);
    auto arg_t = get_input_element_type(0);
    auto output_shape = get_input_partial_shape(0);

    VariableInfo info = {output_shape, arg_t, m_variable_id};
    if (m_variable == nullptr)
        m_variable = std::make_shared<Variable>(info);
    else
        m_variable->update(info);
    set_output_type(0, arg_t, output_shape);
}

shared_ptr<Node> op::v3::ReadValue::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v3_ReadValue_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ReadValue>(new_args.at(0), m_variable_id);
}

bool op::v3::ReadValue::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v3_ReadValue_visit_attributes);
    visitor.on_attribute("variable_id", m_variable_id);
    return true;
}

op::v6::ReadValue::ReadValue(const Output<Node>& init_value, const shared_ptr<Variable>& variable)
    : ReadValueBase({init_value})
{
    m_variable = variable;
    constructor_validate_and_infer_types();
}

void op::v6::ReadValue::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v6_ReadValue_validate_and_infer_types);
    const auto arg_t = get_input_element_type(0);
    auto output_shape = get_input_partial_shape(0);
    NGRAPH_CHECK(m_variable, "Variable is not initialized.");
    VariableInfo var_info = {output_shape, element::dynamic, m_variable->get_info().variable_id};
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(var_info.data_type, m_variable->get_info().data_type, arg_t),
        "Variables types are inconsistent.");
    NODE_VALIDATION_CHECK(
        this,
        PartialShape::merge_into(var_info.data_shape, m_variable->get_info().data_shape),
        "Variable shape and output shape are inconsistent.");
    m_variable->update(var_info);
    set_output_type(0, arg_t, output_shape);
}

shared_ptr<Node> op::v6::ReadValue::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v6_ReadValue_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ReadValue>(new_args.at(0), m_variable);
}

bool op::v6::ReadValue::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v6_ReadValue_visit_attributes);
    visitor.on_attribute("variable_id", m_variable);
    return true;
}

void op::v6::ReadValue::revalidate_and_infer_types()
{
    VariableInfo var_info{
        PartialShape::dynamic(), element::dynamic, m_variable->get_info().variable_id};
    m_variable->update(var_info);
    Node::revalidate_and_infer_types();
}