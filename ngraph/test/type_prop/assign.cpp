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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/variable.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, assign_variable_not_found)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    try
    {
        auto space_to_depth = make_shared<opset5::Assign>(A, "variable_id");
        // Should have thrown, so fail if it didn't
        FAIL() << "Should not find variable with variable_id";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Can't find variable with id = variable_id"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, assign_deduce)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    auto read_value = make_shared<opset5::ReadValue>(input, "variable_id");
    auto assign = make_shared<opset5::Assign>(read_value, "variable_id");

    ASSERT_EQ(assign->get_element_type(), element::f32);
    ASSERT_EQ(assign->get_shape(), (Shape{1, 2, 64, 64}));
}

TEST(type_prop, assign_read_value_new_shape)
{
    auto input = make_shared<op::Parameter>(element::f16, Shape{4, 3, 2, 1});

    auto variable =
        std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "ID"});
    auto read_value = make_shared<opset6::ReadValue>(input, variable);
    auto assign = make_shared<opset6::Assign>(read_value, variable);

    ASSERT_EQ(assign->get_element_type(), element::f16);
    ASSERT_EQ(assign->get_shape(), (Shape{4, 3, 2, 1}));

    auto f = std::make_shared<Function>(ResultVector{}, SinkVector{assign}, ParameterVector{input});

    input->set_partial_shape({3, {4, 5}, 8});
    f->validate_nodes_and_infer_types();

    ASSERT_EQ(assign->get_element_type(), element::f16);
    ASSERT_EQ(assign->get_output_partial_shape(0), (PartialShape{3, {4, 5}, 8}));
    ASSERT_EQ(variable->get_info().data_type, element::f16);
    ASSERT_EQ(variable->get_info().data_shape, (PartialShape{3, {4, 5}, 8}));
}
