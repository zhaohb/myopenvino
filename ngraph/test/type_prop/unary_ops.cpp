//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//  Licensed under the Apache License, Version 2.0 (the "License");
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

#include <vector>
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace ngraph;

template <class T>
class UnaryOperator : public testing::Test
{
};

TYPED_TEST_CASE_P(UnaryOperator);

TYPED_TEST_P(UnaryOperator, shape_inference_Shape1)
{
    auto param = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto op = std::make_shared<TypeParam>(param);
    ASSERT_EQ(op->get_shape(), (Shape{2, 2}));
}
TYPED_TEST_P(UnaryOperator, shape_inference_Shape2)
{
    auto param = std::make_shared<op::Parameter>(element::i32, Shape{21, 15, 2});
    auto op = std::make_shared<TypeParam>(param);
    ASSERT_EQ(op->get_shape(), (Shape{21, 15, 2}));
}

TYPED_TEST_P(UnaryOperator, input_type_inference_F32)
{
    auto param = std::make_shared<op::Parameter>(element::f32, Shape{10, 2, 2});
    auto op = std::make_shared<TypeParam>(param);
    ASSERT_EQ(op->get_element_type(), element::f32);
}

TYPED_TEST_P(UnaryOperator, input_type_inference_I64)
{
    auto param = std::make_shared<op::Parameter>(element::i64, Shape{41, 28, 2});
    auto op = std::make_shared<TypeParam>(param);
    ASSERT_EQ(op->get_element_type(), element::i64);
}

TYPED_TEST_P(UnaryOperator, input_type_inference_U16)
{
    auto param = std::make_shared<op::Parameter>(element::u16, Shape{100, 200, 7});
    auto op = std::make_shared<TypeParam>(param);
    ASSERT_EQ(op->get_element_type(), element::u16);
}

TYPED_TEST_P(UnaryOperator, incompatible_input_type_Shape1)
{
    const auto param = std::make_shared<op::Parameter>(element::boolean, Shape{100, 2, 50});
    ASSERT_THROW(std::make_shared<TypeParam>(param), ngraph::NodeValidationFailure);
}

TYPED_TEST_P(UnaryOperator, incompatible_input_type_Shape2)
{
    const auto param = std::make_shared<op::Parameter>(element::boolean, Shape{40, 17, 50});
    ASSERT_THROW(std::make_shared<TypeParam>(param), ngraph::NodeValidationFailure);
}

TYPED_TEST_P(UnaryOperator, dynamic_rank_input_shape_2D)
{
    const PartialShape param_shape{Dimension::dynamic(), 10};
    const auto param = std::make_shared<op::Parameter>(element::f32, param_shape);
    const auto op = std::make_shared<TypeParam>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{Dimension(), 10}));
}

TYPED_TEST_P(UnaryOperator, dynamic_rank_input_shape_3D)
{
    const PartialShape param_shape{100, Dimension::dynamic(), 58};
    const auto param = std::make_shared<op::Parameter>(element::f32, param_shape);
    const auto op = std::make_shared<TypeParam>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{100, Dimension(), 58}));
}

TYPED_TEST_P(UnaryOperator, dynamic_rank_input_shape_full)
{
    const auto param = std::make_shared<op::Parameter>(element::f64, PartialShape::dynamic());
    const auto op = std::make_shared<TypeParam>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

REGISTER_TYPED_TEST_CASE_P(UnaryOperator,
                           shape_inference_Shape1,
                           shape_inference_Shape2,
                           input_type_inference_F32,
                           input_type_inference_I64,
                           input_type_inference_U16,
                           incompatible_input_type_Shape1,
                           incompatible_input_type_Shape2,
                           dynamic_rank_input_shape_2D,
                           dynamic_rank_input_shape_3D,
                           dynamic_rank_input_shape_full);

using Types = ::testing::Types<op::Acos, op::Asin, op::Abs>;

INSTANTIATE_TYPED_TEST_CASE_P(type_prop, UnaryOperator, Types);
