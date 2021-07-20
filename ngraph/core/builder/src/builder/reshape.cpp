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

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>

#include "ngraph/axis_vector.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reduce_prod.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/shape_of.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/op/transpose.hpp"
#include "ngraph/op/variadic_split.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace ngraph;
using namespace std;

shared_ptr<Node> builder::opset1::reshape(const Output<Node>& value, const Shape& shape)
{
    if (value.get_partial_shape().same_scheme(shape))
    {
        return value.get_node_shared_ptr();
    }
    else if (is_scalar(shape))
    {
        auto value_rank = value.get_shape().size();
        AxisVector axes_vector(value_rank);
        std::iota(axes_vector.begin(), axes_vector.end(), 0);
        auto axes = op::Constant::create(element::i64, Shape{value_rank}, axes_vector);
        return std::make_shared<op::Squeeze>(value, axes);
    }
    else
    {
        auto out_pattern = op::Constant::create(
            element::i64, Shape{shape.size()}, vector<int64_t>(shape.begin(), shape.end()));

        return make_shared<ngraph::opset1::Reshape>(value, out_pattern, false)
            ->add_provenance_group_members_above({value});
    }
}

shared_ptr<Node> builder::opset1::reorder_axes(const Output<Node>& value, vector<size_t> axes_order)
{
    const auto axes_order_const =
        op::Constant::create(element::i64,
                             Shape{axes_order.size()},
                             vector<int64_t>(axes_order.begin(), axes_order.end()));
    return make_shared<ngraph::opset1::Transpose>(value, axes_order_const)
        ->add_provenance_group_members_above({value});
}

shared_ptr<Node> builder::opset1::transpose(const Output<Node>& value)
{
    // This part is left to preserve backward compatibility and ensure passing ONNX tests.
    if (value.get_partial_shape().is_static())
    {
        vector<size_t> axes_order(value.get_shape().size());
        iota(begin(axes_order), end(axes_order), 0);
        reverse(begin(axes_order), end(axes_order));
        return builder::opset1::reorder_axes(value, axes_order);
    }

    const auto input_rank =
        std::make_shared<ngraph::opset1::ShapeOf>(std::make_shared<ngraph::opset1::ShapeOf>(value));
    const auto neg_one = ngraph::opset1::Constant::create(element::i64, Shape{}, {-1});
    const auto start_node = std::make_shared<ngraph::opset1::Add>(input_rank, neg_one);
    const auto reverse_axes_order =
        std::make_shared<ngraph::opset1::Range>(reshape(start_node, Shape{}), // start
                                                neg_one,                      // stop (exclusive)
                                                neg_one);                     // step
    return std::make_shared<ngraph::opset1::Transpose>(value, reverse_axes_order)
        ->add_provenance_group_members_above({value});
}

namespace ngraph
{
    namespace builder
    {
        namespace opset1
        {
            namespace
            {
                ///
                /// \brief      Return the node representing normalized axis with respect to
                ///             provided rank.
                ///
                /// \param[in]  node_rank  The node representing rank used for normalization.
                /// \param[in]  axis       The axis value to be normalized.
                ///
                /// \return     The new Constant node representing normalized axis value.
                ///
                std::shared_ptr<Node>
                    get_normalized_axis_node(const std::shared_ptr<Node> node_rank, int64_t axis)
                {
                    auto axis_node =
                        ngraph::opset1::Constant::create(element::i64, Shape{1}, {axis});
                    // shortcut for alredy positive value
                    if (axis >= 0)
                    {
                        return axis_node;
                    }

                    // TODO: What if axis value is beyond acceptable values? [-node_rank,
                    // node_rank-1]
                    return make_shared<ngraph::opset1::Add>(node_rank, axis_node);
                }
            } // opset1
        }     // builder
    }         // ngraph
}

shared_ptr<Node> builder::opset1::flatten(const Output<Node>& value, int axis)
{
    // First dimension of output tensor is the product of [d_0, ... d_{axis-1}] dimensions of
    // input tensor. The last dimension is the product of the rest of input tensor dimensions:
    // [d_{axis}, ..., d_n]
    shared_ptr<Node> output_shape;
    if (axis == 0)
    {
        output_shape = ngraph::opset1::Constant::create(element::i64, Shape{2}, {1, -1});
    }
    else if (axis == 1)
    {
        output_shape = ngraph::opset1::Constant::create(element::i64, Shape{2}, {0, -1});
    }
    else
    {
        const auto value_shape = make_shared<ngraph::opset1::ShapeOf>(value);
        const auto value_rank = make_shared<ngraph::opset1::ShapeOf>(value_shape);
        const auto axis_node = get_normalized_axis_node(value_rank, axis);

        const auto first_part_dims = make_shared<ngraph::opset1::StridedSlice>(
            value_shape,
            ngraph::opset1::Constant::create(element::i64, {1}, {0}),
            axis_node,
            vector<int64_t>{},
            vector<int64_t>{});
        const auto first_part_dims_length = make_shared<ngraph::opset1::ReduceProd>(
            first_part_dims, ngraph::opset1::Constant::create(element::i64, {}, {0}), true);

        const auto remaining_part_length =
            ngraph::opset1::Constant::create(element::i64, {1}, {-1});

        output_shape = make_shared<ngraph::opset1::Concat>(
            OutputVector{first_part_dims_length, remaining_part_length}, 0);
    }
    return make_shared<ngraph::opset1::Reshape>(value, output_shape, true)
        ->add_provenance_group_members_above({value});
}

shared_ptr<Node> builder::opset1::expand_dims(const Output<Node>& value, size_t axis)
{
    Shape output_shape(value.get_shape());
    // Add empty axis at specified position.
    auto empty_axis_it = begin(output_shape);
    advance(empty_axis_it, axis);
    output_shape.insert(empty_axis_it, 1);
    return builder::opset1::reshape(value, output_shape);
}

shared_ptr<Node> builder::opset1::squeeze(const Output<Node>& value, vector<size_t> axes)
{
    if (axes.empty())
    {
        return value.get_node_shared_ptr();
    }

    Shape in_shape{value.get_shape()};
    for (size_t idx = 0; idx < axes.size(); ++idx)
    {
        in_shape.at(axes.at(idx)) = 0;
    }
    Shape output_shape;
    for (auto axis : in_shape)
    {
        if (axis != 0)
        {
            output_shape.push_back(axis);
        }
    }
    return builder::opset1::reshape(value, output_shape);
}

shared_ptr<Node> builder::opset1::collapse(const Output<Node>& value,
                                           const size_t start_axis,
                                           const size_t end_axis)
{
    if (start_axis == end_axis)
    {
        return value.get_node_shared_ptr();
    }

    if (value.get_partial_shape().is_static())
    {
        auto shape = value.get_shape();
        // Multiply all alements of shape from start_axis to end_axis inclusive
        size_t collapsed_axis_size = accumulate(next(begin(shape), start_axis),
                                                next(begin(shape), end_axis + 1),
                                                size_t{1},
                                                multiplies<size_t>());
        Shape output_shape{};
        output_shape.insert(begin(output_shape), begin(shape), next(begin(shape), start_axis));
        output_shape.insert(end(output_shape), collapsed_axis_size);
        output_shape.insert(end(output_shape), next(begin(shape), end_axis + 1), end(shape));
        return builder::opset1::reshape(value, output_shape);
    }

    const auto shape = make_shared<ngraph::opset1::ShapeOf>(value);
    const auto rank = make_shared<ngraph::opset1::ShapeOf>(shape);

    // Split lengths used in VariadicSplit
    const auto start_axis_node = ngraph::opset1::Constant::create(element::i64, {1}, {start_axis});
    const auto end_axis_node = ngraph::opset1::Constant::create(element::i64, {1}, {end_axis + 1});
    const auto collapsed_axis =
        make_shared<ngraph::opset1::Subtract>(end_axis_node, start_axis_node);
    const auto post_axis = make_shared<ngraph::opset1::Subtract>(rank, end_axis_node);

    const auto split_lengths = make_shared<ngraph::opset1::Concat>(
        OutputVector{start_axis_node, collapsed_axis, post_axis}, 0);
    const auto split_axis = ngraph::opset1::Constant::create(element::i64, {}, {0});
    const auto split_node =
        make_shared<ngraph::opset1::VariadicSplit>(shape, split_axis, split_lengths);

    const auto reduced_axis = ngraph::opset1::Constant::create(element::i64, {1}, {0});
    const auto collapsed_axis_size =
        make_shared<ngraph::opset1::ReduceProd>(split_node->output(1), reduced_axis, true);

    const auto collapsed_shape = make_shared<ngraph::opset1::Concat>(
        OutputVector{split_node->output(0), collapsed_axis_size, split_node->output(2)}, 0);

    return make_shared<ngraph::opset1::Reshape>(value, collapsed_shape, false);
}
