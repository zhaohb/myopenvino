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
#include "op/loop.hpp"

#include <iterator>
#include <memory>

#include "core/graph.hpp"
#include "core/null_node.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/function.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                namespace
                {
                    /// \brief      Check if termination condition is true during all Loop
                    ///             iterations.
                    ///             It allows to replace termination condition body output with
                    ///             Constant.
                    ///             As a result ngraph Loop shape inference is able to handle more
                    ///             cases.
                    ///
                    /// \param[in]  body_out_cond   Termination loop condition input of the body of
                    ///                             the Loop (value updated during Loop iterations).
                    ///
                    /// \return true if termination condition is true and it cannot be changed
                    ///         during Loop iterations, false otherwise.
                    bool is_termination_condition_always_true(
                        const Output<ngraph::Node>& body_out_cond)
                    {
                        // If body termination condition input matches Indentity op pattern the has
                        // value of loop_cond - true
                        // Identity op for boolean value is represented by LogicalOr op whose second
                        // input is always false
                        if (is_type<default_opset::LogicalOr>(body_out_cond.get_node_shared_ptr()))
                        {
                            const auto second_input = body_out_cond.get_node_shared_ptr()
                                                          ->input_value(1)
                                                          .get_node_shared_ptr();
                            if (ngraph::op::is_constant(second_input) &&
                                second_input->get_element_type() == element::boolean &&
                                as_type_ptr<default_opset::Constant>(second_input)
                                        ->cast_vector<bool>()
                                        .at(0) == false)
                            {
                                return true;
                            }
                        }
                        return false;
                    }
                }

                OutputVector loop(const Node& node)
                {
                    const auto& ng_inputs = node.get_ng_inputs();

                    const OutputVector loop_carried_dependencies{std::next(ng_inputs.begin(), 2),
                                                                 ng_inputs.end()};

                    const Subgraph& body_graph{node.get_attribute_value<Subgraph>("body")};
                    auto body_outputs = body_graph.get_ng_outputs();
                    const auto& body_inputs = body_graph.get_ng_parameters();

                    // optional inputs
                    Output<ngraph::Node> trip_count;
                    // trip count skipped or has value max(int64_t) means infinitive loop
                    if (ngraph::op::is_null(ng_inputs.at(0)) ||
                        (ngraph::op::is_constant(ng_inputs.at(0).get_node_shared_ptr()) &&
                         as_type_ptr<default_opset::Constant>(ng_inputs.at(0).get_node_shared_ptr())
                                 ->cast_vector<int64_t>()[0] ==
                             std::numeric_limits<int64_t>::max()))
                    {
                        // -1 means infinite Loop
                        trip_count = ngraph::op::Constant::create(ngraph::element::i64, {1}, {-1});
                    }
                    else
                    {
                        trip_count = ng_inputs.at(0);
                    }

                    Output<ngraph::Node>
                        termination_cond; // true means that first interation should be run
                    if (ngraph::op::is_null(
                            ng_inputs.at(1).get_node_shared_ptr())) // termination condition skipped
                    {
                        termination_cond =
                            ngraph::op::Constant::create(ngraph::element::boolean, {1}, {true});
                    }
                    else if (ngraph::op::is_constant(ng_inputs.at(1).get_node_shared_ptr()) &&
                             as_type_ptr<default_opset::Constant>(
                                 ng_inputs.at(1).get_node_shared_ptr())
                                     ->cast_vector<bool>()[0] == false)
                    {
                        // no iteration is performed so initial values are returned
                        OutputVector node_outputs;
                        // final values
                        for (const auto& dep : loop_carried_dependencies)
                        {
                            node_outputs.push_back(dep);
                        }
                        // scan outputs
                        for (const auto& dep : loop_carried_dependencies)
                        {
                            node_outputs.push_back(dep);
                        }
                        return node_outputs;
                    }
                    else
                    {
                        termination_cond = ng_inputs.at(1);
                    }

                    const int64_t concat_axis = 0;
                    const auto concat_axis_const =
                        ngraph::op::Constant::create(ngraph::element::i64, {1}, {concat_axis});
                    // add dimension along which scan outputs will be concatenated
                    for (size_t i = loop_carried_dependencies.size() + 1; i < body_outputs.size();
                         ++i)
                    {
                        body_outputs[i] = std::make_shared<default_opset::Unsqueeze>(
                            body_outputs[i], concat_axis_const);
                    }

                    const auto& body_loop_out_cond = body_outputs.at(0).get_node_shared_ptr();
                    // optimization allow to improve nG Loop shape inference
                    if (is_termination_condition_always_true(body_loop_out_cond))
                    {
                        body_outputs[0] =
                            ngraph::op::Constant::create(ngraph::element::boolean, {1}, {true});
                    }

                    CHECK_VALID_NODE(node,
                                     body_inputs.size() >= loop_carried_dependencies.size() + 2,
                                     "The provided loop body graph inputs size (",
                                     body_inputs.size(),
                                     "), is not greater than the sum of loop carried dependencies "
                                     "and two mandatory"
                                     " inputs (",
                                     loop_carried_dependencies.size() + 2,
                                     ")");

                    CHECK_VALID_NODE(node,
                                     body_outputs.size() >= loop_carried_dependencies.size() + 1,
                                     "The provided loop body graph outputs size (",
                                     body_outputs.size(),
                                     ") is not greater than number of outputs. Required at least: ",
                                     loop_carried_dependencies.size() + 1);

                    ParameterVector body_params(body_inputs.begin() + 2, body_inputs.end());
                    body_params.emplace(body_params.begin(),
                                        body_inputs[0]); // current iteration body input
                    const auto body = std::make_shared<ngraph::Function>(body_outputs, body_params);
                    auto loop = std::make_shared<default_opset::Loop>(trip_count, termination_cond);
                    default_opset::Loop::SpecialBodyPorts spec_ports{0, 0};
                    loop->set_special_body_ports(spec_ports);
                    loop->set_function(body);

                    // Setting up other Loop body inputs.
                    // body_inputs[0] is iteration number, body_inputs[1] is termination condition
                    auto body_inputs_it = std::next(body_inputs.begin(), 2);
                    // body_outputs[0] is termination condition output
                    auto body_outputs_it = std::next(body_outputs.begin(), 1);

                    // Set-up loop carried dependencies and final output values
                    OutputVector final_values;
                    for (const auto& dep : loop_carried_dependencies)
                    {
                        loop->set_merged_input(*body_inputs_it++, dep, *body_outputs_it);
                        final_values.push_back(loop->get_iter_value(*body_outputs_it++, -1));
                    }

                    const auto& outputs_from_parent = body_graph.get_outputs_from_parent();
                    CHECK_VALID_NODE(node,
                                     std::distance(body_inputs_it, body_inputs.end()) ==
                                         outputs_from_parent.size(),
                                     "Expected number of invariant parameters is"
                                     " not equal number of provided outputs from parent scope");

                    // Set-up parameters from parent graph which are not changed during Loop's
                    // iterations
                    for (auto out_from_parent_it = outputs_from_parent.begin();
                         body_inputs_it != body_inputs.end() &&
                         out_from_parent_it != outputs_from_parent.end();
                         ++body_inputs_it, ++out_from_parent_it)
                    {
                        loop->set_invariant_input(*body_inputs_it, *out_from_parent_it);
                    }

                    // Set-up scan outputs
                    OutputVector scan_outputs;
                    for (; body_outputs_it != body_outputs.end(); body_outputs_it++)
                    {
                        // start=0, stride=1, part_size=1, end=-1, axis=0
                        scan_outputs.push_back(loop->get_concatenated_slices(
                            *body_outputs_it, 0, 1, 1, -1, concat_axis));
                    }
                    loop->validate_and_infer_types();

                    OutputVector node_outputs;
                    for (const auto& v : final_values)
                    {
                        node_outputs.push_back(v);
                    }
                    for (const auto& v : scan_outputs)
                    {
                        node_outputs.push_back(v);
                    }
                    return node_outputs;
                }
            } // namespace set_1
        }     // namespace op
    }         // namespace onnx_import
} // namespace ngraph
