// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/low_latency.hpp"

#include <memory>

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/variant.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::LowLatency, "LowLatency", 0);

ngraph::pass::LowLatency::LowLatency()
{
    auto tensor_iterator = ngraph::pattern::wrap_type<opset6::TensorIterator, opset6::Loop>();
    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        const auto& sub_graph_op =
            std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp>(m.get_match_root());
        if (!sub_graph_op)
        {
            return false;
        }

        if (const auto& loop = std::dynamic_pointer_cast<opset6::Loop>(sub_graph_op))
        {
            const auto& trip_count =
                std::dynamic_pointer_cast<opset6::Constant>(loop->get_input_node_shared_ptr(0));
            const auto& num_iter = loop->get_num_iterations();
            if (trip_count && num_iter > 0 && trip_count->get_output_target_inputs(0).size() == 1)
            {
                auto single_iter =
                    std::make_shared<opset6::Constant>(ngraph::element::i64, Shape{}, 1);
                replace_node(trip_count, single_iter);
            }
            else
            {
                // count of iterations is dynamic;
                return false;
            }
        }
        // Mark the TI layer to be unrolled. Enable unconditional ti unrolling for all plugins.
        auto& rt_info = sub_graph_op->get_rt_info();
        rt_info["UNROLL_TI"] = std::make_shared<ngraph::VariantWrapper<int64_t>>(1);

        int64_t variable_id = 0;
        std::vector<std::shared_ptr<ngraph::op::Sink>> assigns;
        const auto& func = sub_graph_op->get_function();
        for (const auto& in : sub_graph_op->get_input_descriptions())
        {
            // Process all back edges
            if (const auto& merged_in =
                    std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp::MergedInputDescription>(
                        in))
            {
                // Insert ReadValue nodes: Parameter -> (new ReadValue) -> consumers
                const auto& inputs_to = func->get_parameters()
                                            .at(merged_in->m_body_parameter_index)
                                            ->get_output_target_inputs(0);
                const std::string variable_name(sub_graph_op->get_friendly_name() + "/" +
                                                func->get_parameters()
                                                    .at(merged_in->m_body_parameter_index)
                                                    ->get_friendly_name() +
                                                "/variable_" + std::to_string(variable_id));
                auto variable = std::make_shared<Variable>(
                    VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name});
                auto read_value = std::make_shared<opset6::ReadValue>(
                    func->get_parameters().at(merged_in->m_body_parameter_index), variable);
                read_value->set_friendly_name(variable_name);
                for (const auto& input_to : inputs_to)
                {
                    input_to.replace_source_output(read_value->output(0));
                }

                // insert Assign nodes: provider -> (new Assign) -> Result
                const auto res = func->get_results().at(merged_in->m_body_value_index);
                auto assign = std::make_shared<opset6::Assign>(res->input_value(0), variable);
                // control dependency so that ReadValue is processed before Assign
                assign->add_control_dependency(read_value);
                assigns.emplace_back(assign);
            }
            variable_id++;
        }
        // save Assign in the func so that it gets into graph traversals and isn't deleted.
        func->add_sinks(assigns);
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "LowLatency");
    register_matcher(m, callback);
}
