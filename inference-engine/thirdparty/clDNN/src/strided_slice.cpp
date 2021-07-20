/*
// Copyright (c) 2019-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "strided_slice_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include "data_inst.h"
#include <string>
#include <vector>

namespace cldnn {
primitive_type_id strided_slice::type_id() {
    static primitive_type_base<strided_slice> instance;
    return &instance;
}

layout strided_slice_inst::calc_output_layout(strided_slice_node const& node) {
    auto desc = node.get_primitive();
    auto input_layout = node.input(0).get_output_layout();
    auto output_format = input_layout.format;
    if ((output_format == format::bfzyx) && (node.get_primitive()->shrink_axis_mask.size() > 0)) {
        output_format = format::bfyx;
    }
    return layout{input_layout.data_type, output_format, desc->out_size};
}

std::string strided_slice_inst::to_string(strided_slice_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite strided_slice_info;
    strided_slice_info.add("input id", input.id());
    strided_slice_info.add("begin_param id", node.get_dependency(1).id());
    strided_slice_info.add("end_param id", node.get_dependency(2).id());
    strided_slice_info.add("stride_param id", node.get_dependency(3).id());
    strided_slice_info.add("begin mask", node.get_primitive()->begin_mask);
    strided_slice_info.add("end mask", node.get_primitive()->end_mask);
    strided_slice_info.add("new axis mask", node.get_primitive()->new_axis_mask);
    strided_slice_info.add("shrink axis mask", node.get_primitive()->shrink_axis_mask);
    strided_slice_info.add("begin_param shape", node.get_dependency(1).get_output_layout().size.to_string());
    strided_slice_info.add("end_param shape", node.get_dependency(2).get_output_layout().size.to_string());
    strided_slice_info.add("stride_param shape", node.get_dependency(3).get_output_layout().size.to_string());

    node_info->add("strided_slice info", strided_slice_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

strided_slice_inst::typed_primitive_inst(network_impl& network, strided_slice_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
