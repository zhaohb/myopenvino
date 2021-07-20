/*
// Copyright (c) 2019 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/quantize.hpp"
#include "primitive_inst.h"
#include "data_inst.h"
#include "kernel_selector/core/actual_kernels/quantize/quantize_kernel_params.h"
#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<quantize> : public typed_program_node_base<quantize> {
    using parent = typed_program_node_base<quantize>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    size_t inputs_count() const { return get_dependencies().size(); }
    int get_levels() const { return get_primitive()->levels; }
    bool get_packed_binary_output() const { return get_output_layout().data_type == data_types::bin; }
    bool get_scale_shift_opt() const { return scale_shift_opt; }
    bool get_need_pre_shift() const { return need_pre_shift; }
    bool get_need_post_scale() const { return need_post_scale; }
    bool get_need_post_shift() const { return need_post_shift; }
    bool get_need_clamp() const { return need_clamp; }
    bool get_per_tensor_input_scale() const { return per_tensor_input_scale; }
    bool get_per_tensor_input_shift() const { return per_tensor_input_shift; }
    bool get_per_tensor_input_range() const { return per_tensor_input_range; }
    bool get_per_tensor_output_scale() const { return per_tensor_output_scale; }
    bool get_per_tensor_output_shift() const { return per_tensor_output_shift; }
    float get_input_scale_val() const { return in_scale; }
    float get_input_shift_val() const { return in_shift; }
    float get_input_lo_val() const { return in_lo; }
    float get_input_hi_val() const { return in_hi; }
    float get_output_scale_val() const { return out_scale; }
    float get_output_shift_val() const { return out_shift; }

    void set_scale_shift_opt() { scale_shift_opt = true; }
    void set_need_post_scale() { need_post_scale = true; }
    void set_need_post_shift() { need_post_shift = true; }
    void set_need_pre_shift() { need_pre_shift = true; }
    void set_need_clamp() { need_clamp = true; }
    void set_per_tensor_input_scale() { per_tensor_input_scale = true; }
    void set_per_tensor_input_shift() { per_tensor_input_shift = true; }
    void set_per_tensor_input_range() { per_tensor_input_range = true; }
    void set_per_tensor_output_scale() { per_tensor_output_scale = true; }
    void set_per_tensor_output_shift() { per_tensor_output_shift = true; }
    // Clamp is needed to avoid inf and -inf which are converted to undefined "inf" constant in opencl
    void set_input_scale_val(float val) { in_scale = clamp(val); }
    void set_input_shift_val(float val) { in_shift = clamp(val); }
    void set_input_lo_val(float val) { in_lo = clamp(val); }
    void set_input_hi_val(float val) { in_hi = clamp(val); }
    void set_output_scale_val(float val) { out_scale = clamp(val); }
    void set_output_shift_val(float val) { out_shift = clamp(val); }

    std::shared_ptr<kernel_selector::fuse_params> get_fuse_params() const override {
        return std::make_shared<kernel_selector::quantize_fuse_params>(scale_shift_opt,
                                                                       need_post_scale,
                                                                       need_post_shift,
                                                                       need_pre_shift,
                                                                       need_clamp,
                                                                       per_tensor_input_range,
                                                                       per_tensor_input_scale,
                                                                       per_tensor_input_shift,
                                                                       per_tensor_output_scale,
                                                                       per_tensor_output_shift,
                                                                       in_lo,
                                                                       in_hi,
                                                                       in_scale,
                                                                       in_shift,
                                                                       out_scale,
                                                                       out_shift);
    }

private:
    inline float clamp(float val) const {
        return std::max(std::numeric_limits<float>::lowest(), std::min(std::numeric_limits<float>::max(), val));
    }

    bool scale_shift_opt = false;
    bool need_post_scale = false;
    bool need_post_shift = false;
    bool need_pre_shift = false;
    bool need_clamp = false;

    bool per_tensor_input_range = false;
    bool per_tensor_input_scale = false;
    bool per_tensor_input_shift = false;
    bool per_tensor_output_scale = false;
    bool per_tensor_output_shift = false;

    float in_lo = 0.0f;
    float in_hi = 0.0f;
    float in_scale = 0.0f;
    float in_shift = 0.0f;
    float out_scale = 0.0f;
    float out_shift = 0.0f;
};

using quantize_node = typed_program_node<quantize>;

template <>
class typed_primitive_inst<quantize> : public typed_primitive_inst_base<quantize> {
    using parent = typed_primitive_inst_base<quantize>;

public:
    static layout calc_output_layout(quantize_node const& node);
    static std::string to_string(quantize_node const& node);

public:
    typed_primitive_inst(network_impl& network, quantize_node const& desc);
};

using quantize_inst = typed_primitive_inst<quantize>;

}  // namespace cldnn
