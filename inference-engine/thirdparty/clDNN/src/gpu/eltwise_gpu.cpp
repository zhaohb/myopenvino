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

#include "eltwise_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"
#include <vector>

namespace cldnn {
namespace gpu {

struct eltwise_gpu : typed_primitive_gpu_impl<eltwise> {
    using parent = typed_primitive_gpu_impl<eltwise>;
    using parent::parent;

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<eltwise>& instance,
                                                int32_t split) const override {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);
        return args;
    }

public:
    static primitive_impl* create(const eltwise_node& arg) {
        auto ew_params = get_default_params<kernel_selector::eltwise_params>(arg);
        auto ew_optional_params =
            get_default_optional_params<kernel_selector::eltwise_optional_params>(arg.get_program());

        for (size_t i = 1; i < arg.inputs_count(); i++) {
            ew_params.inputs.push_back(convert_data_tensor(arg.input(i).get_output_layout()));
        }

        const auto& primitive = arg.get_primitive();

        ew_params.operations.push_back({{kernel_selector::eltwise_params::InputType::Buffer(0),
                                         kernel_selector::eltwise_params::InputType::Buffer(1)},
                                        convert_to_eltwise_mode(primitive->mode)});

        for (uint32_t i = 2; i < static_cast<uint32_t>(arg.inputs_count()); i++) {
            ew_params.operations.push_back({{kernel_selector::eltwise_params::InputType::Intermediate(i - 2),
                                             kernel_selector::eltwise_params::InputType::Buffer(i)},
                                            convert_to_eltwise_mode(primitive->mode)});
        }

        if (primitive->mode == eltwise_mode::sum) {
            ew_params.coefficients = primitive->coefficients;
        }

        for (size_t i = 0; i < ew_params.inputs.size(); i++) {
            if (!ew_params.inputs[i].SameDims(ew_params.output)) {
                std::vector<int32_t> input_size = arg.input(i).get_output_layout().size.raw.vector();
                std::vector<int32_t> output_size = arg.get_output_layout().size.raw.vector();
                bool broadcast = false;
                for (size_t d = 0; d < output_size.size(); d++) {
                    if (output_size[d] != 1 && input_size[d] == 1)
                        broadcast = true;
                }
                if (broadcast) {
                    ew_params.broadcast = true;
                    break;
                } else {
                    ew_params.layoutBased = true;
                    break;
                }
            }
        }

        // stride
        if (!primitive->stride.empty()) {
            const auto& stride = primitive->stride;
            ew_params.stride.resize(stride.size());
            for (size_t i = 0; i < primitive->stride.size(); i++) {
                ew_params.stride[i] = {(uint32_t)stride[i].spatial[0],
                                       (uint32_t)stride[i].spatial[1],
                                       (uint32_t)stride[i].spatial[2]};
            }
        }

        // check if strides are the same
        if (!ew_params.stride.empty()) {
            const auto& stride = ew_params.stride[0];
            for (size_t i = 1; i < ew_params.stride.size(); i++) {
                if (stride.x != ew_params.stride[i].x || stride.y != ew_params.stride[i].y)
                    ew_params.layoutBased = true;
            }
        } else if (!ew_params.inputs[0].SameDimsSizes(ew_params.inputs[1])) {
            ew_params.broadcast = true;
        }

        // TODO [LOW PRECISION]: check if this parameter's really needed. Maybe data types are enough
        bool quantization = true;
        for (size_t i = 0; i < arg.inputs_count(); i++) {
            if (arg.input(i).get_output_layout().data_type != data_types::u8 &&
                arg.input(i).get_output_layout().data_type != data_types::i8) {
                quantization = false;
            }
        }
        ew_params.int8_quantization = quantization;

        auto& kernel_selector = kernel_selector::eltwise_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto eltwise = new eltwise_gpu(arg, best_kernels[0]);

        return eltwise;
    }
};

namespace detail {

attach_eltwise_gpu::attach_eltwise_gpu() {
    implementation_map<eltwise>::add(
        {{ std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i8, format::yxfb), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i32, format::yxfb), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i64, format::yxfb), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i64, format::bfyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i8, format::byxf), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i32, format::byxf), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i64, format::byxf), eltwise_gpu::create },
         // block f16
         { std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv16), eltwise_gpu::create },
         // 3D
         { std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::u8, format::bfzyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i32, format::bfzyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i64, format::bfzyx), eltwise_gpu::create },
         // 4D
         { std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i8, format::bfwzyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::u8, format::bfwzyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i32, format::bfwzyx), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i64, format::bfwzyx), eltwise_gpu::create },

         { std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_zyx_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_zyx_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_zyx_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_zyx_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i32, format::b_fs_zyx_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i64, format::b_fs_zyx_fsv16), eltwise_gpu::create },

         { std::make_tuple(engine_types::ocl, data_types::f32, format::bs_fs_zyx_bsv16_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f16, format::bs_fs_zyx_bsv16_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i8, format::bs_fs_zyx_bsv16_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i32, format::bs_fs_zyx_bsv16_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i64, format::bs_fs_zyx_bsv16_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f32, format::bs_fs_yx_bsv16_fsv16), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f16, format::bs_fs_yx_bsv16_fsv16), eltwise_gpu::create },
         // MMAD
         { std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv4), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv4), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv4), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv32), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv32), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv32), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv32), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_zyx_fsv32), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_zyx_fsv32), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_zyx_fsv32), eltwise_gpu::create },
         { std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_zyx_fsv32), eltwise_gpu::create },

         //
         { std::make_tuple(engine_types::ocl, data_types::f16, format::fs_b_yx_fsv32), eltwise_gpu::create }});
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
