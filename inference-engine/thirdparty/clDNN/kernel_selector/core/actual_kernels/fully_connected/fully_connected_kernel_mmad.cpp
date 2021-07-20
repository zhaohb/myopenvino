﻿// Copyright (c) 2016-2020 Intel Corporation
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

#include "fully_connected_kernel_mmad.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey FullyConnectedKernelMMAD::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableDifferentInputWeightsTypes();
    k.EnableDifferentTypes();

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::bf);

    k.EnableBiasPerOutput();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    return k;
}

bool FullyConnectedKernelMMAD::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    auto fc_params = static_cast<const fully_connected_params&>(params);
    auto input = fc_params.inputs[0];
    if (input.GetLayout() == DataLayout::bfyx &&
        (input.X().LogicalDimPadded() != 1 || input.Y().LogicalDimPadded() != 1 || input.Z().LogicalDimPadded() != 1)) {
        return false;
    }

    return true;
}

FullyConnectedKernelMMAD::FullyConnectedTuningData FullyConnectedKernelMMAD::GetTuningParams(const fully_connected_params& params) const {
    FullyConnectedTuningData tuning_data;

    const auto& input = params.inputs[0];
    const auto& output = params.output;
    size_t input_feature = input.Feature().v;
    size_t input_batch = input.Batch().v;
    size_t output_feature = output.Feature().v;
    size_t output_batch = output.Batch().v;
    // for 3d case
    if (output.GetLayout() == DataLayout::bfyx) {
        input_batch *= input.Feature().v;
        input_feature = input.Y().v;
        output_batch *= output.Feature().v;
        output_feature = output.Y().v;
    }

    tuning_data.sub_group_size = 8;
    if (input.X().v == 1 && input.Z().v == 1 && input.Batch().v == 1 &&
        ((input.Y().v == 1 && output.GetLayout() != DataLayout::bfyx) || (input.Feature().v == 1 && output.GetLayout() == DataLayout::bfyx)) ) {
        // Known cases for TGL where simd16 works better than simd8
        bool simd16_exception_1 = input.Feature().v == 25088 && output.Feature().v == 512;
        bool simd16_exception_2 = input.Feature().v == 21504 && output.Feature().v == 512;

        if (simd16_exception_1 || simd16_exception_2)
            tuning_data.sub_group_size = 16;
    }

    size_t sub_group_pack_size = tuning_data.sub_group_size * tuning_data.pack_size;

    tuning_data.feature_blocks_count = input.GetLayout() == DataLayout::bfyx && input_feature % sub_group_pack_size != 0 ?
                                       input_feature / sub_group_pack_size :
                                       input.GetLayout() != DataLayout::bfyx && tuning_data.sub_group_size == 16 ?
                                       CeilDiv(input_feature, 32) % 2 == 0 ? CeilDiv(input_feature, 64) : CeilDiv(input_feature, 64) - 1 :
                                       CeilDiv(input_feature, sub_group_pack_size);

    bool slm_div_factor_exception = input_batch == 300 && input_feature == 2048 &&
                                    output_batch == 300 && (output_feature == 324 || output_feature == 81);

    if (tuning_data.feature_blocks_count && tuning_data.sub_group_size == 8 && !slm_div_factor_exception)
        while (tuning_data.feature_blocks_count % (tuning_data.slm_div_factor * 2) == 0 &&
               (tuning_data.slm_div_factor * 2 <= params.engineInfo.maxWorkGroupSize / tuning_data.sub_group_size))
            tuning_data.slm_div_factor *= 2;

    tuning_data.work_group_size = tuning_data.slm_div_factor * tuning_data.sub_group_size;

    tuning_data.full_unroll_factor = tuning_data.feature_blocks_count / tuning_data.slm_div_factor;

    if (tuning_data.sub_group_size == 16) {
        tuning_data.unroll_factor = 1;
    } else {
        size_t temp_unroll_factor = 3;

        if (tuning_data.full_unroll_factor > 3) {
            while (tuning_data.full_unroll_factor % temp_unroll_factor)
                temp_unroll_factor--;
            tuning_data.unroll_factor = temp_unroll_factor;
        } else {
            tuning_data.unroll_factor = tuning_data.full_unroll_factor;
        }
    }

    return tuning_data;
}

FullyConnectedKernelMMAD::DispatchData FullyConnectedKernelMMAD::SetDefault(const fully_connected_params& params,
                                                                            int) const {
    FullyConnectedTuningData tuning_data = GetTuningParams(params);
    auto dispatchData = Parent::SetDefault(params);
    const auto& output = params.output;

    std::vector<size_t> global = { Align(output.Feature().v, tuning_data.sub_group_size) * tuning_data.slm_div_factor, output.Batch().v, 1 };
    if (output.GetLayout() == DataLayout::bfyx)
        global = { Align(output.Y().v, tuning_data.sub_group_size) * tuning_data.slm_div_factor, output.Batch().v, output.Feature().v };

    dispatchData.gws = global;
    dispatchData.lws = { tuning_data.work_group_size, 1, 1 };

    return dispatchData;
}

JitConstants FullyConnectedKernelMMAD::GetJitConstants(const fully_connected_params& params,
                                                       const DispatchData& runInfo) const {
    FullyConnectedTuningData tuning_data = GetTuningParams(params);

    auto jit = Parent::GetJitConstants(params, runInfo);

    auto& input = params.inputs[0];
    auto& output = params.output;
    auto& weights = params.weights;

    size_t sub_group_pack_size = tuning_data.sub_group_size * tuning_data.pack_size;

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", tuning_data.sub_group_size));
    if (tuning_data.sub_group_size == 8) {
        if (input.GetDims().size() == 5) {
            jit.AddConstant(MakeJitConstant("FILTER_GET_OFFSET(f)", "GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX(FILTER, f, 0, 0, 0)"));
        } else {
            jit.AddConstant(MakeJitConstant("FILTER_GET_OFFSET(f)", "GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX(FILTER, f, 0, 0, 0, 0)"));
        }
    } else {
        if (input.GetDims().size() == 5) {
            jit.AddConstant(MakeJitConstant("FILTER_GET_OFFSET(f)", "GET_FILTER_OS_IS_YX_ISA8_OSV16_ISV4_INDEX(FILTER, f, 0, 0, 0)"));
        } else {
            jit.AddConstant(MakeJitConstant("FILTER_GET_OFFSET(f)", "GET_FILTER_OS_IS_ZYX_ISA8_OSV16_ISV4_INDEX(FILTER, f, 0, 0, 0, 0)"));
        }
    }

    jit.Merge(MakeTypeJitConstants(input.GetDType() == Datatype::UINT8 ? Datatype::UINT32 : Datatype::INT32, "INPUT_PACKED"));
    jit.Merge(MakeTypeJitConstants(weights.GetDType() == WeightsType::UINT8 ? Datatype::UINT32 : Datatype::INT32, "FILTER_PACKED"));

    auto filter_spatial_size = weights.X().v * weights.Y().v * weights.Z().v;
    auto filter_spatial_pitch = 8 * sub_group_pack_size;
    auto filter_fblock_pitch = tuning_data.sub_group_size == 8 ?
                               filter_spatial_size * filter_spatial_pitch :
                               filter_spatial_size * filter_spatial_pitch * 2;

    jit.AddConstant(MakeJitConstant("FILTER_SPATIAL_SIZE", filter_spatial_size));
    jit.AddConstant(MakeJitConstant("MMAD_FILTER_SPATIAL_PITCH", filter_spatial_pitch));
    jit.AddConstant(MakeJitConstant("MMAD_FILTER_FBLOCK_PITCH", filter_fblock_pitch));

    size_t input_x_pitch = input.X().pitch;
    size_t input_y_pitch = input.Y().pitch;
    size_t input_z_pitch = input.Z().pitch;

    if (input.GetLayout() == DataLayout::bfyx) {
        jit.AddConstant(MakeJitConstant("MMAD_INPUT_FBLOCK_PITCH", sub_group_pack_size));
    } else if (input.GetLayout() == DataLayout::b_fs_yx_fsv32 || input.GetLayout() == DataLayout::b_fs_zyx_fsv32) {
        input_x_pitch = 32;
        input_y_pitch *= 32;
        input_z_pitch *= 32;
        jit.AddConstant(MakeJitConstant("MMAD_INPUT_FBLOCK_PITCH", input.Feature().pitch * sub_group_pack_size));
    }

    bool has_feature_leftovers = (input.GetLayout() == DataLayout::bfyx && input.Feature().v % sub_group_pack_size) ||
                                 (input.GetLayout() != DataLayout::bfyx && tuning_data.sub_group_size == 16 && CeilDiv(input.Feature().v, 32) % 2);

    if (output.GetLayout() == DataLayout::bfyx)
        has_feature_leftovers = input.Y().v % sub_group_pack_size;

    jit.AddConstant(MakeJitConstant("HAS_FEATURE_LEFTOVERS", has_feature_leftovers));
    jit.AddConstant(MakeJitConstant("FEATURE_BLOCKS_COUNT", tuning_data.feature_blocks_count));
    jit.AddConstant(MakeJitConstant("SLM_DIV_FACTOR", tuning_data.slm_div_factor));
    jit.AddConstant(MakeJitConstant("UNROLL_FACTOR", tuning_data.unroll_factor));
    jit.AddConstant(MakeJitConstant("FULL_UNROLL_FACTOR", tuning_data.full_unroll_factor));
    jit.AddConstant(MakeJitConstant("WORK_GROUP_SIZE", tuning_data.work_group_size));

    jit.AddConstant(MakeJitConstant("MMAD_INPUT_SPATIAL_PITCH", input_x_pitch));
    jit.AddConstant(MakeJitConstant("MMAD_INPUT_X_PITCH", input_x_pitch));
    jit.AddConstant(MakeJitConstant("MMAD_INPUT_Y_PITCH", input_y_pitch));
    jit.AddConstant(MakeJitConstant("MMAD_INPUT_Z_PITCH", input_z_pitch));

    bool split_spatial = input.X().pad.Total() != 0 || input.Y().pad.Total() != 0 || input.Z().pad.Total() != 0;
    bool spatial_major = DataTensor::Channelndex(input.GetLayout(), Tensor::DataChannelName::X) <
                         DataTensor::Channelndex(input.GetLayout(), Tensor::DataChannelName::FEATURE);

    jit.AddConstant(MakeJitConstant("SPLIT_SPATIAL", split_spatial));
    jit.AddConstant(MakeJitConstant("SPATIAL_MAJOR", spatial_major));

    if (output.GetLayout() == DataLayout::bfyx) {
        jit.AddConstant(MakeJitConstant("FEATURE_PITCH", input.Y().pitch));
        jit.AddConstant(MakeJitConstant("OUT_FEATURE_NUM", output.Y().v));
        jit.AddConstant(MakeJitConstant("IN_FEATURE_NUM", input.Y().v));
        jit.AddConstant(MakeJitConstant("IS_3D", true));
    } else {
        jit.AddConstant(MakeJitConstant("FEATURE_PITCH", input.Feature().pitch));
        jit.AddConstant(MakeJitConstant("OUT_FEATURE_NUM", output.Feature().v));
        jit.AddConstant(MakeJitConstant("IN_FEATURE_NUM", input.Feature().v));
    }

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        std::vector<std::string> idx_order = {"batch", "feature", "0", "0"};
        if (output.GetLayout() == DataLayout::bfyx)
            idx_order = {"batch", "skip_f", "feature", "0"};

        FusedOpsConfiguration conf = { "", idx_order, "dequantized", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

KernelsData FullyConnectedKernelMMAD::GetKernelsData(const Params& params, const optional_params& options) const {
    auto fc_params = static_cast<const fully_connected_params&>(params);
    auto& input = fc_params.inputs[0];

    auto w_layout = GetTuningParams(fc_params).sub_group_size == 16 ?
                    input.GetDims().size() == 4 ? WeightsLayout::os_is_yx_isa8_osv16_isv4 : WeightsLayout::os_is_zyx_isa8_osv16_isv4 :
                    input.GetDims().size() == 4 ? WeightsLayout::os_is_yx_isa8_osv8_isv4 : WeightsLayout::os_is_zyx_isa8_osv8_isv4;

    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    options,
                                                    input.GetLayout(),
                                                    w_layout,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }
    return res;
}

KernelsPriority FullyConnectedKernelMMAD::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_7;
}
}  // namespace kernel_selector
