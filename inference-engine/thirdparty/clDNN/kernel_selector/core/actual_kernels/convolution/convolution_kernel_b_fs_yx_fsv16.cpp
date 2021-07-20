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

#include "convolution_kernel_b_fs_yx_fsv16.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <algorithm>

namespace kernel_selector {

ConvolutionKernel_b_fs_yx_fsv16::ConvolutionKernel_b_fs_yx_fsv16() : ConvolutionKernelBase("convolution_gpu_bfyx_f16") {
    std::vector<size_t> outputBlockWidths = { 2, 4, 8 };
    std::vector<std::string> executionModes = ConvolutionKernelBase::autoTuneOptions;

    for (auto w : outputBlockWidths) {
        for (auto exeMode : executionModes) {
            autoTuneOptions.emplace_back(AutoTuneOption{ w, exeMode });
        }
    }
}

ConvolutionKernel_b_fs_yx_fsv16::AutoTuneOption ConvolutionKernel_b_fs_yx_fsv16::GetAutoTuneOptions(const Params& params,
                                                                                                    int /*autoTuneIndex*/) const {
    const convolution_params& cp = static_cast<const convolution_params&>(params);
    auto x = cp.output.X().v;
    auto f = cp.output.Feature().v;
    if (x * f <= 256) {
        if (x <= 8 || x * f <= 128)
            return { 2, DEFAULT };
        else
            return { 4, DEFAULT };
    } else if (x * f <= 1536) {
        return { 4, DEFAULT };
    } else {
        if (x >= 8  && x < 12 && x * f < 2600)
            return { 4, DEFAULT };
        else if (x < 12 && x * f < 8192)
            return { 8, DEFAULT };
        else
            return { 8, AGE_BASED };
    }
}

float ConvolutionKernel_b_fs_yx_fsv16::EstimateOccupancy(const convolution_params& params,
                                                         const ConvolutionTuningData& tuning_data) const {
    auto tuneOptions = GetAutoTuneOptions(params, 0);
    auto blockWidth = tuneOptions.blockWidth;

    auto x = params.output.X().v;
    auto y = params.output.Y().v;
    auto f = params.output.Feature().v;
    auto b = params.output.Batch().v;

    auto threads = CeilDiv(x, blockWidth) * y * CeilDiv(f, tuning_data.feature_block_size) * tuning_data.slm_div_factor * b;

    return static_cast<float>(threads) / static_cast<float>(params.engineInfo.maxThreadsPerDevice);
}

ConvolutionKernel_b_fs_yx_fsv16::ConvolutionTuningData ConvolutionKernel_b_fs_yx_fsv16::GetTuningParams(const convolution_params& params) const {
    ConvolutionTuningData tuning_data;

    const auto& input = params.inputs[0];

    size_t ic_blocks = CeilDiv(input.Feature().v / params.groups, tuning_data.feature_block_size);

    size_t max_slm_div_factor = params.engineInfo.maxWorkGroupSize / tuning_data.sub_group_size;

    bool slm_exception = params.output.X().v == 3 && params.output.Y().v == 3 && params.output.ElementSize() == 4 && params.output.Feature().v <= 512;

    if (params.engineInfo.deviceType == dev_type::integrated_gpu && params.engineInfo.bIMADSupport && !slm_exception)
        while (ic_blocks % (tuning_data.slm_div_factor * 2) == 0 && (tuning_data.slm_div_factor * 2 <= max_slm_div_factor) &&
               EstimateOccupancy(params, tuning_data) < 4.0)
            tuning_data.slm_div_factor *= 2;

    tuning_data.work_group_size = tuning_data.slm_div_factor * tuning_data.sub_group_size;

    return tuning_data;
}

ParamsKey ConvolutionKernel_b_fs_yx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);

    k.EnableDifferentTypes();

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    // TODO Add bias per output support to kernel
    // k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableSplitSupport();
    k.EnableBatching();
    k.EnableDepthwiseSeparableOpt();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    k.EnableGroupedConvolution();
    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_b_fs_yx_fsv16::SetDefault(const convolution_params& params,
                                                                                int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(params);

    ConvolutionTuningData tuning_data = GetTuningParams(params);

    const auto& out = params.output;

    auto autoTune = GetAutoTuneOptions(params, autoTuneIndex);
    dispatchData.cldnnStyle.blockWidth = autoTune.blockWidth;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    dispatchData.gws[0] = CeilDiv(x, autoTune.blockWidth) * y;
    dispatchData.gws[1] = Align(f, tuning_data.feature_block_size) * tuning_data.slm_div_factor;
    dispatchData.gws[2] = b;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = tuning_data.work_group_size;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_b_fs_yx_fsv16::GetKernelsPriority(const Params& params, const optional_params& /*options*/) const {
    const auto& p = static_cast<const convolution_params&>(params);

    return p.output.Batch().v == 1 ? FORCE_PRIORITY_2 :  FORCE_PRIORITY_7;
}

bool ConvolutionKernel_b_fs_yx_fsv16::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o) || !CovolutionCheckInput(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    ConvolutionTuningData tuning_data = GetTuningParams(params);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    if (params.groups > 1) {
        auto outFeaturesPerGroup = output.Feature().v / params.groups;
        auto inFeaturesPerGroup = input.Feature().v / params.groups;
        auto multipleGroupsInputPreload = (tuning_data.feature_block_size % outFeaturesPerGroup == 0) &&
                                          (tuning_data.feature_block_size % inFeaturesPerGroup == 0) &&
                                          (tuning_data.feature_block_size / outFeaturesPerGroup > 1) &&
                                          (tuning_data.feature_block_size / inFeaturesPerGroup > 1) &&
                                          (outFeaturesPerGroup != 1) &&
                                          (inFeaturesPerGroup != 1);
        auto grouped = inFeaturesPerGroup % tuning_data.sub_group_size == 0 &&
                       (outFeaturesPerGroup % tuning_data.sub_group_size == 0 || tuning_data.sub_group_size % outFeaturesPerGroup == 0);

        if (!multipleGroupsInputPreload && !grouped)
            return false;
    }

    // Check that padding before features doesn't miss-align the blocks
    if (input.Feature().pad.before % tuning_data.feature_block_size != 0 || output.Feature().pad.before % tuning_data.feature_block_size != 0)
        return false;

    if (!params.bias.empty() && params.bias[0].GetDType() != input.GetDType())
        return false;

    return true;
}

JitConstants ConvolutionKernel_b_fs_yx_fsv16::GetJitConstants(const convolution_params& params,
                                                              const DispatchData& dispatchData) const {
    auto input = params.inputs[0];
    auto output = params.output;
    auto jit = Parent::GetJitConstants(params, dispatchData);

    ConvolutionTuningData tuning_data = GetTuningParams(params);

    auto blockWidth = dispatchData.cldnnStyle.blockWidth;
    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_vec = { "_VEC",
                                           {"b", "(feature_block * 16)", "y", "x"},
                                           "dst",
                                           input_dt,
                                           blockWidth,
                                           LoadType::LT_ALIGNED_READ,
                                           BoundaryCheck::ENABLED,
                                           IndexType::TENSOR_COORD,
                                           Tensor::DataChannelName::X };
        FusedOpsConfiguration conf_scalar = { "_SCALAR",
                                              {"b", "(feature_block * 16)", "y", "(x + i)"},
                                              "dst[i]",
                                              input_dt,
                                              1,
                                              LoadType::LT_ALIGNED_READ,
                                              BoundaryCheck::ENABLED,
                                              IndexType::TENSOR_COORD,
                                              Tensor::DataChannelName::X };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_vec, conf_scalar}));
    }

    size_t input_line_size = std::min(params.stride.x * (blockWidth - 1) + (params.weights.X().v - 1)*params.dilation.x + 1,
                                      input.X().v + input.X().pad.Total());

    auto outFeaturesPerGroup = output.Feature().v / params.groups;
    auto inFeaturesPerGroup = input.Feature().v / params.groups;
    auto multipleGroupsInputPreload = (tuning_data.feature_block_size % outFeaturesPerGroup == 0) &&
                                      (tuning_data.feature_block_size % inFeaturesPerGroup == 0) &&
                                      (tuning_data.feature_block_size / outFeaturesPerGroup > 1) &&
                                      (tuning_data.feature_block_size / inFeaturesPerGroup > 1);

    if (multipleGroupsInputPreload)
        jit.AddConstant(MakeJitConstant("MULTIPLE_GROUPS_INPUT_PRELOAD", 1));

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", blockWidth));
    jit.AddConstant(MakeJitConstant("INPUT_LINE_SIZE", input_line_size));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", tuning_data.sub_group_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(output.X().v, blockWidth)));
    jit.AddConstant(MakeJitConstant("SLM_DIV_FACTOR", tuning_data.slm_div_factor));
    jit.AddConstant(MakeJitConstant("WORK_GROUP_SIZE", tuning_data.work_group_size));
    jit.AddConstant(MakeJitConstant("IC_BLOCKS", CeilDiv(inFeaturesPerGroup, tuning_data.feature_block_size)));
    if (params.output.Feature().v % tuning_data.feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
    }
    if (inFeaturesPerGroup % tuning_data.feature_block_size != 0 && !multipleGroupsInputPreload) {
        jit.AddConstant(MakeJitConstant("INPUT_LEFTOVERS", 1));
    }

    return jit;
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16::GetTunedKernelsDataByIndex(const Params& params,
                                                                        const optional_params& options,
                                                                        const int autoTuneIndex) const {
    auto tuneOptions = GetAutoTuneOptions(params, autoTuneIndex);
    return GetCommonKernelsData(params, options, tuneOptions.exeMode, autoTuneIndex);
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16::GetKernelsDataForAutoTune(const Params& params,
                                                                       const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelsData res = {};

    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params, options, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

}  // namespace kernel_selector
