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

#include "convolution_kernel_mmad_b_fs_yx_fsv32_dw.h"
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include <iostream>
#include <core/common/kernel_selector_utils.h>

namespace kernel_selector {

ParamsKey ConvolutionKernel_mmad_b_fs_yx_fsv32_dw::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_WEIGHTS);
    k.EnableDifferentTypes();
    k.DisableTuning();
    k.EnableGroupedConvolution();
    k.EnableDepthwiseSeparableOpt();
    k.EnableDifferentInputWeightsTypes();
    return k;
}


bool ConvolutionKernel_mmad_b_fs_yx_fsv32_dw::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o)) {
        return false;
    }

    auto params = dynamic_cast<const convolution_params&>(p);

    if (params.inputs[0].Feature().v != params.groups || params.output.Feature().v != params.groups)
        return false;

    if ((params.quantization == QuantizationType::ASYMMETRIC_DATA || params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS)
        && !params.HasCompensation()) {
        return false;
    }

    return true;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_mmad_b_fs_yx_fsv32_dw::SetDefault(const convolution_params& cp,
                                                                                        int /*autoTuneIndex*/) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(cp);

    dispatchData.gws = { cp.output.Feature().v, cp.output.X().v * cp.output.Y().v, cp.output.Batch().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, cp.engineInfo);

    return dispatchData;
}

KernelsPriority ConvolutionKernel_mmad_b_fs_yx_fsv32_dw::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_3;
}

// TODO: optimize this kernel
JitConstants ConvolutionKernel_mmad_b_fs_yx_fsv32_dw::GetJitConstants(const convolution_params& params,
                                                                      const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_scalar = {"", {"b", "f", "y", "x"}, "res", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return jit;
}


KernelsData ConvolutionKernel_mmad_b_fs_yx_fsv32_dw::GetKernelsData(const Params& params,
                                                                    const optional_params& options) const {
    KernelsData kd = GetTunedKernelsDataByIndex(params, options);
    return kd;
}

}  // namespace kernel_selector
