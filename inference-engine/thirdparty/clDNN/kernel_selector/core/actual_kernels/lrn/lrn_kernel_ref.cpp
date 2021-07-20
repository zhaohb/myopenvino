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


#include "lrn_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
ParamsKey LRNKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableLRNMode(LRNMode::WITHIN_CHANNEL);
    k.EnableLRNMode(LRNMode::ACROSS_CHANNEL);
    k.EnableLRNKernelDividerMode(KernelDividerMode::DYNAMIC);
    k.EnableLRNKernelDividerMode(KernelDividerMode::FIXED);
    k.EnableDifferentTypes();
    return k;
}

JitConstants LRNKernelRef::GetJitConstants(const lrn_params& params, const LRNKernelRef::Parent::DispatchData& dispatchData) const {
    const uint32_t round_norm_size = (params.localSize / 2) * 2 + 1;
    uint32_t numElement = round_norm_size * round_norm_size;
    const auto& input_dt = params.inputs[0].GetDType();

    if (params.normMode == LRNMode::ACROSS_CHANNEL) {
        numElement = round_norm_size;
    }

    const float num_element_div = 1.f / static_cast<float>(numElement);

    JitConstants jit = Parent::GetJitConstants(params, dispatchData);
    jit.AddConstants({
        MakeJitConstant("NUM_ELEMENTS_DIV", num_element_div),
        MakeJitConstant("GWS_BATCH", 2),
        MakeJitConstant("GWS_FEATURE", 1),
        MakeJitConstant("GWS_YX", 0),
    });

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {"", {"b", "f", "y", "x"}, "lrn_result", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

LRNKernelRef::Parent::DispatchData LRNKernelRef::SetDefault(const lrn_params& params) const {
    DispatchData dispatchData = Parent::SetDefault(params);

    const auto& out = params.output;

    dispatchData.gws = { out.X().v * out.Y().v, out.Feature().v, out.Batch().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData LRNKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

KernelsPriority LRNKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
