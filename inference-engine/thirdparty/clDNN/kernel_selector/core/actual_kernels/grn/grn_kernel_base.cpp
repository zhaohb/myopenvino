﻿// Copyright (c) 2020 Intel Corporation
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


#include "grn_kernel_base.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
JitConstants GRNKernelBase::GetJitConstants(const grn_params& params, GRNKernelBase::DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstant(MakeJitConstant("BIAS", params.bias));

    return jit;
}

GRNKernelBase::DispatchData GRNKernelBase::SetDefault(const grn_params& params) const {
    const auto& output = params.output;

    DispatchData dispatchData;
    dispatchData.gws = { output.Batch().v, output.Y().v, output.X().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData GRNKernelBase::GetCommonKernelsData(const Params& params,
                                                const optional_params& options) const {
    assert(params.GetType() == KernelType::GRN);

    if (!Validate(params, options))
        return {};

    const grn_params& orgParams = static_cast<const grn_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<grn_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params));

    return {kd};
}

}  // namespace kernel_selector
