﻿// Copyright (c) 2018-2021 Intel Corporation
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


#include "mvn_kernel_base.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {

bool MVNKernelBase::Validate(const Params& params, const optional_params&) const {
    const mvn_params& orgParams = static_cast<const mvn_params&>(params);

    for (auto& fused_op : orgParams.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

JitConstants MVNKernelBase::GetJitConstants(const mvn_params& params, MVNKernelBase::DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("EPSILON", params.epsilon),
        MakeJitConstant(toString(params.mvnMode), ""),
        MakeJitConstant("NORMALIZE_VARIANCE", params.mvnNormalizeVariance),
        MakeJitConstant("EPS_" + toString(params.mvnEpsMode), ""),
    });

    return jit;
}

MVNKernelBase::DispatchData MVNKernelBase::SetDefault(const mvn_params& params) const {
    const auto& output = params.output;

    DispatchData dispatchData;
    if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
        dispatchData.gws = {output.Batch().v, output.Feature().v, 1};
    } else {
        dispatchData.gws = {output.Batch().v, 1, 1};
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData MVNKernelBase::GetCommonKernelsData(const Params& params,
                                                const optional_params& options) const {
    assert(params.GetType() == KernelType::MVN);

    if (!Validate(params, options))
        return {};

    const mvn_params& orgParams = static_cast<const mvn_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<mvn_params>(params);

    auto finalKernelName = GetKernelName(orgParams);
    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, options);
    auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     finalKernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params));

    return {kd};
}

Datatype MVNKernelBase::GetActivationType(const mvn_params& params) const {
    if (params.output.GetDType() == Datatype::F16)
        return Datatype::F16;
    return Datatype::F32;
}

}  // namespace kernel_selector
