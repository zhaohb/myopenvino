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


#include "fully_connected_kernel_bf_io_gemm.h"
#include <vector>

namespace kernel_selector {

ParamsKey FullyConnected_bf_io_GEMM::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableAllInputLayout();
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableBiasPerOutput();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    // bfyx -> bf layout transformation works incorrectly when tensor has paddings, so offset support is disabled for now.
    // k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

FullyConnected_bf_io_GEMM::DispatchData FullyConnected_bf_io_GEMM::SetDefault(const fully_connected_params& params,
                                                                              int autoTuneIndex) const {
    auto dispatchData = Parent::SetDefault(params, autoTuneIndex);

    const uint32_t localWorkSizeX = 64;
    const uint32_t globalWorkSizeX = localWorkSizeX;

    dispatchData.gws = { globalWorkSizeX, params.output.Feature().v, 1 };
    dispatchData.lws = { localWorkSizeX, 1, 1 };

    return dispatchData;
}

KernelsPriority FullyConnected_bf_io_GEMM::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_6;
}

JitConstants FullyConnected_bf_io_GEMM::GetJitConstants(const fully_connected_params& params,
                                                        const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    if (params.inputs[0].GetDType() == Datatype::F16) {
        jit.AddConstant(MakeJitConstant("__fc_f16", ""));
    } else {
        jit.AddConstant(MakeJitConstant("__fc_f32", ""));
    }

    const uint32_t localWorkSizeX = 64;
    const uint32_t globalWorkSizeX = localWorkSizeX;
    const uint32_t vecSize = 4;
    size_t matrixLineSize = params.inputs[0].Batch().pitch;

    jit.AddConstants({
        MakeJitConstant("LAST_INPUT_SIZE_REMAINDER", matrixLineSize % (globalWorkSizeX * vecSize)),
        MakeJitConstant("LAST_INPUT_SIZE_DIV_4", matrixLineSize % vecSize),
    });

    return jit;
}

KernelsData FullyConnected_bf_io_GEMM::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    options,
                                                    DataLayout::bf,
                                                    WeightsLayout::oiyx,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}
}  // namespace kernel_selector
