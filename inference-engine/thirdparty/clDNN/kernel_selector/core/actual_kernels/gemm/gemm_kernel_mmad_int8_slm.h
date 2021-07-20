// Copyright (c) 2020 Intel Corporation
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

#pragma once

#include "gemm_kernel_base.h"
#include <vector>

namespace kernel_selector {
class GemmKernelMMADslmInt8 : public GemmKernelBase {
public:
    using Parent = GemmKernelBase;
    using DispatchData = CommonDispatchData;
    struct GemmTuningData {
        size_t size_m;
        size_t size_n;
        size_t size_k;

        const size_t slm_tile_size = 32;
        const size_t simd_size = 8;
        const size_t pack_size = 4;
        const size_t max_slm_preloading_size = 256;
        size_t slm_decimation_factor = 2;
    };

    GemmKernelMMADslmInt8() : GemmKernelBase("gemm_mmad_int8_slm") {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION,
                 FusedOpType::SCALE,
                 FusedOpType::ELTWISE };
    }
    bool Validate(const Params& params, const optional_params& options) const override;
    JitConstants GetJitConstants(const gemm_params& params) const override;
    DispatchData SetDefault(const gemm_params& params) const override;
    GemmTuningData InitGemmTuningData(const gemm_params& params) const;
    GemmTuningData SetTuningParams(const gemm_params& params) const;
    size_t GetMmadOperationsNumber(const GemmTuningData& tuning_data) const;
    bool HasLeftovers(const GemmTuningData& tuning_data) const;
};
}  // namespace kernel_selector
