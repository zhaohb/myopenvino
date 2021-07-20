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

#include "deconvolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class DeconvolutionKernel_imad_ref : public DeconvolutionKernelBase {
public:
    using Parent = DeconvolutionKernelBase;
    DeconvolutionKernel_imad_ref() : DeconvolutionKernelBase("deconvolution_gpu_imad_ref") {}
    virtual ~DeconvolutionKernel_imad_ref() = default;

    ParamsKey GetSupportedKey() const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const deconvolution_params &params) const override;
    CommonDispatchData SetDefault(const deconvolution_params& params) const override;
    JitConstants GetJitConstants(const deconvolution_params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE,
            FusedOpType::SCALE,
            FusedOpType::QUANTIZE
        };
    }

    size_t GetTileIFM(const deconvolution_params& params) const;
};

}  // namespace kernel_selector
