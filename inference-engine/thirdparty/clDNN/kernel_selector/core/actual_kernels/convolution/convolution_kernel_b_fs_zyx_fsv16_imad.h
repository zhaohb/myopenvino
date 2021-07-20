/*
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
*/

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class Convolution_kernel_b_fs_zyx_fsv16_imad : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    Convolution_kernel_b_fs_zyx_fsv16_imad() : ConvolutionKernelBase("convolution_gpu_b_fs_zyx_fsv16_imad") {}
    virtual ~Convolution_kernel_b_fs_zyx_fsv16_imad() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& params, const optional_params& options) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;
    bool NeedPaddedInput() const override { return true; }
    WeightsLayout GetPreferredWeightsLayout(const convolution_params& p) const override {
        return p.groups > 1 ? WeightsLayout::g_os_is_zyx_osv16_isv16 : WeightsLayout::os_is_zyx_osv16_isv16;
    }

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }

    struct BlockParams {
        size_t output_block_width;
        size_t output_block_height;
        size_t output_block_depth;

        size_t output_block_features;

        size_t input_block_width;
        size_t input_block_height;
        size_t input_block_depth;

        size_t feature_slm_split;
    };

    BlockParams GetBlockParams(const convolution_params& params) const;
    float EstimateBlockParamsRatio(const convolution_params& params, const BlockParams& block) const;
    float EstimateRegPressure(const convolution_params& params, const BlockParams& block) const;
    float EstimateOccupancy(const convolution_params& params, const BlockParams& block) const;
    float EstimateSLMUsage(const convolution_params& params, const BlockParams& block) const;
};
}  // namespace kernel_selector
