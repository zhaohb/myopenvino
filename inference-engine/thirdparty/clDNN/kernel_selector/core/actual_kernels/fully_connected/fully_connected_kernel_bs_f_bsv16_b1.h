// Copyright (c) 2016 Intel Corporation
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

#include "fully_connected_kernel_base.h"

namespace kernel_selector {

class FullyConnected_bs_f_bsv16_b1 : public FullyConnectedKernelBase {
public:
    FullyConnected_bs_f_bsv16_b1() : FullyConnectedKernelBase("fully_connected_gpu_bs_f_bsv16_b1") {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const fully_connected_params& params,
                                 const FullyConnectedKernelBase::DispatchData& dispatchData) const override;
    DispatchData SetDefault(const fully_connected_params& arg, int autoTuneIndex = -1) const override;
};
}  // namespace kernel_selector
