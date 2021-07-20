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

#include "depth_to_space_kernel_base.h"

namespace kernel_selector {
class DepthToSpaceKernelBlock2Opt : public DepthToSpaceKernelBase {
public:
    using Parent = DepthToSpaceKernelBase;

    DepthToSpaceKernelBlock2Opt() : DepthToSpaceKernelBase("depth_to_space_block2_opt") {}
    virtual ~DepthToSpaceKernelBlock2Opt() {}

    bool Validate(const Params&, const optional_params&) const override;
    JitConstants GetJitConstants(const depth_to_space_params& params) const override;
    CommonDispatchData SetDefault(const depth_to_space_params& params) const override;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
