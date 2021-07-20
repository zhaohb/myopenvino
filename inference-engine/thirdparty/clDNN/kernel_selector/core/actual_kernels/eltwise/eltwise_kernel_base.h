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


#pragma once

#include "kernel_base_opencl.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// eltwise_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct eltwise_params : public base_params {
    eltwise_params() : base_params(KernelType::ELTWISE) {}

    struct InputType {
        EltwiseInputMode mode = EltwiseInputMode::INPUT_BUFFER;
        uint32_t index = 0;     // for inputs results;
        uint32_t tmpIndex = 0;  // for temp results;
        float scalar = 0.f;

        static InputType Buffer(uint32_t index) {
            eltwise_params::InputType input;
            input.mode = EltwiseInputMode::INPUT_BUFFER;
            input.index = index;
            return input;
        }

        static InputType UnorderedAccessBuffer(uint32_t index, uint32_t tmpIndex) {
            eltwise_params::InputType input;
            input.mode = EltwiseInputMode::UNORDERED_ACCESS_INPUT_BUFFER;
            input.index = index;
            input.tmpIndex = tmpIndex;
            return input;
        }

        static InputType Intermediate(uint32_t tmpIndex) {
            eltwise_params::InputType input;
            input.mode = EltwiseInputMode::INTERMEDIATE_RESULTS_INDEX;
            input.tmpIndex = tmpIndex;
            return input;
        }

        static InputType Scalar(float s) {
            eltwise_params::InputType input;
            input.mode = EltwiseInputMode::SCALAR;
            input.scalar = s;
            return input;
        }

        static InputType OutBuffer() {
            eltwise_params::InputType output;
            output.mode = EltwiseInputMode::OUTPUT_BUFFER;
            return output;
        }
    };

    struct Node {
        std::vector<InputType> inputs;
        EltwiseMode mode;
    };

    struct UpdateInputData {
        uint32_t inputId;
        uint32_t tmpId;
    };

    std::vector<eltwise_params::Node> operations;
    std::vector<float> coefficients;
    std::vector<UpdateInputData> updateInputIds;
    std::vector<uSize> stride;

    bool layoutBased = false;
    bool int8_quantization = false;
    bool broadcast = false;

    virtual ParamsKey GetParamsKey() const;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// eltwise_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct eltwise_optional_params : optional_params {
    eltwise_optional_params() : optional_params(KernelType::ELTWISE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// fuse_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct eltwise_fuse_params : fuse_params {
    EltwiseMode mode;

    eltwise_fuse_params(EltwiseMode mode) : fuse_params(KernelType::ELTWISE), mode(mode) {}
};

struct scale_fuse_params : fuse_params {
    scale_fuse_params() : fuse_params(KernelType::SCALE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EltwiseKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class EltwiseKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~EltwiseKernelBase() {}

    using DispatchData = CommonDispatchData;
    JitConstants GetJitConstantsCommon(const eltwise_params& params, bool useVload8) const;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    virtual JitConstants GetJitConstants(const eltwise_params& params) const;
    virtual JitConstants GetOperationsJitConstants(const eltwise_params& params, bool useVload8, size_t blockSize = 1) const;
    virtual JitConstants MakeLoadJitConstants(const eltwise_params& params, bool useVload8) const;
    virtual JitConstants MakeIndexJitConstants(const eltwise_params& params, bool useVload8) const;
    virtual JitConstants MakeInputDeclsJitConstants(const eltwise_params& params, bool useVload8) const;
    virtual DispatchData SetDefault(const eltwise_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params& options) const;
    Datatype GetAccumulatorType(const eltwise_params &params) const;
};
}  // namespace kernel_selector
