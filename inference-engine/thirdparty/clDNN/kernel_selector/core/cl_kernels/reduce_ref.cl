// Copyright (c) 2019 Intel Corporation
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

#include "include/include_all.cl"

inline uint FUNC(calc_linear_offset)(uint b, uint f, uint w, uint z, uint y, uint x)
{
    uint index = b * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z * OUTPUT_SIZE_W * OUTPUT_FEATURE_NUM +
                 f * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z * OUTPUT_SIZE_W +
                 w * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z +
                 z * OUTPUT_SIZE_X * OUTPUT_SIZE_Y +
                 y * OUTPUT_SIZE_X +
                 x;

    return index;
}

KERNEL(reduce_ref)(
    const __global INPUT0_TYPE* data,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint xy   = (uint)get_global_id(0);
    const uint wz   = (uint)get_global_id(1);
    const uint bf   = (uint)get_global_id(2);
    const uint x    = xy % OUTPUT_SIZE_X;
    const uint y    = xy / OUTPUT_SIZE_X;

    const uint b    = bf / OUTPUT_FEATURE_NUM;
    const uint f    = bf % OUTPUT_FEATURE_NUM;
#if INPUT0_DIMS == 4
    const uint w    = 0;
    const uint z    = 0;
    const uint out_idx = OUTPUT_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
    const uint z    = wz % OUTPUT_SIZE_Z;
    const uint w    = 0;
    const uint out_idx = OUTPUT_GET_INDEX(b, f, z, y, x);
#elif INPUT0_DIMS == 6
    const uint z    = wz % OUTPUT_SIZE_Z;
    const uint w    = wz / OUTPUT_SIZE_Z;
    const uint out_idx = OUTPUT_GET_INDEX(b, f, w, z, y, x);
#endif

    const uint linear_idx = FUNC_CALL(calc_linear_offset)(b, f, w, z, y, x);
    if (linear_idx >= COMPUTATIONAL_OPERATIONS_NUMBER)
        return;

#ifdef REDUCE_BATCH
    const uint batch_out = 0;
    const uint batch_max_val = INPUT0_BATCH_NUM;
#else
    const uint batch_out = BATCH_NUM_IDX_COMP(linear_idx);
    const uint batch_max_val = batch_out + 1;
#endif

#ifdef REDUCE_FEATURE
    const uint feature_out = 0;
    const uint feature_max_val = INPUT0_FEATURE_NUM;
#else
    const uint feature_out = FEATURE_NUM_IDX_COMP(linear_idx);
    const uint feature_max_val = feature_out + 1;
#endif

#if INPUT0_DIMS == 6
#ifdef REDUCE_W
    const uint w_out = 0;
    const uint w_max_val = INPUT0_SIZE_W;
#else
    const uint w_out = SIZE_W_IDX_COMP(linear_idx);
    const uint w_max_val = w_out + 1;
#endif
#else
    const uint w_out = 0;
    const uint w_max_val = 1;
#endif

#if INPUT0_DIMS == 6 || INPUT0_DIMS == 5
#ifdef REDUCE_Z
    const uint z_out = 0;
    const uint z_max_val = INPUT0_SIZE_Z;
#else
    const uint z_out = SIZE_Z_IDX_COMP(linear_idx);
    const uint z_max_val = z_out + 1;
#endif
#else
    const uint z_out = 0;
    const uint z_max_val = 1;
#endif

#ifdef REDUCE_Y
    const uint y_out = 0;
    const uint y_max_val = INPUT0_SIZE_Y;
#else
    const uint y_out = SIZE_Y_IDX_COMP(linear_idx);
    const uint y_max_val = y_out + 1;
#endif

#ifdef REDUCE_X
    const uint x_out = 0;
    const uint x_max_val = INPUT0_SIZE_X;
#else
    const uint x_out = SIZE_X_IDX_COMP(linear_idx);
    const uint x_max_val = x_out + 1;
#endif
    ACCUMULATOR_TYPE acc = ACCUMULATOR_VAL_ZERO;
    uint counter = 0;
    for (uint bi = batch_out; bi < batch_max_val; ++bi) {
        for (uint fi = feature_out; fi < feature_max_val; ++fi) {
            for (uint wi = w_out; wi < w_max_val; ++wi) {
                for (uint zi = z_out; zi < z_max_val; ++zi) {
                    for (uint yi = y_out; yi < y_max_val; ++yi) {
                        for (uint xi = x_out; xi < x_max_val; ++xi) {
#if INPUT0_DIMS == 6
                            const uint input_idx = INPUT0_GET_INDEX(bi, fi, wi, zi, yi, xi);
#elif INPUT0_DIMS == 5
                            const uint input_idx = INPUT0_GET_INDEX(bi, fi, zi, yi, xi);
#else
                            const uint input_idx = INPUT0_GET_INDEX(bi, fi, yi, xi);

#endif
#ifdef REDUCE_SUM_MODE
                            acc += data[input_idx];
#elif REDUCE_MAX_MODE
                            if (counter == 0)
                                acc = data[input_idx];
                            else
                                acc = data[input_idx] > acc ? data[input_idx] : acc;
#elif REDUCE_MIN_MODE
                            if (counter == 0)
                                acc = data[input_idx];
                            else
                                acc = data[input_idx] < acc ? data[input_idx] : acc;
#elif REDUCE_MEAN_MODE
                            acc += data[input_idx];
#elif REDUCE_PROD_MODE
                            if (counter == 0)
                                acc = data[input_idx];
                            else
                                acc *= data[input_idx];
#elif REDUCE_AND_MODE
                            if (counter == 0)
                                acc = data[input_idx];
                            else
                                acc = acc && data[input_idx];
#elif REDUCE_OR_MODE
                            if (counter == 0)
                                acc = data[input_idx];
                            else
                                acc = acc || data[input_idx];
#elif REDUCE_SUM_SQUARE_MODE
                            acc += data[input_idx] * data[input_idx];
#elif REDUCE_L1_MODE
                        #if !INPUT0_IS_FP
                            acc += TO_ACCUMULATOR_TYPE(fabs(TO_FINAL_ACCUMULATOR_TYPE(data[input_idx])));
                        #else
                            acc += fabs(data[input_idx]);
                        #endif
#elif REDUCE_L2_MODE
                            acc += data[input_idx] * data[input_idx];
#elif REDUCE_LOG_SUM_MODE
                            acc += data[input_idx];
#elif REDUCE_LOG_SUM_EXP_MODE
                        #if !INPUT0_IS_FP
                            acc += TO_ACCUMULATOR_TYPE(exp(TO_FINAL_ACCUMULATOR_TYPE(data[input_idx])));
                        #else
                            acc += exp(data[input_idx]);
                        #endif
#endif
                            counter++;
                        }
                    }
                }
            }
        }
    }

    FINAL_ACCUMULATOR_TYPE final_acc = TO_FINAL_ACCUMULATOR_TYPE(acc);
#if REDUCE_MEAN_MODE
    if (counter != 0) final_acc /= counter;
#endif
#if REDUCE_L2_MODE
    final_acc = sqrt(final_acc);
#endif
#if REDUCE_LOG_SUM_MODE || REDUCE_LOG_SUM_EXP_MODE
    final_acc = log(final_acc);
#endif

    OUTPUT_TYPE final_result;
    ACTIVATION_TYPE reduce_result = TO_ACTIVATION_TYPE(final_acc);
#if HAS_FUSED_OPS
    FUSED_OPS;
    final_result = FUSED_OPS_RESULT;
#else
    final_result = TO_OUTPUT_TYPE(ACTIVATION(reduce_result, ACTIVATION_PARAMS));
#endif
    output[out_idx] = final_result;
}
