// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if GNA_LIB_VER == 2

#include <cstdint>

#define nLayerKind operation
#define intel_layer_kind_t gna_layer_operation
#define intel_gna_proc_t uint32_t


/**
 * Rounds a number up, to the nearest multiple of significance
 * Used for calculating the memory sizes of GNA data buffers
 *
 * @param number        Memory size or a number to round up.
 * @param significance  Informs the function how to round up. The function "ceils"
 *                      the number to the lowest possible value divisible by "significance".
 * @return Rounded integer value.
 */
#define ALIGN(number, significance) ((((number) + (significance) - 1) / (significance)) * (significance))

/**
 * Rounds a number up, to the nearest multiple of 64
 * Used for calculating memory sizes of GNA data arrays
 */
#define ALIGN64(number) ALIGN(number, 64)

#else

#include <gna-api.h>
#include <gna-api-types-xnn.h>

#define gna_pwl_segment_t intel_pwl_segment_t
#define gna_compound_bias_t intel_compound_bias_t
#define gna_nnet_layer_t intel_nnet_layer_t
#define gna_nnet_type_t intel_nnet_type_t
#define gna_affine_func_t intel_affine_func_t
#define gna_affine_layer_t intel_affine_layer_t
#define gna_convolutional_layer_t intel_convolutional_layer_t
#endif
