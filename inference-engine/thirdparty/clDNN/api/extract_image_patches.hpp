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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief The ExtractImagePatches operation collects patches from the input tensor, as if applying a convolution.
/// All extracted patches are stacked in the depth dimension of the output.
/// @details The ExtractImagePatches operation is similar to the TensorFlow*
/// operation ExtractImagePatches.
/// This op extracts patches of shape `sizes` which are `strides` apart in the
/// input image. The output elements are taken from the input at intervals
/// given by the `rate` argument, as in dilated convolutions.
/// The result is a 4D tensor containing image patches with size
/// `size[0] * size[1] * depth` vectorized in the "depth" dimension.
/// The "auto_pad" attribute has no effect on the size of each patch, it
/// determines how many patches are extracted.
struct extract_image_patches : public primitive_base<extract_image_patches> {
    CLDNN_DECLARE_PRIMITIVE(extract_image_patches)

    /// @brief Constructs select primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id containing input 4-D tensor.
    /// @param sizes Vector with sizes.
    /// @param strides Vector with strides.
    /// @param rates Vector with rates.
    /// @param auto_pad How the padding is calculated.
    /// @param output_shape Tensor with shape of output layout
    extract_image_patches(const primitive_id& id,
                          const primitive_id& input,
                          const std::vector<unsigned int>& sizes,
                          const std::vector<unsigned int>& strides,
                          const std::vector<unsigned int>& rates,
                          const std::string& auto_pad,
                          const tensor& output_shape,
                          const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          sizes(sizes),
          strides(strides),
          rates(rates),
          auto_pad(auto_pad),
          output_shape(output_shape) {}

    /// @brief Vector with sizes
    std::vector<unsigned int> sizes;
    /// @brief Vector with strides
    std::vector<unsigned int> strides;
    /// @brief Vector with rates
    std::vector<unsigned int> rates;
    /// @brief Mode how the padding is calculated
    std::string auto_pad;
    /// @brief Shape of output layout
    tensor output_shape;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
