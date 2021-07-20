//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cassert>
#include <cfenv>
#include <cmath>
#include <functional>
#include <numeric>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/runtime/reference/helpers.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/split.hpp"
#include "ngraph/util.hpp"

// can't be removed currently due to arm-plugin dependency
#include "ngraph/runtime/reference/convolution_backprop_data.hpp"
namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace
            {
                constexpr size_t in_batch_axis = 0;
                constexpr size_t in_channel_axis = 1;
                constexpr size_t filter_out_ch_axis = 0;
                constexpr size_t filter_in_ch_axis = 1;
                constexpr size_t out_batch_axis = 0;
                constexpr size_t out_channel_axis = 1;
                constexpr size_t spatial_axis = 2;

                struct ConvolutionParams
                {
                    std::vector<int> strides;
                    std::vector<int> dilation;
                    std::vector<int> pads_begin;
                    std::vector<int> pads_end;

                    ConvolutionParams(const Strides& strides_,
                                      const Strides& dilation_,
                                      const CoordinateDiff& pads_begin_,
                                      const CoordinateDiff& pads_end_)
                        : strides{strides_.begin(), strides_.end()}
                        , dilation{dilation_.begin(), dilation_.end()}
                        , pads_begin{pads_begin_.begin(), pads_begin_.end()}
                        , pads_end{pads_end_.begin(), pads_end_.end()} {};
                };

                template <typename Int>
                constexpr inline bool in_range(Int val, std::pair<Int, Int> range) noexcept
                {
                    return val >= range.first && val < range.second;
                }

                template <typename T>
                void convolve_3D_channels(const ConvolutionParams& p,
                                          const T* batch,
                                          const Shape& batch_shape,
                                          const T* filter,
                                          const Shape& filter_shape,
                                          T*& out)
                {
                    const int input_size_z = batch_shape[1];
                    const int input_size_y = batch_shape[2];
                    const int input_size_x = batch_shape[3];
                    const int filter_size_z = filter_shape[1];
                    const int filter_size_y = filter_shape[2];
                    const int filter_size_x = filter_shape[3];
                    const int dilated_filter_size_z =
                        filter_size_z + (filter_size_z - 1) * (p.dilation[0] - 1);
                    const int dilated_filter_size_y =
                        filter_size_y + (filter_size_y - 1) * (p.dilation[1] - 1);
                    const int dilated_filter_size_x =
                        filter_size_x + (filter_size_x - 1) * (p.dilation[2] - 1);

                    const Shape input_channel_shape(++batch_shape.begin(), batch_shape.end());
                    const size_t input_channel_size = shape_size(input_channel_shape);
                    const Shape filter_channel_shape(++filter_shape.begin(), filter_shape.end());
                    const size_t filter_channel_size = shape_size(filter_channel_shape);

                    for (int i_z = -p.pads_begin[0];
                         i_z <= (p.pads_end[0] + input_size_z - dilated_filter_size_z);
                         i_z += p.strides[0])
                    {
                        for (int i_y = -p.pads_begin[1];
                             i_y <= (p.pads_end[1] + input_size_y - dilated_filter_size_y);
                             i_y += p.strides[1])
                        {
                            for (int i_x = -p.pads_begin[2];
                                 i_x <= (p.pads_end[2] + input_size_x - dilated_filter_size_x);
                                 i_x += p.strides[2])
                            {
                                auto input_channel = batch;
                                auto filter_channel = filter;
                                T sum = 0;
                                size_t filter_channels_count = filter_shape[0];
                                while (filter_channels_count--)
                                {
                                    for (int f_z = 0; f_z < filter_size_z; ++f_z)
                                    {
                                        for (int f_y = 0; f_y < filter_size_y; ++f_y)
                                        {
                                            for (int f_x = 0; f_x < filter_size_x; ++f_x)
                                            {
                                                int rel_i_z = i_z + (f_z * p.dilation[0]);
                                                int rel_i_y = i_y + (f_y * p.dilation[1]);
                                                int rel_i_x = i_x + (f_x * p.dilation[2]);

                                                bool padding =
                                                    !(in_range(rel_i_x, {0, input_size_x}) &&
                                                      in_range(rel_i_y, {0, input_size_y}) &&
                                                      in_range(rel_i_z, {0, input_size_z}));
                                                if (padding)
                                                    continue;

                                                int f_buf_idx =
                                                    (f_z * filter_size_y * filter_size_x) +
                                                    (f_y * filter_size_x) + f_x;
                                                int i_buf_idx =
                                                    (rel_i_z * input_size_y * input_size_x) +
                                                    (rel_i_y * input_size_x) + rel_i_x;
                                                sum += static_cast<T>(input_channel[i_buf_idx]) *
                                                       static_cast<T>(filter_channel[f_buf_idx]);
                                            }
                                        }
                                    }
                                    input_channel += input_channel_size;
                                    filter_channel += filter_channel_size;
                                }
                                *out = sum;
                                ++out;
                            }
                        }
                    }
                }

                void extend_to_3D(ConvolutionParams& p, Shape& in_shape, Shape& filter_shape)
                {
                    int spatial_rank = in_shape.size() - 2;
                    if (spatial_rank < 3)
                    {
                        int missing_dims = 3 - spatial_rank;
                        p.dilation.insert(
                            std::prev(p.dilation.end(), spatial_rank), missing_dims, 1);
                        p.strides.insert(std::prev(p.strides.end(), spatial_rank), missing_dims, 1);
                        p.pads_begin.insert(
                            std::prev(p.pads_begin.end(), spatial_rank), missing_dims, 0);
                        p.pads_end.insert(
                            std::prev(p.pads_end.end(), spatial_rank), missing_dims, 0);
                        in_shape.insert(std::next(in_shape.end(), -spatial_rank), missing_dims, 1);
                        filter_shape.insert(
                            std::prev(filter_shape.end(), spatial_rank), missing_dims, 1);
                    }
                }
            }

            template <typename T>
            void convolution(const T* in,
                             const T* f,
                             T* out,
                             const Shape& in_shape,
                             const Shape& f_shape,
                             const Shape& out_shape,
                             const Strides& strides,
                             const Strides& dilation,
                             const CoordinateDiff& pads_begin,
                             const CoordinateDiff& pads_end)

            {
                // this implementation supports 1D, 2D and 3D convolutions
                NGRAPH_CHECK(in_shape.size() >= 3 && in_shape.size() <= 5,
                             "Unsupported input rank: ",
                             in_shape);

                NGRAPH_CHECK(f_shape.size() >= 3 && f_shape.size() <= 5,
                             "Unsupported kernel rank: ",
                             f_shape);

                // here we are converting all param types to int's to avoid arithmetic issues
                // (e.g signed + unsigned) in indexes calculation later
                ConvolutionParams params{strides, dilation, pads_begin, pads_end};

                // here we are extending spatial dimensions to 3D, because we are going to use 3D
                // convolution implementation to convolve also in 1D & 2D case
                Shape input_shape{in_shape};
                Shape filters_shape{f_shape};
                if (in_shape.size() < 5)
                {
                    extend_to_3D(params, input_shape, filters_shape);
                }

                const size_t batches_count = input_shape[in_batch_axis];
                const Shape batch_shape(++input_shape.begin(), input_shape.end());
                const size_t batch_size = shape_size(batch_shape);

                const size_t filters_count = filters_shape[filter_out_ch_axis];
                const Shape filter_shape(++filters_shape.begin(), filters_shape.end());
                const size_t filter_size = shape_size(filter_shape);

                auto batch = in;
                for (size_t batch_idx = 0; batch_idx < batches_count; ++batch_idx)
                {
                    auto filter = f;
                    for (size_t f_idx = 0; f_idx < filters_count; ++f_idx)
                    {
                        convolve_3D_channels(params, batch, batch_shape, filter, filter_shape, out);
                        filter += filter_size;
                    }
                    batch += batch_size;
                }
            }

            // DEPRECATED, can't be removed currently due to kmb-plugin dependency (#47799)
            template <typename INPUT,
                      typename FILTER,
                      typename OUTPUT,
                      typename ACCU = typename widen<OUTPUT>::type>
            void convolution(const INPUT* in,
                             const FILTER* f,
                             OUTPUT* out,
                             const Shape& in_shape,
                             const Shape& f_shape,
                             const Shape& out_shape,
                             const Strides& strides,
                             const Strides& dilation,
                             const CoordinateDiff& pads_begin,
                             const CoordinateDiff& pads_end,
                             const Strides&)

            {
                static_assert(std::is_same<INPUT, FILTER>::value,
                              "input and filter types must be the same");
                static_assert(std::is_same<INPUT, OUTPUT>::value,
                              "input and output types must be the same");

                convolution(in,
                            f,
                            out,
                            in_shape,
                            f_shape,
                            out_shape,
                            strides,
                            dilation,
                            pads_begin,
                            pads_end);
            }

        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
