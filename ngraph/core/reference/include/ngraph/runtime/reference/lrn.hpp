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

#include <algorithm>
#include <cmath>
#include <numeric>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            static size_t point_to_flat_idx(const Shape& shape, const std::vector<size_t>& point)
            {
                size_t idx = point[0];
                for (int i = 1; i < point.size(); i++)
                {
                    idx *= shape[i];
                    idx += point[i];
                }
                return idx;
            }

            static std::vector<size_t> slice_indices(const Shape& full_shape,
                                                     const std::vector<size_t>& begin,
                                                     const Shape& slice_shape)
            {
                size_t slice_size = shape_size(slice_shape);
                size_t rank = begin.size();
                auto coord = begin;
                std::vector<size_t> indices;
                indices.reserve(slice_size);
                indices.push_back(point_to_flat_idx(full_shape, coord));
                for (int i = 0; i < slice_size - 1; i++)
                {
                    for (int r = rank - 1; r >= 0; r--)
                    {
                        coord[r]++;
                        if (coord[r] < (begin[r] + slice_shape[r]))
                            break;
                        coord[r] = begin[r];
                    }
                    indices.push_back(point_to_flat_idx(full_shape, coord));
                }
                return indices;
            }

            template <typename T>
            static T sum_region_across_axes(const T* arg, const std::vector<size_t>& indices)
            {
                T square_sum = 0;
                for (auto index : indices)
                {
                    square_sum += arg[index] * arg[index];
                }
                return square_sum;
            }

            template <typename T>
            void lrn(const T* arg,
                     const AxisSet& axes,
                     T* out,
                     const Shape& arg_shape,
                     double dalpha,
                     double dbeta,
                     double dbias,
                     size_t size)
            {
                T alpha = static_cast<T>(dalpha);
                T beta = static_cast<T>(dbeta);
                T bias = static_cast<T>(dbias);
                T scale = alpha / std::pow(size, axes.size());

                std::vector<size_t> begin_area(arg_shape.size());
                Shape area_shape(arg_shape.size(), 1);
                std::vector<bool> axes_map(arg_shape.size(), false);
                for (const auto& axis_coord : axes)
                {
                    axes_map[axis_coord] = true;
                }

                CoordinateTransform input_transform(arg_shape);
                for (const Coordinate& in_coord : input_transform)
                {
                    // area determined by in_coord local neighborhood
                    for (size_t i = 0; i < axes_map.size(); i++)
                    {
                        if (axes_map[i])
                        {
                            begin_area[i] = std::max<int>(0, in_coord.at(i) - (size - 1) / 2);
                            area_shape[i] = std::min<int>(arg_shape.at(i),
                                                          in_coord.at(i) + (size - 1) / 2 + 1) -
                                            begin_area[i];
                        }
                        else
                        {
                            begin_area[i] = in_coord.at(i);
                        }
                    }

                    T square_sum = sum_region_across_axes(
                        arg, slice_indices(arg_shape, begin_area, area_shape));
                    auto index = input_transform.index(in_coord);
                    T x = arg[index];
                    out[index] = x / (std::pow(bias + scale * square_sum, beta));
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
