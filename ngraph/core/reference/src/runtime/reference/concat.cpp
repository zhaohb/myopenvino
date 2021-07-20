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

#include <cstring>

#include "ngraph/runtime/reference/concat.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace
            {
                std::vector<size_t> calculate_shape_sizes(const std::vector<Shape>& in_shapes)
                {
                    std::vector<size_t> sizes;
                    sizes.reserve(in_shapes.size());
                    std::transform(begin(in_shapes),
                                   end(in_shapes),
                                   std::back_inserter(sizes),
                                   [](const Shape& shape) { return shape_size(shape); });
                    return sizes;
                }
            } // namespace

            void concat(const std::vector<const char*>& args,
                        char* out,
                        const std::vector<Shape>& in_shapes,
                        const Shape& out_shape,
                        int64_t concatenation_axis,
                        size_t elem_size)
            {
                size_t steps = 1;
                for (int i = 0; i < concatenation_axis; ++i)
                {
                    steps *= out_shape[i];
                }

                const auto& shape_sizes = calculate_shape_sizes(in_shapes);

                size_t out_offset = 0;
                for (size_t step = 0; step < steps; ++step)
                {
                    for (size_t in_index = 0; in_index < args.size(); ++in_index)
                    {
                        const size_t size = shape_sizes[in_index] / steps;
                        const size_t in_offset = step * size;

                        std::memcpy(&out[out_offset * elem_size],
                                    &args[in_index][in_offset * elem_size],
                                    size * elem_size);

                        out_offset += size;
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
