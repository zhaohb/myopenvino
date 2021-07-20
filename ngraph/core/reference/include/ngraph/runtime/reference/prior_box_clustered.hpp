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

#include <cmath>

#include "ngraph/axis_vector.hpp"
#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/prior_box_clustered.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void prior_box_clustered(const T* data,
                                     const T* img,
                                     float* dst_data,
                                     const Shape& out_shape,
                                     const op::PriorBoxClusteredAttrs& attrs)
            {
                size_t num_priors_ = attrs.widths.size();

                auto variances = attrs.variances;
                if (variances.empty())
                    variances.push_back(0.1f);

                // Execute
                const int64_t layer_width = data[1];
                const int64_t layer_height = data[0];

                int64_t img_width = img[1];
                int64_t img_height = img[0];

                // TODO: Uncomment after PriorBoxClustered is aligned with the specification.

                //                int img_width = img_w_ == 0 ? img[1] : img_w_;
                //                int img_height = img_h_ == 0 ? img[0] : img_h_;

                //                float step_w = attrs.step_widths == 0 ? step_ : attrs.step_widths;
                //                float step_h = attrs.step_heights == 0 ? step_ :
                //                attrs.step_heights;

                float step_w = attrs.step_widths;
                float step_h = attrs.step_heights;

                if (step_w == 0 && step_h == 0)
                {
                    step_w = static_cast<float>(img_width) / layer_width;
                    step_h = static_cast<float>(img_height) / layer_height;
                }

                size_t var_size = variances.size();
                for (int64_t h = 0; h < layer_height; ++h)
                {
                    for (int64_t w = 0; w < layer_width; ++w)
                    {
                        float center_x = (w + attrs.offset) * step_w;
                        float center_y = (h + attrs.offset) * step_h;

                        for (size_t s = 0; s < num_priors_; ++s)
                        {
                            float box_width = attrs.widths[s];
                            float box_height = attrs.heights[s];

                            float xmin = (center_x - box_width / 2.0f) / img_width;
                            float ymin = (center_y - box_height / 2.0f) / img_height;
                            float xmax = (center_x + box_width / 2.0f) / img_width;
                            float ymax = (center_y + box_height / 2.0f) / img_height;

                            if (attrs.clip)
                            {
                                xmin = (std::min)((std::max)(xmin, 0.0f), 1.0f);
                                ymin = (std::min)((std::max)(ymin, 0.0f), 1.0f);
                                xmax = (std::min)((std::max)(xmax, 0.0f), 1.0f);
                                ymax = (std::min)((std::max)(ymax, 0.0f), 1.0f);
                            }

                            auto get_idx = [&](uint64_t cnt) -> uint64_t {
                                return h * layer_width * num_priors_ * cnt + w * num_priors_ * cnt +
                                       s * cnt;
                            };

                            uint64_t idx = get_idx(4);
                            dst_data[idx + 0] = xmin;
                            dst_data[idx + 1] = ymin;
                            dst_data[idx + 2] = xmax;
                            dst_data[idx + 3] = ymax;

                            idx = get_idx(var_size);
                            for (size_t j = 0; j < var_size; j++)
                                dst_data[idx + j + out_shape[1]] = variances[j];
                        }
                    }
                }
            }
        }
    }
}
