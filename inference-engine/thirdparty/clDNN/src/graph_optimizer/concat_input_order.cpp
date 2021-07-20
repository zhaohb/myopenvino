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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "pooling_inst.h"
#include "convolution_inst.h"
#include "fully_connected_inst.h"
#include "data_inst.h"
#include "memory_impl.h"
#include "program_impl.h"

#include <vector>
#include <tuple>

using namespace cldnn;

namespace {

using shuffle_range = std::pair<int32_t, int32_t>;

bool can_shuffle_features(program_node& node) {
    if (node.is_type<convolution>()) {
        auto& conv_node = node.as<convolution>();
        auto& wei_node = conv_node.weights();

        return conv_node.get_groups() == 1 && conv_node.get_split() == 1 &&
            conv_node.get_deformable_groups() == 1 && !conv_node.get_transposed() &&
            !conv_node.activations_zero_points_term() &&
            wei_node.is_type<data>() && wei_node.is_constant() && !wei_node.is_output();
    }
    if (node.is_type<fully_connected>()) {
        auto& fc_node = node.as<fully_connected>();
        auto& wei_node = fc_node.weights();

        return wei_node.is_type<data>() && wei_node.is_constant() && !wei_node.is_output();
    }

    bool pass_through = false;
    pass_through |= node.is_type<activation>();
    pass_through |= node.is_type<pooling>();
    // General conditions for pass-through layers
    pass_through &= !node.is_output() && node.get_dependencies().size() == 1 && !node.has_fused_primitives();
    if (pass_through) {
        // Primitives that are feature order invariant, pass-through shuffled features to users
        for (auto& user : node.get_users()) {
            if (!can_shuffle_features(*user))
                return false;
        }
        return true;
    }

    return false;
}

void shuffle_weights(data_node& node, const std::vector<shuffle_range>& ranges) {
    // Correct for shuffled features by shuffling input feature dimension in weights.
    // This allows to restore correct feature order on output and only changes calculation order.
    auto wei_layout = node.get_output_layout();
    auto& old_weights_memory = node.get_attached_memory();
    bool need_reset = static_cast<bool>(wei_layout.data_padding) || wei_layout.format.is_blocked();
    auto new_weights_memory = old_weights_memory.get_engine()->allocate_memory(wei_layout, old_weights_memory.get_net_id(), need_reset);

    auto bytes_per_elem = data_type_traits::size_of(wei_layout.data_type);
    auto old_ptr = static_cast<char*>(old_weights_memory.lock());
    auto new_ptr = static_cast<char*>(new_weights_memory->lock());
    for (int32_t ofi = 0; ofi < wei_layout.size.batch[0]; ++ofi) {
        int32_t new_ifi = 0;
        for (auto& range : ranges) {
            for (int32_t ifi = range.first; ifi < range.second; ++ifi, ++new_ifi) {
                for (int32_t wi = 0; wi < wei_layout.size.spatial[3]; ++wi) {
                    for (int32_t zi = 0; zi < wei_layout.size.spatial[2]; ++zi) {
                        for (int32_t yi = 0; yi < wei_layout.size.spatial[1]; ++yi) {
                            for (int32_t xi = 0; xi < wei_layout.size.spatial[0]; ++xi) {
                                auto old_coords = tensor(batch(ofi), feature(ifi), spatial(xi, yi, zi, wi));
                                auto new_coords = tensor(batch(ofi), feature(new_ifi), spatial(xi, yi, zi, wi));
                                auto old_offset = wei_layout.get_linear_offset(old_coords);
                                auto new_offset = wei_layout.get_linear_offset(new_coords);
                                for (size_t byte = 0; byte < bytes_per_elem; ++byte) {
                                    new_ptr[new_offset * bytes_per_elem + byte] = old_ptr[old_offset * bytes_per_elem + byte];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    old_weights_memory.unlock();
    new_weights_memory->unlock();

    node.attach_memory(*new_weights_memory, false);
}

void shuffle_features(program_node& node, const std::vector<shuffle_range>& ranges) {
    if (node.is_type<convolution>()) {
        auto& conv = node.as<convolution>();
        shuffle_weights(conv.weights().as<data>(), ranges);
    } else if (node.is_type<fully_connected>()) {
        auto& fc = node.as<fully_connected>();
        shuffle_weights(fc.weights().as<data>(), ranges);
    } else {
        // General case for pass-through layers
        for (auto& user : node.get_users()) {
            shuffle_features(*user, ranges);
        }
    }
}

}  // namespace

void concat_input_order::run(program_impl& p) {
    for (auto node : p.get_processing_order()) {
        // Check that optimization can be performed:
        // 1. Not an output
        // 2. Concatenation along features
        // 3. Currently only fsv16 format on input/output
        // 4. Not already aligned
        // 5. Users can accept shuffled features
        // 6. No fused primitives
        if (!node->is_type<concatenation>() || node->is_output())
            continue;

        auto& concat_node = node->as<concatenation>();
        auto prim = concat_node.get_primitive();

        bool along_f = prim->axis == concatenation::along_f;
        size_t inputs_count = prim->input_size();
        bool no_fusing = !concat_node.has_fused_primitives() && concat_node.get_dependencies().size() == inputs_count;

        auto out_format = concat_node.get_output_layout().format;
        bool correct_format = out_format == format::b_fs_yx_fsv16;
        tensor::value_type alignment = 1;
        if (out_format == format::b_fs_yx_fsv16)
            alignment = 16;

        bool single_format = true;
        std::vector<tensor::value_type> feature_sizes;
        feature_sizes.reserve(inputs_count);
        for (size_t input_idx = 0; input_idx < inputs_count; ++input_idx) {
            auto& dep = concat_node.get_dependency(input_idx);
            auto dep_layout = dep.get_output_layout();
            single_format &= dep_layout.format == out_format;
            feature_sizes.push_back(dep_layout.size.feature[0]);
        }
        // Alignment is not optimal if aligned input follows unaligned one
        bool already_aligned = true;
        for (size_t i = 1; i < feature_sizes.size(); ++i) {
            bool current_aligned = feature_sizes[i] % alignment == 0;
            bool previous_aligned = feature_sizes[i - 1] % alignment == 0;
            already_aligned &= previous_aligned || !current_aligned;
        }
        // Check that we can fuse shuffling to users
        bool can_shuffle_users = true;
        for (auto user : concat_node.get_users()) {
            can_shuffle_users &= can_shuffle_features(*user);
        }

        if (!along_f || !no_fusing || !correct_format || !single_format || already_aligned || !can_shuffle_users)
            continue;

        // Perform the optimization
        // Calculate new input order - first inputs preserving alignment, then rest
        std::vector<size_t> new_order;
        new_order.reserve(inputs_count);
        for (size_t i = 0; i < feature_sizes.size(); ++i) {
            if (feature_sizes[i] % alignment == 0)
                new_order.push_back(i);
        }
        for (size_t i = 0; i < feature_sizes.size(); ++i) {
            if (feature_sizes[i] % alignment != 0)
                new_order.push_back(i);
        }
        // Calculate new ranges
        int32_t current_offset = 0;
        std::vector<shuffle_range> original_ranges;
        original_ranges.reserve(inputs_count);
        for (auto& feature_size : feature_sizes) {
            original_ranges.emplace_back(current_offset, current_offset + feature_size);
            current_offset += feature_size;
        }
        std::vector<shuffle_range> shuffled_ranges;
        shuffled_ranges.reserve(inputs_count);
        for (auto& ord : new_order) {
            shuffled_ranges.push_back(original_ranges[ord]);
        }
        // Change input order
        std::vector<program_node*> new_dependencies = {};
        new_dependencies.reserve(inputs_count);
        for (auto& ord : new_order) {
            new_dependencies.push_back(&concat_node.get_dependency(ord));
        }
        // Update in place with const cast instead of replacing
        auto& dependencies = concat_node.get_dependencies();
        auto& mutable_dependencies = const_cast<std::vector<program_node*>&>(dependencies);
        for (size_t i = 0; i < new_dependencies.size(); ++i) {
            mutable_dependencies[i] = new_dependencies[i];
        }
        std::vector<primitive_id> new_input_ids;
        new_input_ids.reserve(inputs_count);
        for (auto& ord : new_order) {
            new_input_ids.push_back(prim->input[ord]);
        }
        auto mutable_prim = std::const_pointer_cast<concatenation>(prim);
        mutable_prim->input = new_input_ids;
        // Correct users for shuffled features
        for (auto& user : concat_node.get_users()) {
            shuffle_features(*user, shuffled_ranges);
        }
    }
}

