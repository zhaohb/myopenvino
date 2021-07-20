/*
// Copyright (c) 2019-2020 Intel Corporation
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

#include "non_max_suppression_inst.h"
#include "primitive_inst.h"
#include "network_impl.h"
#include "register_gpu.hpp"
#include "cpu_impl_helpers.hpp"

#include <vector>
#include <queue>
#include <algorithm>
#include <utility>
#include <tuple>

namespace cldnn {
namespace {

using namespace cldnn::cpu;

struct result_indices {
    float score;
    int batch_index;
    int class_index;
    int box_index;
};

struct boxInfo {
    float score;
    int idx;
    int suppress_begin_index;
};

std::vector<result_indices> run_nms(
    const vector2D<bounding_box>& boxes,
    const vector3D<float>& scores,
    int num_select_per_class,
    float score_threshold,
    float iou_threshold,
    float soft_nms_sigma,
    bool sort_result_descending
) {
    auto less = [](const boxInfo& l, const boxInfo& r) {
        return l.score < r.score || ((l.score == r.score) && (l.idx > r.idx));
    };
    float scale = 0.0f;
    if (soft_nms_sigma > 0.0f) {
        scale = -0.5f / soft_nms_sigma;
    }
    auto coeff = [&](float iou) {
        const float weight = std::exp(scale * iou * iou);
        return iou <= iou_threshold ? weight : 0.0f;
    };
    std::vector<result_indices> result;

    for (size_t bi = 0; bi < boxes.size(); ++bi) {
        for (size_t ci = 0; ci < scores[bi].size(); ++ci) {
            std::vector<result_indices> fb;

            std::priority_queue<boxInfo, std::vector<boxInfo>, decltype(less)> sorted_boxes(less);
            for (size_t bbi = 0; bbi < boxes[bi].size(); ++bbi) {
                if (scores[bi][ci][bbi] > score_threshold)
                    sorted_boxes.emplace(boxInfo({scores[bi][ci][bbi], static_cast<int>(bbi), 0}));
            }
            fb.reserve(sorted_boxes.size());

            while (static_cast<int>(fb.size()) < num_select_per_class && !sorted_boxes.empty()) {
                boxInfo currBox = sorted_boxes.top();
                float origScore = currBox.score;
                sorted_boxes.pop();

                bool box_is_selected = true;
                for (int idx = static_cast<int>(fb.size()) - 1; idx >= currBox.suppress_begin_index; idx--) {
                    float iou_boxes = iou(boxes[bi][currBox.idx], boxes[bi][fb[idx].box_index]);

                    currBox.score *= coeff(iou_boxes);
                    if (iou_boxes >= iou_threshold) {
                        box_is_selected = false;
                        break;
                    }
                    if (currBox.score <= score_threshold)
                        break;
                }
                currBox.suppress_begin_index = static_cast<int>(fb.size());
                if (box_is_selected) {
                    if (currBox.score == origScore) {
                        fb.push_back(result_indices{ currBox.score, static_cast<int>(bi), static_cast<int>(ci), currBox.idx });
                        continue;
                    }
                    if (currBox.score > score_threshold) {
                        sorted_boxes.push(currBox);
                    }
                }
            }
            std::move(fb.begin(), fb.end(), std::back_inserter(result));
        }
    }

    if (sort_result_descending) {
        std::sort(result.begin(), result.end(),
                [](const result_indices& l, const result_indices& r) {
                    return (l.score > r.score) ||
                           (l.score == r.score && l.batch_index < r.batch_index) ||
                           (l.score == r.score && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
                           (l.score == r.score && l.batch_index == r.batch_index && l.class_index == r.class_index && l.box_index < r.box_index);
                });
    }
    return result;
}

template <typename T>
vector2D<bounding_box> load_boxes_impl(memory_impl& mem, bool center_point) {
    vector2D<bounding_box> result;
    auto lay = mem.get_layout();
    auto batch_size = lay.size.batch[0];
    auto boxes_num = lay.size.feature[0];
    result.resize(batch_size);

    mem_lock<T> boxes_lock(mem);
    auto ptr = boxes_lock.data();

    for (int bi = 0; bi < batch_size; ++bi) {
        result[bi].reserve(boxes_num);
        for (int bxi = 0; bxi < boxes_num; ++bxi) {
            int offset = bi * boxes_num * 4 + bxi * 4;
            if (center_point) {
                result[bi].emplace_back(
                    static_cast<float>(ptr[offset + 0]),
                    static_cast<float>(ptr[offset + 1]),
                    static_cast<float>(ptr[offset + 2]),
                    static_cast<float>(ptr[offset + 3]),
                    bounding_box::center_point_construct_tag());
            } else {
                result[bi].emplace_back(
                    static_cast<float>(ptr[offset + 1]),
                    static_cast<float>(ptr[offset + 0]),
                    static_cast<float>(ptr[offset + 3]),
                    static_cast<float>(ptr[offset + 2]),
                    bounding_box::two_corners_construct_tag());
            }
        }
    }

    return result;
}

vector2D<bounding_box> load_boxes(memory_impl& mem, bool center_point) {
    auto data_type = mem.get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::f16:
        return load_boxes_impl<data_type_to_type<data_types::f16>::type>(mem, center_point);
    case cldnn::data_types::f32:
        return load_boxes_impl<data_type_to_type<data_types::f32>::type>(mem, center_point);
    default:
        throw std::runtime_error("Non max supression - unsupported boxes data type");
    }
}

template <typename T>
vector3D<float> load_scores_impl(memory_impl& mem) {
    auto lay = mem.get_layout();
    auto batch_size = lay.size.batch[0];
    auto classes_num = lay.size.feature[0];
    auto boxes_num = lay.size.spatial[1];

    vector3D<float> result(batch_size, vector2D<float>(classes_num));

    mem_lock<T> lock(mem);
    auto ptr = lock.data();

    for (int bi = 0; bi < batch_size; ++bi) {
        for (int ci = 0; ci < classes_num; ++ci) {
            result[bi][ci].reserve(boxes_num);
            for (int bxi = 0; bxi < boxes_num; ++bxi) {
                auto offset = bi * boxes_num * classes_num + ci * boxes_num + bxi;
                result[bi][ci].emplace_back(static_cast<float>(ptr[offset]));
            }
        }
    }

    return result;
}

vector3D<float> load_scores(memory_impl& mem) {
    auto data_type = mem.get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::f16:
        return load_scores_impl<data_type_to_type<data_types::f16>::type>(mem);
    case cldnn::data_types::f32:
        return load_scores_impl<data_type_to_type<data_types::f32>::type>(mem);
    default:
        throw std::runtime_error("Non max supression - unsupported scores data type");
    }
}

template <typename T, typename MemT>
T load_scalar_impl(memory_impl& mem) {
    mem_lock<MemT> lock(mem);
    auto ptr = lock.data();

    return static_cast<T>(ptr[0]);
}

template <typename T>
T load_scalar(memory_impl& mem) {
    auto data_type = mem.get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::i32:
        return load_scalar_impl<T, data_type_to_type<data_types::i32>::type>(mem);
    case cldnn::data_types::f16:
        return load_scalar_impl<T, data_type_to_type<data_types::f16>::type>(mem);
    case cldnn::data_types::f32:
        return load_scalar_impl<T, data_type_to_type<data_types::f32>::type>(mem);
    default:
        throw std::runtime_error("Non max supression - unsupported data type");
    }
}

template <typename T>
void store_result_impl(memory_impl& mem, const std::vector<result_indices>& result) {
    mem_lock<T> lock(mem);
    auto ptr = lock.data();

    auto output_size = static_cast<size_t>(mem.get_layout().size.batch[0]);
    auto results_size = result.size();

    size_t si = 0;
    for (; si < std::min(output_size, results_size); ++si) {
        auto offset = si * 3;
        ptr[offset + 0] = static_cast<T>(result[si].batch_index);
        ptr[offset + 1] = static_cast<T>(result[si].class_index);
        ptr[offset + 2] = static_cast<T>(result[si].box_index);
    }
    for (; si < output_size; ++si) {
        auto offset = si * 3;
        ptr[offset + 0] = static_cast<T>(-1);
        ptr[offset + 1] = static_cast<T>(-1);
        ptr[offset + 2] = static_cast<T>(-1);
    }
}

void store_result(memory_impl& mem, const std::vector<result_indices>& result) {
    auto data_type = mem.get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::i32:
        store_result_impl<data_type_to_type<data_types::i32>::type>(mem, result);
        break;
    case cldnn::data_types::f16:
        store_result_impl<data_type_to_type<data_types::f16>::type>(mem, result);
        break;
    case cldnn::data_types::f32:
        store_result_impl<data_type_to_type<data_types::f32>::type>(mem, result);
        break;
    default:
        throw std::runtime_error("Non max supression - unsupported output data type");
    }
}

void store_first_output(memory_impl& mem, const std::vector<result_indices>& result) {
    auto data_type = mem.get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::i32:
        store_result_impl<data_type_to_type<data_types::i32>::type>(mem, result);
        break;
    case cldnn::data_types::i64:
        store_result_impl<data_type_to_type<data_types::i32>::type>(mem, result);
        break;
    default:
        throw std::runtime_error("Non max supression - unsupported output data type");
    }
}

template <typename T>
void store_second_output_impl(memory_impl& mem, const std::vector<result_indices>& result) {
    mem_lock<T> lock(mem);
    auto ptr = lock.data();

    auto output_size = static_cast<size_t>(mem.get_layout().size.batch[0]);
    auto results_size = result.size();

    size_t si = 0;
    for (; si < std::min(output_size, results_size); ++si) {
        auto offset = si * 3;
        ptr[offset + 0] = static_cast<T>(result[si].batch_index);
        ptr[offset + 1] = static_cast<T>(result[si].class_index);
        ptr[offset + 2] = static_cast<T>(result[si].score);
    }
    for (; si < output_size; ++si) {
        auto offset = si * 3;
        ptr[offset + 0] = static_cast<T>(-1);
        ptr[offset + 1] = static_cast<T>(-1);
        ptr[offset + 2] = static_cast<T>(-1);
    }
}

void store_second_output(memory_impl& mem, const std::vector<result_indices>& result) {
    auto data_type = mem.get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::f16:
        store_second_output_impl<data_type_to_type<data_types::f16>::type>(mem, result);
        break;
    case cldnn::data_types::f32:
        store_second_output_impl<data_type_to_type<data_types::f32>::type>(mem, result);
        break;
    default:
        throw std::runtime_error("Non max supression - unsupported second output data type");
    }
}

template <typename T>
void store_third_output_impl(memory_impl& mem, const std::vector<result_indices>& result) {
    mem_lock<T> lock(mem);
    auto ptr = lock.data();
    ptr[0] = static_cast<T>(result.size());
}

void store_third_output(memory_impl& mem, const std::vector<result_indices>& result) {
    auto data_type = mem.get_layout().data_type;
    switch (data_type) {
    case cldnn::data_types::i32:
        store_third_output_impl<data_type_to_type<data_types::i32>::type>(mem, result);
        break;
    case cldnn::data_types::i64:
        store_third_output_impl<data_type_to_type<data_types::i32>::type>(mem, result);
        break;
    default:
        throw std::runtime_error("Non max supression - unsupported third output data type");
    }
}

void run(non_max_suppression_inst& instance) {
    auto prim = instance.node.get_primitive();

    auto boxes = load_boxes(instance.input_boxes_mem(), prim->center_point_box);
    auto scores = load_scores(instance.input_scores_mem());

    int num_select_per_class = 0;
    float iou_threshold = 1.f;
    float score_threshold = 0.f;
    float soft_nms_sigma = 0.f;

    if (instance.has_num_select_per_class()) {
        num_select_per_class = load_scalar<int>(instance.num_select_per_class_mem());
    }

    if (instance.has_iou_threshold()) {
        iou_threshold = load_scalar<float>(instance.iou_threshold_mem());
    }

    if (instance.has_score_threshold()) {
        score_threshold = load_scalar<float>(instance.score_threshold_mem());
    }

    if (instance.has_soft_nms_sigma()) {
        soft_nms_sigma = load_scalar<float>(instance.soft_nms_sigma_mem());
    }

    auto result = run_nms(boxes, scores, num_select_per_class, score_threshold, iou_threshold, soft_nms_sigma, prim->sort_result_descending);

    if (instance.has_third_output()) {
        store_third_output(instance.third_output_mem(), result);
    }

    if (instance.has_second_output()) {
        store_second_output(instance.second_output_mem(), result);
        store_first_output(instance.output_memory(), result);
        return;
    }

    store_result(instance.output_memory(), result);
}

struct non_max_suppression_cpu : typed_primitive_impl<non_max_suppression> {
    using parent = typed_primitive_impl<non_max_suppression>;

    non_max_suppression_cpu() : parent(kernel_selector::weights_reorder_params(), "non_max_suppression_cpu") {}

    virtual event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& event,
                                         typed_primitive_inst<non_max_suppression>& instance) {
        for (auto e : event) {
            e->wait();
        }

        auto ev = instance.get_network().get_engine().create_user_event(instance.get_network().get_id(), false);

        run(instance);

        dynamic_cast<cldnn::user_event*>(ev.get())->set();  // set as complete
        return ev;
    }

    static primitive_impl* create(const non_max_suppression_node&) {
        return new non_max_suppression_cpu();
    }
};
}  // namespace

namespace gpu {
namespace detail {

attach_non_max_suppression_gpu::attach_non_max_suppression_gpu() {
    implementation_map<non_max_suppression>::add({
        {std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), non_max_suppression_cpu::create},
        {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), non_max_suppression_cpu::create},
        {std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), non_max_suppression_cpu::create}
    });
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
