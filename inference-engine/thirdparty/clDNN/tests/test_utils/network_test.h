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
#pragma once

#include <gtest/gtest.h>
#include "test_utils/test_utils.h"

#include "api/data.hpp"
#include "api/engine.hpp"
#include "api/fully_connected.hpp"
#include "api/input_layout.hpp"
#include "api/layout.hpp"
#include "api/memory.hpp"
#include "api/primitive.hpp"
#include "api/tensor.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>


namespace tests {

// =====================================================================================================================
// Typed comparison
template <typename T>
struct typed_comparator {
    static ::testing::AssertionResult compare(const char* lhs_expr, const char* rhs_expr, T ref, T val) {
        return ::testing::internal::EqHelper<false>::Compare(lhs_expr, rhs_expr, ref, val);
    }
};

template <>
struct typed_comparator<float> {
    static ::testing::AssertionResult compare(const char* lhs_expr, const char* rhs_expr, float ref, float val) {
        return ::testing::internal::CmpHelperFloatingPointEQ<float>(lhs_expr, rhs_expr, ref, val);
    }
};

template <>
struct typed_comparator<FLOAT16> {
    static ::testing::AssertionResult compare(const char* lhs_expr, const char* rhs_expr, FLOAT16 ref, FLOAT16 val) {
        double abs_error = std::abs(0.05 * (double)ref);
        return ::testing::internal::DoubleNearPredFormat(lhs_expr, rhs_expr, "5 percent", (double)ref, (double)val, abs_error);
    }
};

#define TYPED_ASSERT_EQ(ref, val)                                                       \
    ASSERT_PRED_FORMAT2(typed_comparator<typename std::remove_reference<decltype(ref)>::type>::compare, ref, val)

#define TYPED_EXPECT_EQ(ref, val)                                                       \
    EXPECT_PRED_FORMAT2(typed_comparator<typename std::remove_reference<decltype(ref)>::type>::compare, ref, val)

// =====================================================================================================================
// Reference tensor
struct reference_tensor {
    virtual void compare(cldnn::memory actual) = 0;
};

template <typename T, size_t N>
struct reference_tensor_typed : reference_tensor {};

template <typename T>
struct reference_tensor_typed<T, 1> : reference_tensor {
    using vector_type = VF<T>;
    reference_tensor_typed(vector_type data) : reference(std::move(data)) {}

    void compare(cldnn::memory actual) override {
        auto ptr = actual.pointer<T>();
        for (size_t bi = 0; bi < reference.size(); ++bi) {
            auto coords = cldnn::tensor(cldnn::batch(bi), cldnn::feature(0), cldnn::spatial(0, 0, 0, 0));
            size_t offset = actual.get_layout().get_linear_offset(coords);
            auto& ref = reference[bi];
            auto& val = ptr[offset];
            TYPED_EXPECT_EQ(ref, val) << " at bi=" << bi;
        }
    }

    void fill_memory(cldnn::memory mem) {
        auto ptr = mem.pointer<T>();
        for (size_t bi = 0; bi < reference.size(); ++bi) {
            auto coords = cldnn::tensor(cldnn::batch(bi), cldnn::feature(0), cldnn::spatial(0, 0, 0, 0));
            size_t offset = mem.get_layout().get_linear_offset(coords);
            ptr[offset] = reference[bi];
        }
    }

    cldnn::tensor get_shape() {
        return cldnn::tensor(cldnn::batch(reference.size()));
    }

    vector_type reference;
};

template <typename T>
struct reference_tensor_typed<T, 2> : reference_tensor {
    using vector_type = VVF<T>;
    reference_tensor_typed(vector_type data) : reference(std::move(data)) {}

    void compare(cldnn::memory actual) override {
        auto ptr = actual.pointer<T>();
        for (size_t bi = 0; bi < reference.size(); ++bi) {
            for (size_t fi = 0; fi < reference[0].size(); ++fi) {
                auto coords = cldnn::tensor(cldnn::batch(bi), cldnn::feature(fi), cldnn::spatial(0, 0, 0, 0));
                size_t offset = actual.get_layout().get_linear_offset(coords);
                auto& ref = reference[bi][fi];
                auto& val = ptr[offset];
                TYPED_EXPECT_EQ(ref, val) << "at bi=" << bi << " fi=" << fi;
            }
        }
    }

    void fill_memory(cldnn::memory mem) {
        auto ptr = mem.pointer<T>();
        for (size_t bi = 0; bi < reference.size(); ++bi) {
            for (size_t fi = 0; fi < reference[0].size(); ++fi) {
                auto coords = cldnn::tensor(cldnn::batch(bi), cldnn::feature(fi), cldnn::spatial(0, 0, 0, 0));
                size_t offset = mem.get_layout().get_linear_offset(coords);
                ptr[offset] = reference[bi][fi];
            }
        }
    }

    cldnn::tensor get_shape() {
        return cldnn::tensor(cldnn::batch(reference.size()), cldnn::feature(reference[0].size()));
    }

    vector_type reference;
};

template <typename T>
struct reference_tensor_typed<T, 4> : reference_tensor {
    using vector_type = VVVVF<T>;
    reference_tensor_typed(vector_type data) : reference(std::move(data)) {}
    void compare(cldnn::memory actual) override {
        auto ptr = actual.pointer<T>();
        for (size_t bi = 0; bi < reference.size(); ++bi) {
            for (size_t fi = 0; fi < reference[0].size(); ++fi) {
                for (size_t yi = 0; yi < reference[0][0].size(); ++yi) {
                    for (size_t xi = 0; xi < reference[0][0][0].size(); ++xi) {
                        auto coords = cldnn::tensor(cldnn::batch(bi), cldnn::feature(fi), cldnn::spatial(xi, yi, 0, 0));
                        size_t offset = actual.get_layout().get_linear_offset(coords);
                        auto& ref = reference[bi][fi][yi][xi];
                        auto& val = ptr[offset];
                        TYPED_EXPECT_EQ(ref, val) << "at bi=" << bi << " fi=" << fi << " yi=" << yi << " xi=" << xi;
                    }
                }
            }
        }
    }

    void fill_memory(cldnn::memory mem) {
        auto ptr = mem.pointer<T>();
        for (size_t bi = 0; bi < reference.size(); ++bi) {
            for (size_t fi = 0; fi < reference[0].size(); ++fi) {
                for (size_t yi = 0; yi < reference[0][0].size(); ++yi) {
                    for (size_t xi = 0; xi < reference[0][0][0].size(); ++xi) {
                        auto coords = cldnn::tensor(cldnn::batch(bi), cldnn::feature(fi), cldnn::spatial(xi, yi, 0, 0));
                        size_t offset = mem.get_layout().get_linear_offset(coords);
                        ptr[offset] = reference[bi][fi][yi][xi];
                    }
                }
            }
        }
    }

    cldnn::tensor get_shape() {
        return cldnn::tensor(cldnn::batch(reference.size()),
                             cldnn::feature(reference[0].size()),
                             cldnn::spatial(reference[0][0][0].size(), reference[0][0].size()));
    }

    vector_type reference;
};

// =====================================================================================================================
// Reference calculations
template <typename InputT>
struct fully_connected_accumulator {
    using type = float;
};

template <>
struct fully_connected_accumulator<uint8_t> {
    using type = int;
};

template <>
struct fully_connected_accumulator<int8_t> {
    using type = int;
};

template <typename OutputT,
          typename InputT,
          typename WeightsT,
          typename BiasT,
          typename AccT = typename fully_connected_accumulator<InputT>::type>
VVF<OutputT> fully_connected_reference_typed(VVVVF<InputT>& input, VVVVF<WeightsT>& weights, VF<BiasT>& bias) {
    size_t input_f = input[0].size();
    size_t input_y = input[0][0].size();
    size_t input_x = input[0][0][0].size();
    size_t output_b = input.size();
    size_t output_f = weights.size();
    auto output = VVF<OutputT>(output_b, VF<OutputT>(output_f));
    for (size_t bi = 0; bi < output_b; ++bi) {
        for (size_t ofi = 0; ofi < output_f; ++ofi) {
            AccT acc = static_cast<AccT>(0);
            for (size_t ifi = 0; ifi < input_f; ++ifi) {
                for (size_t yi = 0; yi < input_y; ++yi) {
                    for (size_t xi = 0; xi < input_x; ++xi) {
                        acc += static_cast<AccT>(input[bi][ifi][yi][xi]) * static_cast<AccT>(weights[ofi][ifi][yi][xi]);
                    }
                }
            }
            output[bi][ofi] = static_cast<OutputT>(acc) + static_cast<OutputT>(bias[ofi]);
        }
    }
    return output;
}

template <typename OutputT,
          typename InputT,
          typename WeightsT,
          typename BiasT,
          typename AccT = typename fully_connected_accumulator<InputT>::type>
VVVVF<OutputT> fully_connected_reference_typed_3d(VVVVF<InputT>& input, VVVVF<WeightsT>& weights, VF<BiasT>& bias) {
    size_t input_f = input[0].size();
    size_t input_y = input[0][0].size();
    size_t input_x = input[0][0][0].size();
    size_t output_b = input.size();        // input is assumed to be bfyx
    size_t output_f = weights.size();    // weights is assumed to be bfyx
    VVVVF<OutputT> output(output_b, VVVF<OutputT>(input_f, VVF<OutputT>(output_f, VF<OutputT>(1))));
    OutputT res;
    for (size_t b = 0; b < output_b; ++b) {
        for (size_t n = 0; n < input_f; ++n) {
            for (size_t f = 0; f < output_f; ++f) {
                res = bias[f];
                for (size_t y = 0; y < input_y; ++y) {
                    for (size_t x = 0; x < input_x; ++x) {
                        res += (OutputT)input[b][n][y][x] * (OutputT)weights[f][y][0][0];
                    }
                }
                output[b][n][f][0] = (OutputT)res;
            }
        }
    }
    return output;
}

// =====================================================================================================================
// Network test
struct reference_node_interface {
    using ptr = std::shared_ptr<reference_node_interface>;

    virtual reference_tensor& get_reference() = 0;
    virtual cldnn::primitive_id get_id() = 0;
};

template <typename T, size_t N>
struct reference_node : reference_node_interface {
    using ptr = std::shared_ptr<reference_node>;

    reference_node(cldnn::primitive_id id, reference_tensor_typed<T, N> data)
        : id(id), reference(std::move(data)) {}

    cldnn::primitive_id id;
    reference_tensor_typed<T, N> reference;

    reference_tensor& get_reference() override { return reference; }
    cldnn::primitive_id get_id() override { return id; }
};

class network_test {
public:
    explicit network_test(cldnn::engine eng) : eng(eng) {}

    template <typename T, size_t N>
    typename reference_node<T, N>::ptr add_input_layout(cldnn::primitive_id id,
                                                        cldnn::format::type fmt,
                                                        typename reference_tensor_typed<T, N>::vector_type data) {
        auto output = reference_tensor_typed<T, N>(std::move(data));
        auto shape = output.get_shape();
        auto lt = cldnn::layout(cldnn::type_to_data_type<T>::value, fmt, shape);
        topo.add(cldnn::input_layout(id, lt));
        auto mem = cldnn::memory::allocate(eng, lt);
        output.fill_memory(mem);
        inputs.emplace(id, mem);
        return add_node(id, std::move(output), {});
    }

    template <typename T, size_t N>
    typename reference_node<T, N>::ptr add_data(cldnn::primitive_id id,
                                                cldnn::format::type fmt,
                                                typename reference_tensor_typed<T, N>::vector_type data) {
        auto output = reference_tensor_typed<T, N>(std::move(data));
        auto shape = output.get_shape();
        auto lt = cldnn::layout(cldnn::type_to_data_type<T>::value, fmt, shape);
        auto mem = cldnn::memory::allocate(eng, lt);
        output.fill_memory(mem);
        topo.add(cldnn::data(id, mem));
        return add_node(id, std::move(output), {});
    }

    template <typename T, typename InputT, size_t InputN, typename WeightsT, typename BiasT>
    typename reference_node<T, 2>::ptr add_fully_connected(cldnn::primitive_id id,
                                                           std::shared_ptr<reference_node<InputT, InputN>> input,
                                                           std::shared_ptr<reference_node<WeightsT, InputN>> weights,
                                                           std::shared_ptr<reference_node<BiasT, 2>> bias,
                                                           cldnn::implementation_desc force = cldnn::implementation_desc{ cldnn::format::any, "" }) {
        topo.add(cldnn::fully_connected(id, input->id, weights->id, bias->id, cldnn::type_to_data_type<T>::value));
        if (force.output_format != cldnn::format::any || force.kernel_name != "")
            forced_impls[id] = force;
        VVF<T> output_data = fully_connected_reference_typed<T>(input->reference.reference,
                                                                weights->reference.reference,
                                                                bias->reference.reference[0]);
        return add_node(id, reference_tensor_typed<T, 2>(output_data), { input, weights, bias });
    }

    template <typename T, typename InputT, size_t InputN, typename WeightsT, typename BiasT>
    typename reference_node<T, 4>::ptr add_fully_connected_3d(cldnn::primitive_id id,
                                                           std::shared_ptr<reference_node<InputT, InputN>> input,
                                                           std::shared_ptr<reference_node<WeightsT, InputN>> weights,
                                                           std::shared_ptr<reference_node<BiasT, 2>> bias,
                                                           cldnn::implementation_desc force = cldnn::implementation_desc{cldnn::format::any, ""},
                                                           size_t input_dim_size = 3) {
        topo.add(cldnn::fully_connected(id, input->id, weights->id, bias->id, cldnn::type_to_data_type<T>::value, cldnn::padding(), input_dim_size));
        if (force.output_format != cldnn::format::any || force.kernel_name != "")
            forced_impls[id] = force;
        VVVVF<T> output_data = fully_connected_reference_typed_3d<T>(input->reference.reference,
                                                                     weights->reference.reference,
                                                                     bias->reference.reference[0]);
        return add_node(id, reference_tensor_typed<T, 4>(output_data), {input, weights, bias});
    }

    cldnn::network build_network(cldnn::build_options opts) {
        opts.set_option(cldnn::build_option::force_implementations(forced_impls));
        auto net = cldnn::network(eng, topo, opts);
        for (auto& in_data : inputs) {
            net.set_input_data(in_data.first, in_data.second);
        }
        return net;
    }

    void run(cldnn::build_options opts) {
        auto net = build_network(opts);
        std::stringstream network_info;
        network_info << "Executed kernels: " << std::endl;
        for (auto info : net.get_primitives_info()) {
            if (info.kernel_id == "")
                continue;
            network_info << "  " << info.original_id << " " << info.kernel_id << std::endl;
        }
        SCOPED_TRACE("\n" + network_info.str());

        auto result = net.execute();
        for (auto out : result) {
            auto out_id = out.first;
            bool tested = false;
            for (auto ref : outputs) {
                if (out_id != ref->get_id())
                    continue;
                SCOPED_TRACE("Compare layer: " + out_id);
                ref->get_reference().compare(out.second.get_memory());
                tested = true;
                break;
            }
            EXPECT_TRUE(tested) << "could not find reference for " << out_id;
        }
    }

protected:
    template <typename T, size_t N>
    typename reference_node<T, N>::ptr add_node(cldnn::primitive_id id,
                                                reference_tensor_typed<T, N> data,
                                                std::vector<reference_node_interface::ptr> inputs) {
        auto node = std::make_shared<reference_node<T, N>>(id, std::move(data));
        for (auto& input : inputs) {
            outputs.erase(input);
        }
        outputs.insert(node);
        return node;
    }

    cldnn::engine eng;
    cldnn::topology topo;
    std::map<cldnn::primitive_id, cldnn::implementation_desc> forced_impls;
    std::map<cldnn::primitive_id, cldnn::memory> inputs;
    std::set<reference_node_interface::ptr> outputs;
};

// =====================================================================================================================
// Random data generation
template <typename T>
struct type_test_ranges {
    static constexpr int min = -1;
    static constexpr int max = 1;
    static constexpr int k = 8;
};

template <>
struct type_test_ranges<uint8_t> {
    static constexpr int min = 0;
    static constexpr int max = 255;
    static constexpr int k = 1;
};

template <>
struct type_test_ranges<int8_t> {
    static constexpr int min = -127;
    static constexpr int max = 127;
    static constexpr int k = 1;
};

template <typename T>
VF<T> generate_smart_random_1d(size_t a) {
    return generate_random_1d<T>(a, type_test_ranges<T>::min, type_test_ranges<T>::max, type_test_ranges<T>::k);
}

template <typename T>
VVF<T> generate_smart_random_2d(size_t a, size_t b) {
    return generate_random_2d<T>(a, b, type_test_ranges<T>::min, type_test_ranges<T>::max, type_test_ranges<T>::k);
}

template <typename T>
VVVVF<T> generate_smart_random_4d(size_t a, size_t b, size_t c, size_t d) {
    return generate_random_4d<T>(a, b, c, d, type_test_ranges<T>::min, type_test_ranges<T>::max, type_test_ranges<T>::k);
}

}  // namespace tests
