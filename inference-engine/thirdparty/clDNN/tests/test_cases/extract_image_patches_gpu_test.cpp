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
#include <gtest/gtest.h>

#include <api/memory.hpp>
#include <api/input_layout.hpp>
#include <api/extract_image_patches.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/data.hpp>

#include <test_utils/test_utils.h>

using namespace cldnn;
using namespace tests;

TEST(extract_image_patches_gpu, basic) {
    //  Input  : 1x1x10x10
    //  Output : 1x9x2x2

    tensor output_shape = {1, 9, 2, 2};
    const auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 1;
    auto in_rows = 10;
    auto in_cols = 10;
    std::vector<unsigned int> sizes = {3, 3};
    std::vector<unsigned int> strides = {5, 5};
    std::vector<unsigned int> rates = {1, 1};
    std::string auto_pad = "valid";

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<float> inputVals(batch * depth * in_rows * in_cols);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 1;
        return n++;
    });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(extract_image_patches("extract_image_patches", "Input0", sizes, strides, rates, auto_pad, output_shape));

    network network(engine, topology);
    network.set_input_data("Input0", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> answers = {
         1,  6,
        51, 56,

         2,  7,
        52, 57,

         3,  8,
        53, 58,

        11, 16,
        61, 66,

        12, 17,
        62, 67,

        13, 18,
        63, 68,

        21, 26,
        71, 76,

        22, 27,
        72, 77,

        23, 28,
        73, 78
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(extract_image_patches_gpu, basic2) {
    //  Input  : 1x1x10x10
    //  Output : 1x16x1x1

    const auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 1;
    auto in_rows = 10;
    auto in_cols = 10;
    std::vector<unsigned int> sizes = {4, 4};
    std::vector<unsigned int> strides = {8, 8};
    std::vector<unsigned int> rates = {1, 1};
    std::string auto_pad = "valid";
    tensor output_shape = {1, 16, 1, 1};

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<float> inputVals(batch * depth * in_rows * in_cols);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 1;
        return n++;
    });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(extract_image_patches("extract_image_patches", "Input0", sizes, strides, rates, auto_pad, output_shape));

    network network(engine, topology);
    network.set_input_data("Input0", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> answers = {
         1,
         2,
         3,
         4,
        11,
        12,
        13,
        14,
        21,
        22,
        23,
        24,
        31,
        32,
        33,
        34
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(extract_image_patches_gpu, basic3) {
    //  Input  : 1x1x10x10
    //  Output : 1x16x2x2

    const auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 1;
    auto in_rows = 10;
    auto in_cols = 10;
    std::vector<unsigned int> sizes = {4, 4};
    std::vector<unsigned int> strides = {9, 9};
    std::vector<unsigned int> rates = {1, 1};
    std::string auto_pad = "same_upper";
    tensor output_shape = {1, 16, 2, 2};

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<float> inputVals(batch * depth * in_rows * in_cols);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 1;
        return n++;
    });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(extract_image_patches("extract_image_patches", "Input0", sizes, strides, rates, auto_pad, output_shape));

    network network(engine, topology);
    network.set_input_data("Input0", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> answers = {
         0,   0,
         0,  89,

         0,   0,
        81,  90,

         0,   0,
        82,   0,

         0,   0,
        83,   0,

         0,   9,
         0,  99,

         1,  10,
        91, 100,

         2,   0,
        92,   0,

         3,   0,
        93,   0,

         0,  19,
         0,   0,

        11,  20,
         0,   0,

        12,   0,
         0,   0,

        13,   0,
         0,   0,

         0,  29,
         0,   0,

        21,  30,
         0,   0,

        22,   0,
         0,   0,

        23,   0,
         0,   0,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(extract_image_patches_gpu, basic3_same_lower) {
    //  Input  : 1x1x10x10
    //  Output : 1x16x2x2

    const auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 1;
    auto in_rows = 10;
    auto in_cols = 10;
    std::vector<unsigned int> sizes = {4, 4};
    std::vector<unsigned int> strides = {9, 9};
    std::vector<unsigned int> rates = {1, 1};
    std::string auto_pad = "same_lower";
    tensor output_shape = {1, 16, 2, 2};

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<float> inputVals(batch * depth * in_rows * in_cols);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 1;
        return n++;
    });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(extract_image_patches("extract_image_patches", "Input0", sizes, strides, rates, auto_pad, output_shape));

    network network(engine, topology);
    network.set_input_data("Input0", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> answers = {
         0,   0,
         0,  78,

         0,   0,
         0,  79,

         0,   0,
        71,  80,

         0,   0,
        72,   0,

         0,   0,
         0,  88,

         0,   0,
         0,  89,

         0,   0,
        81,  90,

         0,   0,
        82,   0,

         0,   8,
         0,  98,

         0,   9,
         0,  99,

         1,  10,
        91, 100,

         2,   0,
        92,   0,

         0,  18,
         0,   0,

         0,  19,
         0,   0,

        11,  20,
         0,   0,

        12,   0,
         0,   0,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(extract_image_patches_gpu, basic3_enough_space) {
    //  Input  : 1x1x10x10
    //  Output : 1x9x2x2

    const auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 1;
    auto in_rows = 10;
    auto in_cols = 10;
    std::vector<unsigned int> sizes = {3, 3};
    std::vector<unsigned int> strides = {7, 7};
    std::vector<unsigned int> rates = {1, 1};
    std::string auto_pad = "same_upper";
    tensor output_shape = {1, 9, 2, 2};

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<float> inputVals(batch * depth * in_rows * in_cols);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 1;
        return n++;
    });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(extract_image_patches("extract_image_patches", "Input0", sizes, strides, rates, auto_pad, output_shape));

    network network(engine, topology);
    network.set_input_data("Input0", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> answers = {
         1,   8,
        71,  78,

         2,   9,
        72,  79,

         3,  10,
        73,  80,

        11,  18,
        81,  88,

        12,  19,
        82,  89,

        13,  20,
        83,  90,

        21,  28,
        91,  98,

        22,  29,
        92,  99,

        23,  30,
        93, 100,
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(extract_image_patches_gpu, basic4) {
    //  Input  : 1x1x10x10
    //  Output : 1x9x2x2

    const auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 1;
    auto in_rows = 10;
    auto in_cols = 10;
    std::vector<unsigned int> sizes = {3, 3};
    std::vector<unsigned int> strides = {5, 5};
    std::vector<unsigned int> rates = {2, 2};
    std::string auto_pad = "valid";
    tensor output_shape = {1, 9, 2, 2};

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<float> inputVals(batch * depth * in_rows * in_cols);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 1;
        return n++;
    });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(extract_image_patches("extract_image_patches", "Input0", sizes, strides, rates, auto_pad, output_shape));

    network network(engine, topology);
    network.set_input_data("Input0", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> answers = {
         1,   6,
        51,  56,

         3,   8,
        53,  58,

         5,  10,
        55,  60,

        21,  26,
        71,  76,

        23,  28,
        73,  78,

        25,  30,
        75,  80,

        41,  46,
        91,  96,

        43,  48,
        93,  98,

        45,  50,
        95, 100
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}

TEST(extract_image_patches_gpu, basic5) {
    //  Input  : 1x2x5x5
    //  Output : 1x8x2x2

    const auto& engine = get_test_engine();
    auto batch = 1;
    auto depth = 2;
    auto in_rows = 5;
    auto in_cols = 5;
    std::vector<unsigned int> sizes = {2, 2};
    std::vector<unsigned int> strides = {3, 3};
    std::vector<unsigned int> rates = {1, 1};
    std::string auto_pad = "valid";
    tensor output_shape = {1, 8, 2, 2};

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { batch, depth, in_cols, in_rows } });

    std::vector<float> inputVals(batch * depth * in_rows * in_cols);
    std::generate(inputVals.begin(), inputVals.end(), []() {
        static float n = 1;
        return n++;
    });

    set_values(input, inputVals);

    topology topology;
    topology.add(input_layout("Input0", input.get_layout()));
    topology.add(extract_image_patches("extract_image_patches", "Input0", sizes, strides, rates, auto_pad, output_shape));

    network network(engine, topology);
    network.set_input_data("Input0", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "extract_image_patches");

    auto output = outputs.at("extract_image_patches").get_memory();
    auto output_ptr = output.pointer<float>();

    std::vector<float> answers = {
         1,  4,
        16, 19,

        26, 29,
        41, 44,

         2,  5,
        17, 20,

        27, 30,
        42, 45,

         6,  9,
        21, 24,

        31, 34,
        46, 49,

         7, 10,
        22, 25,

        32, 35,
        47, 50
    };

    ASSERT_EQ(answers.size(), output_ptr.size());
    for (size_t i = 0; i < answers.size(); ++i) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i])) << i;
    }
}
