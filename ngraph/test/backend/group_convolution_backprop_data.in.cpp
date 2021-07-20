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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

static void GroupConvolutionBackPropDataTest(const std::vector<float>& inputs,
                                             const Shape inputs_shape,
                                             const std::vector<float>& filters,
                                             const Shape filter_shape,
                                             const std::vector<float>& outputs,
                                             const Shape outputs_shape,
                                             const Strides& strides,
                                             const CoordinateDiff& padding,
                                             const Strides& dilations)
{
    CoordinateDiff pads_begin{padding};
    CoordinateDiff pads_end{padding};
    const op::PadType auto_pad{op::PadType::EXPLICIT};

    auto inputs_param = make_shared<op::Parameter>(element::f32, inputs_shape);
    auto filter_param = make_shared<op::Parameter>(element::f32, filter_shape);
    auto conv_backprop_data = make_shared<op::v1::GroupConvolutionBackpropData>(
        inputs_param, filter_param, strides, pads_begin, pads_end, dilations, auto_pad);
    auto f = make_shared<Function>(conv_backprop_data, ParameterVector{inputs_param, filter_param});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(inputs_shape, inputs);
    test_case.add_input<float>(filter_shape, filters);
    test_case.add_expected_output<float>(outputs_shape, outputs);
    test_case.run();
}

// --------------------- 1D group convolution ------------------------------------------
// clang-format off
NGRAPH_TEST(${BACKEND_NAME}, group_convolution_backprop_data_1D_1group_1batch_1channel)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{1, 1, 4};
    const std::vector<float> inputs{1.0f, 3.0f, 3.0f, 0.0f};

    const Shape filter_shape{1, 1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 6};
    const std::vector<float> outputs{2.0f, 6.0f, 7.0f, 3.0f, 3.0f, 0.0f};

    GroupConvolutionBackPropDataTest(inputs,
                                     inputs_shape,
                                     filters,
                                     filter_shape,
                                     outputs,
                                     outputs_shape,
                                     strides,
                                     padding,
                                     dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, group_convolution_backprop_data_1D_2group_1batch_2channel)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{1, 2, 4};
    const std::vector<float> inputs{1.0f, 3.0f, 3.0f, 0.0f,
                                    1.0f, 2.0f, 1.0f, 3.0f};

    const Shape filter_shape{2, 1, 1, 3};
    const std::vector<float> filters{1.0f, 0.0f, 3.0f, 3.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 2, 6};
    const std::vector<float> outputs{
        1.0f, 3.0f, 6.0f, 9.0f, 9.0f, 0.0f, 3.0f, 6.0f, 4.0f, 11.0f, 1.0f, 3.0f};

    GroupConvolutionBackPropDataTest(inputs,
                                     inputs_shape,
                                     filters,
                                     filter_shape,
                                     outputs,
                                     outputs_shape,
                                     strides,
                                     padding,
                                     dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, group_convolution_backprop_data_1D_2group_1batch_2_filters_2channel)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{1, 4, 4};
    const std::vector<float> inputs{1.0f, 3.0f, 3.0f, 0.0f,
                                    1.0f, 2.0f, -1.0f, -3.0f,
                                    -3.0f, 0.0f, 1.0f, 2.0f,
                                    0.0f, -2.0f, 3.0f, -1.0f};

    const Shape filter_shape{2, 2, 1, 3};
    const std::vector<float> filters{
        1.0f, 0.0f, 3.0f, 3.0f, 0.0f, 1.0f, -3.0f, 0.0f, 1.0f, 3.0f, 2.0f, -1.0f};

    const Shape outputs_shape{1, 2, 6};
    const std::vector<float> outputs{
        4.0f, 9.0f, 4.0f, 2.0f, 8.0f, -3.0f, 9.0f, -6.0f, -1.0f, -1.0f, -4.0f, 3.0f};

    GroupConvolutionBackPropDataTest(inputs,
                                     inputs_shape,
                                     filters,
                                     filter_shape,
                                     outputs,
                                     outputs_shape,
                                     strides,
                                     padding,
                                     dilations);
}

NGRAPH_TEST(${BACKEND_NAME}, group_convolution_backprop_data_1D_2group_2batch_2channel)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{2, 2, 4};
    const std::vector<float> inputs{// -- batch 1 --
                                    1.0f, 3.0f, 0.0f, 1.0f,
                                    1.0f, 3.0f, 0.0f, 2.0f,
                                    // -- batch 2 --
                                    1.0f, 3.0f, 0.0f, 1.0f,
                                    1.0f, 3.0f, 0.0f, 2.0f};

    const Shape filter_shape{2, 1, 1, 3};
    const std::vector<float> filters{1.0f, 0.0f, 3.0f, 3.0f, 0.0f, 1.0f};

    const Shape outputs_shape{2, 2, 6};
    const std::vector<float> outputs{1.0f, 3.0f, 3.0f, 10.0f, 0.0f, 3.0f,
                                     3.0f, 9.0f, 1.0f,  9.0f, 0.0f, 2.0f,
                                     1.0f, 3.0f, 3.0f, 10.0f, 0.0f, 3.0f,
                                     3.0f, 9.0f, 1.0f,  9.0f, 0.0f, 2.0f};

    GroupConvolutionBackPropDataTest(inputs,
                                     inputs_shape,
                                     filters,
                                     filter_shape,
                                     outputs,
                                     outputs_shape,
                                     strides,
                                     padding,
                                     dilations);
}
// clang-format on

// --------------------- 2D group convolution ------------------------------------------
NGRAPH_TEST(${BACKEND_NAME}, group_convolution_backprop_data_2D)
{
    const CoordinateDiff output_padding{1, 1};
    const CoordinateDiff pads_begin{1, 1};
    const CoordinateDiff pads_end{1, 1};
    Strides strides{2, 2};
    Strides dilations{1, 1};
    const op::PadType auto_pad{op::PadType::EXPLICIT};

    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 1, 3, 3});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1, 3, 3});

    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, strides, pads_begin, pads_end, dilations, auto_pad, output_padding);

    auto function = make_shared<Function>(NodeVector{gcbd}, ParameterVector{data, filters});
    auto test_case = test::TestCase<TestEngine>(function);

    // X
    test_case.add_input<float>(vector<float>{0.16857791f,
                                             -0.15161794f,
                                             0.08540368f,
                                             0.1820628f,
                                             -0.21746576f,
                                             0.08245695f,
                                             0.1431433f,
                                             -0.43156421f,
                                             0.30591947f});
    // W
    test_case.add_input<float>({-0.06230065f,
                                0.37932432f,
                                -0.25388849f,
                                0.33878803f,
                                0.43709868f,
                                -0.22477469f,
                                0.04118127f,
                                -0.44696793f,
                                0.06373066f});
    test_case.add_expected_output(
        Shape{1, 1, 6, 6},
        vector<float>{
            0.07368518f,  -0.08925839f, -0.06627201f, 0.06301362f,  0.03732984f,  -0.01919658f,
            -0.00628807f, -0.02817563f, -0.01472169f, 0.04392925f,  -0.00689478f, -0.01549204f,
            0.07957941f,  -0.11459791f, -0.09505399f, 0.07681622f,  0.03604182f,  -0.01853423f,
            -0.0270785f,  -0.00680824f, -0.06650258f, 0.08004665f,  0.07918708f,  -0.0724144f,
            0.06256775f,  -0.17838378f, -0.18863615f, 0.20064656f,  0.133717f,    -0.06876295f,
            -0.06398046f, -0.00864975f, 0.19289537f,  -0.01490572f, -0.13673618f, 0.01949645f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, group_convolution_backprop_data_2D_output_shape)
{
    Strides strides{1, 1};
    Strides dilations{1, 1};

    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1, 10});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1, 1, 5});
    auto output_shape = op::Constant::create(element::i64, Shape{2}, {1, 14});

    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, output_shape, strides, dilations, op::PadType::SAME_UPPER);

    auto function = make_shared<Function>(NodeVector{gcbd}, ParameterVector{data, filters});
    auto test_case = test::TestCase<TestEngine>(function);

    // X
    test_case.add_input<float>(
        vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    // W
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});
    test_case.add_expected_output(Shape{1, 1, 1, 14},
                                  vector<float>{0.0f,
                                                1.0f,
                                                4.0f,
                                                10.0f,
                                                18.0f,
                                                27.0f,
                                                36.0f,
                                                45.0f,
                                                54.0f,
                                                63.0f,
                                                62.0f,
                                                50.0f,
                                                26.0f,
                                                9.0f});
    test_case.run();
}