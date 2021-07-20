//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_floor)
{
    Shape in_shape{1, 1, 3, 3};
    Shape out_shape{1, 1, 2, 2};
    const Strides& strides{1, 1};
    const Shape& pads_begin{0, 0};
    const Shape& pads_end{0, 0};
    const Shape& kernel{2, 2};
    const bool exclude_pad = true;
    const op::RoundingType rounding_type = op::RoundingType::FLOOR;
    const op::PadType pad_type = op::PadType::NOTSET;

    auto A = make_shared<op::Parameter>(element::f32, in_shape);
    auto avgPool = make_shared<op::v1::AvgPool>(
        A, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_type, pad_type);
    auto f = make_shared<Function>(avgPool, ParameterVector{A});

    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> result{3, 4, 6, 7};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(out_shape, result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_ceil)
{
    Shape in_shape{1, 1, 4, 4};
    Shape out_shape{1, 1, 2, 2};
    const Strides& strides{1, 1};
    const Shape& pads_begin{0, 0};
    const Shape& pads_end{0, 0};
    const Shape& kernel{3, 3};
    const bool exclude_pad = true;
    const op::RoundingType rounding_type = op::RoundingType::CEIL;
    const op::PadType pad_type = op::PadType::NOTSET;

    auto A = make_shared<op::Parameter>(element::f32, in_shape);
    auto avgPool = make_shared<op::v1::AvgPool>(
        A, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_type, pad_type);
    auto f = make_shared<Function>(avgPool, ParameterVector{A});

    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> result{6, 7, 10, 11};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(out_shape, result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_pad)
{
    Shape in_shape{1, 1, 2, 2};
    Shape out_shape{1, 1, 3, 3};
    const Strides& strides{1, 1};
    const Shape& pads_begin{1, 1};
    const Shape& pads_end{1, 1};
    const Shape& kernel{2, 2};
    const bool exclude_pad = true;
    const op::RoundingType rounding_type = op::RoundingType::CEIL;
    const op::PadType pad_type = op::PadType::NOTSET;

    auto A = make_shared<op::Parameter>(element::f32, in_shape);
    auto avgPool = make_shared<op::v1::AvgPool>(
        A, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_type, pad_type);
    auto f = make_shared<Function>(avgPool, ParameterVector{A});

    std::vector<float> a{1, 2, 3, 4};
    std::vector<float> result{1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(out_shape, result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_same_upper)
{
    Shape in_shape{1, 1, 3, 3};
    Shape out_shape{1, 1, 3, 3};
    const Strides& strides{1, 1};
    const Shape& pads_begin{0, 0};
    const Shape& pads_end{0, 0};
    const Shape& kernel{2, 2};
    const bool exclude_pad = false;
    const op::RoundingType rounding_type = op::RoundingType::CEIL;
    const op::PadType pad_type = op::PadType::SAME_UPPER;

    auto A = make_shared<op::Parameter>(element::f32, in_shape);
    auto avgPool = make_shared<op::v1::AvgPool>(
        A, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_type, pad_type);
    auto f = make_shared<Function>(avgPool, ParameterVector{A});

    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> result{3, 4, 2.25, 6, 7, 3.75, 3.75, 4.25, 2.25};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(out_shape, result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_3d)
{
    Shape in_shape{1, 1, 2, 2, 2};
    Shape out_shape{1, 1, 2, 2, 1};
    const Strides& strides{1, 1, 1};
    const Shape& pads_begin{0, 0, 0};
    const Shape& pads_end{0, 0, 0};
    const Shape& kernel{1, 1, 2};
    const bool exclude_pad = true;
    const op::RoundingType rounding_type = op::RoundingType::CEIL;
    const op::PadType pad_type = op::PadType::VALID;

    auto A = make_shared<op::Parameter>(element::f32, in_shape);
    auto avgPool = make_shared<op::v1::AvgPool>(
        A, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_type, pad_type);
    auto f = make_shared<Function>(avgPool, ParameterVector{A});

    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> result{1.5, 3.5, 5.5, 7.5};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(out_shape, result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, avg_pool_2d_same_lower)
{
    Shape in_shape{1, 1, 3, 3};
    Shape out_shape{1, 1, 3, 3};
    const Strides& strides{1, 1};
    const Shape& pads_begin{0, 0};
    const Shape& pads_end{0, 0};
    const Shape& kernel{2, 2};
    const bool exclude_pad = false;
    const op::RoundingType rounding_type = op::RoundingType::CEIL;
    const op::PadType pad_type = op::PadType::SAME_LOWER;

    auto A = make_shared<op::Parameter>(element::f32, in_shape);
    auto avgPool = make_shared<op::v1::AvgPool>(
        A, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_type, pad_type);
    auto f = make_shared<Function>(avgPool, ParameterVector{A});

    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> result{0.25, 0.75, 1.25, 1.25, 3, 4, 2.75, 6, 7};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(out_shape, result);
    test_case.run();
}
