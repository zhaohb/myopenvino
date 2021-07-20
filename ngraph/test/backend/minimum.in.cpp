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

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

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

NGRAPH_TEST(${BACKEND_NAME}, minimum)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Minimum>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 8, -8, 17, -0.5, 0.5, 2, 1};
    std::vector<float> b{1, 2, 4, 8, 0, 0, 1, 1.5};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {1, 2, -8, 8, -.5, 0, 1, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, minimum_int32)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Minimum>(A, B), ParameterVector{A, B});

    std::vector<int32_t> a{1, 8, -8, 17, -5, 67635216, 2, 1};
    std::vector<int32_t> b{1, 2, 4, 8, 0, 18448, 1, 6};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<int32_t>({a, b});
    test_case.add_expected_output<int32_t>(shape, {1, 2, -8, 8, -5, 18448, 1, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, minimum_int64)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Minimum>(A, B), ParameterVector{A, B});

    std::vector<int64_t> a{1, 8, -8, 17, -5, 67635216, 2, 17179887632};
    std::vector<int64_t> b{1, 2, 4, 8, 0, 18448, 1, 280592};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<int64_t>({a, b});
    test_case.add_expected_output<int64_t>(shape, {1, 2, -8, 8, -5, 18448, 1, 280592});
    test_case.run();
}

// TODO Refactor to use TestCase if u16 will be handled correctly
NGRAPH_TEST(${BACKEND_NAME}, minimum_u16)
{
    const Shape shape{3};
    const auto A = make_shared<op::Parameter>(element::u16, shape);
    const auto B = make_shared<op::Parameter>(element::u16, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Minimum>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u16, shape);
    copy_data(a, std::vector<uint16_t>{3, 2, 1});
    auto b = backend->create_tensor(element::u16, shape);
    copy_data(b, std::vector<uint16_t>{1, 4, 4});
    auto result = backend->create_tensor(element::u16, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(test::all_close((std::vector<uint16_t>{1, 2, 1}), read_vector<uint16_t>(result)));
}
