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

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, evaluate_ctc_greedy_decoder_seq_len)
{
    const int N = 1;
    const int T = 3;
    const int C = 3;
    const auto data_shape = Shape{N, T, C};
    const auto seq_len_shape = Shape{N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto seq_len = make_shared<op::Parameter>(element::i32, seq_len_shape);
    auto blanck_index = op::Constant::create<int32_t>(element::i32, Shape{}, {2});
    auto decoder = make_shared<op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, blanck_index, false);
    auto function = make_shared<Function>(decoder, ParameterVector{data, seq_len});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({0.1f, 0.2f, 0.f, 0.4f, 0.3f, 0.f, 0.5f, 0.6f, 0.f});
    test_case.add_input<int32_t>({2});
    test_case.add_expected_output(Shape{N, T}, vector<int32_t>{1, 0, -1});
    test_case.add_expected_output(Shape{N}, vector<int32_t>{2});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_ctc_greedy_decoder_seq_len_merge)
{
    const int N = 1;
    const int T = 3;
    const int C = 3;
    const auto data_shape = Shape{N, T, C};
    const auto seq_len_shape = Shape{N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto seq_len = make_shared<op::Parameter>(element::i32, seq_len_shape);
    auto blanck_index = op::Constant::create<int32_t>(element::i32, Shape{}, {2});
    auto decoder = make_shared<op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, blanck_index, true);
    auto function = make_shared<Function>(decoder, ParameterVector{data, seq_len});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({0.1f, 0.2f, 0.f, 0.4f, 0.3f, 0.f, 0.5f, 0.6f, 0.f});
    test_case.add_input<int32_t>({2});
    test_case.add_expected_output(Shape{N, T}, vector<int32_t>{1, 0, -1});
    test_case.add_expected_output(Shape{N}, vector<int32_t>{2});

    test_case.run();
}

// Test DISABLED as CTCGreedyDecoderSeqLen with fp16 input data for some platforms have fails.
NGRAPH_TEST(${BACKEND_NAME}, DISABLED_evaluate_ctc_greedy_decoder_seq_len_f16)
{
    const int N = 1;
    const int T = 3;
    const int C = 3;
    const auto data_shape = Shape{N, T, C};
    const auto seq_len_shape = Shape{N};

    auto data = make_shared<op::Parameter>(element::f16, data_shape);
    auto seq_len = make_shared<op::Parameter>(element::i32, seq_len_shape);
    auto blanck_index = op::Constant::create<int32_t>(element::i32, Shape{}, {2});
    auto decoder = make_shared<op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, blanck_index, true);
    auto function = make_shared<Function>(decoder, ParameterVector{data, seq_len});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float16>({0.1f, 0.2f, 0.f, 0.4f, 0.3f, 0.f, 0.5f, 0.6f, 0.f});
    test_case.add_input<int32_t>({2});
    test_case.add_expected_output(Shape{N, T}, vector<int32_t>{1, 0, -1});
    test_case.add_expected_output(Shape{N}, vector<int32_t>{2});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_ctc_greedy_decoder_seq_len_multiple_batches)
{
    const int N = 2;
    const int T = 3;
    const int C = 3;
    const auto data_shape = Shape{N, T, C};
    const auto seq_len_shape = Shape{N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto seq_len = make_shared<op::Parameter>(element::i32, seq_len_shape);
    auto blanck_index = op::Constant::create<int32_t>(element::i32, Shape{}, {2});
    auto decoder = make_shared<op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, blanck_index, false);
    auto function = make_shared<Function>(decoder, ParameterVector{data, seq_len});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({0.1f,
                                0.2f,
                                0.f,
                                0.15f,
                                0.25f,
                                0.f,
                                0.4f,
                                0.3f,
                                0.f,
                                0.45f,
                                0.35f,
                                0.f,
                                0.5f,
                                0.6f,
                                0.f,
                                0.55f,
                                0.65f,
                                0.f});

    test_case.add_input<int32_t>({1, 1});

    test_case.add_expected_output(Shape{N, T}, vector<int32_t>{1, -1, -1, 0, -1, -1});
    test_case.add_expected_output(Shape{N}, vector<int32_t>{1, 1});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_ctc_greedy_decoder_seq_len_multiple_batches2)
{
    const int N = 3;
    const int T = 3;
    const int C = 3;
    const auto data_shape = Shape{N, T, C};
    const auto seq_len_shape = Shape{N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto seq_len = make_shared<op::Parameter>(element::i32, seq_len_shape);
    auto blanck_index = op::Constant::create<int32_t>(element::i32, Shape{}, {2});
    auto decoder = make_shared<op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, blanck_index, false);
    auto function = make_shared<Function>(decoder, ParameterVector{data, seq_len});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({0.1f,  0.2f,  0.f, 0.15f, 0.25f, 0.f, 0.4f,  0.3f,  0.f,
                                0.45f, 0.35f, 0.f, 0.5f,  0.6f,  0.f, 0.55f, 0.65f, 0.f,
                                0.1f,  0.2f,  0.f, 0.15f, 0.25f, 0.f, 0.4f,  0.3f,  0.f});

    test_case.add_input<int32_t>({2, 3, 1});

    test_case.add_expected_output(Shape{N, T}, vector<int32_t>{1, 1, -1, 0, 1, 1, 1, -1, -1});
    test_case.add_expected_output(Shape{N}, vector<int32_t>{2, 3, 1});

    test_case.run();
}
