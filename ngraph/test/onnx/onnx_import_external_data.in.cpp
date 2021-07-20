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

#include "default_opset.hpp"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/type/element_type.hpp"
#include "onnx_import/onnx.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;
using namespace ngraph::onnx_import;
using namespace ngraph::test;

static std::string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_data)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/external_data/external_data.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_data_from_stream)
{
    std::string path =
        file_util::path_join(SERIALIZED_ZOO, "onnx/external_data/external_data.prototxt");
    std::ifstream stream{path, std::ios::in | std::ios::binary};
    ASSERT_TRUE(stream.is_open());
    const auto function = onnx_import::import_onnx_model(stream, path);

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();

    stream.close();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_data_optinal_fields)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/external_data/external_data_optional_fields.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_data_in_different_paths)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/external_data/external_data_different_paths.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // first input: {3.f}, second: {1.f, 2.f, 5.f} read from external files
    test_case.add_input<float>({2.f, 7.f, 7.f});

    test_case.add_expected_output<float>({2.f, 4.f, 5.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_two_tensors_data_in_the_same_file)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/external_data/external_data_two_tensors_data_in_the_same_file.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // first input: {3, 2, 1}, second: {1, 2, 3} read from external file
    test_case.add_input<int32_t>({2, 3, 1});

    test_case.add_expected_output<int32_t>({3, 3, 3});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_invalid_external_data_exception)
{
    try
    {
        auto function = onnx_import::import_onnx_model(file_util::path_join(
            SERIALIZED_ZOO, "onnx/external_data/external_data_file_not_found.prototxt"));
        FAIL() << "Incorrect path to external data not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_PRED_FORMAT2(
            testing::IsSubstring,
            std::string("not_existed_file.data, offset: 4096, data_lenght: 16, sha1_digest: 0)"),
            error.what());
    }
    catch (...)
    {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_invalid_up_dir_path)
{
    try
    {
        auto function = onnx_import::import_onnx_model(file_util::path_join(
            SERIALIZED_ZOO,
            "onnx/external_data/inner_scope/external_data_file_in_up_dir.prototxt"));
        FAIL() << "Incorrect path to external data not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("tensor.data, offset: 4096, "
                                        "data_lenght: 16, sha1_digest: 0)"),
                            error.what());
    }
    catch (...)
    {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_external_data_sanitize_path)
{
    const auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/external_data/external_data_sanitize_test.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}
