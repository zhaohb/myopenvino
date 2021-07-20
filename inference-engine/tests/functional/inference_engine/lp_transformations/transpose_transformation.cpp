// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/transpose.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/transpose_function.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ngraph::pass;

class TransposeTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ngraph::Shape inputShape;
    std::vector<int> transposeConstValues;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

inline std::ostream& operator<<(std::ostream& os, const std::vector<int>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

class TransposeTransformation : public LayerTransformation, public testing::WithParamInterface<TransposeTransformationTestValues> {
public:
    void SetUp() override {
        const TransposeTransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::TransposeFunction::getOriginal(
            testValues.inputShape,
            testValues.transposeConstValues,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::TransposeTransformation, ngraph::opset1::Transpose>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::TransposeFunction::getReference(
            testValues.inputShape,
            testValues.transposeConstValues,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<TransposeTransformationTestValues> obj) {
        const TransposeTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.inputShape << "_" <<
            testValues.transposeConstValues << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore;
        return result.str();
    }
};

const std::vector<TransposeTransformationTestValues> testValues = {
    // U8: per-tensor quantization
    {
        ngraph::Shape({ 1, 1000, 1, 1}),
        { 0, 1, 3, 2},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                { {128}, ngraph::element::f32, {}, true, 1, ngraph::element::u8, true },
                {0.1f}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                { {128}, ngraph::element::f32, {}, true, 1, ngraph::element::u8, true },
                {0.1f}
            }
        }
    },
    // U8: per-tensor quantization
    {
        ngraph::Shape({ 1, 1000, 1, 1}),
        { 0, 1, 3, 2},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {128}, {0.1f}}
        }
    },
    {
        ngraph::Shape({ 1, 16, 512 }),
        { 0, 2, 1 },
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {128}, {0.1f}}
        }
    },
    // U8: per-channel quantization
    {
        ngraph::Shape({ 1, 3, 1, 1}),
        { 0, 1, 3, 2},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {
                { ngraph::element::f32 },
                {{ 128, 64, 32 }, ngraph::element::f32, { 1, 3, 1, 1 }},
                {{ 0.3f, 0.2f, 0.1f }, ngraph::element::f32, { 1, 3, 1, 1 }}
            }
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {
                { ngraph::element::f32 },
                {{ 128, 64, 32 }, ngraph::element::f32, { 1, 3, 1, 1 }},
                {{ 0.3f, 0.2f, 0.1f }, ngraph::element::f32, { 1, 3, 1, 1 }}
            }
        }
    },
    // empty
    {
        ngraph::Shape({ 1, 1000, 1, 1}),
        { 0, 1, 3, 2},
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {}
        }
    },
};

TEST_P(TransposeTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    TransposeTransformation,
    ::testing::ValuesIn(testValues),
    TransposeTransformation::getTestCaseName);

} // namespace
