// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/squeeze.hpp>
#include <low_precision/transformer.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/squeeze_function.hpp"

using namespace testing;
using namespace ngraph::pass;

using ngraph::builder::subgraph::SqueezeFunction;

inline std::ostream& operator<<(std::ostream& os, const std::vector<float>& values) {
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

class SqueezeTransformationTestValues {
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
    std::vector<float> axes;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

class SqueezeTransformation : public LayerTransformation, public testing::WithParamInterface<SqueezeTransformationTestValues> {
public:
    void SetUp() override {
        const SqueezeTransformationTestValues testValues = GetParam();

        actualFunction = ngraph::builder::subgraph::SqueezeFunction::getOriginal(
            testValues.inputShape,
            testValues.axes,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::SqueezeTransformation, ngraph::opset1::Squeeze>(testValues.params);

        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::SqueezeFunction::getReference(
            testValues.inputShape,
            testValues.axes,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<SqueezeTransformationTestValues> obj) {
        const SqueezeTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.inputShape << "_" <<
            testValues.axes << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore;

        return result.str();
    }
};

TEST_P(SqueezeTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<SqueezeTransformationTestValues> testValues = {
    {
        ngraph::Shape{ 1, 1, 16, 16 }, // Input shape
        { 1.0f }, // Squeeze axes
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false), // Layer params

        /* Actual */
        {
            ngraph::element::u8, // Precision before dequantization
            /* Dequantization */
            {
                {ngraph::element::f32}, // Convert
                {-0.32f}, // Subtract
                {0.45f} // Multiply
            }
        },
        /* Expected */
        {
            ngraph::element::u8, // Precision before dequantization
            /* Dequantization before */
            {},
            ngraph::element::u8, // Precision after dequantization
            /* Dequantization after */
            {
                {ngraph::element::f32}, // Convert
                {-0.32f}, // Subtract
                {0.45f} // Multiply
            }
        }
    },
    {
        ngraph::Shape{ 1, 1, 1, 1000 }, // Input shape
        {1.0f }, // Squeeze axes
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true), // Layer params

        /* Actual */
        {
            ngraph::element::i8, // Precision before dequantization
            /* Dequantization */
            {
                {ngraph::element::f32}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        },
        /* Expected */
        {
            ngraph::element::i8, // Precision before dequantization
            /* Dequantization before */
            {},
            ngraph::element::i8, // Precision after dequantization
            /* Dequantization after */
            {
                {ngraph::element::f32}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        }
    },
    {
        ngraph::Shape{ 1, 1, 1000 }, // Input shape
        { 1.0f }, // Squeeze axes
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false), // Layer params

        /* Actual */
        {
            ngraph::element::f32, // Precision before dequantization
            /* Dequantization */
            {
                {}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        },
        /* Expected */
        {
            ngraph::element::f32, // Precision before dequantization
            /* Dequantization before */
            {},
            ngraph::element::f32, // Precision after dequantization
            /* Dequantization after */
            {
                {}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        }
    },
    {
        ngraph::Shape{ 1, 1, 1000, 1000 }, // Input shape
        { 0.0f}, // Squeeze axes
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true), // Layer params

        /* Actual */
        {
            ngraph::element::f32, // Precision before dequantization
            /* Dequantization */
            {
                {}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        },
        /* Expected */
        {
            ngraph::element::f32, // Precision before dequantization
            /* Dequantization before */
            {},
            ngraph::element::f32, // Precision after dequantization
            /* Dequantization after */
            {
                {}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    smoke_LPT,
    SqueezeTransformation,
    ::testing::ValuesIn(testValues),
    SqueezeTransformation::getTestCaseName);
