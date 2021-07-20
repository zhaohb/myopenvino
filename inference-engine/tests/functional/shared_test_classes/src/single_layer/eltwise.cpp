// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/single_layer/eltwise.hpp"

namespace LayerTestsDefinitions {

std::string EltwiseLayerTest::getTestCaseName(testing::TestParamInfo<EltwiseTestParams> obj) {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    ngraph::helpers::InputLayerType secondaryInputType;
    CommonTestUtils::OpType opType;
    ngraph::helpers::EltwiseTypes eltwiseOpType;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, eltwiseOpType, secondaryInputType, opType, netPrecision, inPrc, outPrc, inLayout, targetName, additional_config) =
        obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    results << "eltwiseOpType=" << eltwiseOpType << "_";
    results << "secondaryInputType=" << secondaryInputType << "_";
    results << "opType=" << opType << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "inPRC=" << inPrc.name() << "_";
    results << "outPRC=" << outPrc.name() << "_";
    results << "inL=" << inLayout << "_";
    results << "trgDev=" << targetName;
    return results.str();
}

InferenceEngine::Blob::Ptr EltwiseLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    const auto opType = std::get<1>(GetParam());
    switch (opType) {
        case ngraph::helpers::EltwiseTypes::POWER:
        case ngraph::helpers::EltwiseTypes::FLOOR_MOD:
            return info.getPrecision().is_float() ? FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 2, 128):
                                                    FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 4, 2);
        case ngraph::helpers::EltwiseTypes::DIVIDE:
            return info.getPrecision().is_float() ? FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 2, 128):
                                                    FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 100, 101);
        default:
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
    }
}

void EltwiseLayerTest::SetUp() {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    ngraph::helpers::InputLayerType secondaryInputType;
    CommonTestUtils::OpType opType;
    ngraph::helpers::EltwiseTypes eltwiseType;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, eltwiseType, secondaryInputType, opType, netPrecision, inPrc, outPrc, inLayout, targetDevice, additional_config) =
        this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> inputShape1, inputShape2;
    if (inputShapes.size() == 1) {
        inputShape1 = inputShape2 = inputShapes.front();
    } else if (inputShapes.size() == 2) {
        inputShape1 = inputShapes.front();
        inputShape2 = inputShapes.back();
    } else {
        THROW_IE_EXCEPTION << "Incorrect number of input shapes";
    }

    configuration.insert(additional_config.begin(), additional_config.end());
    auto input = ngraph::builder::makeParams(ngPrc, {inputShape1});

    std::vector<size_t> shape_input_secondary;
    switch (opType) {
        case CommonTestUtils::OpType::SCALAR: {
            shape_input_secondary = std::vector<size_t>({1});
            break;
        }
        case CommonTestUtils::OpType::VECTOR:
            shape_input_secondary = inputShape2;
            break;
        default:
            FAIL() << "Unsupported Secondary operation type";
    }

    std::shared_ptr<ngraph::Node> secondaryInput;
    if (eltwiseType == ngraph::helpers::EltwiseTypes::DIVIDE ||
        eltwiseType == ngraph::helpers::EltwiseTypes::FLOOR_MOD ||
        eltwiseType == ngraph::helpers::EltwiseTypes::MOD) {
        std::vector<float> data(ngraph::shape_size(shape_input_secondary));
        data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(shape_input_secondary), 10, 2);
        secondaryInput = ngraph::builder::makeConstant(ngPrc, shape_input_secondary, data);
    } else if (eltwiseType == ngraph::helpers::EltwiseTypes::POWER && secondaryInputType == ngraph::helpers::InputLayerType::CONSTANT) {
        // to avoid floating point overflow on some platforms, let's fill the constant with small numbers.
        secondaryInput = ngraph::builder::makeConstant<float>(ngPrc, shape_input_secondary, {}, true, 3);
    } else {
        secondaryInput = ngraph::builder::makeInputLayer(ngPrc, secondaryInputType, shape_input_secondary);
        if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            input.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
        }
    }

    auto eltwise = ngraph::builder::makeEltwise(input[0], secondaryInput, eltwiseType);
    function = std::make_shared<ngraph::Function>(eltwise, input, "Eltwise");
}
} // namespace LayerTestsDefinitions