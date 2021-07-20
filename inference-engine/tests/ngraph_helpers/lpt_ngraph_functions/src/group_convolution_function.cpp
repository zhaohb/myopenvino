// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/group_convolution_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"

#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "low_precision/common/dequantization_op.hpp"

using namespace ngraph::opset1;
using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<Node> createWeightsOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const size_t inputChannelsCount,
    const size_t outputChannelsCount,
    const size_t groupCount,
    const size_t kernelSize,
    const std::vector<float>& weightsValues,
    const FakeQuantizeOnWeights& fakeQuantizeOnWeights,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationOnWeights) {
    std::shared_ptr<Node> weights;
    if (fakeQuantizeOnWeights.empty() && dequantizationOnWeights.empty()) {
        weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
            weightsValues.size() == 1ul ?
                std::vector<float>(outputChannelsCount * inputChannelsCount, weightsValues[0]) :
                weightsValues);
    } else {
        const size_t inputChannelsPerGroup = inputChannelsCount / groupCount;
        weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ outputChannelsCount, inputChannelsPerGroup, kernelSize, kernelSize },
            weightsValues.size() == 1ul ?
                std::vector<float>(outputChannelsCount * kernelSize * kernelSize * inputChannelsPerGroup, weightsValues[0]) :
                weightsValues);

        if (!fakeQuantizeOnWeights.empty()) {
            weights = ngraph::builder::makeFakeQuantize(
                weights,
                precision,
                fakeQuantizeOnWeights.quantizationLevel,
                { outputChannelsCount, 1, 1, 1 },
                fakeQuantizeOnWeights.inputLowValues,
                fakeQuantizeOnWeights.inputHighValues,
                fakeQuantizeOnWeights.outputLowValues,
                fakeQuantizeOnWeights.outputHighValues);
        }

        if (!dequantizationOnWeights.empty()) {
            weights = ngraph::builder::subgraph::makeDequantization(weights, dequantizationOnWeights);
        }

        weights = std::make_shared<ngraph::opset1::Reshape>(
            weights,
            ngraph::opset1::Constant::create(
                element::i64,
                Shape{ 5 },
                std::vector<size_t>({ groupCount, outputChannelsCount / groupCount, inputChannelsPerGroup, 7, 7 })),
            true);
    }

    return weights;
}

std::shared_ptr<ngraph::Function> GroupConvolutionFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::Shape& outputShape,
    const size_t groupCount,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    std::shared_ptr<ngraph::opset1::Constant> weightsConst,
    const ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    const auto dequantization = makeDequantization(input, dequantizationBefore);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = outputShape[1];
    const size_t kernelSize = 7ul;

    const size_t weightsSize = weightsConst->cast_vector<float>().size();
    if ((weightsSize != 1ul) && (weightsSize != (inputChannelsCount * outputChannelsCount))) {
        throw std::runtime_error("unexpected actual weights values size");
    }

    std::shared_ptr<ngraph::Node> weights = createWeightsOriginal(
        weightsConst->get_element_type(),
        inputShape,
        inputChannelsCount,
        outputChannelsCount,
        groupCount,
        kernelSize,
        weightsConst->cast_vector<float>(),
        fakeQuantizeOnWeights,
        {});

    const auto convolution = std::make_shared<ngraph::opset1::GroupConvolution>(
        dequantization,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    convolution->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "GroupConvolutionTransformation");
}

std::shared_ptr<ngraph::Function> GroupConvolutionFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::Shape& outputShape,
    const size_t groupCount,
    const FakeQuantizeOnData& fakeQuantizeOnData,
    const FakeQuantizeOnWeights& fakeQuantizeOnWeights) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));

    std::shared_ptr<ngraph::Node> fakeQuantizeOnActivations;
    if (fakeQuantizeOnData.empty()) {
        fakeQuantizeOnActivations = nullptr;
    } else {
        fakeQuantizeOnActivations = std::make_shared<ngraph::opset1::FakeQuantize>(
            input,
            std::make_shared<Constant>(precision, Shape{ 1, fakeQuantizeOnData.inputLowValues.size(), 1, 1 }, fakeQuantizeOnData.inputLowValues),
            std::make_shared<Constant>(precision, Shape{ 1, fakeQuantizeOnData.inputHighValues.size(), 1, 1 }, fakeQuantizeOnData.inputHighValues),
            std::make_shared<Constant>(precision, Shape{ 1, fakeQuantizeOnData.outputLowValues.size(), 1, 1 }, fakeQuantizeOnData.outputLowValues),
            std::make_shared<Constant>(precision, Shape{ 1, fakeQuantizeOnData.outputHighValues.size(), 1, 1 }, fakeQuantizeOnData.outputHighValues),
            fakeQuantizeOnData.quantizationLevel);
    }

    // TODO: pass as argument
    //const size_t groupCount = 3ul;
    const size_t outputChannelsCount = outputShape[1];
    const size_t kernelSize = 7ul;
    const size_t inputChannelsCount = inputShape[1];

    std::vector<float> weightsValues = { 1.f };
    std::shared_ptr<ngraph::Node> weights = createWeightsOriginal(
        precision,
        inputShape,
        inputChannelsCount,
        outputChannelsCount,
        groupCount,
        kernelSize,
        weightsValues,
        fakeQuantizeOnWeights,
        {});

    const auto convolution = std::make_shared<ngraph::opset1::GroupConvolution>(
        fakeQuantizeOnActivations == nullptr ? input : fakeQuantizeOnActivations,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "GroupConvolutionTransformation");
}

std::shared_ptr<ngraph::Function> GroupConvolutionFunction::get(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::Shape& outputShape,
    const size_t groupCount,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    std::shared_ptr<ngraph::opset1::Constant> weightsConst,
    const ngraph::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationOnWeights,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const ngraph::element::Type precisionAfterDequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const size_t inputChannelsCount = inputShape[1];

    const size_t outputChannelsCount = outputShape[1];
    const size_t kernelSize = 7ul;
    const size_t inputChannelsInGroup = inputChannelsCount / groupCount;
    const size_t outputChannelsInGroup = outputChannelsCount / groupCount;

    const size_t weightsSize = weightsConst->cast_vector<float>().size();
    if ((weightsSize != 1ul) && (weightsSize != (inputChannelsCount * outputChannelsCount))) {
        throw std::runtime_error("unexpected actual weights values size");
    }

    std::shared_ptr<ngraph::Node> weights;
    if (fakeQuantizeOnWeights.empty() && dequantizationOnWeights.empty()) {
        const ngraph::Shape weightsShape = ngraph::Shape{ groupCount, outputChannelsInGroup, inputChannelsInGroup, kernelSize, kernelSize };
        weights = ngraph::opset1::Constant::create(
            weightsConst->get_element_type(),
            weightsShape,
            weightsSize == 1ul ? std::vector<float>(
                groupCount * outputChannelsInGroup * inputChannelsInGroup * kernelSize * kernelSize,
                weightsConst->cast_vector<float>()[0]) : weightsConst->cast_vector<float>());
    } else {
        weights = createWeightsOriginal(
            weightsConst->get_element_type(),
            inputShape,
            inputChannelsCount,
            outputChannelsCount,
            groupCount,
            kernelSize,
            weightsConst->cast_vector<float>(),
            fakeQuantizeOnWeights,
            dequantizationOnWeights);
    }

    auto convolutionOriginal = ngraph::opset1::GroupConvolution(
        ngraph::op::TemporaryReplaceOutputType(deqBefore, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(weights, element::f32).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    std::shared_ptr<ngraph::opset1::GroupConvolution> convolution = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::GroupConvolution>>(
        convolutionOriginal,
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{});
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(convolution, precisionAfterOperation);

    const auto deqAfter = makeDequantization(convolution, dequantizationAfter);
    deqAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(deqAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
