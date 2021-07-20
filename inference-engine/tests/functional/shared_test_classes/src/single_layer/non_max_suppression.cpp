// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/non_max_suppression.hpp"

namespace LayerTestsDefinitions {

using namespace ngraph;
using namespace InferenceEngine;
using namespace FuncTestUtils::PrecisionUtils;

std::string NmsLayerTest::getTestCaseName(testing::TestParamInfo<NmsParams> obj) {
    InputShapeParams inShapeParams;
    InputPrecisions inPrecisions;
    int32_t maxOutBoxesPerClass;
    float iouThr, scoreThr, softNmsSigma;
    op::v5::NonMaxSuppression::BoxEncodingType boxEncoding;
    bool sortResDescend;
    element::Type outType;
    std::string targetDevice;
    std::tie(inShapeParams, inPrecisions, maxOutBoxesPerClass, iouThr, scoreThr, softNmsSigma, boxEncoding, sortResDescend, outType, targetDevice) = obj.param;

    size_t numBatches, numBoxes, numClasses;
    std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

    Precision paramsPrec, maxBoxPrec, thrPrec;
    std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

    std::ostringstream result;
    result << "numBatches=" << numBatches << "_numBoxes=" << numBoxes << "_numClasses=" << numClasses << "_";
    result << "paramsPrec=" << paramsPrec << "_maxBoxPrec=" << maxBoxPrec << "_thrPrec=" << thrPrec << "_";
    result << "maxOutBoxesPerClass=" << maxOutBoxesPerClass << "_";
    result << "iouThr=" << iouThr << "_scoreThr=" << scoreThr << "_softNmsSigma=" << softNmsSigma << "_";
    result << "boxEncoding=" << boxEncoding << "_sortResDescend=" << sortResDescend << "_outType=" << outType << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void NmsLayerTest::GenerateInputs() {
    size_t it = 0;
    for (const auto &input : cnnNetwork.getInputsInfo()) {
        const auto &info = input.second;
        Blob::Ptr blob;

        if (it == 1) {
            blob = make_blob_with_precision(info->getTensorDesc());
            blob->allocate();
            CommonTestUtils::fill_data_random_float<Precision::FP32>(blob, 1, 0, 1000);
        } else {
            blob = GenerateInput(*info);
        }
        inputs.push_back(blob);
        it++;
    }
}

void NmsLayerTest::Compare(const std::vector<std::vector<std::uint8_t>> &expectedOutputs, const std::vector<Blob::Ptr> &actualOutputs) {
    for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1; outputIndex >=0 ; outputIndex--) {
        const auto& expected = expectedOutputs[outputIndex];
        const auto& actual = actualOutputs[outputIndex];

        const auto &expectedBuffer = expected.data();
        auto memory = as<MemoryBlob>(actual);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        const auto actualBuffer = lockedMemory.as<const uint8_t *>();

        if (outputIndex == 2) {
            if (expected.size() != actual->byteSize())
                throw std::runtime_error("Expected and actual size 3rd output have different size");
        }

        const auto &precision = actual->getTensorDesc().getPrecision();
        size_t size = expected.size() / actual->getTensorDesc().getPrecision().size();
        switch (precision) {
            case Precision::FP32: {
                LayerTestsCommon::Compare(reinterpret_cast<const float *>(expectedBuffer), reinterpret_cast<const float *>(actualBuffer), size, threshold);
                const auto fBuffer = lockedMemory.as<const float *>();
                for (int i = size; i < actual->size(); i++) {
                    ASSERT_TRUE(fBuffer[i] == -1.f) << "Invalid default value: " << fBuffer[i] << " at index: " << i;
                }
                break;
            }
            case Precision::I32: {
                LayerTestsCommon::Compare(reinterpret_cast<const int32_t *>(expectedBuffer), reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
                const auto iBuffer = lockedMemory.as<const int *>();
                for (int i = size; i < actual->size(); i++) {
                    ASSERT_TRUE(iBuffer[i] == -1) << "Invalid default value: " << iBuffer[i] << " at index: " << i;
                }
                break;
            }
            default:
                FAIL() << "Comparator for " << precision << " precision isn't supported";
        }
    }
}

void NmsLayerTest::SetUp() {
    InputShapeParams inShapeParams;
    InputPrecisions inPrecisions;
    size_t maxOutBoxesPerClass;
    float iouThr, scoreThr, softNmsSigma;
    op::v5::NonMaxSuppression::BoxEncodingType boxEncoding;
    bool sortResDescend;
    element::Type outType;
    std::tie(inShapeParams, inPrecisions, maxOutBoxesPerClass, iouThr, scoreThr, softNmsSigma, boxEncoding, sortResDescend, outType,
             targetDevice) = this->GetParam();

    size_t numBatches, numBoxes, numClasses;
    std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

    Precision paramsPrec, maxBoxPrec, thrPrec;
    std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

    numOfSelectedBoxes = std::min(numBoxes, maxOutBoxesPerClass) * numBatches * numClasses;

    const std::vector<size_t> boxesShape{numBatches, numBoxes, 4}, scoresShape{numBatches, numClasses, numBoxes};
    auto ngPrc = convertIE2nGraphPrc(paramsPrec);
    auto params = builder::makeParams(ngPrc, {boxesShape, scoresShape});
    auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(params));

    auto nms = builder::makeNms(paramOuts[0], paramOuts[1], convertIE2nGraphPrc(maxBoxPrec), convertIE2nGraphPrc(thrPrec), maxOutBoxesPerClass, iouThr,
                                scoreThr, softNmsSigma, boxEncoding, sortResDescend, outType);
    auto nms_0_identity = std::make_shared<opset5::Multiply>(nms->output(0), opset5::Constant::create(outType, Shape{1}, {1}));
    auto nms_1_identity = std::make_shared<opset5::Multiply>(nms->output(1), opset5::Constant::create(ngPrc, Shape{1}, {1}));
    auto nms_2_identity = std::make_shared<opset5::Multiply>(nms->output(2), opset5::Constant::create(outType, Shape{1}, {1}));
    function = std::make_shared<Function>(OutputVector{nms_0_identity, nms_1_identity, nms_2_identity}, params, "NMS");
}

}  // namespace LayerTestsDefinitions
