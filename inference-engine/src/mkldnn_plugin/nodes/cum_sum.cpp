// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "list.hpp"
#include "base.hpp"

#include <string>
#include <vector>
#include "ie_parallel.hpp"
#include "ie_precision.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class CumSumImpl: public ExtLayerBase {
    enum { CUM_SUM_DATA, AXIS, numOfInputs };
    bool exclusive;
    bool reverse;
    size_t numOfDims;
    size_t axis = 0;
    std::vector<size_t> shape;

public:
    explicit CumSumImpl(const CNNLayer* layer) {
        try {
            layerName = layer->name;
            if ((layer->insData.size() != numOfInputs && layer->insData.size() != (numOfInputs - 1)) || layer->outData.size() != 1)
                THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "' has incorrect number of input/output edges!";

            const auto &dataTensor = layer->insData[CUM_SUM_DATA].lock()->getTensorDesc();
            const auto &dataShape = dataTensor.getDims();
            if (dataShape.size() < 1) {
                THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "' doesn't support 'data' input tensor with rank: " << dataShape.size();
            }
            numOfDims = dataShape.size();

            exclusive = layer->GetParamAsBool("exclusive", false);
            reverse = layer->GetParamAsBool("reverse", false);

            const auto& dataPrecision = dataTensor.getPrecision();
            if (dataPrecision != Precision::I8 && dataPrecision != Precision::U8 && dataPrecision != Precision::I16 && dataPrecision != Precision::I32 &&
                dataPrecision != Precision::FP32 && dataPrecision != Precision::I64 && dataPrecision != Precision::U64 && dataPrecision != Precision::BF16)
                THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "' has unsupported 'data' input precision: " << dataPrecision.name();

            if (layer->insData.size() == numOfInputs) {
                const auto& axisTensor = layer->insData[AXIS].lock()->getTensorDesc();
                const auto& axisTensorPrec = layer->insData[AXIS].lock()->getTensorDesc().getPrecision();
                if (axisTensorPrec != Precision::I32 && axisTensorPrec != Precision::I64)
                    THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "' has unsupported 'axis' input precision: " << axisTensorPrec.name();

                const auto axisTensorRank = axisTensor.getDims().size();
                if (axisTensorRank != 0)
                    THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "' doesn't support 'axis' input tensor with rank: " << axisTensorRank;
            }

            if (dataShape != layer->outData[0]->getTensorDesc().getDims())
                THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "' has different 'data' input and output dimensions";

            shape = dataShape;

            LayerConfig config;
            for (size_t i = 0; i < layer->insData.size(); i++) {
                DataConfig inConfig;
                inConfig.inPlace = -1;
                inConfig.constant = false;

                Precision inPrecision = i == 1 ? Precision(Precision::I32) : layer->insData[i].lock()->getTensorDesc().getPrecision();
                if (inPrecision == Precision::BF16)
                    inPrecision = Precision::FP32;
                const SizeVector& inDims = layer->insData[i].lock()->getTensorDesc().getDims();
                inConfig.desc = TensorDesc(inPrecision, inDims, InferenceEngine::TensorDesc::getLayoutByDims(inDims));

                config.inConfs.push_back(inConfig);
            }
            DataConfig outConfig;
            outConfig.inPlace = -1;
            outConfig.constant = false;
            Precision outPrecision = layer->insData[CUM_SUM_DATA].lock()->getTensorDesc().getPrecision();
            if (outPrecision == Precision::BF16)
                outPrecision = Precision::FP32;
            const SizeVector& outDims = layer->outData[0]->getTensorDesc().getDims();
            outConfig.desc = TensorDesc(outPrecision, outDims, InferenceEngine::TensorDesc::getLayoutByDims(outDims));

            config.outConfs.push_back(outConfig);

            config.dynBatchSupport = false;
            confs.push_back(config);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        if (inputs.size() == numOfInputs)
            axis = getAxis(inputs[AXIS], inputs[CUM_SUM_DATA]);

        const auto &dataPrecision = inputs[CUM_SUM_DATA]->getTensorDesc().getPrecision();
        switch (dataPrecision) {
            case Precision::I8   : { execImpl<int8_t>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            case Precision::U8   : { execImpl<uint8_t>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            case Precision::I16  : { execImpl<int16_t>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            case Precision::I32  : { execImpl<int32_t>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            case Precision::FP32 : { execImpl<float>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            case Precision::I64  : { execImpl<int64_t>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            case Precision::U64  : { execImpl<uint64_t>(inputs[CUM_SUM_DATA], outputs[0]); break; }
            default : {
                if (resp) {
                    std::string errorMsg = "CumSum layer with name '" + layerName + "' has unsupported 'data' input precision: " + dataPrecision.name();
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
            }
        }
        return OK;
    }

private:
    template <typename dataType>
    void execImpl(const Blob::CPtr& _input, const Blob::Ptr& _output) {
        const auto *input = _input->cbuffer().as<const dataType *>() + _input->getTensorDesc().getBlockingDesc().getOffsetPadding();
        auto *output = _output->buffer().as<dataType *>() + _output->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const std::vector<size_t> strides = _input->getTensorDesc().getBlockingDesc().getStrides();

        if (reverse) {
            if (exclusive) {
                cumSum<true, true, dataType>(input, output, strides);
            } else {
                cumSum<true, false, dataType>(input, output, strides);
            }
        } else {
            if (exclusive) {
                cumSum<false, true, dataType>(input, output, strides);
            } else {
                cumSum<false, false, dataType>(input, output, strides);
            }
        }
    }

    template <bool reverse, bool exclusive, typename dataType>
    void cumSum(const dataType *input, dataType *output, const std::vector<size_t> &strides) {
        SizeVector iterationRange(numOfDims - 1);
        size_t j = 0;
        for (size_t i = 0; i < shape.size(); i++) {
            if (i == axis)
                continue;
            iterationRange[j++] = shape[i];
        }
        size_t work_amount_dst = std::accumulate(iterationRange.begin(), iterationRange.end(), 1, std::multiplies<size_t>());
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            SizeVector counters(numOfDims - 1, 0);
            splitter(work_amount_dst, nthr, ithr, start, end);

            parallelItInit(start, counters, iterationRange);

            for (size_t iwork = start; iwork < end; ++iwork) {
                std::vector<size_t> forStartOffset(numOfDims);
                forStartOffset[axis] = 0;
                for (int64_t offsetIdx = 0, countersIdx = 0; offsetIdx < numOfDims; ++offsetIdx) {
                    if (offsetIdx == axis) {
                        continue;
                    }
                    forStartOffset[offsetIdx] = counters[countersIdx++];
                }

                size_t startOffset = getStartOffset(forStartOffset, strides);

                const dataType *inputStart = input + startOffset;
                dataType *outputStart = output + startOffset;

                size_t offset = strides[axis];
                if (reverse) {
                    if (exclusive) {
                        outputStart[offset*(shape[axis] - 1)] = 0;
                        for (int64_t i = shape[axis] - 2; i >= 0; i--) {
                            outputStart[i*offset] = inputStart[(i+1)*offset] + outputStart[(i+1)*offset];
                        }
                    } else {
                        outputStart[offset*(shape[axis] - 1)] = inputStart[offset * (shape[axis] - 1)];
                        for (int64_t i = shape[axis] - 2; i >= 0; i--) {
                            outputStart[i*offset] = inputStart[i*offset] + outputStart[(i+1)*offset];
                        }
                    }
                } else {
                    if (exclusive) {
                        outputStart[0] = 0;
                        for (size_t i = 1; i < shape[axis]; i++) {
                            outputStart[i*offset] = inputStart[(i-1)*offset] + outputStart[(i-1)*offset];
                        }
                    } else {
                        outputStart[0] = inputStart[0];
                        for (size_t i = 1; i < shape[axis]; i++) {
                            outputStart[i*offset] = inputStart[i*offset] + outputStart[(i-1)*offset];
                        }
                    }
                }

                parallelItStep(counters, iterationRange);
            }
        });
    }

    void parallelItInit(size_t start, std::vector<size_t>& counters, const std::vector<size_t>& iterationRange) {
        auto itCounter = counters.rbegin();
        auto itWork = iterationRange.rbegin();
        while (itCounter != counters.rend() && itWork != iterationRange.rend()) {
            *itCounter = start % *itWork;
            start /= *itWork;
            ++itCounter;
            ++itWork;
        }
    }

    inline void parallelItStep(std::vector<size_t>& counters, const std::vector<size_t>& iterationRange) {
        auto itCounter = counters.rbegin();
        auto itWork = iterationRange.rbegin();

        while (itCounter != counters.rend() && itWork != iterationRange.rend()) {
            *itCounter = (*itCounter + 1) % *itWork;
            if (*itCounter != 0) {
                break;
            }
            ++itCounter;
            ++itWork;
        }
    }

    inline size_t getStartOffset(const std::vector<size_t> &forStartOffset, const std::vector<size_t>& strides) const {
        size_t startOffset = 0;
        for (size_t idx = 0; idx < forStartOffset.size(); ++idx) {
            startOffset += forStartOffset[idx] * strides[idx];
        }
        return startOffset;
    }

    size_t getAxis(const Blob::CPtr& _axis, const Blob::CPtr& _data) const {
        const auto& axisPrecision = _axis->getTensorDesc().getPrecision();
        const int64_t dataShapeSize = static_cast<int64_t>(_data->getTensorDesc().getDims().size());
        int64_t axisValueFromBlob;
        switch (axisPrecision) {
            case Precision::I32 : {
                const auto *axisPtr = _axis->cbuffer().as<const int32_t *>();
                axisValueFromBlob = static_cast<int64_t>(axisPtr[0]);
                break;
            }
            case Precision::I64 : {
                const auto *axisPtr = _axis->cbuffer().as<const int64_t *>();
                axisValueFromBlob = axisPtr[0];
                break;
            }
            default : {
                THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "'  doesn't support 'axis' input with precision: " << axisPrecision.name();
            }
        }
        if (axisValueFromBlob < -dataShapeSize || axisValueFromBlob > dataShapeSize - 1)
            THROW_IE_EXCEPTION << "CumSum layer with name '" << layerName << "'  has axis with a value out of range: " << axisValueFromBlob;
        return axisValueFromBlob >= 0 ? axisValueFromBlob : (axisValueFromBlob + dataShapeSize);
    }

private:
    std::string layerName;
};

REG_FACTORY_FOR(CumSumImpl, CumSum);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine