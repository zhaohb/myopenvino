// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <utility>
#include <algorithm>
#include <memory>
#include <string>
#include <map>

#include <ie_blob.h>
#include <description_buffer.hpp>
#include <debug.h>
#include <ie_layouts.h>
#include <threading/ie_executor_manager.hpp>
#include <blob_transform.hpp>
#include <ie_parallel.hpp>
#include <ie_memcpy.h>
#include <precision_utils.h>

#include "template/template_config.hpp"
#include "template_infer_request.hpp"
#include "template_executable_network.hpp"
#include "template_plugin.hpp"
#include "template_itt.hpp"

using namespace TemplatePlugin;
using namespace InferenceEngine;

using Time = std::chrono::high_resolution_clock;

// ! [infer_request:ctor]
TemplateInferRequest::TemplateInferRequest(const InferenceEngine::InputsDataMap&                     networkInputs,
                                           const InferenceEngine::OutputsDataMap&                    networkOutputs,
                                           const std::shared_ptr<TemplatePlugin::ExecutableNetwork>& executableNetwork) :
    InferRequestInternal(networkInputs, networkOutputs),
    _executableNetwork(executableNetwork) {
    // TODO: allocate infer request device and host buffers if needed, fill actual list of profiling tasks

    auto requestID = std::to_string(_executableNetwork->_requestId.fetch_add(1));

    std::string name = _executableNetwork->_function->get_friendly_name() + "_Req" + requestID;
    _profilingTask = {
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_Preprocess"),
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_Postprocess"),
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_StartPipline"),
        openvino::itt::handle("Template" + std::to_string(_executableNetwork->_cfg.deviceId) + "_" + name + "_WaitPipline"),
    };

    _executable = _executableNetwork->_plugin->_backend->compile(_executableNetwork->_function);
    _parameters = _executableNetwork->_function->get_parameters();
    _results = _executableNetwork->_function->get_results();

    allocateDeviceBuffers();
    allocateBlobs();
}
// ! [infer_request:ctor]

// ! [infer_request:dtor]
TemplateInferRequest::~TemplateInferRequest() {
    _executableNetwork->_requestId--;
}
// ! [infer_request:dtor]

void TemplateInferRequest::allocateDeviceBuffers() {
    // Allocate plugin backend specific memory handles
    _inputTensors.resize(_networkInputs.size());
    _outputTensors.resize(_networkOutputs.size());
}

template<typename BlobDataMap, typename GetNetworkPrecisionF>
static void AllocateImpl(const BlobDataMap& blobDataMap,
                         BlobMap& blobMap,
                         BlobMap& networkBlobMap,
                         GetNetworkPrecisionF&& GetNetworkPrecision) {
    for (auto&& blobData : blobDataMap) {
        auto& dims = blobData.second->getTensorDesc().getDims();
        auto& precision = blobData.second->getTensorDesc().getPrecision();
        auto layout = blobData.second->getTensorDesc().getLayout();
        Blob::Ptr blob;
        switch (precision) {
        case Precision::U8: {
            blob = InferenceEngine::make_shared_blob<std::uint8_t>({precision, dims, layout});
        } break;
        case Precision::FP32 : {
            blob = InferenceEngine::make_shared_blob<float>({precision, dims, layout});
        } break;
        default: THROW_IE_EXCEPTION << "Template Plugin: Unsupported Input/Output Presision";
        }
        blob->allocate();
        blobMap[blobData.first] = blob;

        auto networkPresion = GetNetworkPrecision(blobData.first);
        Blob::Ptr networkBlob;
        switch (networkPresion) {
        case ngraph::element::Type_t::f32 : {
            if (precision == Precision::FP32) {
                networkBlob = blob;
            } else {
                networkBlob = InferenceEngine::make_shared_blob<float>({Precision::FP32, dims, layout});
            }
        } break;
        default: THROW_IE_EXCEPTION << "Template Plugin: Unsupported network Input/Output Presision";
        }
        if (blob != networkBlob) {
            networkBlob->allocate();
        }
        networkBlobMap[blobData.first] = networkBlob;
    }
}

void TemplateInferRequest::allocateBlobs() {
    auto&& parameters = _executableNetwork->_function->get_parameters();
    AllocateImpl(_networkInputs, _inputs, _deviceInputs, [&] (const std::string& blobName) {
        return parameters.at(_executableNetwork->_inputIndex.at(blobName))->get_element_type();
    });
    auto&& results = _executableNetwork->_function->get_results();
    AllocateImpl(_networkOutputs, _outputs, _networkOutputBlobs, [&] (const std::string& blobName) {
        return results.at(_executableNetwork->_outputIndex.at(blobName))->get_element_type();
    });
}

// ! [infer_request:infer_impl]
void TemplateInferRequest::InferImpl() {
    // TODO: fill with actual list of pipeline stages, which are executed synchronously for sync infer requests
    inferPreprocess();
    startPipeline();
    waitPipeline();  // does nothing in current implementation
    inferPostprocess();
}
// ! [infer_request:infer_impl]

template<typename SrcT, typename DstT>
static void blobCopy(const Blob::Ptr& src, const Blob::Ptr& dst) {
    std::copy_n(InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const SrcT*>(),
                src->size(),
                InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<DstT*>());
}

static void blobCopy(const Blob::Ptr& src, const Blob::Ptr& dst) {
    switch (src->getTensorDesc().getPrecision()) {
        case Precision::U8 : {
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::U8 : break;
                case Precision::FP32 : {
                    blobCopy<std::uint8_t, float>(src, dst);
                } break;
                default : {
                    THROW_IE_EXCEPTION << "Unsupported precision conversion from "
                        << src->getTensorDesc().getPrecision() <<" to " << dst->getTensorDesc().getPrecision();
                }
            }
        } break;
        case Precision::FP32 : {
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::FP32 : break;
                case Precision::U8 : {
                    blobCopy<float, std::uint8_t>(src, dst);
                } break;
                default : {
                    THROW_IE_EXCEPTION << "Unsupported precision conversion from "
                        << src->getTensorDesc().getPrecision() <<" to " << dst->getTensorDesc().getPrecision();
                }
            }
        } break;
        default : {
            THROW_IE_EXCEPTION << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision();
        }
    }
}

// ! [infer_request:infer_preprocess]
void TemplateInferRequest::inferPreprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[Preprocess]);
    auto start = Time::now();
    // NOTE: After InferRequestInternal::execDataPreprocessing call
    //       input can points to other memory region than it was allocated in constructor.
    InferRequestInternal::execDataPreprocessing(_deviceInputs);
    for (auto&& networkInput : _deviceInputs) {
        auto index = _executableNetwork->_inputIndex[networkInput.first];
        const auto& parameter = _parameters[index];
        const auto& parameterShape = parameter->get_shape();
        const auto& parameterType = parameter->get_element_type();
        _inputTensors[index] = _executableNetwork->_plugin->_backend->create_tensor(parameterType, parameterShape,
            InferenceEngine::as<InferenceEngine::MemoryBlob>(networkInput.second)->rmap().as<void*>());
    }
    for (auto&& output : _outputs) {
        auto outputBlob = output.second;
        auto networkOutput = _networkOutputBlobs[output.first];
        auto index = _executableNetwork->_outputIndex[output.first];
        if (outputBlob->getTensorDesc().getPrecision() == networkOutput->getTensorDesc().getPrecision()) {
            networkOutput = outputBlob;
        }
        const auto& result = _results[index];
        const auto& resultShape = result->get_shape();
        const auto& resultType = result->get_element_type();
        _outputTensors[index] = _executableNetwork->_plugin->_backend->create_tensor(resultType, resultShape,
            InferenceEngine::as<InferenceEngine::MemoryBlob>(networkOutput)->wmap().as<void*>());
    }
    _durations[Preprocess] = Time::now() - start;
}
// ! [infer_request:infer_preprocess]

// ! [infer_request:start_pipeline]
void TemplateInferRequest::startPipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[StartPipeline])
    auto start = Time::now();
    _executable->call(_outputTensors, _inputTensors);
    _durations[StartPipeline] = Time::now() - start;
}
// ! [infer_request:start_pipeline]

void TemplateInferRequest::waitPipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[WaitPipeline])
    auto start = Time::now();
    // TODO: Wait pipeline using driver API or other synchronizations methods
    // NOTE: not used in current implementation since `startPipeline` executes pipiline synchronously
    _durations[WaitPipeline] = Time::now() - start;
}

// ! [infer_request:infer_postprocess]
void TemplateInferRequest::inferPostprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, _profilingTask[Postprocess]);
    auto start = Time::now();
    for (auto&& output : _outputs) {
        auto outputBlob = output.second;
        auto networkOutput = _networkOutputBlobs[output.first];
        // perform precision conversion of network output's precision and computational
        // graph output's precision are different
        if (outputBlob->getTensorDesc().getPrecision() != networkOutput->getTensorDesc().getPrecision()) {
            blobCopy(networkOutput, outputBlob);
        }
    }
    _durations[Postprocess] = Time::now() - start;
}
// ! [infer_request:infer_postprocess]

// ! [infer_request:get_performance_counts]
std::map<std::string, InferenceEngineProfileInfo> TemplateInferRequest::GetPerformanceCounts() const {
    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    InferenceEngineProfileInfo info;
    info.execution_index = 0;
    info.status = InferenceEngineProfileInfo::EXECUTED;

    info.cpu_uSec = info.realTime_uSec = _durations[Preprocess].count();
    perfMap["1. input preprocessing"] = info;
    info.cpu_uSec = info.realTime_uSec = 0;
    perfMap["2. input transfer to a device"] = info;
    info.cpu_uSec = info.realTime_uSec = _durations[StartPipeline].count();
    perfMap["3. execution time"] = info;
    info.cpu_uSec = info.realTime_uSec = 0;
    perfMap["4. output transfer from a device"] = info;
    info.cpu_uSec = info.realTime_uSec = _durations[Postprocess].count();
    perfMap["5. output postprocessing"] = info;
    return perfMap;
}
// ! [infer_request:get_performance_counts]
