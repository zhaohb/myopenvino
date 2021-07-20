// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_infer_request.h"
#include "mkldnn_extension_utils.h"
#include <vector>
#include <string>
#include <map>
#include <blob_factory.hpp>
#include <nodes/mkldnn_concat_node.h>
#include <nodes/mkldnn_split_node.h>
#include <ie_compound_blob.h>
#include <ie_common.h>
#include "mkldnn_exec_network.h"
#include "mkldnn_itt.h"
#include "nodes/common/cpu_convert.h"
#include "mkldnn_memory_state.h"
#include "nodes/mkldnn_memory_node.hpp"
#include "nodes/common/cpu_memcpy.h"
#include "mkldnn_async_infer_request.h"

MKLDNNPlugin::MKLDNNInferRequest::MKLDNNInferRequest(InferenceEngine::InputsDataMap     networkInputs,
                                                     InferenceEngine::OutputsDataMap    networkOutputs,
                                                     MKLDNNExecNetwork::Ptr             execNetwork_)
: InferRequestInternal(networkInputs, networkOutputs)
, execNetwork(execNetwork_) {
    auto id = (execNetwork->_numRequests)++;
    profilingTask = openvino::itt::handle("MKLDNN_INFER_" + execNetwork->_name + "_" + std::to_string(id));

    if (execNetwork->_graphs.size() == 0)
        THROW_IE_EXCEPTION << "No graph was found";
    graph = execNetwork->_graphs.begin()->get();
    for (const auto& it : _networkInputs) {
        MKLDNNInferRequest::GetBlob(it.first);
    }
    // Allocate all output blobs
    for (const auto& it : _networkOutputs) {
        MKLDNNInferRequest::GetBlob(it.first);
    }

    // Save all MemoryLayer data tensors. Will use insight about mechanics
    // of MemoryLayer implementation. It uses output edge of MemoryLayer
    // producer as storage for tensor to keep it between infer calls.
    IE_SUPPRESS_DEPRECATED_START
    if (execNetwork->_numRequests > 1 || execNetwork->QueryState().size() == 0) {
        for (auto &node : graph->GetNodes()) {
            if (node->getType() == MemoryInput) {
                auto memoryNode = dynamic_cast<MKLDNNMemoryInputNode*>(node.get());
                auto state_store = memoryNode->getStore();
                auto state_name = memoryNode->getId();

                // Remove suffix with pair ID. Internal information.
                auto suffix_idx = state_name.find("/id=");
                if (suffix_idx != std::string::npos)
                    state_name = state_name.substr(0, suffix_idx);

                memoryStates.emplace_back(new MKLDNNVariableState(state_name, state_store));
           }
        }
    } else {
        memoryStates = execNetwork->QueryState();
    }
    IE_SUPPRESS_DEPRECATED_END
}

MKLDNNPlugin::MKLDNNInferRequest::~MKLDNNInferRequest() {
    --(execNetwork->_numRequests);
}

void MKLDNNPlugin::MKLDNNInferRequest::pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob, InferenceEngine::Precision inPrec) {
    bool needConvert = inPrec != inputBlob->getTensorDesc().getPrecision();

    if (inputBlob->cbuffer().as<const void *>() == nullptr) {
        THROW_IE_EXCEPTION << "Input blob has no allocated memory";
    }

    InferenceEngine::Blob::Ptr iconv;
    if (needConvert) {
        iconv = make_blob_with_precision(inPrec, InferenceEngine::TensorDesc(inPrec, inputBlob->getTensorDesc().getDims(),
                                         inputBlob->getTensorDesc().getLayout()));
        iconv->allocate();
        if (inputBlob->size() != iconv->size())
            THROW_IE_EXCEPTION << "Can't copy tensor: input and converted tensors have different number of elements: " << inputBlob->size() << " and "
                               << iconv->size();

        void *srcData = inputBlob->cbuffer().as<void *>();
        void *dstData = iconv->buffer().as<void *>();
        if (dstData == nullptr) {
            THROW_IE_EXCEPTION << "Converted input blob has no allocated memory";
        }
        cpu_convert(srcData, dstData, inputBlob->getTensorDesc().getPrecision(), iconv->getTensorDesc().getPrecision(), iconv->size());
    }

    graph->PushInputData(inputName, needConvert ? iconv : inputBlob);
}

void MKLDNNPlugin::MKLDNNInferRequest::PushInputData() {
    for (auto input : _inputs) {
        if (!_networkInputs[input.first]) {
            THROW_IE_EXCEPTION << "Input blobs map contains not registered during IInferencePlugin::LoadNetwork blob with name " << input.first;
        }
        auto inPrec = input.second->getTensorDesc().getPrecision();

        switch (inPrec) {
            // these precisions are supported by mkldnn, so we push the blob directly
            case InferenceEngine::Precision::I8:
            case InferenceEngine::Precision::I32:
            case InferenceEngine::Precision::BF16:
            case InferenceEngine::Precision::FP32: {
                break;
            }
            // these precisions are supported by mkldnn, so we push the blob directly
            // BUT if a mean image exists, we convert the blob and send FP32
            case InferenceEngine::Precision::U8:
            case InferenceEngine::Precision::BOOL: {
                if (graph->hasMeanImageFor(input.first))
                    inPrec = InferenceEngine::Precision::FP32;
                break;
            }
            // these precisions are unsupported by mkldnn, so we convert the blob and send I32
            case InferenceEngine::Precision::U16:
            case InferenceEngine::Precision::I16:
            case InferenceEngine::Precision::I64:
            case InferenceEngine::Precision::U64: {
                inPrec = InferenceEngine::Precision::I32;
                break;
            }
            default:
                THROW_IE_EXCEPTION << "Unsupported input precision " << input.second->getTensorDesc().getPrecision();
        }

        // User can initialize input via setBlob API using tensorDesc with default (ANY) layout.
        // Currently IE doesn't specify behavior in such scenario, so we assume real layout is equal to the network input.
        if (input.second->getTensorDesc().getLayout() == InferenceEngine::ANY) {
            input.second->getTensorDesc().setLayout(_networkInputs[input.first]->getLayout());
        }

        pushInput(input.first, input.second, inPrec);
    }
}

void MKLDNNPlugin::MKLDNNInferRequest::PushStates() {
    for (auto &node : graph->GetNodes()) {
        if (node->getType() == MemoryInput) {
            auto cur_node = dynamic_cast<MKLDNNMemoryInputNode*>(node.get());
            auto cur_id = cur_node->getId();
            for (const auto& state : memoryStates) {
                if (state->GetName() == cur_id) {
                    auto cur_state_mem = cur_node->getStore();
                    auto data_ptr = state->GetState()->cbuffer().as<void*>();
                    auto data_size = state->GetState()->byteSize();
                    auto cur_state_mem_buf = static_cast<uint8_t*>(cur_state_mem->GetPtr());

                    cpu_memcpy(cur_state_mem_buf, data_ptr, data_size);
                }
            }
        }
    }
}

void MKLDNNPlugin::MKLDNNInferRequest::PullStates() {
    for (auto &node : graph->GetNodes()) {
        if (node->getType() == MemoryInput) {
            auto cur_node = dynamic_cast<MKLDNNMemoryInputNode*>(node.get());
            auto cur_id = cur_node->getId();
            for (const auto& state : memoryStates) {
                if (state->GetName() == cur_id) {
                    auto cur_state_mem = cur_node->getStore();
                    auto data_ptr = state->GetState()->cbuffer().as<void*>();
                    auto data_size = state->GetState()->byteSize();
                    auto cur_state_mem_buf = static_cast<uint8_t*>(cur_state_mem->GetPtr());

                    cpu_memcpy(data_ptr, cur_state_mem_buf, data_size);
                }
            }
        }
    }
}


void MKLDNNPlugin::MKLDNNInferRequest::InferImpl() {
    using namespace openvino::itt;
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, profilingTask);

    graph = execNetwork->_graphs.local().get();

    ThrowIfCanceled();

    execDataPreprocessing(_inputs);

    changeDefaultPtr();

    ThrowIfCanceled();

    PushInputData();

    if (memoryStates.size() != 0) {
        PushStates();
    }

    graph->Infer(this, m_curBatch);

    if (memoryStates.size() != 0) {
        PullStates();
    }

    ThrowIfCanceled();

    graph->PullOutputData(_outputs);
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> MKLDNNPlugin::MKLDNNInferRequest::GetPerformanceCounts() const {
    if (!graph || !graph->IsReady())
        THROW_IE_EXCEPTION << "Graph is not ready!";
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
    graph->GetPerfData(perfMap);
    return perfMap;
}

InferenceEngine::Blob::Ptr MKLDNNPlugin::MKLDNNInferRequest::GetBlob(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, "GetBlob");

    if (!graph || !graph->IsReady())
        THROW_IE_EXCEPTION << "Graph is not ready!";

    InferenceEngine::Blob::Ptr data;

    InferenceEngine::BlobMap blobs;
    graph->getInputBlobs(blobs);

    if (blobs.find(name) != blobs.end()) {
        // ROI blob is returned only if it was set previously.
        auto it = _preProcData.find(name);
        if (it != _preProcData.end()) {
            data = it->second->getRoiBlob();
            return data;
        }

        if (_inputs.find(name) != _inputs.end()) {
            data = _inputs[name];
            checkBlob(data, name, true);
            return data;
        }

        InferenceEngine::TensorDesc desc = blobs[name]->getTensorDesc();
        InferenceEngine::Precision originPrecision = blobs[name]->getTensorDesc().getPrecision();
        if (_networkInputs.find(name) != _networkInputs.end()) {
            InferenceEngine::Layout l = _networkInputs[name]->getLayout();
            InferenceEngine::Precision p = _networkInputs[name]->getPrecision();
            InferenceEngine::SizeVector dims = _networkInputs[name]->getTensorDesc().getDims();

            desc = InferenceEngine::TensorDesc(p, dims, l);
        }

        _inputs[name] = make_blob_with_precision(desc);
        _inputs[name]->allocate();
        if (desc.getPrecision() == originPrecision &&
                graph->_meanImages.find(name) == graph->_meanImages.end() && !graph->getProperty().batchLimit) {
            externalPtr[name] = _inputs[name]->buffer();
        }
        data = _inputs[name];
        checkBlob(data, name, true);
        return data;
    }
    blobs.clear();
    graph->getOutputBlobs(blobs);
    if (blobs.find(name) != blobs.end()) {
        if (_outputs.find(name) != _outputs.end()) {
            data = _outputs[name];
            checkBlob(data, name, false);
            return data;
        }

        InferenceEngine::TensorDesc desc = blobs[name]->getTensorDesc();

        // WA: need to avoid exception thrown when we compare blocking desc in SetBlob
        // in situation if we push output blobs as inputs for next network (in Hetero plugin)
        // it may be that output tensor desc will be different from real input tensor desc for next network
        // because the optimal descriptor was chosen (e.g. inPlace case for Split node)
        auto currBlockDesc = InferenceEngine::BlockingDesc(desc.getBlockingDesc().getBlockDims(), desc.getBlockingDesc().getOrder());
        desc = InferenceEngine::TensorDesc(desc.getPrecision(), desc.getDims(), currBlockDesc);

        _outputs[name] = make_blob_with_precision(desc);
        _outputs[name]->allocate();
        if (desc.getPrecision() == InferenceEngine::Precision::FP32 && !graph->getProperty().batchLimit) {
            externalPtr[name] = _outputs[name]->buffer();
        }
        data = _outputs[name];
        checkBlob(data, name, false);
        return data;
    }
    THROW_IE_EXCEPTION << "Cannot find blob with name: " << name;
}

void MKLDNNPlugin::MKLDNNInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, "SetBlob");
    if (name.empty()) {
        THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
    }

    if (!data)
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
    const bool compoundBlobPassed = data->is<InferenceEngine::CompoundBlob>();
    if (!compoundBlobPassed && data->buffer() == nullptr)
        THROW_IE_EXCEPTION << "Input data was not allocated. Input name: \'" << name << "\'";
    if (data->size() == 0) {
        THROW_IE_EXCEPTION << "Input data is empty. Input name: \'" << name << "\'";
    }

    InferenceEngine::InputInfo::Ptr foundInput;
    InferenceEngine::DataPtr foundOutput;
    size_t dataSize = data->size();
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        if (foundInput->getPrecision() != data->getTensorDesc().getPrecision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set input blob with precision: "
                               << data->getTensorDesc().getPrecision() << ", if CNNNetwork input blob precision is: " << foundInput->getPrecision();
        }

        const bool preProcRequired = preProcessingRequired(foundInput, data);
        if (compoundBlobPassed && !preProcRequired) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "cannot set compound blob: supported only for input pre-processing";
        }

        if (preProcRequired) {
            if (_preProcData.find(name) == _preProcData.end()) {
                _preProcData.emplace(name, InferenceEngine::CreatePreprocDataHelper());
            }
            _preProcData[name]->isApplicable(data, _inputs[name]);
            // Stores the given blob as ROI blob. It will be used to fill in network input during
            // pre-processing
            _preProcData[name]->setRoiBlob(data);
        } else {
            size_t inputSize = foundInput->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                ? InferenceEngine::details::product(foundInput->getTensorDesc().getDims())
                : 1;
            if (dataSize != inputSize) {
                THROW_IE_EXCEPTION << "Input blob size is not equal network input size ("
                                   << dataSize << "!=" << inputSize << ").";
            }

            if (foundInput->getTensorDesc().getDims() != data->getTensorDesc().getDims()) {
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set input blob. Dimensions mismatch.";
            }

            if (data->getTensorDesc().getLayout() != InferenceEngine::Layout::ANY && foundInput->getTensorDesc().getLayout() != InferenceEngine::Layout::ANY &&
                foundInput->getTensorDesc().getBlockingDesc() != data->getTensorDesc().getBlockingDesc()) {
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set input blob. Blocking descriptor mismatch.";
            }

            if (data->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32 &&
                graph->_meanImages.find(name) == graph->_meanImages.end() && !graph->getProperty().batchLimit) {
                externalPtr[name] = data->buffer();
            } else if (externalPtr.find(name) != externalPtr.end()) {
                externalPtr.erase(name);
            }
            _inputs[name] = data;
        }
    } else {
        if (compoundBlobPassed) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "cannot set compound blob: supported only for input pre-processing";
        }
        if (foundOutput->getPrecision() != data->getTensorDesc().getPrecision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set output blob with precision: "
                               << data->getTensorDesc().getPrecision() << ", if CNNNetwork output blob precision is: " << foundOutput->getPrecision();
        }
        size_t outputSize = foundOutput->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
            ? InferenceEngine::details::product(foundOutput->getDims())
            : 1;
        if (dataSize != outputSize) {
            THROW_IE_EXCEPTION << "Output blob size is not equal network output size ("
                               << dataSize << "!=" << outputSize << ").";
        }
        if (foundOutput->getTensorDesc().getDims() != data->getTensorDesc().getDims()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set output Blob. Dimensions mismatch.";
        }
        if (data->getTensorDesc().getLayout() != InferenceEngine::Layout::ANY && foundOutput->getTensorDesc().getLayout() != InferenceEngine::Layout::ANY &&
            foundOutput->getTensorDesc().getBlockingDesc() != data->getTensorDesc().getBlockingDesc()) {
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set output blob. Blocking descriptor mismatch.";
        }
        if (data->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32 &&
                !graph->getProperty().batchLimit) {
            externalPtr[name] = data->buffer();
        } else if (externalPtr.find(name) != externalPtr.end()) {
            externalPtr.erase(name);
        }
        _outputs[name] = data;
    }
}

static inline void changeEdgePtr(const MKLDNNPlugin::MKLDNNEdgePtr &edge, void *newPtr) {
    edge->getMemory().GetPrimitivePtr()->set_data_handle(newPtr);
}

void MKLDNNPlugin::MKLDNNInferRequest::changeDefaultPtr() {
    for (auto& it : externalPtr) {
        auto input = graph->inputNodes.find(it.first);
        if (input != graph->inputNodes.end()) {
            if (input->second->getChildEdgeAt(0)->getMemory().GetPrimitive().get_data_handle() == it.second)
                continue;
            // Input cannot be in-place with other primitives
            bool canBeInPlace = true;
            for (size_t i = 0; canBeInPlace && i < input->second->getChildEdges().size(); i++) {
                auto& child = input->second->getChildEdgeAt(i)->getChild();
                if (child->isConstant())
                    canBeInPlace = false;
                auto* concat = dynamic_cast<MKLDNNConcatNode *>(child.get());
                if (canBeInPlace && concat && concat->isOptimized())
                    canBeInPlace = false;

                // Cannot be in-place before split because split is using different ptrs without offsets
                auto* split = dynamic_cast<MKLDNNSplitNode *>(child.get());
                if (canBeInPlace && split)
                    canBeInPlace = false;

                if (child->isInplace())
                    canBeInPlace = false;
                for (size_t j = 0; canBeInPlace && j < child->getChildEdges().size(); j++) {
                    if (child->getChildEdgeAt(j)->getMemory().GetPrimitive().get_data_handle() ==
                            input->second->getChildEdgeAt(i)->getMemory().GetPrimitive().get_data_handle())
                        canBeInPlace = false;
                }
            }
            for (size_t i = 0; canBeInPlace && i < input->second->getChildEdges().size(); i++) {
                changeEdgePtr(input->second->getChildEdgeAt(i), it.second);
            }
            continue;
        }

        MKLDNNNodePtr output;
        for (auto& out : graph->outputNodes) {
            if (out->getName() == "out_" + it.first) {
                output = out;
                break;
            }
        }
        if (output) {
            if (output->getParentEdgeAt(0)->getMemory().GetPrimitive().get_data_handle() == it.second)
                continue;
            bool canBeInPlace = true;
            void * defaultPtr = output->getParentEdgeAt(0)->getMemory().GetPrimitivePtr()->get_data_handle();
            // Cannot be in-place after concat because concat is using different ptrs without offsets
            auto parent = output->getParentEdgeAt(0)->getParent();
            MKLDNNNodePtr previousParent;
            do {
                previousParent = parent;
                if (parent->getChildEdges().size() != 1 || parent->isConstant() || parent->isInplace()) {
                    canBeInPlace = false;
                    break;
                }

                for (size_t i = 0; i < parent->getParentEdges().size(); i++) {
                    if (parent->getParentEdgeAt(i)->getMemory().GetPrimitivePtr()->get_data_handle() == defaultPtr) {
                        parent = parent->getParentEdgeAt(i)->getParent();
                        break;
                    }
                }
            } while (previousParent != parent);
            if (canBeInPlace)
                changeEdgePtr(output->getParentEdgeAt(0), it.second);
            continue;
        }
        THROW_IE_EXCEPTION << "Cannot find input/output blob: " << it.first;
    }
}


void MKLDNNPlugin::MKLDNNInferRequest::SetBatch(int new_batch) {
    if (!graph->getProperty().enableDynamicBatch)
        THROW_IE_EXCEPTION << "Dynamic batch is not enabled.";

    if (new_batch < 1 || new_batch > graph->getProperty().batchLimit) {
        THROW_IE_EXCEPTION << "Invalid dynamic batch size " << new_batch <<
            " for this request.";
    }

    m_curBatch = new_batch;
}

std::vector<InferenceEngine::IVariableStateInternal::Ptr> MKLDNNPlugin::MKLDNNInferRequest::QueryState() {
    return memoryStates;
}

void MKLDNNPlugin::MKLDNNInferRequest::SetAsyncRequest(MKLDNNAsyncInferRequest* asyncRequest) {
    _asyncRequest = asyncRequest;
}

void MKLDNNPlugin::MKLDNNInferRequest::ThrowIfCanceled() const {
    if (_asyncRequest != nullptr) {
        _asyncRequest->ThrowIfCanceled();
    }
}