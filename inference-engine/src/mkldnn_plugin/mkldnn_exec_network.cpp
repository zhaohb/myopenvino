// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <precision_utils.h>
#include <legacy/net_pass.h>
#include "mkldnn_exec_network.h"

#include "mkldnn_async_infer_request.h"
#include "mkldnn_infer_request.h"
#include "mkldnn_memory_state.h"
#include "mkldnn_itt.h"
#include "nodes/mkldnn_memory_node.hpp"
#include <legacy/ie_util_internal.hpp>
#include <legacy/graph_tools.hpp>
#include <threading/ie_executor_manager.hpp>

#include <threading/ie_cpu_streams_executor.hpp>
#include <ie_system_conf.h>
#include <threading/ie_thread_affinity.hpp>
#include <algorithm>
#include <unordered_set>
#include <utility>
#include <cstring>
#include <legacy/details/ie_cnn_network_tools.h>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

InferenceEngine::InferRequestInternal::Ptr
MKLDNNExecNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                          InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<MKLDNNInferRequest>(networkInputs, networkOutputs, std::static_pointer_cast<MKLDNNExecNetwork>(shared_from_this()));
}

MKLDNNExecNetwork::MKLDNNExecNetwork(const InferenceEngine::CNNNetwork &network,
                                     const Config &cfg,
                                     const MKLDNNExtensionManager::Ptr& extMgr,
                                     NumaNodesWeights &numaNodesWeights) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault{nullptr, nullptr},
    extensionManager(extMgr),
    _cfg{cfg},
    _name{network.getName()} {
    OV_ITT_TASK_CHAIN(taskChain, MKLDNNPlugin::itt::domains::MKLDNN_LT, "MKLDNNExecNetwork", "cloneNet");

    // we are cloning network if we have statistics and we can transform network.
    _clonedNetwork = cloneNetwork(network);

    if (_cfg.lpTransformsMode == Config::LPTransformsMode::On) {
        // Check if network is INT8 or Binary.
        // BF16 transformations were disabled since CPU plug-in doesn't support mixed precision execution:
        // BF16 + INT8 or BF16 + BIN.
        bool isFloatModel = true;
        CNNNetworkIterator iter(network);
        while (iter != CNNNetworkIterator()) {
            if (CaselessEq<std::string>()((*iter)->type, "FakeQuantize")) {
                isFloatModel = false;
                break;
            }
            iter++;
        }

        auto changePrecisionBF16 = [&](Precision current, Precision target) {
            InputsDataMap inputs = _clonedNetwork.getInputsInfo();
            OutputsDataMap outputs = _clonedNetwork.getOutputsInfo();
            CNNNetworkIterator iter(_clonedNetwork);
            while (iter != CNNNetworkIterator()) {
                //  check, if memory output node needs to be transformed
                if (current == Precision::FP32 &&
                    (*iter)->type == "Memory" && (*iter)->outData.size() == 0 &&
                    (*iter)->insData[0].lock()->getPrecision() == current) {
                    (*iter)->insData[0].lock()->setPrecision(target);
                }

                for (size_t o = 0; o < (*iter)->outData.size(); o++) {
                    if (inputs.find((*iter)->outData[o]->getName()) == inputs.end()
                        && outputs.find((*iter)->outData[o]->getName()) == outputs.end()
                        && !CaselessEq<std::string>()((*iter)->type, "const")
                        && (*iter)->outData[o]->getPrecision() == current) {
                        (*iter)->outData[o]->setPrecision(target);
                    }
                }
                iter++;
            }
        };

        if (with_cpu_x86_avx512_core() && isFloatModel) {
            // If enforceBF16 flag was set, BF16 transformation applies for all layers supported by CPU plugin.
            // Otherwise, only layers marked as BF16 in '_clonedNetwork' will be performed in bfloat16 mode.
            // CPU plugin throws an exception, if marked as BF16 layers have not supported by CPU plugin.
            if (cfg.enforceBF16 == true)
                changePrecisionBF16(Precision::FP32, Precision::BF16);
        } else {
            changePrecisionBF16(Precision::BF16, Precision::FP32);
        }
    }

    OV_ITT_TASK_NEXT(taskChain, "createConstInputs");
    auto createConstInputTo = [&](CNNLayerPtr layer, Blob::Ptr blob, const std::vector<size_t>& shape, const std::string& name) {
        LayerParams attrs = {layer->name + "_const_" + name, "Const", blob->getTensorDesc().getPrecision()};
        auto constLayer = std::make_shared<InferenceEngine::CNNLayer>(attrs);
        constLayer->blobs["custom"] = blob;

        const TensorDesc& td = {blob->getTensorDesc().getPrecision(), shape, TensorDesc::getLayoutByDims(shape)};

        DataPtr newEdgeAfterLayer(new Data(constLayer->name, td));
        newEdgeAfterLayer->setName(constLayer->name);
        getCreatorLayer(newEdgeAfterLayer) = constLayer;
        getInputTo(newEdgeAfterLayer).clear();

        IE_SUPPRESS_DEPRECATED_START
        auto icnnnet = static_cast<ICNNNetwork::Ptr>(_clonedNetwork);
        IE_SUPPRESS_DEPRECATED_END
        auto implNetwork = std::dynamic_pointer_cast<details::CNNNetworkImpl>(icnnnet);
        IE_ASSERT(implNetwork != nullptr);
        implNetwork->addData(constLayer->name.c_str(), newEdgeAfterLayer);
        implNetwork->addLayer(constLayer);

        constLayer->outData.push_back(newEdgeAfterLayer);
        getInputTo(newEdgeAfterLayer)[layer->name] = layer;
        layer->insData.push_back(newEdgeAfterLayer);
    };

    // The code block below transforms legacy layers to the form more compatible with opset1 in order to simplify future migration
    // TODO: remove after plug-in is migrated on opset1
    auto all_layers = details::CNNNetSortTopologically(_clonedNetwork);
    for (auto &layer : all_layers) {
        if (layer->type == "ScaleShift" && layer->insData.size() == 1) {
            auto constDimsRank = layer->insData[0].lock()->getDims().size();

            Blob::Ptr scalesBlob = layer->blobs["weights"];
            if (scalesBlob != nullptr) {
                std::vector<size_t> shape(constDimsRank, 1);
                shape[shape.size() > 1 ? 1 : 0] = scalesBlob->size();

                createConstInputTo(layer, scalesBlob, shape, "weights");
            }

            Blob::Ptr shiftBlob = layer->blobs["biases"];
            if (shiftBlob != nullptr) {
                std::vector<size_t> shape(constDimsRank, 1);
                shape[shape.size() > 1 ? 1 : 0] = shiftBlob->size();

                createConstInputTo(layer, shiftBlob, shape, "biases");
            } else if (scalesBlob != nullptr) {
                Blob::Ptr biases = make_shared_blob<float>(scalesBlob->getTensorDesc());
                if (biases == nullptr)
                    THROW_IE_EXCEPTION << "Cannot make 'biases' shared blob";
                biases->allocate();
                auto biasesPtr = biases->buffer().as<float*>();
                for (size_t i = 0; i < biases->size(); i++)
                    biasesPtr[i] = 0;

                std::vector<size_t> shape(constDimsRank, 1);
                shape[shape.size() > 1 ? 1 : 0] = biases->size();

                createConstInputTo(layer, biases, shape, "biases");
            }
        } else if (layer->type == "PReLU" && layer->insData.size() == 1) {
            Blob::Ptr scalesBlob = layer->blobs["weights"];
            if (scalesBlob != nullptr) {
                std::vector<size_t> shape(layer->insData[0].lock()->getDims().size(), 1);
                shape[shape.size() > 1 ? 1 : 0] = scalesBlob->size();

                createConstInputTo(layer, scalesBlob, shape, "weights");
            }
        } else if (layer->type == "DeformableConvolution") {
            auto * defConvLayer = dynamic_cast<DeformableConvolutionLayer*>(layer.get());
            if (defConvLayer == nullptr)
                THROW_IE_EXCEPTION << "Cannot convert deformable convolution layer.";

            Blob::Ptr weightsBlob = defConvLayer->blobs["weights"];
            if (weightsBlob != nullptr) {
                std::vector<size_t> shape;

                if (defConvLayer->_group != 1) {
                    shape.push_back(defConvLayer->_group);
                }
                shape.push_back(defConvLayer->_out_depth);
                shape.push_back(defConvLayer->input()->getDims()[1]);
                for (int i = 1; i <= defConvLayer->_kernel.size(); i++) {
                    shape.push_back(defConvLayer->_kernel[defConvLayer->_kernel.size() - i]);
                }

                createConstInputTo(layer, weightsBlob, shape, "weights");

                defConvLayer->blobs.clear();
                defConvLayer->_weights = nullptr;
            }
        } else if (layer->type == "BinaryConvolution") {
            auto * binConvLayer = dynamic_cast<BinaryConvolutionLayer*>(layer.get());
            if (binConvLayer == nullptr)
                THROW_IE_EXCEPTION << "Cannot convert binary convolution layer.";

            Blob::Ptr weightsBlob = binConvLayer->blobs["weights"];
            if (weightsBlob != nullptr) {
                std::vector<size_t> shape;

                if (binConvLayer->_group != 1) {
                    shape.push_back(binConvLayer->_group);
                }
                shape.push_back(binConvLayer->_out_depth);
                shape.push_back(binConvLayer->input()->getDims()[1]);
                for (int i = 1; i <= binConvLayer->_kernel.size(); i++) {
                    shape.push_back(binConvLayer->_kernel[binConvLayer->_kernel.size() - i]);
                }

                createConstInputTo(layer, weightsBlob, shape, "weights");

                binConvLayer->blobs.clear();
                binConvLayer->_weights = nullptr;
            }
        }
    }

    OV_ITT_TASK_SKIP(taskChain);

    if (_cfg.batchLimit > 1) {
        // check topology for applicability
        if (!CanProcessDynBatch(_clonedNetwork)) {
            THROW_IE_EXCEPTION << "MKLDNNGraph::CreateGraph: such topology cannot be compiled for dynamic batch!";
        }
    }

    if (cfg.exclusiveAsyncRequests) {
        // special case when all InferRequests are muxed into a single queue
        _taskExecutor = InferenceEngine::ExecutorManager::getInstance()->getExecutor("CPU");
    } else {
        auto streamsExecutorConfig = InferenceEngine::IStreamsExecutor::Config::MakeDefaultMultiThreaded(_cfg.streamExecutorConfig);
        streamsExecutorConfig._name = "CPUStreamsExecutor";
        _taskExecutor = InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(streamsExecutorConfig);
    }
    if (0 != cfg.streamExecutorConfig._streams) {
        _callbackExecutor = InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(
            IStreamsExecutor::Config{"CPUCallbackExecutor", 1, 0, IStreamsExecutor::ThreadBindingType::NONE});
    } else {
        _callbackExecutor = _taskExecutor;
    }

    _graphs = decltype(_graphs) {[&] {
        // TODO: Remove `cloneNet` to `localNetwork` when `MKLDNNGraph::CreateGraph`
        //       is fixed and does not change content of network passed (CVS-26420)
        auto localNetwork = cloneNetwork(_clonedNetwork);

        auto graph = std::make_shared<MKLDNNGraph>();
        {
            std::unique_lock<std::mutex> lock{_cfgMutex};
            graph->setConfig(_cfg);
        }
        int numaNode = 0;
        auto* streamExecutor = dynamic_cast<InferenceEngine::IStreamsExecutor*>(_taskExecutor.get());
        if (nullptr != streamExecutor) {
            numaNode = streamExecutor->GetNumaNodeId();
        }

        graph->CreateGraph(localNetwork, extensionManager, numaNodesWeights[numaNode]);
        return graph;
    }};

    _taskExecutor->runAndWait({std::thread::hardware_concurrency(), [this] {_graphs.local();}});

    // Save all MemoryLayer data tensors. Will use insight about mechanics
    // of MemoryLayer implementation. It uses output edge of MemoryLayer
    // producer as storage for tensor to keep it between infer calls.
    if (_graphs.size() == 1) {
        for (auto &node : _graphs.begin()->get()->GetNodes()) {
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
    }
}

void MKLDNNExecNetwork::setProperty(const std::map<std::string, std::string> &properties) {
    {
        std::lock_guard<std::mutex> lock{_cfgMutex};
        _cfg.readProperties(properties);
    }
    for (auto g : _graphs) {
        g->setProperty(properties);
    }
}

InferenceEngine::IInferRequest::Ptr MKLDNNExecNetwork::CreateInferRequest() {
    return CreateAsyncInferRequestFromSync<MKLDNNAsyncInferRequest>();
}

InferenceEngine::CNNNetwork MKLDNNExecNetwork::GetExecGraphInfo() {
    if (_graphs.size() == 0)
        THROW_IE_EXCEPTION << "No graph was found";

    return _graphs.begin()->get()->dump();
}

Parameter MKLDNNExecNetwork::GetConfig(const std::string &name) const {
    if (_graphs.size() == 0)
        THROW_IE_EXCEPTION << "No graph was found";
    Config engConfig = _graphs.begin()->get()->getProperty();
    auto it = engConfig._config.find(name);
    if (it != engConfig._config.end()) {
        return it->second;
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork config key: " << name;
    }
}

InferenceEngine::Parameter MKLDNNExecNetwork::GetMetric(const std::string &name) const {
    if (_graphs.size() == 0)
        THROW_IE_EXCEPTION << "No graph was found";

    if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_SET_METRIC_RETURN(NETWORK_NAME, _graphs.begin()->get()->GetName());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(NETWORK_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto && key : _graphs.begin()->get()->getProperty()._config) {
            configKeys.push_back(key.first);
        }
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        Config engConfig = _graphs.begin()->get()->getProperty();
        auto option = engConfig._config.find(CONFIG_KEY(CPU_THROUGHPUT_STREAMS));
        IE_ASSERT(option != engConfig._config.end());
        auto streams = std::stoi(option->second);
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, static_cast<unsigned int>(
            streams ? streams : 1));
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork metric: " << name;
    }
}

bool MKLDNNExecNetwork::CanProcessDynBatch(const InferenceEngine::CNNNetwork &network) const {
    InputsDataMap inputs = network.getInputsInfo();

    CNNLayerSet inputLayers;
    std::unordered_set<CNNLayer *> allLayers;

    if (inputs.empty())
        return false;

    auto & secondLayers = getInputTo(inputs.begin()->second->getInputData());
    if (secondLayers.empty())
        return false;

    bool check_result = true;
    details::UnorderedDFS(allLayers, secondLayers.begin()->second, [&](CNNLayerPtr layer) {
        auto type = TypeFromName(layer->type);
        // This is WA for Tile layer
        auto tileLayer = dynamic_cast<TileLayer *>(layer.get());
        if (tileLayer && tileLayer->axis)
            return;

        auto reshapeLayer = dynamic_cast<ReshapeLayer *>(layer.get());
        if (reshapeLayer &&
            type == Reshape &&
            (reshapeLayer->outData[0]->getTensorDesc().getDims()[0] ==
             reshapeLayer->insData[0].lock()->getTensorDesc().getDims()[0])) {
            return;
        }

        if (type != Input &&
            type != Output &&
            type != Convolution &&
            type != Deconvolution &&
            type != Activation &&
            type != Depthwise &&
            type != Lrn &&
            type != Pooling &&
            type != FullyConnected &&
            type != Gemm &&
            type != SoftMax &&
            type != Split &&
            type != Concatenation &&
            type != Eltwise &&
            type != Crop &&
            type != BatchNormalization &&
            type != Copy) {
            check_result = false;
        }
    }, false);

    return check_result;
}

IE_SUPPRESS_DEPRECATED_START
std::vector<IVariableStateInternal::Ptr> MKLDNNExecNetwork::QueryState() {
    return memoryStates;
}
IE_SUPPRESS_DEPRECATED_END
