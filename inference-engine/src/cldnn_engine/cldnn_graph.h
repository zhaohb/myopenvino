// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef NOMINMAX
# define NOMINMAX
#endif

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include "ie_blob.h"
#include "cpp/ie_cnn_network.h"

#include <api/network.hpp>
#include <api/topology.hpp>

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include "cldnn_custom_layer.h"
#include "cldnn_config.h"
#include "cldnn_remote_context.h"
#include "cldnn_program.h"

namespace CLDNNPlugin {

class CLDNNGraph {
public:
    typedef std::shared_ptr<CLDNNGraph> Ptr;

    CLDNNGraph(InferenceEngine::CNNNetwork& network, InferenceEngine::gpu::ClContext::Ptr context, Config config, uint16_t stream_id = 0);
    explicit CLDNNGraph(std::shared_ptr<CLDNNGraph> graph, uint16_t stream_id = 0);
    InferenceEngine::CNNNetwork GetExecGraphInfo();

    bool IsLoaded() const;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const;
    void UpdatePerfStatistics();

    const Config& getConfig() const { return m_config; }
    InferenceEngine::gpu::ClContext::Ptr GetContext() { return m_context; }
    std::shared_ptr<const cldnn::engine> GetEngine() const { return getContextImpl(m_context)->GetEngine(); }
    int GetMaxDynamicBatchSize() const { return getConfig().max_dynamic_batch; }
    const std::map<std::string, cldnn::layout>& GetInputLayouts() const { return m_program->GetInputLayouts(); }
    size_t GetNetworksCount() const { return m_networks.size(); }
    std::shared_ptr<cldnn::network> GetNetwork(size_t idx = 0) const;
    InferenceEngine::SizeVector GetOutputSize(std::string outName) const;
    std::string MapOutputName(std::string outName) const;
    std::string getName() const { return m_networkName; }

protected:
    std::string m_networkName;
    Config m_config;

    InferenceEngine::gpu::ClContext::Ptr m_context;
    std::vector<std::shared_ptr<cldnn::network>> m_networks;
    std::map<std::string, cldnn::primitive_id> primitiveIDs;
    std::map<cldnn::primitive_id, std::vector<std::string>> primitivesToIRLayersMap;
    std::map<cldnn::primitive_id, std::string> IRToNgraphLayersMap;
    std::map<std::string, std::vector<cldnn::primitive_id>> prevPrimitiveIDs;

    std::map<cldnn::primitive_id, std::pair<std::string, PerfCounter>> perfMap;
    std::map<cldnn::primitive_id, std::string> implementationsMap;
    std::vector<cldnn::primitive_id> profilingIDs;

    std::map<std::string, InferenceEngine::SizeVector> outputDims;

    std::shared_ptr<Program> m_program;
    uint16_t m_stream_id;

    std::shared_ptr<cldnn::network> BuildNetwork(std::shared_ptr<cldnn::program> program);
    void Build();
    void UpdateLayersMaps();
    void UpdateImplementationsMap();
    InferenceEngine::CNNNetwork GetExecGraphInfoByPrimitivesInfo(std::vector<cldnn::primitive_info>& pi,
                                                                 bool filter_const_primitives = true);
};

}  // namespace CLDNNPlugin
