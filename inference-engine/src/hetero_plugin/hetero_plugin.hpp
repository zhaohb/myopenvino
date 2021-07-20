// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "description_buffer.hpp"
#include "ie_icore.hpp"
#include "cpp_interfaces/impl/ie_plugin_internal.hpp"
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <utility>

namespace HeteroPlugin {

class Engine : public InferenceEngine::InferencePluginInternal {
public:
    using Configs = std::map<std::string, std::string>;
    using DeviceMetaInformationMap = std::unordered_map<std::string, Configs>;

    Engine();

    InferenceEngine::ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(const InferenceEngine::CNNNetwork &network, const Configs &config) override;

    void SetConfig(const Configs &config) override;

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork &network,
                                                     const Configs& config) const override;

    InferenceEngine::Parameter GetMetric(const std::string& name, const std::map<std::string,
                                         InferenceEngine::Parameter> & options) const override;

    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string,
                                         InferenceEngine::Parameter> & options) const override;

    InferenceEngine::ExecutableNetwork ImportNetworkImpl(std::istream& heteroModel, const Configs& config) override;

    DeviceMetaInformationMap GetDevicePlugins(const std::string& targetFallback,
        const Configs & localConfig) const;

private:
    Configs GetSupportedConfig(const Configs& config, const std::string & deviceName) const;
};
}  // namespace HeteroPlugin
