// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_plugin_config.hpp>
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <cpp_interfaces/exception2status.hpp>

#include "template_config.hpp"
#include "template/template_config.hpp"

using namespace TemplatePlugin;

Configuration::Configuration() { }

Configuration::Configuration(const ConfigMap& config, const Configuration & defaultCfg, bool throwOnUnsupported) {
    *this = defaultCfg;
    // If plugin needs to use InferenceEngine::StreamsExecutor it should be able to process its configuration
    auto streamExecutorConfigKeys = _streamsExecutorConfig.SupportedKeys();
    for (auto&& c : config) {
        const auto& key = c.first;
        const auto& value = c.second;

        if (TEMPLATE_CONFIG_KEY(THROUGHPUT_STREAMS) == key) {
            _streamsExecutorConfig.SetConfig(CONFIG_KEY(CPU_THROUGHPUT_STREAMS), value);
        } else if (streamExecutorConfigKeys.end() !=
            std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), key)) {
            _streamsExecutorConfig.SetConfig(key, value);
        } else if (CONFIG_KEY(DEVICE_ID) == key) {
            deviceId = std::stoi(value);
            if (deviceId > 0) {
                THROW_IE_EXCEPTION << "Device ID " << deviceId << " is not supported";
            }
        } else if (CONFIG_KEY(PERF_COUNT) == key) {
            perfCount = (CONFIG_VALUE(YES) == value);
        } else if (throwOnUnsupported) {
            THROW_IE_EXCEPTION << NOT_FOUND_str << ": " << key;
        }
    }
}

InferenceEngine::Parameter Configuration::Get(const std::string& name) const {
    if (name == CONFIG_KEY(DEVICE_ID)) {
        return {std::to_string(deviceId)};
    } else if (name == CONFIG_KEY(PERF_COUNT)) {
        return {perfCount};
    } else if (name == TEMPLATE_CONFIG_KEY(THROUGHPUT_STREAMS) || name == CONFIG_KEY(CPU_THROUGHPUT_STREAMS)) {
        return {std::to_string(_streamsExecutorConfig._streams)};
    } else if (name == CONFIG_KEY(CPU_BIND_THREAD)) {
        return const_cast<InferenceEngine::IStreamsExecutor::Config&>(_streamsExecutorConfig).GetConfig(name);
    } else if (name == CONFIG_KEY(CPU_THREADS_NUM)) {
        return {std::to_string(_streamsExecutorConfig._threads)};
    } else if (name == CONFIG_KEY_INTERNAL(CPU_THREADS_PER_STREAM)) {
        return {std::to_string(_streamsExecutorConfig._threadsPerStream)};
    } else {
        THROW_IE_EXCEPTION << NOT_FOUND_str << ": " << name;
    }
}
