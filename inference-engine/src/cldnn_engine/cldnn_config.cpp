// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sys/stat.h>

#include <cldnn/cldnn_config.hpp>
#include "cldnn_config.h"
#include "cpp_interfaces/exception2status.hpp"
#include "details/ie_exception.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "ie_api.h"
#include "file_utils.h"
#include "cldnn_itt.h"

#ifdef _WIN32
# include <direct.h>
#ifdef ENABLE_UNICODE_PATH_SUPPORT
# define mkdir(dir, mode) _wmkdir(dir)
#else
# define mkdir(dir, mode) _mkdir(dir)
#endif  // ENABLE_UNICODE_PATH_SUPPORT
#endif  // _WIN32

using namespace InferenceEngine;

namespace CLDNNPlugin {

static void createDirectory(std::string _path) {
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring widepath = FileUtils::multiByteCharToWString(_path.c_str());
    const wchar_t* path = widepath.c_str();
#else
    const char* path = _path.c_str();
#endif

    auto err = mkdir(path, 0755);
    if (err != 0 && errno != EEXIST) {
        THROW_IE_EXCEPTION << "Couldn't create directory! (err=" << err << "; errno=" << errno << ")";
    }
}

void Config::UpdateFromMap(const std::map<std::string, std::string>& configMap) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "Config::UpdateFromMap");
    for (auto& kvp : configMap) {
        std::string key = kvp.first;
        std::string val = kvp.second;

        if (key.compare(PluginConfigParams::KEY_PERF_COUNT) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                useProfiling = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                useProfiling = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_DYN_BATCH_ENABLED) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enableDynamicBatch = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enableDynamicBatch = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_DUMP_KERNELS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                dumpCustomKernels = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                dumpCustomKernels = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_PLUGIN_PRIORITY) == 0) {
            std::stringstream ss(val);
            uint32_t uVal(0);
            ss >> uVal;
            if (ss.fail()) {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
            switch (uVal) {
                case 0:
                    queuePriority = cldnn::priority_mode_types::disabled;
                    break;
                case 1:
                    queuePriority = cldnn::priority_mode_types::low;
                    break;
                case 2:
                    queuePriority = cldnn::priority_mode_types::med;
                    break;
                case 3:
                    queuePriority = cldnn::priority_mode_types::high;
                    break;
                default:
                    THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported queue priority value: " << uVal;
            }

        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE) == 0) {
            std::stringstream ss(val);
            uint32_t uVal(0);
            ss >> uVal;
            if (ss.fail()) {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
            switch (uVal) {
                case 0:
                    queueThrottle = cldnn::throttle_mode_types::disabled;
                    break;
                case 1:
                    queueThrottle = cldnn::throttle_mode_types::low;
                    break;
                case 2:
                    queueThrottle = cldnn::throttle_mode_types::med;
                    break;
                case 3:
                    queueThrottle = cldnn::throttle_mode_types::high;
                    break;
                default:
                    THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported queue throttle value: " << uVal;
            }
        } else if (key.compare(PluginConfigParams::KEY_CONFIG_FILE) == 0) {
            std::stringstream ss(val);
            std::istream_iterator<std::string> begin(ss);
            std::istream_iterator<std::string> end;
            std::vector<std::string> configFiles(begin, end);
            for (auto& file : configFiles) {
                CLDNNCustomLayer::LoadFromFile(file, customLayers);
            }
        } else if (key.compare(PluginConfigParams::KEY_TUNING_MODE) == 0) {
            if (val.compare(PluginConfigParams::TUNING_DISABLED) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_disabled;
            } else if (val.compare(PluginConfigParams::TUNING_CREATE) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_tune_and_cache;
            } else if (val.compare(PluginConfigParams::TUNING_USE_EXISTING) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_use_cache;
            } else if (val.compare(PluginConfigParams::TUNING_UPDATE) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_use_and_update;
            } else if (val.compare(PluginConfigParams::TUNING_RETUNE) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_retune_and_cache;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported tuning mode value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_TUNING_FILE) == 0) {
            tuningConfig.cache_file_path = val;
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_MEM_POOL) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                memory_pool_on = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                memory_pool_on = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported memory pool flag value: " << val;
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_GRAPH_DUMPS_DIR) == 0) {
            if (!val.empty()) {
                graph_dumps_dir = val;
                createDirectory(graph_dumps_dir);
            }
        } else if (key.compare(PluginConfigParams::KEY_CACHE_DIR) == 0) {
            if (!val.empty()) {
                kernels_cache_dir = val;
                createDirectory(kernels_cache_dir);
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_SOURCES_DUMPS_DIR) == 0) {
            if (!val.empty()) {
                sources_dumps_dir = val;
                createDirectory(sources_dumps_dir);
            }
        } else if (key.compare(PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                exclusiveAsyncRequests = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                exclusiveAsyncRequests = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS) == 0) {
            if (val.compare(PluginConfigParams::GPU_THROUGHPUT_AUTO) == 0) {
                throughput_streams = 2;
            } else {
                int val_i;
                try {
                    val_i = std::stoi(val);
                } catch (const std::exception&) {
                    THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS
                                       << ". Expected only positive numbers (#streams) or "
                                       << "PluginConfigParams::GPU_THROUGHPUT_AUTO";
                }
                if (val_i > 0)
                    throughput_streams = static_cast<uint16_t>(val_i);
            }
        } else if (key.compare(PluginConfigParams::KEY_DEVICE_ID) == 0) {
            // Validate if passed value is postivie number.
            try {
                int val_i = std::stoi(val);
                (void)val_i;
            } catch (const std::exception&) {
                THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_DEVICE_ID
                    << ". DeviceIDs are only represented by positive numbers";
            }
            // Set this value.
            device_id = val;
        } else if (key.compare(PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enableInt8 = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enableInt8 = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                nv12_two_inputs = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                nv12_two_inputs = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported NV12 flag value: " << val;
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enable_fp16_for_quantized_models = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enable_fp16_for_quantized_models = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported KEY_CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS flag value: " << val;
            }
        } else {
            THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property key by plugin: " << key;
        }

        adjustKeyMapValues();
    }
}

void Config::adjustKeyMapValues() {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "Config::AdjustKeyMapValues");
    if (useProfiling)
        key_config_map[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::NO;

    if (dumpCustomKernels)
        key_config_map[PluginConfigParams::KEY_DUMP_KERNELS] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_DUMP_KERNELS] = PluginConfigParams::NO;

    if (exclusiveAsyncRequests)
        key_config_map[PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = PluginConfigParams::NO;

    if (memory_pool_on)
        key_config_map[CLDNNConfigParams::KEY_CLDNN_MEM_POOL] = PluginConfigParams::YES;
    else
        key_config_map[CLDNNConfigParams::KEY_CLDNN_MEM_POOL] = PluginConfigParams::NO;

    if (enableDynamicBatch)
        key_config_map[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::NO;

    if (nv12_two_inputs)
        key_config_map[CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS] = PluginConfigParams::YES;
    else
        key_config_map[CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS] = PluginConfigParams::NO;

    if (enable_fp16_for_quantized_models)
        key_config_map[CLDNNConfigParams::KEY_CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS] = PluginConfigParams::YES;
    else
        key_config_map[CLDNNConfigParams::KEY_CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS] = PluginConfigParams::NO;

    {
        std::string qp = "0";
        switch (queuePriority) {
        case cldnn::priority_mode_types::low: qp = "1"; break;
        case cldnn::priority_mode_types::med: qp = "2"; break;
        case cldnn::priority_mode_types::high: qp = "3"; break;
        default: break;
        }
        key_config_map[CLDNNConfigParams::KEY_CLDNN_PLUGIN_PRIORITY] = qp;
    }
    {
        std::string qt = "0";
        switch (queueThrottle) {
        case cldnn::throttle_mode_types::low: qt = "1"; break;
        case cldnn::throttle_mode_types::med: qt = "2"; break;
        case cldnn::throttle_mode_types::high: qt = "3"; break;
        default: break;
        }
        key_config_map[CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE] = qt;
    }
    {
        std::string tm = PluginConfigParams::TUNING_DISABLED;
        switch (tuningConfig.mode) {
        case cldnn::tuning_mode::tuning_tune_and_cache: tm = PluginConfigParams::TUNING_CREATE; break;
        case cldnn::tuning_mode::tuning_use_cache: tm = PluginConfigParams::TUNING_USE_EXISTING; break;
        case cldnn::tuning_mode::tuning_use_and_update: tm = PluginConfigParams::TUNING_UPDATE; break;
        case cldnn::tuning_mode::tuning_retune_and_cache: tm = PluginConfigParams::TUNING_RETUNE; break;
        default: break;
        }
        key_config_map[PluginConfigParams::KEY_TUNING_MODE] = tm;
        key_config_map[PluginConfigParams::KEY_TUNING_FILE] = tuningConfig.cache_file_path;
    }

    key_config_map[CLDNNConfigParams::KEY_CLDNN_GRAPH_DUMPS_DIR] = graph_dumps_dir;
    key_config_map[CLDNNConfigParams::KEY_CLDNN_SOURCES_DUMPS_DIR] = sources_dumps_dir;
    key_config_map[PluginConfigParams::KEY_CACHE_DIR] = kernels_cache_dir;

    key_config_map[PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS] = std::to_string(throughput_streams);
    key_config_map[PluginConfigParams::KEY_DEVICE_ID] = device_id;
    key_config_map[PluginConfigParams::KEY_CONFIG_FILE] = "";
}
}  // namespace CLDNNPlugin
