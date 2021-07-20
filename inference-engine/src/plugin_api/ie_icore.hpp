// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for ICore interface
 * @file ie_icore.hpp
 */

#pragma once

#include <array>
#include <memory>
#include <string>

#include <ie_parameter.hpp>
#include <cpp/ie_cnn_network.h>
#include <cpp/ie_executable_network.hpp>

#include "threading/ie_itask_executor.hpp"

namespace InferenceEngine {

/**
 * @interface ICore
 * @brief Minimal ICore interface to allow plugin to get information from Core Inference Engine class.
 * @ingroup ie_dev_api_plugin_api
 */
class ICore {
public:
    /**
     * @brief Returns global to Inference Engine class task executor
     * @return Reference to task executor
     */
    virtual std::shared_ptr<ITaskExecutor> GetTaskExecutor() const = 0;

    /**
     * @brief Reads IR xml and bin (with the same name) files
     * @param model string with IR
     * @param weights shared pointer to constant blob with weights
     * @return CNNNetwork
     */
    virtual CNNNetwork ReadNetwork(const std::string& model, const Blob::CPtr& weights) const = 0;

    /**
     * @brief Reads IR xml and bin files
     * @param modelPath path to IR file
     * @param binPath path to bin file, if path is empty, will try to read bin file with the same name as xml and
     * if bin file with the same name was not found, will load IR without weights.
     * @return CNNNetwork
     */
    virtual CNNNetwork ReadNetwork(const std::string& modelPath, const std::string& binPath) const = 0;

    /**
     * @brief Creates an executable network from a network object.
     *
     * Users can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     *
     * @param network CNNNetwork object acquired from Core::ReadNetwork
     * @param deviceName Name of device to load network to
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation
     * @return An executable network reference
     */
    virtual ExecutableNetwork LoadNetwork(const CNNNetwork& network, const std::string& deviceName,
                                          const std::map<std::string, std::string>& config = {}) = 0;

    /**
     * @brief Creates an executable network from a previously exported network
     * @param deviceName Name of device load executable network on
     * @param networkModel network model stream
     * @param config Optional map of pairs: (config parameter name, config parameter value) relevant only for this load
     * operation*
     * @return An executable network reference
     */
    virtual ExecutableNetwork ImportNetwork(std::istream& networkModel, const std::string& deviceName = {},
                                            const std::map<std::string, std::string>& config = {}) = 0;

    /**
     * @brief Query device if it supports specified network with specified configuration
     *
     * @param deviceName A name of a device to query
     * @param network Network object to query
     * @param config Optional map of pairs: (config parameter name, config parameter value)
     * @return An object containing a map of pairs a layer name -> a device name supporting this layer.
     */
    virtual QueryNetworkResult QueryNetwork(const CNNNetwork& network, const std::string& deviceName,
                                            const std::map<std::string, std::string>& config) const = 0;

    /**
     * @brief Gets general runtime metric for dedicated hardware.
     *
     * The method is needed to request common device properties
     * which are executable network agnostic. It can be device name, temperature, other devices-specific values.
     *
     * @param deviceName - A name of a device to get a metric value.
     * @param name - metric name to request.
     * @return Metric value corresponding to metric key.
     */
    virtual Parameter GetMetric(const std::string& deviceName, const std::string& name) const = 0;

    /**
     * @brief Default virtual destructor
     */
    virtual ~ICore() = default;
};

/**
 * @brief Type of magic value
 * @ingroup ie_dev_api_plugin_api
 */
using ExportMagic = std::array<char, 4>;

/**
 * @brief Magic number used by ie core to identify exported network with plugin name
 * @ingroup ie_dev_api_plugin_api
 */
constexpr static const ExportMagic exportMagic = {{0x1, 0xE, 0xE, 0x1}};

/**
 * @private
 */
class INFERENCE_ENGINE_API_CLASS(DeviceIDParser) {
    std::string deviceName;
    std::string deviceID;
public:
    explicit DeviceIDParser(const std::string& deviceNameWithID);

    std::string getDeviceID() const;
    std::string getDeviceName() const;

    static std::vector<std::string> getHeteroDevices(std::string fallbackDevice);
    static std::vector<std::string> getMultiDevices(std::string devicesList);
};

}  // namespace InferenceEngine
