// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file contains a wrapper class for handling plugin instantiation and releasing resources
 *
 * @file ie_plugin_ptr.hpp
 */
#pragma once

#include <string>

#include "details/ie_so_pointer.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"

namespace InferenceEngine {
namespace details {

/**
 * @brief This class defines the name of the fabric for creating an IInferencePlugin object in DLL
 */
template <>
class SOCreatorTrait<IInferencePlugin> {
public:
    /**
     * @brief A name of the fabric for creating IInferencePlugin object in DLL
     */
    static constexpr auto name = "CreatePluginEngine";
};

}  // namespace details

/**
 * @brief A C++ helper to work with objects created by the plugin.
 *
 * Implements different interfaces.
 */
using InferenceEnginePluginPtr = InferenceEngine::details::SOPointer<IInferencePlugin>;

}  // namespace InferenceEngine
