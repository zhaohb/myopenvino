// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <memory>
#include <ie_iextension.h>
#include <legacy/ie_layers.h>
#include "nodes/list.hpp"

namespace MKLDNNPlugin {

class MKLDNNExtensionManager {
public:
    using Ptr = std::shared_ptr<MKLDNNExtensionManager>;
    MKLDNNExtensionManager() = default;
    InferenceEngine::ILayerImpl::Ptr CreateImplementation(const std::shared_ptr<ngraph::Node>& op);
    std::shared_ptr<InferenceEngine::ILayerImplFactory> CreateExtensionFactory(const InferenceEngine::CNNLayerPtr& Layer);
    void AddExtension(InferenceEngine::IExtensionPtr extension);

private:
    std::vector<InferenceEngine::IExtensionPtr> _extensions;
};

}  // namespace MKLDNNPlugin
