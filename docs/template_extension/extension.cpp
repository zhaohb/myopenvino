// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "extension.hpp"
#include "cpu_kernel.hpp"
#include "op.hpp"
#ifdef OPENCV_IMPORT_ENABLED
#include "fft_op.hpp"
#include "fft_kernel.hpp"
#endif
#include <ngraph/ngraph.hpp>
#ifdef NGRAPH_ONNX_IMPORT_ENABLED
#include <onnx_import/onnx_utils.hpp>
#endif

#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace TemplateExtension;


//! [extension:ctor]
Extension::Extension() {
#ifdef NGRAPH_ONNX_IMPORT_ENABLED
    ngraph::onnx_import::register_operator(
        Operation::type_info.name, 1, "custom_domain", [](const ngraph::onnx_import::Node& node) -> ngraph::OutputVector {
            ngraph::OutputVector ng_inputs{node.get_ng_inputs()};
            int64_t add = node.get_attribute_value<int64_t>("add");
            return {std::make_shared<Operation>(ng_inputs.at(0), add)};
    });
    #ifdef OPENCV_IMPORT_ENABLED
    ngraph::onnx_import::register_operator(
            FFTOp::type_info.name, 1, "custom_domain", [](const ngraph::onnx_import::Node& node) -> ngraph::OutputVector {
            ngraph::OutputVector ng_inputs{node.get_ng_inputs()};
            bool inverse = node.get_attribute_value<int64_t>("inverse");
            return {std::make_shared<FFTOp>(ng_inputs.at(0), inverse)};
    });
    #endif
#endif
}
//! [extension:ctor]

//! [extension:dtor]
Extension::~Extension() {
#ifdef NGRAPH_ONNX_IMPORT_ENABLED
    ngraph::onnx_import::unregister_operator(Operation::type_info.name, 1, "custom_domain");
#ifdef OPENCV_IMPORT_ENABLED
    ngraph::onnx_import::unregister_operator(FFTOp::type_info.name, 1, "custom_domain");
#endif // OPENCV_IMPORT_ENABLED
#endif // NGRAPH_ONNX_IMPORT_ENABLED
}
//! [extension:dtor]

//! [extension:GetVersion]
void Extension::GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept {
    static InferenceEngine::Version ExtensionDescription = {
        {1, 0},           // extension API version
        "1.0",
        "template_ext"    // extension description message
    };

    versionInfo = &ExtensionDescription;
}
//! [extension:GetVersion]

//! [extension:getOpSets]
std::map<std::string, ngraph::OpSet> Extension::getOpSets() {
    std::map<std::string, ngraph::OpSet> opsets;
    ngraph::OpSet opset;
    opset.insert<Operation>();
#ifdef OPENCV_IMPORT_ENABLED
    opset.insert<FFTOp>();
#endif
    opsets["custom_opset"] = opset;
    return opsets;
}
//! [extension:getOpSets]

//! [extension:getImplTypes]
std::vector<std::string> Extension::getImplTypes(const std::shared_ptr<ngraph::Node> &node) {
    if (std::dynamic_pointer_cast<Operation>(node)) {
        return {"CPU"};
    }
#ifdef OPENCV_IMPORT_ENABLED
    if (std::dynamic_pointer_cast<FFTOp>(node)) {
        return {"CPU"};
    }
#endif
    return {};
}
//! [extension:getImplTypes]

//! [extension:getImplementation]
InferenceEngine::ILayerImpl::Ptr Extension::getImplementation(const std::shared_ptr<ngraph::Node> &node, const std::string &implType) {
    if (implType == "CPU") {
        if (std::dynamic_pointer_cast<Operation>(node)) {
            return std::make_shared<OpImplementation>(node);
        }
#ifdef OPENCV_IMPORT_ENABLED
        if (std::dynamic_pointer_cast<FFTOp>(node) && implType == "CPU") {
            return std::make_shared<FFTImpl>(node);
        }
#endif
    }
    return nullptr;
}
//! [extension:getImplementation]

//! [extension:CreateExtension]
// Exported function
INFERENCE_EXTENSION_API(InferenceEngine::StatusCode) InferenceEngine::CreateExtension(InferenceEngine::IExtension *&ext,
                                                                                      InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        ext = new Extension();
        return OK;
    } catch (std::exception &ex) {
        if (resp) {
            std::string err = ((std::string) "Couldn't create extension: ") + ex.what();
            err.copy(resp->msg, 255);
        }
        return InferenceEngine::GENERAL_ERROR;
    }
}
//! [extension:CreateExtension]
