# Custom ONNX* Operators {#openvino_docs_IE_DG_Extensibility_DG_Custom_ONNX_Ops}

The ONNX\* importer provides a mechanism to register custom ONNX operators based on predefined or custom nGraph operations.
The function responsible for registering a new operator is called `ngraph::onnx_import::register_operator` and is defined in `onnx_import/onnx_utils.hpp`.

## Register Custom ONNX Operator Based on Predefined nGraph Operations

The steps below explain how to register a custom ONNX operator, for example, CustomRelu, in a domain called `com.example`.
CustomRelu is defined as follows:
```
x >= 0 => f(x) = x * alpha
x <  0 => f(x) = x * beta
```
where `alpha` and `beta` are float constants.

1. Include headers:
@snippet onnx_custom_op/onnx_custom_op.cpp onnx_custom_op:headers

2. Register the CustomRelu operator in the ONNX importer:
@snippet onnx_custom_op/onnx_custom_op.cpp onnx_custom_op:register_operator
The `register_operator` function takes four arguments: op_type, opset version, domain, and a function object.
The function object is a user-defined function that takes `ngraph::onnx_import::Node` as an input and based on that, returns a graph with nGraph operations.
The `ngraph::onnx_import::Node` class represents a node in an ONNX model. It provides functions to fetch input node(s) using `get_ng_inputs`, attribute value using `get_attribute_value`, and many more. See `onnx_import/core/node.hpp` for full class declaration.

New operator registration must happen before an ONNX model is read. For example, if an model uses the `CustomRelu` operator, call `register_operator("CustomRelu", ...)` before InferenceEngine::Core::ReadNetwork.
Reregistering ONNX operators within the same process is supported. If you register an existing operator, you get a warning.

The example below demonstrates an exemplary model that requires a previously created `CustomRelu` operator:
@snippet onnx_custom_op/onnx_custom_op.cpp onnx_custom_op:model


To create a graph with nGraph operations, visit [Custom nGraph Operations](AddingNGraphOps.md).
For a complete list of predefined nGraph operators, visit [Available Operations Sets](../../ops/opset.md).

If you do not need an operator anymore, unregister it by calling `unregister_operator`. The function takes three arguments: `op_type`, `version`, and `domain`.
@snippet onnx_custom_op/onnx_custom_op.cpp onnx_custom_op:unregister_operator

## Register Custom ONNX Operator Based on Custom nGraph Operations

The same principles apply when registering a custom ONNX operator based on custom nGraph operations.
This example shows how to register a custom ONNX operator based on `Operation` presented in [this tutorial](AddingNGraphOps.md), which is used in [TemplateExtension](Extension.md).
@snippet template_extension/extension.cpp extension:ctor

Here, the `register_operator` function is called in the constructor of Extension. The constructor makes sure that the function is called before InferenceEngine::Core::ReadNetwork, because InferenceEngine::Core::AddExtension must be called before a model with a custom operator is read.

The example below demonstrates how to unregister an operator from the destructor of Extension:
@snippet template_extension/extension.cpp extension:dtor

> **NOTE**: It is mandatory to unregister a custom ONNX operator if it is defined in a dynamic shared library.

## Requirements for Building with CMake

A program that uses the `register_operator` functionality requires `ngraph` and `onnx_importer` libraries in addition to the Inference Engine.
The `onnx_importer` is a component of the `ngraph` package , so `find_package(ngraph REQUIRED COMPONENTS onnx_importer)` can find both.
The `ngraph` package exposes two variables, `${NGRAPH_LIBRARIES}` and `${ONNX_IMPORTER_LIBRARIES}`, which reference the `ngraph` and `onnx_importer` libraries.
Those variables need to be passed to the `target_link_libraries` command in the CMakeLists.txt file.

See CMakeLists.txt below for reference:
@snippet onnx_custom_op/CMakeLists.txt cmake:onnx_custom_op
