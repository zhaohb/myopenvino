# Custom nGraph Operation {#openvino_docs_IE_DG_Extensibility_DG_AddingNGraphOps}

Inference Engine Extension API enables you to register operation sets (opsets) with custom nGraph operations to support models with operations which OpenVINO™ does not support out-of-the-box.

## Operation Class

To add your custom nGraph operation, create a new class that extends `ngraph::Op`, which is in turn derived from `ngraph::Node`, the base class for all graph operations in nGraph. Follow the steps below:

1. Add the `NGRAPH_RTTI_DECLARATION` and `NGRAPH_RTTI_DEFINITION` macros which define a `NodeTypeInfo` object that identifies the type of the operation to the graph users and helps with dynamic type resolution. The type info of an nGraph operation currently consists of a string identifier and a version number, but this may change in the future.

2. Implement constructors that optionally take the operation inputs and attributes as parameters. 

3. Override the shape inference method `validate_and_infer_types`. This method is called multiple times during graph manipulations to determine the shapes and element types of the operations outputs. To access the input shapes and input element types, use the `get_input_partial_shape()` and `get_input_element_type()` methods of `ngraph::Node`. Set the inferred shape and element type of the output using `set_output_type`.

4. Override the `clone_with_new_inputs` method, which enables graph manipulation routines to create copies of this operation and connect it to different nodes during optimization.

5. Override the `visit_attributes` method, which enables serialization and deserialization of operation attributes. An `AttributeVisitor` is passed to the method, and the implementation is expected to walk over all the attributes in the op using the type-aware `on_attribute` helper. Helpers are already implemented for standard C++ types like `int64_t`, `float`, `bool`, `vector`, and for existing nGraph defined types.

6. Override `evaluate`, which is an optional method that enables the application of constant folding if there is a custom operation on the constant branch.

Based on that, declaration of an operation class can look as follows:

@snippet template_extension/op.hpp op:header

### Class Fields

The provided implementation has several fields:

 * `add` of type `int64_t` is an attribute of a custom operation.
 * `type_info` of type `ngraph::NodeTypeInfo` defines the type and version of an operation.

### Operation Constructors

nGraph operation contains two constructors: 
* Default constructor, which enables you to create an operation without attributes 
* Constructor that creates and validates an operation with specified inputs and attributes

@snippet template_extension/op.cpp op:ctor

### `validate_and_infer_types()`

`ngraph::Node::validate_and_infer_types` method validates operation attributes and calculates output shapes using attributes of the operation.

@snippet template_extension/op.cpp op:validate

### `clone_with_new_inputs()`

`ngraph::Node::clone_with_new_inputs` method creates a copy of the nGraph operation with new inputs.

@snippet template_extension/op.cpp op:copy

### `visit_attributes()`

`ngraph::Node::visit_attributes` method enables you to visit all operation attributes.

@snippet template_extension/op.cpp op:visit_attributes

### `evaluate()`

`ngraph::Node::evaluate` method enables you to apply constant folding to an operation.

@snippet template_extension/op.cpp op:evaluate

## Register Custom Operations in Extension Class

To add custom operations to the [Extension](Extension.md) class, create an operation set with custom operations and implement the `InferenceEngine::IExtension::getOpSets` method:

@snippet template_extension/extension.cpp extension:getOpSets

This method returns a map of opsets that exist in the extension library.

nGraph provides an opset mechanism to group operations into clusters. S. Different opsets distinguish between different versions of one operation.

When specifying opset names, follow the rules below:
* Use unique opset names.
* Do not use the following built-in opset names: `extension`, `experimental`, `opset1`, `opset2`, `opset3`, ... , `opsetN`.
* Make sure that the Model Optimizer and your extension use the same opset names.
* IR v10 operations have the mandatory `version` attribute specifying the opset.
Operations from the default opset cannot be redefined.

Use a custom opset to create a new operation or extend functionality of an existing operation from another opset.
