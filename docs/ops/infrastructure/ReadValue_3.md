## ReadValue <a name="ReadValue"></a> {#openvino_docs_ops_infrastructure_ReadValue_3}

**Versioned name**: *ReadValue-3*

**Category**: *Infrastructure*

**Short description**: *ReadValue* returns value of the `variable_id` variable.

**Detailed description**: 

*ReadValue* returns value from the corresponding `variable_id` variable if the variable was set already by *Assign* operation and was not reset. 
The operation checks that the type and shape of the output are the same as
declared in `variable_id` and returns an error otherwise. If the corresponding variable was not set or was reset,
the operation returns the value from the 1 input, and initializes the `variable_id` shape and type
with the shape and type from the 1 input.

**Attributes**:

* *variable_id*

  * **Description**: identificator of the variable to be read
  * **Range of values**: any non-empty string
  * **Type**: string
  * **Default value**: None
  * **Required**: *Yes*

**Inputs**

*   **1**: `init_value` - input tensor with constant values of any supported type. **Required**.

**Outputs**

*   **1**: tensor with the same shape and type as `init_value`.

**Example**

```xml
<layer ... type="ReadValue" ...>
    <data variable_id="lstm_state_1"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>1</dim>
            <dim>3</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
    </output>
</layer>
```