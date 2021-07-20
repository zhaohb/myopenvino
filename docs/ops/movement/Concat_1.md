## Concat <a name="Concat"></a> {#openvino_docs_ops_movement_Concat_1}

**Versioned name**: *Concat-1*

**Category**: data movement operation.

**Short description**: Concatenates arbitrary number of input tensors to a single output tensor along one axis.

**Attributes**:

* *axis*

  * **Description**: *axis* specifies dimension to concatenate along
  * **Range of values**: integer number. Negative value means counting dimension from the end
  * **Type**: int
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1..N**: Arbitrary number of input tensors of type *T*. Types of all tensors should match. Rank of all tensors should match. The rank is positive, so scalars as inputs are not allowed. Shapes for all inputs should match at every position except `axis` position. At least one input is required.

**Outputs**:

*   **1**: Tensor of the same type *T* as input tensor and shape `[d1, d2, ..., d_axis, ...]`, where `d_axis` is a sum of sizes of input tensors along `axis` dimension.

**Types**

* *T*: any numeric type.

**Example**

```xml
<layer id="1" type="Concat">
    <data axis="1" />
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>8</dim>  <!-- axis for concatenation -->
            <dim>50</dim>
            <dim>50</dim>
        </port>
        <port id="1">
            <dim>1</dim>
            <dim>16</dim>  <!-- axis for concatenation -->
            <dim>50</dim>
            <dim>50</dim>
        </port>
        <port id="2">
            <dim>1</dim>
            <dim>32</dim>  <!-- axis for concatenation -->
            <dim>50</dim>
            <dim>50</dim>
        </port>
    </input>
    <output>
        <port id="0">
            <dim>1</dim>
            <dim>56</dim>  <!-- concatenated axis: 8 + 16 + 32 = 48 -->
            <dim>50</dim>
            <dim>50</dim>
        </port>
    </output>
</layer>

```