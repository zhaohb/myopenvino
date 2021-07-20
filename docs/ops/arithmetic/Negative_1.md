## Negative <a name="Negative"></a> {#openvino_docs_ops_arithmetic_Negative_1}

**Versioned name**: *Negative-1*

**Category**: Arithmetic unary operation 

**Short description**: *Negative* performs element-wise negative operation with given tensor.

**Attributes**:

    No attributes available.

**Inputs**

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise negative operation. A tensor of type T.

**Types**

* *T*: any numeric type.

*Negative* does the following with the input tensor *a*:

\f[
a_{i} = -a_{i}
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Negative">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```