## Mish <a name="Mish"></a> {#openvino_docs_ops_activation_Mish_4}

**Versioned name**: *Mish-4*

**Category**: *Activation*

**Short description**: Mish is a Self Regularized Non-Monotonic Neural Activation Function.

**Detailed description**: Mish is a self regularized non-monotonic neural activation function proposed in the [article](https://arxiv.org/abs/1908.08681).

**Attributes**: operation has no attributes.

**Inputs**:

*   **1**: Input tensor *x* of any floating point type T. Required.

**Outputs**:

*   **1**: Floating point tensor with shape and type matching the input tensor.

**Types**

* *T*: any floating point type.

**Mathematical Formulation**

   For each element from the input tensor calculates corresponding
    element in the output tensor with the following formula:
\f[
Mish(x) = x*tanh(ln(1.0+e^{x}))
\f]

**Examples**

```xml
<layer ... type="Mish">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </input>
    <output>
        <port id="3">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```
