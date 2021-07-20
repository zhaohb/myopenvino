## LRN <a name="LRN"></a> {#openvino_docs_ops_normalization_LRN_1}

**Versioned name**: *LRN-1*

**Category**: *Normalization*

**Short description**: Local response normalization.

**Attributes**:

* *alpha*

  * **Description**: *alpha* represents the scaling attribute for the normalizing sum. For example, *alpha* equal 0.0001 means that the normalizing sum is multiplied by 0.0001.
  * **Range of values**: no restrictions
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

* *beta*

  * **Description**: *beta* represents the exponent for the normalizing sum. For example, *beta* equal 0.75 means that the normalizing sum is raised to the power of 0.75.
  * **Range of values**: positive number
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

* *bias*

  * **Description**: *bias* represents the offset. Usually positive number to avoid dividing by zero.
  * **Range of values**: no restrictions
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

* *size*

  * **Description**: *size* represents the side length of the region to be used for the normalization sum. The region can have one or more dimensions depending on the second input axes indices.
  * **Range of values**: positive integer
  * **Type**: int
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: `data` - input tensor of any floating point type and arbitrary shape. Required.

* **2**: `axes` - specifies indices of dimensions in `data` that define normalization slices. Required.

**Outputs**

* **1**: Output tensor of the same shape and type as the `data` input tensor.

**Detailed description**:
Local Response Normalization performs a normalization over local input regions.
Each input value is divided by
\f[ (bias + \frac{alpha}{{size}^{len(axes)}} \cdot \sum_{i} data_{i})^{beta} \f]
The sum is taken over a region of a side length `size` and number of dimensions equal to number of axes.
The region is centered at the input value that's being normalized (with zero padding added if needed).

Here is an example for 4D `data` input tensor and `axes = [1]`:
```
sqr_sum[a, b, c, d] =
    sum(data[a, max(0, b - size / 2) : min(data.shape[1], b + size / 2 + 1), c, d] ** 2)
output = data / (bias + (alpha / size ** len(axes)) * sqr_sum) ** beta
```

Example for 4D `data` input tensor and `axes = [2, 3]`:
```
sqr_sum[a, b, c, d] =
    sum(data[a, b, max(0, c - size / 2) : min(data.shape[2], c + size / 2 + 1),  max(0, d - size / 2) : min(data.shape[3], d + size / 2 + 1)] ** 2)
output = data / (bias + (alpha / size ** len(axes)) * sqr_sum) ** beta
```

**Example**

```xml
<layer id="1" type="LRN" ...>
    <data alpha="1.0e-04" beta="0.75" size="5" bias="1"/>
    <input>
        <port id="0">
            <dim>6</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="1">
            <dim>1</dim>         <!-- value is [1] that means independent normalization for each pixel along channels -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>6</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
    </output>
</layer>
```
