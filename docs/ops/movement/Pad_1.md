## Pad <a name="Pad"></a> {#openvino_docs_ops_movement_Pad_1}

**Versioned name**: *Pad-1*

**Category**: *Data movement operations*

**Short description**: *Pad* operation extends an input tensor on edges. The amount and value of padded elements are defined by inputs and attributes.

**Attributes**

* *pad_mode*

  * **Description**: *pad_mode* specifies the method used to generate new element values.
  * **Range of values**: Name of the method in string format:
    * `constant` - padded values are equal to the value of the *pad_value* operation attribute.
    * `edge` - padded values are copied from the respective edge of the input `data` tensor.
    * `reflect` - padded values are a reflection of the input `data` tensor; values on the edges are not duplicated. `pads_begin[D]` and `pads_end[D]` must be not greater than `data.shape[D] – 1` for any valid `D`.
    * `symmetric` - padded values are symmetrically added from the input `data` tensor. This method is similar to the `reflect`, but values on edges are duplicated. Refer to the examples below for more details. `pads_begin[D]` and `pads_end[D]` must be not greater than `data.shape[D]` for any valid `D`.
  * **Type**: string
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: `data` - input tensor to be padded. Required.

* **2**: `pads_begin` - specifies the number of padding elements at the beginning of each axis. A list of non-negative integers. The length of the list must be equal to the number of dimensions in the input tensor. Required.

* **3**: `pads_end` - specifies the number of padding elements at the beginning of each axis. A list of non-negative integers. The length of the list must be equal to the number of dimensions in the input tensor. Required.

* **4**: `pad_value` - scalar tensor of type matching type of elements in `data` tensor to be replicated in padded area. Used with the `pad_mode = "constant"` only. All new elements are populated with this value. Optional for `pad_mode = "constant"`. If not provided, 0 of appropriate type is used. Shouldn't be set for other `pad_mode` values.


**Outputs**

* **1**: Output padded tensor with dimensions `pads_begin[D] + data.shape[D] + pads_end[D]` for each `D` from `0` to `len(data.shape) - 1`.


**Detailed Description**

The attributes specify a number of elements to add along each axis and a rule by which new element values are generated: for example, whether they are filled with a given constant or generated based on the input tensor content.

The following examples illustrate how output tensor is generated for the *Pad* layer for a given input tensor:
```
INPUT =
[[ 1  2  3  4 ]
[  5  6  7  8 ]
[  9 10 11 12 ]]
```
with the following attributes:
```
pads_begin = [0, 1]
pads_end = [2, 3]
```
depending on the *pad_mode*.
* `pad_mode = "constant"`:
```
OUTPUT =
[[ 0  1  2  3  4  0  0  0 ]
[  0  5  6  7  8  0  0  0 ]
[  0  9 10 11 12  0  0  0 ]
[  0  0  0  0  0  0  0  0 ]
[  0  0  0  0  0  0  0  0 ]]
```
* `pad_mode = "edge"`:
```
OUTPUT =
[[ 1  1  2  3  4  4  4  4 ]
[  5  5  6  7  8  8  8  8 ]
[  9  9 10 11 12 12 12 12 ]
[  9  9 10 11 12 12 12 12 ]
[  9  9 10 11 12 12 12 12 ]]
```
* `pad_mode = "reflect"`:
```
OUTPUT =
[[ 2  1  2  3  4  3  2  1 ]
[  6  5  6  7  8  7  6  5 ]
[ 10  9 10 11 12 11 10  9 ]
[  6  5  6  7  8  7  6  5 ]
[  2  1  2  3  4  3  2  1 ]]
```
* `pad_mode = "symmetric"`:
```
OUTPUT =
[[ 1  1  2  3  4  4  3  2 ]
[  5  5  6  7  8  8  7  6 ]
[  9  9 10 11 12 12 11 10 ]
[  9  9 10 11 12 12 11 10 ]
[  5  5  6  7  8  8  7  6 ]]
```

**Example**

```xml
<layer ... type="Pad" ...>
    <data pad_mode="constant"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>32</dim>
            <dim>40</dim>
        </port>
        <port id="1">
            <dim>4</dim>     <!-- pads_begin = [0, 5, 2, 1]  -->
        </port>
        <port id="2">
            <dim>4</dim>     <!-- pads_end = [1, 0, 3, 7] -->
        </port>
        <port id="3">
                             <!-- pad_value = 15.0 -->
        </port>
    </input>
    <output>
        <port id="0">
            <dim>2</dim>     <!-- 2 = 0 + 1 + 1 = pads_begin[0] + input.shape[0] + pads_end[0] -->
            <dim>8</dim>     <!-- 8 = 5 + 3 + 0 = pads_begin[1] + input.shape[1] + pads_end[1] -->
            <dim>37</dim>    <!-- 37 = 2 + 32 + 3 = pads_begin[2] + input.shape[2] + pads_end[2] -->
            <dim>48</dim>    <!-- 48 = 1 + 40 + 7 = pads_begin[3] + input.shape[3] + pads_end[3] -->
                             <!-- all new elements are filled with 15.0 value -->
        </port>
    </output>
</layer>
```