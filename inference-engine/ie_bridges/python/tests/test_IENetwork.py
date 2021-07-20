"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import pytest
import warnings
import numpy as np

from openvino.inference_engine import IECore, IENetwork, DataPtr, InputInfoPtr, PreProcessInfo
from conftest import model_path


test_net_xml, test_net_bin = model_path()


def test_create_ie_network_deprecated():
    with warnings.catch_warnings(record=True) as w:
        net = IENetwork(model=test_net_xml, weights=test_net_bin)
        assert isinstance(net, IENetwork)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "Reading network using constructor is deprecated. " \
               "Please, use IECore.read_network() method instead" in str(w[0].message)


def test_incorrect_xml_deprecated():
    with warnings.catch_warnings(record=True) as w:
        with pytest.raises(Exception) as e:
            IENetwork(model="./model.xml", weights=test_net_bin)
        assert "Path to the model ./model.xml doesn't exist or it's a directory" in str(e.value)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "Reading network using constructor is deprecated. " \
               "Please, use IECore.read_network() method instead" in str(w[0].message)


def test_incorrect_bin_deprecated():
    with warnings.catch_warnings(record=True) as w:
        with pytest.raises(Exception) as e:
            IENetwork(model=test_net_xml, weights="./model.bin")
        assert "Path to the weights ./model.bin doesn't exist or it's a directory" in str(e.value)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "Reading network using constructor is deprecated. " \
               "Please, use IECore.read_network() method instead" in str(w[0].message)


def test_name():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.name == "test_model"


def test_inputs_deprecated():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    with warnings.catch_warnings(record=True) as w:
        inp = net.inputs
        assert isinstance(inp['data'], DataPtr)
        assert inp['data'].layout == "NCHW"
        assert inp['data'].precision == "FP32"
        assert inp['data'].shape == [1, 3, 32, 32]
    assert len(w) == 1
    assert "'inputs' property of IENetwork class is deprecated. " \
               "To access DataPtrs user need to use 'input_data' property " \
               "of InputInfoPtr objects which " \
               "can be accessed by 'input_info' property." in str(w[-1].message)


def test_input_info():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert isinstance(net.input_info['data'], InputInfoPtr)
    assert net.input_info['data'].layout == "NCHW"
    assert net.input_info['data'].precision == "FP32"
    assert isinstance(net.input_info['data'].input_data, DataPtr)
    assert isinstance(net.input_info['data'].preprocess_info, PreProcessInfo)


def test_input_info_precision_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.input_info['data'].layout == "NCHW"
    net.input_info['data'].layout = "NHWC"
    assert net.input_info['data'].layout == "NHWC"


def test_input_input_info_layout_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.input_info['data'].precision == "FP32"
    net.input_info['data'].precision = "I8"
    assert net.input_info['data'].precision == "I8"


def test_input_unsupported_precision_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    with pytest.raises(ValueError) as e:
        net.input_info['data'].precision = "BLA"
    assert "Unsupported precision BLA! List of supported precisions: " in str(e.value)


def test_input_unsupported_layout_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    with pytest.raises(ValueError) as e:
        net.input_info['data'].layout = "BLA"
    assert "Unsupported layout BLA! List of supported layouts: " in str(e.value)


def test_outputs():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert isinstance(net.outputs['fc_out'], DataPtr)
    assert net.outputs['fc_out'].layout == "NC"
    assert net.outputs['fc_out'].precision == "FP32"
    assert net.outputs['fc_out'].shape == [1, 10]


def test_output_precision_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.outputs['fc_out'].precision == "FP32"
    net.outputs['fc_out'].precision = "I8"
    assert net.outputs['fc_out'].precision == "I8"


def test_output_unsupported_precision_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    with pytest.raises(ValueError) as e:
        net.outputs['fc_out'].precision = "BLA"
    assert "Unsupported precision BLA! List of supported precisions: " in str(e.value)


def test_add_ouputs():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.add_outputs('28/Reshape')
    net.add_outputs(['29/WithoutBiases'])
    assert sorted(net.outputs) == ['28/Reshape', '29/WithoutBiases', 'fc_out']


def test_add_outputs_with_port():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.add_outputs(('28/Reshape', 0))
    net.add_outputs([('29/WithoutBiases', 0)])
    assert sorted(net.outputs) == ['28/Reshape', '29/WithoutBiases', 'fc_out']


def test_add_outputs_with_and_without_port():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.add_outputs('28/Reshape')
    net.add_outputs([('29/WithoutBiases', 0)])
    assert sorted(net.outputs) == ['28/Reshape', '29/WithoutBiases', 'fc_out']


def test_batch_size_getter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.batch_size == 1


def test_batch_size_setter():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.batch_size = 4
    assert net.batch_size == 4
    assert net.input_info['data'].input_data.shape == [4, 3, 32, 32]


def test_batch_size_after_reshape():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.reshape({'data': [4, 3, 32, 32]})
    assert net.batch_size == 4
    assert net.input_info['data'].input_data.shape == [4, 3, 32, 32]
    net.reshape({'data': [8, 3, 32, 32]})
    assert net.batch_size == 8
    assert net.input_info['data'].input_data.shape == [8, 3, 32, 32]


def test_serialize():
    import ngraph as ng
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.serialize("./serialized_net.xml", "./serialized_net.bin")
    serialized_net = ie.read_network(model="./serialized_net.xml", weights="./serialized_net.bin")
    func_net = ng.function_from_cnn(net)
    ops_net = func_net.get_ordered_ops()
    ops_net_names = [op.friendly_name for op in ops_net]
    func_serialized_net = ng.function_from_cnn(serialized_net)
    ops_serialized_net = func_serialized_net.get_ordered_ops()
    ops_serialized_net_names = [op.friendly_name for op in ops_serialized_net]
    assert ops_serialized_net_names == ops_net_names
    os.remove("./serialized_net.xml")
    os.remove("./serialized_net.bin")


def test_reshape():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.reshape({"data": (2, 3, 32, 32)})


def test_read_net_from_buffer_deprecated():
    with warnings.catch_warnings(record=True) as w:
        with open(test_net_bin, 'rb') as f:
            bin = f.read()
        with open(test_net_xml, 'rb') as f:
            xml = f.read()
        net = IENetwork(model=xml, weights=bin, init_from_buffer=True)
        assert isinstance(net, IENetwork)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "Reading network using constructor is deprecated. " \
               "Please, use IECore.read_network() method instead" in str(w[0].message)


def test_net_from_buffer_valid_deprecated():
    ie = IECore()
    with open(test_net_bin, 'rb') as f:
        bin = f.read()
    with open(model_path()[0], 'rb') as f:
        xml = f.read()
    net = ie.read_network(model=xml, weights=bin, init_from_buffer=True)
    ref_net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.name == ref_net.name
    assert net.batch_size == ref_net.batch_size
    ii_net = net.input_info
    ii_net2 = ref_net.input_info
    o_net = net.outputs
    o_net2 = ref_net.outputs
    assert ii_net.keys() == ii_net2.keys()
    assert o_net.keys() == o_net2.keys()


def test_multi_out_data():
    # Regression test 23965
    # Check that DataPtr for all output layers not copied between outputs map  items
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    net.add_outputs(['28/Reshape'])
    assert "28/Reshape" in net.outputs and "fc_out" in net.outputs
    assert isinstance(net.outputs["28/Reshape"], DataPtr)
    assert isinstance(net.outputs["fc_out"], DataPtr)
    assert net.outputs["28/Reshape"].name == "28/Reshape" and net.outputs["28/Reshape"].shape == [1, 5184]
    assert net.outputs["fc_out"].name == "fc_out" and net.outputs["fc_out"].shape == [1, 10]
    pass

def test_tensor_names():
    model = """
            <net name="Network" version="10">
                <layers>
                    <layer name="in1" type="Parameter" id="0" version="opset1">
                        <data element_type="f32" shape="1,3,22,22"/>
                        <output>
                            <port id="0" precision="FP32" names="input">
                                <dim>1</dim>
                                <dim>3</dim>
                                <dim>22</dim>
                                <dim>22</dim>
                            </port>
                        </output>
                    </layer>
                    <layer name="activation" id="1" type="ReLU" version="opset1">
                        <input>
                            <port id="1" precision="FP32">
                                <dim>1</dim>
                                <dim>3</dim>
                                <dim>22</dim>
                                <dim>22</dim>
                            </port>
                        </input>
                        <output>
                            <port id="2" precision="FP32" names="relu_t, identity_t">
                                <dim>1</dim>
                                <dim>3</dim>
                                <dim>22</dim>
                                <dim>22</dim>
                            </port>
                        </output>
                    </layer>
                    <layer name="output" type="Result" id="2" version="opset1">
                        <input>
                            <port id="0" precision="FP32">
                                <dim>1</dim>
                                <dim>3</dim>
                                <dim>22</dim>
                                <dim>22</dim>
                            </port>
                        </input>
                    </layer>
                </layers>
                <edges>
                    <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
                    <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
                </edges>
            </net>
            """
    ie = IECore()
    weights = b''
    net = ie.read_network(model=model.encode('utf-8'), weights=weights, init_from_buffer=True)
    assert net.get_ov_name_for_tensor("relu_t") == "activation"
    assert net.get_ov_name_for_tensor("identity_t") == "activation"
    assert net.get_ov_name_for_tensor("input") == "in1"
