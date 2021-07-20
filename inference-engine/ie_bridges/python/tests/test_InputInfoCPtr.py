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

import pytest

from openvino.inference_engine import InputInfoCPtr, DataPtr, IECore, TensorDesc
from conftest import model_path


test_net_xml, test_net_bin = model_path()


def test_name(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert isinstance(exec_net.input_info['data'], InputInfoCPtr)
    assert exec_net.input_info['data'].name == "data", "Incorrect name"
    del exec_net


def test_precision(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert isinstance(exec_net.input_info['data'], InputInfoCPtr)
    assert exec_net.input_info['data'].precision == "FP32", "Incorrect precision"
    del exec_net


def test_no_precision_setter(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    with pytest.raises(AttributeError) as e:
        exec_net.input_info['data'].precision = "I8"
    assert "attribute 'precision' of 'openvino.inference_engine.ie_api.InputInfoCPtr' " \
           "objects is not writable" in str(e.value)
    del exec_net


def test_input_data(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    assert isinstance(exec_net.input_info['data'], InputInfoCPtr)
    assert isinstance(exec_net.input_info['data'].input_data, DataPtr), "Incorrect precision for layer 'fc_out'"
    del exec_net


def test_tensor_desc(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device, num_requests=5)
    tensor_desc = exec_net.input_info['data'].tensor_desc
    assert isinstance(tensor_desc, TensorDesc)
    assert tensor_desc.layout == "NCHW"
