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
from sys import platform
import numpy as np
from pathlib import Path

from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork
from conftest import model_path, plugins_path, model_onnx_path, model_prototxt_path


test_net_xml, test_net_bin = model_path()
test_net_onnx = model_onnx_path()
test_net_prototxt = model_prototxt_path()
plugins_xml, plugins_win_xml, plugins_osx_xml = plugins_path()


def test_init_ie_core_no_cfg():
    ie = IECore()
    assert isinstance(ie, IECore)


def test_init_ie_core_with_cfg():
    ie = IECore(plugins_xml)
    assert isinstance(ie, IECore)


def test_get_version(device):
    ie = IECore()
    version = ie.get_versions(device)
    assert isinstance(version, dict), "Returned version must be a dictionary"
    assert device in version, "{} plugin version wasn't found in versions"
    assert hasattr(version[device], "major"), "Returned version has no field 'major'"
    assert hasattr(version[device], "minor"), "Returned version has no field 'minor'"
    assert hasattr(version[device], "description"), "Returned version has no field 'description'"
    assert hasattr(version[device], "build_number"), "Returned version has no field 'build_number'"


def test_load_network(device):
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, device)
    assert isinstance(exec_net, ExecutableNetwork)


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device independent test")
def test_load_network_wrong_device():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    with pytest.raises(RuntimeError) as e:
        ie.load_network(net, "BLA")
    assert 'Device with "BLA" name is not registered in the InferenceEngine' in str(e.value)


def test_query_network(device):
    import ngraph as ng
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    query_res = ie.query_network(net, device)
    func_net = ng.function_from_cnn(net)
    ops_net = func_net.get_ordered_ops()
    ops_net_names = [op.friendly_name for op in ops_net]
    assert [key for key in query_res.keys() if key not in ops_net_names] == [], \
        "Not all network layers present in query_network results"
    assert next(iter(set(query_res.values()))) == device, "Wrong device for some layers"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device independent test")
def test_register_plugin():
    ie = IECore()
    ie.register_plugin("MKLDNNPlugin", "BLA")
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, "BLA")
    assert isinstance(exec_net, ExecutableNetwork), "Cannot load the network to the registered plugin with name 'BLA'"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU", reason="Device independent test")
def test_register_plugins():
    ie = IECore()
    if platform == "linux" or platform == "linux2":
        ie.register_plugins(plugins_xml)
    elif platform == "darwin":
        ie.register_plugins(plugins_osx_xml)
    elif platform == "win32":
        ie.register_plugins(plugins_win_xml)

    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    exec_net = ie.load_network(net, "CUSTOM")
    assert isinstance(exec_net,
                      ExecutableNetwork), "Cannot load the network to the registered plugin with name 'CUSTOM' " \
                                          "registred in the XML file"


@pytest.mark.skip(reason="Need to figure out if it's expected behaviour (fails with C++ API as well")
def test_unregister_plugin(device):
    ie = IECore()
    ie.unregister_plugin(device)
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    with pytest.raises(RuntimeError) as e:
        ie.load_network(net, device)
    assert f"Device with '{device}' name is not registered in the InferenceEngine" in str(e.value)


def test_available_devices(device):
    ie = IECore()
    devices = ie.available_devices
    assert device in devices, f"Current device '{device}' is not listed in available devices '{', '.join(devices)}'"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_list_of_str():
    ie = IECore()
    param = ie.get_metric("CPU", "OPTIMIZATION_CAPABILITIES")
    assert isinstance(param, list), "Parameter value for 'OPTIMIZATION_CAPABILITIES' " \
                                    f"metric must be a list but {type(param)} is returned"
    assert all(isinstance(v, str) for v in param), "Not all of the parameter values for 'OPTIMIZATION_CAPABILITIES' " \
                                                   "metric are strings!"



@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_tuple_of_two_ints():
    ie = IECore()
    param = ie.get_metric("CPU", "RANGE_FOR_STREAMS")
    assert isinstance(param, tuple), "Parameter value for 'RANGE_FOR_STREAMS' " \
                                     f"metric must be tuple but {type(param)} is returned"
    assert all(isinstance(v, int) for v in param), "Not all of the parameter values for 'RANGE_FOR_STREAMS' " \
                                                   "metric are integers!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_tuple_of_three_ints():
    ie = IECore()
    param = ie.get_metric("CPU", "RANGE_FOR_ASYNC_INFER_REQUESTS")
    assert isinstance(param, tuple), "Parameter value for 'RANGE_FOR_ASYNC_INFER_REQUESTS' " \
                                     f"metric must be tuple but {type(param)} is returned"
    assert all(isinstance(v, int) for v in param), "Not all of the parameter values for " \
                                                   "'RANGE_FOR_ASYNC_INFER_REQUESTS' metric are integers!"


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Cannot run test on device {os.environ.get('TEST_DEVICE')}, Plugin specific test")
def test_get_metric_str():
    ie = IECore()
    param = ie.get_metric("CPU", "FULL_DEVICE_NAME")
    assert isinstance(param, str), "Parameter value for 'FULL_DEVICE_NAME' " \
                                   f"metric must be string but {type(param)} is returned"


def test_read_network_from_xml():
    ie = IECore()
    net = ie.read_network(model=test_net_xml, weights=test_net_bin)
    assert isinstance(net, IENetwork)

    net = ie.read_network(model=test_net_xml)
    assert isinstance(net, IENetwork)


def test_read_network_as_path():
    ie = IECore()

    net = ie.read_network(model=Path(test_net_xml), weights=test_net_bin)
    assert isinstance(net, IENetwork)

    net = ie.read_network(model=test_net_xml, weights=Path(test_net_bin))
    assert isinstance(net, IENetwork)

    net = ie.read_network(model=Path(test_net_xml))
    assert isinstance(net, IENetwork)


def test_read_network_from_onnx():
    ie = IECore()
    net = ie.read_network(model=test_net_onnx)
    assert isinstance(net, IENetwork)

def test_read_network_from_onnx_as_path():
    ie = IECore()
    net = ie.read_network(model=Path(test_net_onnx))
    assert isinstance(net, IENetwork)

def test_read_network_from_prototxt():
    ie = IECore()
    net = ie.read_network(model=test_net_prototxt)
    assert isinstance(net, IENetwork)

def test_read_network_from_prototxt_as_path():
    ie = IECore()
    net = ie.read_network(model=Path(test_net_prototxt))
    assert isinstance(net, IENetwork)

def test_incorrect_xml():
    ie = IECore()
    with pytest.raises(Exception) as e:
        ie.read_network(model="./model.xml", weights=Path(test_net_bin))
    assert "Path to the model ./model.xml doesn't exist or it's a directory" in str(e.value)


def test_incorrect_bin():
    ie = IECore()
    with pytest.raises(Exception) as e:
        ie.read_network(model=test_net_xml, weights="./model.bin")
    assert "Path to the weights ./model.bin doesn't exist or it's a directory" in str(e.value)


def test_read_net_from_buffer():
    ie = IECore()
    with open(test_net_bin, 'rb') as f:
        bin = f.read()
    with open(model_path()[0], 'rb') as f:
        xml = f.read()
    net = ie.read_network(model=xml, weights=bin, init_from_buffer=True)
    assert isinstance(net, IENetwork)


def test_net_from_buffer_valid():
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
