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

from openvino.inference_engine import PreProcessInfo, IECore, TensorDesc, Blob, PreProcessChannel,\
    MeanVariant, ResizeAlgorithm, ColorFormat
from conftest import model_path


test_net_xml, test_net_bin = model_path()


def test_preprocess_info():
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    assert isinstance(net.input_info["data"].preprocess_info, PreProcessInfo)


def test_color_format():
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    preprocess_info = net.input_info["data"].preprocess_info
    assert preprocess_info.color_format == ColorFormat.RAW


def test_color_format_setter():
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    preprocess_info = net.input_info["data"].preprocess_info
    preprocess_info.color_format = ColorFormat.BGR
    assert preprocess_info.color_format == ColorFormat.BGR


def test_resize_algorithm():
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    preprocess_info = net.input_info["data"].preprocess_info
    assert preprocess_info.resize_algorithm == ResizeAlgorithm.NO_RESIZE


def test_resize_algorithm_setter():
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    preprocess_info = net.input_info["data"].preprocess_info
    preprocess_info.resize_algorithm = ResizeAlgorithm.RESIZE_BILINEAR
    assert preprocess_info.resize_algorithm == ResizeAlgorithm.RESIZE_BILINEAR


def test_mean_variant():
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    preprocess_info = net.input_info["data"].preprocess_info
    assert preprocess_info.mean_variant == MeanVariant.NONE


def test_mean_variant_setter():
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    preprocess_info = net.input_info["data"].preprocess_info
    preprocess_info.mean_variant = MeanVariant.MEAN_IMAGE
    assert preprocess_info.mean_variant == MeanVariant.MEAN_IMAGE


def test_get_number_of_channels():
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    assert net.input_info["data"].preprocess_info.get_number_of_channels() == 0


def test_init():
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    net.input_info['data'].preprocess_info.init(5)
    assert net.input_info["data"].preprocess_info.get_number_of_channels() == 5


def test_set_mean_image():
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    tensor_desc = TensorDesc("FP32", [0, 127, 127], "CHW")
    mean_image_blob = Blob(tensor_desc)
    preprocess_info = net.input_info["data"].preprocess_info
    preprocess_info.set_mean_image(mean_image_blob)
    assert preprocess_info.mean_variant == MeanVariant.MEAN_IMAGE


def test_get_pre_process_channel():
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    preprocess_info = net.input_info["data"].preprocess_info
    preprocess_info.init(1)
    pre_process_channel = preprocess_info[0]
    assert isinstance(pre_process_channel, PreProcessChannel)


def test_set_mean_image_for_channel():
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    tensor_desc = TensorDesc("FP32", [127, 127], "HW")
    mean_image_blob = Blob(tensor_desc)
    preprocess_info = net.input_info["data"].preprocess_info
    preprocess_info.init(1)
    preprocess_info.set_mean_image_for_channel(mean_image_blob, 0)
    pre_process_channel = preprocess_info[0]
    assert isinstance(pre_process_channel.mean_data, Blob)
    assert pre_process_channel.mean_data.tensor_desc.dims == [127, 127]
    assert preprocess_info.mean_variant == MeanVariant.MEAN_IMAGE


def test_resize_algorithm_set(device):
    ie_core = IECore()
    net = ie_core.read_network(model=test_net_xml, weights=test_net_bin)
    preprocess_info = net.input_info["data"].preprocess_info
    preprocess_info.resize_algorithm = ResizeAlgorithm.RESIZE_BILINEAR
    exec_net = ie_core.load_network(network=net, device_name=device, num_requests=1)
    request = exec_net.requests[0]
    pp = request.preprocess_info["data"]
    assert pp.resize_algorithm == ResizeAlgorithm.RESIZE_BILINEAR
