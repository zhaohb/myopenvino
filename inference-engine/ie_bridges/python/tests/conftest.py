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


def model_path(is_myriad=False):
    path_to_repo = os.environ["MODELS_PATH"]
    if not is_myriad:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.bin')
    else:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.bin')
    return (test_xml, test_bin)


def model_onnx_path():
    path_to_repo = os.environ["MODELS_PATH"]
    test_onnx = os.path.join(path_to_repo, "models", "test_model", 'test_model.onnx')
    return test_onnx

def model_prototxt_path():
    path_to_repo = os.environ["MODELS_PATH"]
    test_prototxt = os.path.join(path_to_repo, "models", "test_model", 'test_model.prototxt')
    return test_prototxt

def image_path():
    path_to_repo = os.environ["DATA_PATH"]
    path_to_img = os.path.join(path_to_repo, 'validation_set', '224x224', 'dog.bmp')
    return path_to_img


def plugins_path():
    path_to_repo = os.environ["DATA_PATH"]
    plugins_xml = os.path.join(path_to_repo, 'ie_class', 'plugins.xml')
    plugins_win_xml = os.path.join(path_to_repo, 'ie_class', 'plugins_win.xml')
    plugins_osx_xml = os.path.join(path_to_repo, 'ie_class', 'plugins_apple.xml')
    return (plugins_xml, plugins_win_xml, plugins_osx_xml)


@pytest.fixture(scope='session')
def device():
    return os.environ.get("TEST_DEVICE") if os.environ.get("TEST_DEVICE") else "CPU"
