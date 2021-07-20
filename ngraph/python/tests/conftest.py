# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import os
import pytest

import tests

from pathlib import Path


def _get_default_model_zoo_dir():
    return Path(os.getenv("ONNX_HOME", Path.home() / ".onnx/model_zoo"))


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default="CPU",
        choices=["CPU", "GPU", "FPGA", "HDDL", "MYRIAD", "HETERO"],
        help="Select target device",
    )
    parser.addoption(
        "--model_zoo_dir",
        default=_get_default_model_zoo_dir(),
        type=str,
        help="location of the model zoo",
    )
    parser.addoption(
        "--model_zoo_xfail",
        action="store_true",
        help="treat model zoo known issues as xfails instead of failures",
    )


def pytest_configure(config):
    backend_name = config.getvalue("backend")
    tests.BACKEND_NAME = backend_name
    tests.MODEL_ZOO_DIR = Path(config.getvalue("model_zoo_dir"))
    tests.MODEL_ZOO_XFAIL = config.getvalue("model_zoo_xfail")

    # register additional markers
    config.addinivalue_line("markers", "skip_on_cpu: Skip test on CPU")
    config.addinivalue_line("markers", "skip_on_gpu: Skip test on GPU")
    config.addinivalue_line("markers", "skip_on_fpga: Skip test on FPGA")
    config.addinivalue_line("markers", "skip_on_hddl: Skip test on HDDL")
    config.addinivalue_line("markers", "skip_on_myriad: Skip test on MYRIAD")
    config.addinivalue_line("markers", "skip_on_hetero: Skip test on HETERO")
    config.addinivalue_line("markers", "onnx_coverage: Collect ONNX operator coverage")


def pytest_collection_modifyitems(config, items):
    backend_name = config.getvalue("backend")
    tests.MODEL_ZOO_DIR = Path(config.getvalue("model_zoo_dir"))
    tests.MODEL_ZOO_XFAIL = config.getvalue("model_zoo_xfail")

    keywords = {
        "CPU": "skip_on_cpu",
        "GPU": "skip_on_gpu",
        "FPGA": "skip_on_fpga",
        "HDDL": "skip_on_hddl",
        "MYRIAD": "skip_on_myriad",
        "HETERO": "skip_on_hetero",
    }

    skip_markers = {
        "CPU": pytest.mark.skip(reason="Skipping test on the CPU backend."),
        "GPU": pytest.mark.skip(reason="Skipping test on the GPU backend."),
        "FPGA": pytest.mark.skip(reason="Skipping test on the FPGA backend."),
        "HDDL": pytest.mark.skip(reason="Skipping test on the HDDL backend."),
        "MYRIAD": pytest.mark.skip(reason="Skipping test on the MYRIAD backend."),
        "HETERO": pytest.mark.skip(reason="Skipping test on the HETERO backend."),
    }

    for item in items:
        skip_this_backend = keywords[backend_name]
        if skip_this_backend in item.keywords:
            item.add_marker(skip_markers[backend_name])
