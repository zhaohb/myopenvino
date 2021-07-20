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
"""
Package: ngraph
Low level wrappers for the nGraph c++ api.
"""

# flake8: noqa

import os
import sys

if sys.platform == "win32":
    # ngraph.dll is installed 3 directories above by default
    # and this path needs to be visible to the _pyngraph module
    #
    ngraph_dll = os.path.join(os.path.dirname(__file__), "..", "..", "..")

    # PIP installs openvino and ngraph dlls 2 directories above in openvino.libs by default
    # and this path needs to be visible to the _pyngraph modules
    #
    openvino_dlls = os.path.join(os.path.dirname(__file__), "..", "..", "openvino", "libs")
    # If you're using a custom installation of openvino,
    # add the location of openvino dlls to your system PATH.
    if (3, 8) <= sys.version_info:
        # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
        os.add_dll_directory(os.path.abspath(ngraph_dll))
        os.add_dll_directory(os.path.abspath(openvino_dlls))
    else:
        os.environ["PATH"] = os.path.abspath(openvino_dlls) + ";" + os.path.abspath(ngraph_dll) + ";" + os.environ["PATH"]

from _pyngraph import Dimension
from _pyngraph import Function
from _pyngraph import Input
from _pyngraph import Output
from _pyngraph import Node
from _pyngraph import Type
from _pyngraph import PartialShape
from _pyngraph import Shape
from _pyngraph import Strides
from _pyngraph import CoordinateDiff
from _pyngraph import AxisSet
from _pyngraph import AxisVector
from _pyngraph import Coordinate
from _pyngraph import Output

from _pyngraph import util
