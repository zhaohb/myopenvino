"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the 'License');
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an 'AS IS' BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


import os
import sys

if sys.platform == "win32":
    # PIP installs openvino dlls 3 directories above in openvino.libs by default
    # and this path needs to be visible to the openvino modules
    #
    # If you're using a custom installation of openvino,
    # add the location of openvino dlls to your system PATH.
    openvino_dlls = os.path.join(os.path.dirname(__file__), "..", "..", "openvino", "libs")
    if (3, 8) <= sys.version_info:
        # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
        os.add_dll_directory(os.path.abspath(openvino_dlls))
    else:
        os.environ["PATH"] = os.path.abspath(openvino_dlls) + ";" + os.environ["PATH"]

from .ie_api import *
__all__ = ['IENetwork', "TensorDesc", "IECore", "Blob", "PreProcessInfo", "get_version"]
__version__ = get_version()

