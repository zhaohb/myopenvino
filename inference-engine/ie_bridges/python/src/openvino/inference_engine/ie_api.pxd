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


from .cimport ie_api_impl_defs as C
from .ie_api_impl_defs cimport CBlob, CTensorDesc, InputInfo, CPreProcessChannel, CPreProcessInfo

import os

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr

cdef class Blob:
    cdef CBlob.Ptr _ptr
    cdef public object _array_data
    cdef public object _initial_shape

cdef class BlobBuffer:
    cdef CBlob.Ptr ptr
    cdef char*format
    cdef vector[Py_ssize_t] shape
    cdef vector[Py_ssize_t] strides
    cdef reset(self, CBlob.Ptr &, vector[size_t] representation_shape = ?)
    cdef char*_get_blob_format(self, const CTensorDesc & desc)

    cdef public:
        total_stride, item_size

cdef class InferRequest:
    cdef C.InferRequestWrap *impl

    cpdef BlobBuffer _get_blob_buffer(self, const string & blob_name)

    cpdef infer(self, inputs = ?)
    cpdef async_infer(self, inputs = ?)
    cpdef wait(self, timeout = ?)
    cpdef get_perf_counts(self)
    cdef void user_callback(self, int status) with gil
    cdef public:
        _inputs_list, _outputs_list, _py_callback, _py_data, _py_callback_used, _py_callback_called, _user_blobs

cdef class IENetwork:
    cdef C.IENetwork impl

cdef class ExecutableNetwork:
    cdef unique_ptr[C.IEExecNetwork] impl
    cdef C.IECore ie_core_impl
    cpdef wait(self, num_requests = ?, timeout = ?)
    cpdef get_idle_request_id(self)
    cdef public:
        _requests, _infer_requests

cdef class IECore:
    cdef C.IECore impl
    cpdef IENetwork read_network(self, model : [str, bytes, os.PathLike], weights : [str, bytes, os.PathLike] = ?, bool init_from_buffer = ?)
    cpdef ExecutableNetwork load_network(self, IENetwork network, str device_name, config = ?, int num_requests = ?)
    cpdef ExecutableNetwork import_network(self, str model_file, str device_name, config = ?, int num_requests = ?)


cdef class DataPtr:
    cdef C.DataPtr _ptr
    cdef C.IENetwork * _ptr_network

cdef class CDataPtr:
    cdef C.CDataPtr _ptr

cdef class TensorDesc:
    cdef C.CTensorDesc impl

cdef class InputInfoPtr:
    cdef InputInfo.Ptr _ptr
    cdef C.IENetwork * _ptr_network

cdef class InputInfoCPtr:
    cdef InputInfo.CPtr _ptr

cdef class PreProcessInfo:
    cdef CPreProcessInfo* _ptr
    cpdef object _user_data

cdef class PreProcessChannel:
    cdef CPreProcessChannel.Ptr _ptr
