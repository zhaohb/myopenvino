# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.pair cimport pair

from ..inference_engine.ie_api_impl_defs cimport IENetwork

cdef extern from "test_utils_api_impl.hpp" namespace "InferenceEnginePython":
    cdef pair[bool, string] CompareNetworks(IENetwork lhs, IENetwork rhs)
