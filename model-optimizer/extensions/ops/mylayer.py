# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.extractor import bool_to_str
from mo.graph.graph import Node, Graph
from mo.middle.passes.convert_data_type import np_data_type_to_destination_type
from mo.ops.op import Op


class MyLayer(Op):
    op = 'MyLayer'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'kind': 'op',
            'type': __class__.op,
            'op': __class__.op,
            'version': 'extension',

            'type_infer': self.type_infer,
            'infer': self.infer,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        import pdb
        #pdb.set_trace()
        version = self.get_opset()
        return []

    @staticmethod
    def type_infer(node):
        # the output is always integer since the layer outputs a bucket index
        if node.get_opset() == "extension":
            node.out_port(0).set_data_type(np.float)
        else:
            assert node.output_type in [np.int64, np.int32], \
                'Bucketize `output_type` attribute must be int32 or int64, `{}` found'.format(np.dtype(node.output_type).name)
            node.out_port(0).set_data_type(node.output_type)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        import pdb
        #pdb.set_trace()
        assert len(node.in_nodes()) == 1, \
            "Incorrect number of inputs for {} node".format(node.id)
        if node.get_opset() == "extension":
            output_type = np.float
        else:
            assert node.has_valid('output_type'), \
                '`output_type` attribute is not set for Bucketize node `{}`'.format(node_name)
            assert node.output_type in [np.int64, np.int32], \
                'Bucketize `output_type` attribute must be int32 or int64, `{}` found'.format(np.dtype(node.output_type).name)
            output_type = node.output_type

        output_shape = node.in_port(0).data.get_shape()
        node.out_port(0).data.set_shape(output_shape)

