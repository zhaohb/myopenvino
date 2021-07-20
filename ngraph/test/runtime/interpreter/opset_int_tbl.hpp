//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#ifndef NGRAPH_OP
#warning "NGRAPH_OP not defined"
#define NGRAPH_OP(x, y)
#endif

NGRAPH_OP(Abs, op::v0)
NGRAPH_OP(BatchNormInference, op::v0)
NGRAPH_OP(Ceiling, op::v0)
NGRAPH_OP(Convert, op::v0)
NGRAPH_OP(CTCGreedyDecoder, op::v0)
NGRAPH_OP(CumSum, ngraph::op::v0)
NGRAPH_OP(DetectionOutput, op::v0)
NGRAPH_OP(Elu, op::v0)
NGRAPH_OP(FakeQuantize, op::v0)
NGRAPH_OP(Gelu, op::v0)
NGRAPH_OP(GRN, op::v0)
NGRAPH_OP(HardSigmoid, op::v0)
NGRAPH_OP(LRN, ngraph::op::v0)
NGRAPH_OP(MVN, ngraph::op::v0)
NGRAPH_OP(NormalizeL2, op::v0)
NGRAPH_OP(PriorBox, ngraph::op::v0)
NGRAPH_OP(Proposal, ngraph::op::v0)
NGRAPH_OP(PSROIPooling, op::v0)
NGRAPH_OP(RegionYolo, op::v0)
NGRAPH_OP(Relu, op::v0)
NGRAPH_OP(ReorgYolo, op::v0)
NGRAPH_OP(ReverseSequence, op::v0)
NGRAPH_OP(RNNCell, op::v0)
NGRAPH_OP(Selu, op::v0)
NGRAPH_OP(Sign, op::v0)
NGRAPH_OP(SquaredDifference, op::v0)
NGRAPH_OP(TensorIterator, op::v0)
NGRAPH_OP(ROIPooling, op::v0)

NGRAPH_OP(AvgPool, op::v1)
NGRAPH_OP(ConvertLike, op::v1)
NGRAPH_OP(Convolution, ngraph::op::v1)
NGRAPH_OP(ConvolutionBackpropData, ngraph::op::v1)
NGRAPH_OP(GroupConvolution, ngraph::op::v1)
NGRAPH_OP(GroupConvolutionBackpropData, ngraph::op::v1)
NGRAPH_OP(LessEqual, op::v1)
NGRAPH_OP(LogicalAnd, op::v1)
NGRAPH_OP(LogicalOr, op::v1)
NGRAPH_OP(LogicalXor, op::v1)
NGRAPH_OP(LogicalNot, op::v1)
NGRAPH_OP(MaxPool, op::v1)
NGRAPH_OP(Mod, op::v1)
NGRAPH_OP(OneHot, op::v1)
NGRAPH_OP(Pad, op::v1)
NGRAPH_OP(Split, op::v1)
NGRAPH_OP(Reshape, op::v1)
NGRAPH_OP(Select, op::v1)
NGRAPH_OP(GatherTree, op::v1)

NGRAPH_OP(Bucketize, op::v3)
NGRAPH_OP(EmbeddingBagOffsetsSum, ngraph::op::v3)
NGRAPH_OP(EmbeddingBagPackedSum, ngraph::op::v3)
NGRAPH_OP(ExtractImagePatches, op::v3)
NGRAPH_OP(EmbeddingSegmentsSum, ngraph::op::v3)
NGRAPH_OP(GRUCell, ngraph::op::v3)
NGRAPH_OP(NonZero, op::v3)
NGRAPH_OP(ScatterNDUpdate, op::v3)
NGRAPH_OP(ShapeOf, op::v3)

NGRAPH_OP(CTCLoss, op::v4)
NGRAPH_OP(LSTMCell, op::v4)
NGRAPH_OP(Proposal, op::v4)

NGRAPH_OP(BatchNormInference, op::v5)
NGRAPH_OP(GatherND, op::v5)
NGRAPH_OP(GRUSequence, op::v5)
NGRAPH_OP(LogSoftmax, op::v5)
NGRAPH_OP(LSTMSequence, op::v5)
NGRAPH_OP(Loop, op::v5)
NGRAPH_OP(LSTMSequence, op::v5)
NGRAPH_OP(NonMaxSuppression, op::v5)
NGRAPH_OP(RNNSequence, op::v5)
NGRAPH_OP(Round, op::v5)

NGRAPH_OP(CTCGreedyDecoderSeqLen, op::v6)
NGRAPH_OP(GatherElements, op::v6)
NGRAPH_OP(MVN, ngraph::op::v6)
