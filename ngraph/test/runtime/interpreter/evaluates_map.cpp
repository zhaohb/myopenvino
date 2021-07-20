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

#include "evaluates_map.hpp"

#include "backend.hpp"
#include "ngraph/ops.hpp"

#include <ngraph/runtime/reference/abs.hpp>
#include <ngraph/runtime/reference/avg_pool.hpp>
#include <ngraph/runtime/reference/batch_norm.hpp>
#include <ngraph/runtime/reference/bucketize.hpp>
#include <ngraph/runtime/reference/ceiling.hpp>
#include <ngraph/runtime/reference/convert.hpp>
#include <ngraph/runtime/reference/convolution.hpp>
#include <ngraph/runtime/reference/convolution_backprop_data.hpp>
#include <ngraph/runtime/reference/ctc_greedy_decoder.hpp>
#include <ngraph/runtime/reference/ctc_greedy_decoder_seq_len.hpp>
#include <ngraph/runtime/reference/ctc_loss.hpp>
#include <ngraph/runtime/reference/cum_sum.hpp>
#include <ngraph/runtime/reference/detection_output.hpp>
#include <ngraph/runtime/reference/elu.hpp>
#include <ngraph/runtime/reference/embedding_bag_offsets_sum.hpp>
#include <ngraph/runtime/reference/embedding_bag_packed_sum.hpp>
#include <ngraph/runtime/reference/embedding_segments_sum.hpp>
#include <ngraph/runtime/reference/extract_image_patches.hpp>
#include <ngraph/runtime/reference/fake_quantize.hpp>
#include <ngraph/runtime/reference/gather_elements.hpp>
#include <ngraph/runtime/reference/gather_nd.hpp>
#include <ngraph/runtime/reference/gather_tree.hpp>
#include <ngraph/runtime/reference/gelu.hpp>
#include <ngraph/runtime/reference/grn.hpp>
#include <ngraph/runtime/reference/group_convolution.hpp>
#include <ngraph/runtime/reference/group_convolution_backprop_data.hpp>
#include <ngraph/runtime/reference/gru_cell.hpp>
#include <ngraph/runtime/reference/hard_sigmoid.hpp>
#include <ngraph/runtime/reference/log_softmax.hpp>
#include <ngraph/runtime/reference/lrn.hpp>
#include <ngraph/runtime/reference/lstm_cell.hpp>
#include <ngraph/runtime/reference/mod.hpp>
#include <ngraph/runtime/reference/mvn.hpp>
#include <ngraph/runtime/reference/non_max_suppression.hpp>
#include <ngraph/runtime/reference/normalize_l2.hpp>
#include <ngraph/runtime/reference/one_hot.hpp>
#include <ngraph/runtime/reference/pad.hpp>
#include <ngraph/runtime/reference/prior_box.hpp>
#include <ngraph/runtime/reference/proposal.hpp>
#include <ngraph/runtime/reference/psroi_pooling.hpp>
#include <ngraph/runtime/reference/region_yolo.hpp>
#include <ngraph/runtime/reference/reorg_yolo.hpp>
#include <ngraph/runtime/reference/reverse_sequence.hpp>
#include <ngraph/runtime/reference/rnn_cell.hpp>
#include <ngraph/runtime/reference/roi_pooling.hpp>
#include <ngraph/runtime/reference/scatter_nd_update.hpp>
#include <ngraph/runtime/reference/select.hpp>
#include <ngraph/runtime/reference/selu.hpp>
#include <ngraph/runtime/reference/sequences.hpp>
#include <ngraph/runtime/reference/sign.hpp>
#include <ngraph/runtime/reference/squared_difference.hpp>
#include <ngraph/runtime/reference/tensor_iterator.hpp>

using namespace ngraph;
using namespace std;

namespace
{
    template <element::Type_t ET>
    bool evaluate(shared_ptr<Node> op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        return false;
    }

    namespace bucketize_v3
    {
        template <element::Type_t t1, element::Type_t t2, element::Type_t t3>
        inline void evaluate(const shared_ptr<op::v3::Bucketize>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            using T3 = typename element_type_traits<t3>::value_type;

            runtime::reference::bucketize<T1, T2, T3>(inputs[0]->get_data_ptr<T1>(),
                                                      inputs[1]->get_data_ptr<T2>(),
                                                      outputs[0]->get_data_ptr<T3>(),
                                                      op->get_input_shape(0),
                                                      op->get_input_shape(1),
                                                      op->get_with_right_bound());
        }

        static inline constexpr uint16_t getElementMask(element::Type_t type1,
                                                        element::Type_t type2)
        {
            return (static_cast<uint8_t>(type1)) | (static_cast<uint8_t>(type2) << 8);
        }

    } // namespace bucketize_v3

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::Bucketize>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (bucketize_v3::getElementMask(op->get_input_element_type(0),
                                             op->get_input_element_type(1)))
        {
        case bucketize_v3::getElementMask(element::Type_t::f32, element::Type_t::f32):
            bucketize_v3::evaluate<element::Type_t::f32, element::Type_t::f32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f32, element::Type_t::f16):
            bucketize_v3::evaluate<element::Type_t::f32, element::Type_t::f16, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f32, element::Type_t::i32):
            bucketize_v3::evaluate<element::Type_t::f32, element::Type_t::i32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f32, element::Type_t::i64):
            bucketize_v3::evaluate<element::Type_t::f32, element::Type_t::i64, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f16, element::Type_t::f32):
            bucketize_v3::evaluate<element::Type_t::f16, element::Type_t::f32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f16, element::Type_t::f16):
            bucketize_v3::evaluate<element::Type_t::f16, element::Type_t::f16, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f16, element::Type_t::i32):
            bucketize_v3::evaluate<element::Type_t::f16, element::Type_t::i32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f16, element::Type_t::i64):
            bucketize_v3::evaluate<element::Type_t::f32, element::Type_t::i64, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i32, element::Type_t::f32):
            bucketize_v3::evaluate<element::Type_t::i32, element::Type_t::f32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i32, element::Type_t::f16):
            bucketize_v3::evaluate<element::Type_t::i32, element::Type_t::f16, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i32, element::Type_t::i32):
            bucketize_v3::evaluate<element::Type_t::i32, element::Type_t::i32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i32, element::Type_t::i64):
            bucketize_v3::evaluate<element::Type_t::i32, element::Type_t::i64, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i64, element::Type_t::f32):
            bucketize_v3::evaluate<element::Type_t::i64, element::Type_t::f32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i64, element::Type_t::f16):
            bucketize_v3::evaluate<element::Type_t::i64, element::Type_t::f16, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i64, element::Type_t::i32):
            bucketize_v3::evaluate<element::Type_t::i64, element::Type_t::i32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i64, element::Type_t::i64):
            bucketize_v3::evaluate<element::Type_t::i64, element::Type_t::i64, ET>(
                op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::Convolution>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto& out_shape = outputs[0]->get_shape();
        const auto& in_shape = inputs[0]->get_shape();
        const auto& filter_shape = inputs[1]->get_shape();
        runtime::reference::convolution<typename element_type_traits<ET>::value_type>(
            in_data_ptr,
            filter_data,
            out_data_ptr,
            in_shape,
            filter_shape,
            out_shape,
            op->get_strides(),
            op->get_dilations(),
            op->get_pads_begin(),
            op->get_pads_end());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::ConvolutionBackpropData>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto& out_shape = outputs[0]->get_shape();
        const auto& in_shape = inputs[0]->get_shape();
        const auto& filter_shape = inputs[1]->get_shape();
        Strides in_dilation(std::vector<size_t>(in_shape.size() - 2));
        std::fill(in_dilation.begin(), in_dilation.end(), 1);
        runtime::reference::convolution_backprop_in<typename element_type_traits<ET>::value_type>(
            in_data_ptr,
            filter_data,
            out_data_ptr,
            in_shape,
            filter_shape,
            out_shape,
            in_dilation,
            op->get_dilations(),
            op->get_pads_begin(),
            op->get_pads_end(),
            op->get_strides());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::GroupConvolution>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto& out_shape = outputs[0]->get_shape();
        const auto& in_shape = inputs[0]->get_shape();
        const auto& filter_shape = inputs[1]->get_shape();
        runtime::reference::group_convolution<typename element_type_traits<ET>::value_type>(
            in_data_ptr,
            filter_data,
            out_data_ptr,
            in_shape,
            filter_shape,
            out_shape,
            op->get_strides(),
            op->get_dilations(),
            op->get_pads_begin(),
            op->get_pads_end());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::GroupConvolutionBackpropData>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto filter_data_ptr = inputs[1]->get_data_ptr<ET>();
        const auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_shape = inputs[0]->get_shape();
        const auto filter_shape = inputs[1]->get_shape();
        const auto out_shape = outputs[0]->get_shape();
        runtime::reference::group_convolution_backprop_data<
            typename element_type_traits<ET>::value_type>(in_data_ptr,
                                                          filter_data_ptr,
                                                          out_data_ptr,
                                                          in_shape,
                                                          filter_shape,
                                                          out_shape,
                                                          op->get_strides(),
                                                          op->get_dilations(),
                                                          op->get_pads_begin(),
                                                          op->get_pads_end());
        return true;
    }

    namespace cum_sum_v0
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v0::CumSum>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::cumsum<T1, T2>(inputs[0]->get_data_ptr<T1>(),
                                               inputs[1]->get_data_ptr<T2>(),
                                               outputs[0]->get_data_ptr<T1>(),
                                               inputs[0]->get_shape(),
                                               op->is_exclusive(),
                                               op->is_reverse());
        }
    } // namespace cum_sum_v0

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::CumSum>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i64:
            cum_sum_v0::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        default: cum_sum_v0::evaluate<ET, element::Type_t::i32>(op, outputs, inputs); break;
        }
        return true;
    }

    namespace embedding_offsets_sum_v3
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v3::EmbeddingSegmentsSum>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::embeddingSegmentsSum<T1, T2>(
                inputs[0]->get_data_ptr<T1>(),
                inputs[1]->get_data_ptr<T2>(),
                inputs[2]->get_data_ptr<T2>(),
                inputs.size() > 4 ? inputs[4]->get_data_ptr<T2>() : nullptr,
                inputs.size() > 5 ? inputs[5]->get_data_ptr<T1>() : nullptr,
                outputs[0]->get_data_ptr<T1>(),
                inputs[0]->get_shape(),
                inputs[1]->get_shape(),
                outputs[0]->get_shape());
        }
    } // namespace embedding_offsets_sum_v3

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::EmbeddingSegmentsSum>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i32:
            embedding_offsets_sum_v3::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        case element::Type_t::i64:
            embedding_offsets_sum_v3::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    namespace embedding_bag_offsets_sum_v3
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v3::EmbeddingBagOffsetsSum>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::embeddingBagOffsetsSum<T1, T2>(
                inputs[0]->get_data_ptr<T1>(),
                inputs[1]->get_data_ptr<T2>(),
                inputs[2]->get_data_ptr<T2>(),
                inputs.size() > 3 ? inputs[3]->get_data_ptr<T2>() : nullptr,
                inputs.size() > 4 ? inputs[4]->get_data_ptr<T1>() : nullptr,
                outputs[0]->get_data_ptr<T1>(),
                shape_size(inputs[1]->get_shape()),
                outputs[0]->get_shape());
        }
    } // namespace embedding_bag_offsets_sum_v3

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::EmbeddingBagOffsetsSum>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i32:
            embedding_bag_offsets_sum_v3::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        case element::Type_t::i64:
            embedding_bag_offsets_sum_v3::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    namespace embedding_bag_packed_sum_v3
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v3::EmbeddingBagPackedSum>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::embeddingBagPackedSum<T1, T2>(
                inputs[0]->get_data_ptr<T1>(),
                inputs[1]->get_data_ptr<T2>(),
                inputs.size() > 2 ? inputs[2]->get_data_ptr<T1>() : nullptr,
                outputs[0]->get_data_ptr<T1>(),
                inputs[1]->get_shape(),
                outputs[0]->get_shape());
        }
    } // namespace embedding_bag_packed_sum_v3

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::EmbeddingBagPackedSum>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i32:
            embedding_bag_packed_sum_v3::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        case element::Type_t::i64:
            embedding_bag_packed_sum_v3::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        default: return false;
        }

        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::MVN>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::mvn<T>(inputs[0]->get_data_ptr<ET>(),
                                   outputs[0]->get_data_ptr<ET>(),
                                   inputs[0]->get_shape(),
                                   op->get_normalize_variance(),
                                   op->get_reduction_axes(),
                                   op->get_eps());
        return true;
    }

    namespace mvn_6_axes
    {
        template <typename T>
        AxisSet mvn_6_reduction_axes(const HostTensorPtr& axes_input, size_t rank)
        {
            T* a = axes_input->get_data_ptr<T>();
            auto v = std::vector<T>(a, a + axes_input->get_shape()[0]);
            std::vector<size_t> axes(v.size(), 0);
            for (int i = 0; i < v.size(); i++)
            {
                if (v[i] < 0)
                {
                    if (rank + v[i] < 0)
                    {
                        throw ngraph_error("Unexpected axis");
                    }
                    axes[i] = (size_t)(rank + v[i]);
                }
                else
                {
                    axes[i] = (size_t)(v[i]);
                }
            }
            return AxisSet(axes);
        }
    } // mvn_6_axes

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v6::MVN>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        AxisSet reduction_axes;
        auto rank = inputs[0]->get_shape().size();
        if (inputs[1]->get_element_type() == element::i64)
        {
            reduction_axes = mvn_6_axes::mvn_6_reduction_axes<int64_t>(inputs[1], rank);
        }
        else if (inputs[1]->get_element_type() == element::i32)
        {
            reduction_axes = mvn_6_axes::mvn_6_reduction_axes<int32_t>(inputs[1], rank);
        }
        else
        {
            throw ngraph_error("Unexpected indices type");
        }
        runtime::reference::mvn_6<T>(inputs[0]->get_data_ptr<ET>(),
                                     outputs[0]->get_data_ptr<ET>(),
                                     inputs[0]->get_shape(),
                                     reduction_axes,
                                     op->get_normalize_variance(),
                                     op->get_eps(),
                                     op->get_eps_mode());
        return true;
    }

    namespace nms_v5
    {
        using V5BoxEncoding = op::v5::NonMaxSuppression::BoxEncodingType;

        struct InfoForNMS5
        {
            int64_t max_output_boxes_per_class;
            float iou_threshold;
            float score_threshold;
            float soft_nms_sigma;
            Shape out_shape;
            Shape boxes_shape;
            Shape scores_shape;
            std::vector<float> boxes_data;
            std::vector<float> scores_data;
            size_t out_shape_size;
            bool sort_result_descending;
            ngraph::element::Type output_type;
        };

        constexpr size_t boxes_port = 0;
        constexpr size_t scores_port = 1;

        PartialShape
            infer_selected_indices_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                         int64_t max_output_boxes_per_class)
        {
            const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
            const auto scores_ps = inputs[scores_port]->get_partial_shape();

            // NonMaxSuppression produces triplets
            // that have the following format: [batch_index, class_index, box_index]
            PartialShape result = {Dimension::dynamic(), 3};

            if (boxes_ps.rank().is_static() && scores_ps.rank().is_static())
            {
                const auto num_boxes_boxes = boxes_ps[1];
                if (num_boxes_boxes.is_static() && scores_ps[0].is_static() &&
                    scores_ps[1].is_static())
                {
                    const auto num_boxes = num_boxes_boxes.get_length();
                    const auto num_classes = scores_ps[1].get_length();

                    result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes *
                                scores_ps[0].get_length();
                }
            }
            return result;
        }

        std::vector<int64_t> get_integers(const std::shared_ptr<HostTensor>& input,
                                          const Shape& shape)
        {
            size_t input_size = shape_size(shape);
            std::vector<int64_t> result(input_size);

            switch (input->get_element_type())
            {
            case element::Type_t::i8:
            {
                auto p = input->get_data_ptr<int8_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::i16:
            {
                auto p = input->get_data_ptr<int16_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::i32:
            {
                auto p = input->get_data_ptr<int32_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::i64:
            {
                auto p = input->get_data_ptr<int64_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::u8:
            {
                auto p = input->get_data_ptr<uint8_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::u16:
            {
                auto p = input->get_data_ptr<uint16_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::u32:
            {
                auto p = input->get_data_ptr<uint32_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::u64:
            {
                auto p = input->get_data_ptr<uint64_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            default:
                throw std::runtime_error("Unsupported data type in op NonMaxSuppression-5");
                break;
            }

            return result;
        }

        std::vector<float> get_floats(const std::shared_ptr<HostTensor>& input, const Shape& shape)
        {
            size_t input_size = shape_size(shape);
            std::vector<float> result(input_size);

            switch (input->get_element_type())
            {
            case element::Type_t::bf16:
            {
                bfloat16* p = input->get_data_ptr<bfloat16>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = float(p[i]);
                }
            }
            break;
            case element::Type_t::f16:
            {
                float16* p = input->get_data_ptr<float16>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = float(p[i]);
                }
            }
            break;
            case element::Type_t::f32:
            {
                float* p = input->get_data_ptr<float>();
                memcpy(result.data(), p, input_size * sizeof(float));
            }
            break;
            default:
                throw std::runtime_error("Unsupported data type in op NonMaxSuppression-5");
                break;
            }

            return result;
        }

        void normalize_corner(float* boxes, const Shape& boxes_shape)
        {
            size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
            for (size_t i = 0; i < total_num_of_boxes; ++i)
            {
                float* current_box = boxes + 4 * i;

                float y1 = current_box[0];
                float x1 = current_box[1];
                float y2 = current_box[2];
                float x2 = current_box[3];

                float ymin = std::min(y1, y2);
                float ymax = std::max(y1, y2);
                float xmin = std::min(x1, x2);
                float xmax = std::max(x1, x2);

                current_box[0] = ymin;
                current_box[1] = xmin;
                current_box[2] = ymax;
                current_box[3] = xmax;
            }
        }

        void normalize_center(float* boxes, const Shape& boxes_shape)
        {
            size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
            for (size_t i = 0; i < total_num_of_boxes; ++i)
            {
                float* current_box = boxes + 4 * i;

                float x_center = current_box[0];
                float y_center = current_box[1];
                float width = current_box[2];
                float height = current_box[3];

                float y1 = y_center - height / 2.0;
                float x1 = x_center - width / 2.0;
                float y2 = y_center + height / 2.0;
                float x2 = x_center + width / 2.0;

                current_box[0] = y1;
                current_box[1] = x1;
                current_box[2] = y2;
                current_box[3] = x2;
            }
        }

        void normalize_box_encoding(float* boxes,
                                    const Shape& boxes_shape,
                                    const V5BoxEncoding box_encoding)
        {
            if (box_encoding == V5BoxEncoding::CORNER)
            {
                normalize_corner(boxes, boxes_shape);
            }
            else
            {
                normalize_center(boxes, boxes_shape);
            }
        }

        std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                              const Shape& boxes_shape,
                                              const V5BoxEncoding box_encoding)
        {
            auto result = get_floats(boxes, boxes_shape);
            normalize_box_encoding(result.data(), boxes_shape, box_encoding);
            return result;
        }

        std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores,
                                               const Shape& scores_shape)
        {
            auto result = get_floats(scores, scores_shape);
            return result;
        }

        InfoForNMS5 get_info_for_nms5_eval(const std::shared_ptr<op::v5::NonMaxSuppression>& nms5,
                                           const std::vector<std::shared_ptr<HostTensor>>& inputs)
        {
            InfoForNMS5 result;

            result.max_output_boxes_per_class =
                inputs.size() > 2 ? get_integers(inputs[2], Shape({}))[0] : 0;
            result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], Shape({}))[0] : 0.0f;
            result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], Shape({}))[0] : 0.0f;
            result.soft_nms_sigma = inputs.size() > 5 ? get_floats(inputs[5], Shape({}))[0] : 0.0f;

            auto selected_indices_shape =
                infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
            result.out_shape = selected_indices_shape.to_shape();

            result.boxes_shape = inputs[boxes_port]->get_shape();
            result.scores_shape = inputs[scores_port]->get_shape();

            result.boxes_data = prepare_boxes_data(
                inputs[boxes_port], result.boxes_shape, nms5->get_box_encoding());
            result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

            result.out_shape_size = shape_size(result.out_shape);

            result.sort_result_descending = nms5->get_sort_result_descending();

            result.output_type = nms5->get_output_type();

            return result;
        }

    } // namespace nms_v5

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::NonMaxSuppression>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        auto info = nms_v5::get_info_for_nms5_eval(op, inputs);

        std::vector<int64_t> selected_indices(info.out_shape_size);
        std::vector<float> selected_scores(info.out_shape_size);
        int64_t valid_outputs = 0;

        runtime::reference::non_max_suppression(info.boxes_data.data(),
                                                info.boxes_shape,
                                                info.scores_data.data(),
                                                info.scores_shape,
                                                info.max_output_boxes_per_class,
                                                info.iou_threshold,
                                                info.score_threshold,
                                                info.soft_nms_sigma,
                                                selected_indices.data(),
                                                info.out_shape,
                                                selected_scores.data(),
                                                info.out_shape,
                                                &valid_outputs,
                                                info.sort_result_descending);

        auto selected_scores_type =
            (inputs.size() < 4) ? element::f32 : inputs[3]->get_element_type();

        runtime::reference::nms5_postprocessing(outputs,
                                                info.output_type,
                                                selected_indices,
                                                selected_scores,
                                                valid_outputs,
                                                selected_scores_type);
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::LRN>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::lrn<T>(inputs[0]->get_data_ptr<ET>(),
                                   op->get_reduction_axes(),
                                   outputs[0]->get_data_ptr<ET>(),
                                   inputs[0]->get_shape(),
                                   op->get_alpha(),
                                   op->get_beta(),
                                   op->get_bias(),
                                   op->get_nsize());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::GRN>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::grn<T>(inputs[0]->get_data_ptr<ET>(),
                                   outputs[0]->get_data_ptr<ET>(),
                                   op->get_bias(),
                                   inputs[0]->get_shape());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::DetectionOutput>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::referenceDetectionOutput<T> refDetOut(op->get_attrs(),
                                                                  op->get_input_shape(0),
                                                                  op->get_input_shape(2),
                                                                  op->get_output_shape(0));
        if (op->get_input_size() == 3)
        {
            refDetOut.run(inputs[0]->get_data_ptr<const T>(),
                          inputs[1]->get_data_ptr<const T>(),
                          inputs[2]->get_data_ptr<const T>(),
                          nullptr,
                          nullptr,
                          outputs[0]->get_data_ptr<T>());
        }
        else if (op->get_input_size() == 5)
        {
            refDetOut.run(inputs[0]->get_data_ptr<const T>(),
                          inputs[1]->get_data_ptr<const T>(),
                          inputs[2]->get_data_ptr<const T>(),
                          inputs[3]->get_data_ptr<const T>(),
                          inputs[4]->get_data_ptr<const T>(),
                          outputs[0]->get_data_ptr<T>());
        }
        else
        {
            throw ngraph_error("DetectionOutput layer supports only 3 or 5 inputs");
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::ScatterNDUpdate>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        auto idxType = op->get_input_element_type(1);
        if (idxType == element::i32)
        {
            runtime::reference::scatterNdUpdate<T, int32_t>(
                inputs[0]->get_data_ptr<const T>(),
                inputs[1]->get_data_ptr<const int32_t>(),
                inputs[2]->get_data_ptr<const T>(),
                outputs[0]->get_data_ptr<T>(),
                op->get_input_shape(0),
                op->get_input_shape(1),
                op->get_input_shape(2));
        }
        else if (idxType == element::i64)
        {
            runtime::reference::scatterNdUpdate<T, int64_t>(
                inputs[0]->get_data_ptr<const T>(),
                inputs[1]->get_data_ptr<const int64_t>(),
                inputs[2]->get_data_ptr<const T>(),
                outputs[0]->get_data_ptr<T>(),
                op->get_input_shape(0),
                op->get_input_shape(1),
                op->get_input_shape(2));
        }
        else
        {
            throw ngraph_error(
                "ScatterNDUpdate layer support only i32 and i64 'indices' input precision!");
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::Select>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;

        runtime::reference::select<T>(inputs[0]->get_data_ptr<const char>(),
                                      inputs[1]->get_data_ptr<const T>(),
                                      inputs[2]->get_data_ptr<const T>(),
                                      outputs[0]->get_data_ptr<T>(),
                                      op->get_input_shape(0),
                                      op->get_input_shape(1),
                                      op->get_input_shape(2),
                                      op->get_auto_broadcast());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::AvgPool>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::avg_pool<T>(inputs[0]->get_data_ptr<T>(),
                                        outputs[0]->get_data_ptr<T>(),
                                        inputs[0]->get_shape(),
                                        op->get_output_shape(0),
                                        op->get_kernel(),
                                        op->get_strides(),
                                        op->get_pads_begin(),
                                        op->get_pads_end(),
                                        !op->get_exclude_pad());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::HardSigmoid>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::hard_sigmoid<T>(inputs[0]->get_data_ptr<T>(),
                                            inputs[1]->get_data_ptr<const T>()[0],
                                            inputs[2]->get_data_ptr<const T>()[0],
                                            outputs[0]->get_data_ptr<T>(),
                                            shape_size(outputs[0]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Elu>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::elu<T>(inputs[0]->get_data_ptr<T>(),
                                   outputs[0]->get_data_ptr<T>(),
                                   shape_size(inputs[0]->get_shape()),
                                   op->get_alpha());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::PriorBox>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::prior_box<T>(inputs[0]->get_data_ptr<T>(),
                                         inputs[1]->get_data_ptr<T>(),
                                         outputs[0]->get_data_ptr<float>(),
                                         outputs[0]->get_shape(),
                                         op->get_attrs());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Proposal>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::proposal_v0<T>(inputs[0]->get_data_ptr<T>(),
                                           inputs[1]->get_data_ptr<T>(),
                                           inputs[2]->get_data_ptr<T>(),
                                           outputs[0]->get_data_ptr<T>(),
                                           inputs[0]->get_shape(),
                                           inputs[1]->get_shape(),
                                           inputs[2]->get_shape(),
                                           outputs[0]->get_shape(),
                                           op.get()->get_attrs());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v4::Proposal>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::proposal_v4<T>(inputs[0]->get_data_ptr<T>(),
                                           inputs[1]->get_data_ptr<T>(),
                                           inputs[2]->get_data_ptr<T>(),
                                           outputs[0]->get_data_ptr<T>(),
                                           outputs[1]->get_data_ptr<T>(),
                                           inputs[0]->get_shape(),
                                           inputs[1]->get_shape(),
                                           inputs[2]->get_shape(),
                                           outputs[0]->get_shape(),
                                           outputs[1]->get_shape(),
                                           op.get()->get_attrs());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::Mod>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::mod<T>(inputs[0]->get_data_ptr<T>(),
                                   inputs[1]->get_data_ptr<T>(),
                                   outputs[0]->get_data_ptr<T>(),
                                   inputs[0]->get_shape(),
                                   inputs[1]->get_shape(),
                                   op->get_auto_broadcast());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Selu>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::selu<T>(inputs[0]->get_data_ptr<T>(),
                                    inputs[1]->get_data_ptr<T>(),
                                    inputs[2]->get_data_ptr<T>(),
                                    outputs[0]->get_data_ptr<T>(),
                                    shape_size(inputs[0]->get_shape()),
                                    shape_size(inputs[1]->get_shape()),
                                    shape_size(inputs[2]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Ceiling>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::ceiling<T>(inputs[0]->get_data_ptr<T>(),
                                       outputs[0]->get_data_ptr<T>(),
                                       shape_size(inputs[0]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Gelu>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::gelu<T>(inputs[0]->get_data_ptr<T>(),
                                    outputs[0]->get_data_ptr<T>(),
                                    shape_size(inputs[0]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Relu>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::relu<T>(inputs[0]->get_data_ptr<T>(),
                                    outputs[0]->get_data_ptr<T>(),
                                    shape_size(inputs[0]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Sign>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::sign<T>(inputs[0]->get_data_ptr<T>(),
                                    outputs[0]->get_data_ptr<T>(),
                                    shape_size(inputs[0]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Abs>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::abs<T>(inputs[0]->get_data_ptr<T>(),
                                   outputs[0]->get_data_ptr<T>(),
                                   shape_size(inputs[0]->get_shape()));
        return true;
    }

    namespace ctc_loss_v4
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v4::CTCLoss>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::CTCLoss<T1, T2>(inputs[0]->get_data_ptr<T1>(),
                                                inputs[0]->get_shape(),
                                                inputs[1]->get_data_ptr<T2>(),
                                                inputs[2]->get_data_ptr<T2>(),
                                                inputs[3]->get_data_ptr<T2>(),
                                                inputs[4]->get_data_ptr<T2>(),
                                                op->get_preprocess_collapse_repeated(),
                                                op->get_ctc_merge_repeated(),
                                                op->get_unique(),
                                                outputs[0]->get_data_ptr<T1>());
        }
    } // namespace ctc_loss_v4

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v4::CTCLoss>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i32:
            ctc_loss_v4::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        case element::Type_t::i64:
            ctc_loss_v4::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::BatchNormInference>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::batch_norm_inference<T>(op->get_eps_value(),
                                                    inputs[0]->get_data_ptr<T>(),
                                                    inputs[1]->get_data_ptr<T>(),
                                                    inputs[2]->get_data_ptr<T>(),
                                                    inputs[3]->get_data_ptr<T>(),
                                                    inputs[4]->get_data_ptr<T>(),
                                                    outputs[0]->get_data_ptr<T>(),
                                                    inputs[2]->get_shape());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::BatchNormInference>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::batch_norm_inference<T>(op->get_eps_value(),
                                                    inputs[1]->get_data_ptr<const T>(),
                                                    inputs[2]->get_data_ptr<const T>(),
                                                    inputs[0]->get_data_ptr<const T>(),
                                                    inputs[3]->get_data_ptr<const T>(),
                                                    inputs[4]->get_data_ptr<const T>(),
                                                    outputs[0]->get_data_ptr<T>(),
                                                    op->get_input_shape(0));
        return true;
    }

    namespace reverse_sequence_v0
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v0::ReverseSequence>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::reverse_sequence<T1, T2>(inputs[0]->get_data_ptr<T1>(),
                                                         outputs[0]->get_data_ptr<T1>(),
                                                         inputs[0]->get_shape(),
                                                         op->get_batch_axis(),
                                                         op->get_sequence_axis(),
                                                         inputs[1]->get_data_ptr<T2>());
        }
    } // namespace reverse_sequence_v0

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::ReverseSequence>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::boolean:
            reverse_sequence_v0::evaluate<ET, element::Type_t::boolean>(op, outputs, inputs);
            break;
        case element::Type_t::i8:
            reverse_sequence_v0::evaluate<ET, element::Type_t::i8>(op, outputs, inputs);
            break;
        case element::Type_t::i16:
            reverse_sequence_v0::evaluate<ET, element::Type_t::i16>(op, outputs, inputs);
            break;
        case element::Type_t::i32:
            reverse_sequence_v0::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        case element::Type_t::i64:
            reverse_sequence_v0::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        case element::Type_t::u8:
            reverse_sequence_v0::evaluate<ET, element::Type_t::u8>(op, outputs, inputs);
            break;
        case element::Type_t::u16:
            reverse_sequence_v0::evaluate<ET, element::Type_t::u16>(op, outputs, inputs);
            break;
        case element::Type_t::u32:
            reverse_sequence_v0::evaluate<ET, element::Type_t::u32>(op, outputs, inputs);
            break;
        case element::Type_t::u64:
            reverse_sequence_v0::evaluate<ET, element::Type_t::u64>(op, outputs, inputs);
            break;
        case element::Type_t::f16:
            reverse_sequence_v0::evaluate<ET, element::Type_t::f16>(op, outputs, inputs);
            break;
        case element::Type_t::f32:
            reverse_sequence_v0::evaluate<ET, element::Type_t::f32>(op, outputs, inputs);
            break;
        case element::Type_t::f64:
            reverse_sequence_v0::evaluate<ET, element::Type_t::f64>(op, outputs, inputs);
            break;
        default: return false;
        }
#undef REF_CALL
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::ExtractImagePatches>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::extract_image_patches<T>(op,
                                                     inputs[0]->get_data_ptr<T>(),
                                                     outputs[0]->get_data_ptr<T>(),
                                                     inputs[0]->get_shape(),
                                                     outputs[0]->get_shape());
        return true;
    }

    namespace convert_v0
    {
        template <element::Type_t ti, element::Type_t to>
        inline void evaluate(const shared_ptr<op::v0::Convert>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using TI = typename element_type_traits<ti>::value_type;
            using TO = typename element_type_traits<to>::value_type;
            runtime::reference::convert<TI, TO>(inputs[0]->get_data_ptr<TI>(),
                                                outputs[0]->get_data_ptr<TO>(),
                                                shape_size(inputs[0]->get_shape()));
        }
    } // namespace convert_v0

    template <element::Type_t OUT_ET>
    bool evaluate(const shared_ptr<op::v0::Convert>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[0]->get_element_type())
        {
        case element::Type_t::boolean:
            convert_v0::evaluate<element::Type_t::boolean, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::i8:
            convert_v0::evaluate<element::Type_t::i8, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::i16:
            convert_v0::evaluate<element::Type_t::i16, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::i32:
            convert_v0::evaluate<element::Type_t::i32, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::i64:
            convert_v0::evaluate<element::Type_t::i64, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::u8:
            convert_v0::evaluate<element::Type_t::u8, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::u16:
            convert_v0::evaluate<element::Type_t::u16, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::u32:
            convert_v0::evaluate<element::Type_t::u32, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::u64:
            convert_v0::evaluate<element::Type_t::u64, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::f16:
            convert_v0::evaluate<element::Type_t::f16, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::f32:
            convert_v0::evaluate<element::Type_t::f32, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::f64:
            convert_v0::evaluate<element::Type_t::f64, OUT_ET>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    namespace convert_like_v1
    {
        template <element::Type_t ti, element::Type_t to>
        inline void evaluate(const shared_ptr<op::v1::ConvertLike>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using TI = typename element_type_traits<ti>::value_type;
            using TO = typename element_type_traits<to>::value_type;
            runtime::reference::convert<TI, TO>(inputs[0]->get_data_ptr<TI>(),
                                                outputs[0]->get_data_ptr<TO>(),
                                                shape_size(inputs[0]->get_shape()));
        }

    } // namespace convert_like_v1

    template <element::Type_t OUT_ET>
    bool evaluate(const shared_ptr<op::v1::ConvertLike>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[0]->get_element_type())
        {
        case element::Type_t::boolean:
            convert_like_v1::evaluate<element::Type_t::boolean, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::u8:
            convert_like_v1::evaluate<element::Type_t::u8, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::u16:
            convert_like_v1::evaluate<element::Type_t::u16, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::u32:
            convert_like_v1::evaluate<element::Type_t::u32, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::u64:
            convert_like_v1::evaluate<element::Type_t::u64, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::i8:
            convert_like_v1::evaluate<element::Type_t::i8, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::i16:
            convert_like_v1::evaluate<element::Type_t::i16, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::i32:
            convert_like_v1::evaluate<element::Type_t::i32, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::i64:
            convert_like_v1::evaluate<element::Type_t::i64, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::bf16:
            convert_like_v1::evaluate<element::Type_t::bf16, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::f16:
            convert_like_v1::evaluate<element::Type_t::f16, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::f32:
            convert_like_v1::evaluate<element::Type_t::f32, OUT_ET>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::OneHot>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        switch (inputs[0]->get_element_type())
        {
        case element::Type_t::i32:
            runtime::reference::
                one_hot<typename element_type_traits<element::Type_t::i32>::value_type, T>(
                    inputs[0]->get_data_ptr<element::Type_t::i32>(),
                    outputs[0]->get_data_ptr<T>(),
                    inputs[0]->get_shape(),
                    outputs[0]->get_shape(),
                    op->get_axis(),
                    inputs[2]->get_data_ptr<T>()[0],
                    inputs[3]->get_data_ptr<T>()[0]);
            break;
        case element::Type_t::i64:
            runtime::reference::
                one_hot<typename element_type_traits<element::Type_t::i64>::value_type, T>(
                    inputs[0]->get_data_ptr<element::Type_t::i64>(),
                    outputs[0]->get_data_ptr<T>(),
                    inputs[0]->get_shape(),
                    outputs[0]->get_shape(),
                    op->get_axis(),
                    inputs[2]->get_data_ptr<T>()[0],
                    inputs[3]->get_data_ptr<T>()[0]);
            break;
        default:
            std::stringstream ss;
            ss << "Unhandled input precision " << inputs[0]->get_element_type().get_type_name()
               << " in v1::OneHot evaluate call";
            throw ngraph_error(ss.str());
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::RNNCell>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::rnn_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                        inputs[0]->get_shape(),
                                        inputs[1]->get_data_ptr<ET>(),
                                        inputs[1]->get_shape(),
                                        inputs[2]->get_data_ptr<ET>(),
                                        inputs[2]->get_shape(),
                                        inputs[3]->get_data_ptr<ET>(),
                                        inputs[3]->get_shape(),
                                        inputs[4]->get_data_ptr<ET>(),
                                        inputs[4]->get_shape(),
                                        outputs[0]->get_data_ptr<ET>(),
                                        op->get_activations().front(),
                                        op->get_clip());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v4::LSTMCell>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::lstm_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                         inputs[0]->get_shape(),
                                         inputs[1]->get_data_ptr<ET>(),
                                         inputs[1]->get_shape(),
                                         inputs[2]->get_data_ptr<ET>(),
                                         inputs[2]->get_shape(),
                                         inputs[3]->get_data_ptr<ET>(),
                                         inputs[3]->get_shape(),
                                         inputs[4]->get_data_ptr<ET>(),
                                         inputs[4]->get_shape(),
                                         inputs[5]->get_data_ptr<ET>(),
                                         inputs[5]->get_shape(),
                                         outputs[0]->get_data_ptr<ET>(),
                                         outputs[1]->get_data_ptr<ET>(),
                                         op->get_activations()[0],
                                         op->get_activations()[1],
                                         op->get_activations()[2],
                                         op->get_clip());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::GRUCell>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::gru_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                        inputs[0]->get_shape(),
                                        inputs[1]->get_data_ptr<ET>(),
                                        inputs[1]->get_shape(),
                                        inputs[2]->get_data_ptr<ET>(),
                                        inputs[2]->get_shape(),
                                        inputs[3]->get_data_ptr<ET>(),
                                        inputs[3]->get_shape(),
                                        inputs[4]->get_data_ptr<ET>(),
                                        inputs[4]->get_shape(),
                                        outputs[0]->get_data_ptr<ET>(),
                                        op->get_activations()[0],
                                        op->get_activations()[1],
                                        op->get_clip(),
                                        op->get_linear_before_reset());
        return true;
    }

    namespace rnn_seq_v5
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v5::RNNSequence>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::rnn_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
                                                     inputs[0]->get_shape(),
                                                     inputs[1]->get_data_ptr<char>(),
                                                     inputs[1]->get_shape(),
                                                     inputs[2]->get_data_ptr<char>(),
                                                     inputs[2]->get_shape(),
                                                     inputs[3]->get_data_ptr<char>(),
                                                     inputs[3]->get_shape(),
                                                     inputs[4]->get_data_ptr<char>(),
                                                     inputs[4]->get_shape(),
                                                     inputs[5]->get_data_ptr<char>(),
                                                     inputs[5]->get_shape(),
                                                     outputs[0]->get_data_ptr<char>(),
                                                     outputs[1]->get_data_ptr<char>(),
                                                     op->get_activations()[0],
                                                     op->get_clip(),
                                                     op->get_direction());
        }
    } // namespace rnn_seq_v5

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::RNNSequence>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[2]->get_element_type())
        {
        case element::Type_t::i64:
        case element::Type_t::u64:
            rnn_seq_v5::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        case element::Type_t::i32:
        case element::Type_t::u32:
            rnn_seq_v5::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    namespace lstm_seq_v5
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v5::LSTMSequence>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::lstm_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
                                                      inputs[0]->get_shape(),
                                                      inputs[1]->get_data_ptr<char>(),
                                                      inputs[1]->get_shape(),
                                                      inputs[2]->get_data_ptr<char>(),
                                                      inputs[2]->get_shape(),
                                                      inputs[3]->get_data_ptr<char>(),
                                                      inputs[3]->get_shape(),
                                                      inputs[4]->get_data_ptr<char>(),
                                                      inputs[4]->get_shape(),
                                                      inputs[5]->get_data_ptr<char>(),
                                                      inputs[5]->get_shape(),
                                                      inputs[6]->get_data_ptr<char>(),
                                                      inputs[6]->get_shape(),
                                                      outputs[0]->get_data_ptr<char>(),
                                                      outputs[1]->get_data_ptr<char>(),
                                                      outputs[2]->get_data_ptr<char>(),
                                                      op->get_activations()[0],
                                                      op->get_activations()[1],
                                                      op->get_activations()[2],
                                                      op->get_clip(),
                                                      op->get_direction());
        }
    } // namespace lstm_seq_v5

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::LSTMSequence>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[3]->get_element_type())
        {
        case element::Type_t::i64:
        case element::Type_t::u64:
            lstm_seq_v5::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        case element::Type_t::i32:
        case element::Type_t::u32:
            lstm_seq_v5::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    namespace ti_v0
    {
        runtime::reference::custom_evaluate_function evaluate =
            [](const std::shared_ptr<ngraph::Function>& function,
               const HostTensorVector& inputs,
               HostTensorVector& outputs) -> void {
            const auto& parameters = function->get_parameters();
            const auto& parametersNumber = parameters.size();
            const auto& inputsNumber = inputs.size();
            NGRAPH_CHECK(parametersNumber == inputsNumber,
                         "Got function (",
                         function->get_friendly_name(),
                         ") with ",
                         parametersNumber,
                         " parameters, but ",
                         inputsNumber,
                         " input blobs");

            auto inputTensors = std::vector<std::shared_ptr<runtime::Tensor>>{};
            for (const auto& parameter : parameters)
            {
                const auto& parameterIndex = function->get_parameter_index(parameter);
                const auto& parameterShape = parameter->get_shape();
                const auto& parameterType = parameter->get_element_type();
                const auto& parameterSize = shape_size(parameterShape) * parameterType.size();

                const auto& input = inputs[parameterIndex];
                const auto& inputSize = input->get_size_in_bytes();
                NGRAPH_CHECK(parameterSize == inputSize,
                             "Got parameter (",
                             parameter->get_friendly_name(),
                             ") of size ",
                             parameterSize,
                             " bytes, but corresponding input with index ",
                             parameterIndex,
                             " has ",
                             inputSize,
                             " bytes");

                auto tensor = std::make_shared<runtime::HostTensor>(parameterType, parameterShape);
                tensor->write(input->get_data_ptr(), parameterSize);
                inputTensors.push_back(tensor);
            }

            const auto& results = function->get_results();
            std::vector<std::shared_ptr<ngraph::runtime::Tensor>> outputTensors;
            outputTensors.reserve(results.size());
            for (size_t i = 0; i < results.size(); ++i)
            {
                outputTensors.push_back(std::make_shared<HostTensor>());
            }
            runtime::Backend::set_backend_shared_library_search_directory("");
            auto backend = runtime::Backend::create("INTERPRETER");
            auto handle = backend->compile(function);
            handle->call_with_validate(outputTensors, inputTensors);

            outputs.reserve(outputTensors.size());
            for (const auto& tensor : outputTensors)
            {
                auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
                outputs.push_back(host_tensor);
            }
        };
    } // namespace ti_v0

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::TensorIterator>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        runtime::reference::tensor_iterator(op->get_num_iterations(),
                                            op->get_function(),
                                            op->get_output_descriptions(),
                                            op->get_input_descriptions(),
                                            outputs,
                                            inputs,
                                            ti_v0::evaluate);
        return true;
    }

    namespace gru_seq_v5
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v5::GRUSequence>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::gru_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
                                                     inputs[0]->get_shape(),
                                                     inputs[1]->get_data_ptr<char>(),
                                                     inputs[1]->get_shape(),
                                                     inputs[2]->get_data_ptr<char>(),
                                                     inputs[2]->get_shape(),
                                                     inputs[3]->get_data_ptr<char>(),
                                                     inputs[3]->get_shape(),
                                                     inputs[4]->get_data_ptr<char>(),
                                                     inputs[4]->get_shape(),
                                                     inputs[5]->get_data_ptr<char>(),
                                                     inputs[5]->get_shape(),
                                                     outputs[0]->get_data_ptr<char>(),
                                                     outputs[1]->get_data_ptr<char>(),
                                                     op->get_activations()[0],
                                                     op->get_activations()[1],
                                                     op->get_clip(),
                                                     op->get_direction(),
                                                     op->get_linear_before_reset());
        }
    } // namespace gru_seq_v5

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::GRUSequence>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[2]->get_element_type())
        {
        case element::Type_t::i64:
        case element::Type_t::u64:
            gru_seq_v5::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        case element::Type_t::i32:
        case element::Type_t::u32:
            gru_seq_v5::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }
    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::ROIPooling>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::roi_pooling<T>(inputs[0]->get_data_ptr<const T>(),
                                           inputs[1]->get_data_ptr<const T>(),
                                           outputs[0]->get_data_ptr<T>(),
                                           op->get_input_shape(0),
                                           op->get_input_shape(1),
                                           op->get_output_shape(0),
                                           op->get_spatial_scale(),
                                           op->get_method());
        return true;
    }
    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::ReorgYolo>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        runtime::reference::reorg_yolo(inputs[0]->get_data_ptr<char>(),
                                       outputs[0]->get_data_ptr<char>(),
                                       inputs[0]->get_shape(),
                                       op->get_strides().at(0),
                                       inputs[0]->get_element_type().size());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::RegionYolo>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::region_yolo<T>(inputs[0]->get_data_ptr<const T>(),
                                           outputs[0]->get_data_ptr<T>(),
                                           inputs[0]->get_shape(),
                                           op->get_num_coords(),
                                           op->get_num_classes(),
                                           op->get_num_regions(),
                                           op->get_do_softmax(),
                                           op->get_mask());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::Pad>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::pad(inputs[0]->get_data_ptr<char>(),
                                inputs[1]->get_data_ptr<char>(),
                                outputs[0]->get_data_ptr<char>(),
                                shape_size(inputs[0]->get_shape()),
                                inputs[1]->get_shape(),
                                outputs[0]->get_shape(),
                                op->get_pads_end(),
                                op->get_pads_begin(),
                                op->get_pad_mode());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::GatherTree>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::gather_tree(inputs[0]->get_data_ptr<const char>(),
                                        inputs[1]->get_data_ptr<const char>(),
                                        inputs[2]->get_data_ptr<const char>(),
                                        inputs[3]->get_data_ptr<const char>(),
                                        outputs[0]->get_data_ptr<char>(),
                                        op->get_input_shape(0),
                                        op->get_input_shape(1),
                                        op->get_input_shape(2),
                                        op->get_input_shape(3),
                                        inputs[1]->get_element_type());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::FakeQuantize>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::fake_quantize<T>(inputs[0]->get_data_ptr<const T>(),
                                             inputs[1]->get_data_ptr<const T>(),
                                             inputs[2]->get_data_ptr<const T>(),
                                             inputs[3]->get_data_ptr<const T>(),
                                             inputs[4]->get_data_ptr<const T>(),
                                             outputs[0]->get_data_ptr<T>(),
                                             op->get_input_shape(0),
                                             op->get_input_shape(1),
                                             op->get_input_shape(2),
                                             op->get_input_shape(3),
                                             op->get_input_shape(4),
                                             op->get_levels());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::NormalizeL2>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::normalize_l2<T>(inputs[0]->get_data_ptr<const T>(),
                                            outputs[0]->get_data_ptr<T>(),
                                            op->get_input_shape(0),
                                            op->get_reduction_axes(),
                                            op->get_eps(),
                                            op->get_eps_mode());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::CTCGreedyDecoder>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::ctc_greedy_decoder<T>(inputs[0]->get_data_ptr<const T>(),
                                                  inputs[1]->get_data_ptr<const T>(),
                                                  outputs[0]->get_data_ptr<T>(),
                                                  inputs[0]->get_shape(),
                                                  inputs[1]->get_shape(),
                                                  outputs[0]->get_shape(),
                                                  op->get_ctc_merge_repeated());
        return true;
    }

    namespace ctc_greedy_decoder_v6
    {
        template <element::Type_t T1, element::Type_t T2, element::Type_t TOUT>
        inline void evaluate(const shared_ptr<op::v6::CTCGreedyDecoderSeqLen>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using TF = typename element_type_traits<T1>::value_type;
            using TI = typename element_type_traits<T2>::value_type;
            using TIND1 = typename element_type_traits<TOUT>::value_type;
            if (op->get_sequence_length_type() == element::i32)
            {
                runtime::reference::ctc_greedy_decoder_seq_len<TF>(
                    inputs[0]->get_data_ptr<const TF>(),
                    inputs[1]->get_data_ptr<const TI>(),
                    inputs[2]->get_data_ptr<const TI>(),
                    outputs[0]->get_data_ptr<TIND1>(),
                    outputs[1]->get_data_ptr<int32_t>(),
                    inputs[0]->get_shape(),
                    outputs[0]->get_shape(),
                    op->get_merge_repeated());
            }
            else if (op->get_sequence_length_type() == element::i64)
            {
                runtime::reference::ctc_greedy_decoder_seq_len<TF>(
                    inputs[0]->get_data_ptr<const TF>(),
                    inputs[1]->get_data_ptr<const TI>(),
                    inputs[2]->get_data_ptr<const TI>(),
                    outputs[0]->get_data_ptr<TIND1>(),
                    outputs[1]->get_data_ptr<int64_t>(),
                    inputs[0]->get_shape(),
                    outputs[0]->get_shape(),
                    op->get_merge_repeated());
            }
        }
    }
    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v6::CTCGreedyDecoderSeqLen>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto& dataType = inputs[0]->get_element_type();
        const auto& seqLenType = inputs[1]->get_element_type();
        if (dataType == element::Type_t::f16 && seqLenType == element::Type_t::i32)
        {
            ctc_greedy_decoder_v6::evaluate<element::Type_t::f16, element::Type_t::i32, ET>(
                op, outputs, inputs);
        }
        else if (dataType == element::Type_t::f32 && seqLenType == element::Type_t::i32)
        {
            ctc_greedy_decoder_v6::evaluate<element::Type_t::f32, element::Type_t::i32, ET>(
                op, outputs, inputs);
        }
        else if (dataType == element::Type_t::f64 && seqLenType == element::Type_t::i32)
        {
            ctc_greedy_decoder_v6::evaluate<element::Type_t::f64, element::Type_t::i32, ET>(
                op, outputs, inputs);
        }
        else if (dataType == element::Type_t::f16 && seqLenType == element::Type_t::i64)
        {
            ctc_greedy_decoder_v6::evaluate<element::Type_t::f16, element::Type_t::i64, ET>(
                op, outputs, inputs);
        }
        else if (dataType == element::Type_t::f32 && seqLenType == element::Type_t::i64)
        {
            ctc_greedy_decoder_v6::evaluate<element::Type_t::f32, element::Type_t::i64, ET>(
                op, outputs, inputs);
        }
        else if (dataType == element::Type_t::f64 && seqLenType == element::Type_t::i64)
        {
            ctc_greedy_decoder_v6::evaluate<element::Type_t::f64, element::Type_t::i64, ET>(
                op, outputs, inputs);
        }
        else
        {
            return false;
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::SquaredDifference>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::squared_difference<T>(inputs[0]->get_data_ptr<const T>(),
                                                  inputs[1]->get_data_ptr<const T>(),
                                                  outputs[0]->get_data_ptr<T>(),
                                                  inputs[0]->get_shape(),
                                                  inputs[1]->get_shape(),
                                                  ngraph::op::AutoBroadcastSpec::NUMPY);
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v6::GatherElements>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        Shape params_shape = inputs[0]->get_shape();
        Shape indices_shape = inputs[1]->get_shape();

        outputs[0]->set_shape(indices_shape);

        if (inputs[1]->get_element_type() == element::i64)
        {
            runtime::reference::gather_elements<T, int64_t>(inputs[0]->get_data_ptr<ET>(),
                                                            inputs[1]->get_data_ptr<int64_t>(),
                                                            outputs[0]->get_data_ptr<ET>(),
                                                            inputs[0]->get_shape(),
                                                            inputs[1]->get_shape(),
                                                            outputs[0]->get_shape(),
                                                            op->get_axis());
        }
        else if (inputs[1]->get_element_type() == element::i32)
        {
            runtime::reference::gather_elements<T, int32_t>(inputs[0]->get_data_ptr<ET>(),
                                                            inputs[1]->get_data_ptr<int32_t>(),
                                                            outputs[0]->get_data_ptr<ET>(),
                                                            inputs[0]->get_shape(),
                                                            inputs[1]->get_shape(),
                                                            outputs[0]->get_shape(),
                                                            op->get_axis());
        }
        else
        {
            throw ngraph_error("Unexpected indices type");
        }

        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::GatherND>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        if (op->get_input_element_type(1) == element::i64)
        {
            runtime::reference::gather_nd<T, int64_t>(inputs[0]->get_data_ptr<T>(),
                                                      inputs[1]->get_data_ptr<int64_t>(),
                                                      outputs[0]->get_data_ptr<T>(),
                                                      op->get_input_shape(0),
                                                      op->get_input_shape(1),
                                                      op->get_output_shape(0),
                                                      op->get_batch_dims());
        }
        else if (op->get_input_element_type(1) == element::i32)
        {
            runtime::reference::gather_nd<T, int32_t>(inputs[0]->get_data_ptr<T>(),
                                                      inputs[1]->get_data_ptr<int32_t>(),
                                                      outputs[0]->get_data_ptr<T>(),
                                                      op->get_input_shape(0),
                                                      op->get_input_shape(1),
                                                      op->get_output_shape(0),
                                                      op->get_batch_dims());
        }
        else
        {
            throw ngraph_error("Unexpected indices type for GatherND operation");
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::LogSoftmax>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        int64_t i_axis = op->get_axis();
        if (i_axis < 0)
        {
            i_axis += inputs[0]->get_partial_shape().rank().get_length();
        }
        runtime::reference::log_softmax<T>(inputs[0]->get_data_ptr<const T>(),
                                           outputs[0]->get_data_ptr<T>(),
                                           op->get_output_shape(0),
                                           AxisSet{(size_t)i_axis});
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::PSROIPooling>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::psroi_pooling<T>(inputs[0]->get_data_ptr<T>(),
                                             inputs[0]->get_shape(),
                                             inputs[1]->get_data_ptr<T>(),
                                             inputs[1]->get_shape(),
                                             outputs[0]->get_data_ptr<T>(),
                                             outputs[0]->get_shape(),
                                             op->get_mode(),
                                             op->get_spatial_scale(),
                                             op->get_spatial_bins_x(),
                                             op->get_spatial_bins_y());

        return true;
    }

    template <typename T>
    bool evaluate_node(std::shared_ptr<Node> node,
                       const HostTensorVector& outputs,
                       const HostTensorVector& inputs)
    {
        auto element_type = node->get_output_element_type(0);
        if (is_type<op::v1::Select>(node))
        {
            element_type = node->get_input_element_type(1);
        }
        else if (is_type<op::v0::PriorBox>(node))
        {
            element_type = node->get_input_element_type(0);
        }
        for (size_t i = 1; i < node->outputs().size(); i++)
        {
            if (is_type<op::v5::NonMaxSuppression>(node) && i == 1)
            {
                continue;
            }
            if (element_type != node->get_output_element_type(i))
            {
                throw std::logic_error("Output node element types is not equal");
            }
        }
        switch (element_type)
        {
        case element::Type_t::boolean:
            return evaluate<element::Type_t::boolean>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::bf16:
            return evaluate<element::Type_t::bf16>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::f16:
            return evaluate<element::Type_t::f16>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::f64:
            return evaluate<element::Type_t::f64>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::f32:
            return evaluate<element::Type_t::f32>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::i8:
            return evaluate<element::Type_t::i8>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::i16:
            return evaluate<element::Type_t::i16>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::i32:
            return evaluate<element::Type_t::i32>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::i64:
            return evaluate<element::Type_t::i64>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::u8:
            return evaluate<element::Type_t::u8>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::u16:
            return evaluate<element::Type_t::u16>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::u32:
            return evaluate<element::Type_t::u32>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::u64:
            return evaluate<element::Type_t::u64>(as_type_ptr<T>(node), outputs, inputs);
        default:
            throw ngraph_error(std::string("Unhandled data type ") +
                               node->get_element_type().get_type_name() +
                               std::string("in evaluate_node()"));
        }
    }
} // namespace

runtime::interpreter::EvaluatorsMap& runtime::interpreter::get_evaluators_map()
{
    static runtime::interpreter::EvaluatorsMap evaluatorsMap{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, evaluate_node<NAMESPACE::NAME>},

#include "opset_int_tbl.hpp"

#undef NGRAPH_OP
    };
    return evaluatorsMap;
}
