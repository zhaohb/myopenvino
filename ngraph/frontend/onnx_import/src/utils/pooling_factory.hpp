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
#pragma once

#include <memory>
#include <type_traits>

#include "ngraph/node.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace pooling
        {
            ///
            /// \brief      Factory class which generates sub-graphs for ONNX 'regular' pooling
            ///             operators.
            ///
            /// \note       This factory is intended for creating pooling operations like:
            ///             - AveragePool
            ///             - MaxPool
            ///
            ///             This class holds all common attributes like srides, dilations,
            ///             paddings, kernel shape and auto_pad type.
            class PoolingFactory
            {
            public:
                explicit PoolingFactory(const Node& node);
                virtual ~PoolingFactory() = default;

                ///
                /// \brief      Creates average pooling ONNX operation.
                /// \return     Vector of output nodes.
                ///
                OutputVector make_avg_pool() const;

                ///
                /// \brief      Creates max pooling ONNX operation.
                /// \return     Vector of output nodes.
                ///
                OutputVector make_max_pool() const;

            protected:
                Node m_onnx_node;
                const OutputVector m_inputs;
                Shape m_kernel_shape;
                Strides m_strides;
                Strides m_dilations;
                Shape m_padding_below;
                Shape m_padding_above;
                ngraph::op::PadType m_auto_pad;
                ngraph::op::RoundingType m_rounding_type;
            };
        } // namespace pooling
    }     // namespace onnx_import
} // namespace ngraph
