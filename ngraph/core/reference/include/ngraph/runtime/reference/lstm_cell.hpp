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

#include <cmath>
#include <ngraph/runtime/reference/add.hpp>
#include <ngraph/runtime/reference/clamp.hpp>
#include <ngraph/runtime/reference/matmul.hpp>
#include <ngraph/runtime/reference/multiply.hpp>
#include <ngraph/runtime/reference/relu.hpp>
#include <ngraph/runtime/reference/sigmoid.hpp>
#include <ngraph/runtime/reference/split.hpp>
#include <ngraph/runtime/reference/subtract.hpp>
#include <ngraph/runtime/reference/tanh.hpp>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void lstm_cell(const T* X,
                           const Shape& X_shape,
                           const T* H,
                           const Shape& H_shape,
                           const T* C,
                           const Shape& C_shape,
                           const T* W,
                           const Shape& W_shape,
                           const T* R,
                           const Shape& R_shape,
                           const T* B,
                           const Shape& B_shape,
                           T* out_Ht,
                           T* out_Ct,
                           const std::string& activation_f,
                           const std::string& activation_g,
                           const std::string& activation_h,
                           float clip)
            {
                // ------ VARIABLE'S NAMES AND ACRONYM DEFINITIONS ------
                // The names used below are analogous to the one used in ONNX documentation.
                //
                // ------ ACRONYMS ------
                // i - input gate
                // o - output gate
                // f - forget gate
                // c - cell gate
                // t - time step (t-1 means previous time step)
                // Wb - W bias vectors for input, output, forget, and cell gates.
                // Rb - R bias vectors for input, output, forget, and cell gates.
                // P  - The peephole weights for input, output and forget gates.
                // ------ VARIABLE NAMES ------
                // X       - The input data tensor. Shape: [batch_size, input_size].
                // W       - The weight matrix for input, forget, cell and output gates
                //           Shape: [4*hidden_size, input_size]
                // R       - The recurrence weight matrix for input, forget, cell and output gates.
                //           Shape: [4*hidden_size, hidden_size].
                // H_t     - The hidden state tensor at current time step. Shape: [batch_size,
                // hidden_size].
                // C_t     - The cell state tensor at current time step. Shape: [batch_size,
                // hidden_size].
                // bias    - The sum of biases (weight and recurrence) for input, forget, cell and
                // output gates.
                //           Shape: [4 * hidden_size]
                // p_[iof] - The peephole weight vector for respectively: input, output, and forget
                // gates.
                //           Each peephole has shape [hidden_size].
                //
                // (.) - Denotes element-wise multiplication.
                // *   - Denotes dot product.
                //
                // ---- Equations ----
                // f, g, h - are activation functions.
                // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
                // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
                // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
                // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
                // Ct = ft (.) Ct-1 + it (.) ct
                // Ht = ot (.) h(Ct)
                // --------------------
                Shape gate_shape{X_shape[0], H_shape[1]};
                Shape all_gates_shape{X_shape[0], 4 * H_shape[1]};
                auto gate_shape_size = X_shape[0] * H_shape[1];
                auto all_gates_shape_size = gate_shape_size * 4;
                // Xt*(W^T)
                std::vector<T> Xt_W(all_gates_shape_size);
                reference::matmul(
                    X, W, Xt_W.data(), X_shape, W_shape, all_gates_shape, false, true);

                // Ht-1*(R^T)
                std::vector<T> Ht_R(all_gates_shape_size);
                reference::matmul(
                    H, R, Ht_R.data(), H_shape, R_shape, all_gates_shape, false, true);

                // Ht-1*(R^T) + Wb + Rb
                std::vector<T> Ht_R_B(all_gates_shape_size);
                reference::add(Ht_R.data(),
                               B,
                               Ht_R_B.data(),
                               all_gates_shape,
                               B_shape,
                               op::AutoBroadcastSpec::NUMPY);

                // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
                std::vector<T> XHB(all_gates_shape_size);
                reference::add(Xt_W.data(),
                               Ht_R_B.data(),
                               XHB.data(),
                               all_gates_shape,
                               all_gates_shape,
                               op::AutoBroadcastSpec::NUMPY);

                std::vector<std::vector<T>> X_W_fico(4, std::vector<T>(all_gates_shape_size / 4));
                std::vector<char*> pointers = {reinterpret_cast<char*>(X_W_fico[0].data()),
                                               reinterpret_cast<char*>(X_W_fico[1].data()),
                                               reinterpret_cast<char*>(X_W_fico[2].data()),
                                               reinterpret_cast<char*>(X_W_fico[3].data())};
                // split on gates
                reference::split(reinterpret_cast<char*>(XHB.data()),
                                 all_gates_shape,
                                 sizeof(T),
                                 1,
                                 4,
                                 pointers.data());

                auto clip_activation = [&clip](
                    std::vector<T>& gate, const std::string& activation, bool enable_clip = true) {
                    if (clip > 0.f && enable_clip)
                    {
                        reference::clamp(gate.data(),
                                         gate.data(),
                                         static_cast<T>(-clip),
                                         static_cast<T>(clip),
                                         gate.size());
                    }
                    if (activation == "relu")
                    {
                        reference::relu(gate.data(), gate.data(), gate.size());
                    }
                    else if (activation == "sigmoid")
                    {
                        reference::sigmoid(gate.data(), gate.data(), gate.size());
                    }
                    else if (activation == "tanh")
                    {
                        reference::tanh(gate.data(), gate.data(), gate.size());
                    }
                    else
                    {
                        throw ngraph_error("Activation function " + activation +
                                           " is not supported.");
                    }
                };

                // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
                clip_activation(X_W_fico[0], activation_f);
                // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
                clip_activation(X_W_fico[1], activation_f);
                // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
                clip_activation(X_W_fico[2], activation_g);
                // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
                clip_activation(X_W_fico[3], activation_f);

                vector<T> mul1(gate_shape_size);
                vector<T> mul2(gate_shape_size);
                vector<T> Ct(gate_shape_size);
                // ft (.) Ct-1
                reference::multiply(X_W_fico[0].data(),
                                    C,
                                    mul1.data(),
                                    gate_shape,
                                    C_shape,
                                    op::AutoBroadcastSpec::NUMPY);
                // it (.) ct
                reference::multiply(X_W_fico[1].data(),
                                    X_W_fico[2].data(),
                                    mul2.data(),
                                    gate_shape,
                                    gate_shape,
                                    op::AutoBroadcastSpec::NUMPY);
                // Ct = ft (.) Ct-1 + it (.) ct
                reference::add(mul1.data(),
                               mul2.data(),
                               Ct.data(),
                               gate_shape,
                               gate_shape,
                               op::AutoBroadcastSpec::NUMPY);
                std::memcpy(out_Ct, Ct.data(), Ct.size() * sizeof(T));
                clip_activation(Ct, activation_h, false);

                // Ht = ot (.) h(Ct)
                reference::multiply(X_W_fico[3].data(),
                                    Ct.data(),
                                    out_Ht,
                                    gate_shape,
                                    gate_shape,
                                    op::AutoBroadcastSpec::NUMPY);
            }
        }
    }
}
