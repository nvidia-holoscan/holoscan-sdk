/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOSCAN_OPERATORS_PING_TENSOR_RX_PING_TENSOR_RX_HPP
#define HOLOSCAN_OPERATORS_PING_TENSOR_RX_PING_TENSOR_RX_HPP

#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

/**
 * @brief Simple tensor receive operator.
 *
 * This is an example of a native operator with one input port.
 *
 * This operator is intended for use in test cases and example applications.
 *
 * On each tick, it receives a TensorMap and loops over each tensor in the map. For each, it will
 * print the tensor's name and shape.
 *
 * ==Named Inputs==
 *
 * - **in** : nvidia::gxf::Tensor(s)
 *   - One or more received tensors (i.e. a TensorMap).
 */
class PingTensorRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTensorRxOp)

  PingTensorRxOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;

 private:
  size_t count_ = 1;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_PING_TENSOR_RX_PING_TENSOR_RX_HPP */
