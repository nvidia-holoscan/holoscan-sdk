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

#ifndef HOLOSCAN_OPERATORS_PING_TX_HPP
#define HOLOSCAN_OPERATORS_PING_TX_HPP

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

/**
 * @brief Simple transmitter operator.
 *
 * On each tick, it transmits an integer to the "out" port.
 *
 * ==Named Outputs==
 *
 * - **out** : int
 *   - An index value that increments by one on each call to `compute`. The starting value is 1.
 */
class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

  int index() const { return index_; }

 private:
  int index_ = 1;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_PING_TX_HPP */
