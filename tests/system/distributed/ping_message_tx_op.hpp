/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TESTS_CORE_PING_MESSAGE_TX_OP_HPP
#define TESTS_CORE_PING_MESSAGE_TX_OP_HPP

#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "ping_message_rx_op.hpp"  // MessageType

namespace holoscan {
namespace ops {

class PingMessageTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMessageTxOp)

  PingMessageTxOp() = default;

  void initialize() override;

  void set_message_types(std::vector<MessageType> types) { types_ = std::move(types); }

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;

 private:
  std::vector<MessageType> types_ = {MessageType::FLOAT};
  size_t current_index_ = 0;
};
}  // namespace ops
}  // namespace holoscan

#endif /* TESTS_CORE_PING_MESSAGE_TX_OP_HPP */
