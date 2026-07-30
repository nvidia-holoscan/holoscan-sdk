/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
