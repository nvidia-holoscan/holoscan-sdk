/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TESTS_CORE_PING_TX_OP_HPP
#define TESTS_CORE_PING_TX_OP_HPP

#include <holoscan/holoscan.hpp>

namespace holoscan {
namespace ops {

class PingMultiTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMultiTxOp)

  PingMultiTxOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;
};

}  // namespace ops
}  // namespace holoscan

#endif /* TESTS_CORE_PING_TX_OP_HPP */
