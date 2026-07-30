/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TESTS_CORE_PING_RX_OP_HPP
#define TESTS_CORE_PING_RX_OP_HPP

#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan {
namespace ops {

class PingMultiRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMultiRxOp)

  PingMultiRxOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;

 private:
  int count_ = 1;
};

}  // namespace ops
}  // namespace holoscan

#endif /* TESTS_CORE_PING_RX_OP_HPP */
