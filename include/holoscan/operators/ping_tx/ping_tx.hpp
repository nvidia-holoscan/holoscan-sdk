/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
