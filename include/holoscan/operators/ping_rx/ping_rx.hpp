/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_OPERATORS_PING_RX_HPP
#define HOLOSCAN_OPERATORS_PING_RX_HPP

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

/**
 * @brief Simple receiver operator.
 *
 * This is an example of a native operator with one input port.
 * On each tick, it receives an integer from the "in" port.
 *
 * ==Named Inputs==
 *
 * - **in** : any
 *   - A received value.
 */
class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_PING_RX_HPP */
