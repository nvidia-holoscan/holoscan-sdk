/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PING_RX_ASYNC_OP_HPP
#define PING_RX_ASYNC_OP_HPP

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>

namespace holoscan::ops {

class PingRxAsyncOp : public PingRxOp {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(PingRxAsyncOp, PingRxOp)

  PingRxAsyncOp() = default;

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    auto maybe_value = op_input.receive<int>("in");
    if (!maybe_value) {
      auto error_msg = fmt::format("Operator '{}' did not receive a valid value.", this->name());
      HOLOSCAN_LOG_INFO(error_msg);
      return;
    }
    int value = maybe_value.value();
    HOLOSCAN_LOG_INFO("Rx message value: {}", value);
  }
};

}  // namespace holoscan::ops

#endif /* PING_RX_ASYNC_OP_HPP */
