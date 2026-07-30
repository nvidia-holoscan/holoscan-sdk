/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PING_SIMPLE_ASYNC_BUFFER_CPP_PING_TX_ASYNC_OP_HPP
#define PING_SIMPLE_ASYNC_BUFFER_CPP_PING_TX_ASYNC_OP_HPP

#include <thread>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>

namespace holoscan::ops {

class PingTxAsyncOp : public PingTxOp {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(PingTxAsyncOp, PingTxOp)

  PingTxAsyncOp() = default;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    PingTxOp::compute(op_input, op_output, context);
    HOLOSCAN_LOG_INFO("tx");
  }
};

}  // namespace holoscan::ops

#endif /* PING_SIMPLE_ASYNC_BUFFER_CPP_PING_TX_ASYNC_OP_HPP */
