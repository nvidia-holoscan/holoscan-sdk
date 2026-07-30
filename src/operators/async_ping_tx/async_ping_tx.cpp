/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/operators/async_ping_tx/async_ping_tx.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

namespace holoscan::ops {

void AsyncPingTxOp::async_ping() {
  HOLOSCAN_LOG_INFO("Async ping tx thread entering.");
  while (true) {
    if (should_stop_) {
      HOLOSCAN_LOG_INFO("Async ping tx thread exiting.");
      return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(delay_.get()));

    if (async_condition()->event_state() == AsynchronousEventState::EVENT_WAITING) {
      async_condition()->event_state(AsynchronousEventState::EVENT_DONE);
    }
  }
}

void AsyncPingTxOp::setup(OperatorSpec& spec) {
  spec.output<int>("out");

  spec.param(delay_, "delay", "Ping delay in ms", "Ping delay in ms", 10L);
  spec.param(count_, "count", "Ping count", "Ping count", 0UL);
}

void AsyncPingTxOp::start() {
  async_thread_ = std::thread([this] { async_ping(); });
}

void AsyncPingTxOp::compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
                            [[maybe_unused]] ExecutionContext& context) {
  ++index_;
  if (index_ == count_) {
    // Reached max count of ticks
    async_condition()->event_state(AsynchronousEventState::EVENT_NEVER);
  } else {
    async_condition()->event_state(AsynchronousEventState::EVENT_WAITING);
  }

  int value = index_;
  op_output.emit(value, "out");
}

void AsyncPingTxOp::stop() {
  should_stop_ = true;
  async_thread_.join();
}

}  // namespace holoscan::ops
