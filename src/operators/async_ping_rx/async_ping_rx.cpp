/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/operators/async_ping_rx/async_ping_rx.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

namespace holoscan::ops {

void AsyncPingRxOp::async_ping() {
  HOLOSCAN_LOG_INFO("Async ping rx thread entering.");
  while (true) {
    if (should_stop_) {
      HOLOSCAN_LOG_INFO("Async ping rx thread exiting.");
      return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(delay_.get()));

    if (async_condition()->event_state() == AsynchronousEventState::EVENT_WAITING) {
      async_condition()->event_state(AsynchronousEventState::EVENT_DONE);
    }
  }
}

void AsyncPingRxOp::setup(OperatorSpec& spec) {
  spec.input<int>("in");
  spec.param(delay_, "delay", "Ping delay in ms", "Ping delay in ms", 10L);
}

void AsyncPingRxOp::start() {
  async_thread_ = std::thread([this] { async_ping(); });
}

void AsyncPingRxOp::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
                            [[maybe_unused]] ExecutionContext& context) {
  auto maybe_value = op_input.receive<int>("in");
  if (!maybe_value) {
    auto error_msg = fmt::format("Operator '{}' failed to receive message from port 'in': {}",
                                 name_,
                                 maybe_value.error().what());
    HOLOSCAN_LOG_ERROR(error_msg);
    throw std::runtime_error(error_msg);
  }
  int value = maybe_value.value();
  HOLOSCAN_LOG_INFO("Rx message value: {}", value);

  async_condition()->event_state(AsynchronousEventState::EVENT_WAITING);
}

void AsyncPingRxOp::stop() {
  should_stop_ = true;
  async_thread_.join();
}

}  // namespace holoscan::ops
