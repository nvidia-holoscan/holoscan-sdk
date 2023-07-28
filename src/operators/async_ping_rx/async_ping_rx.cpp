/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/operators/async_ping_rx/async_ping_rx.hpp"

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

    if (async_condition_->event_state() ==
        AsynchronousEventState::EVENT_WAITING) {
      async_condition_->event_state(AsynchronousEventState::EVENT_DONE);
    }
  }
}

void AsyncPingRxOp::setup(OperatorSpec& spec) {
  spec.input<int>("in");
  spec.param(delay_, "delay", "Ping delay in ms", "Ping delay in ms", 10L);
  spec.param(async_condition_,
             "async_condition",
             "asynchronous condition",
             "AsynchronousCondition adding async support to the operator");
}

void AsyncPingRxOp::initialize() {
  // Set up prerequisite parameters before calling Operator::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'async_condition_'
  auto has_async_condition =
      std::find_if(args().begin(), args().end(), [](const auto& arg) {
        return (arg.name() == "async_condition");
      });

  // Create the AsynchronousCondition if there is no argument provided.
  if (has_async_condition == args().end()) {
    async_condition_ =
        frag->make_condition<holoscan::AsynchronousCondition>("async_condition");
    add_arg(async_condition_.get());
  }

  Operator::initialize();
}

void AsyncPingRxOp::start() {
  async_thread_ = std::thread([this] { async_ping(); });
}

void AsyncPingRxOp::compute(InputContext& op_input, OutputContext&, ExecutionContext&) {
  auto value = op_input.receive<int>("in").value();
  HOLOSCAN_LOG_INFO("Rx message value: {}", value);

  async_condition_->event_state(AsynchronousEventState::EVENT_WAITING);
}

void AsyncPingRxOp::stop() {
  should_stop_ = true;
  async_thread_.join();
}

}  // namespace holoscan::ops
