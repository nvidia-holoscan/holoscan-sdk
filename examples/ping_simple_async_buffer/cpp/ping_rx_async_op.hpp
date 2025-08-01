/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
