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
