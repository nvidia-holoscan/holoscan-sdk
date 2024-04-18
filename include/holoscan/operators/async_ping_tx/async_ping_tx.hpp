/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_ASYNC_PING_TX_HPP
#define HOLOSCAN_OPERATORS_ASYNC_PING_TX_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

/**
 * @brief Simple asynchronous transmitter operator.
 *
 * ==Named Outputs==
 *
 * - **out** : int
 *   - An index value that increments by one on each call to `compute`. The starting value
 *     is 1.
 *
 * ==Parameters==
 *
 * - **delay**: Ping delay in ms. Optional (default: `10L`)
 * - **count**: Ping count. Optional (default: `0UL`)
 * - **async_condition**: AsynchronousCondition adding async support to the operator.
 *   Optional (default: `nullptr`)
 */
class AsyncPingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncPingTxOp)

  AsyncPingTxOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override;
  void stop() override;

  void async_ping();

 private:
  Parameter<int64_t> delay_;
  Parameter<uint64_t> count_;
  Parameter<std::shared_ptr<AsynchronousCondition>> async_condition_;

  // internal state
  std::atomic<uint64_t> index_{0};
  std::atomic<bool> should_stop_{false};
  std::thread async_thread_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_ASYNC_PING_TX_HPP */
