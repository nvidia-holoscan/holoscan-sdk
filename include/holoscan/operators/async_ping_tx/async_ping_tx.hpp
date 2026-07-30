/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
 */
class AsyncPingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncPingTxOp)

  AsyncPingTxOp() = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;
  void stop() override;

  void async_ping();

 private:
  Parameter<int64_t> delay_;
  Parameter<uint64_t> count_;

  // internal state
  std::atomic<uint64_t> index_{0};
  std::atomic<bool> should_stop_{false};
  std::thread async_thread_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_ASYNC_PING_TX_HPP */
