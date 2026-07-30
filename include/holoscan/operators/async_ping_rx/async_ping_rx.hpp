/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_OPERATORS_ASYNC_PING_RX_HPP
#define HOLOSCAN_OPERATORS_ASYNC_PING_RX_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

/**
 * @brief Simple asynchronous receiver operator.
 *
 * ==Named Inputs==
 *
 * - **in** : any
 *   - A received value.
 *
 * ==Parameters==
 *
 * - **delay**: Ping delay in ms. Optional (default: `10L`)
 * - **async_condition**: AsynchronousCondition adding async support to the operator.
 *   Optional (default: `nullptr`)
 */
class AsyncPingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncPingRxOp)

  AsyncPingRxOp() = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;
  void stop() override;

  void async_ping();

 private:
  Parameter<int64_t> delay_;

  // internal state
  std::atomic<bool> should_stop_{false};
  std::thread async_thread_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_ASYNC_PING_RX_HPP */
