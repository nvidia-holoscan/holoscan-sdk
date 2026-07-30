/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/operators/ping_rx/ping_rx.hpp>

namespace holoscan::ops {

void PingRxOp::setup(OperatorSpec& spec) {
  spec.input<int>("in");
}

void PingRxOp::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
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
}

}  // namespace holoscan::ops
