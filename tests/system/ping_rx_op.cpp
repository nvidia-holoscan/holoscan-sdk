/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ping_rx_op.hpp"

#include <vector>

namespace holoscan {
namespace ops {

void PingMultiRxOp::setup(OperatorSpec& spec) {
  spec.input<std::vector<int>>("receivers", IOSpec::kAnySize);
}

void PingMultiRxOp::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
                            [[maybe_unused]] ExecutionContext& context) {
  auto value_vector = op_input.receive<std::vector<int>>("receivers").value();

  HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());
  for (int i = 0; i < value_vector.size(); ++i) {
    HOLOSCAN_LOG_INFO("Rx message value{}: {}", i + 1, value_vector[i]);
  }
}

}  // namespace ops
}  // namespace holoscan
