/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/operators/test_ops/rx_dtype_test.hpp>

#include <any>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/operator_spec.hpp>

namespace holoscan::ops {

void DataTypeRxTestOp::setup(OperatorSpec& spec) {
  spec.input<std::any>("in");
}

void DataTypeRxTestOp::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
                               [[maybe_unused]] ExecutionContext& context) {
  auto message = op_input.receive<std::any>("in");
  if (!message) {
    throw std::runtime_error("No message received");
  }
  auto value = message.value();
  HOLOSCAN_LOG_INFO("Received message of type: {}", value.type().name());
}

}  // namespace holoscan::ops
