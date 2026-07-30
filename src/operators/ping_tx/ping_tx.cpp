/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/operators/ping_tx/ping_tx.hpp>

#include <memory>

namespace holoscan::ops {

void PingTxOp::setup(OperatorSpec& spec) {
  spec.output<int>("out");
}

void PingTxOp::compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
                       [[maybe_unused]] ExecutionContext& context) {
  int value = index_++;
  op_output.emit(value, "out");
}

}  // namespace holoscan::ops
