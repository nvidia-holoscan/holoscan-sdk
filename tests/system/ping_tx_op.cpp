/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ping_tx_op.hpp"

#include <memory>

namespace holoscan {
namespace ops {

void PingMultiTxOp::setup(OperatorSpec& spec) {
  spec.output<int>("out1");
  spec.output<int>("out2");
}

void PingMultiTxOp::compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
                            [[maybe_unused]] ExecutionContext& context) {
  int value1 = 1;
  op_output.emit(value1, "out1");

  int value2 = 100;
  op_output.emit(value2, "out2");
}

}  // namespace ops
}  // namespace holoscan
