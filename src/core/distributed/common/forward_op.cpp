/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/distributed/common/forward_op.hpp>

#include <holoscan/core/io_context.hpp>

namespace holoscan::ops {

void ForwardOp::setup(OperatorSpec& spec) {
  spec.input<std::any>("in");
  spec.output<std::any>("out");
}

void ForwardOp::compute(InputContext& op_input, OutputContext& op_output,
                        [[maybe_unused]] ExecutionContext& context) {
  auto in_message = op_input.receive<std::any>("in");
  if (in_message) {
    auto value = in_message.value();
    if (value.type() == typeid(holoscan::gxf::Entity)) {
      // emit as entity
      auto entity = std::any_cast<holoscan::gxf::Entity>(value);
      op_output.emit(entity, "out");
    } else {
      // emit as std::any
      op_output.emit(value, "out");
    }
  }
}

}  // namespace holoscan::ops
