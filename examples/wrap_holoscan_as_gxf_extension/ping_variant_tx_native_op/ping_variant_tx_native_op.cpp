/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ping_variant_tx_native_op.hpp"

using namespace holoscan;

namespace myops {

void PingVarTxNativeOp::setup(OperatorSpec& spec) {
  HOLOSCAN_LOG_INFO("PingVarTxNativeOp::setup() called.");
  spec.output<holoscan::gxf::Entity>("out");

  spec.param(custom_resource_,
             "custom_resource",
             "CustomResource",
             "This is a sample parameter for a custom resource.");
}

void PingVarTxNativeOp::compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
                             ExecutionContext& context) {
  HOLOSCAN_LOG_INFO("PingVarTxNativeOp::compute() called.");

  // Check if custom_resource_ is set and print a message.
  if (custom_resource_.get()) {
    HOLOSCAN_LOG_INFO(
        "PingVarTxNativeOp::compute() - custom_resource_ is set - custom_int: {}, float: {}",
        custom_resource_->get_custom_int(),
        custom_resource_->get_float());
  } else {
    HOLOSCAN_LOG_INFO("PingVarTxNativeOp::compute() - custom_resource_ is not set.");
  }

  // Create a new message (Entity)
  auto out_message = holoscan::gxf::Entity::New(&context);
  // Send the empty message.
  op_output.emit(out_message, "out");
}

}  // namespace myops
