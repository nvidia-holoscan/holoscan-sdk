/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ping_tx_native_op.hpp"

#include <holoscan/core/gxf/gxf_tensor.hpp>

using namespace holoscan;

namespace myops {

void PingTxNativeOp::setup(OperatorSpec& spec) {
  HOLOSCAN_LOG_INFO("PingTxNativeOp::setup() called.");
  spec.output<gxf::Entity>("out");
}

void PingTxNativeOp::compute(InputContext& op_input, OutputContext& op_output,
                             ExecutionContext& context) {
  HOLOSCAN_LOG_INFO("PingTxNativeOp::compute() called.");

  // Create a new message (Entity)
  auto out_message = gxf::Entity::New(&context);
  // Send the empty message.
  op_output.emit(out_message, "out");
}

}  // namespace myops
