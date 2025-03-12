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

#include "ping_variant_rx_native_op.hpp"

#include <memory>
#include <vector>

using namespace holoscan;

namespace myops {

void PingVarRxNativeOp::setup(OperatorSpec& spec) {
  HOLOSCAN_LOG_INFO("PingVarRxNativeOp::setup() called.");
  spec.input<holoscan::gxf::Entity>("in");
  spec.input<std::vector<holoscan::gxf::Entity>>("receivers", IOSpec::kAnySize);

  spec.param(boolean_condition_,
             "boolean_condition",
             "BooleanCondition",
             "BooleanCondition",
             ParameterFlag::kOptional);
}

void PingVarRxNativeOp::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
                             [[maybe_unused]] ExecutionContext& context) {
  HOLOSCAN_LOG_INFO("PingVarRxNativeOp::compute() called.");

  // The type of `in_message` is 'holoscan::gxf::Entity'.
  auto in_message = op_input.receive<holoscan::gxf::Entity>("in");
  auto receivers_message =
      op_input.receive<std::vector<holoscan::gxf::Entity>>("receivers").value();
  HOLOSCAN_LOG_INFO("Number of pings received: {}", count_++);
  HOLOSCAN_LOG_INFO("Number of receivers: {}", receivers_message.size());
}

}  // namespace myops
