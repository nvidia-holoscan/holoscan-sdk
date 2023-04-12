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

#include "ping_rx_native_op.hpp"

#include <holoscan/core/gxf/gxf_tensor.hpp>

using namespace holoscan;

namespace myops {

void PingRxNativeOp::setup(OperatorSpec& spec) {
  HOLOSCAN_LOG_INFO("PingRxNativeOp::setup() called.");
  spec.input<gxf::Entity>("in");
}

void PingRxNativeOp::compute(InputContext& op_input, OutputContext& op_output,
                             ExecutionContext& context) {
  HOLOSCAN_LOG_INFO("PingRxNativeOp::compute() called.");

  // The type of `in_message` is 'holoscan::gxf::Entity'.
  auto in_message = op_input.receive<holoscan::gxf::Entity>("in");
  HOLOSCAN_LOG_INFO("Number of pings received: {}", count_++);
}

}  // namespace myops
