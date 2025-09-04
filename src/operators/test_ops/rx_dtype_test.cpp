/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/operators/test_ops/rx_dtype_test.hpp"

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
