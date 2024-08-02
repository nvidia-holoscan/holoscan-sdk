/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ping_rx_op.hpp"

#include <vector>

namespace holoscan {
namespace ops {

void PingMultiRxOp::setup(OperatorSpec& spec) {
    spec.input<std::vector<int>>("receivers", IOSpec::kAnySize);
}

void PingMultiRxOp::compute(InputContext& op_input, OutputContext&, ExecutionContext&) {
  auto value_vector = op_input.receive<std::vector<int>>("receivers").value();

  HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());
  for (int i = 0; i < value_vector.size(); ++i) {
    HOLOSCAN_LOG_INFO("Rx message value{}: {}", i + 1, value_vector[i]);
  }
}

}  // namespace ops
}  // namespace holoscan
