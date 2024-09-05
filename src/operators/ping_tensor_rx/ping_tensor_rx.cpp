/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/operators/ping_tensor_rx/ping_tensor_rx.hpp"

#include <cuda_runtime.h>

#include <string>

namespace holoscan::ops {

void PingTensorRxOp::setup(OperatorSpec& spec) {
  spec.input<TensorMap>("in");
}

void PingTensorRxOp::compute(InputContext& op_input, OutputContext&, ExecutionContext&) {
  auto maybe_in_message = op_input.receive<holoscan::TensorMap>("in");
  if (!maybe_in_message) {
    HOLOSCAN_LOG_ERROR("Failed to receive message from port 'in'");
    return;
  }
  auto in_message = maybe_in_message.value();
  // Loop over any tensors found, printing their names and shapes.
  for (auto& [key, tensor] : in_message) {
    if (tensor->data() == nullptr) {
      HOLOSCAN_LOG_ERROR("Received tensor named '{}' with null data", key);
      continue;
    }
    HOLOSCAN_LOG_INFO("{} received message {}: Tensor key: '{}', shape: ({})",
                      name(),
                      count_++,
                      key,
                      fmt::join(tensor->shape(), ", "));
  }
}

}  // namespace holoscan::ops
