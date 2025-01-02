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

void PingTensorRxOp::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
                             [[maybe_unused]] ExecutionContext& context) {
  auto maybe_in_message = op_input.receive<holoscan::TensorMap>("in");
  if (!maybe_in_message) {
    HOLOSCAN_LOG_ERROR("Failed to receive message from port 'in'");
    return;
  }
  cudaStream_t stream = op_input.receive_cuda_stream("in", false);

  HOLOSCAN_LOG_INFO("{} received {}default CUDA stream from port 'in'",
                    name(),
                    stream == cudaStreamDefault ? "" : "non-");

  // have this operator wait for any work on the stream to complete
  cudaError_t status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("Failed to synchronize stream: {}", cudaGetErrorString(status));
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
