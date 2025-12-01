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

#include "holoscan/operators/ping_tensor_rx/ping_tensor_rx.hpp"

#include <cuda_runtime.h>

#include <string>

namespace holoscan::ops {

void PingTensorRxOp::setup(OperatorSpec& spec) {
  spec.input<TensorMap>("in");
  spec.param(receive_as_tensormap_,
             "receive_as_tensormap",
             "receive as tensormap",
             "Whether to receive the tensor as a TensorMap. If false, "
             "receive<std::shared_ptr<Tensor>> is used instead",
             true);
}

void PingTensorRxOp::compute(InputContext& op_input, OutputContext&, ExecutionContext& context) {
  if (receive_as_tensormap_.get()) {
    auto maybe_in_message = op_input.receive<holoscan::TensorMap>("in");
    if (!maybe_in_message) {
      HOLOSCAN_LOG_ERROR("Failed to receive message from port 'in'");
      return;
    }

    if (context.is_gpu_available()) {
      cudaStream_t stream = op_input.receive_cuda_stream("in", false);

      HOLOSCAN_LOG_INFO("{} received {}default CUDA stream from port 'in'",
                        name(),
                        stream == cudaStreamDefault ? "" : "non-");

      // have this operator wait for any work on the stream to complete
      cudaError_t status = cudaStreamSynchronize(stream);
      if (status != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Failed to synchronize stream: {}", cudaGetErrorString(status));
      }
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
  } else {
    auto maybe_tensor = op_input.receive<std::shared_ptr<Tensor>>("in");
    if (!maybe_tensor) {
      HOLOSCAN_LOG_ERROR("Failed to receive std::shared_ptr<Tensor> from port 'in'");
      return;
    }

    if (context.is_gpu_available()) {
      cudaStream_t stream = op_input.receive_cuda_stream("in", false);
      HOLOSCAN_LOG_INFO("{} received {}default CUDA stream from port 'in'",
                        name(),
                        stream == cudaStreamDefault ? "" : "non-");

      // have this operator wait for any work on the stream to complete
      cudaError_t status = cudaStreamSynchronize(stream);
      if (status != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Failed to synchronize stream: {}", cudaGetErrorString(status));
      }
    }

    auto tensor = maybe_tensor.value();
    HOLOSCAN_LOG_INFO("{} received message {}: Tensor shape: ({})",
                      name(),
                      count_++,
                      fmt::join(tensor->shape(), ", "));
  }
}
}  // namespace holoscan::ops
