/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ping_tensor_rx_op.hpp"

#include <cuda_runtime.h>

#include <string>

namespace holoscan {
namespace ops {

void PingTensorRxOp::setup(OperatorSpec& spec) {
  spec.input<TensorMap>("in");

  spec.param(tensor_name_,
             "tensor_name",
             "output tensor name",
             "output tensor name (default: tensor)",
             std::string{"tensor"});
}

void PingTensorRxOp::compute(InputContext& op_input, OutputContext&, ExecutionContext&) {
  auto value = op_input.receive<TensorMap>("in").value();
  auto& tensor = value[tensor_name_.get()];
  if (tensor->data() == nullptr) {
    HOLOSCAN_LOG_ERROR("Received tensor with null data");
    return;
  }
  uint8_t data = 0;
  cudaMemcpy(&data, tensor->data(), 1, cudaMemcpyDeviceToHost);
  HOLOSCAN_LOG_INFO(
      "Rx message value - name:{}, data[0]:{}, nbytes:{}", name(), data, tensor->nbytes());
}

}  // namespace ops
}  // namespace holoscan
