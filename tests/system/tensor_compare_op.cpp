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

#include "tensor_compare_op.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <string>
#include <vector>
#include "holoscan/utils/cuda_macros.hpp"

namespace holoscan {
namespace ops {

void TensorCompareOp::setup(OperatorSpec& spec) {
  spec.input<TensorMap>("input1");
  spec.input<TensorMap>("input2");
}

void TensorCompareOp::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
                              [[maybe_unused]] ExecutionContext& context) {
  auto input1 = op_input.receive<TensorMap>("input1").value();
  if (input1.size() != 1) {
    HOLOSCAN_LOG_ERROR("Expected one tensor at `input1`");
    return;
  }
  auto tensor1 = input1.begin()->second;

  auto input2 = op_input.receive<TensorMap>("input2").value();
  if (input2.size() != 1) {
    HOLOSCAN_LOG_ERROR("Expected one tensor at `input2`");
    return;
  }
  auto tensor2 = input2.begin()->second;

  if (tensor1->nbytes() != tensor2->nbytes()) {
    HOLOSCAN_LOG_ERROR(
        "Expected same size but got {} and {}", tensor1->nbytes(), tensor2->nbytes());
    return;
  }

  std::vector<uint8_t> data1(tensor1->nbytes());
  HOLOSCAN_CUDA_CALL(
      cudaMemcpy(data1.data(), tensor1->data(), tensor1->nbytes(), cudaMemcpyDeviceToHost));

  std::vector<uint8_t> data2(tensor2->nbytes());
  HOLOSCAN_CUDA_CALL(
      cudaMemcpy(data2.data(), tensor2->data(), tensor2->nbytes(), cudaMemcpyDeviceToHost));

  auto result = std::mismatch(data1.begin(), data1.end(), data2.begin());
  if (result.first != data1.end()) {
    HOLOSCAN_LOG_ERROR("Inputs differ at index {}: {} != {}",
                       ssize_t(std::distance(data1.begin(), result.first)),
                       *result.first,
                       *result.second);
    return;
  }
}

}  // namespace ops
}  // namespace holoscan
