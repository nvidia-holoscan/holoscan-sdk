/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
  cudaMemcpyKind copy_kind1;
  if (tensor1->device().device_type == kDLCUDAManaged) {
    copy_kind1 = cudaMemcpyDefault;
  } else if (tensor1->device().device_type == kDLCUDAHost) {
    copy_kind1 = cudaMemcpyHostToHost;
  } else {
    copy_kind1 = cudaMemcpyDeviceToHost;
  }
  HOLOSCAN_CUDA_CALL(cudaMemcpy(data1.data(), tensor1->data(), tensor1->nbytes(), copy_kind1));

  std::vector<uint8_t> data2(tensor2->nbytes());
  cudaMemcpyKind copy_kind2;
  if (tensor2->device().device_type == kDLCUDAManaged) {
    copy_kind2 = cudaMemcpyDefault;
  } else if (tensor2->device().device_type == kDLCUDAHost) {
    copy_kind2 = cudaMemcpyHostToHost;
  } else {
    copy_kind2 = cudaMemcpyDeviceToHost;
  }
  HOLOSCAN_CUDA_CALL(cudaMemcpy(data2.data(), tensor2->data(), tensor2->nbytes(), copy_kind2));

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
