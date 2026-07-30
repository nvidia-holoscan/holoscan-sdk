/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>

#include <holoscan/utils/cuda_macros.hpp>

namespace {

constexpr int kThreadsPerBlock = 256;

int num_blocks(int size) {
  return (size + kThreadsPerBlock - 1) / kThreadsPerBlock;
}

__global__ void add_constant_kernel(const int* __restrict__ input, int* __restrict__ output,
                                    int constant, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] + constant;
  }
}

__global__ void subtract_constant_kernel(const int* __restrict__ input, int* __restrict__ output,
                                         int constant, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] - constant;
  }
}

__global__ void multiply_kernel(const int* lhs, const int* rhs, int* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = lhs[idx] * rhs[idx];
  }
}

}  // namespace

void launch_add_constant_kernel(const int* input, int* output, int constant, int size,
                                cudaStream_t stream) {
  add_constant_kernel<<<num_blocks(size), kThreadsPerBlock, 0, stream>>>(
      input, output, constant, size);
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetLastError(), "Failed to launch add_constant_kernel");
}

void launch_subtract_constant_kernel(const int* input, int* output, int constant, int size,
                                     cudaStream_t stream) {
  subtract_constant_kernel<<<num_blocks(size), kThreadsPerBlock, 0, stream>>>(
      input, output, constant, size);
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetLastError(), "Failed to launch subtract_constant_kernel");
}

void launch_multiply_kernel(const int* lhs, const int* rhs, int* output, int size,
                            cudaStream_t stream) {
  multiply_kernel<<<num_blocks(size), kThreadsPerBlock, 0, stream>>>(lhs, rhs, output, size);
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetLastError(), "Failed to launch multiply_kernel");
}
