/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_kernels.cuh"

// Simple CUDA kernel that adds a value to each element
__global__ void add_value_kernel(int* data, int value, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] += value;
  }
}

// Launch function for the add value kernel
void launch_add_value_kernel(int* data, int value, int size, cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  add_value_kernel<<<num_blocks, block_size, 0, stream>>>(data, value, size);
}
