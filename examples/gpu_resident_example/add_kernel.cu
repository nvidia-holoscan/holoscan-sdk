/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <cstdio>

// Simple CUDA kernel that adds 5 to each element in the input array
__global__ void add_five_kernel(int* input, int* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    output[idx] = input[idx] + 5;
  }
  if (idx == 0) {
    printf("add_five_kernel: %d\n", output[idx]);
  }
}

// Host function to launch the kernel
void launch_add_five_kernel(int* input, int* output, int size, cudaStream_t stream) {
  // Use 256 threads per block
  int threadsPerBlock = 256;
  int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  add_five_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(input, output, size);
}
