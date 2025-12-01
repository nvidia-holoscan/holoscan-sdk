/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdio.h>
#include <cuda_runtime.h>

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
