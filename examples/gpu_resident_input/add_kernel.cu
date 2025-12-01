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

#include "holoscan/core/executors/gpu_resident/gpu_resident_dev.cuh"

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

// Kernel to generate fixed sequential numbers for testing
__global__ void generate_fixed_data_kernel(int* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Generate fixed sequential numbers starting from 1
  // This makes it easy to verify the results: output[i] = i + 1
  if (idx < size) {
    output[idx] = idx + 1;
  }

  if (idx == 0) {
    printf("generate_fixed_data_kernel: generating %d fixed numbers\n", size);
  }
}

// Kernel to mark data as ready (launched after random data generation completes)
__global__ void mark_data_ready_kernel(unsigned int* data_ready_address, int size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("mark_data_ready_kernel: marking %d elements ready\n", size);
    gpu_resident_mark_data_ready_dev(data_ready_address);
  }
}

// Host function to launch the kernel
void launch_add_five_kernel(int* input, int* output, int size, cudaStream_t stream) {
  // Use 256 threads per block
  int threadsPerBlock = 256;
  int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  add_five_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(input, output, size);
}

// Kernel to verify the computed results
// Expected: input was (idx + 1), and we added 5 three times, so result should be (idx + 1) + 15 =
// idx + 16
__global__ void verify_results_kernel(int* input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    int expected = idx + 16;  // Original (idx + 1) + 15 from three add_five_kernels
    int actual = input[idx];

    if (actual != expected) {
      printf("ERROR at index %d: expected %d, got %d\n", idx, expected, actual);
    }
  }

  // Print summary from first thread
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("verify_results_kernel: verification complete for %d elements\n", size);
  }
}

void launch_data_ready_handler_kernel(unsigned int* data_ready_address, int* output, int size,
                                      cudaStream_t stream) {
  // Use 256 threads per block
  int threadsPerBlock = 256;
  int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  // First, generate fixed sequential data
  generate_fixed_data_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(output, size);

  // Then mark data as ready (stream ordering ensures this runs after data generation)
  mark_data_ready_kernel<<<1, 1, 0, stream>>>(data_ready_address, size);
}

void launch_verify_results_kernel(int* input, int size, cudaStream_t stream) {
  // Use 256 threads per block
  int threadsPerBlock = 256;
  int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  verify_results_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(input, size);
}
