/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
