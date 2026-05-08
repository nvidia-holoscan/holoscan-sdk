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

#include "multi_io_test_kernels.cuh"

__global__ void init_pattern_kernel(int* data, int port_id, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = port_id * 1000 + i;
  }
}

__global__ void copy_add_kernel(int* dst, const int* src, int add_val, int size) {
  for (int i = 0; i < size; ++i) {
    dst[i] = src[i] + add_val;
  }
}

void launch_init_pattern_kernel(int* data, int port_id, int size, cudaStream_t stream) {
  init_pattern_kernel<<<1, 1, 0, stream>>>(data, port_id, size);
}

void launch_copy_add_kernel(int* dst, const int* src, int add_val, int size, cudaStream_t stream) {
  copy_add_kernel<<<1, 1, 0, stream>>>(dst, src, add_val, size);
}
