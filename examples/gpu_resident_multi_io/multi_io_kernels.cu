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

#include <stdio.h>
#include <cuda_runtime.h>

#include "holoscan/core/executors/gpu_resident/gpu_resident_dev.cuh"

namespace {
constexpr int kModulo = 101;

__device__ unsigned int g_source_start = 0;

__global__ void source_emit_kernel(int* out0, int* out1, int size) {
  unsigned int start = atomicAdd(&g_source_start, 1) % kModulo;

  for (int idx = 0; idx < size; ++idx) {
    out0[idx] = static_cast<int>((start + idx) % kModulo);
    out1[idx] = static_cast<int>((start + idx + 1) % kModulo);
  }

  printf("source_emit_kernel: start=%u out0[0]=%d out1[0]=%d\n", start, out0[0], out1[0]);
}

__global__ void add_sub_kernel(const int* in0, const int* in1, int* sum_out, int* diff_out,
                               int size) {
  for (int idx = 0; idx < size; ++idx) {
    sum_out[idx] = in0[idx] + in1[idx];
    diff_out[idx] = in0[idx] - in1[idx];
  }

  printf("add_sub_kernel: sum[0]=%d diff[0]=%d\n", sum_out[0], diff_out[0]);
}

__global__ void final_add_kernel(const int* in_sum, const int* in_diff, int* out, int size) {
  for (int idx = 0; idx < size; ++idx) {
    out[idx] = in_sum[idx] + in_diff[idx];
  }

  printf("final_add_kernel: out[0]=%d\n", out[0]);
}

__global__ void mark_data_ready_kernel(unsigned int* data_ready_address) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gpu_resident_mark_data_ready_dev(data_ready_address);
  }
}
}  // namespace

void launch_source_emit_kernel(int* out0, int* out1, int size, cudaStream_t stream) {
  source_emit_kernel<<<1, 1, 0, stream>>>(out0, out1, size);
}

void launch_add_sub_kernel(const int* in0,
                           const int* in1,
                           int* sum_out,
                           int* diff_out,
                           int size,
                           cudaStream_t stream) {
  add_sub_kernel<<<1, 1, 0, stream>>>(in0, in1, sum_out, diff_out, size);
}

void launch_final_add_kernel(const int* in_sum,
                             const int* in_diff,
                             int* out,
                             int size,
                             cudaStream_t stream) {
  final_add_kernel<<<1, 1, 0, stream>>>(in_sum, in_diff, out, size);
}

void launch_mark_data_ready_kernel(unsigned int* data_ready_address, cudaStream_t stream) {
  mark_data_ready_kernel<<<1, 1, 0, stream>>>(data_ready_address);
}
