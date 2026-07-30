/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
