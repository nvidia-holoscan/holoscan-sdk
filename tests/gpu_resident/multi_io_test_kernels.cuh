/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_TESTS_GPU_RESIDENT_MULTI_IO_TEST_KERNELS_CUH
#define HOLOSCAN_TESTS_GPU_RESIDENT_MULTI_IO_TEST_KERNELS_CUH

#include <cuda_runtime.h>

// Sets every element to a deterministic pattern: data[i] = port_id * 1000 + i.
// Launched with a single thread.
void launch_init_pattern_kernel(int* data, int port_id, int size, cudaStream_t stream);

// dst[i] = src[i] + add_val for each element.
// Launched with a single thread.
void launch_copy_add_kernel(int* dst, const int* src, int add_val, int size, cudaStream_t stream);

#endif  // HOLOSCAN_TESTS_GPU_RESIDENT_MULTI_IO_TEST_KERNELS_CUH
