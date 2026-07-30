/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_TESTS_GPU_RESIDENT_TEST_KERNELS_CUH
#define HOLOSCAN_TESTS_GPU_RESIDENT_TEST_KERNELS_CUH

#include <cuda_runtime.h>

// Launch function for the add value kernel
// Adds 'value' to each element in the 'data' array of 'size' elements
void launch_add_value_kernel(int* data, int value, int size, cudaStream_t stream);

#endif  // HOLOSCAN_TESTS_GPU_RESIDENT_TEST_KERNELS_CUH
