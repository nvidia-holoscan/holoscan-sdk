/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RESOURCES_CUDA_GREEN_CONTEXT_CPP_TEST_KERNEL_CU_HPP
#define RESOURCES_CUDA_GREEN_CONTEXT_CPP_TEST_KERNEL_CU_HPP

#include <cuda_runtime.h>

// matrix multiplication
void asyncLaunchMatrixMultiplyKernel(float* A, float* B, float* C, int N, cudaStream_t stream);

#endif /* RESOURCES_CUDA_GREEN_CONTEXT_CPP_TEST_KERNEL_CU_HPP */
