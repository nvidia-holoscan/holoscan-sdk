/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

void launch_add_constant_kernel(const int* input, int* output, int constant, int size,
                                cudaStream_t stream);

void launch_subtract_constant_kernel(const int* input, int* output, int constant, int size,
                                     cudaStream_t stream);

void launch_multiply_kernel(const int* lhs, const int* rhs, int* output, int size,
                            cudaStream_t stream);
