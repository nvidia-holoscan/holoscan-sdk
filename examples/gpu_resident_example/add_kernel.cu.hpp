/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

// Host function to launch the add five kernel
void launch_add_five_kernel(int* input, int* output, int size, cudaStream_t stream);
