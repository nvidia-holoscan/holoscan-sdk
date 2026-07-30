/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

void launch_source_emit_kernel(int* out0, int* out1, int size, cudaStream_t stream);
void launch_add_sub_kernel(const int* in0, const int* in1, int* sum_out, int* diff_out, int size,
                           cudaStream_t stream);
void launch_final_add_kernel(const int* in_sum, const int* in_diff, int* out, int size,
                             cudaStream_t stream);
void launch_mark_data_ready_kernel(unsigned int* data_ready_address, cudaStream_t stream);
