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

#pragma once

#include <cuda_runtime.h>

void launch_source_emit_kernel(int* out0, int* out1, int size, cudaStream_t stream);
void launch_add_sub_kernel(const int* in0, const int* in1, int* sum_out, int* diff_out, int size,
                           cudaStream_t stream);
void launch_final_add_kernel(const int* in_sum, const int* in_diff, int* out, int size,
                             cudaStream_t stream);
void launch_mark_data_ready_kernel(unsigned int* data_ready_address, cudaStream_t stream);
