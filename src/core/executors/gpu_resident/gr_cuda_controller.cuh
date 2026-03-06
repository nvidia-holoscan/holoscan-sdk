/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_GR_CUDA_CONTROLLER_CUH
#define HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_GR_CUDA_CONTROLLER_CUH

#include <cuda_runtime.h>

// Forward declare the CUDA kernels
extern "C" {

/**
 * @brief Get the time from %globaltimer register in nanoseconds
 * We use CUDA C++ to get the time, and in the backend it uses PTX instructions to get the time in
 * device code.
 *
 * @return unsigned long long The time in nanoseconds
 */
__device__ unsigned long long gettime_ns();

/**
 * @brief Mark the end of the while loop and store the execution time in the device in microseconds.
 * The device memory address of the array of execution times is passed as an argument. Store the
 * execution time in the array up to the specified number of samples.
 *
 * If the start time is not provided (NULL), then don't compute and store the execution time.
 *
 * If sync_with_host is true, then a system-wide fence is performed after the end of the WHILE loop.
 * This ensures that all the data is visible to the host. However, this can slow down the compute
 * pipeline, and is recommended only when host is controlling the GPU-resident execution, i.e., for
 * debugging, development and testing purposes.
 *
 * @param data_ready_device Pointer to the data ready flag in device memory.
 * @param result_ready_device Pointer to the result ready flag in device memory.
 * @param execution_times_us Pointer to the array of execution times in microseconds.
 * @param num_samples Total number of samples to collect.
 * @param start_time_ns Pointer to the start time of the while loop in nanoseconds.
 * @param actual_samples_collected Pointer to store the actual number of samples collected.
 * @param sync_with_host Whether to synchronize with the host after the end of the WHILE loop.
 */
__global__ void while_end_marker(unsigned int* data_ready_device, unsigned int* result_ready_device,
                                 unsigned int* execution_times_us, unsigned int num_samples,
                                 unsigned long long* start_time_ns,
                                 unsigned int* actual_samples_collected, bool sync_with_host);

__global__ void start_perf_timer(unsigned long long* start_time_ns);

__global__ void while_controller(unsigned int* data_ready_device, unsigned int* result_ready_device,
                                 unsigned int* tear_down_device,
                                 unsigned int sleep_interval_us,
                                 cudaGraphConditionalHandle while_handle,
                                 cudaGraphConditionalHandle if_handle);
}

#endif  // HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_GR_CUDA_CONTROLLER_CUH
