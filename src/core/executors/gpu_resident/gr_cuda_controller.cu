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

#include <cuda_runtime.h>
#include <cuda/std/atomic>
#include <cuda/std/chrono>

#include <cstdio>

#include "gr_cuda_controller.cuh"

#include <holoscan/core/executors/gpu_resident/controlcommand.hpp>
#include <holoscan/core/executors/gpu_resident/gpu_resident_dev.cuh>

extern "C" {

__device__ unsigned long long gettime_ns() {
  return cuda::std::chrono::system_clock::now().time_since_epoch().count();
}

__device__ inline void while_end_marker_body(
    unsigned int* data_ready_device, unsigned int* result_ready_device,
    unsigned int* execution_times_us, unsigned int num_samples, unsigned long long* start_time_ns,
    unsigned int* actual_samples_collected, bool sync_with_host) {
  if (sync_with_host) {
    __threadfence_system();
  }

  if (execution_times_us && actual_samples_collected && *actual_samples_collected < num_samples) {
    // end the perf measurement timer and store the execution time
    unsigned long long current_time_ns = gettime_ns();
    unsigned long long execution_time_ns = current_time_ns - *start_time_ns;
    unsigned int execution_time_us = execution_time_ns / 1000;
    execution_times_us[*actual_samples_collected] = execution_time_us;
    // Increment the actual number of samples collected
    (*actual_samples_collected)++;
  }

  // Mark result as ready and data as not ready using device functions
  gpu_resident_mark_data_not_ready_dev(data_ready_device);
  gpu_resident_mark_result_ready_dev(result_ready_device);
}

__global__ void while_end_marker(unsigned int* data_ready_device, unsigned int* result_ready_device,
                                 unsigned int* execution_times_us, unsigned int num_samples,
                                 unsigned long long* start_time_ns,
                                 unsigned int* actual_samples_collected, bool sync_with_host) {
  while_end_marker_body(data_ready_device,
                        result_ready_device,
                        execution_times_us,
                        num_samples,
                        start_time_ns,
                        actual_samples_collected,
                        sync_with_host);
}

__global__ void while_end_marker_debug(unsigned int* data_ready_device,
                                       unsigned int* result_ready_device,
                                       unsigned int* execution_times_us, unsigned int num_samples,
                                       unsigned long long* start_time_ns,
                                       unsigned int* actual_samples_collected,
                                       bool sync_with_host) {
  printf("while_end_marker: data_ready=%u result_ready=%u samples=%u/%u sync_with_host=%d\n",
         *data_ready_device,
         *result_ready_device,
         actual_samples_collected ? *actual_samples_collected : 0,
         num_samples,
         static_cast<int>(sync_with_host));
  while_end_marker_body(data_ready_device,
                        result_ready_device,
                        execution_times_us,
                        num_samples,
                        start_time_ns,
                        actual_samples_collected,
                        sync_with_host);
}

__global__ void start_perf_timer(unsigned long long* start_time_ns) {
  *start_time_ns = gettime_ns();
}

__device__ inline void while_controller_body(unsigned int* data_ready_device,
                                             unsigned int* tear_down_device,
                                             unsigned int sleep_interval_us,
                                             cudaGraphConditionalHandle while_handle,
                                             cudaGraphConditionalHandle if_handle) {
  // Create cuda::std::atomic_ref for CPU-GPU synchronization
  cuda::std::atomic_ref<unsigned int> data_ready_atomic(*data_ready_device);
  cuda::std::atomic_ref<unsigned int> tear_down_atomic(*tear_down_device);

  // Atomic load with acquire ordering ensures we see the latest value
  unsigned int data_ready = data_ready_atomic.load(cuda::std::memory_order_acquire);
  unsigned int tear_down = tear_down_atomic.load(cuda::std::memory_order_acquire);

  if (data_ready == static_cast<unsigned int>(holoscan::ControlCommand::DATA_NOT_READY)) {
    // data is not ready, don't do anything and sleep for the specified interval
    // for loop is used because nanosleep works in max of 1 ms granularity
    for (unsigned int i = 0; i < sleep_interval_us; i++) {
      unsigned int sleep_duration_ns = 1000;  // 1 us = 1000 ns
      asm volatile("nanosleep.u32 %0;" ::"r"(sleep_duration_ns));
    }
    cudaGraphSetConditional(if_handle, 0);
  } else if (data_ready == static_cast<unsigned int>(holoscan::ControlCommand::DATA_READY)) {
    // set the if conditional handle to true
    cudaGraphSetConditional(if_handle, 1);
  }
  if (tear_down == static_cast<unsigned int>(holoscan::ControlCommand::TEAR_DOWN)) {
    cudaGraphSetConditional(while_handle, 0);
    cudaGraphSetConditional(if_handle, 0);
    return;
  }
}

__global__ void while_controller(unsigned int* data_ready_device, unsigned int* result_ready_device,
                                 unsigned int* tear_down_device, unsigned int sleep_interval_us,
                                 cudaGraphConditionalHandle while_handle,
                                 cudaGraphConditionalHandle if_handle) {
  (void)result_ready_device;
  while_controller_body(
      data_ready_device, tear_down_device, sleep_interval_us, while_handle, if_handle);
}

__global__ void while_controller_debug(unsigned int* data_ready_device,
                                       unsigned int* result_ready_device,
                                       unsigned int* tear_down_device,
                                       unsigned int sleep_interval_us,
                                       cudaGraphConditionalHandle while_handle,
                                       cudaGraphConditionalHandle if_handle) {
  cuda::std::atomic_ref<unsigned int> data_ready_atomic(*data_ready_device);
  cuda::std::atomic_ref<unsigned int> result_ready_atomic(*result_ready_device);
  cuda::std::atomic_ref<unsigned int> tear_down_atomic(*tear_down_device);

  printf("while_controller: data_ready=%u result_ready=%u tear_down=%u sleep_interval_us=%u\n",
         data_ready_atomic.load(cuda::std::memory_order_acquire),
         result_ready_atomic.load(cuda::std::memory_order_acquire),
         tear_down_atomic.load(cuda::std::memory_order_acquire),
         sleep_interval_us);

  while_controller_body(
      data_ready_device, tear_down_device, sleep_interval_us, while_handle, if_handle);
}

}  // extern "C"
