/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdio>

#include "gr_cuda_controller.cuh"
#include "holoscan/core/executors/gpu_resident/controlcommand.hpp"
#include "holoscan/core/executors/gpu_resident/gpu_resident_dev.cuh"

extern "C" {

__global__ void while_end_marker(unsigned int* data_ready_device,
                                 unsigned int* result_ready_device) {
  // Mark result as ready and data as not ready using device functions
  gpu_resident_mark_result_ready_dev(result_ready_device);
  gpu_resident_mark_data_not_ready_dev(data_ready_device);
}

__global__ void while_controller(unsigned int* data_ready_device, unsigned int* result_ready_device,
                                 unsigned int* tear_down_device,
                                 cudaGraphConditionalHandle while_handle,
                                 cudaGraphConditionalHandle if_handle) {
  // Create cuda::std::atomic_ref for CPU-GPU synchronization
  cuda::std::atomic_ref<unsigned int> data_ready_atomic(*data_ready_device);
  cuda::std::atomic_ref<unsigned int> tear_down_atomic(*tear_down_device);

  // Atomic load with acquire ordering ensures we see the latest value
  unsigned int data_ready = data_ready_atomic.load(cuda::std::memory_order_acquire);
  unsigned int tear_down = tear_down_atomic.load(cuda::std::memory_order_acquire);

  if (data_ready == static_cast<unsigned int>(holoscan::ControlCommand::DATA_NOT_READY)) {
    // data is not ready, don't do anything and sleep for 500 us
    for (int i = 0; i < 500; i++) {
      unsigned int sleep_duration_ns = 1000000;  // 1 us = 1,000,000 ns
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

}  // extern "C"
