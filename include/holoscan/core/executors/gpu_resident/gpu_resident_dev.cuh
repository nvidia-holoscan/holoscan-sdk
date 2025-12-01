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

#ifndef HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_GPU_RESIDENT_DEV_CUH
#define HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_GPU_RESIDENT_DEV_CUH

#include <cuda_runtime.h>
#include <cuda/std/atomic>

#include "controlcommand.hpp"

// Device functions for GPU-resident CUDA graph

// The functions that use data ready, result ready and other shared memory
// addresses, leverages cuda::std::atomic_ref for correct memory ordering.

/**
 * @brief Mark data as ready for processing.
 *
 * @param data_ready_device Pointer to the data ready flag in device memory.
 */
__device__ __forceinline__ void gpu_resident_mark_data_ready_dev(unsigned int* data_ready_device) {
  cuda::std::atomic_ref<unsigned int> data_ready_atomic(*data_ready_device);
  data_ready_atomic.store(static_cast<unsigned int>(holoscan::ControlCommand::DATA_READY),
                          cuda::std::memory_order_release);
}

/**
 * @brief Mark data as not ready for processing.
 *
 * @param data_ready_device Pointer to the data ready flag in device memory.
 */
__device__ __forceinline__ void gpu_resident_mark_data_not_ready_dev(
    unsigned int* data_ready_device) {
  cuda::std::atomic_ref<unsigned int> data_ready_atomic(*data_ready_device);
  data_ready_atomic.store(static_cast<unsigned int>(holoscan::ControlCommand::DATA_NOT_READY),
                          cuda::std::memory_order_release);
}

/**
 * @brief Mark result as ready for consumption.
 *
 * @param result_ready_device Pointer to the result ready flag in device memory.
 */
__device__ __forceinline__ void gpu_resident_mark_result_ready_dev(
    unsigned int* result_ready_device) {
  cuda::std::atomic_ref<unsigned int> result_ready_atomic(*result_ready_device);
  result_ready_atomic.store(static_cast<unsigned int>(holoscan::ControlCommand::RESULT_READY),
                            cuda::std::memory_order_release);
}

/**
 * @brief Mark result as not ready for consumption.
 *
 * @param result_ready_device Pointer to the result ready flag in device memory.
 */
__device__ __forceinline__ void gpu_resident_mark_result_not_ready_dev(
    unsigned int* result_ready_device) {
  cuda::std::atomic_ref<unsigned int> result_ready_atomic(*result_ready_device);
  result_ready_atomic.store(static_cast<unsigned int>(holoscan::ControlCommand::RESULT_NOT_READY),
                            cuda::std::memory_order_release);
}

#endif  // HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_GPU_RESIDENT_DEV_CUH
