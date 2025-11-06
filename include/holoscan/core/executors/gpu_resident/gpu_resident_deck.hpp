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

#ifndef HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_GPU_RESIDENT_DECK_HPP
#define HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_GPU_RESIDENT_DECK_HPP

#include <cuda_runtime.h>
#include <atomic>
#include <future>
#include <memory>

#include "holoscan/utils/cuda/buffer.hpp"

namespace holoscan {

/**
 * @brief GPU-resident deck is the CPU-side software component that is
 * responsible for communication with the asynchronously running GPU-resident
 * CUDA workload.
 *
 */
class GPUResidentDeck {
 public:
  GPUResidentDeck();
  ~GPUResidentDeck();

  /**
   * @brief This function launches an executable CUDA graph asynchronously.
   *
   * @param graph The executable CUDA graph to launch asynchronously.
   * @return std::future<void> A future that will be set when the CUDA graph has finished executing.
   */
  std::future<void> launch_cuda_graph(cudaGraphExec_t graph);

  void* data_ready_device_address() { return cpu_data_ready_trigger_->device_data(); }
  void* result_ready_device_address() { return cpu_result_ready_trigger_->device_data(); }
  void* tear_down_device_address() { return cpu_tear_down_trigger_->device_data(); }

  /**
   * @brief Sets the timeout for the GPU-resident CUDA graph execution. If
   * timeout is zero, then the asynchronous execution will wait until an
   * external tear down is triggered.
   *
   * @param timeout_ms The timeout in milliseconds.
   */
  void timeout_ms(unsigned long long timeout_ms) { timeout_ms_ = timeout_ms; }

  /**
   * @brief Indicates whether the result of a single iteration of the
   * GPU-resident CUDA graph is ready or not.
   *
   * @return true if the result is ready, false otherwise.
   */
  bool result_ready();

  /**
   * @brief Sends a tear down signal to the GPU-resident CUDA graph. The timeout
   * has to be set to zero for this to work for now.
   * In the future, we will support ignoring the timeout when tear down is triggered.
   *
   */
  void tear_down();

  /**
   * @brief This function informs GPU-resident CUDA graph that the data is ready
   * for the main workload.
   *
   */
  void set_data_ready();

  /**
   * @brief Indicates whether the GPU-resident CUDA graph has been launched.
   *
   * @return true if the CUDA graph has been launched, false otherwise.
   */
  bool is_launched() const;

 private:
  /// We create three different status buffers to track :
  /// - whether data is ready for the CUDA workload
  /// - whether result is ready
  /// - whether the tear down is needed
  /// A single buffer could have been used, but simultaneous operation on
  /// multiple threads work more reliably with different buffer addresses. It also helps manage a
  /// clear implementation.
  std::shared_ptr<holoscan::utils::cuda::CudaHostMappedBuffer> cpu_data_ready_trigger_;
  std::shared_ptr<holoscan::utils::cuda::CudaHostMappedBuffer> cpu_result_ready_trigger_;
  std::shared_ptr<holoscan::utils::cuda::CudaHostMappedBuffer> cpu_tear_down_trigger_;

  /// RAII for cudaStream_t is not being used here, as we specifically keep
  /// track of the streams in this class.
  cudaStream_t execution_stream_;
  cudaStream_t status_stream_;

  unsigned long long timeout_ms_ = 0;

  /// Atomic variable to track whether the CUDA graph has been launched (true) or
  /// torn down (false)
  std::atomic<bool> graph_launched_{false};
};

}  // namespace holoscan
#endif  // HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_GPU_RESIDENT_DECK_HPP
