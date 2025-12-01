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

#include "holoscan/core/executors/gpu_resident/gpu_resident_deck.hpp"

#include <cuda_runtime.h>

#include <cuda/std/atomic>

#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include "holoscan/core/executors/gpu_resident/controlcommand.hpp"
#include "holoscan/utils/cuda/buffer.hpp"
#include "holoscan/utils/cuda_macros.hpp"

namespace holoscan {

GPUResidentDeck::GPUResidentDeck() {
  cpu_data_ready_trigger_ =
      std::make_shared<holoscan::utils::cuda::CudaHostMappedBuffer>(sizeof(ControlCommand));
  cpu_result_ready_trigger_ =
      std::make_shared<holoscan::utils::cuda::CudaHostMappedBuffer>(sizeof(ControlCommand));
  cpu_tear_down_trigger_ =
      std::make_shared<holoscan::utils::cuda::CudaHostMappedBuffer>(sizeof(ControlCommand));

  // Initialize CUDA streams
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaStreamCreateWithFlags(&execution_stream_, cudaStreamNonBlocking),
      "Failed to create execution stream");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamCreateWithFlags(&status_stream_, cudaStreamNonBlocking),
                                 "Failed to create status stream");
}

GPUResidentDeck::~GPUResidentDeck() {
  try {
    // Clean up CUDA streams
    if (execution_stream_) {
      HOLOSCAN_CUDA_CALL_ERR_MSG(cudaStreamDestroy(execution_stream_),
                                 "Failed to destroy execution stream");
    }
    if (status_stream_) {
      HOLOSCAN_CUDA_CALL_ERR_MSG(cudaStreamDestroy(status_stream_),
                                 "Failed to destroy status stream");
    }
  } catch (...) {
    // discard any exception raised by logging calls
  }
}

std::future<void> GPUResidentDeck::launch_cuda_graph(cudaGraphExec_t graph_exec) {
  // Launch the CUDA graph asynchronously and return a future
  return std::async(std::launch::async, [this, graph_exec]() {
    // Use cuda::std::atomic_ref for CPU-GPU visibility
    cuda::std::atomic_ref<unsigned int> data_ready_atomic(
        *reinterpret_cast<unsigned int*>(cpu_data_ready_trigger_->data()));
    cuda::std::atomic_ref<unsigned int> result_ready_atomic(
        *reinterpret_cast<unsigned int*>(cpu_result_ready_trigger_->data()));
    cuda::std::atomic_ref<unsigned int> tear_down_atomic(
        *reinterpret_cast<unsigned int*>(cpu_tear_down_trigger_->data()));

    // Atomic stores with memory_order_release to ensure GPU sees the updates
    data_ready_atomic.store(static_cast<unsigned int>(holoscan::ControlCommand::DATA_NOT_READY),
                            cuda::std::memory_order_release);
    result_ready_atomic.store(static_cast<unsigned int>(holoscan::ControlCommand::RESULT_NOT_READY),
                              cuda::std::memory_order_release);
    tear_down_atomic.store(static_cast<unsigned int>(holoscan::ControlCommand::INVALID),
                           cuda::std::memory_order_release);
    HOLOSCAN_LOG_DEBUG("GPU-resident CUDA graph is being launched.");
    // Launch the graph on the execution stream
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGraphLaunch(graph_exec, execution_stream_),
                                   "Failed to launch CUDA graph");
    // Set the atomic flag to indicate the graph has been launched
    graph_launched_.store(true);
    // check if timeout is zero
    // if not zero, then sleep for timeout and then send a tear down trigger
    // if zero, then wait for the stream to synchronize
    if (timeout_ms_ != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms_));
      tear_down();
    } else {
      HOLOSCAN_LOG_DEBUG("GPU-resident execution stream is being synchronized.");
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(execution_stream_),
                                     "Failed to synchronize execution stream");
    }
  });
}

bool GPUResidentDeck::result_ready() {
  cuda::std::atomic_ref<unsigned int> result_ready_atomic(
      *reinterpret_cast<unsigned int*>(cpu_result_ready_trigger_->data()));
  return result_ready_atomic.load(cuda::std::memory_order_acquire) ==
         static_cast<unsigned int>(holoscan::ControlCommand::RESULT_READY);
}

void GPUResidentDeck::tear_down() {
  HOLOSCAN_LOG_DEBUG("Tearing down GPU-resident workload.");
  cuda::std::atomic_ref<unsigned int> tear_down_atomic(
      *reinterpret_cast<unsigned int*>(cpu_tear_down_trigger_->data()));
  tear_down_atomic.store(static_cast<unsigned int>(holoscan::ControlCommand::TEAR_DOWN),
                         cuda::std::memory_order_release);
  graph_launched_.store(false);
  HOLOSCAN_LOG_INFO(
      "Torn down GPU-resident workload. Waiting for execution stream to synchronize.");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(execution_stream_),
                                 "Failed to synchronize execution stream");
}

void GPUResidentDeck::set_data_ready() {
  cuda::std::atomic_ref<unsigned int> data_ready_atomic(
      *reinterpret_cast<unsigned int*>(cpu_data_ready_trigger_->data()));
  data_ready_atomic.store(static_cast<unsigned int>(holoscan::ControlCommand::DATA_READY),
                          cuda::std::memory_order_release);

  // mark result as not ready atomically
  cuda::std::atomic_ref<unsigned int> result_ready_atomic(
      *reinterpret_cast<unsigned int*>(cpu_result_ready_trigger_->data()));
  result_ready_atomic.store(static_cast<unsigned int>(holoscan::ControlCommand::RESULT_NOT_READY),
                            cuda::std::memory_order_release);
}

bool GPUResidentDeck::is_launched() const {
  return graph_launched_.load();
}

}  // namespace holoscan
