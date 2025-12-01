/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_EXECUTION_CONTEXT_HPP
#define HOLOSCAN_CORE_EXECUTION_CONTEXT_HPP

#include <memory>
#include <string>
#include <vector>

// Forward declaration to avoid including <cuda_runtime.h>
extern "C" {
typedef struct CUstream_st* cudaStream_t;
}

#include "./common.hpp"
#include "./errors.hpp"
#include "./expected.hpp"
#include "./io_context.hpp"
#include "./operator_status.hpp"
#include "holoscan/profiler/profiler.hpp"

// ExecutionContext CUDA stream profiling events (use same green color for consistency)
PROF_DEFINE_EVENT(event_allocate_cuda_stream, "allocate_cuda_stream", 0x99, 0xFF, 0x00);
PROF_DEFINE_EVENT(event_synchronize_streams, "synchronize_streams", 0x99, 0xFF, 0x00);
PROF_DEFINE_EVENT(event_device_from_stream, "device_from_stream", 0x99, 0xFF, 0x00);

namespace holoscan {

/**
 * @brief Class to hold the execution context.
 *
 * This class provides the execution context for the operator.
 */
class ExecutionContext {
 public:
  /**
   * @brief Construct a new Execution Context object.
   */
  ExecutionContext() = default;

  virtual ~ExecutionContext() = default;

  /**
   * @brief Get the input context.
   *
   * @return The shared pointer to the input context.
   */
  std::shared_ptr<InputContext> input() const { return input_context_; }

  /**
   * @brief Get the output context.
   *
   * @return The shared pointer to the output context.
   */
  std::shared_ptr<OutputContext> output() const { return output_context_; }

  /**
   * @brief Get the context.
   *
   * @return The pointer to the context.
   */
  void* context() const { return context_; }

  /// @brief allocate a new GXF CudaStream object and return the contained cudaStream_t
  virtual expected<cudaStream_t, RuntimeError> allocate_cuda_stream(
      [[maybe_unused]] const std::string& stream_name = "") {
    return make_unexpected(RuntimeError(
        ErrorCode::kFailure, "allocate_cuda_stream not implemented in base ExecutionContext"));
  }

  /// @brief synchronize all of the streams in cuda_streams with target_cuda_stream
  virtual void synchronize_streams(
      [[maybe_unused]] const std::vector<std::optional<cudaStream_t>>& cuda_streams,
      [[maybe_unused]] cudaStream_t target_cuda_stream) {
    HOLOSCAN_LOG_ERROR("synchronize_streams not implemented in base ExecutionContext");
  }

  /// @brief determine the CUDA device corresponding to the given stream
  virtual expected<int, RuntimeError> device_from_stream([[maybe_unused]] cudaStream_t stream) {
    return make_unexpected(RuntimeError(
        ErrorCode::kFailure, "device_from_stream not implemented in base ExecutionContext"));
  }

  /// @brief check if GPU capability is present on the system
  virtual bool is_gpu_available() const {
    throw std::runtime_error("is_gpu_available is not implemented");
  }

  /**
   * @brief Find an operator by name.
   *
   * If the operator name is not provided, the current operator is returned.
   *
   * @param op_name The name of the operator.
   * @return A shared pointer to the operator or nullptr if the operator is not found.
   */
  virtual std::shared_ptr<Operator> find_operator(
      [[maybe_unused]] const std::string& op_name = "") {
    return nullptr;
  }

  /**
   * @brief Get the status of the operator.
   *
   * If the operator name is not provided, the status of the current operator is returned.
   *
   * @param op_name The name of the operator.
   * @return The status of the operator or an error if the operator is not found.
   */
  virtual expected<holoscan::OperatorStatus, RuntimeError> get_operator_status(
      [[maybe_unused]] const std::string& op_name = "") {
    return make_unexpected(RuntimeError(
        ErrorCode::kFailure, "get_operator_status not implemented in base ExecutionContext"));
  }

 protected:
  std::shared_ptr<InputContext> input_context_ = nullptr;    ///< The input context.
  std::shared_ptr<OutputContext> output_context_ = nullptr;  ///< The output context.
  void* context_ = nullptr;                                  ///< The context.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_EXECUTION_CONTEXT_HPP */
