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

#ifndef HOLOSCAN_CORE_CUDA_OBJECT_HANDLER_HPP
#define HOLOSCAN_CORE_CUDA_OBJECT_HANDLER_HPP

#include <optional>
#include <string>
#include <vector>

// Forward declaration to avoid including <cuda_runtime.h>
extern "C" {
typedef struct CUstream_st* cudaStream_t;
}

namespace holoscan {

class Operator;

/**
 * @brief Pure virtual base class for handling CUDA streams in operators.
 *
 * This class defines the interface for managing CUDA streams in Holoscan operators.
 * It provides methods for stream synchronization, allocation, and management.
 */
class CudaObjectHandler {
 public:
  virtual ~CudaObjectHandler() = default;

  /**
   * @brief Initialize the handler from an operator.
   *
   * @param op The operator this instance is attached to.
   */
  virtual void init_from_operator(Operator* op) = 0;

  /**
   * @brief Add a CUDA stream to an output port.
   *
   * @param stream The stream to add
   * @param output_port_name The name of the output port
   * @return 0 if successful, otherwise an error code
   */
  virtual int add_stream(const cudaStream_t stream, const std::string& output_port_name) = 0;

  /**
   * @brief Get the CUDA stream for a given input port.
   *
   * @param context context object (e.g. gxf_context_t for the GXF backend)
   * @param input_port_name The name of the input port
   * @param allocate If true, allocate a new stream if none exists
   * @param sync_to_default If true, synchronize to default stream
   * @return The CUDA stream to use
   */
  virtual cudaStream_t get_cuda_stream(void* context, const std::string& input_port_name,
                                       bool allocate = false, bool sync_to_default = true) = 0;

  /**
   * @brief Get all CUDA streams for a given input port.
   *
   * @param context context object (e.g. gxf_context_t for the GXF backend)
   * @param input_port_name The name of the input port
   * @return Vector of optional CUDA streams (one per message)
   */
  virtual std::vector<std::optional<cudaStream_t>> get_cuda_streams(
      void* context, const std::string& input_port_name) = 0;

  /**
   * @brief Synchronize streams with a target stream.
   *
   * @param cuda_streams The streams to synchronize
   * @param target_stream The stream to synchronize to
   * @param sync_to_default_stream If true, also sync to default stream
   * @return 0 if successful, error code otherwise
   */
  virtual int synchronize_streams(std::vector<cudaStream_t> cuda_streams,
                                  cudaStream_t target_stream,
                                  bool sync_to_default_stream = true) = 0;

  /**
   * @brief Release all internally allocated CUDA streams.
   *
   * @param context context object (e.g. gxf_context_t for the GXF backend)
   * @return 0 if successful, error code otherwise
   */
  virtual int release_internal_streams(void* context) = 0;

  /**
   * @brief Clear all received streams.
   *
   * This is used to refresh the state before each compute call.
   */
  virtual void clear_received_streams() = 0;

  /**
   * @brief Check if GPU capability is present on the system.
   *
   * @return true if GPU(s) are available, false if no GPU is present
   */
  virtual bool is_gpu_available() const = 0;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CUDA_OBJECT_HANDLER_HPP */
