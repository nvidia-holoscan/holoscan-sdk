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

#ifndef HOLOSCAN_CORE_GXF_GXF_EXECUTION_CONTEXT_HPP
#define HOLOSCAN_CORE_GXF_GXF_EXECUTION_CONTEXT_HPP

#include <gxf/core/gxf.h>

#include <memory>
#include <string>
#include <vector>

#include "../execution_context.hpp"
#include "./gxf_cuda.hpp"
#include "./gxf_io_context.hpp"

namespace holoscan::gxf {

class GXFWrapper;  // forward declaration

/**
 * @brief Class to hold the execution context for GXF Operator.
 *
 * This class provides the execution context for the operator using GXF.
 */
class GXFExecutionContext : public holoscan::ExecutionContext {
 public:
  /**
   * @brief Construct a new GXFExecutionContext object
   *
   * @param context The pointer to the GXF context.
   * @param op The pointer to the operator.
   */
  GXFExecutionContext(gxf_context_t context, Operator* op);

  /**
   * @brief Construct a new GXFExecutionContext object
   *
   * @param context The pointer to the GXF context.
   * @param gxf_input_context The shared pointer to the GXFInputContext object.
   * @param gxf_output_context The shared pointer to the GXFOutputContext object.
   */
  GXFExecutionContext(gxf_context_t context, std::shared_ptr<GXFInputContext> gxf_input_context,
                      std::shared_ptr<GXFOutputContext> gxf_output_context);

  ~GXFExecutionContext() override = default;
  /**
   * @brief Get the GXF input context.
   *
   * @return The pointer to the GXFInputContext object.
   */
  std::shared_ptr<GXFInputContext> gxf_input() { return gxf_input_context_; }

  /**
   * @brief Get the GXF output context.
   *
   * @return The pointer to the GXFOutputContext object.
   */
  std::shared_ptr<GXFOutputContext> gxf_output() { return gxf_output_context_; }

  /// @brief allocate a new GXF CudaStream object and return the cudaStream_t corresponding to it
  expected<cudaStream_t, RuntimeError> allocate_cuda_stream(
      const std::string& stream_name) override;

  // @brief synchronize all of the streams in cuda_streams with target_cuda_stream
  void synchronize_streams(const std::vector<std::optional<cudaStream_t>>& cuda_streams,
                           cudaStream_t target_cuda_stream) override;

  // @brief determine the CUDA device corresponding to the given stream
  expected<int, RuntimeError> device_from_stream(cudaStream_t stream) override;

  /**
   * @brief Return the CudaStreamHandle corresponding to a given cudaStream_t.
   *
   * This will only work with a cudaStream_t that was allocated as a CudaStream object by GXF.
   * The stream should correspond to a CudaStreamId that was received on one of the Operator's
   * input ports or a stream that was allocated via `allocate_cuda_stream`.
   *
   * @param stream_handle A CUDA stream object.
   * @return The GXF CudaStream handle if found, or unexpected if not found.
   */
  expected<gxf::CudaStreamHandle, RuntimeError> stream_handle_from_stream(cudaStream_t stream);

  std::shared_ptr<CudaObjectHandler> cuda_object_handler() { return cuda_object_handler_; }

  /// @brief initialize the CudaObjectHandler for the Operator
  void init_cuda_object_handler(Operator* op);

  /// @brief release any internal stream objects allocated by the operator
  void release_internal_cuda_streams();

  /// @brief clear the handler's received stream mappings from a prior `Operator::compute` call.
  void clear_received_streams();

 protected:
  friend class holoscan::gxf::GXFWrapper;
  friend class holoscan::gxf::GXFInputContext;
  friend class holoscan::gxf::GXFOutputContext;

  /// @brief allocate a new GXF CudaStream object and return the GXF Handle to it
  expected<CudaStreamHandle, RuntimeError> allocate_cuda_stream_handle(
      const std::string& stream_name);

  std::shared_ptr<GXFInputContext> gxf_input_context_{};    ///< The GXF input context.
  std::shared_ptr<GXFOutputContext> gxf_output_context_{};  ///< The GXF output context.
  std::shared_ptr<CudaObjectHandler> cuda_object_handler_{};
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_EXECUTION_CONTEXT_HPP */
