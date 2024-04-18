/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef INCLUDE_HOLOSCAN_UTILS_CUDA_STREAM_HANDLER_HPP
#define INCLUDE_HOLOSCAN_UTILS_CUDA_STREAM_HANDLER_HPP

#include <memory>
#include <vector>

#include "../core/operator_spec.hpp"
#include "../core/parameter.hpp"
#include "../core/resources/gxf/cuda_stream_pool.hpp"
#include "gxf/cuda/cuda_stream.hpp"
// keep the following two gxf/cuda imports in the header for backwards compatibility with 1.0
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"

namespace holoscan {

/**
 * This class handles usage of CUDA streams for operators.
 *
 * When using CUDA operations the default stream '0' synchronizes with all other streams in the same
 * context, see
 * https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior.
 * This can reduce performance. The CudaStreamHandler class manages streams across operators and
 * makes sure that CUDA operations are properly chained.
 *
 * Usage:
 * - add an instance of CudaStreamHandler to your operator
 * - call CudaStreamHandler::register_interface(spec) from the operator setup() function
 * - in the compute() function call CudaStreamHandler::from_message(), this will get the CUDA stream
 *   from the message of the previous operator. When the operator receives multiple messages, then
 *   call CudaStreamHandler::from_messages(). This will synchronize with multiple streams.
 * - when executing CUDA functions CudaStreamHandler::get() to get the CUDA stream which should
 *   be used by your CUDA function
 * - before publishing the output message(s) of your operator call CudaStreamHandler::to_message()
 *   on each message. This will add the CUDA stream used by the CUDA functions in your operator to
 *   the output message.
 */
class CudaStreamHandler {
 public:
  /**
   * @brief Destroy the CudaStreamHandler object
   */
  ~CudaStreamHandler();

  /**
   * Define the parameters used by this class.
   *
   * @param spec      OperatorSpec to define the cuda_stream_pool parameter
   * @param required  if set then it's required that the CUDA stream pool is specified
   */
  void define_params(OperatorSpec& spec, bool required = false);

  /**
   * Define the parameters used by this class.
   *
   * This method is deprecated in favor of `define_params`.
   *
   * @deprecated since 1.0
   * @param spec      OperatorSpec to define the cuda_stream_pool parameter
   * @param required  if set then it's required that the CUDA stream pool is specified
   */
  void defineParams(OperatorSpec& spec, bool required = false);

  /**
   * Get the CUDA stream for the operation from the incoming message
   *
   * @param context
   * @param message
   * @return gxf_result_t
   */
  gxf_result_t from_message(gxf_context_t context,
                            const nvidia::gxf::Expected<nvidia::gxf::Entity>& message);

  /**
   * Get the CUDA stream for the operation from the incoming message
   *
   * This method is deprecated in favor of `from_message`.
   *
   * @deprecated since 1.0
   * @param context
   * @param message
   * @return gxf_result_t
   */
  gxf_result_t fromMessage(gxf_context_t context,
                           const nvidia::gxf::Expected<nvidia::gxf::Entity>& message);

  /**
   * Get the CUDA stream for the operation from the incoming messages
   *
   * @param context
   * @param messages
   * @return gxf_result_t
   */
  gxf_result_t from_messages(gxf_context_t context,
                             const std::vector<nvidia::gxf::Entity>& messages);
  /**
   * Get the CUDA stream for the operation from the incoming messages
   *
   * This method is deprecated in favor of `from_messages`.
   *
   * @deprecated since 1.0
   * @param context
   * @param messages
   * @return gxf_result_t
   */
  gxf_result_t fromMessages(gxf_context_t context,
                            const std::vector<nvidia::gxf::Entity>& messages);

  /**
   * Add the used CUDA stream to the outgoing message
   *
   * @param message
   * @return gxf_result_t
   */
  gxf_result_t to_message(nvidia::gxf::Expected<nvidia::gxf::Entity>& message);

  /**
   * Add the used CUDA stream to the outgoing message
   *
   * This method is deprecated in favor of `to_message`.
   *
   * @deprecated since 1.0
   * @param message
   * @return gxf_result_t
   */
  gxf_result_t toMessage(nvidia::gxf::Expected<nvidia::gxf::Entity>& message);

  /**
   * Get the CUDA stream handle which should be used for CUDA commands
   *
   * @param context
   * @return nvidia::gxf::Handle<nvidia::gxf::CudaStream>
   */
  nvidia::gxf::Handle<nvidia::gxf::CudaStream> get_stream_handle(gxf_context_t context);

  /**
   * Get the CUDA stream handle which should be used for CUDA commands
   *
   * This method is deprecated in favor of `get_stream_handle`.
   *
   * @deprecated since 1.0
   * @param context
   * @return nvidia::gxf::Handle<nvidia::gxf::CudaStream>
   */
  nvidia::gxf::Handle<nvidia::gxf::CudaStream> getStreamHandle(gxf_context_t context);

  /**
   * Get the CUDA stream which should be used for CUDA commands.
   *
   * If no message stream is set and no stream can be allocated, return the default stream.
   *
   * @param context
   * @return cudaStream_t
   */
  cudaStream_t get_cuda_stream(gxf_context_t context);

  /**
   * Get the CUDA stream which should be used for CUDA commands.
   *
   * If no message stream is set and no stream can be allocated, return the default stream.
   *
   * This method is deprecated in favor of `get_cuda_stream`.
   *
   * @deprecated since 1.0
   * @param context
   * @return cudaStream_t
   */
  cudaStream_t getCudaStream(gxf_context_t context);

 private:
  /**
   * Allocate the internal CUDA stream
   *
   * @param context
   * @return gxf_result_t
   */
  gxf_result_t allocate_internal_stream(gxf_context_t context);

  /// if set then it's required that the CUDA stream pool is specified, if this is not the case
  /// an error is generated
  bool cuda_stream_pool_required_ = false;

  /// CUDA stream pool used to allocate the internal CUDA stream
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_;

  /// If the CUDA stream pool is not set and we can't use the incoming CUDA stream, issue
  /// a warning once.
  bool default_stream_warning_ = false;

  /// Array of CUDA events used to synchronize the internal CUDA stream with multiple incoming
  /// streams
  std::vector<cudaEvent_t> cuda_events_;

  /// The CUDA stream which is attached to the incoming message
  nvidia::gxf::Handle<nvidia::gxf::CudaStream> message_cuda_stream_handle_;

  /// Allocated internal CUDA stream handle
  nvidia::gxf::Handle<nvidia::gxf::CudaStream> cuda_stream_handle_;
};

}  // namespace holoscan

#endif /* INCLUDE_HOLOSCAN_UTILS_CUDA_STREAM_HANDLER_HPP */
