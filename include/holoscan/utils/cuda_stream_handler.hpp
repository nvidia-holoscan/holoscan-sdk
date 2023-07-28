/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>
#include <vector>

#include "../core/operator_spec.hpp"
#include "../core/parameter.hpp"
#include "../core/resources/gxf/cuda_stream_pool.hpp"
#include "gxf/cuda/cuda_stream.hpp"
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
 * - call CudaStreamHandler::registerInterface(spec) from the operator setup() function
 * - in the compute() function call CudaStreamHandler::fromMessage(), this will get the CUDA stream
 *   from the message of the previous operator. When the operator receives multiple messages, then
 *   call CudaStreamHandler::fromMessages(). This will synchronize with multiple streams.
 * - when executing CUDA functions CudaStreamHandler::get() to get the CUDA stream which should
 *   be used by your CUDA function
 * - before publishing the output message(s) of your operator call CudaStreamHandler::toMessage() on
 *   each message. This will add the CUDA stream used by the CUDA functions in your operator to
 *   the output message.
 */
class CudaStreamHandler {
 public:
  /**
   * @brief Destroy the CudaStreamHandler object
   */
  ~CudaStreamHandler() {
    for (auto&& event : cuda_events_) {
      const cudaError_t result = cudaEventDestroy(event);
      if (cudaSuccess != result) {
        HOLOSCAN_LOG_ERROR("Failed to destroy CUDA event: %s", cudaGetErrorString(result));
      }
    }
    cuda_events_.clear();
  }

  /**
   * Define the parameters used by this class.
   *
   * @param spec      OperatorSpec to define the cuda_stream_pool parameter
   * @param required  if set then it's required that the CUDA stream pool is specified
   */
  void defineParams(OperatorSpec& spec, bool required = false) {
    spec.param(cuda_stream_pool_,
               "cuda_stream_pool",
               "CUDA Stream Pool",
               "Instance of gxf::CudaStreamPool.");
    cuda_stream_pool_required_ = required;
  }

  /**
   * Get the CUDA stream for the operation from the incoming message
   *
   * @param context
   * @param message
   * @return gxf_result_t
   */
  gxf_result_t fromMessage(gxf_context_t context,
                           const nvidia::gxf::Expected<nvidia::gxf::Entity>& message) {
    // if the message contains a stream use this
    const auto maybe_cuda_stream_id = message.value().get<nvidia::gxf::CudaStreamId>();
    if (maybe_cuda_stream_id) {
      const auto maybe_cuda_stream_handle = nvidia::gxf::Handle<nvidia::gxf::CudaStream>::Create(
          context, maybe_cuda_stream_id.value()->stream_cid);
      if (maybe_cuda_stream_handle) {
        message_cuda_stream_handle_ = maybe_cuda_stream_handle.value();
      }
    } else {
      // if no stream had been found, allocate a stream and use that
      gxf_result_t result = allocateInternalStream(context);
      if (result != GXF_SUCCESS) { return result; }
      message_cuda_stream_handle_ = cuda_stream_handle_;
    }
    return GXF_SUCCESS;
  }

  /**
   * Get the CUDA stream for the operation from the incoming messages
   *
   * @param context
   * @param messages
   * @return gxf_result_t
   */
  gxf_result_t fromMessages(gxf_context_t context,
                            const std::vector<nvidia::gxf::Entity>& messages) {
    const gxf_result_t result = allocateInternalStream(context);
    if (result != GXF_SUCCESS) { return result; }

    if (!cuda_stream_handle_) {
      // if no CUDA stream can be allocated because no stream pool is set, then don't sync
      // with incoming streams. CUDA operations of this operator will use the default stream
      // which sync with all other streams by default.
      return GXF_SUCCESS;
    }

    // iterate through all messages and use events to chain incoming streams with the internal
    // stream
    auto event_it = cuda_events_.begin();
    for (auto& msg : messages) {
      const auto maybe_cuda_stream_id = msg.get<nvidia::gxf::CudaStreamId>();
      if (maybe_cuda_stream_id) {
        const auto maybe_cuda_stream_handle = nvidia::gxf::Handle<nvidia::gxf::CudaStream>::Create(
            context, maybe_cuda_stream_id.value()->stream_cid);
        if (maybe_cuda_stream_handle) {
          const cudaStream_t cuda_stream = maybe_cuda_stream_handle.value()->stream().value();
          cudaError_t result;

          // allocate a new event if needed
          if (event_it == cuda_events_.end()) {
            cudaEvent_t cuda_event;
            result = cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming);
            if (cudaSuccess != result) {
              HOLOSCAN_LOG_ERROR("Failed to create input CUDA event: %s",
                                 cudaGetErrorString(result));
              return GXF_FAILURE;
            }
            cuda_events_.push_back(cuda_event);
            event_it = cuda_events_.end();
            --event_it;
          }

          result = cudaEventRecord(*event_it, cuda_stream);
          if (cudaSuccess != result) {
            HOLOSCAN_LOG_ERROR("Failed to record event for message stream: %s",
                          cudaGetErrorString(result));
            return GXF_FAILURE;
          }
          result = cudaStreamWaitEvent(cuda_stream_handle_->stream().value(), *event_it);
          if (cudaSuccess != result) {
            HOLOSCAN_LOG_ERROR("Failed to record wait on message event: %s",
                               cudaGetErrorString(result));
            return GXF_FAILURE;
          }
          ++event_it;
        }
      }
    }
    message_cuda_stream_handle_ = cuda_stream_handle_;
    return GXF_SUCCESS;
  }

  /**
   * Add the used CUDA stream to the outgoing message
   *
   * @param message
   * @return gxf_result_t
   */
  gxf_result_t toMessage(nvidia::gxf::Expected<nvidia::gxf::Entity>& message) {
    if (message_cuda_stream_handle_) {
      const auto maybe_stream_id = message.value().add<nvidia::gxf::CudaStreamId>();
      if (!maybe_stream_id) {
        HOLOSCAN_LOG_ERROR("Failed to add CUDA stream id to output message.");
        return nvidia::gxf::ToResultCode(maybe_stream_id);
      }
      maybe_stream_id.value()->stream_cid = message_cuda_stream_handle_.cid();
    }
    return GXF_SUCCESS;
  }

  /**
   * Get the CUDA stream handle which should be used for CUDA commands
   *
   * @param context
   * @return nvidia::gxf::Handle<nvidia::gxf::CudaStream>
   */
  nvidia::gxf::Handle<nvidia::gxf::CudaStream> getStreamHandle(gxf_context_t context) {
    // If there is a message stream handle, return this
    if (message_cuda_stream_handle_) { return message_cuda_stream_handle_; }

    // else allocate an internal CUDA stream and return it
    allocateInternalStream(context);
    return cuda_stream_handle_;
  }

  /**
   * Get the CUDA stream which should be used for CUDA commands.
   *
   * If no message stream is set and no stream can be allocated, return the default stream.
   *
   * @param context
   * @return cudaStream_t
   */
  cudaStream_t getCudaStream(gxf_context_t context) {
    const nvidia::gxf::Handle<nvidia::gxf::CudaStream> cuda_stream_handle =
        getStreamHandle(context);
    if (cuda_stream_handle) { return cuda_stream_handle->stream().value(); }
    if (!default_stream_warning_) {
      default_stream_warning_ = true;
      HOLOSCAN_LOG_WARN(
          "Parameter `cuda_stream_pool` is not set, using the default CUDA stream for CUDA "
          "operations.");
    }
    return cudaStreamDefault;
  }

 private:
  /**
   * Allocate the internal CUDA stream
   *
   * @param context
   * @return gxf_result_t
   */
  gxf_result_t allocateInternalStream(gxf_context_t context) {
    // Create the CUDA stream if it does not yet exist.
    if (!cuda_stream_handle_) {
      // Check if a cuda stream pool is given.
      const bool has_cuda_stream_pool_ = cuda_stream_pool_.has_value() && cuda_stream_pool_.get();
      if (!has_cuda_stream_pool_) {
        // If the cuda stream pool is required return an error
        if (cuda_stream_pool_required_) {
          HOLOSCAN_LOG_ERROR("'cuda_stream_pool' is required but not set.");
          return GXF_FAILURE;
        }
        return GXF_SUCCESS;
      }

      // get Handle to underlying nvidia::gxf::CudaStreamPool from
      // std::shared_ptr<holoscan::CudaStreamPool>
      const auto cuda_stream_pool = nvidia::gxf::Handle<nvidia::gxf::CudaStreamPool>::Create(
          context, cuda_stream_pool_.get()->gxf_cid());
      if (cuda_stream_pool) {
        // allocate a stream
        auto maybe_stream = cuda_stream_pool.value()->allocateStream();
        if (!maybe_stream) {
          HOLOSCAN_LOG_ERROR("Failed to allocate CUDA stream");
          return nvidia::gxf::ToResultCode(maybe_stream);
        }
        cuda_stream_handle_ = std::move(maybe_stream.value());
      }
    }
    return GXF_SUCCESS;
  }

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
