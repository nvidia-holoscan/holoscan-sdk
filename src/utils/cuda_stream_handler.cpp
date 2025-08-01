/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/utils/cuda_stream_handler.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

namespace holoscan {

CudaStreamHandler::~CudaStreamHandler() {
  for (auto&& event : cuda_events_) {
    const cudaError_t result = cudaEventDestroy(event);
    if (cudaSuccess != result) {
      try {
        HOLOSCAN_LOG_ERROR("Failed to destroy CUDA event: {}", cudaGetErrorString(result));
      } catch (std::exception& e) {
      }
    }
  }
  cuda_events_.clear();
}

void CudaStreamHandler::define_params(OperatorSpec& spec, bool required) {
  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "CUDA Stream Pool",
             "Instance of gxf::CudaStreamPool.");
  cuda_stream_pool_required_ = required;
}

void CudaStreamHandler::defineParams(OperatorSpec& spec, bool required) {
  static bool warned = false;
  if (!warned) {
    warned = true;
    HOLOSCAN_LOG_WARN(
        "CudaStreamHandler's `defineParams` method has been renamed to `define_params`. "
        "The old name is deprecated and may be removed in a future release.");
  }
  return define_params(spec, required);
}

gxf_result_t CudaStreamHandler::from_message(
    gxf_context_t context, const nvidia::gxf::Expected<nvidia::gxf::Entity>& message) {
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
    gxf_result_t result = allocate_internal_stream(context);
    if (result != GXF_SUCCESS) {
      return result;
    }
    message_cuda_stream_handle_ = cuda_stream_handle_;
  }
  return GXF_SUCCESS;
}

gxf_result_t CudaStreamHandler::fromMessage(
    gxf_context_t context, const nvidia::gxf::Expected<nvidia::gxf::Entity>& message) {
  static bool warned = false;
  if (!warned) {
    warned = true;
    HOLOSCAN_LOG_WARN(
        "CudaStreamHandler's `fromMessage` method has been renamed to `from_message`. "
        "The old name is deprecated and may be removed in a future release.");
  }
  return from_message(context, message);
}

gxf_result_t CudaStreamHandler::from_messages(gxf_context_t context,
                                              const std::vector<holoscan::gxf::Entity>& messages) {
  // call the common internal version using the pointer to the vector data, this only works
  // if the size of the nvidia and holoscan gxf::Entity versions is identical
  static_assert(sizeof(holoscan::gxf::Entity) == sizeof(nvidia::gxf::Entity));
  return from_messages(context, messages.size(), messages.data());
}

gxf_result_t CudaStreamHandler::from_messages(gxf_context_t context,
                                              const std::vector<nvidia::gxf::Entity>& messages) {
  // call the common internal version using the pointer to the vector data, this only works
  // if the size of the nvidia and holoscan gxf::Entity versions is identical
  static_assert(sizeof(holoscan::gxf::Entity) == sizeof(nvidia::gxf::Entity));
  return from_messages(context, messages.size(), messages.data());
}

gxf_result_t CudaStreamHandler::from_messages(gxf_context_t context, size_t message_count,
                                              const nvidia::gxf::Entity* messages) {
  const gxf_result_t result = allocate_internal_stream(context);
  if (result != GXF_SUCCESS) {
    return result;
  }

  if (!cuda_stream_handle_) {
    // if no CUDA stream can be allocated because no stream pool is set, then don't sync
    // with incoming streams. CUDA operations of this operator will use the default stream
    // which sync with all other streams by default.
    return GXF_SUCCESS;
  }

  // iterate through all messages and use events to chain incoming streams with the internal
  // stream
  auto event_it = cuda_events_.begin();
  for (size_t index = 0; index < message_count; ++index) {
    const auto maybe_cuda_stream_id = messages[index].get<nvidia::gxf::CudaStreamId>();
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
            HOLOSCAN_LOG_ERROR("Failed to create input CUDA event: {}", cudaGetErrorString(result));
            return GXF_FAILURE;
          }
          cuda_events_.push_back(cuda_event);
          event_it = cuda_events_.end();
          --event_it;
        }

        result = cudaEventRecord(*event_it, cuda_stream);
        if (cudaSuccess != result) {
          HOLOSCAN_LOG_ERROR("Failed to record event for message stream: {}",
                             cudaGetErrorString(result));
          return GXF_FAILURE;
        }
        result = cudaStreamWaitEvent(cuda_stream_handle_->stream().value(), *event_it);
        if (cudaSuccess != result) {
          HOLOSCAN_LOG_ERROR("Failed to record wait on message event: {}",
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

gxf_result_t CudaStreamHandler::fromMessages(gxf_context_t context,
                                             const std::vector<nvidia::gxf::Entity>& messages) {
  static bool warned = false;
  if (!warned) {
    warned = true;
    HOLOSCAN_LOG_WARN(
        "CudaStreamHandler's `fromMessages` method has been renamed to `from_messages`. "
        "The old name is deprecated and may be removed in a future release.");
  }
  return from_messages(context, messages);
}

gxf_result_t CudaStreamHandler::to_message(nvidia::gxf::Expected<nvidia::gxf::Entity>& message) {
  if (message_cuda_stream_handle_) {
    const auto maybe_stream_id = message.value().add<nvidia::gxf::CudaStreamId>("cuda_stream_id_");
    if (!maybe_stream_id) {
      HOLOSCAN_LOG_ERROR("Failed to add CUDA stream id to output message.");
      return nvidia::gxf::ToResultCode(maybe_stream_id);
    }
    maybe_stream_id.value()->stream_cid = message_cuda_stream_handle_.cid();
  }
  return GXF_SUCCESS;
}

gxf_result_t CudaStreamHandler::toMessage(nvidia::gxf::Expected<nvidia::gxf::Entity>& message) {
  static bool warned = false;
  if (!warned) {
    warned = true;
    HOLOSCAN_LOG_WARN(
        "CudaStreamHandler's `toMessage` method has been renamed to `to_message`. "
        "The old name is deprecated and may be removed in a future release.");
  }
  return to_message(message);
}

nvidia::gxf::Handle<nvidia::gxf::CudaStream> CudaStreamHandler::get_stream_handle(
    gxf_context_t context) {
  // If there is a message stream handle, return this
  if (message_cuda_stream_handle_) {
    return message_cuda_stream_handle_;
  }

  // else allocate an internal CUDA stream and return it
  allocate_internal_stream(context);
  return cuda_stream_handle_;
}

nvidia::gxf::Handle<nvidia::gxf::CudaStream> CudaStreamHandler::getStreamHandle(
    gxf_context_t context) {
  static bool warned = false;
  if (!warned) {
    warned = true;
    HOLOSCAN_LOG_WARN(
        "CudaStreamHandler's `getStreamHandle` method has been renamed to `get_stream_handle`. "
        "The old name is deprecated and may be removed in a future release.");
  }
  return get_stream_handle(context);
}

cudaStream_t CudaStreamHandler::get_cuda_stream(gxf_context_t context) {
  const nvidia::gxf::Handle<nvidia::gxf::CudaStream> cuda_stream_handle =
      get_stream_handle(context);
  if (cuda_stream_handle) {
    return cuda_stream_handle->stream().value();
  }
  if (!default_stream_warning_) {
    default_stream_warning_ = true;
    HOLOSCAN_LOG_WARN(
        "Parameter `cuda_stream_pool` is not set, using the default CUDA stream for CUDA "
        "operations.");
  }
  return cudaStreamDefault;
}

cudaStream_t CudaStreamHandler::getCudaStream(gxf_context_t context) {
  static bool warned = false;
  if (!warned) {
    warned = true;
    HOLOSCAN_LOG_WARN(
        "CudaStreamHandler's `getCudaStream` method has been renamed to `get_cuda_stream`. "
        "The old name is deprecated and may be removed in a future release.");
  }
  return get_cuda_stream(context);
}

gxf_result_t CudaStreamHandler::allocate_internal_stream(gxf_context_t context) {
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

}  // namespace holoscan
