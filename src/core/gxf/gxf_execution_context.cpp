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

#include "holoscan/core/gxf/gxf_execution_context.hpp"

#include <gxf/core/gxf.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/gxf/gxf_io_context.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/gxf/gxf_wrapper.hpp"
#include "holoscan/core/operator.hpp"

namespace holoscan::gxf {

GXFExecutionContext::GXFExecutionContext(gxf_context_t context, Operator* op) {
  gxf_input_context_ = std::make_shared<GXFInputContext>(this, op);
  gxf_output_context_ = std::make_shared<GXFOutputContext>(this, op);

  context_ = context;
  input_context_ = gxf_input_context_.get();
  output_context_ = gxf_output_context_.get();
}

GXFExecutionContext::GXFExecutionContext(gxf_context_t context,
                                         std::shared_ptr<GXFInputContext> gxf_input_context,
                                         std::shared_ptr<GXFOutputContext> gxf_output_context)
    : gxf_input_context_(std::move(gxf_input_context)),
      gxf_output_context_(std::move(gxf_output_context)) {
  context_ = context;
}

void GXFExecutionContext::init_cuda_object_handler(Operator* op) {
  cuda_object_handler_ = std::make_shared<CudaObjectHandler>();
  cuda_object_handler_->init_from_operator(op);
}

void GXFExecutionContext::release_internal_cuda_streams() {
  if (cuda_object_handler_ != nullptr) { cuda_object_handler_->release_internal_streams(context_); }
}

void GXFExecutionContext::clear_received_streams() {
  if (cuda_object_handler_ != nullptr) { cuda_object_handler_->clear_received_streams(); }
}

void GXFExecutionContext::synchronize_streams(
    const std::vector<std::optional<cudaStream_t>>& cuda_streams, cudaStream_t target_cuda_stream) {
  if (cuda_object_handler_ == nullptr) {
    throw std::runtime_error("Failed to sync streams: cuda_object_handler_ is nullptr");
  }
  // create new vector omitting the nullopt values
  std::vector<cudaStream_t> streams;
  streams.reserve(cuda_streams.size());
  for (const auto& stream : cuda_streams) {
    if (stream.has_value()) { streams.push_back(stream.value()); }
  }
  auto gxf_result = cuda_object_handler_->synchronize_streams(streams, target_cuda_stream);
  if (gxf_result != GXF_SUCCESS) {
    throw std::runtime_error(fmt::format("Failed to sync streams: {}", GxfResultStr(gxf_result)));
  }
}

expected<CudaStreamHandle, RuntimeError> GXFExecutionContext::allocate_cuda_stream_handle(
    const std::string& stream_name) {
  if (cuda_object_handler_ != nullptr) {
    return cuda_object_handler_->allocate_internal_stream(context_, stream_name);
  }
  return make_unexpected(RuntimeError(
      ErrorCode::kFailure, "CudaObjectHandler is not initialized, could not allocate a stream"));
}

expected<cudaStream_t, RuntimeError> GXFExecutionContext::allocate_cuda_stream(
    const std::string& stream_name) {
  auto maybe_stream_handle = allocate_cuda_stream_handle(stream_name);
  if (!maybe_stream_handle) { return forward_error(maybe_stream_handle); }
  return maybe_stream_handle.value()->stream().value();
}

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
expected<gxf::CudaStreamHandle, RuntimeError> GXFExecutionContext::stream_handle_from_stream(
    cudaStream_t stream) {
  if (cuda_object_handler_ != nullptr) {
    return cuda_object_handler_->stream_handle_from_stream(stream);
  }
  return make_unexpected(RuntimeError(
      ErrorCode::kFailure, "CudaObjectHandler is not initialized, could not find stream handle"));
}

// @brief determine the CUDA device corresponding to the given stream
expected<int, RuntimeError> GXFExecutionContext::device_from_stream(cudaStream_t stream) {
  auto maybe_handle = stream_handle_from_stream(stream);
  if (maybe_handle) { return maybe_handle.value()->dev_id(); }
  return make_unexpected(RuntimeError(ErrorCode::kFailure,
                                      "device_from_stream only supports retrieving the device ID "
                                      "from streams being managed by Holoscan SDK"));
}

}  // namespace holoscan::gxf
