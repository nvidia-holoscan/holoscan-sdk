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

#include "holoscan/core/gxf/gxf_cuda.hpp"

#include <gxf/core/gxf.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/utils/cuda_macros.hpp"

namespace holoscan::gxf {

namespace {
// Check GPU presence once and cache the result (thread-safe via C++11 static initialization)
bool check_gpu_present() {
  static const bool gpu_present = []() {
    int gpu_count = 0;
    cudaError_t cuda_err = HOLOSCAN_CUDA_CALL_WARN_MSG(
        cudaGetDeviceCount(&gpu_count),
        "Initializing CudaObjectHandler with support for CPU data only");

    if (cuda_err == cudaSuccess && gpu_count > 0) {
      HOLOSCAN_LOG_DEBUG("Detected {} GPU(s), CudaObjectHandler will support GPU operations.",
                         gpu_count);
      return true;
    } else {
      HOLOSCAN_LOG_DEBUG(
          "No GPU detected or CUDA unavailable, CudaObjectHandler functionality will be "
          "limited to CPU data.");
      return false;
    }
  }();

  return gpu_present;
}
}  // namespace

CudaObjectHandler::~CudaObjectHandler() {
  if (event_created_) {
    const cudaError_t result = cudaEventDestroy(cuda_event_);
    if (cudaSuccess != result) {
      try {
        HOLOSCAN_LOG_WARN("Failed to destroy CUDA event: {}", cudaGetErrorString(result));
      } catch (std::exception& e) {
      }
    }
  }
}

void CudaObjectHandler::init_from_operator(Operator* op) {
  if (op == nullptr) {
    HOLOSCAN_LOG_ERROR(
        "Operator is nullptr in CudaObjectHandler::init_from_operator. "
        "Cannot initialize CUDA stream handler.");
    return;
  }

  // Check for GPU presence (cached check, only done once globally)
  gpu_present_ = check_gpu_present();
  auto params = op->spec()->params();
  // If the operator has a parameter named cuda_stream_pool, reuse that
  auto param_iter = params.find("cuda_stream_pool");
  if (param_iter != params.end()) {
    try {
      auto param_wrapper = param_iter->second;
      HOLOSCAN_LOG_TRACE("cuda_stream_pool parameter found with type: {}",
                         param_wrapper.value().type().name());
      HOLOSCAN_LOG_TRACE(
          "trying to cast to type: {}",
          typeid(holoscan::MetaParameter<std::shared_ptr<holoscan::CudaStreamPool>>).name());
      auto pool_param =
          std::any_cast<holoscan::MetaParameter<std::shared_ptr<holoscan::CudaStreamPool>>*>(
              param_wrapper.value());
      if (pool_param->has_value() && pool_param->get() != nullptr) {
        cuda_stream_pool_ = pool_param->get();
        return;
      }
    } catch (const std::bad_any_cast& e) {
      HOLOSCAN_LOG_ERROR(
          "Failed to cast cuda_stream_pool parameter of operator '{}': {}", op->name(), e.what());
    }
  }

  std::shared_ptr<CudaStreamPool> cuda_stream_pool_ptr = nullptr;
  for (auto& resource : op->resources()) {
    // If the resource is a
    cuda_stream_pool_ptr = std::dynamic_pointer_cast<CudaStreamPool>(resource.second);
    if (cuda_stream_pool_ptr) {
      HOLOSCAN_LOG_DEBUG("Operator '{}': Found CudaStreamPool in resources", op->name());
      cuda_stream_pool_ = cuda_stream_pool_ptr;

      return;
    }
  }

  // Before adding default cuda_stream_pool, check if cuda_green_context_pool and cuda_green_context
  // are already in resources. A CudaGreenContextPool is mandatory. A CudaGreenContext is optional
  // and the default green context will be created if none is provided.
  std::shared_ptr<CudaGreenContextPool> cuda_green_context_pool_ptr = nullptr;
  for (auto& resource : op->resources()) {
    auto pool_resource = std::dynamic_pointer_cast<CudaGreenContextPool>(resource.second);
    if (pool_resource &&
        std::string(pool_resource->name()) != "fragment_default_green_context_pool") {
      HOLOSCAN_LOG_DEBUG("Operator '{}': Found CudaGreenContextPool in resources", op->name());
      cuda_green_context_pool_ptr = pool_resource;
      break;
    }
  }

  // Try to use the default green context pool from the fragment
  if (!cuda_green_context_pool_ptr) {
    HOLOSCAN_LOG_DEBUG("Operator '{}': using the default CudaGreenContextPool", op->name());
    cuda_green_context_pool_ptr = op->fragment()->get_default_green_context_pool();
  }

  if (cuda_green_context_pool_ptr) {
    HOLOSCAN_LOG_DEBUG("Operator '{}': initializing the CudaGreenContextPool", op->name());
    if (op->graph_entity()) {
      cuda_green_context_pool_ptr->gxf_eid(op->graph_entity()->eid());
    }
    cuda_green_context_pool_ptr->initialize();
  }

  // check if a CudaGreenContext is already in resources
  std::shared_ptr<CudaGreenContext> cuda_green_context_ptr = nullptr;
  for (auto& resource : op->resources()) {
    auto context_resource = std::dynamic_pointer_cast<CudaGreenContext>(resource.second);
    if (context_resource) {
      HOLOSCAN_LOG_DEBUG("Operator '{}': Found CudaGreenContext in resources", op->name());
      cuda_green_context_ptr = context_resource;
      break;
    }
  }

  // Create default Green context if not found in resources or spec parameters
  if (!cuda_green_context_ptr && cuda_green_context_pool_ptr) {
    HOLOSCAN_LOG_DEBUG("Operator '{}': creating the default CudaGreenContext", op->name());
    cuda_green_context_ptr =
        op->fragment()->make_resource<CudaGreenContext>(cuda_green_context_pool_ptr);
  }

  // Initialize the Green context
  if (cuda_green_context_ptr) {
    HOLOSCAN_LOG_DEBUG("Operator '{}': initializing the CudaGreenContext", op->name());
    if (op->graph_entity()) {
      cuda_green_context_ptr->gxf_eid(op->graph_entity()->eid());
    }
    cuda_green_context_ptr->initialize();
  }

  if (!cuda_stream_pool_ptr) {
    HOLOSCAN_LOG_DEBUG("Operator '{}': Creating default CudaStreamPool", op->name());
    // Add a CudaStreamPool with initial capacity 1 on the active device
    if (gpu_present_) {
      // determine the currently active GPU device
      int device = 0;
      cudaError_t cuda_status = HOLOSCAN_CUDA_CALL_WARN_MSG(
          cudaGetDevice(&device), "Failed to determine the currently active GPU device");
      if (cuda_status == cudaSuccess) {
        // Note: `op` will have already been initialized, so do not use
        // `op->add_cuda_stream_pool`. Instead, manually handle creating and initializing
        // the stream pool here.
        auto cuda_stream_pool = op->fragment()->make_resource<CudaStreamPool>(
            fmt::format("{}_stream_pool", op->name()).c_str(),
            device,
            0,
            0,
            1,
            0,
            cuda_green_context_ptr);
        // assign this new stream pool resource to the same entity as the operator
        if (op->graph_entity()) {
          cuda_stream_pool->gxf_eid(op->graph_entity()->eid());
        }
        HOLOSCAN_LOG_DEBUG("Operator '{}': Initializing CudaStreamPool in parameters", op->name());
        cuda_stream_pool->initialize();
        op->add_arg(cuda_stream_pool);
        cuda_stream_pool_ = cuda_stream_pool;
        HOLOSCAN_LOG_DEBUG(
            "No cuda_stream_pool parameter or resource found for operator '{}'."
            "Added a default CUDA stream pool with initial size 1 on device {}.",
            op->name(),
            device);
      }
    }
  }
}

gxf_result_t CudaObjectHandler::streams_from_message(gxf_context_t context,
                                                     const nvidia::gxf::Entity& message,
                                                     const std::string& input_name) {
  // truncate possible :0, :1, etc. from end of a multi-receiver (IOSpec::kAny) port name
  std::string input_key = input_name.substr(0, input_name.find(':'));

  const auto maybe_cuda_stream_id = message.get<nvidia::gxf::CudaStreamId>();
  if (maybe_cuda_stream_id) {
    HOLOSCAN_LOG_TRACE("CudaStreamId found for input '{}' (key='{}')", input_name, input_key);
    const auto& cuda_stream_id = maybe_cuda_stream_id.value();
    auto& id_vector = received_cuda_stream_ids_[input_key];
    id_vector.emplace_back(*(cuda_stream_id.get()));

    // TODO(grelee): populate this eagerly here, or delay until a handle is actually requested?
    const auto maybe_cuda_stream_handle =
        CudaStreamHandle::Create(context, cuda_stream_id->stream_cid);
    if (maybe_cuda_stream_handle) {
      auto& handle_vector = received_cuda_stream_handles_[input_key];
      auto& stream_handle = maybe_cuda_stream_handle.value();
      handle_vector.push_back(stream_handle);

      // keep reverse mapping from stream->handle to allow later use from add_stream
      auto maybe_stream = stream_handle->stream();
      if (maybe_stream) {
        stream_to_stream_handle_[maybe_stream.value()] = stream_handle;
      }
    }
  } else {
    auto& id_vector = received_cuda_stream_ids_[input_key];
    id_vector.push_back(std::nullopt);
    auto& handle_vector = received_cuda_stream_handles_[input_key];
    handle_vector.push_back(std::nullopt);
    // TODO: could allocate from the internal stream pool here, but to avoid overhead, we can
    // instead delay that to the point at which the user requests it.
    HOLOSCAN_LOG_TRACE("No CudaStreamId found for input '{}' (key='{}')", input_name, input_key);
  }
  return GXF_SUCCESS;
}

expected<gxf_uid_t, ErrorCode> CudaObjectHandler::get_output_stream_cid(
    const std::string& output_port_name) {
  if (emitted_cuda_stream_cids_.size() > 0) {
    auto emitted_iter = emitted_cuda_stream_cids_.find(output_port_name);
    if (emitted_iter != emitted_cuda_stream_cids_.end()) {
      HOLOSCAN_LOG_TRACE("get_output_stream_ci: found a stream cid for port '{}'",
                         output_port_name);
      return emitted_iter->second;
    }
  } else {
    // If the user didn't explicitly set a stream to output, use the internally allocated stream
    auto allocated_iter = allocated_cuda_stream_handles_.find("_internal");
    if (allocated_iter != allocated_cuda_stream_handles_.end()) {
      HOLOSCAN_LOG_TRACE("Using internally allocated stream for output port '{}'",
                         output_port_name);
      auto stream_handle = allocated_iter->second;
      return stream_handle.cid();
    }

    // If no internal stream could be allocated, emit the "_internal" received stream on the output
    auto received_iter = received_cuda_stream_handles_.find("_internal");
    if (received_iter != received_cuda_stream_handles_.end() && !received_iter->second.empty()) {
      HOLOSCAN_LOG_TRACE("Using received stream as the stream emitted for output port '{}'",
                         output_port_name);
      auto stream_handle = received_iter->second[0];
      if (!stream_handle.has_value()) {
        return make_unexpected<ErrorCode>(ErrorCode::kNotFound);
      }
      return stream_handle.value().cid();
    }
  }
  return make_unexpected<ErrorCode>(ErrorCode::kNotFound);
}

gxf_result_t CudaObjectHandler::add_stream(const CudaStreamHandle& stream_handle,
                                           const std::string& output_port_name) {
  HOLOSCAN_LOG_TRACE("Adding stream to output port '{}'", output_port_name);
  emitted_cuda_stream_cids_.emplace(output_port_name, stream_handle.cid());
  return GXF_SUCCESS;
}

int CudaObjectHandler::add_stream(const cudaStream_t stream, const std::string& output_port_name) {
  HOLOSCAN_LOG_TRACE("Adding stream to output port '{}'", output_port_name);
  auto it = stream_to_stream_handle_.find(stream);
  if (it != stream_to_stream_handle_.end()) {
    const auto& stream_handle = it->second;
    emitted_cuda_stream_cids_.emplace(output_port_name, stream_handle.cid());
    return static_cast<int>(GXF_SUCCESS);
  }
  return static_cast<int>(GXF_FAILURE);
}

expected<std::vector<std::optional<CudaStreamHandle>>, RuntimeError>
CudaObjectHandler::get_cuda_stream_handles(gxf_context_t context,
                                           const std::string& input_port_name) {
  // truncate possible :0, :1, etc. from end of a multi-receiver (IOSpec::kAny) port name
  std::string input_key = input_port_name.substr(0, input_port_name.find(':'));

  // If there is already a stream handle for this input port, return that
  auto it = received_cuda_stream_handles_.find(input_key);
  if (it != received_cuda_stream_handles_.end()) {
    return it->second;
  }
  HOLOSCAN_LOG_TRACE("get_cuda_stream_handles: no stream handles found");

  auto out = std::vector<std::optional<CudaStreamHandle>>{};

  // If the message contained a stream ID, retrieve the corresponding CudaStreamHandle
  auto id_it = received_cuda_stream_ids_.find(input_key);
  if (id_it != received_cuda_stream_ids_.end()) {
    const auto& stream_id_vec = id_it->second;
    if (stream_id_vec.empty()) {
      return out;
    }
    out.reserve(stream_id_vec.size());
    for (auto& maybe_id : stream_id_vec) {
      if (maybe_id) {
        // get the GXF Handle<CudaStream> object corresponding to the component ID
        const auto maybe_cuda_stream_handle =
            CudaStreamHandle::Create(context, maybe_id.value().stream_cid);
        if (maybe_cuda_stream_handle) {
          // TODO: also update received_cuda_stream_handles_ on demand in this function?
          out.emplace_back(maybe_cuda_stream_handle.value());
        } else {
          return make_unexpected<RuntimeError>(RuntimeError(
              ErrorCode::kFailure, "Failed to create CudaStreamHandle from stream ID."));
        }
      } else {
        HOLOSCAN_LOG_TRACE("get_cuda_stream_handles: no stream ids found");
        out.emplace_back(std::nullopt);
      }
    }
    return out;
  }
  auto err_msg = fmt::format("input_port_name '{}' (base name: '{}') not found",
                              input_port_name, input_key);
  return make_unexpected<RuntimeError>(RuntimeError(ErrorCode::kFailure, err_msg));
}

expected<CudaStreamHandle, RuntimeError> CudaObjectHandler::get_cuda_stream_handle(
    gxf_context_t context, const std::string& input_port_name, bool allocate,
    bool sync_to_default) {
  // truncate possible :0, :1, etc. from end of a multi-receiver (IOSpec::kAny) port name
  std::string input_key = input_port_name.substr(0, input_port_name.find(':'));

  // TODO (grelee): remove allocate=false code paths from this function?
  // If there is already a stream handle for this input port, return that
  auto received_iter = received_cuda_stream_handles_.find(input_key);

  CudaStreamHandle output_stream;
  if (allocate) {
    HOLOSCAN_LOG_TRACE("\tallocate = true branch");
    // Use the internally allocated stream as the output and synchronize all other streams to it.
    // This allocation currently requires a CUDA stream pool parameter to be set on the operator.
    // Except for the first time this is called, the already allocated "_internal" stream will be
    // used.
    auto maybe_stream = allocate_internal_stream(context, "_internal");
    if (maybe_stream.has_value()) {
      HOLOSCAN_LOG_TRACE("\t\tinternal stream allocation succeeded");
      output_stream = maybe_stream.value();

      // Set the active device to match the internal stream
      int gpu_id = output_stream->dev_id();
      HOLOSCAN_CUDA_CALL_ERR_MSG(cudaSetDevice(gpu_id), "Failed to set device id {}", gpu_id);

      // If there were any input streams, synchronize these to the allocated stream
      if (received_iter != received_cuda_stream_handles_.end()) {
        const auto& stream_handle_vec = received_iter->second;
        auto vec_size = stream_handle_vec.size();
        if (vec_size > 0) {
          HOLOSCAN_LOG_TRACE("\t\tSynchronizing input streams to the allocated stream");
          // synchronize all input streams to the internally allocated stream (or default stream)
          synchronize_streams(stream_handle_vec, output_stream, sync_to_default);
        }
      } else {
        HOLOSCAN_LOG_TRACE("\t\tSynchronizing internal stream with the default stream");
        synchronize_streams({output_stream->stream().value()}, cudaStreamDefault);
      }
      return output_stream;
    } else {
      HOLOSCAN_LOG_TRACE("\t\tinternal stream allocation failed");
    }
  }
  HOLOSCAN_LOG_TRACE("\tOn no-allocation code path now");

  auto internal_iter = received_cuda_stream_handles_.find("_internal");
  if (internal_iter != received_cuda_stream_handles_.end() && !internal_iter->second.empty()) {
    HOLOSCAN_LOG_TRACE("\t\tfound existing _internal received stream");
    // A different stream from a prior `receive_cuda_stream` call was already selected for use as
    // the internal stream. Return that stream if it was found and synchronize any other streams
    // to it.
    auto maybe_stream = internal_iter->second[0];
    if (!maybe_stream.has_value()) {
      auto err_msg =
          fmt::format("_internal stream was unexpectedly without a value", input_port_name);
      HOLOSCAN_LOG_TRACE(err_msg);
      return make_unexpected<RuntimeError>(RuntimeError(ErrorCode::kNotFound, err_msg));
    }
    output_stream = maybe_stream.value();
    int gpu_id = output_stream->dev_id();
    HOLOSCAN_CUDA_CALL_ERR_MSG(cudaSetDevice(gpu_id), "Failed to set device id {}", gpu_id);

    if (received_iter != received_cuda_stream_handles_.end()) {
      HOLOSCAN_LOG_TRACE("\t\tSynchronizing any received streams to the existing _internal one");
      const auto& stream_handle_vec = received_iter->second;
      synchronize_streams(stream_handle_vec, output_stream, sync_to_default);
    }
  }

  // Error if no stream was found from this call or a prior receive_stream call
  if (received_iter == received_cuda_stream_handles_.end() || received_iter->second.empty()) {
    auto err_msg =
        fmt::format("No stream found for input port '{}' and allocate=false", input_port_name);
    HOLOSCAN_LOG_TRACE(err_msg);
    return make_unexpected<RuntimeError>(RuntimeError(ErrorCode::kNotFound, err_msg));
  }

  // Select one of the received streams as the internal stream to use
  const auto& stream_handle_vec = received_iter->second;
  auto vec_size = stream_handle_vec.size();
  HOLOSCAN_LOG_TRACE("\t\tFound {} streams on the input", vec_size);

  // Use the first stream found on the input and synchronize any other input streams to it
  bool stream_found = false;
  for (size_t i = 0; i < vec_size; ++i) {
    // Find the first stream with a value
    if (stream_handle_vec[i].has_value()) {
      HOLOSCAN_LOG_TRACE("\t\tFound first stream at index {}", i);
      stream_found = true;
      output_stream = stream_handle_vec[i].value();
      if (!output_stream->stream().has_value()) {
        HOLOSCAN_LOG_TRACE(
            "\t\tCudaStreamHandle found on input '{}', does not contain a stream value.",
            input_port_name);
        continue;
      }

      // Sync any remaining streams with this first one (or sync all to the default stream)
      if (i + 1 < vec_size) {
        HOLOSCAN_LOG_TRACE("\t\tSynchronizing remaining streams as well");
        std::vector<std::optional<CudaStreamHandle>> remaining_streams(
            stream_handle_vec.begin() + i + 1, stream_handle_vec.end());
        synchronize_streams(std::move(remaining_streams), output_stream, sync_to_default);
      } else {
        HOLOSCAN_LOG_TRACE("\t\tSynchronizing received stream with the default stream");
        synchronize_streams({output_stream->stream().value()}, cudaStreamDefault);
      }
      break;
    }
  }
  if (!stream_found) {
    HOLOSCAN_LOG_TRACE("\tNo stream found or allocated: CudaStreamHandle could not be returned");
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kNotFound, "No CUDA stream was found and allocate=false."));
  }
  // mark this received stream as the "_internal" stream that will be automatically emitted
  if (internal_iter == received_cuda_stream_handles_.end() || internal_iter->second.empty()) {
    HOLOSCAN_LOG_TRACE("\t\tSetting the _internal received stream");
    auto& handle_vector = received_cuda_stream_handles_["_internal"];
    handle_vector.push_back(output_stream);
  }

  // Set the device to the device of the selected stream
  int gpu_id = output_stream->dev_id();
  HOLOSCAN_CUDA_CALL_ERR_MSG(cudaSetDevice(gpu_id), "Failed to set device id {}", gpu_id);

  return output_stream;
}

expected<nvidia::gxf::Handle<nvidia::gxf::CudaStreamPool>, RuntimeError>
CudaObjectHandler::cuda_stream_pool_handle(gxf_context_t context) {
  // Check if a cuda stream pool is given.
  const bool has_cuda_stream_pool_ = cuda_stream_pool_.has_value() && cuda_stream_pool_.get();
  if (!has_cuda_stream_pool_) {
    // HOLOSCAN_LOG_DEBUG("No CUDA stream pool available, cannot allocate a stream.");
    return make_unexpected<RuntimeError>(RuntimeError(
        ErrorCode::kFailure, "No CUDA stream pool available, cannot allocate a stream."));
  }
  // get Handle to underlying nvidia::gxf::CudaStreamPool from
  // std::shared_ptr<holoscan::CudaStreamPool>
  const auto maybe_cuda_stream_pool = nvidia::gxf::Handle<nvidia::gxf::CudaStreamPool>::Create(
      context, cuda_stream_pool_.get()->gxf_cid());
  if (!maybe_cuda_stream_pool) {
    // HOLOSCAN_LOG_DEBUG("Failed to retrieve the internal stream pool.");
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kFailure, "Failed to retrieve the internal stream pool."));
  }
  return maybe_cuda_stream_pool.value();
}

expected<CudaStreamHandle, RuntimeError> CudaObjectHandler::allocate_cuda_stream(
    gxf_context_t context) {
  auto maybe_gxf_stream_pool_handle = cuda_stream_pool_handle(context);
  if (!maybe_gxf_stream_pool_handle) {
    return forward_error(maybe_gxf_stream_pool_handle);
  }
  auto gxf_stream_pool_handle = maybe_gxf_stream_pool_handle.value();

  // allocate a stream
  auto maybe_stream = gxf_stream_pool_handle->allocateStream();
  if (!maybe_stream) {
    // HOLOSCAN_LOG_DEBUG("Failed to allocate a new stream from the internal stream pool.");
    return make_unexpected<RuntimeError>(RuntimeError(
        ErrorCode::kFailure, "Failed to allocate a new stream from the internal stream pool."));
  }
  return maybe_stream.value();
}

cudaStream_t CudaObjectHandler::stream_from_stream_handle(CudaStreamHandle stream_handle) {
  if (stream_handle.is_null()) {
    HOLOSCAN_LOG_TRACE("CudaStreamHandle is null, returning cudaStreamDefault.");
    return cudaStreamDefault;
  }
  const auto& maybe_stream = stream_handle->stream();
  if (maybe_stream.has_value()) {
    return maybe_stream.value();
  }
  HOLOSCAN_LOG_TRACE(
      "CudaStream object does not contain a stream value, returning cudaStreamDefault.");
  return cudaStreamDefault;
}

expected<CudaStreamHandle, RuntimeError> CudaObjectHandler::stream_handle_from_stream(
    cudaStream_t stream) {
  auto stream_it = stream_to_stream_handle_.find(stream);
  if (stream_it != stream_to_stream_handle_.end()) {
    return stream_it->second;
  }
  return make_unexpected<RuntimeError>(
      RuntimeError(ErrorCode::kNotFound,
                   "No CudaStreamHandle is currently mapped to the provided CUDA stream."));
}

cudaStream_t CudaObjectHandler::get_cuda_stream(void* context, const std::string& input_port_name,
                                                bool allocate, bool sync_to_default) {
  auto maybe_cuda_stream_handle =
      get_cuda_stream_handle(context, input_port_name, allocate, sync_to_default);
  if (maybe_cuda_stream_handle.has_value()) {
    return stream_from_stream_handle(maybe_cuda_stream_handle.value());
  }
  if (allocate) {
    HOLOSCAN_LOG_DEBUG(
        "Failed to allocate a stream for input port '{}': {}. Returning cudaStreamDefault",
        input_port_name,
        maybe_cuda_stream_handle.error().what());
  } else {
    HOLOSCAN_LOG_TRACE("Failed to find stream on input port '{}': {}. Returning cudaStreamDefault",
                       input_port_name,
                       maybe_cuda_stream_handle.error().what());
  }
  return cudaStreamDefault;
}

std::vector<std::optional<cudaStream_t>> CudaObjectHandler::get_cuda_streams(
    void* context, const std::string& input_port_name) {
  auto maybe_cuda_stream_handle_vec = get_cuda_stream_handles(context, input_port_name);
  if (maybe_cuda_stream_handle_vec.has_value()) {
    auto out = std::vector<std::optional<cudaStream_t>>{};
    const auto& cuda_stream_handle_vec = maybe_cuda_stream_handle_vec.value();
    out.reserve(cuda_stream_handle_vec.size());
    for (auto& maybe_stream_handle : cuda_stream_handle_vec) {
      if (maybe_stream_handle) {
        out.push_back(stream_from_stream_handle(maybe_stream_handle.value()));
      } else {
        out.push_back(std::nullopt);
      }
    }
    return out;
  } else {
    HOLOSCAN_LOG_ERROR("get_cuda_stream_handles failed for port '{}': {}",
                       input_port_name,
                       maybe_cuda_stream_handle_vec.error().what());
  }
  throw std::runtime_error(fmt::format("get_cuda_streams failed for port '{}': ",
                                       maybe_cuda_stream_handle_vec.error().what()));
}

gxf_result_t CudaObjectHandler::synchronize_streams(
    std::vector<std::optional<CudaStreamHandle>> stream_handles,
    CudaStreamHandle target_stream_handle, bool sync_to_default_stream) {
  if (stream_handles.empty()) {
    return GXF_SUCCESS;
  }

  auto maybe_target_cuda_stream = target_stream_handle->stream();
  if (!maybe_target_cuda_stream) {
    HOLOSCAN_LOG_ERROR("target_stream_handle does not contain a stream value.");
    return GXF_FAILURE;
  }
  const cudaStream_t target_cuda_stream = maybe_target_cuda_stream.value();

  // iterate through all handles, storing the cudaStream_t for each that has a value
  std::vector<cudaStream_t> cuda_streams;
  cuda_streams.reserve(stream_handles.size());
  for (auto& maybe_stream_handle : stream_handles) {
    if (maybe_stream_handle) {
      cuda_streams.push_back(stream_from_stream_handle(maybe_stream_handle.value()));
    }
  }
  auto gxf_result = synchronize_streams(cuda_streams, target_cuda_stream, sync_to_default_stream);
  return static_cast<gxf_result_t>(gxf_result);
}

int CudaObjectHandler::synchronize_streams(std::vector<cudaStream_t> cuda_streams,
                                           cudaStream_t target_cuda_stream,
                                           bool sync_to_default_stream) {
  HOLOSCAN_LOG_DEBUG("Synchronizing {} streams to target stream.", cuda_streams.size());
  // exit early if there is nothing to synchronize
  if (target_cuda_stream == cudaStreamDefault) {
    sync_to_default_stream = false;
  }
  if (cuda_streams.empty() && !sync_to_default_stream) {
    return GXF_SUCCESS;
  }

  if (!event_created_) {
    // Create the internal event if needed
    cudaError_t result =
        HOLOSCAN_CUDA_CALL(cudaEventCreateWithFlags(&cuda_event_, cudaEventDisableTiming));
    if (cudaSuccess != result) {
      HOLOSCAN_LOG_ERROR("Stream synchronization failed");
      return static_cast<int>(GXF_FAILURE);
    }
    event_created_ = true;
  }

  // iterate through all messages and use events to chain incoming streams with the target stream
  for (auto& cuda_stream : cuda_streams) {
    cudaError_t result;
    if (cuda_stream == target_cuda_stream) {
      HOLOSCAN_LOG_TRACE("Skipping synchronization of stream with itself.");
      continue;
    }
    result = HOLOSCAN_CUDA_CALL(cudaEventRecord(cuda_event_, cuda_stream));
    if (cudaSuccess != result) {
      return GXF_FAILURE;
    }

    // Note: cudaStreamWaitEvent will work even for streams associated with separate devices
    result = HOLOSCAN_CUDA_CALL(cudaStreamWaitEvent(target_cuda_stream, cuda_event_));
    if (cudaSuccess != result) {
      HOLOSCAN_LOG_ERROR("Stream synchronization failed");
      return static_cast<int>(GXF_FAILURE);
    }
  }

  // also sync the target stream to the default stream
  if (sync_to_default_stream) {
    cudaError_t result = HOLOSCAN_CUDA_CALL(cudaEventRecord(cuda_event_, target_cuda_stream));
    if (cudaSuccess != result) {
      return GXF_FAILURE;
    }
    result = HOLOSCAN_CUDA_CALL(cudaStreamWaitEvent(cudaStreamDefault, cuda_event_));
    if (cudaSuccess != result) {
      return static_cast<int>(GXF_FAILURE);
    }
  }
  return static_cast<int>(GXF_SUCCESS);
}

expected<CudaStreamHandle, RuntimeError> CudaObjectHandler::allocate_internal_stream(
    gxf_context_t context, const std::string& stream_name) {
  // Create the CUDA stream if it does not yet exist.
  auto it = allocated_cuda_stream_handles_.find(stream_name);
  if (it == allocated_cuda_stream_handles_.end()) {
    HOLOSCAN_LOG_DEBUG("allocating internal stream named '{}'", stream_name);
    auto maybe_stream_handle = allocate_cuda_stream(context);
    if (!maybe_stream_handle) {
      return forward_error(maybe_stream_handle);
    }
    auto stream_handle = maybe_stream_handle.value();
    stream_to_stream_handle_.emplace(stream_handle->stream().value(), stream_handle);
    allocated_cuda_stream_handles_.emplace(stream_name, stream_handle);
  } else {
    HOLOSCAN_LOG_TRACE("reusing previously allocated internal stream named '{}'", stream_name);
  }
  return allocated_cuda_stream_handles_[stream_name];
}

int CudaObjectHandler::release_internal_streams(void* context) {
  if (allocated_cuda_stream_handles_.empty()) {
    return GXF_SUCCESS;
  }

  auto maybe_gxf_stream_pool_handle = cuda_stream_pool_handle(context);
  if (!maybe_gxf_stream_pool_handle) {
    HOLOSCAN_LOG_ERROR(
        "Found internally allocated CUDA streams, but no CUDA stream pool. These streams will "
        "not "
        "be released.");
    return static_cast<int>(GXF_FAILURE);
  }
  auto cuda_stream_pool = maybe_gxf_stream_pool_handle.value();

  gxf_result_t result = GXF_SUCCESS;
  for (const auto& [name, stream_handle] : allocated_cuda_stream_handles_) {
    auto maybe_released = cuda_stream_pool->releaseStream(stream_handle);
    if (!maybe_released) {
      HOLOSCAN_LOG_ERROR("Failed to release internally allocated CUDA stream '{}'", name);
      result = GXF_FAILURE;
    }
  }
  return static_cast<int>(result);
}

void CudaObjectHandler::clear_received_streams() {
  // retain the existing unordered_maps and vectors, but clear the contents
  for (auto& item : received_cuda_stream_ids_) {
    item.second.clear();
  }
  for (auto& item : received_cuda_stream_handles_) {
    item.second.clear();
  }
}

}  // namespace holoscan::gxf
