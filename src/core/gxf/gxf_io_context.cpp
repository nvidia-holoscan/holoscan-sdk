/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/gxf/gxf_io_context.hpp>

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/domain/tensor_map.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/gxf_cuda.hpp>
#include <holoscan/core/gxf/gxf_execution_context.hpp>
#include <holoscan/core/gxf/gxf_operator.hpp>
#include <holoscan/core/gxf/gxf_resource.hpp>
#include <holoscan/core/gxf/gxf_utils.hpp>
#include <holoscan/core/io_spec.hpp>
#include <holoscan/core/message.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/profiler/profiler.hpp>

#include <gxf/core/gxf.h>
#include <gxf/multimedia/video.hpp>
#include <gxf/std/receiver.hpp>
#include <gxf/std/tensor.hpp>
#include <gxf/std/timestamp.hpp>
#include <gxf/std/transmitter.hpp>

namespace holoscan::gxf {

nvidia::gxf::Receiver* get_gxf_receiver(const std::shared_ptr<IOSpec>& input_spec) {
  auto connector = input_spec->connector();
  auto gxf_resource = std::dynamic_pointer_cast<GXFResource>(connector);
  if (gxf_resource == nullptr) {
    if (input_spec->queue_size() == IOSpec::kAnySize) {
      HOLOSCAN_LOG_ERROR(
          "Unable to receive non-vector data from the input port '{}' with the queue size "
          "'IOSpec::kAnySize'. Please call 'op_input.receive<std::vector<T>>()' instead of "
          "'op_input.receive<T>()'.",
          input_spec->name());
      throw std::invalid_argument("Invalid template type for the input port");
    } else {
      HOLOSCAN_LOG_ERROR("Invalid connector type for the input spec '{}'", input_spec->name());
    }
    return nullptr;  // to cause a bad_any_cast
  }

  // Use cached component pointer from GXFComponent (set during gxf_initialize)
  // instead of calling GxfComponentTypeId + GxfComponentPointer on every receive
  return static_cast<nvidia::gxf::Receiver*>(gxf_resource->gxf_cptr());
}

GXFInputContext::GXFInputContext(ExecutionContext* execution_context, Operator* op)
    : InputContext(execution_context, op) {}

GXFInputContext::GXFInputContext(ExecutionContext* execution_context, Operator* op,
                                 std::unordered_map<std::string, std::shared_ptr<IOSpec>>& inputs)
    : InputContext(execution_context, op, inputs) {}

gxf_context_t GXFInputContext::gxf_context() const {
  if (execution_context_) {
    return execution_context_->context();
  }
  return nullptr;
}

bool GXFInputContext::empty_impl(const char* name) {
  std::string input_name = holoscan::get_well_formed_name(name, inputs_);
  auto it = inputs_.find(input_name);
  if (it == inputs_.end()) {
    HOLOSCAN_LOG_ERROR("The input port with name {} is not found", input_name);
    return false;
  }
  auto receiver = get_gxf_receiver(it->second);
  if (!receiver) {
    HOLOSCAN_LOG_ERROR("Invalid receiver found for the input port with name {}", input_name);
    return false;
  }
  return receiver->size() == 0;
}

namespace {

//==============================================================================
// Cached Type ID Helper Functions
//
// These functions provide type-ID-cached versions of entity.add<T>() and entity.get<T>()
// for frequently-used component types. The standard GXF Entity::add<T>() and Entity::get<T>()
// call GxfComponentTypeId() on every invocation, which involves string lookups. By caching
// the type ID in a static variable, we avoid this overhead on every message emit/receive.
//
// Thread safety: These use std::call_once for thread-safe one-time initialization of the
// cached type IDs. Type IDs are assigned once at application startup (during extension
// registration) and remain constant for the lifetime of the GXF context.
//==============================================================================

/// @brief Get holoscan::Message component from entity with cached type ID lookup.
nvidia::gxf::Expected<nvidia::gxf::Handle<holoscan::Message>> get_message(
    nvidia::gxf::Entity& entity) {
  static std::once_flag tid_init_flag;
  static gxf_tid_t tid;
  static gxf_result_t tid_init_result = GXF_SUCCESS;
  std::call_once(tid_init_flag, [&entity]() {
    tid_init_result = GxfComponentTypeId(entity.context(), "holoscan::Message", &tid);
  });
  if (tid_init_result != GXF_SUCCESS) {
    return nvidia::gxf::Unexpected{tid_init_result};
  }

  gxf_uid_t cid;
  void* comp_ptr = nullptr;
  auto result = GxfComponentFindAndGetPtr(entity.context(),
                                          entity.eid(),
                                          entity.entity_item_ptr(),
                                          tid,
                                          nullptr,
                                          nullptr,
                                          &cid,
                                          &comp_ptr);
  if (result != GXF_SUCCESS) {
    return nvidia::gxf::Unexpected{result};
  }
  return nvidia::gxf::Handle<holoscan::Message>::Create(entity.context(), cid, tid, comp_ptr);
}

/// @brief Add holoscan::Message component to entity with cached type ID lookup.
nvidia::gxf::Expected<nvidia::gxf::Handle<holoscan::Message>> add_message(
    nvidia::gxf::Entity& entity) {
  static std::once_flag tid_init_flag;
  static gxf_tid_t tid;
  static gxf_result_t tid_init_result = GXF_SUCCESS;
  std::call_once(tid_init_flag, [&entity]() {
    tid_init_result = GxfComponentTypeId(entity.context(), "holoscan::Message", &tid);
  });
  if (tid_init_result != GXF_SUCCESS) {
    return nvidia::gxf::Unexpected{tid_init_result};
  }

  gxf_uid_t cid;
  void* comp_ptr = nullptr;
  auto result = GxfComponentAddAndGetPtr(
      entity.context(), entity.entity_item_ptr(), tid, nullptr, &cid, &comp_ptr);
  if (result != GXF_SUCCESS) {
    return nvidia::gxf::Unexpected{result};
  }
  return nvidia::gxf::Handle<holoscan::Message>::Create(entity.context(), cid, tid, comp_ptr);
}

/// @brief Get holoscan::MetadataDictionary component from entity with cached type ID lookup.
nvidia::gxf::Expected<nvidia::gxf::Handle<holoscan::MetadataDictionary>> get_metadata(
    nvidia::gxf::Entity& entity, const char* name = "metadata_") {
  static std::once_flag tid_init_flag;
  static gxf_tid_t tid;
  static gxf_result_t tid_init_result = GXF_SUCCESS;
  std::call_once(tid_init_flag, [&entity]() {
    tid_init_result = GxfComponentTypeId(entity.context(), "holoscan::MetadataDictionary", &tid);
  });
  if (tid_init_result != GXF_SUCCESS) {
    return nvidia::gxf::Unexpected{tid_init_result};
  }

  gxf_uid_t cid;
  void* comp_ptr = nullptr;
  auto result = GxfComponentFindAndGetPtr(entity.context(),
                                          entity.eid(),
                                          entity.entity_item_ptr(),
                                          tid,
                                          name,
                                          nullptr,
                                          &cid,
                                          &comp_ptr);
  if (result != GXF_SUCCESS) {
    return nvidia::gxf::Unexpected{result};
  }
  return nvidia::gxf::Handle<holoscan::MetadataDictionary>::Create(
      entity.context(), cid, tid, comp_ptr);
}

/// @brief Add holoscan::MetadataDictionary component to entity with cached type ID lookup.
nvidia::gxf::Expected<nvidia::gxf::Handle<holoscan::MetadataDictionary>> add_metadata(
    nvidia::gxf::Entity& entity, const char* name = "metadata_") {
  static std::once_flag tid_init_flag;
  static gxf_tid_t tid;
  static gxf_result_t tid_init_result = GXF_SUCCESS;
  std::call_once(tid_init_flag, [&entity]() {
    tid_init_result = GxfComponentTypeId(entity.context(), "holoscan::MetadataDictionary", &tid);
  });
  if (tid_init_result != GXF_SUCCESS) {
    return nvidia::gxf::Unexpected{tid_init_result};
  }

  gxf_uid_t cid;
  void* comp_ptr = nullptr;
  auto result = GxfComponentAddAndGetPtr(
      entity.context(), entity.entity_item_ptr(), tid, name, &cid, &comp_ptr);
  if (result != GXF_SUCCESS) {
    return nvidia::gxf::Unexpected{result};
  }
  return nvidia::gxf::Handle<holoscan::MetadataDictionary>::Create(
      entity.context(), cid, tid, comp_ptr);
}

/// @brief Get nvidia::gxf::CudaStreamId component from entity with cached type ID lookup.
nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::CudaStreamId>> get_cuda_stream_id(
    nvidia::gxf::Entity& entity, const char* name = nullptr) {
  static std::once_flag tid_init_flag;
  static gxf_tid_t tid;
  static gxf_result_t tid_init_result = GXF_SUCCESS;
  std::call_once(tid_init_flag, [&entity]() {
    tid_init_result = GxfComponentTypeId(entity.context(), "nvidia::gxf::CudaStreamId", &tid);
  });
  if (tid_init_result != GXF_SUCCESS) {
    return nvidia::gxf::Unexpected{tid_init_result};
  }

  gxf_uid_t cid;
  void* comp_ptr = nullptr;
  auto result = GxfComponentFindAndGetPtr(entity.context(),
                                          entity.eid(),
                                          entity.entity_item_ptr(),
                                          tid,
                                          name,
                                          nullptr,
                                          &cid,
                                          &comp_ptr);
  if (result != GXF_SUCCESS) {
    return nvidia::gxf::Unexpected{result};
  }
  return nvidia::gxf::Handle<nvidia::gxf::CudaStreamId>::Create(
      entity.context(), cid, tid, comp_ptr);
}

/// @brief Add nvidia::gxf::CudaStreamId component to entity with cached type ID lookup.
nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::CudaStreamId>> add_cuda_stream_id(
    nvidia::gxf::Entity& entity, const char* name = "cuda_stream_id_") {
  static std::once_flag tid_init_flag;
  static gxf_tid_t tid;
  static gxf_result_t tid_init_result = GXF_SUCCESS;
  std::call_once(tid_init_flag, [&entity]() {
    tid_init_result = GxfComponentTypeId(entity.context(), "nvidia::gxf::CudaStreamId", &tid);
  });
  if (tid_init_result != GXF_SUCCESS) {
    return nvidia::gxf::Unexpected{tid_init_result};
  }

  gxf_uid_t cid;
  void* comp_ptr = nullptr;
  auto result = GxfComponentAddAndGetPtr(
      entity.context(), entity.entity_item_ptr(), tid, name, &cid, &comp_ptr);
  if (result != GXF_SUCCESS) {
    return nvidia::gxf::Unexpected{result};
  }
  return nvidia::gxf::Handle<nvidia::gxf::CudaStreamId>::Create(
      entity.context(), cid, tid, comp_ptr);
}

//==============================================================================

/**
 * @brief Add or update a CudaStreamId component in an entity.
 *
 * By default (replace_existing=true), if the entity already contains a CudaStreamId, it is
 * updated with the new stream_cid. This is the correct behavior when forwarding a received
 * entity after processing on a different stream - downstream operators should see the most
 * recent stream, not the original upstream stream.
 *
 * The replace_existing=false mode allows adding additional CudaStreamId components for
 * advanced use cases where an entity needs to track multiple streams (e.g., when different
 * parts of an entity were processed on different streams). Note that the standard
 * receive_cuda_stream/receive_cuda_streams APIs only return the FIRST CudaStreamId found
 * via entity.get<CudaStreamId>(). CudaStreamCondition uses findAll and can wait on all streams.
 *
 * @param gxf_entity The entity to add/update the CudaStreamId in.
 * @param stream_cid The component ID of the CudaStream to reference.
 * @param replace_existing If true (default), update existing CudaStreamId if present.
 *                         If false, always add a new CudaStreamId component.
 * @param is_new_entity If true, skip the get() check for existing CudaStreamId since we know
 *                      the entity was just created. This avoids a GXF lookup on every emit
 *                      for the common case of emitting newly-created entities.
 * @return GXF_SUCCESS on success, GXF_FAILURE on error.
 */
gxf_result_t add_stream_id_to_entity(nvidia::gxf::Entity& gxf_entity, gxf_uid_t stream_cid,
                                     bool replace_existing = true, bool is_new_entity = false) {
  // For forwarded entities, check if CudaStreamId already exists and update it
  if (replace_existing && !is_new_entity) {
    // Check if there's already a CudaStreamId in the entity (e.g., when forwarding
    // a received entity). If so, update it instead of adding a new one. This ensures
    // downstream operators see the correct (most recent) stream via entity.get<CudaStreamId>().
    auto existing_stream_id = get_cuda_stream_id(gxf_entity);
    if (!existing_stream_id) {
      // Check if it's an actual error vs just "component not found"
      auto code = nvidia::gxf::ToResultCode(existing_stream_id);
      if (code != GXF_ENTITY_COMPONENT_NOT_FOUND) {
        HOLOSCAN_LOG_ERROR("Failed to get existing CudaStreamId with error: {}",
                           GxfResultStr(code));
        return code;
      }
      // Component not found is expected - fall through to add a new one
    } else {
      existing_stream_id.value()->stream_cid = stream_cid;
      HOLOSCAN_LOG_TRACE("Updated existing CudaStreamId in entity to stream_cid: {}", stream_cid);
      return GXF_SUCCESS;
    }
  }

  // No existing CudaStreamId (or is_new_entity=true or replace_existing=false), add a new one
  const auto maybe_stream_id = add_cuda_stream_id(gxf_entity);
  if (!maybe_stream_id) {
    auto code = nvidia::gxf::ToResultCode(maybe_stream_id);
    HOLOSCAN_LOG_ERROR("Failed to add CUDA stream id to output message with error: {}.",
                       GxfResultStr(code));
    return GXF_FAILURE;
  }
  maybe_stream_id.value()->stream_cid = stream_cid;
  return GXF_SUCCESS;
}

// Check if stream propagation for raw entities is disabled via environment variable.
// This allows users to opt-out of the automatic findAll behavior for performance tuning.
bool is_entity_stream_propagation_disabled() {
  static std::once_flag init_flag;
  static bool disabled = false;
  std::call_once(init_flag, []() {
    const char* env_value = std::getenv("HOLOSCAN_DISABLE_ENTITY_STREAM_PROPAGATION");
    disabled = (env_value != nullptr && std::string(env_value) == "1");
    if (disabled) {
      HOLOSCAN_LOG_INFO(
          "Entity stream propagation disabled via HOLOSCAN_DISABLE_ENTITY_STREAM_PROPAGATION=1");
    }
  });
  return disabled;
}

// Propagate CUDA stream to memory buffers in an entity for stream-aware deallocation.
// This enables allocators like BlockMemoryPool to defer memory reuse until GPU operations
// complete on the specified stream, preventing data corruption from race conditions.
//
// This function iterates through all Tensor and VideoBuffer components in the entity
// and sets the stream on their memory buffers.
void propagate_stream_to_entity_memory_buffers(nvidia::gxf::Entity& gxf_entity,
                                               gxf_context_t gxf_ctx, gxf_uid_t stream_cid) {
  // Check if disabled via environment variable
  if (is_entity_stream_propagation_disabled()) {
    return;
  }

  // Get the cudaStream_t from the stream component
  auto maybe_stream_handle = gxf::CudaStreamHandle::Create(gxf_ctx, stream_cid);
  if (!maybe_stream_handle) {
    HOLOSCAN_LOG_DEBUG("Failed to create CudaStreamHandle for stream propagation");
    return;
  }

  auto stream_result = maybe_stream_handle.value()->stream();
  if (!stream_result) {
    HOLOSCAN_LOG_DEBUG("Failed to get CUDA stream from stream handle");
    return;
  }
  cudaStream_t cuda_stream = stream_result.value();
  void* stream_ptr = static_cast<void*>(cuda_stream);

  // Set stream on all Tensors
  auto tensors = gxf_entity.findAllHeap<nvidia::gxf::Tensor>();
  if (tensors) {
    // Cannot use auto& because value() returns nvidia::gxf::Expected types by value
    for (auto tensor_handle : tensors.value()) {
      auto tensor_ptr = tensor_handle.value();
      tensor_ptr->memory_buffer().setStream(stream_ptr);
    }
  }

  // Set stream on all VideoBuffers
  auto video_buffers = gxf_entity.findAllHeap<nvidia::gxf::VideoBuffer>();
  if (video_buffers) {
    // Cannot use auto& because value() returns nvidia::gxf::Expected types by value
    for (auto vb_handle : video_buffers.value()) {
      auto vb_ptr = vb_handle.value();
      vb_ptr->memory_buffer().setStream(stream_ptr);
    }
  }
}

}  // namespace

gxf_result_t GXFInputContext::retrieve_cuda_streams(nvidia::gxf::Entity& message,
                                                    const std::string& input_name) {
  auto context = gxf_context();
  auto object_handler = gxf_cuda_object_handler();
  if (object_handler == nullptr) {
    HOLOSCAN_LOG_DEBUG("CudaObjectHandler is not initialized, could not retrieve CUDA streams");
    return GXF_FAILURE;
  }

  // Note: For multi-receiver ports, input_name may have a suffix like "receivers:0", "receivers:1".
  // The streams_from_message function strips this suffix when storing streams in the map.
  auto result = object_handler->streams_from_message(context, message, input_name);
  if (result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to retrieve CUDA streams from the incoming message: {}",
                       GxfResultStr(result));
    return result;
  }
  return GXF_SUCCESS;
}

cudaStream_t GXFInputContext::receive_cuda_stream(const char* input_port_name, bool allocate,
                                                  bool sync_to_default) {
  PROF_SCOPED_EVENT(op_->id(), event_receive_cuda_stream);

  std::string input_name = holoscan::get_well_formed_name(input_port_name, inputs_);
  if (inputs_.find(input_name) == inputs_.end()) {
    std::string err_msg = fmt::format("An input port with name '{}' is not found", input_name);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  auto object_handler = cuda_object_handler();
  if (object_handler == nullptr) {
    const std::string err_msg = "CudaObjectHandler not initialized.";
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  auto stream = object_handler->get_cuda_stream(
      execution_context_->context(), input_name, allocate, sync_to_default);
  return stream;
}

std::vector<std::optional<cudaStream_t>> GXFInputContext::receive_cuda_streams(
    const char* input_port_name) {
  PROF_SCOPED_EVENT(op_->id(), event_receive_cuda_streams);

  std::string input_name = holoscan::get_well_formed_name(input_port_name, inputs_);
  if (inputs_.find(input_name) == inputs_.end()) {
    std::string err_msg = fmt::format("An input port with name {} is not found", input_name);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  auto object_handler = cuda_object_handler();
  if (object_handler == nullptr) {
    HOLOSCAN_LOG_ERROR("CudaObjectHandler is not initialized, returning empty vector");
    return std::vector<std::optional<cudaStream_t>>{};
  }
  auto streams = object_handler->get_cuda_streams(execution_context_->context(), input_name);
  return streams;
}

std::any GXFInputContext::receive_impl(const char* name, InputType in_type, bool no_error_message,
                                       bool omit_data_logging, bool allow_any_size) {
  std::string input_name = holoscan::get_well_formed_name(name, inputs_);
  PROF_SCOPED_EVENT(op_->id(), event_receive_impl);
  HOLOSCAN_LOG_TRACE("GXFInputContext::receive_impl for op: {}, input_name: {}, in_type: {}",
                     op_->name(),
                     input_name,
                     magic_enum::enum_name(in_type));

  auto& data_loggers = op_->fragment()->data_loggers();

  // Lambda to extract the first received CUDA stream for logging
  auto get_stream_for_logging = [this](const char* port_name) -> std::optional<cudaStream_t> {
    auto received_streams = receive_cuda_streams(port_name);
    if (!received_streams.empty() && received_streams[0].has_value()) {
      if (received_streams.size() > 1) {
        HOLOSCAN_LOG_DEBUG(
            "{}: Multiple CUDA streams ({}) received on input port '{}', using first stream for "
            "data "
            "logging",
            op_->name(),
            received_streams.size(),
            port_name);
      }
      return received_streams[0];
    }
    return std::nullopt;
  };

  auto it = inputs_.find(input_name);
  if (it == inputs_.end()) {
    if (no_error_message) {
      return kNoReceivedMessage;
    }
    // Show error message because the input name is not found.
    const auto input_port_size = inputs_.size();
    if (input_port_size == 1) {
      auto no_accessible_error_message = NoAccessibleMessageType(fmt::format(
          "The operator({}) has only one port with label '{}' but the non-existent port label "
          "'{}' was specified in the receive() method",
          op_->name(),
          inputs_.begin()->first,
          input_name));
      return no_accessible_error_message;
    } else {
      if (input_port_size == 0) {
        auto no_accessible_error_message = NoAccessibleMessageType(
            fmt::format("The operator({}) does not have any input port but '{}' was specified in "
                        "receive() method",
                        op_->name(),
                        input_name));

        return no_accessible_error_message;
      }

      auto msg_buf = fmt::memory_buffer();
      auto& op_inputs = op_->spec()->inputs();
      for (const auto& [label, _] : op_inputs) {
        if (&label == &(op_inputs.begin()->first)) {
          fmt::format_to(std::back_inserter(msg_buf), "{}", label);
        } else {
          fmt::format_to(std::back_inserter(msg_buf), ", {}", label);
        }
      }
      auto no_accessible_error_message = NoAccessibleMessageType(
          fmt::format("The operator({}) does not have an input port with label "
                      "'{}'. It should be one of ({:.{}}) "
                      "in receive() method",
                      op_->name(),
                      input_name,
                      msg_buf.data(),
                      msg_buf.size()));
      return no_accessible_error_message;
    }
  }
  auto io_spec = it->second;
  if (!allow_any_size &&
      io_spec->queue_size() == static_cast<int64_t>(holoscan::IOSpec::kAnySize)) {
    auto error_message = fmt::format(
        "Unable to receive non-vector data from the input port '{}' with the queue size "
        "'IOSpec::kAnySize'. Please call 'op_input.receive<std::vector<T>>()' instead of "
        "'op_input.receive<T>()'.",
        input_name);
    HOLOSCAN_LOG_ERROR(error_message);
    throw std::invalid_argument(error_message);
  }
  auto receiver = get_gxf_receiver(io_spec);
  if (!receiver) {
    auto no_accessible_error_message = NoAccessibleMessageType(fmt::format(
        "{}: Invalid receiver found for the input port with name {}", op_->name(), input_name));

    return no_accessible_error_message;
  }

  auto maybe_entity = receiver->receive();
  if (!maybe_entity || maybe_entity.value().is_null()) {
    return kNoReceivedMessage;  // to indicate that there is no data
  }
  auto& entity = maybe_entity.value();

  // Update operator metadata using any metadata found in the entity.
  if (op_->is_metadata_enabled()) {
    {
      PROF_SCOPED_EVENT(op_->id(), event_receive_metadata);
      // Merge metadata from all input ports into the dynamic metadata of the operator
      auto maybe_metadata = get_metadata(entity);
      if (!maybe_metadata) {
        // If the operator does not have any metadata it is expected that the MetadataDictionary
        // component will not be present, so don't warn in this case.
        HOLOSCAN_LOG_TRACE(
            "No MetadataDictionary found for input '{}' on operator '{}'", input_name, op_->name());
      } else {
        auto metadata = maybe_metadata.value();
        HOLOSCAN_LOG_TRACE("MetadataDictionary with size {} found for input '{}' of operator '{}'",
                           metadata->size(),
                           input_name,
                           op_->name());
        auto metadata_ptr = metadata.get();
        // use update here to respect the Operator's MetadataPolicy
        op_->metadata()->update(*metadata_ptr);
      }
    }
  }

  // Handle any streams found in the entity
  {
    PROF_SCOPED_EVENT(op_->id(), event_receive_streams);
    retrieve_cuda_streams(entity, input_name);
  }

  // Handle any acquisition timestamps found in the entity
  {
    PROF_SCOPED_EVENT(op_->id(), event_receive_acquisition_timestamps);
    // Use findAll with capacity=1 instead of findAllHeap to avoid heap allocation.
    // We only use front().value() below, so capacity of 1 is sufficient.
    // This avoids both the heap allocation of findAllHeap AND the large 1024-element
    // default stack allocation of findAll<T>() (kMaxComponents = 1024).
    auto timestamp_components = entity.findAll<nvidia::gxf::Timestamp, 1>();
    int64_t gxf_acquisition_timestamp = 0;
    if (!timestamp_components || 0 == timestamp_components->size()) {
      // Requires Timestamp instance for message age
      HOLOSCAN_LOG_TRACE(
          "{}: message received on input port '{}' carries no Timestamp.", op_->name(), input_name);
    } else {
      gxf_acquisition_timestamp = timestamp_components->front().value()->acqtime;
      HOLOSCAN_LOG_TRACE(
          "{}: message received on input port '{}' has GXF Timestamp with acqtime: {}.",
          op_->name(),
          input_name,
          gxf_acquisition_timestamp);
      // Truncate the port name to the base name without :0, etc. in the multi-receiver case.
      auto colon_pos = input_name.find(':');
      if (colon_pos != std::string::npos) {
        input_name = input_name.substr(0, colon_pos);
      }
      acquisition_timestamp_map_[input_name].push_back(gxf_acquisition_timestamp);
    }
  }

  // If the input type is GXFEntity, return the entity directly
  if (in_type == InputType::kGXFEntity) {
    if (!omit_data_logging && !data_loggers.empty()) {
      // Log the entity itself using log_backend_specific
      auto metadata_ptr = op_->is_metadata_enabled() ? op_->metadata() : nullptr;
      const std::string unique_id = io_spec->unique_id();

      // Lazily retrieve CUDA stream only if needed for logging
      bool stream_checked = false;
      std::optional<cudaStream_t> stream_for_logging = std::nullopt;

      for (auto& data_logger : data_loggers) {
        HOLOSCAN_LOG_TRACE("\t log_backend_specific code path");
        if (data_logger->should_log_input()) {
          if (!stream_checked) {
            stream_for_logging = get_stream_for_logging(input_name.c_str());
            stream_checked = true;
          }
          // Create a shared entity to ensure proper lifetime management for async logging
          auto shared_entity_expected = entity.clone();
          if (shared_entity_expected) {
            HOLOSCAN_LOG_TRACE("\t\t calling log_backend_specific");
            PROF_SCOPED_EVENT(op_->id(), event_log_backend_specific);
            data_logger->log_backend_specific(shared_entity_expected.value(),
                                              unique_id,
                                              -1,
                                              metadata_ptr,
                                              IOSpec::IOType::kInput,
                                              stream_for_logging);
          } else {
            HOLOSCAN_LOG_ERROR("{}.{}: failed to create shared entity for logging",
                               op_->name(),
                               input_name,
                               GxfResultStr(shared_entity_expected.error()));
          }
        }
      }
    }
    // Convert nvidia::gxf::Entity to holoscan::gxf::Entity
    holoscan::gxf::Entity entity_wrapper(entity);
    return entity_wrapper;  // to handle gxf::Entity as it is
  }

  auto message = get_message(entity);
  if (!message) {
    // TensorMap case is already logged in the outer InputContext::receive call, so don't log it
    // here as well.
    if (!omit_data_logging && !data_loggers.empty()) {
      HOLOSCAN_LOG_TRACE("\t log_backend_specific code path 2");
      // Log the entity itself using log_backend_specific
      auto metadata_ptr = op_->is_metadata_enabled() ? op_->metadata() : nullptr;
      const std::string unique_id = io_spec->unique_id();

      // Lazily retrieve CUDA stream only if needed for logging
      bool stream_checked = false;
      std::optional<cudaStream_t> stream_for_logging = std::nullopt;

      for (auto& data_logger : data_loggers) {
        if (data_logger->should_log_input()) {
          if (!stream_checked) {
            stream_for_logging = get_stream_for_logging(input_name.c_str());
            stream_checked = true;
          }
          // Create a shared entity to ensure proper lifetime management for async logging
          auto shared_entity_expected = entity.clone();
          if (shared_entity_expected) {
            HOLOSCAN_LOG_TRACE("\t\t calling log_backend_specific");
            PROF_SCOPED_EVENT(op_->id(), event_log_backend_specific);
            data_logger->log_backend_specific(shared_entity_expected.value(),
                                              unique_id,
                                              -1,
                                              metadata_ptr,
                                              IOSpec::IOType::kInput,
                                              stream_for_logging);
          } else {
            HOLOSCAN_LOG_ERROR("{}.{}: failed to create shared entity for logging: {}",
                               op_->name(),
                               input_name,
                               GxfResultStr(shared_entity_expected.error()));
          }
        }
      }
    }
    // Convert nvidia::gxf::Entity to holoscan::gxf::Entity
    holoscan::gxf::Entity entity_wrapper(entity);
    return entity_wrapper;  // to handle gxf::Entity as it is
  }

  // Return the value of the message
  auto message_ptr = message.value();
  auto value = message_ptr->value();

  if (!data_loggers.empty()) {
    PROF_SCOPED_EVENT(op_->id(), event_data_logging);
    auto metadata_ptr = op_->is_metadata_enabled() ? op_->metadata() : nullptr;
    const std::string unique_id = io_spec->unique_id();

    for (auto& data_logger : data_loggers) {
      // TODO(grelee): should we log any information on CUDA streams? If so there are multiple
      // considerations:
      //   1. should we log all streams found on the emitted/received by a port
      //   2. should we log the stream associated with the operator
      //   3. should the value logged be the memory address of the cudaStream_t? (or GXF cid?)
      if (data_logger->should_log_input()) {
        PROF_SCOPED_EVENT(op_->id(), event_log_data);
        data_logger->log_data(value, unique_id, -1, metadata_ptr, IOSpec::IOType::kInput);
      }
    }
  }

  return value;
}

GXFOutputContext::GXFOutputContext(ExecutionContext* execution_context, Operator* op)
    : OutputContext(execution_context, op) {}

GXFOutputContext::GXFOutputContext(
    ExecutionContext* execution_context, Operator* op,
    std::unordered_map<std::string, std::shared_ptr<IOSpec>>& outputs)
    : OutputContext(execution_context, op, outputs) {}

gxf_context_t GXFOutputContext::gxf_context() const {
  if (execution_context_) {
    return execution_context_->context();
  }
  return nullptr;
}

void GXFOutputContext::populate_output_metadata(nvidia::gxf::Handle<MetadataDictionary> metadata,
                                                const std::string& output_name) {
  // insert the operator's metadata into the provided (empty) metadata object
  auto dynamic_metadata = op_->metadata();
  metadata->insert(*dynamic_metadata);
  HOLOSCAN_LOG_TRACE("MetadataDictionary with size {} emitted on output '{}' of operator '{}'",
                     metadata->size(),
                     output_name,
                     op_->name());
}

void GXFOutputContext::set_cuda_stream(const cudaStream_t stream, const char* output_port_name) {
  PROF_SCOPED_EVENT(op_->id(), event_set_cuda_stream);

  std::string output_name = holoscan::get_well_formed_name(output_port_name, outputs_);
  if (outputs_.find(output_name) == outputs_.end()) {
    std::string err_msg = fmt::format("An input port with name '{}' is not found", output_name);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  auto object_handler = cuda_object_handler();
  if (object_handler == nullptr) {
    HOLOSCAN_LOG_ERROR("CudaObjectHandler is not initialized, stream will not be set");
    return;
  }
  auto gxf_result = object_handler->add_stream(stream, output_name);
  if (gxf_result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR(
        "Failure to add CUDA stream to output port '{}': No GXF CudaStreamHandle is currently "
        "mapped to the provided CUDA stream. Only streams that were received from an input port "
        "via one of the `InputContext::receive_cuda_stream*` methods or or were allocated via "
        "`ExecutionContext::allocate_cuda_stream*` can be added to an output port.",
        output_port_name);
  }
}

std::optional<cudaStream_t> GXFOutputContext::stream_to_emit(const char* output_port_name) {
  std::string output_name = holoscan::get_well_formed_name(output_port_name, outputs_);

  auto gxf_cuda_handler = gxf_cuda_object_handler();
  if (!gxf_cuda_handler) {
    return std::nullopt;
  }

  auto maybe_stream_cid = gxf_cuda_handler->get_output_stream_cid(output_name);
  if (!maybe_stream_cid.has_value()) {
    return std::nullopt;
  }

  auto gxf_ctx = gxf_context();
  auto maybe_stream_handle = gxf::CudaStreamHandle::Create(gxf_ctx, maybe_stream_cid.value());
  if (!maybe_stream_handle.has_value()) {
    return std::nullopt;
  }

  cudaStream_t stream = gxf_cuda_handler->stream_from_stream_handle(maybe_stream_handle.value());
  return stream;
}

void GXFOutputContext::emit_impl(std::any data, const char* name, OutputType out_type,
                                 const int64_t acq_timestamp, bool omit_data_logging,
                                 bool skip_stream_propagation, bool is_new_entity) {
  std::string output_name = holoscan::get_well_formed_name(name, outputs_);
  PROF_SCOPED_EVENT(op_->id(), event_emit_impl);
  HOLOSCAN_LOG_TRACE("GXFOutputContext::emit_impl for op: {}, output_name: {}, out_type: {}",
                     op_->name(),
                     output_name,
                     magic_enum::enum_name(out_type));

  // Fast path: direct lookup of the output port
  auto it = outputs_.find(output_name);
  if (it == outputs_.end()) {  // Not found case
    const auto output_port_size = outputs_.size();

    // Handle empty name with execution port case (relatively common)
    if (output_name.empty() && output_port_size == 2 && op_->output_exec_spec()) {
      it = std::find_if(outputs_.begin(),
                        outputs_.end(),
                        [exec_spec = op_->output_exec_spec()](const auto& pair) {
                          return pair.second != exec_spec;
                        });
      if (it != outputs_.end()) {
        output_name = it->first;
      }
    }

    // If still not found, handle error cases and return
    if (it == outputs_.end()) {
      if (output_port_size == 1) {
        HOLOSCAN_LOG_ERROR(
            "The operator({}) has only one port with label '{}' but the non-existent port label "
            "'{}' was specified in the emit() method",
            op_->name(),
            outputs_.begin()->first,
            name);
      } else if (output_port_size == 0) {
        HOLOSCAN_LOG_ERROR(
            "The operator({}) does not have any output port but '{}' was specified in "
            "the emit() method",
            op_->name(),
            output_name);
      } else {
        auto msg_buf = fmt::memory_buffer();
        const auto& op_outputs = op_->spec()->outputs();
        bool is_first = true;
        for (const auto& [label, _] : op_outputs) {
          if (is_first) {
            fmt::format_to(std::back_inserter(msg_buf), "{}", label);
            is_first = false;
          } else {
            fmt::format_to(std::back_inserter(msg_buf), ", {}", label);
          }
        }
        HOLOSCAN_LOG_ERROR(
            "The operator({}) does not have an output port with label '{}'. It should be "
            "one of ({:.{}}) passed to the emit() method.",
            op_->name(),
            output_name,
            msg_buf.data(),
            msg_buf.size());
      }
      return;
    }
  }

  // check if there is a CUDA stream to be emitted on this output port
  bool stream_found = false;
  gxf_uid_t stream_cid{kNullUid};
  {
    PROF_SCOPED_EVENT(op_->id(), event_emit_streams);
    auto object_handler = gxf_cuda_object_handler();
    if (object_handler == nullptr) {
      HOLOSCAN_LOG_DEBUG("CudaObjectHandler not initialized, no streams will be emitted");
    } else {
      auto maybe_stream_cid = object_handler->get_output_stream_cid(output_name);
      stream_found = maybe_stream_cid.has_value();
      if (stream_found) {
        stream_cid = maybe_stream_cid.value();
      }
    }
  }

  const std::shared_ptr<IOSpec>& output_spec = it->second;
  auto connector = output_spec->connector();

  auto gxf_resource = std::dynamic_pointer_cast<GXFResource>(connector);
  if (gxf_resource == nullptr) {
    HOLOSCAN_LOG_ERROR("Invalid resource type for connector {}.{}", op_->name(), output_name);
    return;
  }

  // Use cached component pointer from GXFComponent (set during gxf_initialize)
  // instead of calling GxfComponentTypeId + GxfComponentPointer on every emit
  void* tx_ptr = gxf_resource->gxf_cptr();

  HOLOSCAN_LOG_TRACE("in GXFOutputContext::emit_impl: out_type: {}", static_cast<int>(out_type));
  switch (out_type) {
    case OutputType::kAny: {
      // Create an Entity object and add a Message object to it.
      auto gxf_entity = nvidia::gxf::Entity::New(gxf_context());
      auto buffer = add_message(gxf_entity.value());

      if (op_->is_metadata_enabled() && op_->metadata()->size() > 0) {
        {
          PROF_SCOPED_EVENT(op_->id(), event_emit_metadata);
          auto metadata = add_metadata(gxf_entity.value());
          if (metadata) {
            populate_output_metadata(metadata.value(), output_name);
          } else {
            HOLOSCAN_LOG_ERROR("{}.{}: Failed to attach metadata to output entity: {}",
                               op_->name(),
                               output_name,
                               GxfResultStr(metadata.error()));
          }
        }
      }

      if (stream_found) {
        {
          PROF_SCOPED_EVENT(op_->id(), event_emit_streams);
          // is_new_entity=true: skip get() check since entity was just created via Entity::New()
          auto stream_result = add_stream_id_to_entity(gxf_entity.value(), stream_cid, true, true);
          if (stream_result != GXF_SUCCESS) {
            throw std::runtime_error(
                fmt::format("{}.{}: failed to add CUDA stream to output message: {}",
                            op_->name(),
                            output_name,
                            GxfResultStr(stream_result)));
          }
        }
      }
      if (!omit_data_logging) {
        auto& data_loggers = op_->fragment()->data_loggers();
        HOLOSCAN_LOG_TRACE(
            "{}.{}: number of data loggers: {}", op_->name(), output_name, data_loggers.size());

        if (!data_loggers.empty()) {
          PROF_SCOPED_EVENT(op_->id(), event_data_logging);
          auto metadata_ptr = op_->is_metadata_enabled() ? op_->metadata() : nullptr;
          const std::string unique_id = output_spec->unique_id();

          // Check if a CUDA stream is being emitted on this output port
          auto stream_for_logging = stream_to_emit(output_name.c_str());

          for (auto& data_logger : data_loggers) {
            if (data_logger->should_log_output()) {
              PROF_SCOPED_EVENT(op_->id(), event_log_data);
              data_logger->log_data(data,
                                    unique_id,
                                    acq_timestamp,
                                    metadata_ptr,
                                    IOSpec::IOType::kOutput,
                                    stream_for_logging);
            }
          }
        }
      }

      // Set the data to the value of the Message object. Can only move **after** logging the data
      buffer.value()->set_value(data);

      // Publish the Entity object.
      nvidia::gxf::Expected<void> gxf_result;
      if (acq_timestamp != -1) {
        gxf_result = static_cast<nvidia::gxf::Transmitter*>(tx_ptr)->publish(gxf_entity.value(),
                                                                             acq_timestamp);
      } else {
        gxf_result =
            static_cast<nvidia::gxf::Transmitter*>(tx_ptr)->publish(std::move(gxf_entity.value()));
      }
      if (!gxf_result) {
        auto error_msg = fmt::format("{}.{}: Failed to publish output message with error: {}",
                                     op_->name(),
                                     output_name,
                                     GxfResultStr(gxf_result.error()));
        HOLOSCAN_LOG_ERROR(error_msg);
        throw std::runtime_error(error_msg);
      }
      break;
    }
    case OutputType::kGXFEntity: {
      // Cast to an Entity object and publish it.
      try {
        auto gxf_entity = std::any_cast<nvidia::gxf::Entity>(data);

        if (op_->is_metadata_enabled() && op_->metadata()->size() > 0) {
          {
            PROF_SCOPED_EVENT(op_->id(), event_emit_metadata);
            // The entity may be reused across ticks (e.g. the persistent cache.out_message
            // in InferenceOp's transmit_data_per_model). In that case the "metadata_"
            // component already exists from a prior tick, so add_metadata() would fail with
            // a duplicate-name error. Look up first, fall back to add on first use, and
            // clear any stale entries before re-populating so insert() starts from empty.
            auto metadata = get_metadata(gxf_entity);
            if (!metadata) {
              metadata = add_metadata(gxf_entity);
            }
            if (metadata) {
              metadata.value()->clear();
              populate_output_metadata(metadata.value(), output_name);
            } else {
              HOLOSCAN_LOG_ERROR("{}.{}: Failed to attach metadata to output entity: {}",
                                 op_->name(),
                                 output_name,
                                 GxfResultStr(metadata.error()));
            }
          }
        }

        if (stream_found) {
          {
            PROF_SCOPED_EVENT(op_->id(), event_emit_streams);
            // Pass is_new_entity to skip get() check when entity was just created
            auto stream_result =
                add_stream_id_to_entity(gxf_entity, stream_cid, true, is_new_entity);
            if (stream_result != GXF_SUCCESS) {
              throw std::runtime_error(
                  fmt::format("{}.{}: Failed to add CUDA stream to output message: {}",
                              op_->name(),
                              output_name,
                              GxfResultStr(stream_result)));
            }
            // Propagate stream to memory buffers for stream-aware deallocation
            // Skip if caller already set the stream (e.g., via Entity::add with stream parameter)
            if (!skip_stream_propagation) {
              propagate_stream_to_entity_memory_buffers(gxf_entity, gxf_context(), stream_cid);
            }
          }
        }

        if (!omit_data_logging) {
          auto& data_loggers = op_->fragment()->data_loggers();
          HOLOSCAN_LOG_TRACE("number of data loggers: {}", data_loggers.size());
          if (!data_loggers.empty()) {
            HOLOSCAN_LOG_TRACE("\t log_backend_specific code path");
            auto metadata_ptr = op_->is_metadata_enabled() ? op_->metadata() : nullptr;
            const std::string unique_id = output_spec->unique_id();

            // Check if a CUDA stream is being emitted on this output port
            auto stream_for_logging = stream_to_emit(output_name.c_str());

            for (auto& data_logger : data_loggers) {
              if (data_logger->should_log_output()) {
                // Create a shared entity to ensure proper lifetime management for async logging
                auto shared_entity_expected = gxf_entity.clone();
                if (shared_entity_expected) {
                  HOLOSCAN_LOG_TRACE("\t\t calling log_backend_specific");
                  PROF_SCOPED_EVENT(op_->id(), event_log_backend_specific);
                  data_logger->log_backend_specific(shared_entity_expected.value(),
                                                    unique_id,
                                                    acq_timestamp,
                                                    metadata_ptr,
                                                    IOSpec::IOType::kOutput,
                                                    stream_for_logging);
                } else {
                  HOLOSCAN_LOG_ERROR("{}.{}: Failed to create shared entity for logging: {}",
                                     op_->name(),
                                     output_name,
                                     GxfResultStr(shared_entity_expected.error()));
                }
              }
            }
          }
        }

        nvidia::gxf::Expected<void> gxf_result;
        if (acq_timestamp != -1) {
          gxf_result =
              static_cast<nvidia::gxf::Transmitter*>(tx_ptr)->publish(gxf_entity, acq_timestamp);
        } else {
          gxf_result =
              static_cast<nvidia::gxf::Transmitter*>(tx_ptr)->publish(std::move(gxf_entity));
        }
        if (!gxf_result) {
          auto error_msg = fmt::format("{}.{}: failed to publish output message with error: {}",
                                       op_->name(),
                                       output_name,
                                       GxfResultStr(gxf_result.error()));
          HOLOSCAN_LOG_ERROR(error_msg);
          throw std::runtime_error(error_msg);
        }
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_ERROR(
            "{}.{}: unable to cast to gxf::Entity: {}", op_->name(), output_name, e.what());
      }
      break;
    }
  }
}
}  // namespace holoscan::gxf
