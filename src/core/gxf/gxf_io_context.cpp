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

#include "holoscan/core/gxf/gxf_io_context.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/domain/tensor_map.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_cuda.hpp"
#include "holoscan/core/gxf/gxf_execution_context.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/message.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/profiler/profiler.hpp"

#include "gxf/core/gxf.h"
#include "gxf/std/receiver.hpp"
#include "gxf/std/timestamp.hpp"
#include "gxf/std/transmitter.hpp"

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

  gxf_tid_t rx_tid{};
  gxf_context_t context = gxf_resource->gxf_context();
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentTypeId(context, gxf_resource->gxf_typename(), &rx_tid));
  void* rx_ptr = nullptr;
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentPointer(context, gxf_resource->gxf_cid(), rx_tid, &rx_ptr));
  return static_cast<nvidia::gxf::Receiver*>(rx_ptr);
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
                                       bool omit_data_logging) {
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
            "Multiple CUDA streams ({}) received on input port '{}', using first stream for data "
            "logging",
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
  auto receiver = get_gxf_receiver(io_spec);
  if (!receiver) {
    auto no_accessible_error_message = NoAccessibleMessageType(
        fmt::format("Invalid receiver found for the input port with name {}", input_name));

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
      auto maybe_metadata = entity.get<holoscan::MetadataDictionary>("metadata_");
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
    // Get first timestamp encountered or use findAllHeap to get all timestamps in the Entity?
    // auto maybe_timestamp = entity.get<nvidia::gxf::Timestamp>();
    auto timestamp_components = entity.findAllHeap<nvidia::gxf::Timestamp>();
    int64_t gxf_acquisition_timestamp = 0;
    if (0 == timestamp_components->size()) {
      // Requires Timestamp instance for message age
      HOLOSCAN_LOG_TRACE("Message received on input port '{}' carries no Timestamp.", input_name);
    } else {
      gxf_acquisition_timestamp = timestamp_components->front().value()->acqtime;
      HOLOSCAN_LOG_TRACE("Message received on input port '{}' has GXF Timestamp with acqtime: {}.",
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
            HOLOSCAN_LOG_ERROR("Failed to create shared entity for logging: {}",
                               GxfResultStr(shared_entity_expected.error()));
          }
        }
      }
    }
    // Convert nvidia::gxf::Entity to holoscan::gxf::Entity
    holoscan::gxf::Entity entity_wrapper(entity);
    return entity_wrapper;  // to handle gxf::Entity as it is
  }

  auto message = entity.get<holoscan::Message>();
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
            HOLOSCAN_LOG_ERROR("Failed to create shared entity for logging: {}",
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

void GXFOutputContext::populate_output_metadata(nvidia::gxf::Handle<MetadataDictionary> metadata) {
  // insert the operator's metadata into the provided (empty) metadata object
  auto dynamic_metadata = op_->metadata();
  metadata->insert(*dynamic_metadata);
  HOLOSCAN_LOG_DEBUG("output context: op '{}' metadata.size() = {}", op_->name(), metadata->size());
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

namespace {

gxf_result_t add_stream_id_to_entity(nvidia::gxf::Entity& gxf_entity, gxf_uid_t stream_cid) {
  const auto maybe_stream_id = gxf_entity.add<nvidia::gxf::CudaStreamId>("cuda_stream_id_");
  if (!maybe_stream_id) {
    auto code = nvidia::gxf::ToResultCode(maybe_stream_id);
    HOLOSCAN_LOG_ERROR("Failed to add CUDA stream id to output message with error: {}.",
                       GxfResultStr(code));
    return GXF_FAILURE;
  }
  maybe_stream_id.value()->stream_cid = stream_cid;
  return GXF_SUCCESS;
}

}  // namespace

void GXFOutputContext::emit_impl(std::any data, const char* name, OutputType out_type,
                                 const int64_t acq_timestamp, bool omit_data_logging) {
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
    HOLOSCAN_LOG_ERROR("Invalid resource type");
    return;
  }

  gxf_tid_t tx_tid;
  gxf_context_t context = gxf_resource->gxf_context();
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentTypeId(context, gxf_resource->gxf_typename(), &tx_tid));

  void* tx_ptr;
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentPointer(context, gxf_resource->gxf_cid(), tx_tid, &tx_ptr));

  HOLOSCAN_LOG_TRACE("in GXFOutputContext::emit_impl: out_type: {}", static_cast<int>(out_type));
  switch (out_type) {
    case OutputType::kAny: {
      // Create an Entity object and add a Message object to it.
      auto gxf_entity = nvidia::gxf::Entity::New(gxf_context());
      auto buffer = gxf_entity.value().add<Message>();

      if (op_->is_metadata_enabled() && op_->metadata()->size() > 0) {
        {
          PROF_SCOPED_EVENT(op_->id(), event_emit_metadata);
          auto metadata = gxf_entity.value().add<MetadataDictionary>("metadata_");
          populate_output_metadata(metadata.value());
        }
      }

      if (stream_found) {
        {
          PROF_SCOPED_EVENT(op_->id(), event_emit_streams);
          auto stream_result = add_stream_id_to_entity(gxf_entity.value(), stream_cid);
          if (stream_result != GXF_SUCCESS) {
            throw std::runtime_error(fmt::format("Failed to add CUDA stream to output message: {}",
                                                 GxfResultStr(stream_result)));
          }
        }
      }
      if (!omit_data_logging) {
        auto& data_loggers = op_->fragment()->data_loggers();
        HOLOSCAN_LOG_TRACE("number of data loggers: {}", data_loggers.size());

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
      buffer.value()->set_value(std::move(data));

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
        auto error_msg = fmt::format("Failed to publish output message with error: {}",
                                     GxfResultStr(gxf_result.error()));
        HOLOSCAN_LOG_ERROR(error_msg);
        throw std::runtime_error(error_msg);
      }
      break;
    }
    case OutputType::kGXFEntity: {
      // Cast to an Entity object and publish it.
      try {
        auto gxf_entity = std::any_cast<nvidia::gxf::Entity>(std::move(data));

        if (op_->is_metadata_enabled() && op_->metadata()->size() > 0) {
          {
            PROF_SCOPED_EVENT(op_->id(), event_emit_metadata);
            auto metadata = gxf_entity.add<MetadataDictionary>("metadata_");
            populate_output_metadata(metadata.value());
          }
        }

        if (stream_found) {
          {
            PROF_SCOPED_EVENT(op_->id(), event_emit_streams);
            auto stream_result = add_stream_id_to_entity(gxf_entity, stream_cid);
            if (stream_result != GXF_SUCCESS) {
              throw std::runtime_error(fmt::format(
                  "Failed to add CUDA stream to output message: {}", GxfResultStr(stream_result)));
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
                  HOLOSCAN_LOG_ERROR("Failed to create shared entity for logging: {}",
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
          auto error_msg = fmt::format("Failed to publish output message with error: {}",
                                       GxfResultStr(gxf_result.error()));
          HOLOSCAN_LOG_ERROR(error_msg);
          throw std::runtime_error(error_msg);
        }
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_ERROR("Unable to cast to gxf::Entity: {}", e.what());
      }
      break;
    }
  }
}
}  // namespace holoscan::gxf
