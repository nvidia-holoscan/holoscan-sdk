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

#include "holoscan/core/gxf/gxf_io_context.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/message.hpp"

#include "gxf/std/receiver.hpp"
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
  if (execution_context_) { return execution_context_->context(); }
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

std::any GXFInputContext::receive_impl(const char* name, bool no_error_message) {
  std::string input_name = holoscan::get_well_formed_name(name, inputs_);

  auto it = inputs_.find(input_name);
  if (it == inputs_.end()) {
    if (no_error_message) { return kNoReceivedMessage; }
    // Show error message because the input name is not found.
    if (inputs_.size() == 1) {
      auto no_accessible_error_message = NoAccessibleMessageType(fmt::format(
          "The operator({}) has only one port with label '{}' but the non-existent port label "
          "'{}' was specified in the receive() method",
          op_->name(),
          inputs_.begin()->first,
          name));

      return no_accessible_error_message;
    } else {
      if (inputs_.empty()) {
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

  auto receiver = get_gxf_receiver(it->second);
  if (!receiver) {
    auto no_accessible_error_message = NoAccessibleMessageType(
        fmt::format("Invalid receiver found for the input port with name {}", input_name));

    return no_accessible_error_message;
  }

  auto entity = receiver->receive();
  if (!entity || entity.value().is_null()) {
    return kNoReceivedMessage;  // to indicate that there is no data
  }

  // Update operator metadata using any metadata found in the entity.
  if (op_->is_metadata_enabled()) {
    // Merge metadata from all input ports into the dynamic metadata of the operator
    auto maybe_metadata = entity.value().get<holoscan::MetadataDictionary>("metadata_");
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

  auto message = entity.value().get<holoscan::Message>();
  if (!message) {
    // Convert nvidia::gxf::Entity to holoscan::gxf::Entity
    holoscan::gxf::Entity entity_wrapper(entity.value());
    return entity_wrapper;  // to handle gxf::Entity as it is
  }

  auto message_ptr = message.value();
  auto value = message_ptr->value();

  return value;
}

GXFOutputContext::GXFOutputContext(ExecutionContext* execution_context, Operator* op)
    : OutputContext(execution_context, op) {}

GXFOutputContext::GXFOutputContext(
    ExecutionContext* execution_context, Operator* op,
    std::unordered_map<std::string, std::shared_ptr<IOSpec>>& outputs)
    : OutputContext(execution_context, op, outputs) {}

gxf_context_t GXFOutputContext::gxf_context() const {
  if (execution_context_) { return execution_context_->context(); }
  return nullptr;
}

void GXFOutputContext::populate_output_metadata(nvidia::gxf::Handle<MetadataDictionary> metadata) {
  // insert the operator's metadata into the provided (empty) metadata object
  auto dynamic_metadata = op_->metadata();
  metadata->insert(*dynamic_metadata);
  HOLOSCAN_LOG_DEBUG("output context: op '{}' metadata.size() = {}", op_->name(), metadata->size());
}

void GXFOutputContext::emit_impl(std::any data, const char* name, OutputType out_type,
                                 const int64_t acq_timestamp) {
  std::string output_name = holoscan::get_well_formed_name(name, outputs_);

  auto it = outputs_.find(output_name);
  if (it == outputs_.end()) {
    // Show error message because the output name is not found.
    if (outputs_.size() == 1) {
      HOLOSCAN_LOG_ERROR(
          "The operator({}) has only one port with label '{}' but the non-existent port label "
          "'{}' was specified in the emit() method",
          op_->name(),
          outputs_.begin()->first,
          name);
      return;
    } else {
      if (outputs_.empty()) {
        HOLOSCAN_LOG_ERROR(
            "The operator({}) does not have any output port but '{}' was specified in "
            "emit() method",
            op_->name(),
            output_name);
        return;
      }

      auto msg_buf = fmt::memory_buffer();
      auto& op_outputs = op_->spec()->outputs();
      for (const auto& [label, _] : op_outputs) {
        if (&label == &(op_outputs.begin()->first)) {
          fmt::format_to(std::back_inserter(msg_buf), "{}", label);
        } else {
          fmt::format_to(std::back_inserter(msg_buf), ", {}", label);
        }
      }
      HOLOSCAN_LOG_ERROR(
          "The operator({}) does not have an output port with label '{}'. It should be "
          "one of ({:.{}}) in emit() method",
          op_->name(),
          output_name,
          msg_buf.data(),
          msg_buf.size());
      return;
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

  switch (out_type) {
    case OutputType::kSharedPointer:
    case OutputType::kAny: {
      // Create an Entity object and add a Message object to it.
      auto gxf_entity = nvidia::gxf::Entity::New(gxf_context());
      auto buffer = gxf_entity.value().add<Message>();
      // Set the data to the value of the Message object.
      buffer.value()->set_value(data);

      if (op_->is_metadata_enabled() && op_->metadata()->size() > 0) {
        auto metadata = gxf_entity.value().add<MetadataDictionary>("metadata_");
        populate_output_metadata(metadata.value());
      }

      // Publish the Entity object.
      // TODO(gbae): Check error message
      if (acq_timestamp != -1) {
        static_cast<nvidia::gxf::Transmitter*>(tx_ptr)->publish(gxf_entity.value(), acq_timestamp);
      } else {
        static_cast<nvidia::gxf::Transmitter*>(tx_ptr)->publish(std::move(gxf_entity.value()));
      }
      break;
    }
    case OutputType::kGXFEntity: {
      // Cast to an Entity object and publish it.
      try {
        auto gxf_entity = std::any_cast<nvidia::gxf::Entity>(data);

        if (op_->is_metadata_enabled() && op_->metadata()->size() > 0) {
          auto metadata = gxf_entity.add<MetadataDictionary>("metadata_");
          populate_output_metadata(metadata.value());
        }

        // TODO(gbae): Check error message
        if (acq_timestamp != -1) {
          static_cast<nvidia::gxf::Transmitter*>(tx_ptr)->publish(gxf_entity, acq_timestamp);
        } else {
          static_cast<nvidia::gxf::Transmitter*>(tx_ptr)->publish(std::move(gxf_entity));
        }
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_ERROR("Unable to cast to gxf::Entity: {}", e.what());
      }
      break;
    }
  }
}

}  // namespace holoscan::gxf
