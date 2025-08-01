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

#include "holoscan/data_loggers/async_console_logger/async_console_logger.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

#include "gxf/core/entity.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/domain/tensor_map.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/metadata.hpp"
#include "holoscan/data_loggers/async_console_logger/async_console_backend.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {
namespace data_loggers {

// AsyncConsoleLogger implementation
void AsyncConsoleLogger::setup(ComponentSpec& spec) {
  // call parent AsyncDataLoggerResource::setup first
  AsyncDataLoggerResource::setup(spec);

  spec.param(serializer_, "serializer", "Serializer", "Serializer to use for logging data");
}

void AsyncConsoleLogger::initialize() {
  // Find if there is an argument for 'serializer'
  auto has_serializer = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "serializer"); });

  std::shared_ptr<SimpleTextSerializer> serializer;
  // Create appropriate serializer if none was provided
  if (has_serializer == args().end()) {
    serializer = fragment()->make_resource<SimpleTextSerializer>("serializer");
    add_arg(Arg("serializer", serializer));
  } else {
    HOLOSCAN_LOG_DEBUG("AsyncConsoleLogger: serializer found: type={}",
                       has_serializer->value().type().name());
    auto serializer_resource = std::any_cast<std::shared_ptr<Resource>>(has_serializer->value());
    HOLOSCAN_LOG_DEBUG("AsyncConsoleLogger: serializer_resource created, but is null: {}",
                       serializer_resource == nullptr);
    serializer = std::dynamic_pointer_cast<SimpleTextSerializer>(serializer_resource);
    if (!serializer) {
      // current code continues silently
      HOLOSCAN_LOG_ERROR("AsyncConsoleLogger: Provided serializer is not a SimpleTextSerializer");
      throw std::runtime_error("AsyncConsoleLogger requires a SimpleTextSerializer");
    }
    HOLOSCAN_LOG_DEBUG("AsyncConsoleLogger: serializer set");
  }

  // Create and configure the backend
  auto backend = std::make_shared<AsyncConsoleBackend>(serializer);
  backend->set_log_metadata(copy_value_from_args<bool>("log_metadata", true));
  backend->set_log_tensor_data_content(
      copy_value_from_args<bool>("log_tensor_data_content", false));

  // Set the backend BEFORE calling parent initialize (which starts worker threads)
  set_backend(backend);

  // Call parent initialize to start the worker threads
  AsyncDataLoggerResource::initialize();
  HOLOSCAN_LOG_DEBUG("AsyncConsoleLogger initialized");
}

bool AsyncConsoleLogger::log_backend_specific(const std::any& data, const std::string& unique_id,
                                              int64_t acquisition_timestamp,
                                              const std::shared_ptr<MetadataDictionary>& metadata,
                                              IOSpec::IOType io_type) {
  HOLOSCAN_LOG_TRACE("AsyncConsoleLogger: log_backend_specific called for unique_id: {}",
                     unique_id);
  // Check if this message should be logged based on allowlist/denylist patterns
  if (!should_log_message(unique_id)) {
    HOLOSCAN_LOG_DEBUG(
        "AsyncConsoleLogger: Backend-specific message '{}' filtered out by allowlist/denylist "
        "patterns",
        unique_id);
    return true;  // Consider filtered messages as successfully "handled"
  }

  // Check for empty data
  if (!data.has_value()) {
    HOLOSCAN_LOG_DEBUG("AsyncConsoleLogger: Skipping empty backend-specific data for message '{}'",
                       unique_id);
    return true;
  }

  // metadata is often updated in-place, so we need to make a copy here at the time of logging if
  // it is to be logged.
  std::shared_ptr<MetadataDictionary> metadata_copy;
  if (metadata) {
    metadata_copy = std::make_shared<MetadataDictionary>(*metadata);
  }
  // Runtime type checking for GXF Entity types
  static const std::type_index nvidia_gxf_entity_type(typeid(nvidia::gxf::Entity));
  static const std::type_index holoscan_gxf_entity_type(typeid(holoscan::gxf::Entity));

  std::type_index data_type_index(data.type());
  bool is_nvidia_entity = (data_type_index == nvidia_gxf_entity_type);
  bool is_holoscan_entity = (data_type_index == holoscan_gxf_entity_type);

  if (is_nvidia_entity || is_holoscan_entity) {
    std::string entity_type_str =
        is_nvidia_entity ? "nvidia::gxf::Entity" : "holoscan::gxf::Entity";

    // Extract tensor components from the entity and log them
    try {
      nvidia::gxf::Entity gxf_entity;

      if (is_nvidia_entity) {
        gxf_entity = std::any_cast<nvidia::gxf::Entity>(data);
      } else {
        // holoscan::gxf::Entity inherits from nvidia::gxf::Entity
        auto holoscan_entity = std::any_cast<holoscan::gxf::Entity>(data);
        gxf_entity = static_cast<nvidia::gxf::Entity>(holoscan_entity);
      }

      // Find and log tensor components within the entity
      auto tensor_components_expected = gxf_entity.findAllHeap<nvidia::gxf::Tensor>();
      if (!tensor_components_expected) {
        HOLOSCAN_LOG_ERROR("Failed to enumerate tensor components: {}",
                           GxfResultStr(tensor_components_expected.error()));
        return false;
      }
      if (!tensor_components_expected->empty()) {
        TensorMap tensor_map;
        for (const auto& gxf_tensor : tensor_components_expected.value()) {
          // Do zero-copy conversion to holoscan::Tensor
          auto maybe_dl_ctx = (*gxf_tensor->get()).toDLManagedTensorContext();
          if (!maybe_dl_ctx) {
            HOLOSCAN_LOG_ERROR(
                "Failed to get std::shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
            continue;
          }
          auto holoscan_tensor = std::make_shared<Tensor>(maybe_dl_ctx.value());
          tensor_map.insert({gxf_tensor->name(), holoscan_tensor});
        }

        if (tensor_map.size() > 0) {
          // Log the tensor map found in the entity

          // TODO(grelee): If we want to log other components then we shouldn't call this logging
          // function directly here to avoid multiple log entries for the entity.
          // (We would use serializer_->serialize_tensormap_to_string(), etc. directly instead.)
          if (!log_tensormap_data(
                  tensor_map, unique_id, acquisition_timestamp, metadata_copy, io_type)) {
            HOLOSCAN_LOG_ERROR("Logging of TensorMap data from Entity failed");
            return false;
          }
        }
      }
      // Get current timestamp
      int64_t current_timestamp = get_timestamp();
      (void)current_timestamp;

      // TODO: handle any other component types (VideoBuffer, etc.)?
    } catch (const std::bad_any_cast& e) {
      HOLOSCAN_LOG_ERROR("AsyncConsoleLogger: Failed to cast entity data to expected type '{}': {}",
                         entity_type_str,
                         e.what());
      return false;
    }

    return true;
  } else {
    HOLOSCAN_LOG_ERROR("AsyncConsoleLogger: Unsupported data type: {}", data.type().name());
    return false;
  }
}

}  // namespace data_loggers
}  // namespace holoscan
