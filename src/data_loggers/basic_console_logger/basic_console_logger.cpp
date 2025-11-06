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

#include "holoscan/data_loggers/basic_console_logger/basic_console_logger.hpp"

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>  // For std::shared_ptr in parameters
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/domain/tensor_map.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/data_logger.hpp"
#include "holoscan/data_loggers/basic_console_logger/simple_text_serializer.hpp"

namespace holoscan {
namespace data_loggers {

// External reference to shared mutex for thread-safe console output coordination
// (to be used as needed to prevent interleaved output from separate loggers)
extern std::mutex console_output_mutex;

void BasicConsoleLogger::setup(ComponentSpec& spec) {
  spec.param(serializer_, "serializer", "Serializer", "Serializer to use for logging data");
  // setup the parameters present on the base DataLoggerResource
  DataLoggerResource::setup(spec);
}

void BasicConsoleLogger::initialize() {
  // Find if there is an argument for 'serializer'
  auto has_serializer = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "serializer"); });

  // Create appropriate serializer if none was provided
  if (has_serializer == args().end()) {
    add_arg(Arg("serializer", fragment()->make_resource<SimpleTextSerializer>("serializer")));
  }

  // call parent initialize after adding missing serializer arg above
  DataLoggerResource::initialize();
}

bool BasicConsoleLogger::log_data(const std::any& data, const std::string& unique_id,
                                  int64_t acquisition_timestamp,
                                  const std::shared_ptr<MetadataDictionary>& metadata,
                                  IOSpec::IOType io_type, std::optional<cudaStream_t> stream) {
  // Check if this message should be logged based on allowlist/denylist patterns
  if (!should_log_message(unique_id)) {
    HOLOSCAN_LOG_DEBUG(
        "BasicConsoleLogger: Message '{}' filtered out by allowlist/denylist patterns", unique_id);
    return true;  // Consider filtered messages as successfully "handled"
  }

  // Check for empty or void data
  if (!data.has_value()) {
    HOLOSCAN_LOG_DEBUG("BasicConsoleLogger: Skipping empty data for message '{}'", unique_id);
    return true;
  }

  // Check for void type specifically
  if (data.type() == typeid(void)) {
    HOLOSCAN_LOG_DEBUG("BasicConsoleLogger: Skipping void type data for message '{}'", unique_id);
    return true;  // Consider void data as successfully "logged"
  }

  if (!serializer_.has_value()) {
    HOLOSCAN_LOG_WARN("BasicConsoleLogger: No serializer set.");
    std::string type_name = data.has_value() ? data.type().name() : "unknown";
    return false;
  }

  // Get current timestamp and assign based on port type
  int64_t current_timestamp = get_timestamp();

  // Runtime type dispatch for specialized serialization
  std::string serialized_content;
  std::string log_category;
  std::string type_name = data.type().name();

  // Check for Tensor type
  if (data.type() == typeid(std::shared_ptr<Tensor>)) {
    // Tensor serialization
    auto tensor = std::any_cast<std::shared_ptr<Tensor>>(data);
    bool log_data_content = should_log_tensor_data_content();
    serialized_content = serializer_->serialize_tensor_to_string(tensor, log_data_content, stream);
    log_category = "Tensor";
  } else if (data.type() == typeid(TensorMap)) {
    // TensorMap serialization
    auto tensor_map = std::any_cast<TensorMap>(data);
    bool log_data_content = should_log_tensor_data_content();
    serialized_content =
        serializer_->serialize_tensormap_to_string(tensor_map, log_data_content, stream);
    log_category = "TensorMap";
  } else {
    // General case
    if (!serializer_->can_handle_message(data.type())) {
      HOLOSCAN_LOG_WARN(
          "BasicConsoleLogger: Cannot handle message '{}' with type '{}'.", unique_id, type_name);
      return false;
    }
    serialized_content = serializer_->serialize_to_string(data);
    log_category = "Message (std::any)";
  }

  if (should_log_metadata() && metadata) {
    serialized_content += "\n" + serializer_->serialize_metadata_to_string(metadata);
  }

  // Check if serialization succeeded
  if (serialized_content.empty() && data.has_value()) {
    HOLOSCAN_LOG_WARN("BasicConsoleLogger: Serialization failed for message '{}' with type '{}'.",
                      unique_id,
                      type_name);
    return false;
  }

  // Format the log message similar to BasicConsoleLogger using HOLOSCAN_LOG_INFO
  {
    std::string stream_info;
    if (stream.has_value()) {
      if (stream.value() == cudaStreamDefault) {
        stream_info = "Stream: default";
      } else if (stream.value() == cudaStreamLegacy) {
        stream_info = "Stream: legacy";
      } else if (stream.value() == cudaStreamPerThread) {
        stream_info = "Stream: per-thread";
      } else {
        stream_info = fmt::format("Stream: 0x{:x}", reinterpret_cast<uintptr_t>(stream.value()));
      }
    } else {
      stream_info = "Stream: none";
    }

    // In general, should protect console output with mutex in case of concurrent access from
    // multiple threads to prevent interleaved output. Should not be necessary here since there is
    // only a single logging statement.
    // std::lock_guard<std::mutex> lock(console_output_mutex);
    HOLOSCAN_LOG_INFO(
        "BasicConsoleLogger[ID:{}][Acquisition Timestamp:{}][{}:{}][Category:{}][Type Name: "
        "{}][{}] {}",
        unique_id,
        acquisition_timestamp,
        io_type == IOSpec::IOType::kOutput ? "Emit Timestamp" : "Receive Timestamp",
        current_timestamp,
        log_category,
        type_name,
        stream_info,
        serialized_content);
  }

  return true;
}

bool BasicConsoleLogger::log_tensor_data(const std::shared_ptr<Tensor>& tensor,
                                         const std::string& unique_id,
                                         int64_t acquisition_timestamp,
                                         const std::shared_ptr<MetadataDictionary>& metadata,
                                         IOSpec::IOType io_type,
                                         std::optional<cudaStream_t> stream) {
  // Convert to std::any and dispatch to unified log_data method
  std::any data = tensor;
  return log_data(std::move(data), unique_id, acquisition_timestamp, metadata, io_type, stream);
}

bool BasicConsoleLogger::log_tensormap_data(const TensorMap& tensor_map,
                                            const std::string& unique_id,
                                            int64_t acquisition_timestamp,
                                            const std::shared_ptr<MetadataDictionary>& metadata,
                                            IOSpec::IOType io_type,
                                            std::optional<cudaStream_t> stream) {
  // Convert to std::any and dispatch to unified log_data method
  std::any data = tensor_map;
  return log_data(std::move(data), unique_id, acquisition_timestamp, metadata, io_type, stream);
}

bool BasicConsoleLogger::log_backend_specific(const std::any& data, const std::string& unique_id,
                                              int64_t acquisition_timestamp,
                                              const std::shared_ptr<MetadataDictionary>& metadata,
                                              IOSpec::IOType io_type,
                                              std::optional<cudaStream_t> stream) {
  // Default implementation: backend-specific logging is not supported
  HOLOSCAN_LOG_DEBUG(
      "BasicConsoleLogger: Backend-specific logging not supported for type '{}'."
      "Please use GXFConsoleLogger instead to log this type.",
      data.type().name());
  // still log metadata and timestamp
  auto result = log_data(data, unique_id, acquisition_timestamp, metadata, io_type, stream);
  if (!result) {
    HOLOSCAN_LOG_ERROR("BasicConsoleLogger: Failed to log metadata for port '{}'.", unique_id);
  }
  // return false since data was not processed
  return false;
}

}  // namespace data_loggers
}  // namespace holoscan
