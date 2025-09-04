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

#include "holoscan/data_loggers/async_console_logger/async_console_backend.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

#include "gxf/core/entity.hpp"
#include "holoscan/core/metadata.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/domain/tensor_map.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {
namespace data_loggers {

// External reference to shared mutex for thread-safe console output coordination
// (to be used as needed to prevent interleaved output from separate loggers)
extern std::mutex console_output_mutex;

// AsyncConsoleBackend implementation
AsyncConsoleBackend::AsyncConsoleBackend(std::shared_ptr<SimpleTextSerializer> serializer)
    : serializer_(std::move(serializer)) {}

bool AsyncConsoleBackend::initialize() {
  initialized_.store(true);
  entries_written_.store(0);
  large_entries_written_.store(0);

  HOLOSCAN_LOG_INFO("AsyncConsoleLogger Initialized");
  return true;
}

void AsyncConsoleBackend::shutdown() {
  HOLOSCAN_LOG_INFO("AsyncConsoleLogger Shutdown - entries: {}, Large entries: {}",
                    entries_written_.load(),
                    large_entries_written_.load());
  initialized_.store(false);
}

bool AsyncConsoleBackend::process_data_entry(const DataEntry& entry) {
  if (!initialized_.load()) {
    return false;
  }

  try {
    bool success = log_entry(entry);
    if (success) {
      entries_written_.fetch_add(1, std::memory_order_relaxed);
    }
    return success;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR(
        "AsyncConsoleLogger Error processing entry {}: {}", entry.unique_id, e.what());
    return false;
  }
}

bool AsyncConsoleBackend::process_large_data_entry(const DataEntry& entry) {
  if (!initialized_.load()) {
    return false;
  }

  try {
    bool success = log_large_entry(entry);
    if (success) {
      large_entries_written_.fetch_add(1, std::memory_order_relaxed);
    }
    return success;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR(
        "AsyncConsoleLogger Error processing large entry {}: {}", entry.unique_id, e.what());
    return false;
  }
}

std::string AsyncConsoleBackend::get_statistics() const {
  std::ostringstream ss;
  ss << "AsyncConsoleBackend - written: " << entries_written_.load()
     << ", Large written: " << large_entries_written_.load()
     << ", Initialized: " << (initialized_.load() ? "Yes" : "No");
  return ss.str();
}

bool AsyncConsoleBackend::log_entry(const DataEntry& entry) {
  if (!serializer_) {
    HOLOSCAN_LOG_ERROR("AsyncConsoleLogger No serializer available for entry: {}", entry.unique_id);
    return false;
  }

  std::string serialized_content = serialize_data_content(entry);
  std::string type_name = get_data_type_name(entry);

  // Add metadata if available and enabled
  if (log_metadata_.load() && entry.metadata) {
    // For async background thread safety, catch any exceptions during metadata serialization
    // This prevents any potential segfault when Python objects are accessed from background
    // threads
    std::string metadata_str;
    try {
      metadata_str = serializer_->serialize_metadata_to_string(entry.metadata);
    } catch (const std::exception& e) {
      metadata_str = std::string("Metadata serialization error: ") + e.what();
    } catch (...) {
      metadata_str = "Metadata serialization error: Unknown exception";
    }

    serialized_content += "\n" + metadata_str;
  }

  // Determine log category based on data type (similar to BasicConsoleLogger)
  std::string log_category;
  switch (entry.type) {
    case DataEntry::Generic:
      log_category = "Message (std::any)";
      break;
    case DataEntry::TensorData:
      log_category = "Tensor";
      break;
    case DataEntry::TensorMapData:
      log_category = "TensorMap";
      break;
    default:
      log_category = "Unknown";
      break;
  }

  // Format the log message similar to BasicConsoleLogger using HOLOSCAN_LOG_INFO
  {
    // In general, should protect console output with mutex in case of concurrent access from
    // multiple threads to prevent interleaved output. Should not be necessary here since there is
    // only a single logging statement.
    // std::lock_guard<std::mutex> lock(console_output_mutex);
    HOLOSCAN_LOG_INFO(
        "AsyncConsoleLogger[ID:{}][Acquisition Timestamp:{}][{}:{}][Category:{}][Type Name: {}] {}",
        entry.unique_id,
        entry.acquisition_timestamp,
        entry.io_type == IOSpec::IOType::kOutput ? "Emit Timestamp" : "Receive Timestamp",
        entry.emit_timestamp,
        log_category,
        type_name,
        serialized_content);
  }
  return true;
}

bool AsyncConsoleBackend::log_large_entry(const DataEntry& entry) {
  if (!serializer_) {
    HOLOSCAN_LOG_ERROR("AsyncConsoleLogger No serializer available for large entry: {}",
                       entry.unique_id);
    return false;
  }

  std::string serialized_content = serialize_large_data_content(entry);
  std::string type_name = get_large_data_type_name(entry);

  // Add metadata if available and enabled
  if (log_metadata_.load() && entry.metadata) {
    // For async background thread safety, catch any exceptions during metadata serialization
    // This prevents any potential segfault when Python objects are accessed from background
    // threads
    std::string metadata_str;
    try {
      metadata_str = serializer_->serialize_metadata_to_string(entry.metadata);
    } catch (const std::exception& e) {
      metadata_str = std::string("Metadata serialization error: ") + e.what();
    } catch (...) {
      metadata_str = "Metadata serialization error: Unknown exception";
    }

    serialized_content += "\n" + metadata_str;
  }
  // Format the log message similar to BasicConsoleLogger using HOLOSCAN_LOG_INFO
  {
    // In general, should protect console output with mutex in case of concurrent access from
    // multiple threads to prevent interleaved output. Should not be necessary here since there is
    // only a single logging statement.
    // std::lock_guard<std::mutex> lock(console_output_mutex);
    HOLOSCAN_LOG_INFO(
        "AsyncConsoleLogger[ID:{}][Acquisition Timestamp:{}][{}:{}][Category:Content][Type Name: "
        "{}] "
        "{}",
        entry.unique_id,
        entry.acquisition_timestamp,
        entry.io_type == IOSpec::IOType::kOutput ? "Emit Timestamp" : "Receive Timestamp",
        entry.emit_timestamp,
        type_name,
        serialized_content);
    return true;
  }
}

std::string AsyncConsoleBackend::get_data_type_name(const DataEntry& entry) const {
  switch (entry.type) {
    case DataEntry::Generic:
      return "Generic";
    case DataEntry::TensorData:
      return "Tensor";
    case DataEntry::TensorMapData:
      return "TensorMap";
    default:
      return "Unknown";
  }
}

std::string AsyncConsoleBackend::get_large_data_type_name(const DataEntry& entry) const {
  switch (entry.type) {
    case DataEntry::TensorData:
      return "Tensor";
    case DataEntry::TensorMapData:
      return "TensorMap";
    default:
      return "Unknown";
  }
}

std::string AsyncConsoleBackend::serialize_data_content(const DataEntry& entry) const {
  try {
    switch (entry.type) {
      case DataEntry::Generic:
        if (std::holds_alternative<std::any>(entry.data)) {
          auto& any_data = std::get<std::any>(entry.data);
          if (serializer_->can_handle_message(any_data.type())) {
            return serializer_->serialize_to_string(any_data);
          }
        }
        return "Generic data (serializer cannot handle type)";

      case DataEntry::TensorData:
        if (std::holds_alternative<std::shared_ptr<holoscan::Tensor>>(entry.data)) {
          auto tensor = std::get<std::shared_ptr<holoscan::Tensor>>(entry.data);
          // For data, we might serialize without tensor content (metadata only)
          return serializer_->serialize_tensor_to_string(tensor, false);
        }
        return "Tensor (null)";

      case DataEntry::TensorMapData:
        if (std::holds_alternative<holoscan::TensorMap>(entry.data)) {
          auto& tensor_map = std::get<holoscan::TensorMap>(entry.data);
          // For data, we might serialize without tensor content (metadata only)
          return serializer_->serialize_tensormap_to_string(tensor_map, false);
        }
        return "TensorMap (empty)";

      default:
        return "Unknown data type";
    }
  } catch (const std::exception& e) {
    return std::string("Serialization error: ") + e.what();
  }
}

std::string AsyncConsoleBackend::serialize_large_data_content(const DataEntry& entry) const {
  try {
    switch (entry.type) {
      case DataEntry::TensorData:
        if (std::holds_alternative<std::shared_ptr<holoscan::Tensor>>(entry.data)) {
          auto tensor = std::get<std::shared_ptr<holoscan::Tensor>>(entry.data);
          return serializer_->serialize_tensor_to_string(tensor, log_tensor_data_content_.load());
        }
        return "Tensor (null)";

      case DataEntry::TensorMapData:
        if (std::holds_alternative<holoscan::TensorMap>(entry.data)) {
          auto& tensor_map = std::get<holoscan::TensorMap>(entry.data);
          return serializer_->serialize_tensormap_to_string(tensor_map,
                                                            log_tensor_data_content_.load());
        }
        return "TensorMap (empty)";

      default:
        return "Unknown large data type";
    }
  } catch (const std::exception& e) {
    return std::string("Serialization error: ") + e.what();
  }
}

}  // namespace data_loggers
}  // namespace holoscan
