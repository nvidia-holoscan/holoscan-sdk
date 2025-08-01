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
#include "holoscan/core/resources/data_logger.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

// Shared mutex for thread-safe console output coordination across all console logger types
std::mutex console_output_mutex;

void DataLoggerResource::setup(ComponentSpec& spec) {
  // logging parameters
  spec.param(log_outputs_, "log_outputs", "Log Outputs", "Whether to log output ports", true);
  spec.param(log_inputs_, "log_inputs", "Log Inputs", "Whether to log input ports", true);
  spec.param(log_tensor_data_content_,
             "log_tensor_data_content",
             "Log Tensor Data Content",
             "Whether to log the actual tensor data (if false, only some Tensor header info will "
             "be logged. When true the full data is also logged)",
             false);
  spec.param(log_metadata_, "log_metadata", "Log Metadata", "Whether to log metadata", true);

  // Filtering parameters
  spec.param(allowlist_patterns_,
             "allowlist_patterns",
             "Allowlist Patterns",
             "Regex patterns for unique_ids to log. If empty all messages not matching a denylist "
             "pattern are logged. Otherwise, there must be a match to one of the allowlist "
             "patterns.",
             std::vector<std::string>{});
  spec.param(denylist_patterns_,
             "denylist_patterns",
             "Denylist Patterns",
             "Regex patterns for unique_ids to log. If specified and there is a match to one of "
             "these patterns, the message is not logged.",
             std::vector<std::string>{});

  // TODO(grelee): should metadata have a separate allowlist/denylist?
}

void DataLoggerResource::initialize() {
  Resource::initialize();

  // Warn if both allowlist and denylist are specified
  bool has_allowlist = allowlist_patterns_.has_value() && !allowlist_patterns_.get().empty();
  bool has_denylist = denylist_patterns_.has_value() && !denylist_patterns_.get().empty();

  if (has_allowlist && has_denylist) {
    HOLOSCAN_LOG_WARN(
        "DataLoggerResource: Both allowlist_patterns and denylist_patterns are specified. "
        "Allowlist takes precedence and denylist will be ignored.");
  }

  // Compile regex patterns during initialization to avoid overhead during runtime
  compile_patterns();
}

int64_t DataLoggerResource::get_timestamp() const {
  auto now = std::chrono::high_resolution_clock::now();
  auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  return micros;
}

bool DataLoggerResource::should_log_message(const std::string& unique_id) const {
  // Patterns are already compiled during initialize()

  // First check denylist
  if (compiled_denylist_pattern_.has_value()) {
    try {
      if (std::regex_search(unique_id, compiled_denylist_pattern_.value())) {
        // Found match in denylist
        return false;
      }
    } catch (const std::regex_error& e) {
      HOLOSCAN_LOG_WARN("DataLoggerResource: Regex error in denylist pattern: {}", e.what());
    }
  }

  // If no allowlist was specified, allow everything
  if (!compiled_allowlist_pattern_.has_value()) {
    return true;
  }

  try {
    if (std::regex_search(unique_id, compiled_allowlist_pattern_.value())) {
      return true;  // Found match in allowlist
    }
  } catch (const std::regex_error& e) {
    HOLOSCAN_LOG_WARN("DataLoggerResource: Regex error in allowlist pattern: {}", e.what());
  }
  return false;
}

void DataLoggerResource::compile_patterns() {
  // Compile allowlist patterns
  compiled_allowlist_pattern_.reset();
  if (allowlist_patterns_.has_value()) {
    const auto& patterns = allowlist_patterns_.get();
    if (!patterns.empty()) {
      std::string combined_pattern;
      for (const auto& pattern : patterns) {
        if (!combined_pattern.empty()) {
          combined_pattern += '|';
        }
        combined_pattern += '(';
        combined_pattern += pattern;
        combined_pattern += ')';
      }

      try {
        compiled_allowlist_pattern_.emplace(combined_pattern, std::regex_constants::optimize);
      } catch (const std::regex_error& e) {
        HOLOSCAN_LOG_ERROR("DataLoggerResource: Invalid combined allowlist regex pattern '{}': {}",
                           combined_pattern,
                           e.what());
      }
    }
  }

  // Compile denylist patterns
  compiled_denylist_pattern_.reset();
  if (denylist_patterns_.has_value()) {
    const auto& patterns = denylist_patterns_.get();
    if (!patterns.empty()) {
      std::string combined_pattern;
      for (const auto& pattern : patterns) {
        if (!combined_pattern.empty()) {
          combined_pattern += '|';
        }
        combined_pattern += '(';
        combined_pattern += pattern;
        combined_pattern += ')';
      }

      try {
        compiled_denylist_pattern_.emplace(combined_pattern, std::regex_constants::optimize);
      } catch (const std::regex_error& e) {
        HOLOSCAN_LOG_ERROR("DataLoggerResource: Invalid combined denylist regex pattern '{}': {}",
                           combined_pattern,
                           e.what());
      }
    }
  }

  patterns_compiled_ = true;

  HOLOSCAN_LOG_DEBUG("DataLoggerResource: Compiled all allowlist and denylist patterns");
}

bool DataLoggerResource::log_backend_specific(const std::any& data, const std::string& unique_id,
                                              int64_t acquisition_timestamp,
                                              const std::shared_ptr<MetadataDictionary>& metadata,
                                              IOSpec::IOType io_type) {
  // Default implementation: backend-specific logging is not supported
  HOLOSCAN_LOG_DEBUG(
      "Backend-specific logging not supported for type '{}'. Please use a logger "
      "implementing log_backend_specific instead to log this type.",
      data.type().name());
  // still log metadata and timestamp
  auto result = log_data(data, unique_id, acquisition_timestamp, metadata, io_type);
  if (!result) {
    HOLOSCAN_LOG_ERROR("DataLoggerResource: Failed to log metadata for port '{}'.", unique_id);
  }
  // return false since data was not processed
  return false;
}

}  // namespace holoscan
