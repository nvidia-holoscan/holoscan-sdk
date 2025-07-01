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
#include <regex>
#include <string>
#include <vector>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

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
             "Regex patterns for unique_ids to always log (if any specified, only matching "
             "messages are logged)",
             std::vector<std::string>{});
  spec.param(denylist_patterns_,
             "denylist_patterns",
             "Denylist Patterns",
             "Regex patterns for unique_ids to never log (ignored if allowlist is specified)",
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

  // If allowlist patterns are specified, unique_id must match at least one
  if (!compiled_allowlist_patterns_.empty()) {
    for (const auto& pattern : compiled_allowlist_patterns_) {
      try {
        if (std::regex_search(unique_id, pattern)) {
          return true;  // Found match in allowlist
        }
      } catch (const std::regex_error& e) {
        HOLOSCAN_LOG_WARN("DataLoggerResource: Regex error in allowlist pattern: {}", e.what());
      }
    }
    return false;
  }

  // No allowlist specified, check denylist
  for (const auto& pattern : compiled_denylist_patterns_) {
    try {
      if (std::regex_search(unique_id, pattern)) {
        // Found match in denylist
        return false;
      }
    } catch (const std::regex_error& e) {
      HOLOSCAN_LOG_WARN("DataLoggerResource: Regex error in denylist pattern: {}", e.what());
    }
  }

  return true;  // No denylist match, should log
}

void DataLoggerResource::compile_patterns() {
  // Compile allowlist patterns
  compiled_allowlist_patterns_.clear();
  if (allowlist_patterns_.has_value()) {
    for (const auto& pattern : allowlist_patterns_.get()) {
      try {
        compiled_allowlist_patterns_.emplace_back(pattern, std::regex_constants::optimize);
      } catch (const std::regex_error& e) {
        HOLOSCAN_LOG_ERROR(
            "DataLoggerResource: Invalid allowlist regex pattern '{}': {}", pattern, e.what());
      }
    }
  }

  // Compile denylist patterns
  compiled_denylist_patterns_.clear();
  if (denylist_patterns_.has_value()) {
    for (const auto& pattern : denylist_patterns_.get()) {
      try {
        compiled_denylist_patterns_.emplace_back(pattern, std::regex_constants::optimize);
      } catch (const std::regex_error& e) {
        HOLOSCAN_LOG_ERROR(
            "DataLoggerResource: Invalid denylist regex pattern '{}': {}", pattern, e.what());
      }
    }
  }

  patterns_compiled_ = true;

  HOLOSCAN_LOG_DEBUG("DataLoggerResource: Compiled {} allowlist patterns and {} denylist patterns",
                     compiled_allowlist_patterns_.size(),
                     compiled_denylist_patterns_.size());
}

}  // namespace holoscan
