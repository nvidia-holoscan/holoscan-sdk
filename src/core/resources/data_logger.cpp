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

#include "holoscan/core/clock.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/scheduler.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

// Shared mutex for thread-safe console output coordination across all console logger types
std::mutex console_output_mutex;

void DataLoggerResource::setup(ComponentSpec& spec) {
  // logging parameters
  spec.param(log_outputs_, "log_outputs", "Log Outputs", "Whether to log output ports", true);
  spec.param(log_inputs_, "log_inputs", "Log Inputs", "Whether to log input ports", true);
  spec.param(log_metadata_, "log_metadata", "Log Metadata", "Whether to log metadata", true);
  spec.param(log_tensor_data_content_,
             "log_tensor_data_content",
             "Log Tensor Data Content",
             "Whether to log the actual tensor data (if false, only some Tensor header info will "
             "be logged. When true the full data is also logged)",
             true);
  spec.param(use_scheduler_clock_,
             "use_scheduler_clock",
             "Use Scheduler Clock",
             "Whether to use the scheduler's clock for timestamps (if false, uses a steady clock"
             "with time offset relative to epoch). If the `clock` parameter is instead specified, "
             "that explicitly provided clock is always used instead.",
             true);
  spec.param(clock_,
             "clock",
             "Clock",
             "An optional, custom clock to be used by the data logger to define the emit/receive "
             "timestamps. The clock->timestamp() value is returned directly when this is "
             "provided. When provided, this clock takes precedence and `use_scheduler_clock` is "
             "ignored.",
             ParameterFlag::kOptional);

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

  if (!use_scheduler_clock_.get()) {
    // get_timestamp will use steady_clock time added to the epoch time offset from system_clock
    time_reference_ = std::chrono::steady_clock::now();

    // use system clock to reliably get an offset relative to epoch
    auto epoch_time = std::chrono::system_clock::now().time_since_epoch();
    time_offset_ = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch_time).count();
  } else {
    if (clock_.has_value() && clock_.get() != nullptr) {
      HOLOSCAN_LOG_INFO(
          "`use_scheduler_clock` is set to true, but a custom clock was also provided. The "
          "provided `clock` will be used.");
    }
  }

  if (clock_.has_value() && clock_.get() != nullptr) {
    try {
      clock_interface_ = std::dynamic_pointer_cast<ClockInterface>(clock_.get());
      if (!clock_interface_) {
        clock_interface_ = std::dynamic_pointer_cast<holoscan::Clock>(clock_.get())->clock_impl();
      }
    } catch (const std::bad_cast& e) {
      std::string error_message = fmt::format(
          "DataLoggerResource: Failed to cast clock parameter to either "
          "std::shared_ptr<holoscan::Clock> or std::shared_ptr<holoscan::ClockInterface>: {}",
          e.what());
      HOLOSCAN_LOG_ERROR(error_message);
      throw std::runtime_error(error_message);
    }
  }
}

int64_t DataLoggerResource::get_timestamp() const {
  // First priority is the clock parameter if one was explicitly set
  if (clock_interface_) {
    return clock_interface_->timestamp();
  }

  // Fallback to the scheduler clock if requested
  if (use_scheduler_clock_.get()) {
    const auto fragment_ptr = fragment();
    if (fragment_ptr) {
      // Use const_cast since getting timestamp is logically const but scheduler() isn't const
      const auto scheduler_ptr = fragment_ptr->scheduler();
      if (scheduler_ptr) {
        const auto clock_ptr = scheduler_ptr->clock();
        if (clock_ptr) {
          return clock_ptr->timestamp();
        }
      }
    }
    HOLOSCAN_LOG_DEBUG("{}: No scheduler clock found, falling back to system clock", name());
  }

  // Fallback to steady clock with epoch offset
  auto now = std::chrono::steady_clock::now();
  auto delta_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - time_reference_).count();
  return time_offset_ + delta_ns;
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
                                              IOSpec::IOType io_type,
                                              std::optional<cudaStream_t> stream) {
  // Default implementation: backend-specific logging is not supported
  HOLOSCAN_LOG_DEBUG(
      "Backend-specific logging not supported for type '{}'. Please use a logger "
      "implementing log_backend_specific instead to log this type.",
      data.type().name());
  // still log metadata and timestamp
  auto result = log_data(data, unique_id, acquisition_timestamp, metadata, io_type, stream);
  if (!result) {
    HOLOSCAN_LOG_ERROR("DataLoggerResource: Failed to log metadata for port '{}'.", unique_id);
  }
  // return false since data was not processed
  return false;
}

}  // namespace holoscan
