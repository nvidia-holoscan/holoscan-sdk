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

#include "holoscan/core/executors/gxf/gxf_logger.hpp"

#include <cstdio>
#include <cstdlib>
#include <string_view>

#include <common/logger.hpp>

#include "holoscan/logger/logger.hpp"

namespace holoscan::gxf {

static nvidia::Severity s_gxf_log_level = nvidia::Severity::INFO;

static void ensure_gxf_log_level(int level) {
  if (level < static_cast<int>(nvidia::Severity::NONE) ||
      level > static_cast<int>(nvidia::Severity::COUNT)) {
    std::fprintf(stderr, "GXFLogger: Invalid log level %d.", level);
    std::abort();
  }
}

void GXFLogger::log(const char* file, int line, const char* /* name */, int level, const char* log,
                    void* /* arg */) {
  if (level == static_cast<int>(nvidia::Severity::ALL) ||
      level == static_cast<int>(nvidia::Severity::COUNT)) {
    HOLOSCAN_LOG_ERROR("Invalid severity level ({}): Log severity cannot be 'ALL' or 'COUNT'.",
                       static_cast<int>(level));
  }

  // Ignore severity if requested
  if (s_gxf_log_level == nvidia::Severity::NONE || level > static_cast<int>(s_gxf_log_level)) {
    return;
  }

  LogLevel holoscan_log_level = LogLevel::INFO;

  switch (level) {
    case static_cast<int>(nvidia::Severity::VERBOSE):
      holoscan_log_level = LogLevel::TRACE;
      break;
    case static_cast<int>(nvidia::Severity::DEBUG):
      holoscan_log_level = LogLevel::DEBUG;
      break;
    case static_cast<int>(nvidia::Severity::INFO):
      holoscan_log_level = LogLevel::INFO;
      break;
    case static_cast<int>(nvidia::Severity::WARNING):
      holoscan_log_level = LogLevel::WARN;
      break;
    case static_cast<int>(nvidia::Severity::ERROR):
      holoscan_log_level = LogLevel::ERROR;
      break;
    case static_cast<int>(nvidia::Severity::PANIC):
      holoscan_log_level = LogLevel::CRITICAL;
      break;
    default:
      holoscan_log_level = LogLevel::INFO;
  }

  std::string_view file_str(file);
  std::string_view file_base = file_str.substr(file_str.find_last_of("/") + 1);

  holoscan::log_message(file_base.data(), line, "gxf", holoscan_log_level, log);
}

void GXFLogger::pattern(const char* /* pattern */) {
  // Do nothing, as the default logger does not support custom patterns
}

const char* GXFLogger::pattern() const {
  // Always return the empty string, as the default logger does not support custom patterns
  return "";
}

void GXFLogger::level(int level) {
  ensure_gxf_log_level(level);
  nvidia::Severity severity = static_cast<nvidia::Severity>(level);

  if (severity == nvidia::Severity::COUNT) {
    std::fprintf(stderr, "GXFLogger: Log severity cannot be 'COUNT'.\n");
    std::abort();
  }
  s_gxf_log_level = severity;
}

int GXFLogger::level() const {
  return static_cast<int>(s_gxf_log_level);
}

void GXFLogger::redirect(int /* level */, void* /* output */) {
  // Do nothing for GXF redirect
}

void* GXFLogger::redirect(int /* level */) const {
  // Do nothing. Always return stderr;
  return stderr;
}

void GXFLogger::set_gxf_log_level(int level) {
  ensure_gxf_log_level(level);
  s_gxf_log_level = static_cast<nvidia::Severity>(level);
}

int GXFLogger::gxf_log_level() {
  return static_cast<int>(s_gxf_log_level);
}

}  // namespace holoscan::gxf
