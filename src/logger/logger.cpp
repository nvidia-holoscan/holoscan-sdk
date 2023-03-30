/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/logger/logger.hpp"

#include <spdlog/cfg/env.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdlib>

namespace holoscan {

static std::shared_ptr<spdlog::logger>& get_logger(const std::string& name = "holoscan") {
  static auto logger = spdlog::stderr_color_mt(name);
  return logger;
}

void Logger::load_env_level() {
  const char* env_p = std::getenv("HOLOSCAN_LOG_LEVEL");

  if (env_p) {
    auto log_level_str = std::string_view(env_p);
    if (log_level_str == "TRACE") {
      set_log_level(LogLevel::TRACE);
    } else if (log_level_str == "DEBUG") {
      set_log_level(LogLevel::DEBUG);
    } else if (log_level_str == "INFO") {
      set_log_level(LogLevel::INFO);
    } else if (log_level_str == "WARN") {
      set_log_level(LogLevel::WARN);
    } else if (log_level_str == "ERROR") {
      set_log_level(LogLevel::ERROR);
    } else if (log_level_str == "CRITICAL") {
      set_log_level(LogLevel::CRITICAL);
    } else if (log_level_str == "OFF") {
      set_log_level(LogLevel::OFF);
    }
  }
}

void Logger::set_level(LogLevel level) {
  get_logger()->set_level(static_cast<spdlog::level::level_enum>(level));
}

LogLevel Logger::level() {
  return static_cast<LogLevel>(get_logger()->level());
}

void Logger::set_pattern(std::string pattern) {
  get_logger()->set_pattern(pattern);
}

bool Logger::should_backtrace() {
  return get_logger()->should_backtrace();
}

void Logger::disable_backtrace() {
  return get_logger()->disable_backtrace();
}

void Logger::enable_backtrace(size_t n_messages) {
  return get_logger()->enable_backtrace(n_messages);
}

void Logger::dump_backtrace() {
  return get_logger()->dump_backtrace();
}

void Logger::flush() {
  return get_logger()->flush();
}

LogLevel Logger::flush_level() {
  return static_cast<LogLevel>(get_logger()->flush_level());
}

void Logger::flush_on(LogLevel level) {
  get_logger()->flush_on(static_cast<spdlog::level::level_enum>(level));
}

void Logger::log_message(const char* file, int line, const char* function_name, LogLevel level,
                         fmt::string_view format, fmt::format_args args) {
  get_logger()->log(spdlog::source_loc{file, line, function_name},
                    static_cast<spdlog::level::level_enum>(level),
                    fmt::vformat(format, args));
}

void Logger::log_message(LogLevel level, fmt::string_view format, fmt::format_args args) {
  get_logger()->log(static_cast<spdlog::level::level_enum>(level), fmt::vformat(format, args));
}

}  // namespace holoscan
