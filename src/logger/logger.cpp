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

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>

namespace holoscan {

bool Logger::log_pattern_set_by_user = false;
bool Logger::log_level_set_by_user = false;

static std::string get_concrete_log_pattern(std::string pattern) {
  // Convert to uppercase
  std::string log_pattern = pattern;
  std::transform(log_pattern.begin(), log_pattern.end(), log_pattern.begin(), [](unsigned char c) {
    return std::toupper(c);
  });

  if (log_pattern == "SHORT") {
    return "[%^%l%$] %v";
  } else if (log_pattern == "DEFAULT") {
    return "[%^%l%$] [%s:%#] %v";
  } else if (log_pattern == "LONG") {
    return "[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] [%s:%#] %v";
  } else if (log_pattern == "FULL") {
    return "[%Y-%m-%d %H:%M:%S.%e] [%t] [%n] [%^%l%$] [%s:%#] %v";
  } else {
    // Return the original pattern
    return pattern;
  }
}

std::string& Logger::pattern() {
  static std::string log_pattern = "[%^%l%$] [%s:%#] %v";
  return log_pattern;
}

static std::shared_ptr<spdlog::logger>& get_logger(const std::string& name = "holoscan") {
  static auto logger = [&name] {
    auto tmp_logger = spdlog::stderr_color_mt(name);
    // Set default log level and pattern
    tmp_logger->set_level(spdlog::level::info);
    tmp_logger->set_pattern(Logger::pattern());
    return tmp_logger;
  }();

  return logger;
}

void set_log_level(LogLevel level) {
  static const char* level_str[7] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL", "OFF"};
  bool is_overridden_by_env = false;

  Logger::set_level(level, &is_overridden_by_env);

  if (is_overridden_by_env) {
    LogLevel new_log_level = Logger::level();
    HOLOSCAN_LOG_DEBUG(
        "Log level would be overridden by HOLOSCAN_LOG_LEVEL environment variable to '{}'",
        level_str[static_cast<int>(new_log_level)]);
  }
}

void set_log_pattern(std::string pattern) {
  bool is_overridden_by_env = false;

  // https://spdlog.docsforge.com/v1.x/0.faq/#colors-do-not-appear-when-using-custom-format
  Logger::set_pattern(pattern, &is_overridden_by_env);

  if (is_overridden_by_env) {
    const char* env_p = std::getenv("HOLOSCAN_LOG_FORMAT");
    if (env_p) {
      HOLOSCAN_LOG_DEBUG(
          "Log format would be overridden by HOLOSCAN_LOG_FORMAT environment variable to '{}'",
          env_p);
    }
  }
}

void Logger::set_level(LogLevel level, bool* is_overridden_by_env) {
  const char* env_p = std::getenv("HOLOSCAN_LOG_LEVEL");
  // Override the environment variable if the user has set the log level
  if (env_p) {
    LogLevel env_level = level;
    std::string log_level(env_p);
    std::transform(log_level.begin(), log_level.end(), log_level.begin(), [](unsigned char c) {
      return std::toupper(c);
    });
    if (log_level == "TRACE") {
      env_level = LogLevel::TRACE;
    } else if (log_level == "DEBUG") {
      env_level = LogLevel::DEBUG;
    } else if (log_level == "INFO") {
      env_level = LogLevel::INFO;
    } else if (log_level == "WARN") {
      env_level = LogLevel::WARN;
    } else if (log_level == "ERROR") {
      env_level = LogLevel::ERROR;
    } else if (log_level == "CRITICAL") {
      env_level = LogLevel::CRITICAL;
    } else if (log_level == "OFF") {
      env_level = LogLevel::OFF;
    }

    if (is_overridden_by_env) { *is_overridden_by_env = true; }
    level = env_level;
  }

  get_logger()->set_level(static_cast<spdlog::level::level_enum>(level));

  Logger::log_level_set_by_user = true;
}

LogLevel Logger::level() {
  return static_cast<LogLevel>(get_logger()->level());
}

void Logger::set_pattern(std::string pattern, bool* is_overridden_by_env) {
  // Consider the pattern set by the user if it is not empty
  if (!pattern.empty()) { Logger::log_pattern_set_by_user = true; }

  // Get the concrete log pattern
  pattern = get_concrete_log_pattern(pattern);

  const char* env_p = std::getenv("HOLOSCAN_LOG_FORMAT");
  // Override the environment variable if the user has set the log level
  if (env_p) {
    std::string env_pattern;
    std::string log_pattern = env_p;
    env_pattern = get_concrete_log_pattern(log_pattern);

    if (is_overridden_by_env) { *is_overridden_by_env = true; }

    pattern = env_pattern;
  }

  if (pattern.empty()) {
    // The following code sets the logger's format pattern, it takes effect only if the pattern
    // hasn't already been set by the user from directly calling Logger::set_pattern, or via the
    // HOLOSCAN_LOG_FORMAT env variable
    if (!Logger::log_pattern_set_by_user) {
      switch (Logger::level()) {
        case LogLevel::OFF:
        case LogLevel::CRITICAL:
        case LogLevel::ERROR:
        case LogLevel::WARN:
        case LogLevel::INFO:
          // [%s:%#] for showing filename:line_number
          pattern = "[%^%l%$] [%s:%#] %v";
          break;
        case LogLevel::DEBUG:
        case LogLevel::TRACE:
          // Display info for [time] [thread] [tool] [level] [filename:line_number] message
          pattern = "[%Y-%m-%d %H:%M:%S.%e][%t][%n][%^%l%$][%s:%#] %v";
      }
    }
  }

  if (!pattern.empty()) {
    Logger::pattern() = pattern;
    get_logger()->set_pattern(pattern);
  }
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
