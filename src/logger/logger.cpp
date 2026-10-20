/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// NOLINTBEGIN(concurrency-mt-unsafe,cppcoreguidelines-pro-bounds-constant-array-index)
// Logger configuration is done from main thread at startup.
// Array access by LogLevel enum is safe since enum values are bounded.

#include <holoscan/logger/logger.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <string>
#include <utility>

#include <holoscan/logger/spdlog_logger.hpp>

namespace holoscan {

constinit bool Logger::log_pattern_set_by_user = false;
constinit bool Logger::log_level_set_by_user = false;

namespace {

// Process-wide spdlog-backed logger instance. The spdlog backend disables automatic registry
// registration, so it does not clash with any other "holoscan" logger by name.
logger::SpdlogLogger& instance() {
  static logger::SpdlogLogger logger("holoscan");
  return logger;
}

std::string get_concrete_log_pattern(std::string pattern) {
  // Convert to uppercase
  std::string log_pattern = pattern;
  std::transform(
      log_pattern.begin(), log_pattern.end(), log_pattern.begin(), [](unsigned char character) {
        return std::toupper(character);
      });

  if (log_pattern == "SHORT") {
    return "[%^%l%$] %v";
  }
  if (log_pattern == "DEFAULT") {
    return "[%^%l%$] [%s:%#] %v";
  }
  if (log_pattern == "LONG") {
    return "[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] [%s:%#] %v";
  }
  if (log_pattern == "FULL") {
    return "[%Y-%m-%d %H:%M:%S.%e] [%t] [%n] [%^%l%$] [%s:%#] %v";
  }

  // Return the original pattern
  return pattern;
}

}  // namespace

std::string& Logger::pattern() {
  return instance().pattern_string();
}

void set_log_level(LogLevel level) {
  static constexpr std::array<const char*, 7> kLevelStrings = {
      "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL", "OFF"};
  bool is_overridden_by_env = false;

  Logger::set_level(level, &is_overridden_by_env);

  if (is_overridden_by_env) {
    const LogLevel new_log_level = Logger::level();
    HOLOSCAN_LOG_DEBUG(
        "Log level would be overridden by HOLOSCAN_LOG_LEVEL environment variable to '{}'",
        kLevelStrings[static_cast<int>(new_log_level)]);
  }
}

void set_log_pattern(std::string pattern) {
  bool is_overridden_by_env = false;

  Logger::set_pattern(std::move(pattern), &is_overridden_by_env);

  if (is_overridden_by_env) {
    const char* env_p = std::getenv("HOLOSCAN_LOG_FORMAT");
    if (env_p != nullptr) {
      HOLOSCAN_LOG_DEBUG(
          "Log format would be overridden by HOLOSCAN_LOG_FORMAT environment variable to '{}'",
          env_p);
    }
  }
}

void Logger::set_level(LogLevel level, bool* is_overridden_by_env) {
  const char* env_p = std::getenv("HOLOSCAN_LOG_LEVEL");
  // Override the requested level if the environment variable is set
  if (env_p != nullptr) {
    LogLevel env_level = level;
    std::string log_level(env_p);
    std::transform(
        log_level.begin(), log_level.end(), log_level.begin(), [](unsigned char character) {
          return std::toupper(character);
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

    if (is_overridden_by_env != nullptr) {
      *is_overridden_by_env = true;
    }
    level = env_level;
  }

  instance().level(static_cast<int>(level));

  Logger::log_level_set_by_user = true;
}

LogLevel Logger::level() {
  return static_cast<LogLevel>(instance().level());
}

void Logger::set_pattern(std::string pattern, bool* is_overridden_by_env) {
  // Consider the pattern set by the user if it is not empty
  if (!pattern.empty()) {
    Logger::log_pattern_set_by_user = true;
  }

  // Get the concrete log pattern
  pattern = get_concrete_log_pattern(pattern);

  const char* env_p = std::getenv("HOLOSCAN_LOG_FORMAT");
  // Override with the environment variable if it is set
  if (env_p != nullptr) {
    std::string log_pattern = env_p;
    std::string env_pattern = get_concrete_log_pattern(std::move(log_pattern));

    if (is_overridden_by_env != nullptr) {
      *is_overridden_by_env = true;
    }

    pattern = std::move(env_pattern);
  }

  if (pattern.empty()) {
    // The following sets the logger's format pattern; it takes effect only if the pattern hasn't
    // already been set by the user directly via Logger::set_pattern, or via HOLOSCAN_LOG_FORMAT.
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
          pattern = "[%Y-%m-%d %H:%M:%S.%e] [%t] [%n] [%^%l%$] [%s:%#] %v";
      }
    }
  }

  if (!pattern.empty()) {
    Logger::pattern() = pattern;
    instance().pattern(pattern.c_str());
  }
}

void Logger::log_message(const char* file, int line, const char* name, LogLevel level,
                         fmt::string_view format, fmt::format_args args) {
  instance().log(file, line, name, static_cast<int>(level), fmt::vformat(format, args).c_str());
}

void Logger::log_message(LogLevel level, fmt::string_view format, fmt::format_args args) {
  instance().log(nullptr, 0, nullptr, static_cast<int>(level), fmt::vformat(format, args).c_str());
}

}  // namespace holoscan

// NOLINTEND(concurrency-mt-unsafe,cppcoreguidelines-pro-bounds-constant-array-index)
