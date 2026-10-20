/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoviz/logger.hpp>

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <string_view>

namespace holoviz {
namespace {

std::mutex g_log_mutex;
LogLevel g_log_level = LogLevel::kInfo;
bool g_log_level_initialized = false;

// Guards the handler pair independently of the level state. dispatch_log() holds it shared
// across the handler invocation so that set_log_handler()/reset_log_handler() (exclusive)
// cannot return while a previous handler or its user_data is still in use.
std::shared_mutex g_handler_mutex;
LogHandler g_log_handler = nullptr;
void* g_log_user_data = nullptr;

const char* log_level_name(LogLevel level) noexcept {
  switch (level) {
    case LogLevel::kTrace:
      return "TRACE";
    case LogLevel::kDebug:
      return "DEBUG";
    case LogLevel::kInfo:
      return "INFO";
    case LogLevel::kWarn:
      return "WARN";
    case LogLevel::kError:
      return "ERROR";
    case LogLevel::kCritical:
      return "CRITICAL";
    case LogLevel::kOff:
      return "OFF";
  }
  return "UNKNOWN";
}

// Case-insensitive ASCII comparison without allocation; parse_log_level() runs on noexcept
// paths where a std::string temporary would terminate on allocation failure.
[[nodiscard]] bool ascii_iequals(std::string_view a, std::string_view b) noexcept {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::toupper(static_cast<unsigned char>(a[i])) !=
        std::toupper(static_cast<unsigned char>(b[i]))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] LogLevel parse_log_level(std::string_view value, LogLevel fallback) noexcept {
  struct Token {
    std::string_view name;
    LogLevel level;
  };
  constexpr std::array<Token, 7> kTokens = {{
      {"TRACE", LogLevel::kTrace},
      {"DEBUG", LogLevel::kDebug},
      {"INFO", LogLevel::kInfo},
      {"WARN", LogLevel::kWarn},
      {"ERROR", LogLevel::kError},
      {"CRITICAL", LogLevel::kCritical},
      {"OFF", LogLevel::kOff},
  }};
  for (const auto& token : kTokens) {
    if (ascii_iequals(value, token.name)) {
      return token.level;
    }
  }
  return fallback;
}

void initialize_log_level_from_env() noexcept {
  if (g_log_level_initialized) {
    return;
  }
  g_log_level_initialized = true;

  const char* env = std::getenv("HOLOVIZ_LOG_LEVEL");
  if (env == nullptr) {
    return;
  }
  g_log_level = parse_log_level(env, g_log_level);
}

void log_to_default_stderr(LogLevel level, const char* file, int line, const char* function,
                           const char* message) noexcept {
  if (message == nullptr) {
    return;
  }

  if (level == LogLevel::kTrace || level == LogLevel::kDebug) {
    static_cast<void>(std::fprintf(stderr,
                                   "[holoviz][%s] %s (%s:%d)\n",
                                   log_level_name(level),
                                   message,
                                   function != nullptr ? function : "",
                                   line));
    static_cast<void>(file);
    return;
  }

  static_cast<void>(file);
  static_cast<void>(line);
  static_cast<void>(function);

  static_cast<void>(std::fprintf(stderr, "[holoviz][%s] %s\n", log_level_name(level), message));
}

void dispatch_log(LogLevel level, const char* file, int line, const char* function,
                  const char* message) noexcept {
  if (message == nullptr) {
    return;
  }

  LogLevel active_level = LogLevel::kInfo;
  {
    const std::lock_guard lock(g_log_mutex);
    initialize_log_level_from_env();
    active_level = g_log_level;
  }

  if (static_cast<int>(level) < static_cast<int>(active_level)) {
    return;
  }

  {
    // Invoke the handler while holding the shared lock so reset/replacement blocks until
    // every in-flight callback has returned (see set_log_handler()/reset_log_handler()).
    const std::shared_lock lock(g_handler_mutex);
    if (g_log_handler != nullptr) {
      g_log_handler(level, file, line, function, message, g_log_user_data);
      return;
    }
  }

  log_to_default_stderr(level, file, line, function, message);
}

}  // namespace

void log_dispatch(const char* file, int line, const char* function, LogLevel level,
                  const char* message) noexcept {
  dispatch_log(level, file, line, function, message);
}

void set_log_level(LogLevel level) noexcept {
  const std::lock_guard lock(g_log_mutex);
  const char* env = std::getenv("HOLOVIZ_LOG_LEVEL");
  if (env != nullptr) {
    level = parse_log_level(env, level);
  }
  g_log_level = level;
  g_log_level_initialized = true;
}

LogLevel log_level() noexcept {
  const std::lock_guard lock(g_log_mutex);
  initialize_log_level_from_env();
  return g_log_level;
}

void set_log_handler(LogHandler handler, void* user_data) noexcept {
  const std::lock_guard lock(g_handler_mutex);
  g_log_handler = handler;
  g_log_user_data = user_data;
}

void reset_log_handler() noexcept {
  const std::lock_guard lock(g_handler_mutex);
  g_log_handler = nullptr;
  g_log_user_data = nullptr;
}

void log_literal(LogLevel level, const char* message) noexcept {
  dispatch_log(level, "holoviz", 0, "log_literal", message);
}

void log_literal(LogLevel level, const char* file, int line, const char* function,
                 const char* message) noexcept {
  dispatch_log(level, file, line, function, message);
}

}  // namespace holoviz
