/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/logger.hpp>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>

#include "transport/iox2/log_bridge.hpp"

namespace holomq {
namespace {

std::mutex g_log_mutex;
LogHandler g_log_handler = nullptr;
void* g_log_user_data = nullptr;
LogLevel g_log_level = LogLevel::Info;
bool g_log_level_initialized = false;
bool g_transport_logging_enabled = false;

const char* log_level_name(LogLevel level) noexcept {
  switch (level) {
    case LogLevel::Trace:
      return "TRACE";
    case LogLevel::Debug:
      return "DEBUG";
    case LogLevel::Info:
      return "INFO";
    case LogLevel::Warn:
      return "WARN";
    case LogLevel::Error:
      return "ERROR";
    case LogLevel::Critical:
      return "CRITICAL";
    case LogLevel::Off:
      return "OFF";
  }
  return "UNKNOWN";
}

[[nodiscard]] LogLevel parse_log_level(std::string_view value, LogLevel fallback) noexcept {
  std::string upper;
  upper.reserve(value.size());
  for (const char ch : value) {
    upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
  }

  if (upper == "TRACE") {
    return LogLevel::Trace;
  }
  if (upper == "DEBUG") {
    return LogLevel::Debug;
  }
  if (upper == "INFO") {
    return LogLevel::Info;
  }
  if (upper == "WARN") {
    return LogLevel::Warn;
  }
  if (upper == "ERROR") {
    return LogLevel::Error;
  }
  if (upper == "CRITICAL") {
    return LogLevel::Critical;
  }
  if (upper == "OFF") {
    return LogLevel::Off;
  }
  return fallback;
}

void initialize_log_level_from_env() noexcept {
  if (g_log_level_initialized) {
    return;
  }
  g_log_level_initialized = true;

  const char* env = std::getenv("HOLOMQ_LOG_LEVEL");
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

  if (level == LogLevel::Trace || level == LogLevel::Debug) {
    static_cast<void>(std::fprintf(stderr,
                                   "[holomq][%s] %s (%s:%d)\n",
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

  static_cast<void>(std::fprintf(stderr, "[holomq][%s] %s\n", log_level_name(level), message));
}

void dispatch_log(LogLevel level, const char* file, int line, const char* function,
                  const char* message) noexcept {
  if (message == nullptr) {
    return;
  }

  LogHandler handler = nullptr;
  void* user_data = nullptr;
  LogLevel active_level = LogLevel::Info;
  {
    const std::lock_guard lock(g_log_mutex);
    initialize_log_level_from_env();
    active_level = g_log_level;
    handler = g_log_handler;
    user_data = g_log_user_data;
  }

  if (static_cast<int>(level) < static_cast<int>(active_level)) {
    return;
  }

  if (handler != nullptr) {
    handler(level, file, line, function, message, user_data);
    return;
  }

  log_to_default_stderr(level, file, line, function, message);
}

void apply_log_level_locked(LogLevel level) noexcept {
  g_log_level = level;
  g_log_level_initialized = true;
#if defined(HOLOMQ_TRANSPORT_IOX2)
  if (g_transport_logging_enabled) {
    sync_iox2_log_level(level);
  }
#endif
}

}  // namespace

void log_dispatch(const char* file, int line, const char* function, LogLevel level,
                  const char* message) noexcept {
  dispatch_log(level, file, line, function, message);
}

void set_log_level(LogLevel level) noexcept {
  const std::lock_guard lock(g_log_mutex);
  const char* env = std::getenv("HOLOMQ_LOG_LEVEL");
  if (env != nullptr) {
    level = parse_log_level(env, level);
  }
  apply_log_level_locked(level);
}

LogLevel log_level() noexcept {
  const std::lock_guard lock(g_log_mutex);
  initialize_log_level_from_env();
  return g_log_level;
}

void set_log_handler(LogHandler handler, void* user_data) noexcept {
  const std::lock_guard lock(g_log_mutex);
  g_log_handler = handler;
  g_log_user_data = user_data;
}

void reset_log_handler() noexcept {
  const std::lock_guard lock(g_log_mutex);
  g_log_handler = nullptr;
  g_log_user_data = nullptr;
}

void log_literal(LogLevel level, const char* message) noexcept {
  dispatch_log(level, "holomq", 0, "log_literal", message);
}

void log_literal(LogLevel level, const char* file, int line, const char* function,
                 const char* message) noexcept {
  dispatch_log(level, file, line, function, message);
}

void enable_transport_logging() noexcept {
#if defined(HOLOMQ_TRANSPORT_IOX2)
  {
    const std::lock_guard lock(g_log_mutex);
    initialize_log_level_from_env();
    g_transport_logging_enabled = true;
    sync_iox2_log_level(g_log_level);
  }
  static_cast<void>(install_iox2_log_bridge());
#endif
}

void enable_transport_logging(LogLevel min_level) noexcept {
#if defined(HOLOMQ_TRANSPORT_IOX2)
  {
    const std::lock_guard lock(g_log_mutex);
    g_transport_logging_enabled = true;
    const char* env = std::getenv("HOLOMQ_LOG_LEVEL");
    if (env != nullptr) {
      min_level = parse_log_level(env, min_level);
    }
    apply_log_level_locked(min_level);
  }
  static_cast<void>(install_iox2_log_bridge());
#else
  static_cast<void>(min_level);
#endif
}

void disable_transport_logging() noexcept {
  const std::lock_guard lock(g_log_mutex);
  g_transport_logging_enabled = false;
}

#if defined(HOLOMQ_TRANSPORT_IOX2)
bool transport_logging_enabled() noexcept {
  const std::lock_guard lock(g_log_mutex);
  return g_transport_logging_enabled;
}
#endif

namespace detail {

void maybe_install_transport_log_bridge() noexcept {
#if defined(HOLOMQ_TRANSPORT_IOX2)
  static_cast<void>(install_iox2_log_bridge());
#endif
}

}  // namespace detail

}  // namespace holomq
