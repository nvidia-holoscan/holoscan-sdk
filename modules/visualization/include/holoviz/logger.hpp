/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoviz/holoviz_export.h>

#include <cstdint>
#include <format>
#include <string>
#include <tuple>
#include <utility>

#define HOLOVIZ_LOG_LEVEL_TRACE 0
#define HOLOVIZ_LOG_LEVEL_DEBUG 1
#define HOLOVIZ_LOG_LEVEL_INFO 2
#define HOLOVIZ_LOG_LEVEL_WARN 3
#define HOLOVIZ_LOG_LEVEL_ERROR 4
#define HOLOVIZ_LOG_LEVEL_CRITICAL 5
#define HOLOVIZ_LOG_LEVEL_OFF 6

#ifndef HOLOVIZ_LOG_ACTIVE_LEVEL
#define HOLOVIZ_LOG_ACTIVE_LEVEL HOLOVIZ_LOG_LEVEL_TRACE
#endif

namespace holoviz {

/// Severity for runtime filtering and compile-time \c HOLOVIZ_LOG_* stripping (\c
/// HOLOVIZ_LOG_ACTIVE_LEVEL).
enum class LogLevel : std::uint8_t {
  kTrace = 0,
  kDebug = 1,
  kInfo = 2,
  kWarn = 3,
  kError = 4,
  kCritical = 5,
  kOff = 6,
};

/// Custom log sink. Must be \c noexcept and must not call HoloViz logging APIs
/// (\c HOLOVIZ_LOG_*, \c set_log_level, etc.) — re-entry deadlocks on the internal mutex.
/// Dispatch holds a shared lock across the callback, so a handler must also not call
/// \ref set_log_handler or \ref reset_log_handler — doing so self-deadlocks.
using LogHandler = void (*)(LogLevel level, const char* file, int line, const char* function,
                            const char* message, void* user_data) noexcept;

HOLOVIZ_EXPORT void log_dispatch(const char* file, int line, const char* function, LogLevel level,
                                 const char* message) noexcept;

class Logger {
 public:
  /// Format and dispatch a log message. On format failure, logs \p format unchanged.
  template <typename... ArgsT>
  static void log(const char* file, int line, const char* function, LogLevel level,
                  const char* format, ArgsT&&... args) {
    try {
      if constexpr (sizeof...(ArgsT) == 0U) {
        log_dispatch(file, line, function, level, format);
        return;
      } else {
        auto bound_args = std::make_tuple(std::forward<ArgsT>(args)...);
        const std::string message = std::apply(
            [&](auto&... bound) { return std::vformat(format, std::make_format_args(bound...)); },
            bound_args);
        log_dispatch(file, line, function, level, message.c_str());
      }
    } catch (...) {
      log_dispatch(file, line, function, level, format);
    }
  }
};

/// Set the minimum runtime log level. \c HOLOVIZ_LOG_LEVEL overrides \p level when set.
HOLOVIZ_EXPORT void set_log_level(LogLevel level) noexcept;

/// Current runtime log level (after env override on first query or \ref set_log_level).
[[nodiscard]] HOLOVIZ_EXPORT LogLevel log_level() noexcept;

/// Install a custom log handler; \p user_data is forwarded on each call. See \ref LogHandler.
/// Blocks until any in-flight invocation of the previous handler returns; afterwards the
/// previous handler and its \c user_data are never used again and may be destroyed.
/// Must not be called from within a handler (self-deadlock).
HOLOVIZ_EXPORT void set_log_handler(LogHandler handler, void* user_data = nullptr) noexcept;

/// Restore default stderr logging. Blocks until any in-flight invocation of the current
/// handler returns; afterwards the handler and its \c user_data are never used again and
/// may be destroyed. Must not be called from within a handler (self-deadlock).
HOLOVIZ_EXPORT void reset_log_handler() noexcept;

/// Log a pre-formatted message; safe for noexcept contexts (no format parsing).
HOLOVIZ_EXPORT void log_literal(LogLevel level, const char* message) noexcept;

/// Log with source location; message must outlive the call.
HOLOVIZ_EXPORT void log_literal(LogLevel level, const char* file, int line, const char* function,
                                const char* message) noexcept;

}  // namespace holoviz

#define HOLOVIZ_LOG_CALL(level, ...) \
  ::holoviz::Logger::log(            \
      __FILE__, __LINE__, static_cast<const char*>(__FUNCTION__), level, __VA_ARGS__)

#if HOLOVIZ_LOG_ACTIVE_LEVEL <= HOLOVIZ_LOG_LEVEL_TRACE
#define HOLOVIZ_LOG_TRACE(...) HOLOVIZ_LOG_CALL(::holoviz::LogLevel::kTrace, __VA_ARGS__)
#else
#define HOLOVIZ_LOG_TRACE(...) (void)0
#endif

#if HOLOVIZ_LOG_ACTIVE_LEVEL <= HOLOVIZ_LOG_LEVEL_DEBUG
#define HOLOVIZ_LOG_DEBUG(...) HOLOVIZ_LOG_CALL(::holoviz::LogLevel::kDebug, __VA_ARGS__)
#else
#define HOLOVIZ_LOG_DEBUG(...) (void)0
#endif

#if HOLOVIZ_LOG_ACTIVE_LEVEL <= HOLOVIZ_LOG_LEVEL_INFO
#define HOLOVIZ_LOG_INFO(...) HOLOVIZ_LOG_CALL(::holoviz::LogLevel::kInfo, __VA_ARGS__)
#else
#define HOLOVIZ_LOG_INFO(...) (void)0
#endif

#if HOLOVIZ_LOG_ACTIVE_LEVEL <= HOLOVIZ_LOG_LEVEL_WARN
#define HOLOVIZ_LOG_WARN(...) HOLOVIZ_LOG_CALL(::holoviz::LogLevel::kWarn, __VA_ARGS__)
#else
#define HOLOVIZ_LOG_WARN(...) (void)0
#endif

#if HOLOVIZ_LOG_ACTIVE_LEVEL <= HOLOVIZ_LOG_LEVEL_ERROR
#define HOLOVIZ_LOG_ERROR(...) HOLOVIZ_LOG_CALL(::holoviz::LogLevel::kError, __VA_ARGS__)
#else
#define HOLOVIZ_LOG_ERROR(...) (void)0
#endif

#if HOLOVIZ_LOG_ACTIVE_LEVEL <= HOLOVIZ_LOG_LEVEL_CRITICAL
#define HOLOVIZ_LOG_CRITICAL(...) HOLOVIZ_LOG_CALL(::holoviz::LogLevel::kCritical, __VA_ARGS__)
#else
#define HOLOVIZ_LOG_CRITICAL(...) (void)0
#endif
