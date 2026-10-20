/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>

#include <cstdint>
#include <format>
#include <string>
#include <tuple>
#include <utility>

#define HOLOMQ_LOG_LEVEL_TRACE 0
#define HOLOMQ_LOG_LEVEL_DEBUG 1
#define HOLOMQ_LOG_LEVEL_INFO 2
#define HOLOMQ_LOG_LEVEL_WARN 3
#define HOLOMQ_LOG_LEVEL_ERROR 4
#define HOLOMQ_LOG_LEVEL_CRITICAL 5
#define HOLOMQ_LOG_LEVEL_OFF 6

#ifndef HOLOMQ_LOG_ACTIVE_LEVEL
#define HOLOMQ_LOG_ACTIVE_LEVEL HOLOMQ_LOG_LEVEL_TRACE
#endif

namespace holomq {

/// Severity for runtime filtering and compile-time \c HOLOMQ_LOG_* stripping (\c
/// HOLOMQ_LOG_ACTIVE_LEVEL).
enum class LogLevel : std::uint8_t {
  Trace = 0,
  Debug = 1,
  Info = 2,
  Warn = 3,
  Error = 4,
  Critical = 5,
  Off = 6,
};

/// Custom log sink. Must be \c noexcept and must not call HoloMQ logging APIs
/// (\c HOLOMQ_LOG_*, \c set_log_level, etc.) — re-entry deadlocks on the internal mutex.
using LogHandler = void (*)(LogLevel level, const char* file, int line, const char* function,
                            const char* message, void* user_data) noexcept;

HOLOMQ_EXPORT void log_dispatch(const char* file, int line, const char* function, LogLevel level,
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
      }

      auto bound_args = std::make_tuple(std::forward<ArgsT>(args)...);
      const std::string message = std::apply(
          [&](auto&... bound) { return std::vformat(format, std::make_format_args(bound...)); },
          bound_args);
      log_dispatch(file, line, function, level, message.c_str());
    } catch (...) {
      log_dispatch(file, line, function, level, format);
    }
  }
};

/// Set the minimum runtime log level. \c HOLOMQ_LOG_LEVEL overrides \p level when set.
HOLOMQ_EXPORT void set_log_level(LogLevel level) noexcept;

/// Current runtime log level (after env override on first query or \ref set_log_level).
[[nodiscard]] HOLOMQ_EXPORT LogLevel log_level() noexcept;

/// Install a custom log handler; \p user_data is forwarded on each call. See \ref LogHandler.
HOLOMQ_EXPORT void set_log_handler(LogHandler handler, void* user_data = nullptr) noexcept;

/// Restore default stderr logging.
HOLOMQ_EXPORT void reset_log_handler() noexcept;

/// Log a pre-formatted message; safe for noexcept contexts (no format parsing).
HOLOMQ_EXPORT void log_literal(LogLevel level, const char* message) noexcept;

/// Log with source location; message must outlive the call.
HOLOMQ_EXPORT void log_literal(LogLevel level, const char* file, int line, const char* function,
                               const char* message) noexcept;

/// Install the iceoryx2 logger bridge. Uses \c HOLOMQ_LOG_LEVEL when set.
HOLOMQ_EXPORT void enable_transport_logging() noexcept;

/// Install the bridge and set HoloMQ/IOX2 log level. \c HOLOMQ_LOG_LEVEL overrides \p min_level
/// when set.
HOLOMQ_EXPORT void enable_transport_logging(LogLevel min_level) noexcept;

/// Best-effort disable; iceoryx2 cannot unset its logger once installed.
HOLOMQ_EXPORT void disable_transport_logging() noexcept;

namespace detail {

HOLOMQ_EXPORT void maybe_install_transport_log_bridge() noexcept;

}  // namespace detail

}  // namespace holomq

#define HOLOMQ_LOG_CALL(level, ...) \
  ::holomq::Logger::log(            \
      __FILE__, __LINE__, static_cast<const char*>(__FUNCTION__), level, __VA_ARGS__)

#if HOLOMQ_LOG_ACTIVE_LEVEL <= HOLOMQ_LOG_LEVEL_TRACE
#define HOLOMQ_LOG_TRACE(...) HOLOMQ_LOG_CALL(::holomq::LogLevel::Trace, __VA_ARGS__)
#else
#define HOLOMQ_LOG_TRACE(...) (void)0
#endif

#if HOLOMQ_LOG_ACTIVE_LEVEL <= HOLOMQ_LOG_LEVEL_DEBUG
#define HOLOMQ_LOG_DEBUG(...) HOLOMQ_LOG_CALL(::holomq::LogLevel::Debug, __VA_ARGS__)
#else
#define HOLOMQ_LOG_DEBUG(...) (void)0
#endif

#if HOLOMQ_LOG_ACTIVE_LEVEL <= HOLOMQ_LOG_LEVEL_INFO
#define HOLOMQ_LOG_INFO(...) HOLOMQ_LOG_CALL(::holomq::LogLevel::Info, __VA_ARGS__)
#else
#define HOLOMQ_LOG_INFO(...) (void)0
#endif

#if HOLOMQ_LOG_ACTIVE_LEVEL <= HOLOMQ_LOG_LEVEL_WARN
#define HOLOMQ_LOG_WARN(...) HOLOMQ_LOG_CALL(::holomq::LogLevel::Warn, __VA_ARGS__)
#else
#define HOLOMQ_LOG_WARN(...) (void)0
#endif

#if HOLOMQ_LOG_ACTIVE_LEVEL <= HOLOMQ_LOG_LEVEL_ERROR
#define HOLOMQ_LOG_ERROR(...) HOLOMQ_LOG_CALL(::holomq::LogLevel::Error, __VA_ARGS__)
#else
#define HOLOMQ_LOG_ERROR(...) (void)0
#endif

#if HOLOMQ_LOG_ACTIVE_LEVEL <= HOLOMQ_LOG_LEVEL_CRITICAL
#define HOLOMQ_LOG_CRITICAL(...) HOLOMQ_LOG_CALL(::holomq::LogLevel::Critical, __VA_ARGS__)
#else
#define HOLOMQ_LOG_CRITICAL(...) (void)0
#endif
