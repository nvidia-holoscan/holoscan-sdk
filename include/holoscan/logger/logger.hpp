/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// The holoscan logger keeps the HOLOSCAN_LOG_* macro surface and spdlog-based behavior, but does
// not depend on GXF (`gxf/logger/logger.hpp`); its backend (holoscan::logger::SpdlogLogger) is a
// standalone unit. The implementation may later be swapped for a faster backend without changing
// this interface.

#include <fmt/format.h>
#include <fmt/ranges.h>  // allows fmt to format std::array, std::vector, etc.

#include <memory>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/holoscan_logger_export.h>

#define HOLOSCAN_LOG_LEVEL_TRACE 0
#define HOLOSCAN_LOG_LEVEL_DEBUG 1
#define HOLOSCAN_LOG_LEVEL_INFO 2
#define HOLOSCAN_LOG_LEVEL_WARN 3
#define HOLOSCAN_LOG_LEVEL_ERROR 4
#define HOLOSCAN_LOG_LEVEL_CRITICAL 5
#define HOLOSCAN_LOG_LEVEL_OFF 6

// Define (or pass via CMake's `target_compile_definitions`) HOLOSCAN_LOG_ACTIVE_LEVEL to one of
// the levels above to skip logging below that level at compile time. If it is left undefined, all
// levels are compiled in.
#ifndef HOLOSCAN_LOG_ACTIVE_LEVEL
#define HOLOSCAN_LOG_ACTIVE_LEVEL HOLOSCAN_LOG_LEVEL_TRACE
#endif

// Workaround for zero-arguments
// (https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0306r2.html)
#define HOLOSCAN_LOG_CALL(level, ...)                                               \
  ::holoscan::Logger::log(std::source_location::current().file_name(),              \
                          static_cast<int>(std::source_location::current().line()), \
                          std::source_location::current().function_name(),          \
                          level,                                                    \
                          __VA_ARGS__)

// clang-format off
#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_TRACE
/**
 * @brief Print a trace message to the log (with file, line, and function name).
 *
 * The format string follows the [fmtlib format string syntax](https://fmt.dev/latest/syntax.html).
 */
#    define HOLOSCAN_LOG_TRACE(...) HOLOSCAN_LOG_CALL(::holoscan::LogLevel::TRACE, __VA_ARGS__)
#else
#    define HOLOSCAN_LOG_TRACE(...) (void)0
#endif

#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_DEBUG
/**
 * @brief Print a debug message to the log (with file, line, and function name).
 *
 * The format string follows the [fmtlib format string syntax](https://fmt.dev/latest/syntax.html).
 */
#    define HOLOSCAN_LOG_DEBUG(...) HOLOSCAN_LOG_CALL(::holoscan::LogLevel::DEBUG, __VA_ARGS__)
#else
#    define HOLOSCAN_LOG_DEBUG(...) (void)0
#endif

#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_INFO
/**
 * @brief Print an info message to the log (with file, line, and function name).
 *
 * The format string follows the [fmtlib format string syntax](https://fmt.dev/latest/syntax.html).
 */
#    define HOLOSCAN_LOG_INFO(...) HOLOSCAN_LOG_CALL(::holoscan::LogLevel::INFO, __VA_ARGS__)
#else
#    define HOLOSCAN_LOG_INFO(...) (void)0
#endif

#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_WARN
/**
 * @brief Print a warning message to the log (with file, line, and function name).
 *
 * The format string follows the [fmtlib format string syntax](https://fmt.dev/latest/syntax.html).
 */
#    define HOLOSCAN_LOG_WARN(...) HOLOSCAN_LOG_CALL(::holoscan::LogLevel::WARN, __VA_ARGS__)
#else
#    define HOLOSCAN_LOG_WARN(...) (void)0
#endif

#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_ERROR
/**
 * @brief Print an error message to the log (with file, line, and function name).
 *
 * The format string follows the [fmtlib format string syntax](https://fmt.dev/latest/syntax.html).
 */
#    define HOLOSCAN_LOG_ERROR(...) HOLOSCAN_LOG_CALL(::holoscan::LogLevel::ERROR, __VA_ARGS__)
#else
#    define HOLOSCAN_LOG_ERROR(...) (void)0
#endif

#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_CRITICAL
/**
 * @brief Print a critical message to the log (with file, line, and function name).
 *
 * The format string follows the [fmtlib format string syntax](https://fmt.dev/latest/syntax.html).
 */
#    define HOLOSCAN_LOG_CRITICAL(...) \
HOLOSCAN_LOG_CALL(::holoscan::LogLevel::CRITICAL, __VA_ARGS__)
#else
#    define HOLOSCAN_LOG_CRITICAL(...) (void)0
#endif
// clang-format on

namespace holoscan {

enum class LogLevel {
  TRACE = 0,     ///< SPDLOG_LEVEL_TRACE
  DEBUG = 1,     ///< SPDLOG_LEVEL_DEBUG
  INFO = 2,      ///< SPDLOG_LEVEL_INFO
  WARN = 3,      ///< SPDLOG_LEVEL_WARN
  ERROR = 4,     ///< SPDLOG_LEVEL_ERROR
  CRITICAL = 5,  ///< SPDLOG_LEVEL_CRITICAL
  OFF = 6,       ///< SPDLOG_LEVEL_OFF
};

/**
 * @brief A logger class that wraps spdlog.
 *
 * GXF-independent logger backed by holoscan::logger::SpdlogLogger. See the
 * [spdlog documentation](https://spdlog.docsforge.com/v1.x/api/spdlog/) for more details of the
 * underlying API.
 */
class HOLOSCAN_LOGGER_EXPORT Logger {
 public:
  static void set_level(LogLevel level, bool* is_overridden_by_env = nullptr);
  [[nodiscard]] static LogLevel level();

  static void set_pattern(std::string pattern = "", bool* is_overridden_by_env = nullptr);
  [[nodiscard]] static std::string& pattern();

  // fmt v11+ make_format_args requires lvalue refs, so this wrapper intentionally keeps args as
  // lvalues instead of forwarding them.
  // NOLINTBEGIN(cppcoreguidelines-missing-std-forward)
  template <typename FormatT, typename... ArgsT>
  static void log(const char* file, int line, const char* function_name, LogLevel level,
                  const FormatT& format, ArgsT&&... args) {
// Version number of the fmt library represented as (major * 10000 + minor * 100 + patch)
#if FMT_VERSION >= 110000
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
    log_message(file, line, function_name, level, format, fmt::make_format_args(args...));
#else
    log_message(file,
                line,
                function_name,
                level,
                format,
                fmt::make_format_args<fmt::buffer_context<fmt::char_t<FormatT> > >(args...));
#endif
  }
  // NOLINTEND(cppcoreguidelines-missing-std-forward)

  // fmt v11+ make_format_args requires lvalue refs, so this wrapper intentionally keeps args as
  // lvalues instead of forwarding them.
  // NOLINTBEGIN(cppcoreguidelines-missing-std-forward)
  template <typename FormatT, typename... ArgsT>
  static void log(LogLevel level, const FormatT& format, ArgsT&&... args) {
#if FMT_VERSION >= 110000
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
    log_message(level, format, fmt::make_format_args(args...));
#else
    log_message(
        level, format, fmt::make_format_args<fmt::buffer_context<fmt::char_t<FormatT> > >(args...));
#endif
  }
  // NOLINTEND(cppcoreguidelines-missing-std-forward)

  /**
   * @brief Flag to indicate if the log pattern was set by the user.
   */
  static bool log_pattern_set_by_user;

  /**
   * @brief Flag to indicate if the log level was set by the user.
   */
  static bool log_level_set_by_user;

 private:
  static void log_message(const char* file, int line, const char* function_name, LogLevel level,
                          fmt::string_view format, fmt::format_args args);
  static void log_message(LogLevel level, fmt::string_view format, fmt::format_args args);
};

/**
 * @brief Set global logging level.
 *
 * If the environment variable `HOLOSCAN_LOG_LEVEL` is set, the log level will be overridden by the
 * value of the environment variable (one of TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, OFF).
 *
 * @param level The new log level.
 */
HOLOSCAN_LOGGER_EXPORT void set_log_level(LogLevel level);

/**
 * @brief Get global logging level.
 *
 * @return The current log level.
 */
[[nodiscard]] inline LogLevel log_level() {
  return Logger::level();
}

/**
 * @brief Set global log format string.
 *
 * If the environment variable `HOLOSCAN_LOG_FORMAT` is set, the log pattern will be overridden by
 * the value of the environment variable. `HOLOSCAN_LOG_FORMAT` accepts the presets SHORT, DEFAULT,
 * LONG, FULL, or a custom spdlog pattern string.
 *
 * @param pattern The format string.
 */
HOLOSCAN_LOGGER_EXPORT void set_log_pattern(std::string pattern = "");

/**
 * @brief Print a trace message to the log.
 *
 * The format string follows the [fmtlib format string syntax](https://fmt.dev/latest/syntax.html).
 */
template <typename FormatT, typename... ArgsT>
inline void log_trace(const FormatT& format, ArgsT&&... args) {
  Logger::log(LogLevel::TRACE, format, std::forward<ArgsT>(args)...);
}

/**
 * @brief Print a debug message to the log.
 *
 * The format string follows the [fmtlib format string syntax](https://fmt.dev/latest/syntax.html).
 */
template <typename FormatT, typename... ArgsT>
inline void log_debug(const FormatT& format, ArgsT&&... args) {
  Logger::log(LogLevel::DEBUG, format, std::forward<ArgsT>(args)...);
}

/**
 * @brief Print an info message to the log.
 *
 * The format string follows the [fmtlib format string syntax](https://fmt.dev/latest/syntax.html).
 */
template <typename FormatT, typename... ArgsT>
inline void log_info(const FormatT& format, ArgsT&&... args) {
  Logger::log(LogLevel::INFO, format, std::forward<ArgsT>(args)...);
}

/**
 * @brief Print a warning message to the log.
 *
 * The format string follows the [fmtlib format string syntax](https://fmt.dev/latest/syntax.html).
 */
template <typename FormatT, typename... ArgsT>
inline void log_warn(const FormatT& format, ArgsT&&... args) {
  Logger::log(LogLevel::WARN, format, std::forward<ArgsT>(args)...);
}

/**
 * @brief Print an error message to the log.
 *
 * The format string follows the [fmtlib format string syntax](https://fmt.dev/latest/syntax.html).
 */
template <typename FormatT, typename... ArgsT>
inline void log_error(const FormatT& format, ArgsT&&... args) {
  Logger::log(LogLevel::ERROR, format, std::forward<ArgsT>(args)...);
}

/**
 * @brief Print a critical message to the log.
 *
 * The format string follows the [fmtlib format string syntax](https://fmt.dev/latest/syntax.html).
 */
template <typename FormatT, typename... ArgsT>
inline void log_critical(const FormatT& format, ArgsT&&... args) {
  Logger::log(LogLevel::CRITICAL, format, std::forward<ArgsT>(args)...);
}

/**
 * @brief Print a message to the log.
 *
 * The format string follows the [fmtlib format string syntax](https://fmt.dev/latest/syntax.html).
 */
template <typename FormatT, typename... ArgsT>
inline void log_message(const char* file, int line, const char* function_name, LogLevel level,
                        const FormatT& format, ArgsT&&... args) {
  Logger::log(file, line, function_name, level, format, std::forward<ArgsT>(args)...);
}

}  // namespace holoscan
