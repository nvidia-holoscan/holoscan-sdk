/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Backend for the holoscan logger. This consolidates what was previously the GXF
// `nvidia::logger::{ILogger, Logger}` interface and the `nvidia::logger::SpdlogLogger` backend into
// a single GXF-independent unit under the `holoscan::logger` namespace.

#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan_logger_export.h>

namespace holoscan::logger {

using LogFunction = std::function<void(const char* file, int line, const char* function_name,
                                       int level, const char* message, void* arg)>;

/// Interface for custom logger implementations.
///
/// ILogger provides an interface for implementing custom logging mechanisms, allowing the
/// integration of various logging systems with the Logger class.
class ILogger {
 public:
  virtual ~ILogger() = default;

  /// Log a message.
  ///
  /// @param file The name of the source file where the log request originated.
  /// @param line The line number in the source file.
  /// @param name The name of the function/module making the log request.
  /// @param level The log level.
  /// @param message The log message.
  /// @param arg An optional argument for additional log information.
  virtual void log(const char* file, int line, const char* name, int level, const char* message,
                   void* arg) = 0;

  /// Set the log message pattern.
  virtual void pattern(const char* pattern) = 0;
  /// Get the current log message pattern.
  [[nodiscard]] virtual const char* pattern() const = 0;

  /// Set the log level.
  virtual void level(int level) = 0;
  /// Get the current log level.
  [[nodiscard]] virtual int level() const = 0;

  /// Redirect log output to the specified file/output for the given log level.
  virtual void redirect(int level, void* output) = 0;
  /// Get the current redirection for the specified log level (nullptr if none).
  [[nodiscard]] virtual void* redirect(int level) const = 0;
};

/// Logger class for logging messages.
///
/// Provides a flexible, customizable logging system that can dispatch to a custom `ILogger` or a
/// custom `LogFunction`. The custom log function takes priority over the custom logger; if neither
/// is set, no logging occurs.
class HOLOSCAN_LOGGER_EXPORT Logger {
 public:
  Logger() = default;

  /// Construct a logger instance with the custom logger.
  HOLOSCAN_LOGGER_EXPORT explicit Logger(std::shared_ptr<ILogger> logger);
  /// Construct a logger instance with the custom log function.
  HOLOSCAN_LOGGER_EXPORT explicit Logger(LogFunction func);
  /// Construct a logger instance with the custom logger or log function.
  HOLOSCAN_LOGGER_EXPORT Logger(std::shared_ptr<ILogger> logger, LogFunction func);

  /// Log a message using either the custom log function (preferred) or the custom logger.
  HOLOSCAN_LOGGER_EXPORT void log(const char* file, int line, const char* name, int level,
                                  const char* message);

  /// Set / get a custom logger.
  HOLOSCAN_LOGGER_EXPORT void logger(std::shared_ptr<ILogger> custom_logger);
  [[nodiscard]] HOLOSCAN_LOGGER_EXPORT std::shared_ptr<ILogger> logger() const;

  /// Set a custom log function and an opaque argument passed back to it.
  HOLOSCAN_LOGGER_EXPORT void func(LogFunction log_func, void* func_arg = nullptr);
  /// Get the current custom log function.
  [[nodiscard]] HOLOSCAN_LOGGER_EXPORT LogFunction func() const;
  /// Get the current custom log function argument.
  [[nodiscard]] HOLOSCAN_LOGGER_EXPORT void* arg() const;

  /// Set / get the log message pattern.
  HOLOSCAN_LOGGER_EXPORT void pattern(const char* pattern);
  [[nodiscard]] HOLOSCAN_LOGGER_EXPORT const char* pattern() const;

  /// Set / get the log level.
  HOLOSCAN_LOGGER_EXPORT void level(int level);
  [[nodiscard]] HOLOSCAN_LOGGER_EXPORT int level() const;

  /// Redirect log output for a given level, and query the current redirection.
  HOLOSCAN_LOGGER_EXPORT void redirect(int level, void* output);
  [[nodiscard]] HOLOSCAN_LOGGER_EXPORT void* redirect(int level) const;

 protected:
  std::shared_ptr<ILogger> logger_;  ///< shared pointer to the logger
  LogFunction func_;                 ///< log function
  void* func_arg_ = nullptr;         ///< log function argument

  std::string pattern_;       ///< log pattern
  int level_ = 0;             ///< log level
  std::vector<void*> sinks_;  ///< log sinks
};

/// Logger backed by spdlog.
///
/// Creates a default spdlog-based `ILogger` when no custom logger/function is supplied.
class HOLOSCAN_LOGGER_EXPORT SpdlogLogger : public Logger {
 public:
  /// Create a logger with the given name and optional custom logger/log function. If neither is
  /// provided, a default spdlog logger is created.
  HOLOSCAN_LOGGER_EXPORT explicit SpdlogLogger(const char* name,
                                               std::shared_ptr<ILogger> logger = nullptr,
                                               LogFunction func = nullptr);

  /// Return a reference to the log pattern string.
  [[nodiscard]] HOLOSCAN_LOGGER_EXPORT std::string& pattern_string();

 protected:
  std::string name_;  ///< logger name
};

}  // namespace holoscan::logger
