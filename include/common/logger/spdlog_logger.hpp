/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMMON_LOGGER_SPDLOG_LOGGER_HPP
#define COMMON_LOGGER_SPDLOG_LOGGER_HPP

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <gxf/logger/logger.hpp>

namespace nvidia {

/// Namespace for the NVIDIA logger functionality.
namespace logger {

class SpdlogLogger : public Logger {
 public:
  /// Create a logger with the given name.
  ///
  /// This constructor creates a logger with the given name and optional logger and log function.
  /// If no logger or log function is provided, a default spdlog logger will be created.
  ///
  /// @param name The name of the logger.
  /// @param logger The logger to use (default: nullptr).
  /// @param func The log function to use (default: nullptr).
  explicit SpdlogLogger(const char* name, const std::shared_ptr<ILogger>& logger = nullptr,
                        const LogFunction& func = nullptr);

  /// Return the log pattern.
  /// @return The reference to the log pattern string.
  std::string& pattern_string();

 protected:
  std::string name_;  ///< logger name
};

}  // namespace logger

}  // namespace nvidia

#endif /* COMMON_LOGGER_SPDLOG_LOGGER_HPP */
