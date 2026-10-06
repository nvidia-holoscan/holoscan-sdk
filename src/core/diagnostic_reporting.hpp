/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <source_location>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan::detail {

// Keep argument evaluation, formatting, and any throwing sink inside the guarded cold path.
template <typename Report>
void report_automatic_diagnostic(LogLevel severity, const Report& report) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    if (automatic_diagnostic_logging_enabled() && log_level() <= severity) {
      report();
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    // Presentation cannot replace the initiating failure or interrupt cleanup.
    return;
  }
#endif
}

}  // namespace holoscan::detail

#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_ERROR
#define HOLOSCAN_LOG_AUTOMATIC_ERROR(...)                                         \
  ::holoscan::detail::report_automatic_diagnostic(                                \
      ::holoscan::LogLevel::ERROR,                                                \
      [&, diagnostic_log_location = std::source_location::current()] {            \
        ::holoscan::Logger::log(diagnostic_log_location.file_name(),              \
                                static_cast<int>(diagnostic_log_location.line()), \
                                diagnostic_log_location.function_name(),          \
                                ::holoscan::LogLevel::ERROR,                      \
                                __VA_ARGS__);                                     \
      })
#else
#define HOLOSCAN_LOG_AUTOMATIC_ERROR(...) (void)0
#endif

#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_WARN
#define HOLOSCAN_LOG_AUTOMATIC_WARN(...)                                                           \
  ::holoscan::detail::report_automatic_diagnostic(                                                 \
      ::holoscan::LogLevel::WARN, [&, diagnostic_log_location = std::source_location::current()] { \
        ::holoscan::Logger::log(diagnostic_log_location.file_name(),                               \
                                static_cast<int>(diagnostic_log_location.line()),                  \
                                diagnostic_log_location.function_name(),                           \
                                ::holoscan::LogLevel::WARN,                                        \
                                __VA_ARGS__);                                                      \
      })
#else
#define HOLOSCAN_LOG_AUTOMATIC_WARN(...) (void)0
#endif
