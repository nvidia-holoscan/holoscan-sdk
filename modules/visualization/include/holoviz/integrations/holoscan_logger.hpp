/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Optional bridge from HoloViz logging to the Holoscan logger.
// Requires Holoscan Core headers and linking holoscan::logger at compile time; do not include in
// standalone HoloViz apps.

#include <holoviz/logger.hpp>

#include <holoscan/logger/logger.hpp>

#include <cstdio>

namespace holoviz {

namespace integrations {
namespace detail {

static_assert(static_cast<int>(LogLevel::kTrace) == static_cast<int>(holoscan::LogLevel::TRACE));
static_assert(static_cast<int>(LogLevel::kDebug) == static_cast<int>(holoscan::LogLevel::DEBUG));
static_assert(static_cast<int>(LogLevel::kInfo) == static_cast<int>(holoscan::LogLevel::INFO));
static_assert(static_cast<int>(LogLevel::kWarn) == static_cast<int>(holoscan::LogLevel::WARN));
static_assert(static_cast<int>(LogLevel::kError) == static_cast<int>(holoscan::LogLevel::ERROR));
static_assert(static_cast<int>(LogLevel::kCritical) ==
              static_cast<int>(holoscan::LogLevel::CRITICAL));
static_assert(static_cast<int>(LogLevel::kOff) == static_cast<int>(holoscan::LogLevel::OFF));

[[nodiscard]] constexpr holoscan::LogLevel to_holoscan_log_level(const LogLevel level) noexcept {
  return static_cast<holoscan::LogLevel>(static_cast<int>(level));
}

inline void holoscan_log_thunk(const LogLevel level, const char* file, const int line,
                               const char* function, const char* message,
                               void* /*user_data*/) noexcept {
  const char* text = message != nullptr ? message : "";
  try {
    holoscan::Logger::log(file, line, function, to_holoscan_log_level(level), "{}", text);
  } catch (...) {
    // Best-effort fallback; do not call HoloViz logging APIs from a LogHandler (re-entry).
    static_cast<void>(std::fprintf(stderr, "[holoviz][holoscan-logger-bridge] %s\n", text));
  }
}

}  // namespace detail
}  // namespace integrations

/// `@brief` Redirect HoloViz library logs to \c holoscan::Logger.
///
/// `@note` Call at process startup. HoloViz filters by \ref set_log_level / \c HOLOVIZ_LOG_LEVEL
/// before the handler runs; align \c holoscan::Logger::set_level for consistent output. The
/// installed handler must not call HoloViz logging APIs (\c HOLOVIZ_LOG_*, \ref set_log_level,
/// etc.) — re-entry deadlocks.
inline void redirect_holoviz_logs_to_holoscan() noexcept {
  set_log_handler(integrations::detail::holoscan_log_thunk, nullptr);
}

}  // namespace holoviz
