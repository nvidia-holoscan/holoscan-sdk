/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Optional bridge from HoloMQ logging to the Holoscan logger.
// Requires Holoscan Core headers and linking holoscan::Logger at compile time; do not include in
// applications that do not link the Holoscan logger.

#include <holomq/logger.hpp>

#include <holoscan/logger/logger.hpp>

#include <cstdio>

namespace holomq {

namespace adapters {
namespace detail {

static_assert(static_cast<int>(LogLevel::Trace) == static_cast<int>(holoscan::LogLevel::TRACE));
static_assert(static_cast<int>(LogLevel::Debug) == static_cast<int>(holoscan::LogLevel::DEBUG));
static_assert(static_cast<int>(LogLevel::Info) == static_cast<int>(holoscan::LogLevel::INFO));
static_assert(static_cast<int>(LogLevel::Warn) == static_cast<int>(holoscan::LogLevel::WARN));
static_assert(static_cast<int>(LogLevel::Error) == static_cast<int>(holoscan::LogLevel::ERROR));
static_assert(static_cast<int>(LogLevel::Critical) ==
              static_cast<int>(holoscan::LogLevel::CRITICAL));
static_assert(static_cast<int>(LogLevel::Off) == static_cast<int>(holoscan::LogLevel::OFF));

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
    // Best-effort fallback; do not call HoloMQ logging APIs from a LogHandler (re-entry).
    static_cast<void>(std::fprintf(stderr, "[holomq][holoscan-logger-bridge] %s\n", text));
  }
}

}  // namespace detail
}  // namespace adapters

/// Redirect HoloMQ library logs (including iceoryx2 transport logs when enabled) to \c
/// holoscan::Logger.
///
/// Call at process startup, before \ref ContextBuilder::create and before \ref
/// enable_transport_logging. HoloMQ filters by \ref set_log_level / \c HOLOMQ_LOG_LEVEL before
/// the handler runs; align
/// \c holoscan::Logger::set_level for consistent output. The installed handler must not call
/// HoloMQ logging APIs (\c HOLOMQ_LOG_*, \ref set_log_level, etc.) — re-entry deadlocks.
inline void redirect_holomq_logs_to_holoscan() noexcept {
  set_log_handler(adapters::detail::holoscan_log_thunk, nullptr);
}

}  // namespace holomq
