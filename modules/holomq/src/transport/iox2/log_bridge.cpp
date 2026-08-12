/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "transport/iox2/log_bridge.hpp"

#include <holomq/logger.hpp>

#include "iox2/log.hpp"
#include "iox2/log_level.hpp"

#include <format>
#include <mutex>
#include <string>

namespace holomq {
namespace {

[[nodiscard]] LogLevel map_iox2_log_level(const iox2::LogLevel level) noexcept {
  switch (level) {
    case iox2::LogLevel::Trace:
      return LogLevel::Trace;
    case iox2::LogLevel::Debug:
      return LogLevel::Debug;
    case iox2::LogLevel::Info:
      return LogLevel::Info;
    case iox2::LogLevel::Warn:
      return LogLevel::Warn;
    case iox2::LogLevel::Error:
      return LogLevel::Error;
    case iox2::LogLevel::Fatal:
      return LogLevel::Critical;
  }
  return LogLevel::Info;
}

[[nodiscard]] iox2::LogLevel map_holomq_log_level(const LogLevel level) noexcept {
  switch (level) {
    case LogLevel::Trace:
      return iox2::LogLevel::Trace;
    case LogLevel::Debug:
      return iox2::LogLevel::Debug;
    case LogLevel::Info:
      return iox2::LogLevel::Info;
    case LogLevel::Warn:
      return iox2::LogLevel::Warn;
    case LogLevel::Error:
      return iox2::LogLevel::Error;
    case LogLevel::Critical:
      return iox2::LogLevel::Fatal;
    case LogLevel::Off:
      // iceoryx2 has no "off" level; keep native diagnostics at Error when HoloMQ is Off.
      return iox2::LogLevel::Error;
  }
  return iox2::LogLevel::Info;
}

class Iox2LogBridge final : public iox2::Log {
 public:
  void log(const iox2::LogLevel log_level, const char* origin, const char* message) override {
    if (origin == nullptr) {
      origin = "unknown";
    }
    if (message == nullptr) {
      message = "";
    }

    if (!transport_logging_enabled()) {
      return;
    }

    try {
      const std::string formatted = std::format("[iox2][{}] {}", origin, message);
      log_literal(map_iox2_log_level(log_level), "iox2", 0, origin, formatted.c_str());
    } catch (...) {
      log_literal(map_iox2_log_level(log_level), "iox2", 0, origin, message);
    }
  }
};

Iox2LogBridge& iox2_log_bridge_instance() noexcept {
  static Iox2LogBridge bridge;
  return bridge;
}

std::once_flag g_install_once;
bool g_bridge_installed = false;

}  // namespace

void sync_iox2_log_level(const LogLevel level) noexcept {
  iox2::set_log_level(map_holomq_log_level(level));
}

bool install_iox2_log_bridge() noexcept {
  std::call_once(g_install_once, []() {
    g_bridge_installed = iox2::set_logger(iox2_log_bridge_instance());
    if (g_bridge_installed) {
      sync_iox2_log_level(log_level());
    }
  });
  return g_bridge_installed;
}

}  // namespace holomq
