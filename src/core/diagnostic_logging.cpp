/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <string_view>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan {
namespace {

enum class EnvironmentPolicy { kUnset, kEnabled, kDisabled, kInvalid };

EnvironmentPolicy read_environment_policy() noexcept {
  const char* value = std::getenv("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING");
  if (value == nullptr || *value == '\0') {
    return EnvironmentPolicy::kUnset;
  }
  const std::string_view text{value};
  const auto equal_ascii = [text](std::string_view upper) noexcept {
    if (text.size() != upper.size()) {
      return false;
    }
    for (std::size_t index = 0; index < text.size(); ++index) {
      const char c = text[index];
      if (c != upper[index] && c != upper[index] + ('a' - 'A')) {
        return false;
      }
    }
    return true;
  };
  if (equal_ascii("ON")) {
    return EnvironmentPolicy::kEnabled;
  }
  return equal_ascii("OFF") ? EnvironmentPolicy::kDisabled : EnvironmentPolicy::kInvalid;
}

struct AutomaticDiagnosticPolicy {
  std::atomic<bool> preference{true};
  const EnvironmentPolicy environment{read_environment_policy()};
  std::atomic<bool> invalid_warning_attempted{false};
};

AutomaticDiagnosticPolicy& policy() noexcept {
  // Initialization never calls a logger. Reentrant sinks can safely query the published state.
  static AutomaticDiagnosticPolicy state;
  return state;
}

void warn_invalid_policy(AutomaticDiagnosticPolicy& state) noexcept {
  if (state.environment != EnvironmentPolicy::kInvalid ||
      state.invalid_warning_attempted.exchange(true, std::memory_order_relaxed)) {
    return;
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    HOLOSCAN_LOG_WARN(
        "Ignoring invalid HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING; "
        "expected ON or OFF. Using the API preference (initially enabled).");
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    // Configuration feedback must not replace the failure being reported.
    return;
  }
#endif
}

}  // namespace

void set_automatic_diagnostic_logging_enabled(bool enabled) noexcept {
  auto& state = policy();
  state.preference.store(enabled, std::memory_order_relaxed);
  warn_invalid_policy(state);
}

bool automatic_diagnostic_logging_enabled() noexcept {
  auto& state = policy();
  const bool enabled = state.environment == EnvironmentPolicy::kEnabled ||
                       (state.environment != EnvironmentPolicy::kDisabled &&
                        state.preference.load(std::memory_order_relaxed));
  warn_invalid_policy(state);
  return enabled;
}

}  // namespace holoscan
