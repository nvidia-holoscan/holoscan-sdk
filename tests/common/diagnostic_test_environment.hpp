/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdlib>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include <holoscan/logger/logger.hpp>

namespace holoscan::test {

// Mutate the same C runtime environment read by std::getenv. The Windows CRT
// treats an empty value as removal; both forms have the same policy meaning.
inline int set_test_environment(const char* name, const char* value) noexcept {
#if defined(_WIN32)
  return _putenv_s(name, value != nullptr ? value : "");
#else
  return value != nullptr ? setenv(name, value, 1) : unsetenv(name);
#endif
}

// Restore both presence and value on ordinary scope exit, including early test returns.
class ScopedTestEnvironment {
 public:
  ScopedTestEnvironment(const char* name, const char* value) : name_(name) {
    if (const char* previous = std::getenv(name)) {
      previous_ = previous;
    }
    EXPECT_EQ(set_test_environment(name_.c_str(), value), 0);
  }
  ~ScopedTestEnvironment() {
    EXPECT_EQ(set_test_environment(name_.c_str(), previous_ ? previous_->c_str() : nullptr), 0);
  }
  ScopedTestEnvironment(const ScopedTestEnvironment&) = delete;
  ScopedTestEnvironment& operator=(const ScopedTestEnvironment&) = delete;

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

// Logger setters honor environment overrides. Clear those overrides for the whole
// scope, restore the logger first, and only then restore the inherited environment.
// Declare this guard before sessions so session cleanup runs under the test logger.
class ScopedDiagnosticLogger {
 public:
  explicit ScopedDiagnosticLogger(LogLevel level)
      : previous_level_(log_level()),
        previous_pattern_(Logger::pattern()),
        previous_level_explicit_(Logger::log_level_set_by_user),
        previous_pattern_explicit_(Logger::log_pattern_set_by_user),
        level_environment_("HOLOSCAN_LOG_LEVEL", nullptr),
        format_environment_("HOLOSCAN_LOG_FORMAT", nullptr) {
    set_log_level(level);
    set_log_pattern("%l: %v");
  }
  ~ScopedDiagnosticLogger() {
    set_log_level(previous_level_);
    set_log_pattern(previous_pattern_);
    Logger::log_level_set_by_user = previous_level_explicit_;
    Logger::log_pattern_set_by_user = previous_pattern_explicit_;
  }
  ScopedDiagnosticLogger(const ScopedDiagnosticLogger&) = delete;
  ScopedDiagnosticLogger& operator=(const ScopedDiagnosticLogger&) = delete;

 private:
  LogLevel previous_level_;
  std::string previous_pattern_;
  bool previous_level_explicit_;
  bool previous_pattern_explicit_;
  ScopedTestEnvironment level_environment_;
  ScopedTestEnvironment format_environment_;
};

}  // namespace holoscan::test
