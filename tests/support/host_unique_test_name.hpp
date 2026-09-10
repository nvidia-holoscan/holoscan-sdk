/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace holoscan::test {

inline std::atomic<std::uint64_t> g_host_unique_test_name_ordinal{};

/// Return a filesystem- and service-name-safe form of one test identity component.
[[nodiscard]] inline std::string sanitize_test_identity(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const char byte : value) {
    const bool safe = (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
                      (byte >= '0' && byte <= '9') || byte == '-' || byte == '_' || byte == '.';
    result.push_back(safe ? byte : '_');
  }
  return result;
}

/// Return the active gtest suite and case as one host-name-safe identity.
[[nodiscard]] inline std::string active_test_identity() {
  const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
  if (test == nullptr) {
    return "unknown-test";
  }
  return sanitize_test_identity(std::string(test->test_suite_name()) + "-" + test->name());
}

/// Build a host-unique name for a test-created archive, service, or graph identity.
///
/// Test-created host-visible names must use this helper or an equivalent identity containing the
/// active test, process, time, and invocation ordinal; fixed names are not parallel-CTest safe.
[[nodiscard]] inline std::string make_host_unique_test_name(std::string_view stem) {
#if defined(_WIN32)
  const auto process = static_cast<long long>(::_getpid());
#else
  const auto process = static_cast<long long>(::getpid());
#endif
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  return sanitize_test_identity(stem) + "-" + active_test_identity() + "-" +
         std::to_string(process) + "-" + std::to_string(tick) + "-" +
         std::to_string(g_host_unique_test_name_ordinal.fetch_add(1U, std::memory_order_relaxed));
}

}  // namespace holoscan::test
