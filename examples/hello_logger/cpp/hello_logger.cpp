/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke test for installed-SDK consumers of holoscan::logger: exercises the HOLOSCAN_LOG_* logging
// API, which is fmt-formatted and pulls in the PUBLIC header-only dependency fmt (logger.hpp
// includes <fmt/format.h>). If find_package(holoscan) cannot resolve fmt::fmt-header-only (logger's
// interface), or logger.hpp / the fmt headers are not installed, this fails to configure/compile
// against the install tree -- which is what we want CI to catch.

// REQ: EXPECTED-021
#if defined(HOLOSCAN_VERIFY_CALLER_OWNED_LOG_POLICY) && defined(HOLOSCAN_LOG_ACTIVE_LEVEL)
#error "The installed holoscan::logger target must not define HOLOSCAN_LOG_ACTIVE_LEVEL"
#endif

#include <holoscan/logger/logger.hpp>

#if defined(HOLOSCAN_VERIFY_CALLER_OWNED_LOG_POLICY)
static_assert(HOLOSCAN_LOG_ACTIVE_LEVEL == HOLOSCAN_LOG_LEVEL_TRACE,
              "logger.hpp must preserve the caller-owned default when no threshold is supplied");
#endif

#include <iostream>

int main() {
  holoscan::set_log_level(holoscan::LogLevel::INFO);

  // fmt-formatted log message (exercises the fmt dependency through the logging macros).
  const int answer = 42;
  HOLOSCAN_LOG_INFO("hello from the holoscan logger: answer={}", answer);

  std::cout << "holoscan logger sample complete\n";
  return 0;
}
