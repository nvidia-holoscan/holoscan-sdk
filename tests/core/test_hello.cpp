/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>

#include <holoscan/holoscan.hpp>

namespace {

// Smoke test confirming that the SDK's hello() entry point writes the canonical greeting
// to stdout. Also validates that the public "holoscan/holoscan.hpp" umbrella header
// compiles cleanly and exposes the holoscan:: namespace.
TEST(HoloscanCoreHelloTest, PrintsHelloHoloscanGreeting) {
  // Key setup: redirect std::cout to an ostringstream before calling hello() so the
  // output can be captured without relying on process-level stdout interception.
  const std::ostringstream output;
  auto* const original_buffer = std::cout.rdbuf(output.rdbuf());

  holoscan::hello();

  // Key assertion: restore the original buffer before asserting to avoid leaving
  // stdout redirected on test failure.
  std::cout.rdbuf(original_buffer);
  EXPECT_EQ(output.str(), "Hello Holoscan!\n");
}

}  // namespace
