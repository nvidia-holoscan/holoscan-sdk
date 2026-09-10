/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke: HOLOSCAN_TRY propagates errors from both value-returning and
// void-returning expected<> functions, and a successful chain returns the
// correct value without short-circuiting.

#include <holoscan/core/expected_macros.hpp>

namespace {

// Returns expected<int,int>; on failure carries error code 7.
tl::expected<int, int> make_value(const bool fail) {
  if (fail) {
    return tl::make_unexpected(7);
  }
  return 42;
}

// Returns expected<void,int>; on failure carries error code 9.
tl::expected<void, int> make_void(const bool fail) {
  if (fail) {
    return tl::make_unexpected(9);
  }
  return {};
}

// Chains make_value and make_void with HOLOSCAN_TRY; any inner failure
// short-circuits and propagates the error to the caller.
tl::expected<int, int> propagate(const bool value_failure, const bool void_failure) {
  HOLOSCAN_TRY(auto value, make_value(value_failure));
  HOLOSCAN_TRY(make_void(void_failure));
  return value;
}

}  // namespace

// REQ: EXPECTED-018
int main() {
  const auto success = propagate(false, false);
  const auto value_failure = propagate(true, false);
  const auto void_failure = propagate(false, true);

  return success && *success == 42 && !value_failure && value_failure.error() == 7 &&
                 !void_failure && void_failure.error() == 9
             ? 0
             : 1;  // exit 1: success value wrong, or error code from value/void failure wrong
}
