/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke: holoscan::expected<T,Error> integrates with HOLOSCAN_TRY for both
// value-bearing and void paths; a successful chain returns without error and
// a failing inner call short-circuits with the holoscan::Error propagated.

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace {

// Returns a fixed value (21) for the downstream validator to check.
holoscan::expected<int, holoscan::Error> produce_value() {
  return 21;
}

// Validates that the value is the expected sentinel; returns an Error otherwise.
holoscan::expected<void, holoscan::Error> validate_value(int value) {
  if (value != 21) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "unexpected smoke value"});
  }
  return {};
}

// REQ: EXPECTED-016
// Chains produce_value() and validate_value() with HOLOSCAN_TRY to exercise
// both the value-extraction and void-propagation forms of the macro.
holoscan::expected<void, holoscan::Error> run_smoke() {
  HOLOSCAN_TRY(auto value, produce_value());
  HOLOSCAN_TRY(validate_value(value));
  return {};
}

}  // namespace

int main() {
  return run_smoke() ? 0 : 1;  // exit 1: run_smoke() returned an unexpected Error
}
