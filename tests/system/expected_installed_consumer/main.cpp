/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Imported Holoscan targets must leave the compile-time logging threshold under caller control.
// Check before logger.hpp supplies its documented default.
#if defined(HOLOSCAN_LOG_ACTIVE_LEVEL)
#error "installed Holoscan targets must not define HOLOSCAN_LOG_ACTIVE_LEVEL for consumers"
#endif

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/logger/logger.hpp>

static_assert(HOLOSCAN_LOG_ACTIVE_LEVEL == HOLOSCAN_LOG_LEVEL_TRACE,
              "logger.hpp must retain its caller-owned default threshold");

namespace {

holoscan::expected<int, holoscan::Error> make_value(const bool fail) {
  if (fail) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotFound, "installed value failure"});
  }
  return 21;
}

holoscan::expected<void, holoscan::Error> validate_value(const int value, const bool fail) {
  if (fail || value != 21) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "installed void failure"});
  }
  return {};
}

holoscan::expected<int, holoscan::Error> propagate_try_forms(const bool value_failure,
                                                             const bool void_failure) {
  HOLOSCAN_TRY(auto value, make_value(value_failure));
  HOLOSCAN_TRY(validate_value(value, void_failure));
  return value * 2;
}

holoscan::expected<int, holoscan::Error> propagate_explicit_forms(const bool value_failure,
                                                                  const bool void_failure) {
  int value = 0;
  HOLOSCAN_ASSIGN_OR_RETURN(value, make_value(value_failure));
  HOLOSCAN_RETURN_IF_ERROR(validate_value(value, void_failure));
  return value * 2;
}

bool has_expected_failure(const holoscan::expected<int, holoscan::Error>& result,
                          const holoscan::ErrorCode code) {
  return !result && result.error().code == code;
}

}  // namespace

// REQ: EXPECTED-018
// REQ: EXPECTED-021
int main() {
  holoscan::set_log_level(holoscan::LogLevel::OFF);
  HOLOSCAN_LOG_INFO("installed package consumer linked the public logger surface");

  const auto try_success = propagate_try_forms(false, false);
  const auto try_value_failure = propagate_try_forms(true, false);
  const auto try_void_failure = propagate_try_forms(false, true);
  const auto explicit_success = propagate_explicit_forms(false, false);
  const auto explicit_value_failure = propagate_explicit_forms(true, false);
  const auto explicit_void_failure = propagate_explicit_forms(false, true);

  return try_success && *try_success == 42 &&
                 has_expected_failure(try_value_failure, holoscan::ErrorCode::kNotFound) &&
                 has_expected_failure(try_void_failure, holoscan::ErrorCode::kInvalidArgument) &&
                 explicit_success && *explicit_success == 42 &&
                 has_expected_failure(explicit_value_failure, holoscan::ErrorCode::kNotFound) &&
                 has_expected_failure(explicit_void_failure, holoscan::ErrorCode::kInvalidArgument)
             ? 0
             : 1;
}
