/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fmt/format.h>
#include <holoscan/core/errors.hpp>

#include <string>

namespace holoscan {

const char* RuntimeError::ErrorStrings[] = {
    "Success",                       // 0
    "Failure",                       // 1
    "InputContext receive() Error",  // 2
    "codec (de)serialize() Error",   // 3
    "Invalid argument",              // 4
    "Not found",                     // 5
    "Duplicate name",                // 6
    "Not Implemented",               // 7
};

RuntimeError::RuntimeError(holoscan::ErrorCode error_code)
    : std::runtime_error(error_string(error_code)) {}

RuntimeError::RuntimeError(holoscan::ErrorCode error_code, const std::string& what_arg)
    : std::runtime_error(construct_error_message(error_code, what_arg.c_str())) {}

RuntimeError::RuntimeError(holoscan::ErrorCode error_code, const char* what_arg)
    : std::runtime_error(construct_error_message(error_code, what_arg)) {}

const char* RuntimeError::error_string(const holoscan::ErrorCode error_code) {
  static_assert(sizeof(ErrorStrings) / sizeof(ErrorStrings[0]) ==
                    static_cast<size_t>(holoscan::ErrorCode::kErrorCodeCount),
                "ErrorStrings array size must match ErrorCode::kErrorCodeCount");
  if (static_cast<int>(error_code) < 0 ||
      static_cast<int>(error_code) >= static_cast<int>(holoscan::ErrorCode::kErrorCodeCount)) {
    return "Unknown error code";
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index) bounds checked above
  return ErrorStrings[static_cast<int>(error_code)];
}

std::string RuntimeError::construct_error_message(const holoscan::ErrorCode error_code,
                                                  const char* what_arg) {
  if (what_arg == nullptr) {
    return error_string(error_code);
  } else {
    return fmt::format("{}: {}", error_string(error_code), what_arg);
  }
}

}  // namespace holoscan
