/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "holoscan/core/errors.hpp"
#include <fmt/format.h>

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
  if (static_cast<int>(error_code) >= static_cast<int>(holoscan::ErrorCode::kErrorCodeCount)) {
    return "Unknown error code";
  }
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
