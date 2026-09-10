/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fmt/format.h>
#include <holoscan/core/errors.hpp>
#include <magic_enum/magic_enum.hpp>

#include <string>
#include <string_view>

namespace holoscan {

std::string_view error_code_name(const ErrorCode code) noexcept {
  const std::string_view name = magic_enum::enum_name(code);
  return name.empty() ? std::string_view{"unknown"} : name;
}

Error make_publication_partial_error(const std::uint32_t delivered_recipient_count,
                                     const std::uint32_t expected_recipient_count) noexcept {
  try {
    return Error{ErrorCode::kPublicationPartial,
                 fmt::format("PUBLICATION_PARTIAL delivered={} expected={}",
                             delivered_recipient_count,
                             expected_recipient_count)};
  } catch (...) {
    return Error{ErrorCode::kPublicationPartial};
  }
}

RuntimeError::RuntimeError(holoscan::ErrorCode error_code)
    : std::runtime_error(std::string(error_code_name(error_code))), code_(error_code) {}

RuntimeError::RuntimeError(holoscan::ErrorCode error_code, const std::string& what_arg)
    : std::runtime_error(construct_error_message(error_code, std::string_view{what_arg})),
      code_(error_code) {}

RuntimeError::RuntimeError(holoscan::ErrorCode error_code, std::string_view what_arg)
    : std::runtime_error(construct_error_message(error_code, what_arg)), code_(error_code) {}

RuntimeError::RuntimeError(holoscan::ErrorCode error_code, const char* what_arg)
    : std::runtime_error(construct_error_message(error_code, what_arg)), code_(error_code) {}

RuntimeError::RuntimeError(const Error& error)
    : std::runtime_error(construct_error_message(
          error.code, error.message.empty() ? nullptr : error.message.c_str())),
      code_(error.code) {}

std::string RuntimeError::construct_error_message(holoscan::ErrorCode error_code,
                                                  const char* what_arg) {
  if (what_arg == nullptr) {
    return std::string(error_code_name(error_code));
  }
  return construct_error_message(error_code, std::string_view{what_arg});
}

std::string RuntimeError::construct_error_message(holoscan::ErrorCode error_code,
                                                  std::string_view what_arg) {
  return fmt::format("{}: {}", error_code_name(error_code), what_arg);
}

}  // namespace holoscan
