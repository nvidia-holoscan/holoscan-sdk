/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/errors.hpp>

#include <magic_enum/magic_enum.hpp>

#include <string>

namespace holomq {

std::string_view error_code_name(const ErrorCode code) noexcept {
  const std::string_view name = magic_enum::enum_name(code);
  return name.empty() ? std::string_view{"unknown"} : name;
}

std::string error_message(const Error& error) {
  const std::string_view code_name = error_code_name(error.code);
  if (error.message.empty()) {
    return std::string(code_name);
  }
  return std::string(code_name) + ": " + error.message;
}

}  // namespace holomq
