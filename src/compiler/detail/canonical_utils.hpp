/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>

namespace holoscan::detail {

[[nodiscard]] inline bool bounded_identity_valid(std::string_view value,
                                                 std::size_t max_length) noexcept {
  if (value.empty() || value.size() > max_length || !valid_utf8(value)) {
    return false;
  }
  const auto ascii_space = [](char character) noexcept {
    switch (character) {
      case ' ':
      case '\t':
      case '\n':
      case '\r':
      case '\f':
      case '\v':
        return true;
      default:
        return false;
    }
  };
  if (ascii_space(value.front()) || ascii_space(value.back())) {
    return false;
  }
  return std::none_of(value.begin(), value.end(), [](char character) noexcept {
    const auto byte = static_cast<std::uint8_t>(character);
    return byte <= 0x1FU || (byte >= 0x7FU && byte <= 0x9FU);
  });
}

inline void canonicalize_tokens(std::vector<std::string>& tokens) {
  std::sort(tokens.begin(), tokens.end());
  tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
}

}  // namespace holoscan::detail
