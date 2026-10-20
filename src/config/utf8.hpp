/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace holoscan::config::detail {

[[nodiscard]] inline std::optional<std::size_t> invalid_utf8_offset(
    std::string_view text) noexcept {
  std::size_t index = 0;
  while (index < text.size()) {
    const auto first = static_cast<unsigned char>(text[index]);
    std::size_t width = 0;
    std::uint32_t codepoint = 0;
    if (first <= 0x7FU) {
      width = 1;
      codepoint = first;
    } else if ((first & 0xE0U) == 0xC0U) {
      width = 2;
      codepoint = first & 0x1FU;
    } else if ((first & 0xF0U) == 0xE0U) {
      width = 3;
      codepoint = first & 0x0FU;
    } else if ((first & 0xF8U) == 0xF0U) {
      width = 4;
      codepoint = first & 0x07U;
    } else {
      return index;
    }
    if (index + width > text.size()) {
      return index;
    }
    for (std::size_t offset = 1; offset < width; ++offset) {
      const auto next = static_cast<unsigned char>(text[index + offset]);
      if ((next & 0xC0U) != 0x80U) {
        return index;
      }
      codepoint = (codepoint << 6U) | (next & 0x3FU);
    }
    if ((width == 2U && codepoint < 0x80U) || (width == 3U && codepoint < 0x800U) ||
        (width == 4U && codepoint < 0x10000U) || codepoint > 0x10FFFFU ||
        (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
      return index;
    }
    index += width;
  }
  return std::nullopt;
}

[[nodiscard]] inline bool valid_utf8(std::string_view text) noexcept {
  return !invalid_utf8_offset(text).has_value();
}

}  // namespace holoscan::config::detail
