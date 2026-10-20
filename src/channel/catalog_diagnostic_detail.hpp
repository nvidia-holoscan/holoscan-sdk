/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace holoscan::channel_internal {

/// @brief Escape prefix introducing one two-hex-digit detail byte.
inline constexpr char kCatalogDetailEscape = '%';

/// @brief Return whether a detail byte would otherwise be read as record structure.
/// @param character Candidate byte from a dynamic detail value.
/// @return True when the byte delimits fields, values, or projected facts.
/// @note Endpoint names are validated as printable UTF-8, which admits every one of these bytes.
[[nodiscard]] constexpr bool catalog_detail_byte_is_structural(const char character) noexcept {
  return character == '|' || character == '=' || character == ';' ||
         character == kCatalogDetailEscape;
}

/// @brief Append one dynamic value with its structural bytes escaped.
/// @param target Detail record under construction.
/// @param value Caller-authored value such as an endpoint name.
/// @note A catalog diagnostic detail record is a `|key=value` list that compiler lowering reparses
/// and reprojects as `;`-joined facts. Without escaping, a valid endpoint name containing those
/// bytes can forge a neighboring field, truncate its own, or split a projected fact, so the
/// machine-readable diagnostic would state facts the catalog never produced.
inline void append_escaped_detail_value(std::string& target, const std::string_view value) {
  constexpr std::string_view kHexDigits = "0123456789ABCDEF";
  for (const char character : value) {
    if (!catalog_detail_byte_is_structural(character)) {
      target.push_back(character);
      continue;
    }
    const auto byte = static_cast<unsigned char>(character);
    target.push_back(kCatalogDetailEscape);
    target.push_back(kHexDigits[byte >> 4U]);
    target.push_back(kHexDigits[byte & 0x0FU]);
  }
}

/// @brief Decode one escaped detail value back to its original bytes.
/// @param value Escaped value recovered from a detail record.
/// @return The original value; a truncated or non-hex escape is preserved verbatim.
/// @note Decoding is total so a malformed record still projects readable evidence instead of
/// failing lowering. Only sequences this encoder can emit are folded back.
[[nodiscard]] inline std::string decode_detail_value(const std::string_view value) {
  const auto hex_digit = [](const char character) -> int {
    if (character >= '0' && character <= '9') {
      return character - '0';
    }
    if (character >= 'A' && character <= 'F') {
      return (character - 'A') + 10;
    }
    return -1;
  };
  std::string decoded;
  decoded.reserve(value.size());
  for (std::size_t index = 0U; index < value.size(); ++index) {
    const int high = index + 2U < value.size() ? hex_digit(value[index + 1U]) : -1;
    const int low = index + 2U < value.size() ? hex_digit(value[index + 2U]) : -1;
    if (value[index] != kCatalogDetailEscape || high < 0 || low < 0) {
      decoded.push_back(value[index]);
      continue;
    }
    const auto character = static_cast<char>((high << 4) | low);
    if (!catalog_detail_byte_is_structural(character)) {
      decoded.push_back(value[index]);
      continue;
    }
    decoded.push_back(character);
    index += 2U;
  }
  return decoded;
}

}  // namespace holoscan::channel_internal
