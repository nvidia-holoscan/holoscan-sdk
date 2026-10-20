/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace holoscan::detail {

enum class HexPrefix : std::uint8_t {
  kNone,
  kSha256,
};

inline void append_percent_encoded_uri_component(std::string& output, std::string_view value) {
  constexpr std::string_view kHexDigits{"0123456789ABCDEF"};
  for (const unsigned char byte : value) {
    const bool unreserved = (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
                            (byte >= '0' && byte <= '9') || byte == '-' || byte == '_' ||
                            byte == '.' || byte == '~';
    if (unreserved) {
      output.push_back(static_cast<char>(byte));
      continue;
    }
    output.push_back('%');
    output.push_back(kHexDigits[(byte >> 4U) & 0x0FU]);
    output.push_back(kHexDigits[byte & 0x0FU]);
  }
}

inline void append_percent_encoded_uri_path_segment(std::string& output, std::string_view segment) {
  if (!output.empty()) {
    output.push_back('/');
  }
  append_percent_encoded_uri_component(output, segment);
}

inline void append_json_string(std::string& output, std::string_view value) {
  constexpr std::string_view kHexDigits{"0123456789abcdef"};
  output.push_back('"');
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (byte < 0x20U) {
          output += "\\u00";
          output.push_back(kHexDigits[(byte >> 4U) & 0x0FU]);
          output.push_back(kHexDigits[byte & 0x0FU]);
        } else {
          output.push_back(static_cast<char>(byte));
        }
        break;
    }
  }
  output.push_back('"');
}

[[nodiscard]] inline std::string json_string(std::string_view value) {
  std::string result;
  result.reserve(value.size() + 2U);
  append_json_string(result, value);
  return result;
}

[[nodiscard]] inline std::string percent_encode_uri_component(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  append_percent_encoded_uri_component(result, value);
  return result;
}

inline void append_lowercase_hex(std::string& output, std::span<const std::byte> bytes,
                                 HexPrefix prefix) {
  constexpr std::string_view kHexDigits{"0123456789abcdef"};
  if (prefix == HexPrefix::kSha256) {
    output += "sha256:";
  }
  for (const std::byte value : bytes) {
    const auto byte = std::to_integer<std::uint8_t>(value);
    output.push_back(kHexDigits[byte >> 4U]);
    output.push_back(kHexDigits[byte & 0x0FU]);
  }
}

[[nodiscard]] inline std::string lowercase_hex_text(std::span<const std::byte> bytes,
                                                    HexPrefix prefix) {
  const std::size_t prefix_size = prefix == HexPrefix::kSha256 ? 7U : 0U;
  std::string result;
  result.reserve(prefix_size + bytes.size() * 2U);
  append_lowercase_hex(result, bytes, prefix);
  return result;
}

}  // namespace holoscan::detail
