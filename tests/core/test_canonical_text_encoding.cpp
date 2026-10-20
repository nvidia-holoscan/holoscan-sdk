/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "core/canonical_text_encoding.hpp"

namespace holoscan::detail {
namespace {

TEST(CanonicalTextEncoding, PercentEncodesEveryByteAsOneUriComponent) {
  std::string input;
  input.reserve(256U);
  for (std::uint16_t value = 0U; value <= 0xFFU; ++value) {
    input.push_back(static_cast<char>(value));
  }

  // Key setup: encode one component containing every possible byte.
  const std::string encoded = percent_encode_uri_component(input);
  std::size_t cursor = 0U;
  constexpr std::string_view kHexDigits{"0123456789ABCDEF"};
  // Key assertion: RFC 3986 unreserved bytes remain literal and every other byte uses uppercase
  // percent encoding without treating embedded NUL or non-ASCII UTF-8 bytes specially.
  for (std::uint16_t value = 0U; value <= 0xFFU; ++value) {
    const auto byte = static_cast<unsigned char>(value);
    const bool unreserved = (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
                            (byte >= '0' && byte <= '9') || byte == '-' || byte == '_' ||
                            byte == '.' || byte == '~';
    if (unreserved) {
      ASSERT_LT(cursor, encoded.size());
      EXPECT_EQ(static_cast<unsigned char>(encoded[cursor++]), byte);
      continue;
    }
    ASSERT_LE(cursor + 3U, encoded.size());
    EXPECT_EQ(encoded[cursor++], '%');
    EXPECT_EQ(encoded[cursor++], kHexDigits[(byte >> 4U) & 0x0FU]);
    EXPECT_EQ(encoded[cursor++], kHexDigits[byte & 0x0FU]);
  }
  EXPECT_EQ(cursor, encoded.size());
  EXPECT_TRUE(percent_encode_uri_component({}).empty());
  EXPECT_EQ(percent_encode_uri_component(std::string_view{"A\0B", 3U}), "A%00B");
  EXPECT_EQ(percent_encode_uri_component("\xC3\xA9"), "%C3%A9");
}

// REQ: COMPILE-068
TEST(CanonicalTextEncoding, JoinsEncodedPathSegmentsWithoutCollapsingDecodedIdentity) {
  std::string first;
  append_percent_encoded_uri_path_segment(first, "a");
  append_percent_encoded_uri_path_segment(first, "b/c");

  std::string second;
  append_percent_encoded_uri_path_segment(second, "a/b");
  append_percent_encoded_uri_path_segment(second, "c");

  EXPECT_EQ(first, "a/b%2Fc");
  EXPECT_EQ(second, "a%2Fb/c");
  EXPECT_NE(first, second);
}

TEST(CanonicalTextEncoding, EscapesEveryByteAsOneJsonString) {
  std::string input;
  input.reserve(256U);
  for (std::uint16_t value = 0U; value <= 0xFFU; ++value) {
    input.push_back(static_cast<char>(value));
  }

  // Key setup: encode one JSON string containing every possible byte and append the same input
  // after caller-owned text.
  const std::string encoded = json_string(input);
  std::string appended{"prefix:"};
  append_json_string(appended, input);

  // Key assertion: JSON syntax bytes use their short escapes, other control bytes use lowercase
  // Unicode escapes, and bytes at or above 0x20 otherwise remain byte-exact.
  std::string expected{"\""};
  constexpr std::string_view kHexDigits{"0123456789abcdef"};
  for (std::uint16_t value = 0U; value <= 0xFFU; ++value) {
    const auto byte = static_cast<unsigned char>(value);
    switch (byte) {
      case '"':
        expected += "\\\"";
        break;
      case '\\':
        expected += "\\\\";
        break;
      case '\b':
        expected += "\\b";
        break;
      case '\f':
        expected += "\\f";
        break;
      case '\n':
        expected += "\\n";
        break;
      case '\r':
        expected += "\\r";
        break;
      case '\t':
        expected += "\\t";
        break;
      default:
        if (byte < 0x20U) {
          expected += "\\u00";
          expected.push_back(kHexDigits[(byte >> 4U) & 0x0FU]);
          expected.push_back(kHexDigits[byte & 0x0FU]);
        } else {
          expected.push_back(static_cast<char>(byte));
        }
        break;
    }
  }
  expected.push_back('"');
  EXPECT_EQ(encoded, expected);
  EXPECT_EQ(appended, "prefix:" + expected);
  EXPECT_EQ(json_string({}), "\"\"");
  EXPECT_EQ(json_string(std::string_view{"A\0B", 3U}), "\"A\\u0000B\"");
  EXPECT_EQ(json_string("\xC3\xA9"), "\"\xC3\xA9\"");
}

TEST(CanonicalTextEncoding, RendersEveryByteWithExplicitDigestPrefixPolicy) {
  std::array<std::byte, 256> bytes{};
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::byte>(index);
  }

  // Key setup: render all byte values with and without the SHA-256 tag and append one fixed digest
  // after existing text.
  const std::string untagged = lowercase_hex_text(bytes, HexPrefix::kNone);
  const std::string tagged = lowercase_hex_text(bytes, HexPrefix::kSha256);
  std::string appended{"prefix:"};
  const std::array<std::byte, 4> fixed{
      std::byte{0x00}, std::byte{0x01}, std::byte{0xAB}, std::byte{0xFF}};
  append_lowercase_hex(appended, fixed, HexPrefix::kSha256);

  // Key assertion: width is exactly two lowercase digits per byte, prefix selection is explicit,
  // and append form preserves caller-owned text without a temporary rendering contract.
  constexpr std::string_view kHexDigits{"0123456789abcdef"};
  ASSERT_EQ(untagged.size(), bytes.size() * 2U);
  ASSERT_EQ(tagged.size(), 7U + bytes.size() * 2U);
  EXPECT_TRUE(tagged.starts_with("sha256:"));
  EXPECT_EQ(tagged.substr(7U), untagged);
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    EXPECT_EQ(untagged[index * 2U], kHexDigits[(index >> 4U) & 0x0FU]);
    EXPECT_EQ(untagged[index * 2U + 1U], kHexDigits[index & 0x0FU]);
  }
  EXPECT_EQ(appended, "prefix:sha256:0001abff");
  EXPECT_EQ(lowercase_hex_text({}, HexPrefix::kNone), "");
  EXPECT_EQ(lowercase_hex_text({}, HexPrefix::kSha256), "sha256:");
}

}  // namespace
}  // namespace holoscan::detail
