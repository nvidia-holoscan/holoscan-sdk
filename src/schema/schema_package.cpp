/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/schema/schema_package.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace holoscan::detail {
namespace {

void append_diagnostic_json_string(std::string& output, const std::string_view value) {
  constexpr std::string_view kHex{"0123456789abcdef"};
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
          output.push_back(kHex[byte >> 4U]);
          output.push_back(kHex[byte & 0x0FU]);
        } else {
          output.push_back(static_cast<char>(byte));
        }
        break;
    }
  }
  output.push_back('"');
}

void append_diagnostic_hex(std::string& output, const std::span<const std::byte> bytes) {
  constexpr std::string_view kHex{"0123456789abcdef"};
  output.push_back('"');
  for (const std::byte byte : bytes) {
    const auto value = std::to_integer<std::uint8_t>(byte);
    output.push_back(kHex[value >> 4U]);
    output.push_back(kHex[value & 0x0FU]);
  }
  output.push_back('"');
}

void append_attach_identity(std::string& output, const SchemaIdentity& identity) {
  output += "{\"canonical_name\":";
  append_diagnostic_json_string(output, identity.canonical_name);
  output += ",\"compatibility_epoch\":";
  output += std::to_string(identity.compatibility_epoch);
  output += ",\"family_id\":";
  append_diagnostic_hex(output, identity.family_id.value);
  output += ",\"file_identifier\":";
  append_diagnostic_json_string(output,
                                std::string_view{identity.file_identifier.value.data(),
                                                 identity.file_identifier.value.size()});
  output.push_back('}');
}

}  // namespace

std::string schema_attach_mismatch_message(const SchemaIdentity& expected_identity,
                                           const SchemaIdentity& local_identity) {
  std::string output{"attached endpoint requires a different schema package; expected="};
  append_attach_identity(output, expected_identity);
  output += "; local=";
  append_attach_identity(output, local_identity);
  return output;
}

}  // namespace holoscan::detail
