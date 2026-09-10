/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "artifact_test_support.hpp"

#include <cstddef>
#include <string>
#include <string_view>

namespace holoscan::detail::artifact_test {

[[nodiscard]] std::string normalized_artifact_json(std::string_view text) {
  if (text.empty()) {
    return {};
  }

  std::string result;
  result.reserve(text.size() * 2U);
  bool in_string = false;
  bool escaped = false;
  std::size_t indent = 0U;
  const auto append_indent = [&result, &indent]() { result.append(indent * 2U, ' '); };
  for (const char value : text) {
    if (in_string) {
      result.push_back(value);
      if (escaped) {
        escaped = false;
      } else if (value == '\\') {
        escaped = true;
      } else if (value == '"') {
        in_string = false;
      }
      continue;
    }
    if (value == ' ' || value == '\t' || value == '\r' || value == '\n') {
      continue;
    }
    if (!result.empty() &&
        ((result.back() == '{' && value != '}') || (result.back() == '[' && value != ']'))) {
      result.push_back('\n');
      append_indent();
    }
    switch (value) {
      case '"':
        in_string = true;
        result.push_back(value);
        break;
      case '{':
      case '[':
        result.push_back(value);
        ++indent;
        break;
      case '}':
      case ']':
        if (indent != 0U) {
          --indent;
        }
        if (!result.empty() && result.back() != '{' && result.back() != '[') {
          result.push_back('\n');
          append_indent();
        }
        result.push_back(value);
        break;
      case ',':
        result.append(",\n");
        append_indent();
        break;
      case ':':
        result.append(": ");
        break;
      default:
        result.push_back(value);
        break;
    }
  }
  result.push_back('\n');
  return result;
}

}  // namespace holoscan::detail::artifact_test
