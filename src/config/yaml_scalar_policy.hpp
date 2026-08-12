/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>

namespace holoscan::config::detail {

[[nodiscard]] inline bool yaml_ascii_equal_ignoring_case(std::string_view left,
                                                         std::string_view right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    const auto fold = [](char value) {
      return value >= 'A' && value <= 'Z' ? static_cast<char>(value - 'A' + 'a') : value;
    };
    if (fold(left[index]) != fold(right[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool unsupported_yaml_implicit(std::string_view token) noexcept {
  constexpr std::array<std::string_view, 12> kWords{
      "null", "true", "false", "yes", "no", "on", "off", "nan", "inf", "infinity", ".nan", ".inf"};
  for (const auto word : kWords) {
    if (yaml_ascii_equal_ignoring_case(token, word)) {
      return token != "null" && token != "true" && token != "false";
    }
  }
  if (token.size() > 1U && (token.front() == '-' || token.front() == '+') &&
      (yaml_ascii_equal_ignoring_case(token.substr(1U), ".inf") ||
       yaml_ascii_equal_ignoring_case(token.substr(1U), ".nan"))) {
    return true;
  }
  return token.size() >= 10U && token[4] == '-' && token[7] == '-' &&
         std::all_of(token.begin(),
                     token.begin() + 4,
                     [](char value) { return value >= '0' && value <= '9'; }) &&
         token[5] >= '0' && token[5] <= '9' && token[6] >= '0' && token[6] <= '9' &&
         token[8] >= '0' && token[8] <= '9' && token[9] >= '0' && token[9] <= '9';
}

[[nodiscard]] inline bool yaml_numeric_candidate(
    std::string_view token, bool preserve_legacy_leading_minus = false) noexcept {
  if (token.empty()) {
    return false;
  }
  if (token.front() >= '0' && token.front() <= '9') {
    return true;
  }
  if (token.front() == '-') {
    if (preserve_legacy_leading_minus) {
      return true;
    }
    return token.size() > 1U &&
           ((token[1] >= '0' && token[1] <= '9') ||
            (token[1] == '.' && ((token.size() > 2U && token[2] >= '0' && token[2] <= '9') ||
                                 yaml_ascii_equal_ignoring_case(token.substr(1U), ".inf") ||
                                 yaml_ascii_equal_ignoring_case(token.substr(1U), ".nan"))));
  }
  if (token.front() == '+') {
    return token.size() > 1U &&
           ((token[1] >= '0' && token[1] <= '9') ||
            (token[1] == '.' && ((token.size() > 2U && token[2] >= '0' && token[2] <= '9') ||
                                 yaml_ascii_equal_ignoring_case(token.substr(1U), ".inf") ||
                                 yaml_ascii_equal_ignoring_case(token.substr(1U), ".nan"))));
  }
  return token.front() == '.' && token.size() > 1U &&
         ((token[1] >= '0' && token[1] <= '9') || yaml_ascii_equal_ignoring_case(token, ".inf") ||
          yaml_ascii_equal_ignoring_case(token, ".nan"));
}

[[nodiscard]] inline bool yaml_plain_string_representable(std::string_view text) noexcept {
  constexpr std::string_view kLeadingIndicators{"-?:,[]{}#&*!|>'\"%@`"};
  if (text.empty() || text.front() == ' ' || text.back() == ' ' ||
      kLeadingIndicators.find(text.front()) != std::string_view::npos || text == "null" ||
      text == "~" || text == "true" || text == "false" ||
      text.find_first_of(",[]{}") != std::string_view::npos || unsupported_yaml_implicit(text) ||
      yaml_numeric_candidate(text, true)) {
    return false;
  }
  return std::all_of(text.begin(), text.end(), [](const unsigned char byte) {
    return byte >= 0x20U && byte != ':' && byte != '#' && byte != '\n' && byte != '\r';
  });
}

}  // namespace holoscan::config::detail
