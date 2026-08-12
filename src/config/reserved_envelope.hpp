/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <holoscan/config/document.hpp>

namespace holoscan::config_io::detail {
namespace reserved_envelope {

[[nodiscard]] inline config::ConfigError failure(std::string_view message) noexcept {
  return config::ConfigError::make(config::ConfigErrorCode::kUnsupportedValue,
                                   message,
                                   {},
                                   {},
                                   {},
                                   config::ConfigOperationStage::kParse);
}

[[nodiscard]] inline bool canonical_f64(std::string_view source, double value) noexcept {
  if (value == 0.0) {
    value = 0.0;
  }
  std::array<char, 128> buffer{};
  const auto rendered = std::to_chars(
      buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::general);
  return rendered.ec == std::errc() &&
         source == std::string_view{buffer.data(),
                                    static_cast<std::size_t>(rendered.ptr - buffer.data())};
}

template <typename Integer>
[[nodiscard]] inline config::ConfigResult<config::ConfigValue> decode_integer(
    std::string_view text) {
  Integer value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() ||
      std::to_string(value) != text) {
    return tl::make_unexpected(failure("invalid scalar escape"));
  }
  if constexpr (std::is_same_v<Integer, std::int64_t>) {
    return config::ConfigValue::signed_integer(value);
  } else {
    return config::ConfigValue::unsigned_integer(value);
  }
}

[[nodiscard]] inline config::ConfigResult<config::ConfigValue> decode_floating(
    std::string_view text) {
  double value = 0.0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value, std::chars_format::general);
  if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() ||
      !std::isfinite(value) || !canonical_f64(text, value)) {
    return tl::make_unexpected(failure("invalid scalar escape"));
  }
  return config::ConfigValue::floating(value);
}

[[nodiscard]] inline config::ConfigResult<config::ConfigValue> decode_scalar(
    const config::ConfigView& value) {
  if (value.kind() != config::ConfigValueKind::kObject || value.size() != 2U) {
    return tl::make_unexpected(failure("malformed scalar escape"));
  }
  auto kind = value.member("kind");
  auto text = value.member("value");
  if (!kind || !text) {
    return tl::make_unexpected(failure("malformed scalar escape"));
  }
  auto kind_text = kind->as_string();
  auto value_text = text->as_string();
  if (!kind_text || !value_text) {
    return tl::make_unexpected(failure("scalar escape fields must be strings"));
  }
  if (*kind_text == "i64") {
    return decode_integer<std::int64_t>(*value_text);
  }
  if (*kind_text == "u64") {
    return decode_integer<std::uint64_t>(*value_text);
  }
  if (*kind_text == "f64") {
    return decode_floating(*value_text);
  }
  return tl::make_unexpected(failure("invalid scalar escape"));
}

[[nodiscard]] inline config::ConfigResult<config::ConfigValue> decode_literal_object(
    const config::ConfigView& value, const config::ConfigValueLimits& limits) {
  if (value.kind() != config::ConfigValueKind::kArray) {
    return tl::make_unexpected(failure("malformed literal-object escape"));
  }
  std::vector<std::string> keys;
  std::vector<config::ConfigValue> values;
  keys.reserve(value.size());
  values.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    auto entry = value.element(index);
    if (!entry || entry->kind() != config::ConfigValueKind::kArray || entry->size() != 2U) {
      return tl::make_unexpected(failure("malformed literal-object entry"));
    }
    auto key = entry->element(0);
    auto item = entry->element(1);
    if (!key || !item) {
      return tl::make_unexpected(failure("malformed literal-object entry"));
    }
    auto decoded_key = key->as_string();
    if (!decoded_key) {
      return tl::make_unexpected(failure("literal-object key must be string"));
    }
    keys.emplace_back(*decoded_key);
    values.push_back(item->to_value());
  }
  std::vector<config::ConfigObjectEntry> entries;
  entries.reserve(keys.size());
  for (std::size_t index = 0; index < keys.size(); ++index) {
    entries.push_back({keys[index], values[index]});
  }
  return config::ConfigValue::object(entries, limits);
}

}  // namespace reserved_envelope

[[nodiscard]] inline config::ConfigResult<config::ConfigValue> decode_reserved_envelope(
    config::ConfigValue object, const config::ConfigValueLimits& limits) {
  auto view = object.view();
  if (view.size() != 1U) {
    return object;
  }
  auto key = view.key(0);
  auto value = view.element(0);
  if (!key || !value) {
    return object;
  }
  if (*key == "$holoscan.config.scalar/v1") {
    return reserved_envelope::decode_scalar(*value);
  }
  if (*key == "$holoscan.config.object/v1") {
    return reserved_envelope::decode_literal_object(*value, limits);
  }
  return object;
}

}  // namespace holoscan::config_io::detail
