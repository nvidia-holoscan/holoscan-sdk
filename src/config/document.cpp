/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <holoscan/config/document.hpp>

#include "config/config_internal.hpp"
#include "config/sha256.hpp"
#include "config/utf8.hpp"
#include "config/yaml_scalar_policy.hpp"

namespace holoscan::config {
namespace {

using detail::ConfigArray;
using detail::ConfigNode;
using detail::ConfigObject;
using detail::valid_utf8;

[[nodiscard]] std::size_t utf8_prefix(std::string_view text, std::size_t capacity) noexcept {
  std::size_t size = std::min(text.size(), capacity);
  while (size > 0 && size < text.size() &&
         (static_cast<unsigned char>(text[size]) & 0xC0U) == 0x80U) {
    --size;
  }
  if (size == 0 || valid_utf8(text.substr(0, size))) {
    return size;
  }
  while (size > 0 && !valid_utf8(text.substr(0, size))) {
    --size;
  }
  return size;
}

template <std::size_t N>
void copy_bounded(std::array<char, N>& target, std::uint16_t& size, bool& truncated,
                  std::string_view source) noexcept {
  const std::size_t copied = utf8_prefix(source, N - 1U);
  if (copied != 0) {
    std::memcpy(target.data(), source.data(), copied);
  }
  target[copied] = '\0';
  size = static_cast<std::uint16_t>(copied);
  truncated = copied != source.size();
}

void append_single_line(std::string& output, std::string_view text) {
  constexpr unsigned char kAsciiDelete = 0x7FU;
  for (const char value : text) {
    const auto byte = static_cast<unsigned char>(value);
    const bool is_control = byte < static_cast<unsigned char>(' ') || byte == kAsciiDelete;
    output.push_back(is_control ? ' ' : value);
  }
}

template <typename ErrorLike>
void append_error_context(std::string& output, const ErrorLike& error) {
  if (!error.message().empty()) {
    output.append(": ");
    append_single_line(output, error.message());
    if (error.message_truncated()) {
      output.append("...");
    }
  }
  if (!error.source().empty()) {
    output.append(" [source: ");
    append_single_line(output, error.source());
    if (error.source_truncated()) {
      output.append("...");
    }
    output.push_back(']');
  }
  if (!error.path().empty()) {
    output.append(" [path: ");
    append_single_line(output, error.path());
    if (error.path_truncated()) {
      output.append("...");
    }
    output.push_back(']');
  }
}

[[nodiscard]] ConfigError type_error(std::string_view message) noexcept {
  return ConfigError::make(ConfigErrorCode::kTypeMismatch, message);
}

[[nodiscard]] ConfigValueKind node_kind(const ConfigNode& node) noexcept {
  return static_cast<ConfigValueKind>(node.storage.index());
}

[[nodiscard]] const ConfigNode::Storage& node_storage(
    const std::shared_ptr<const ConfigNode>& node) noexcept {
  static const ConfigNode::Storage null_storage{std::monostate{}};
  return node ? node->storage : null_storage;
}

[[nodiscard]] ConfigValueKind node_kind(const std::shared_ptr<const ConfigNode>& node) noexcept {
  return static_cast<ConfigValueKind>(node_storage(node).index());
}

[[nodiscard]] bool unsigned_utf8_less(std::string_view left, std::string_view right) noexcept {
  return std::lexicographical_compare(
      left.begin(), left.end(), right.begin(), right.end(), [](char lhs, char rhs) {
        return static_cast<unsigned char>(lhs) < static_cast<unsigned char>(rhs);
      });
}

struct TreeMetrics {
  std::uint64_t nodes{0};
  std::uint32_t depth{0};
};

[[nodiscard]] bool measure_tree(const std::shared_ptr<const ConfigNode>& node, std::uint32_t depth,
                                const ConfigValueLimits& limits, TreeMetrics& metrics) noexcept {
  if (depth == 0 || depth > limits.max_depth || ++metrics.nodes > limits.max_nodes) {
    return false;
  }
  metrics.depth = std::max(metrics.depth, depth);
  const auto& storage = node_storage(node);
  if (const auto* string = std::get_if<std::string>(&storage)) {
    return string->size() <= limits.max_string_bytes;
  }
  if (const auto* array = std::get_if<ConfigArray>(&storage)) {
    if (array->size() > limits.max_container_entries) {
      return false;
    }
    for (const auto& child : *array) {
      if (!measure_tree(child, depth + 1U, limits, metrics)) {
        return false;
      }
    }
  }
  if (const auto* object = std::get_if<ConfigObject>(&storage)) {
    if (object->size() > limits.max_container_entries) {
      return false;
    }
    for (const auto& [key, child] : *object) {
      if (key.size() > limits.max_string_bytes ||
          !measure_tree(child, depth + 1U, limits, metrics)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] ConfigResult<std::size_t> array_index(std::string_view token, std::size_t size,
                                                    bool allow_end = false) {
  if (token.empty() || token == "-" || (token.size() > 1 && token.front() == '0')) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kPathSyntax, "invalid array index"));
  }
  std::uint64_t parsed = 0;
  const auto result = std::from_chars(token.data(), token.data() + token.size(), parsed);
  if (result.ec != std::errc() || result.ptr != token.data() + token.size() ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kPathSyntax, "array index is out of range"));
  }
  const auto index = static_cast<std::size_t>(parsed);
  if (index > size || (!allow_end && index == size)) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kPathNotFound, "array index does not exist"));
  }
  return index;
}

[[nodiscard]] ConfigResult<std::shared_ptr<const ConfigNode>> merge_nodes(
    const std::shared_ptr<const ConfigNode>& baseline,
    const std::shared_ptr<const ConfigNode>& patch) {
  const auto* baseline_object = std::get_if<ConfigObject>(&node_storage(baseline));
  const auto* patch_object = std::get_if<ConfigObject>(&node_storage(patch));
  if (baseline_object == nullptr || patch_object == nullptr) {
    return patch;
  }
  ConfigObject result = *baseline_object;
  for (const auto& [key, patch_value] : *patch_object) {
    auto found = std::find_if(
        result.begin(), result.end(), [&](const auto& item) { return item.first == key; });
    if (found == result.end()) {
      result.emplace_back(key, patch_value);
    } else {
      auto merged = merge_nodes(found->second, patch_value);
      if (!merged) {
        return tl::make_unexpected(merged.error());
      }
      found->second = std::move(*merged);
    }
  }
  return std::make_shared<const ConfigNode>(ConfigNode{std::move(result)});
}

[[nodiscard]] ConfigResult<std::shared_ptr<const ConfigNode>> apply_at(
    const std::shared_ptr<const ConfigNode>& node, const Path& path, std::size_t depth,
    EditKind kind, const std::shared_ptr<const ConfigNode>& value);

[[nodiscard]] ConfigResult<std::shared_ptr<const ConfigNode>> apply_root_edit(
    const std::shared_ptr<const ConfigNode>& node, EditKind kind,
    const std::shared_ptr<const ConfigNode>& value) {
  if (kind == EditKind::kAdd) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kPathConflict, "add target already exists"));
  }
  if (kind == EditKind::kRemove) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kPathConflict, "the document root cannot be removed"));
  }
  if (kind != EditKind::kMergeObject) {
    return value;
  }
  if (node_kind(node) != ConfigValueKind::kObject || node_kind(value) != ConfigValueKind::kObject) {
    return tl::make_unexpected(type_error("object merge requires object values"));
  }
  return merge_nodes(node, value);
}

[[nodiscard]] ConfigResult<std::shared_ptr<const ConfigNode>> apply_object_edit(
    const ConfigObject& object, std::string_view token, const Path& path, std::size_t depth,
    EditKind kind, const std::shared_ptr<const ConfigNode>& value) {
  ConfigObject result = object;
  auto found = std::find_if(
      result.begin(), result.end(), [&](const auto& item) { return item.first == token; });
  const bool leaf = depth + 1U == path.token_count();
  if (found == result.end()) {
    if (kind == EditKind::kAdd && leaf) {
      result.emplace_back(std::string{token}, value);
      return std::make_shared<const ConfigNode>(ConfigNode{std::move(result)});
    }
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kPathNotFound, "object member does not exist"));
  }
  if (leaf && kind == EditKind::kAdd) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kPathConflict, "add target already exists"));
  }
  if (leaf && kind == EditKind::kRemove) {
    result.erase(found);
    return std::make_shared<const ConfigNode>(ConfigNode{std::move(result)});
  }
  auto updated = apply_at(found->second, path, depth + 1U, kind, value);
  if (!updated) {
    return tl::make_unexpected(updated.error());
  }
  found->second = std::move(*updated);
  return std::make_shared<const ConfigNode>(ConfigNode{std::move(result)});
}

[[nodiscard]] ConfigResult<std::shared_ptr<const ConfigNode>> apply_array_edit(
    const ConfigArray& array, std::string_view token, const Path& path, std::size_t depth,
    EditKind kind, const std::shared_ptr<const ConfigNode>& value) {
  ConfigArray result = array;
  const bool leaf = depth + 1U == path.token_count();
  auto index = array_index(token, result.size(), kind == EditKind::kAdd && leaf);
  if (!index) {
    return tl::make_unexpected(index.error());
  }
  if (leaf && kind == EditKind::kAdd) {
    result.insert(result.begin() + static_cast<std::ptrdiff_t>(*index), value);
    return std::make_shared<const ConfigNode>(ConfigNode{std::move(result)});
  }
  if (leaf && kind == EditKind::kRemove) {
    result.erase(result.begin() + static_cast<std::ptrdiff_t>(*index));
    return std::make_shared<const ConfigNode>(ConfigNode{std::move(result)});
  }
  auto updated = apply_at(result[*index], path, depth + 1U, kind, value);
  if (!updated) {
    return tl::make_unexpected(updated.error());
  }
  result[*index] = std::move(*updated);
  return std::make_shared<const ConfigNode>(ConfigNode{std::move(result)});
}

[[nodiscard]] ConfigResult<std::shared_ptr<const ConfigNode>> apply_at(
    const std::shared_ptr<const ConfigNode>& node, const Path& path, std::size_t depth,
    EditKind kind, const std::shared_ptr<const ConfigNode>& value) {
  if (depth == path.token_count()) {
    return apply_root_edit(node, kind, value);
  }

  const std::string_view token = path.token(depth);
  if (const auto* object = std::get_if<ConfigObject>(&node_storage(node))) {
    return apply_object_edit(*object, token, path, depth, kind, value);
  }

  if (const auto* array = std::get_if<ConfigArray>(&node_storage(node))) {
    return apply_array_edit(*array, token, path, depth, kind, value);
  }

  return tl::make_unexpected(
      ConfigError::make(ConfigErrorCode::kPathNotFound, "path traverses a scalar value"));
}

[[nodiscard]] bool overlaps(const Path& left, const Path& right) noexcept {
  const std::size_t common = std::min(left.token_count(), right.token_count());
  for (std::size_t index = 0; index < common; ++index) {
    if (left.token(index) != right.token(index)) {
      return false;
    }
  }
  return true;
}

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
  for (std::size_t index = 0; index < 8U; ++index) {
    output.push_back(static_cast<std::byte>(value >> ((7U - index) * 8U)));
  }
}

[[nodiscard]] ConfigResult<void> ensure_encoding_capacity(const std::vector<std::byte>& output,
                                                          std::size_t maximum, std::size_t count) {
  if (count > maximum - std::min(maximum, output.size())) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "semantic encoding limit exceeded"));
  }
  return {};
}

[[nodiscard]] ConfigResult<void> append_encoding_tag(std::vector<std::byte>& output,
                                                     std::size_t maximum, std::byte tag) {
  auto capacity = ensure_encoding_capacity(output, maximum, 1U);
  if (!capacity) {
    return capacity;
  }
  output.push_back(tag);
  return {};
}

[[nodiscard]] ConfigResult<void> append_tagged_u64(std::vector<std::byte>& output,
                                                   std::size_t maximum, std::byte tag,
                                                   std::uint64_t value) {
  auto capacity = ensure_encoding_capacity(output, maximum, 9U);
  if (!capacity) {
    return capacity;
  }
  output.push_back(tag);
  append_u64(output, value);
  return {};
}

[[nodiscard]] ConfigResult<void> encode_node(const std::shared_ptr<const ConfigNode>& node,
                                             std::vector<std::byte>& output, std::size_t maximum);

[[nodiscard]] ConfigResult<void> encode_string(std::string_view value,
                                               std::vector<std::byte>& output,
                                               std::size_t maximum) {
  auto capacity = ensure_encoding_capacity(output, maximum, 9U + value.size());
  if (!capacity) {
    return capacity;
  }
  output.push_back(std::byte{0x06});
  append_u64(output, value.size());
  output.insert(output.end(),
                reinterpret_cast<const std::byte*>(value.data()),
                reinterpret_cast<const std::byte*>(value.data() + value.size()));
  return {};
}

[[nodiscard]] ConfigResult<void> encode_array(const ConfigArray& values,
                                              std::vector<std::byte>& output, std::size_t maximum) {
  auto capacity = ensure_encoding_capacity(output, maximum, 9U);
  if (!capacity) {
    return capacity;
  }
  output.push_back(std::byte{0x07});
  append_u64(output, values.size());
  for (const auto& value : values) {
    auto encoded = encode_node(value, output, maximum);
    if (!encoded) {
      return encoded;
    }
  }
  return {};
}

[[nodiscard]] ConfigResult<void> encode_object(const ConfigObject& values,
                                               std::vector<std::byte>& output,
                                               std::size_t maximum) {
  std::vector<const ConfigObject::value_type*> sorted;
  sorted.reserve(values.size());
  for (const auto& value : values) {
    sorted.push_back(&value);
  }
  std::sort(sorted.begin(), sorted.end(), [](const auto* left, const auto* right) {
    return unsigned_utf8_less(left->first, right->first);
  });
  auto capacity = ensure_encoding_capacity(output, maximum, 9U);
  if (!capacity) {
    return capacity;
  }
  output.push_back(std::byte{0x08});
  append_u64(output, values.size());
  for (const auto* value : sorted) {
    capacity = ensure_encoding_capacity(output, maximum, 8U + value->first.size());
    if (!capacity) {
      return capacity;
    }
    append_u64(output, value->first.size());
    output.insert(output.end(),
                  reinterpret_cast<const std::byte*>(value->first.data()),
                  reinterpret_cast<const std::byte*>(value->first.data() + value->first.size()));
    auto encoded = encode_node(value->second, output, maximum);
    if (!encoded) {
      return encoded;
    }
  }
  return {};
}

[[nodiscard]] ConfigResult<void> encode_node(const std::shared_ptr<const ConfigNode>& node,
                                             std::vector<std::byte>& output, std::size_t maximum) {
  const auto& storage = node_storage(node);
  switch (node_kind(node)) {
    case ConfigValueKind::kNull:
      return append_encoding_tag(output, maximum, std::byte{0x00});
    case ConfigValueKind::kBool:
      return append_encoding_tag(
          output, maximum, std::get<bool>(storage) ? std::byte{0x02} : std::byte{0x01});
    case ConfigValueKind::kSignedInteger:
      return append_tagged_u64(output,
                               maximum,
                               std::byte{0x03},
                               std::bit_cast<std::uint64_t>(std::get<std::int64_t>(storage)));
    case ConfigValueKind::kUnsignedInteger:
      return append_tagged_u64(output, maximum, std::byte{0x04}, std::get<std::uint64_t>(storage));
    case ConfigValueKind::kFloating: {
      double value = std::get<double>(storage);
      if (value == 0.0) {
        value = 0.0;
      }
      return append_tagged_u64(
          output, maximum, std::byte{0x05}, std::bit_cast<std::uint64_t>(value));
    }
    case ConfigValueKind::kString:
      return encode_string(std::get<std::string>(storage), output, maximum);
    case ConfigValueKind::kArray:
      return encode_array(std::get<ConfigArray>(storage), output, maximum);
    case ConfigValueKind::kObject:
      return encode_object(std::get<ConfigObject>(storage), output, maximum);
  }
  return tl::make_unexpected(
      ConfigError::make(ConfigErrorCode::kUnsupportedValue, "unknown configuration value kind"));
}

[[nodiscard]] ConfigError lossless_edit_error(std::string_view message) {
  return ConfigError::make(ConfigErrorCode::kLosslessPreservationUnavailable, message);
}

void append_json_quoted(std::string& output, std::string_view text) {
  constexpr auto kHex = std::to_array("0123456789abcdef");
  output.push_back('"');
  for (const unsigned char byte : text) {
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
    }
  }
  output.push_back('"');
}

// Each branch preserves an already admitted scalar style; keeping the closed style matrix together
// makes fail-closed lossless replacement behavior auditable.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] ConfigResult<std::string> render_lossless_scalar(const ConfigView& value,
                                                               std::string_view retained_token,
                                                               std::uint8_t source_format) {
  const bool json = source_format == 0U;
  const bool envelope = retained_token.starts_with('{');
  const bool yaml_envelope = !json && envelope;
  std::string result;
  std::string envelope_kind;
  std::string envelope_value;
  switch (value.kind()) {
    case ConfigValueKind::kNull:
      return !json && retained_token == "~" ? std::string{"~"} : std::string{"null"};
    case ConfigValueKind::kBool: {
      auto decoded = value.as_bool();
      return *decoded ? std::string{"true"} : std::string{"false"};
    }
    case ConfigValueKind::kSignedInteger: {
      auto decoded = value.as_signed_integer();
      envelope_kind = "i64";
      envelope_value = std::to_string(*decoded);
      result = envelope_value;
      break;
    }
    case ConfigValueKind::kUnsignedInteger: {
      auto decoded = value.as_unsigned_integer();
      envelope_kind = "u64";
      envelope_value = std::to_string(*decoded);
      result = envelope_value;
      break;
    }
    case ConfigValueKind::kFloating: {
      auto decoded = value.as_floating();
      if (!decoded || !std::isfinite(*decoded)) {
        return tl::make_unexpected(lossless_edit_error(
            "floating replacement cannot be represented by retained scalar style"));
      }
      std::array<char, 768> buffer{};
      const bool exponent_style =
          !yaml_envelope && retained_token.find_first_of("eE") != std::string_view::npos;
      const auto format = yaml_envelope    ? std::chars_format::general
                          : exponent_style ? std::chars_format::scientific
                                           : std::chars_format::fixed;
      const auto encoded =
          std::to_chars(buffer.data(), buffer.data() + buffer.size(), *decoded, format);
      if (encoded.ec != std::errc()) {
        return tl::make_unexpected(lossless_edit_error(
            "floating replacement cannot be represented by retained scalar style"));
      }
      result.assign(buffer.data(), encoded.ptr);
      if (exponent_style && retained_token.find('E') != std::string_view::npos) {
        std::replace(result.begin(), result.end(), 'e', 'E');
      } else if (!yaml_envelope && !exponent_style && result.find('.') == std::string::npos) {
        result += ".0";
      }
      envelope_kind = "f64";
      envelope_value = result;
      break;
    }
    case ConfigValueKind::kString: {
      auto decoded = value.as_string();
      if (json || retained_token.starts_with('"')) {
        append_json_quoted(result, *decoded);
      } else if (retained_token.starts_with('\'')) {
        if (!std::all_of(decoded->begin(), decoded->end(), [](const unsigned char byte) {
              return byte >= 0x20U && byte != 0x7FU && byte != '\n' && byte != '\r';
            })) {
          return tl::make_unexpected(lossless_edit_error(
              "string replacement cannot be represented by retained scalar style"));
        }
        result.push_back('\'');
        for (const char byte : *decoded) {
          result.push_back(byte);
          if (byte == '\'') {
            result.push_back('\'');
          }
        }
        result.push_back('\'');
      } else if (detail::yaml_plain_string_representable(*decoded)) {
        result.assign(*decoded);
      } else {
        return tl::make_unexpected(lossless_edit_error(
            "string replacement cannot be represented by retained scalar style"));
      }
      break;
    }
    case ConfigValueKind::kArray:
    case ConfigValueKind::kObject:
      return tl::make_unexpected(
          lossless_edit_error("lossless replacement requires a scalar leaf"));
  }
  if (envelope) {
    std::string encoded{"{\"$holoscan.config.scalar/v1\":{\"kind\":\""};
    encoded += envelope_kind;
    encoded += "\",\"value\":\"";
    encoded += envelope_value;
    encoded += "\"}}";
    return encoded;
  }
  return result;
}

struct LosslessReplacement {
  std::size_t span_index{0};
  std::size_t begin{0};
  std::size_t end{0};
  std::string bytes;
  std::size_t installed_begin{0};
  std::size_t installed_end{0};
};

[[nodiscard]] ConfigResult<void> validate_nonoverlapping_edits(std::span<const Edit> edits) {
  for (std::size_t left = 0; left < edits.size(); ++left) {
    for (std::size_t right = left + 1U; right < edits.size(); ++right) {
      if (overlaps(edits[left].path(), edits[right].path())) {
        return tl::make_unexpected(ConfigError::make(
            ConfigErrorCode::kPathConflict, "transaction contains duplicate or overlapping paths"));
      }
    }
  }
  return {};
}

[[nodiscard]] ConfigResult<LosslessReplacement> prepare_lossless_replacement(
    const Document& document, const detail::DocumentState& state, const Edit& edit) {
  const auto path = edit.path().to_string();
  if (edit.kind() != EditKind::kReplace || path.empty()) {
    return tl::make_unexpected(lossless_edit_error(
        "lossless edits support only replacement of an existing non-root scalar leaf"));
  }
  auto previous = document.at(edit.path());
  if (!previous) {
    return tl::make_unexpected(previous.error());
  }
  const auto replacement = edit.value().view();
  if (previous->kind() != replacement.kind() || replacement.kind() == ConfigValueKind::kArray ||
      replacement.kind() == ConfigValueKind::kObject) {
    return tl::make_unexpected(
        lossless_edit_error("lossless replacement must retain the existing scalar kind"));
  }
  const auto span = std::find_if(
      state.scalar_spans.begin(),
      state.scalar_spans.end(),
      [&](const detail::LosslessScalarSpan& candidate) { return candidate.path == path; });
  if (span == state.scalar_spans.end() || span->kind != replacement.kind() ||
      span->begin > span->end || span->end > state.raw_source->size()) {
    return tl::make_unexpected(
        lossless_edit_error("lossless scalar path or source span is unstable"));
  }
  const auto token =
      std::string_view{*state.raw_source}.substr(span->begin, span->end - span->begin);
  auto rendered = render_lossless_scalar(replacement, token, *state.source_format);
  if (!rendered) {
    return tl::make_unexpected(rendered.error());
  }
  return LosslessReplacement{static_cast<std::size_t>(span - state.scalar_spans.begin()),
                             span->begin,
                             span->end,
                             std::move(*rendered)};
}

[[nodiscard]] ConfigResult<std::vector<LosslessReplacement>> prepare_lossless_replacements(
    const Document& document, const detail::DocumentState& state, std::span<const Edit> edits) {
  std::vector<LosslessReplacement> replacements;
  if (!state.requires_lossless || edits.empty()) {
    return replacements;
  }
  if (!state.lossless_available || !state.raw_source || !state.source_format) {
    return tl::make_unexpected(lossless_edit_error("lossless edit sidecar is unavailable"));
  }
  replacements.reserve(edits.size());
  for (const auto& edit : edits) {
    auto replacement = prepare_lossless_replacement(document, state, edit);
    if (!replacement) {
      return tl::make_unexpected(replacement.error());
    }
    replacements.push_back(std::move(*replacement));
  }
  std::sort(replacements.begin(), replacements.end(), [](const auto& left, const auto& right) {
    return left.begin < right.begin;
  });
  for (std::size_t index = 1; index < replacements.size(); ++index) {
    if (replacements[index - 1U].end > replacements[index].begin) {
      return tl::make_unexpected(lossless_edit_error("lossless scalar source spans overlap"));
    }
  }
  return replacements;
}

[[nodiscard]] ConfigResult<std::size_t> lossless_output_size(
    std::size_t initial_size, const std::vector<LosslessReplacement>& replacements) {
  constexpr std::size_t kMaximumRetainedBytes = 64U * 1024U * 1024U;
  std::size_t output_size = initial_size;
  for (const auto& replacement : replacements) {
    const std::size_t removed = replacement.end - replacement.begin;
    if (replacement.bytes.size() > kMaximumRetainedBytes - (output_size - removed)) {
      return tl::make_unexpected(ConfigError::make(
          ConfigErrorCode::kLimitExceeded, "lossless edited source exceeds package byte limit"));
    }
    output_size = output_size - removed + replacement.bytes.size();
  }
  return output_size;
}

void update_lossless_span(detail::LosslessScalarSpan& span, std::size_t span_index,
                          const std::vector<LosslessReplacement>& replacements) {
  std::int64_t shift = 0;
  for (const auto& replacement : replacements) {
    if (replacement.span_index == span_index) {
      span.begin = replacement.installed_begin;
      span.end = replacement.installed_end;
      return;
    }
    if (replacement.end <= span.begin) {
      shift += static_cast<std::int64_t>(replacement.bytes.size()) -
               static_cast<std::int64_t>(replacement.end - replacement.begin);
    }
  }
  if (shift != 0) {
    span.begin = static_cast<std::size_t>(static_cast<std::int64_t>(span.begin) + shift);
    span.end = static_cast<std::size_t>(static_cast<std::int64_t>(span.end) + shift);
  }
}

[[nodiscard]] ConfigResult<void> rewrite_lossless_source(
    detail::DocumentState& state, std::vector<LosslessReplacement>& replacements) {
  auto output_size = lossless_output_size(state.raw_source->size(), replacements);
  if (!output_size) {
    return tl::make_unexpected(output_size.error());
  }
  std::string rewritten;
  rewritten.reserve(*output_size);
  std::size_t cursor = 0;
  for (auto& replacement : replacements) {
    rewritten.append(*state.raw_source, cursor, replacement.begin - cursor);
    replacement.installed_begin = rewritten.size();
    rewritten += replacement.bytes;
    replacement.installed_end = rewritten.size();
    cursor = replacement.end;
  }
  rewritten.append(*state.raw_source, cursor, state.raw_source->size() - cursor);

  for (std::size_t span_index = 0; span_index < state.scalar_spans.size(); ++span_index) {
    update_lossless_span(state.scalar_spans[span_index], span_index, replacements);
  }
  state.raw_source = std::make_shared<const std::string>(std::move(rewritten));
  state.lossless_available = true;
  return {};
}

}  // namespace

std::string_view config_error_token(ConfigErrorCode code) noexcept {
  static constexpr std::array<std::string_view, 22> kTokens{
      "CONFIG_INVALID_ARGUMENT",
      "CONFIG_PARSE_ERROR",
      "CONFIG_UNSUPPORTED_FORMAT",
      "CONFIG_UNSUPPORTED_VALUE",
      "CONFIG_INVALID_UTF8",
      "CONFIG_DUPLICATE_KEY",
      "CONFIG_PATH_SYNTAX",
      "CONFIG_PATH_NOT_FOUND",
      "CONFIG_PATH_CONFLICT",
      "CONFIG_TYPE_MISMATCH",
      "CONFIG_OUT_OF_RANGE",
      "CONFIG_LIMIT_EXCEEDED",
      "CONFIG_LOSSLESS_PRESERVATION_UNAVAILABLE",
      "CONFIG_SOURCE_CHANGED",
      "CONFIG_GUARD_BUSY",
      "CONFIG_CODEC_MISSING",
      "CONFIG_CODEC_FAILURE",
      "CONFIG_IO_ERROR",
      "CONFIG_PERMISSION_DENIED",
      "CONFIG_ATOMIC_REPLACE_UNSUPPORTED",
      "CONFIG_DIGEST_MISMATCH",
      "CONFIG_UNSUPPORTED_VERSION"};
  const auto index = static_cast<std::size_t>(code);
  return index < kTokens.size() ? kTokens[index] : std::string_view{"CONFIG_INVALID_ARGUMENT"};
}

ConfigError ConfigError::make(ConfigErrorCode code, std::string_view message, std::string_view path,
                              std::string_view source, std::span<const std::byte> excerpt,
                              ConfigOperationStage stage,
                              ConfigDestinationState destination) noexcept {
  ConfigError result;
  result.code_ = code;
  result.stage_ = stage;
  result.destination_ = destination;
  copy_bounded(result.message_, result.message_size_, result.message_truncated_, message);
  copy_bounded(result.path_, result.path_size_, result.path_truncated_, path);
  copy_bounded(result.source_, result.source_size_, result.source_truncated_, source);
  const std::size_t excerpt_size = std::min(excerpt.size(), kMaxExcerptBytes);
  std::copy_n(excerpt.begin(), excerpt_size, result.excerpt_.begin());
  result.excerpt_size_ = static_cast<std::uint16_t>(excerpt_size);
  result.excerpt_truncated_ = excerpt_size != excerpt.size();
  return result;
}

ConfigError ConfigError::nest(ConfigError root, const ConfigError& cause) noexcept {
  auto copy_cause = [](ConfigErrorCause& target, const ConfigError& source) {
    target.code_ = source.code_;
    target.stage_ = source.stage_;
    target.destination_ = source.destination_;
    target.message_ = source.message_;
    target.path_ = source.path_;
    target.source_ = source.source_;
    target.excerpt_ = source.excerpt_;
    target.message_size_ = source.message_size_;
    target.path_size_ = source.path_size_;
    target.source_size_ = source.source_size_;
    target.excerpt_size_ = source.excerpt_size_;
    target.message_truncated_ = source.message_truncated_;
    target.path_truncated_ = source.path_truncated_;
    target.source_truncated_ = source.source_truncated_;
    target.excerpt_truncated_ = source.excerpt_truncated_;
    target.cause_overflow_ = source.cause_overflow_;
  };

  root.cause_count_ = 1;
  copy_cause(root.causes_[0], cause);
  for (std::size_t index = 0;
       index < cause.cause_count_ && root.cause_count_ < kMaxNestedCauseDepth;
       ++index) {
    root.causes_[root.cause_count_++] = cause.causes_[index];
  }
  if (cause.cause_overflow_ || cause.cause_count_ + 1U > kMaxNestedCauseDepth) {
    root.cause_overflow_ = true;
    root.causes_[root.cause_count_ - 1U].cause_overflow_ = true;
  }
  return root;
}

std::string_view ConfigError::message() const noexcept {
  return {message_.data(), message_size_};
}
std::string_view ConfigError::path() const noexcept {
  return {path_.data(), path_size_};
}
std::string_view ConfigError::source() const noexcept {
  return {source_.data(), source_size_};
}
std::span<const std::byte> ConfigError::excerpt() const noexcept {
  return {excerpt_.data(), excerpt_size_};
}

std::string_view ConfigErrorCause::message() const noexcept {
  return {message_.data(), message_size_};
}
std::string_view ConfigErrorCause::path() const noexcept {
  return {path_.data(), path_size_};
}
std::string_view ConfigErrorCause::source() const noexcept {
  return {source_.data(), source_size_};
}
std::span<const std::byte> ConfigErrorCause::excerpt() const noexcept {
  return {excerpt_.data(), excerpt_size_};
}

std::string format_error(const ConfigError& error) {
  std::string result{error.token()};
  append_error_context(result, error);
  for (std::size_t index = 0; index < error.cause_count(); ++index) {
    const ConfigErrorCause* cause = error.cause(index);
    if (cause == nullptr) {
      continue;
    }
    result.append("; caused by: ");
    result.append(cause->token());
    append_error_context(result, *cause);
  }
  return result;
}

ConfigValue::ConfigValue() noexcept = default;
ConfigValue::ConfigValue(std::shared_ptr<const ConfigNode> node) noexcept
    : node_(std::move(node)) {}

ConfigValue ConfigValue::null() noexcept {
  return {};
}
ConfigResult<ConfigValue> ConfigValue::boolean(bool value) noexcept {
  try {
    return ConfigValue{std::make_shared<const ConfigNode>(ConfigNode{value})};
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "configuration allocation failed"));
  }
}
ConfigResult<ConfigValue> ConfigValue::signed_integer(std::int64_t value) noexcept {
  try {
    return ConfigValue{std::make_shared<const ConfigNode>(ConfigNode{value})};
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "configuration allocation failed"));
  }
}
ConfigResult<ConfigValue> ConfigValue::unsigned_integer(std::uint64_t value) noexcept {
  try {
    return ConfigValue{std::make_shared<const ConfigNode>(ConfigNode{value})};
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "configuration allocation failed"));
  }
}
ConfigResult<ConfigValue> ConfigValue::floating(double value) noexcept {
  if (!std::isfinite(value)) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kUnsupportedValue, "floating value must be finite"));
  }
  if (value == 0.0) {
    value = 0.0;
  }
  try {
    return ConfigValue{std::make_shared<const ConfigNode>(ConfigNode{value})};
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "configuration allocation failed"));
  }
}
ConfigResult<ConfigValue> ConfigValue::string(std::string_view value, ConfigValueLimits limits) {
  if (!valid_utf8(value)) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kInvalidUtf8, "configuration string is not UTF-8"));
  }
  if (value.size() > limits.max_string_bytes || limits.max_depth == 0 || limits.max_nodes == 0 ||
      limits.max_container_entries == 0) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "configuration string limit exceeded"));
  }
  try {
    return ConfigValue{std::make_shared<const ConfigNode>(ConfigNode{std::string{value}})};
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "configuration allocation failed"));
  }
}
ConfigResult<ConfigValue> ConfigValue::array(std::span<const ConfigValue> values,
                                             ConfigValueLimits limits) {
  if (values.size() > limits.max_container_entries || limits.max_depth == 0 ||
      limits.max_nodes == 0 || limits.max_string_bytes == 0 || limits.max_container_entries == 0) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "configuration array limit exceeded"));
  }
  try {
    ConfigArray result;
    result.reserve(values.size());
    for (const auto& value : values) {
      result.push_back(value.node_);
    }
    auto node = std::make_shared<const ConfigNode>(ConfigNode{std::move(result)});
    TreeMetrics metrics;
    if (!measure_tree(node, 1U, limits, metrics)) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kLimitExceeded, "configuration tree limit exceeded"));
    }
    return ConfigValue{std::move(node)};
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "configuration allocation failed"));
  }
}
ConfigResult<ConfigValue> ConfigValue::object(std::span<const ConfigObjectEntry> entries,
                                              ConfigValueLimits limits) {
  if (entries.size() > limits.max_container_entries || limits.max_depth == 0 ||
      limits.max_nodes == 0 || limits.max_string_bytes == 0 || limits.max_container_entries == 0) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "configuration object limit exceeded"));
  }
  try {
    ConfigObject result;
    result.reserve(entries.size());
    std::unordered_set<std::string_view> keys;
    keys.reserve(entries.size());
    for (const auto& entry : entries) {
      if (!valid_utf8(entry.key)) {
        return tl::make_unexpected(
            ConfigError::make(ConfigErrorCode::kInvalidUtf8, "object key is not UTF-8"));
      }
      if (entry.key.size() > limits.max_string_bytes) {
        return tl::make_unexpected(
            ConfigError::make(ConfigErrorCode::kLimitExceeded, "object key limit exceeded"));
      }
      if (!keys.insert(entry.key).second) {
        return tl::make_unexpected(
            ConfigError::make(ConfigErrorCode::kDuplicateKey, "duplicate object key"));
      }
      result.emplace_back(entry.key, entry.value.node_);
    }
    auto node = std::make_shared<const ConfigNode>(ConfigNode{std::move(result)});
    TreeMetrics metrics;
    if (!measure_tree(node, 1U, limits, metrics)) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kLimitExceeded, "configuration tree limit exceeded"));
    }
    return ConfigValue{std::move(node)};
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "configuration allocation failed"));
  }
}

ConfigValueKind ConfigValue::kind() const noexcept {
  return node_kind(node_);
}
ConfigView ConfigValue::view() const noexcept {
  return ConfigView{node_};
}

ConfigValueKind ConfigView::kind() const noexcept {
  return valid_ ? node_kind(node_) : ConfigValueKind::kNull;
}
std::size_t ConfigView::size() const noexcept {
  if (!valid_) {
    return 0;
  }
  const auto& storage = node_storage(node_);
  if (const auto* array = std::get_if<ConfigArray>(&storage)) {
    return array->size();
  }
  if (const auto* object = std::get_if<ConfigObject>(&storage)) {
    return object->size();
  }
  return 0;
}
ConfigResult<bool> ConfigView::as_bool() const noexcept {
  if (valid_) {
    if (const auto* value = std::get_if<bool>(&node_storage(node_))) {
      return *value;
    }
  }
  return tl::make_unexpected(type_error("configuration value is not bool"));
}
ConfigResult<std::int64_t> ConfigView::as_signed_integer() const noexcept {
  if (valid_) {
    if (const auto* value = std::get_if<std::int64_t>(&node_storage(node_))) {
      return *value;
    }
  }
  return tl::make_unexpected(type_error("configuration value is not signed integer"));
}
ConfigResult<std::uint64_t> ConfigView::as_unsigned_integer() const noexcept {
  if (valid_) {
    if (const auto* value = std::get_if<std::uint64_t>(&node_storage(node_))) {
      return *value;
    }
  }
  return tl::make_unexpected(type_error("configuration value is not unsigned integer"));
}
ConfigResult<double> ConfigView::as_floating() const noexcept {
  if (valid_) {
    if (const auto* value = std::get_if<double>(&node_storage(node_))) {
      return *value;
    }
  }
  return tl::make_unexpected(type_error("configuration value is not floating point"));
}
ConfigResult<std::string_view> ConfigView::as_string() const noexcept {
  if (valid_) {
    if (const auto* value = std::get_if<std::string>(&node_storage(node_))) {
      return *value;
    }
  }
  return tl::make_unexpected(type_error("configuration value is not string"));
}
ConfigResult<ConfigView> ConfigView::element(std::size_t index) const noexcept {
  if (!valid_) {
    return tl::make_unexpected(type_error("invalid configuration view"));
  }
  const auto& storage = node_storage(node_);
  if (const auto* array = std::get_if<ConfigArray>(&storage)) {
    if (index >= array->size()) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kPathNotFound, "array index does not exist"));
    }
    return ConfigView{(*array)[index]};
  }
  if (const auto* object = std::get_if<ConfigObject>(&storage)) {
    if (index >= object->size()) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kPathNotFound, "object index does not exist"));
    }
    return ConfigView{(*object)[index].second};
  }
  return tl::make_unexpected(type_error("configuration value is not a container"));
}
ConfigResult<std::string_view> ConfigView::key(std::size_t index) const noexcept {
  if (valid_) {
    if (const auto* object = std::get_if<ConfigObject>(&node_storage(node_))) {
      if (index < object->size()) {
        return (*object)[index].first;
      }
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kPathNotFound, "object index does not exist"));
    }
  }
  return tl::make_unexpected(type_error("configuration value is not an object"));
}
ConfigResult<ConfigView> ConfigView::member(std::string_view key) const noexcept {
  auto found = find_member(key);
  if (!found) {
    return tl::make_unexpected(found.error());
  }
  auto value = found->value_or(ConfigView{});
  if (!value.valid()) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kPathNotFound, "object member does not exist"));
  }
  return value;
}
ConfigResult<std::optional<ConfigView>> ConfigView::find_member(
    std::string_view key) const noexcept {
  if (!valid_) {
    return tl::make_unexpected(type_error("invalid configuration view"));
  }
  const auto* object = std::get_if<ConfigObject>(&node_storage(node_));
  if (object == nullptr) {
    return tl::make_unexpected(type_error("configuration value is not an object"));
  }
  const auto found = std::find_if(
      object->begin(), object->end(), [&](const auto& item) { return item.first == key; });
  return found == object->end() ? std::optional<ConfigView>{}
                                : std::optional<ConfigView>{ConfigView{found->second}};
}
ConfigValue ConfigView::to_value() const noexcept {
  return ConfigValue{node_};
}

ConfigResult<Path> Path::parse(std::string_view text) {
  Path result;
  if (text.empty()) {
    return result;
  }
  if (text.front() != '/') {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kPathSyntax, "non-root path must begin with slash"));
  }
  std::size_t begin = 1;
  while (begin <= text.size()) {
    const std::size_t end = text.find('/', begin);
    const std::string_view encoded = text.substr(begin, end - begin);
    std::string decoded;
    decoded.reserve(encoded.size());
    for (std::size_t index = 0; index < encoded.size(); ++index) {
      if (encoded[index] != '~') {
        decoded.push_back(encoded[index]);
      } else if (index + 1U < encoded.size() && encoded[index + 1U] == '0') {
        decoded.push_back('~');
        ++index;
      } else if (index + 1U < encoded.size() && encoded[index + 1U] == '1') {
        decoded.push_back('/');
        ++index;
      } else {
        return tl::make_unexpected(
            ConfigError::make(ConfigErrorCode::kPathSyntax, "invalid JSON Pointer escape"));
      }
    }
    if (!valid_utf8(decoded)) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kInvalidUtf8, "path token is not UTF-8"));
    }
    result.tokens_.push_back(std::move(decoded));
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
  }
  return result;
}
Path Path::root() noexcept {
  return {};
}
std::string_view Path::token(std::size_t index) const noexcept {
  return index < tokens_.size() ? std::string_view{tokens_[index]} : std::string_view{};
}
std::string Path::to_string() const {
  std::string result;
  for (const auto& token : tokens_) {
    result.push_back('/');
    for (const char value : token) {
      if (value == '~') {
        result += "~0";
      } else if (value == '/') {
        result += "~1";
      } else {
        result.push_back(value);
      }
    }
  }
  return result;
}

// The public value-semantic API deliberately takes a cheap owning view by value.
// NOLINTNEXTLINE(performance-unnecessary-value-param)
ConfigResult<std::vector<std::byte>> encode_semantic_value_v1(ConfigView value,
                                                              SemanticEncodeOptions options) {
  if (!value.valid() || options.max_output_bytes == 0) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kInvalidArgument, "invalid semantic encoding input"));
  }
  std::vector<std::byte> result;
  auto encoded = encode_node(value.node_, result, options.max_output_bytes);
  if (!encoded) {
    return tl::make_unexpected(encoded.error());
  }
  return result;
}

Edit::Edit(EditKind kind, Path path, ConfigValue value) noexcept
    : kind_(kind), path_(std::move(path)), value_(std::move(value)) {}
Edit Edit::add(Path path, ConfigValue value) {
  return {EditKind::kAdd, std::move(path), std::move(value)};
}
Edit Edit::replace(Path path, ConfigValue value) {
  return {EditKind::kReplace, std::move(path), std::move(value)};
}
Edit Edit::remove(Path path) {
  return {EditKind::kRemove, std::move(path), ConfigValue::null()};
}
Edit Edit::merge_object(Path path, ConfigValue object) {
  return {EditKind::kMergeObject, std::move(path), std::move(object)};
}

Document::Document(std::shared_ptr<const detail::DocumentState> state) noexcept
    : state_(std::move(state)) {}
ConfigResult<Document> Document::from_value(ConfigValue root) noexcept {
  try {
    auto state = std::make_shared<detail::DocumentState>();
    state->root = std::move(root.node_);
    auto digest = detail::semantic_digest(state->root);
    if (!digest) {
      return tl::make_unexpected(digest.error());
    }
    state->digest.semantic = *digest;
    return Document{std::move(state)};
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "document allocation failed"));
  }
}
ConfigView Document::root() const noexcept {
  return ConfigView{state_->root};
}
ConfigResult<ConfigView> Document::at(const Path& path) const {
  std::shared_ptr<const ConfigNode> current = state_->root;
  for (std::size_t depth = 0; depth < path.token_count(); ++depth) {
    const auto token = path.token(depth);
    const auto& storage = node_storage(current);
    if (const auto* object = std::get_if<ConfigObject>(&storage)) {
      const auto found = std::find_if(
          object->begin(), object->end(), [&](const auto& item) { return item.first == token; });
      if (found == object->end()) {
        return tl::make_unexpected(ConfigError::make(
            ConfigErrorCode::kPathNotFound, "object member does not exist", path.to_string()));
      }
      current = found->second;
    } else if (const auto* array = std::get_if<ConfigArray>(&storage)) {
      auto index = array_index(token, array->size());
      if (!index) {
        return tl::make_unexpected(index.error());
      }
      current = (*array)[*index];
    } else {
      return tl::make_unexpected(ConfigError::make(
          ConfigErrorCode::kPathNotFound, "path traverses a scalar value", path.to_string()));
    }
  }
  return ConfigView{std::move(current)};
}
ConfigResult<ConfigView> Document::at(std::string_view path) const {
  auto parsed = Path::parse(path);
  if (!parsed) {
    return tl::make_unexpected(parsed.error());
  }
  return at(*parsed);
}
ConfigResult<std::optional<ConfigView>> Document::find(const Path& path) const {
  auto value = at(path);
  if (!value && value.error().code() == ConfigErrorCode::kPathNotFound) {
    return std::optional<ConfigView>{};
  }
  if (!value) {
    return tl::make_unexpected(value.error());
  }
  return std::optional<ConfigView>{*value};
}
ConfigResult<std::optional<ConfigView>> Document::find(std::string_view path) const {
  auto parsed = Path::parse(path);
  if (!parsed) {
    return tl::make_unexpected(parsed.error());
  }
  return find(*parsed);
}
ConfigResult<Document> Document::patched(std::span<const Edit> edits) const {
  try {
    if (!edits.empty() && state_->requires_lossless && state_->contains_yaml_alias) {
      return tl::make_unexpected(
          lossless_edit_error("lossless edits of alias-bearing YAML are unavailable"));
    }
    auto paths_valid = validate_nonoverlapping_edits(edits);
    if (!paths_valid) {
      return tl::make_unexpected(paths_valid.error());
    }
    auto prepared_replacements = prepare_lossless_replacements(*this, *state_, edits);
    if (!prepared_replacements) {
      return tl::make_unexpected(prepared_replacements.error());
    }
    auto replacements = std::move(*prepared_replacements);

    auto root = state_->root;
    for (const auto& edit : edits) {
      auto updated = apply_at(root, edit.path(), 0U, edit.kind(), edit.value().node_);
      if (!updated) {
        return tl::make_unexpected(updated.error());
      }
      root = std::move(*updated);
    }
    TreeMetrics metrics;
    if (!measure_tree(root, 1U, ConfigValueLimits{}, metrics)) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kLimitExceeded, "edited tree limit exceeded"));
    }
    auto state = std::make_shared<detail::DocumentState>(*state_);
    state->root = std::move(root);
    auto semantic = detail::semantic_digest(state->root);
    if (!semantic) {
      return tl::make_unexpected(semantic.error());
    }
    state->digest.semantic = *semantic;

    if (!replacements.empty()) {
      auto rewritten = rewrite_lossless_source(*state, replacements);
      if (!rewritten) {
        return tl::make_unexpected(rewritten.error());
      }
    } else {
      state->lossless_available = state_->lossless_available;
    }
    return Document{std::move(state)};
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "document edit allocation failed"));
  }
}
ConfigDigest Document::digest() const noexcept {
  return state_->digest;
}

ConfigResult<ConfigValue> merge_config_values(const ConfigValue& baseline,
                                              const ConfigValue& patch) {
  try {
    auto merged = merge_nodes(baseline.node_, patch.node_);
    if (!merged) {
      return tl::make_unexpected(merged.error());
    }
    TreeMetrics metrics;
    if (!measure_tree(*merged, 1U, ConfigValueLimits{}, metrics)) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kLimitExceeded, "merged tree limit exceeded"));
    }
    return ConfigValue{std::move(*merged)};
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "object merge allocation failed"));
  }
}

ConfigResult<Document> ConfigIoAccess::from_parsed(ConfigValue root, RawSourceDigest raw_digest,
                                                   std::uint8_t source_format,
                                                   std::shared_ptr<const std::string> raw_source,
                                                   std::vector<detail::LosslessScalarSpan> spans,
                                                   bool requires_lossless) noexcept {
  try {
    auto state = std::make_shared<detail::DocumentState>();
    state->root = std::move(root.node_);
    auto semantic = detail::semantic_digest(state->root);
    if (!semantic) {
      return tl::make_unexpected(semantic.error());
    }
    state->digest.semantic = *semantic;
    state->digest.source = raw_digest;
    state->source_format = source_format & detail::kSourceFormatValueMask;
    state->raw_source = std::move(raw_source);
    state->scalar_spans = std::move(spans);
    state->requires_lossless = requires_lossless;
    state->lossless_available = requires_lossless;
    state->contains_yaml_alias = (source_format & detail::kYamlAliasSourceFlag) != 0U;
    return Document{std::move(state)};
  } catch (const std::bad_alloc&) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "document allocation failed"));
  }
}
const detail::DocumentState& ConfigIoAccess::state(const Document& document) noexcept {
  return *document.state_;
}

namespace detail {
ConfigResult<SemanticDigest> semantic_digest(
    const std::shared_ptr<const ConfigNode>& root) noexcept {
  try {
    std::vector<std::byte> encoded;
    auto result = encode_node(root, encoded, 64U * 1024U * 1024U);
    if (!result) {
      return tl::make_unexpected(result.error());
    }
    Sha256Builder builder;
    constexpr std::string_view domain{"holoscan.config.semantic/v1"};
    builder.add({reinterpret_cast<const std::byte*>(domain.data()), domain.size()});
    const std::byte nul{0};
    builder.add({&nul, 1U});
    builder.add(encoded);
    return SemanticDigest{.canonical_version = 1, .sha256 = builder.finish()};
  } catch (...) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kLimitExceeded, "semantic digest allocation failed"));
  }
}
ConfigResult<RawSourceDigest> raw_digest(std::string_view bytes) noexcept {
  Sha256Builder builder;
  builder.add({reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()});
  return RawSourceDigest{.algorithm_version = 1,
                         .sha256 = builder.finish(),
                         .byte_size = static_cast<std::uint64_t>(bytes.size())};
}
}  // namespace detail

}  // namespace holoscan::config
