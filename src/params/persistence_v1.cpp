/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <holoscan/params/persistence.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <holoscan/config/document.hpp>
#include <holoscan/config/io.hpp>

#include "config/io_internal.hpp"
#include "core/canonical_text_encoding.hpp"
#include "core/parameter_snapshot_archive_access.hpp"

namespace holoscan::params {
namespace {

constexpr std::string_view kArchiveFormat = "holoscan.param-snapshot";
constexpr std::uint64_t kArchiveVersion = 1U;
constexpr std::string_view kDigestDomain = "holoscan.param-snapshot/v1";
constexpr std::size_t kDigestBytes = 32U;
constexpr std::size_t kMaxArchiveRecords = 1U << 20U;

[[nodiscard]] ParamArchiveError archive_error(ParamArchiveErrorCode code, std::string_view token,
                                              std::optional<Error> core = std::nullopt) {
  return ParamArchiveError{
      code, ParamToolError{ParamToolErrorCode::kPersistence, std::string{token}, std::move(core)}};
}

[[nodiscard]] ParamArchiveError archive_config_error(ParamArchiveErrorCode code,
                                                     std::string_view token,
                                                     config::ConfigError error) {
  return ParamArchiveError{code, from_config_error(std::move(error), token)};
}

[[nodiscard]] ParamArchiveErrorCode archive_load_error_code(config::ConfigErrorCode code) noexcept {
  switch (code) {
    case config::ConfigErrorCode::kIoError:
    case config::ConfigErrorCode::kPermissionDenied:
    case config::ConfigErrorCode::kPathNotFound:
    case config::ConfigErrorCode::kAtomicReplaceUnsupported:
    case config::ConfigErrorCode::kSourceChanged:
    case config::ConfigErrorCode::kGuardBusy:
      return ParamArchiveErrorCode::kPersistenceFailure;
    default:
      return ParamArchiveErrorCode::kInvalidArchive;
  }
}

[[nodiscard]] ParamRebindError persistence_rebind_error(const ParamArchiveError& error) {
  return ParamRebindError{
      ParamRebindErrorCode::kInvalidSnapshot,
      ParamRebindStage::kSourceValidation,
      ParamToolError{ParamToolErrorCode::kPersistence, std::string{error.detail().token()}}};
}

struct NodeObjectEntry;

struct Node {
  enum class Kind : std::uint8_t {
    kNull,
    kBool,
    kUnsigned,
    kSigned,
    kFloating,
    kString,
    kArray,
    kObject,
  };

  Kind kind{Kind::kNull};
  bool scalar_escape{};
  bool boolean{};
  std::uint64_t unsigned_integer{};
  std::int64_t signed_integer{};
  double floating{};
  std::string text;
  std::vector<Node> array;
  std::vector<std::string> object_keys;
  std::vector<Node> object_values;

  [[nodiscard]] static Node null() { return {}; }
  [[nodiscard]] static Node boolean_value(bool value) {
    Node result;
    result.kind = Kind::kBool;
    result.boolean = value;
    return result;
  }
  [[nodiscard]] static Node structural(std::uint64_t value) {
    Node result;
    result.kind = Kind::kUnsigned;
    result.unsigned_integer = value;
    return result;
  }
  [[nodiscard]] static Node signed_scalar(std::int64_t value) {
    Node result;
    result.kind = Kind::kSigned;
    result.scalar_escape = true;
    result.signed_integer = value;
    return result;
  }
  [[nodiscard]] static Node unsigned_scalar(std::uint64_t value) {
    Node result;
    result.kind = Kind::kUnsigned;
    result.scalar_escape = true;
    result.unsigned_integer = value;
    return result;
  }
  [[nodiscard]] static Node floating_scalar(double value) {
    Node result;
    result.kind = Kind::kFloating;
    result.scalar_escape = true;
    result.floating = value == 0.0 ? 0.0 : value;
    return result;
  }
  [[nodiscard]] static Node string(std::string value) {
    Node result;
    result.kind = Kind::kString;
    result.text = std::move(value);
    return result;
  }
  [[nodiscard]] static Node array_value(std::vector<Node> value) {
    Node result;
    result.kind = Kind::kArray;
    result.array = std::move(value);
    return result;
  }
  [[nodiscard]] static Node object_value(std::vector<std::pair<std::string, Node>> value);
};

Node Node::object_value(std::vector<std::pair<std::string, Node>> value) {
  Node result;
  result.kind = Kind::kObject;
  result.object_keys.reserve(value.size());
  result.object_values.reserve(value.size());
  for (auto& [key, child] : value) {
    result.object_keys.push_back(std::move(key));
    result.object_values.push_back(std::move(child));
  }
  return result;
}

[[nodiscard]] ParamArchiveResult<config::ConfigValue> checked_config_value(
    config::ConfigResult<config::ConfigValue> value, std::string_view token) {
  if (!value) {
    return make_unexpected(archive_config_error(
        ParamArchiveErrorCode::kInvalidArchive, token, std::move(value.error())));
  }
  return std::move(*value);
}

[[nodiscard]] ParamArchiveResult<config::ConfigValue> to_config_value(const Node& node);

[[nodiscard]] ParamArchiveResult<config::ConfigValue> to_config_array(const Node& node) {
  std::vector<config::ConfigValue> values;
  values.reserve(node.array.size());
  for (const auto& child : node.array) {
    auto value = to_config_value(child);
    if (!value) {
      return make_unexpected(std::move(value.error()));
    }
    values.push_back(std::move(*value));
  }
  return checked_config_value(config::ConfigValue::try_array(values),
                              "PARAM_ARCHIVE_ARRAY_INVALID");
}

[[nodiscard]] ParamArchiveResult<config::ConfigValue> to_config_object(const Node& node) {
  std::vector<config::ConfigValue> values;
  values.reserve(node.object_values.size());
  for (const auto& child : node.object_values) {
    auto value = to_config_value(child);
    if (!value) {
      return make_unexpected(std::move(value.error()));
    }
    values.push_back(std::move(*value));
  }
  std::vector<config::ConfigObjectEntry> entries;
  entries.reserve(node.object_values.size());
  for (std::size_t index = 0; index < node.object_values.size(); ++index) {
    entries.push_back({node.object_keys[index], values[index]});
  }
  return checked_config_value(config::ConfigValue::try_object(entries),
                              "PARAM_ARCHIVE_OBJECT_INVALID");
}

[[nodiscard]] ParamArchiveResult<config::ConfigValue> to_config_value(const Node& node) {
  switch (node.kind) {
    case Node::Kind::kNull:
      return config::ConfigValue::null();
    case Node::Kind::kBool:
      return checked_config_value(config::ConfigValue::try_boolean(node.boolean),
                                  "PARAM_ARCHIVE_VALUE_INVALID");
    case Node::Kind::kUnsigned:
      return checked_config_value(config::ConfigValue::try_unsigned_integer(node.unsigned_integer),
                                  "PARAM_ARCHIVE_VALUE_INVALID");
    case Node::Kind::kSigned:
      return checked_config_value(config::ConfigValue::try_signed_integer(node.signed_integer),
                                  "PARAM_ARCHIVE_VALUE_INVALID");
    case Node::Kind::kFloating:
      return checked_config_value(config::ConfigValue::try_floating(node.floating),
                                  "PARAM_ARCHIVE_VALUE_INVALID");
    case Node::Kind::kString:
      return checked_config_value(config::ConfigValue::try_string(node.text),
                                  "PARAM_ARCHIVE_STRING_INVALID");
    case Node::Kind::kArray:
      return to_config_array(node);
    case Node::Kind::kObject:
      return to_config_object(node);
  }
  return make_unexpected(
      archive_error(ParamArchiveErrorCode::kInvalidArchive, "PARAM_ARCHIVE_NODE_INVALID"));
}

void append_json_string(std::string& output, std::string_view value) {
  holoscan::detail::append_json_string(output, value);
}

void append_scalar_escape(std::string& output, std::string_view kind, std::string_view value) {
  output += "{\"$holoscan.config.scalar/v1\":{\"kind\":\"";
  output += kind;
  output += "\",\"value\":";
  append_json_string(output, value);
  output += "}}";
}

[[nodiscard]] ParamArchiveResult<void> append_json(const Node& node, std::string& output,
                                                   std::size_t max_bytes);

[[nodiscard]] ParamArchiveResult<void> require_output_limit(const std::string& output,
                                                            std::size_t max_bytes) {
  if (output.size() > max_bytes) {
    return make_unexpected(
        archive_error(ParamArchiveErrorCode::kPersistenceFailure, "PARAM_ARCHIVE_OUTPUT_LIMIT"));
  }
  return {};
}

[[nodiscard]] ParamArchiveResult<void> append_json_array(const Node& node, std::string& output,
                                                         std::size_t max_bytes) {
  output.push_back('[');
  for (std::size_t index = 0; index < node.array.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    auto appended = append_json(node.array[index], output, max_bytes);
    if (!appended) {
      return appended;
    }
  }
  output.push_back(']');
  return {};
}

[[nodiscard]] ParamArchiveResult<void> append_json_object(const Node& node, std::string& output,
                                                          std::size_t max_bytes) {
  output.push_back('{');
  for (std::size_t index = 0; index < node.object_values.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    append_json_string(output, node.object_keys[index]);
    output.push_back(':');
    auto appended = append_json(node.object_values[index], output, max_bytes);
    if (!appended) {
      return appended;
    }
  }
  output.push_back('}');
  return {};
}

[[nodiscard]] ParamArchiveResult<void> append_json_floating(const Node& node, std::string& output) {
  std::array<char, 128> buffer{};
  const auto encoded = std::to_chars(
      buffer.data(), buffer.data() + buffer.size(), node.floating, std::chars_format::general);
  if (encoded.ec != std::errc{}) {
    return make_unexpected(
        archive_error(ParamArchiveErrorCode::kInvalidArchive, "PARAM_ARCHIVE_FLOAT_INVALID"));
  }
  append_scalar_escape(output, "f64", std::string_view{buffer.data(), encoded.ptr});
  return {};
}

[[nodiscard]] ParamArchiveResult<void> append_json(const Node& node, std::string& output,
                                                   std::size_t max_bytes) {
  auto within_limit = require_output_limit(output, max_bytes);
  if (!within_limit) {
    return within_limit;
  }
  switch (node.kind) {
    case Node::Kind::kNull:
      output += "null";
      break;
    case Node::Kind::kBool:
      output += node.boolean ? "true" : "false";
      break;
    case Node::Kind::kUnsigned:
      if (node.scalar_escape) {
        append_scalar_escape(output, "u64", std::to_string(node.unsigned_integer));
      } else {
        output += std::to_string(node.unsigned_integer);
      }
      break;
    case Node::Kind::kSigned:
      append_scalar_escape(output, "i64", std::to_string(node.signed_integer));
      break;
    case Node::Kind::kFloating: {
      auto appended = append_json_floating(node, output);
      if (!appended) {
        return appended;
      }
      break;
    }
    case Node::Kind::kString:
      append_json_string(output, node.text);
      break;
    case Node::Kind::kArray: {
      auto appended = append_json_array(node, output, max_bytes);
      if (!appended) {
        return appended;
      }
      break;
    }
    case Node::Kind::kObject: {
      auto appended = append_json_object(node, output, max_bytes);
      if (!appended) {
        return appended;
      }
      break;
    }
  }
  return require_output_limit(output, max_bytes);
}

template <typename Digest>
[[nodiscard]] std::string tagged_digest(Digest value) {
  return holoscan::detail::lowercase_hex_text(value.value, holoscan::detail::HexPrefix::kSha256);
}

[[nodiscard]] std::optional<std::array<std::byte, kDigestBytes>> parse_tagged_digest(
    std::string_view value) {
  if (!value.starts_with("sha256:") || value.size() != 7U + 2U * kDigestBytes) {
    return std::nullopt;
  }
  const auto nibble = [](char character) -> std::optional<std::uint8_t> {
    if (character >= '0' && character <= '9') {
      return static_cast<std::uint8_t>(character - '0');
    }
    if (character >= 'a' && character <= 'f') {
      return static_cast<std::uint8_t>(character - 'a' + 10U);
    }
    return std::nullopt;
  };
  std::array<std::byte, kDigestBytes> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    const auto high = nibble(value[7U + index * 2U]);
    const auto low = nibble(value[8U + index * 2U]);
    if (!high || !low) {
      return std::nullopt;
    }
    result[index] = static_cast<std::byte>((*high << 4U) | *low);
  }
  return result;
}

[[nodiscard]] std::string_view profile_token(Profile profile) noexcept {
  switch (profile) {
    case Profile::kStandard:
      return "standard";
    case Profile::kRobotics:
      return "robotics";
    case Profile::kRealTime:
      return "real-time";
    case Profile::kRegulated:
      return "regulated";
    case Profile::kResearch:
      return "research";
  }
  return {};
}

[[nodiscard]] std::optional<Profile> parse_profile(std::string_view token) noexcept {
  if (token == "standard") {
    return Profile::kStandard;
  }
  if (token == "robotics") {
    return Profile::kRobotics;
  }
  if (token == "real-time") {
    return Profile::kRealTime;
  }
  if (token == "regulated") {
    return Profile::kRegulated;
  }
  if (token == "research") {
    return Profile::kResearch;
  }
  return std::nullopt;
}

[[nodiscard]] std::string_view projection_token(ParamProjectionStatus projection) noexcept {
  switch (projection) {
    case ParamProjectionStatus::kIncluded:
      return "included";
    case ParamProjectionStatus::kRedactedSensitive:
      return "redacted-sensitive";
    case ParamProjectionStatus::kExcludedInternal:
      return "excluded-internal";
    case ParamProjectionStatus::kExcludedTransient:
      return "excluded-transient";
    case ParamProjectionStatus::kExcludedProfilePolicy:
      return "excluded-profile-policy";
    case ParamProjectionStatus::kExcludedUnsupportedCodec:
      return "excluded-unsupported-codec";
  }
  return {};
}

[[nodiscard]] std::optional<ParamProjectionStatus> parse_projection(
    std::string_view token) noexcept {
  if (token == "included") {
    return ParamProjectionStatus::kIncluded;
  }
  if (token == "redacted-sensitive") {
    return ParamProjectionStatus::kRedactedSensitive;
  }
  if (token == "excluded-internal") {
    return ParamProjectionStatus::kExcludedInternal;
  }
  if (token == "excluded-transient") {
    return ParamProjectionStatus::kExcludedTransient;
  }
  if (token == "excluded-profile-policy") {
    return ParamProjectionStatus::kExcludedProfilePolicy;
  }
  if (token == "excluded-unsupported-codec") {
    return ParamProjectionStatus::kExcludedUnsupportedCodec;
  }
  return std::nullopt;
}

[[nodiscard]] std::string_view no_value_reason(ParamProjectionStatus projection) noexcept {
  switch (projection) {
    case ParamProjectionStatus::kRedactedSensitive:
      return "sensitive-redacted";
    case ParamProjectionStatus::kExcludedInternal:
      return "internal-policy";
    case ParamProjectionStatus::kExcludedTransient:
      return "transient";
    case ParamProjectionStatus::kExcludedProfilePolicy:
      return "profile-policy";
    case ParamProjectionStatus::kExcludedUnsupportedCodec:
      return "codec-unavailable";
    case ParamProjectionStatus::kIncluded:
      return {};
  }
  return {};
}

[[nodiscard]] std::string_view state_token(ParamUpdateState state) noexcept {
  if (state == ParamUpdateState::kPublished || state == ParamUpdateState::kObserved ||
      state == ParamUpdateState::kAcceptedNoChange) {
    return "published";
  }
  if (state == ParamUpdateState::kAccepted || state == ParamUpdateState::kSubmitted ||
      state == ParamUpdateState::kQuiescing || state == ParamUpdateState::kRestarting) {
    return "latest-accepted";
  }
  return "unavailable";
}

[[nodiscard]] ParamUpdateState parse_state(std::string_view token) noexcept {
  if (token == "published") {
    return ParamUpdateState::kPublished;
  }
  if (token == "latest-accepted") {
    return ParamUpdateState::kAccepted;
  }
  return ParamUpdateState::kFailedStopped;
}

[[nodiscard]] std::string_view coverage_token(ParamSchemaAvailability availability,
                                              ParamSchemaCoverage coverage) noexcept {
  if (availability == ParamSchemaAvailability::kOpaqueLocal) {
    return "opaque";
  }
  return coverage == ParamSchemaCoverage::kComplete ? "complete" : "partial";
}

[[nodiscard]] std::string_view proof_token(ParamSchemaAvailability availability,
                                           ParamSchemaProof proof) noexcept {
  if (availability == ParamSchemaAvailability::kOpaqueLocal) {
    return "none";
  }
  return proof == ParamSchemaProof::kProven ? "proven" : "author-asserted";
}

[[nodiscard]] std::string_view value_coverage_token(ParamSnapshotValueCoverage coverage) noexcept {
  switch (coverage) {
    case ParamSnapshotValueCoverage::kUnavailable:
      return "unavailable";
    case ParamSnapshotValueCoverage::kPolicyFiltered:
      return "policy-filtered";
    case ParamSnapshotValueCoverage::kAllDeclaredFields:
      return "all-declared-fields";
  }
  return {};
}

[[nodiscard]] Node path_node(std::span<const std::string> segments) {
  std::vector<Node> values;
  values.reserve(segments.size());
  for (const auto& segment : segments) {
    values.push_back(Node::string(segment));
  }
  return Node::object_value({{"segments", Node::array_value(std::move(values))}});
}

[[nodiscard]] Node segments_node(std::span<const std::string> segments) {
  std::vector<Node> values;
  values.reserve(segments.size());
  for (const auto& segment : segments) {
    values.push_back(Node::string(segment));
  }
  return Node::array_value(std::move(values));
}

[[nodiscard]] Node present_node(Node value) {
  return Node::object_value(
      {{"availability", Node::string("present")}, {"value", std::move(value)}});
}

[[nodiscard]] Node unavailable_node(std::string reason) {
  return Node::object_value(
      {{"availability", Node::string("unavailable")}, {"reason", Node::string(std::move(reason))}});
}

[[nodiscard]] std::int64_t sign_extend(std::uint64_t bits, std::uint8_t width) noexcept {
  if (width == 0U || width > 64U) {
    return 0;
  }
  if (width == 64U) {
    return std::bit_cast<std::int64_t>(bits);
  }
  const std::uint64_t mask = (std::uint64_t{1} << width) - 1U;
  bits &= mask;
  const std::uint64_t sign = std::uint64_t{1} << (width - 1U);
  if ((bits & sign) != 0U) {
    bits |= ~mask;
  }
  return std::bit_cast<std::int64_t>(bits);
}

[[nodiscard]] ParamArchiveResult<Node> scalar_node(const ParamValue& value) {
  switch (value.kind()) {
    case ParamScalarKind::kBool:
      if (value.bit_width() != 1U || value.raw_bits() > 1U) {
        return make_unexpected(
            archive_error(ParamArchiveErrorCode::kInvalidArchive, "PARAM_ARCHIVE_BOOL_INVALID"));
      }
      return Node::boolean_value(value.raw_bits() != 0U);
    case ParamScalarKind::kSignedInteger:
      return Node::signed_scalar(sign_extend(value.raw_bits(), value.bit_width()));
    case ParamScalarKind::kUnsignedInteger: {
      if (value.bit_width() == 0U || value.bit_width() > 64U) {
        return make_unexpected(archive_error(ParamArchiveErrorCode::kInvalidArchive,
                                             "PARAM_ARCHIVE_UNSIGNED_INVALID"));
      }
      const std::uint64_t mask = value.bit_width() == 64U
                                     ? std::numeric_limits<std::uint64_t>::max()
                                     : (std::uint64_t{1} << value.bit_width()) - 1U;
      return Node::unsigned_scalar(value.raw_bits() & mask);
    }
    case ParamScalarKind::kFloatingPoint:
      if (value.bit_width() == 32U) {
        const auto bits = static_cast<std::uint32_t>(value.raw_bits());
        const double decoded = static_cast<double>(std::bit_cast<float>(bits));
        if (!std::isfinite(decoded)) {
          return make_unexpected(archive_error(ParamArchiveErrorCode::kInvalidArchive,
                                               "PARAM_ARCHIVE_FLOAT_NONFINITE"));
        }
        return Node::floating_scalar(decoded);
      }
      if (value.bit_width() == 64U) {
        const double decoded = std::bit_cast<double>(value.raw_bits());
        if (!std::isfinite(decoded)) {
          return make_unexpected(archive_error(ParamArchiveErrorCode::kInvalidArchive,
                                               "PARAM_ARCHIVE_FLOAT_NONFINITE"));
        }
        return Node::floating_scalar(decoded);
      }
      break;
    case ParamScalarKind::kDurationNanoseconds:
      if (value.bit_width() == 64U) {
        return Node::signed_scalar(std::bit_cast<std::int64_t>(value.raw_bits()));
      }
      break;
    case ParamScalarKind::kNone:
      break;
  }
  return make_unexpected(
      archive_error(ParamArchiveErrorCode::kInvalidArchive, "PARAM_ARCHIVE_SCALAR_INVALID"));
}

[[nodiscard]] bool unsigned_utf8_less(std::string_view left, std::string_view right) noexcept {
  return std::lexicographical_compare(
      left.begin(), left.end(), right.begin(), right.end(), [](char lhs, char rhs) {
        return static_cast<unsigned char>(lhs) < static_cast<unsigned char>(rhs);
      });
}

[[nodiscard]] bool path_less(const OperatorPath& left, const OperatorPath& right) noexcept {
  return std::lexicographical_compare(
      left.segments().begin(),
      left.segments().end(),
      right.segments().begin(),
      right.segments().end(),
      [](const std::string& lhs, const std::string& rhs) { return unsigned_utf8_less(lhs, rhs); });
}

struct ArchiveBuild {
  Node envelope;
  ParamSaveResult report;
};

[[nodiscard]] ParamArchiveResult<Node> field_node(const ParamSnapshotFieldRecord& field,
                                                  ParamSaveResult& report) {
  if (field.address.field == 0U || field.path.segments().empty() || field.type.id.empty() ||
      field.type.version == 0U) {
    return make_unexpected(archive_error(ParamArchiveErrorCode::kInvalidArchive,
                                         "PARAM_ARCHIVE_FIELD_IDENTITY_INVALID"));
  }

  Node codec;
  if (field.codec) {
    if (field.codec->id.empty() || field.codec->version == 0U) {
      return make_unexpected(archive_error(ParamArchiveErrorCode::kInvalidArchive,
                                           "PARAM_ARCHIVE_CODEC_IDENTITY_INVALID"));
    }
    codec = Node::object_value({{"availability", Node::string("present")},
                                {"id", Node::string(field.codec->id)},
                                {"version", Node::structural(field.codec->version)}});
  } else {
    codec = unavailable_node("no-qualified-codec");
  }

  Node payload;
  switch (field.projection) {
    case ParamProjectionStatus::kIncluded: {
      if (!field.codec || !field.value) {
        ++report.error_count;
        return make_unexpected(archive_error(ParamArchiveErrorCode::kProjectionFailure,
                                             "PARAM_ARCHIVE_INCLUDED_VALUE_MISSING"));
      }
      auto value = scalar_node(*field.value);
      if (!value) {
        ++report.error_count;
        return make_unexpected(std::move(value.error()));
      }
      payload = Node::object_value({{"kind", Node::string("value")}, {"value", std::move(*value)}});
      ++report.included_count;
      break;
    }
    case ParamProjectionStatus::kRedactedSensitive:
      payload = Node::object_value(
          {{"kind", Node::string("no-value")},
           {"reason", Node::string(std::string{no_value_reason(field.projection)})}});
      ++report.redacted_count;
      break;
    case ParamProjectionStatus::kExcludedInternal:
    case ParamProjectionStatus::kExcludedTransient:
    case ParamProjectionStatus::kExcludedProfilePolicy:
      payload = Node::object_value(
          {{"kind", Node::string("no-value")},
           {"reason", Node::string(std::string{no_value_reason(field.projection)})}});
      ++report.excluded_count;
      break;
    case ParamProjectionStatus::kExcludedUnsupportedCodec:
      payload = Node::object_value(
          {{"kind", Node::string("no-value")},
           {"reason", Node::string(std::string{no_value_reason(field.projection)})}});
      ++report.error_count;
      break;
  }

  return Node::object_value(
      {{"field_id", Node::structural(field.address.field)},
       {"path", path_node(field.path.segments())},
       {"type",
        Node::object_value({{"id", Node::string(field.type.id)},
                            {"version", Node::structural(field.type.version)}})},
       {"codec", std::move(codec)},
       {"projection", Node::string(std::string{projection_token(field.projection)})},
       {"payload", std::move(payload)}});
}

[[nodiscard]] ParamArchiveResult<Node> operator_node(const ParamSnapshotOperatorRecord& item,
                                                     ParamSaveResult& report) {
  if (item.path.segments().empty() || !item.effective_profile) {
    return make_unexpected(archive_error(ParamArchiveErrorCode::kInvalidArchive,
                                         "PARAM_ARCHIVE_OPERATOR_IDENTITY_INVALID"));
  }
  const std::string_view profile = profile_token(*item.effective_profile);
  if (profile.empty()) {
    return make_unexpected(
        archive_error(ParamArchiveErrorCode::kInvalidArchive, "PARAM_ARCHIVE_PROFILE_INVALID"));
  }

  Node schema;
  if (item.schema_availability == ParamSchemaAvailability::kOpaqueLocal) {
    schema = Node::object_value(
        {{"availability", Node::string("opaque-local")},
         {"id", unavailable_node("opaque-schema")},
         {"version", unavailable_node("opaque-schema")},
         {"coverage", Node::string("opaque")},
         {"proof", Node::string("none")},
         {"hashes",
          Node::object_value({{"structural", unavailable_node("opaque-schema")},
                              {"contract", unavailable_node("opaque-schema")},
                              {"metadata", unavailable_node("opaque-schema")}})}});
  } else {
    if (item.schema_id.empty() || item.schema_version == 0U) {
      return make_unexpected(archive_error(ParamArchiveErrorCode::kInvalidArchive,
                                           "PARAM_ARCHIVE_SCHEMA_IDENTITY_INVALID"));
    }
    schema = Node::object_value(
        {{"availability", Node::string("declared")},
         {"id", present_node(Node::string(item.schema_id))},
         {"version", present_node(Node::structural(item.schema_version))},
         {"coverage",
          Node::string(
              std::string{coverage_token(item.schema_availability, item.schema_coverage)})},
         {"proof",
          Node::string(std::string{proof_token(item.schema_availability, item.schema_proof)})},
         {"hashes",
          Node::object_value(
              {{"structural", present_node(Node::string(tagged_digest(item.structural_hash)))},
               {"contract", present_node(Node::string(tagged_digest(item.contract_hash)))},
               {"metadata", present_node(Node::string(tagged_digest(item.metadata_hash)))}})}});
  }

  std::vector<const ParamSnapshotFieldRecord*> ordered_fields;
  ordered_fields.reserve(item.fields.size());
  for (const auto& field : item.fields) {
    ordered_fields.push_back(&field);
  }
  std::ranges::sort(ordered_fields, {}, [](const auto* field) { return field->address.field; });
  for (std::size_t index = 1; index < ordered_fields.size(); ++index) {
    if (ordered_fields[index - 1U]->address.field == ordered_fields[index]->address.field) {
      return make_unexpected(
          archive_error(ParamArchiveErrorCode::kInvalidArchive, "PARAM_ARCHIVE_FIELD_DUPLICATE"));
    }
  }
  std::vector<Node> fields;
  fields.reserve(ordered_fields.size());
  for (const auto* field : ordered_fields) {
    auto encoded = field_node(*field, report);
    if (!encoded) {
      return make_unexpected(std::move(encoded.error()));
    }
    fields.push_back(std::move(*encoded));
  }

  return Node::object_value(
      {{"path", path_node(item.path.segments())},
       {"schema", std::move(schema)},
       {"generation", Node::structural(item.generation)},
       {"sequence",
        Node::object_value({{"availability", Node::string("present")},
                            {"accepted", Node::structural(item.accepted_update_sequence)},
                            {"published", Node::structural(item.published_update_sequence)}})},
       {"state", Node::string(std::string{state_token(item.state)})},
       {"value_coverage", Node::string(std::string{value_coverage_token(item.value_coverage)})},
       {"filter", Node::string("policy-default")},
       {"effective_profile", Node::string(std::string{profile})},
       {"fields", Node::array_value(std::move(fields))}});
}

[[nodiscard]] ParamArchiveResult<bool> archive_capture_is_stopped(const ParamSnapshot& snapshot) {
  if (snapshot.consistency() == ParamCaptureConsistency::kDispatcherBarrier ||
      snapshot.view() == ParamReadView::kPending) {
    return make_unexpected(archive_error(ParamArchiveErrorCode::kInvalidArchive,
                                         "PARAM_ARCHIVE_CAPTURE_MODE_UNSUPPORTED"));
  }
  const bool stopped = snapshot.consistency() == ParamCaptureConsistency::kRequireStopped;
  const bool empty_capture =
      snapshot.operators().empty() && !snapshot.accepted_through() && !snapshot.published_through();
  if (stopped &&
      (snapshot.view() != ParamReadView::kPublished ||
       ((!snapshot.accepted_through() || !snapshot.published_through()) && !empty_capture))) {
    return make_unexpected(archive_error(ParamArchiveErrorCode::kInvalidArchive,
                                         "PARAM_ARCHIVE_STOPPED_FRONTIER_INVALID"));
  }
  if (!stopped && (snapshot.accepted_through() || snapshot.published_through())) {
    return make_unexpected(archive_error(ParamArchiveErrorCode::kInvalidArchive,
                                         "PARAM_ARCHIVE_LIVE_FRONTIER_INVALID"));
  }
  return stopped;
}

struct OrderedOperators {
  Profile profile{Profile::kStandard};
  std::vector<const ParamSnapshotOperatorRecord*> records;
};

[[nodiscard]] ParamArchiveResult<OrderedOperators> ordered_archive_operators(
    const ParamSnapshot& snapshot) {
  OrderedOperators result;
  result.profile = holoscan::detail::ParamSnapshotArchiveAccess::archive_profile(snapshot).value_or(
      Profile::kStandard);
  if (!snapshot.operators().empty()) {
    const auto& first_profile = snapshot.operators().front().effective_profile;
    if (!first_profile) {
      return make_unexpected(
          archive_error(ParamArchiveErrorCode::kInvalidArchive, "PARAM_ARCHIVE_PROFILE_MISSING"));
    }
    result.profile = first_profile.value_or(Profile::kStandard);
  }
  result.records.reserve(snapshot.operators().size());
  for (const auto& item : snapshot.operators()) {
    if (!item.effective_profile || *item.effective_profile != result.profile) {
      return make_unexpected(
          archive_error(ParamArchiveErrorCode::kInvalidArchive, "PARAM_ARCHIVE_PROFILE_MIXED"));
    }
    result.records.push_back(&item);
  }
  std::ranges::sort(result.records, [](const auto* lhs, const auto* rhs) {
    return path_less(lhs->path, rhs->path);
  });
  for (std::size_t index = 1; index < result.records.size(); ++index) {
    if (result.records[index - 1U]->path == result.records[index]->path) {
      return make_unexpected(archive_error(ParamArchiveErrorCode::kInvalidArchive,
                                           "PARAM_ARCHIVE_OPERATOR_DUPLICATE"));
    }
  }
  return result;
}

[[nodiscard]] ParamArchiveResult<std::vector<Node>> encode_archive_operators(
    const std::vector<const ParamSnapshotOperatorRecord*>& ordered, ParamSaveResult& report) {
  std::vector<Node> operators;
  operators.reserve(ordered.size());
  for (const auto* item : ordered) {
    auto encoded = operator_node(*item, report);
    if (!encoded) {
      return make_unexpected(std::move(encoded.error()));
    }
    operators.push_back(std::move(*encoded));
  }
  return operators;
}

[[nodiscard]] ParamArchiveResult<Node> archive_scope_node(const ParamSnapshot& snapshot) {
  std::string scope_kind;
  switch (snapshot.scope().kind()) {
    case ParamSnapshotScopeKind::kRoot:
      scope_kind = "root";
      break;
    case ParamSnapshotScopeKind::kGraph:
      scope_kind = "graph";
      break;
    case ParamSnapshotScopeKind::kOperator:
      scope_kind = "operator";
      break;
  }
  if ((snapshot.scope().kind() == ParamSnapshotScopeKind::kRoot) !=
      snapshot.scope().path().segments().empty()) {
    return make_unexpected(
        archive_error(ParamArchiveErrorCode::kInvalidArchive, "PARAM_ARCHIVE_SCOPE_INVALID"));
  }
  return Node::object_value({{"kind", Node::string(std::move(scope_kind))},
                             {"segments", segments_node(snapshot.scope().path().segments())}});
}

[[nodiscard]] Node archive_frontier_node(const ParamSnapshot& snapshot, bool stopped) {
  if (stopped) {
    return Node::object_value(
        {{"availability", Node::string("global")},
         {"accepted_through", Node::structural(snapshot.accepted_through().value_or(0U))},
         {"published_through", Node::structural(snapshot.published_through().value_or(0U))}});
  }
  return Node::object_value({{"availability", Node::string("per-operator")},
                             {"accepted_through", unavailable_node("per-operator")},
                             {"published_through", unavailable_node("per-operator")}});
}

[[nodiscard]] Node archive_source_run_node(const ParamSnapshot& snapshot) {
  const auto& run_identity =
      holoscan::detail::ParamSnapshotArchiveAccess::archive_source_run_identity(snapshot);
  return run_identity ? Node::string(*run_identity) : Node::null();
}

[[nodiscard]] bool archive_has_profile_exclusion(const ParamSnapshot& snapshot) {
  return std::ranges::any_of(snapshot.operators(), [](const auto& item) {
    return std::ranges::any_of(item.fields, [](const auto& field) {
      return field.projection == ParamProjectionStatus::kExcludedProfilePolicy;
    });
  });
}

[[nodiscard]] ParamArchiveResult<ArchiveBuild> build_archive(const ParamSnapshot& snapshot) {
  auto stopped = archive_capture_is_stopped(snapshot);
  if (!stopped) {
    return make_unexpected(std::move(stopped.error()));
  }

  auto ordered = ordered_archive_operators(snapshot);
  if (!ordered) {
    return make_unexpected(std::move(ordered.error()));
  }

  ArchiveBuild result;
  auto operators = encode_archive_operators(ordered->records, result.report);
  if (!operators) {
    return make_unexpected(std::move(operators.error()));
  }

  auto scope = archive_scope_node(snapshot);
  if (!scope) {
    return make_unexpected(std::move(scope.error()));
  }

  Node frontier = archive_frontier_node(snapshot, *stopped);
  const bool has_profile_exclusion = archive_has_profile_exclusion(snapshot);
  result.report.complete_for_normal_persistence =
      result.report.error_count == 0U && !has_profile_exclusion;
  result.envelope = Node::object_value(
      {{"format", Node::string(std::string{kArchiveFormat})},
       {"version", Node::structural(kArchiveVersion)},
       {"profile", Node::string(std::string{profile_token(ordered->profile)})},
       {"source",
        Node::object_value({{"plan", Node::string(tagged_digest(snapshot.plan_digest()))},
                            {"run", archive_source_run_node(snapshot)},
                            {"catalog", Node::string(tagged_digest(snapshot.catalog_digest()))}})},
       {"scope", std::move(*scope)},
       {"capture",
        Node::object_value(
            {{"view",
              Node::string(snapshot.view() == ParamReadView::kPublished ? "published"
                                                                        : "latest-accepted")},
             {"consistency", Node::string(*stopped ? "stopped-published" : "live-per-operator")},
             {"frontier", std::move(frontier)}})},
       {"operators", Node::array_value(std::move(*operators))},
       {"extensions", Node::object_value({})}});
  return result;
}

[[nodiscard]] ParamArchiveResult<config::SemanticDigest> semantic_digest(const Node& envelope) {
  auto value = to_config_value(envelope);
  if (!value) {
    return make_unexpected(std::move(value.error()));
  }
  auto encoded = config::try_encode_semantic_value_v1(value->view());
  if (!encoded) {
    return make_unexpected(archive_config_error(ParamArchiveErrorCode::kInvalidArchive,
                                                "PARAM_ARCHIVE_SEMANTIC_ENCODE_FAILED",
                                                std::move(encoded.error())));
  }
  holoscan::detail::StableHashBuilder hash;
  for (const char byte : kDigestDomain) {
    hash.add_byte(static_cast<std::uint8_t>(byte));
  }
  hash.add_byte(0U);
  for (const std::byte byte : *encoded) {
    hash.add_byte(std::to_integer<std::uint8_t>(byte));
  }
  const ParamSchemaHash finished = hash.finish();
  config::SemanticDigest result;
  result.canonical_version = 1U;
  result.sha256 = finished.value;
  return result;
}

[[nodiscard]] std::string tagged_digest(config::SemanticDigest value) {
  return holoscan::detail::lowercase_hex_text(value.sha256, holoscan::detail::HexPrefix::kSha256);
}

[[nodiscard]] ParamArchiveResult<Node> complete_envelope(const ParamSnapshot& snapshot,
                                                         ParamSaveResult& report,
                                                         config::SemanticDigest& digest) {
  auto built = build_archive(snapshot);
  if (!built) {
    return make_unexpected(std::move(built.error()));
  }
  auto computed = semantic_digest(built->envelope);
  if (!computed) {
    return make_unexpected(std::move(computed.error()));
  }
  digest = *computed;
  report = built->report;
  built->envelope.object_keys.push_back("digest");
  built->envelope.object_values.push_back(Node::string(tagged_digest(*computed)));
  return std::move(built->envelope);
}

[[nodiscard]] ParamArchiveError invalid_archive(std::string_view token) {
  return archive_error(ParamArchiveErrorCode::kInvalidArchive, token);
}

[[nodiscard]] ParamArchiveResult<void> require_exact_keys(
    const config::ConfigView& value, std::span<const std::string_view> expected,
    std::string_view token) {
  if (value.kind() != config::ConfigValueKind::kObject || value.size() != expected.size()) {
    return make_unexpected(invalid_archive(std::string{token}));
  }
  std::vector<bool> observed(expected.size(), false);
  for (std::size_t index = 0; index < value.size(); ++index) {
    auto key = value.try_key(index);
    if (!key) {
      return make_unexpected(archive_config_error(
          ParamArchiveErrorCode::kInvalidArchive, token, std::move(key.error())));
    }
    const auto found = std::ranges::find(expected, *key);
    if (found == expected.end()) {
      return make_unexpected(invalid_archive(std::string{token}));
    }
    observed[static_cast<std::size_t>(found - expected.begin())] = true;
  }
  const bool all_members_present =
      std::ranges::all_of(observed, [](bool value_present) { return value_present; });
  if (!all_members_present) {
    return make_unexpected(invalid_archive(std::string{token}));
  }
  return {};
}

[[nodiscard]] ParamArchiveResult<config::ConfigView> member(const config::ConfigView& value,
                                                            std::string_view key,
                                                            std::string_view token) {
  auto found = value.try_member(key);
  if (!found) {
    return make_unexpected(archive_config_error(
        ParamArchiveErrorCode::kInvalidArchive, token, std::move(found.error())));
  }
  return *found;
}

[[nodiscard]] ParamArchiveResult<std::string> string_value(const config::ConfigView& value,
                                                           std::string_view token) {
  auto text = value.try_as_string();
  if (!text) {
    return make_unexpected(archive_config_error(
        ParamArchiveErrorCode::kInvalidArchive, token, std::move(text.error())));
  }
  return std::string{*text};
}

[[nodiscard]] ParamArchiveResult<std::uint64_t> structural_unsigned(const config::ConfigView& value,
                                                                    std::string_view token) {
  if (value.kind() == config::ConfigValueKind::kUnsignedInteger) {
    auto decoded = value.try_as_unsigned_integer();
    if (decoded) {
      return *decoded;
    }
    return make_unexpected(archive_config_error(
        ParamArchiveErrorCode::kInvalidArchive, token, std::move(decoded.error())));
  }
  if (value.kind() == config::ConfigValueKind::kSignedInteger) {
    auto decoded = value.try_as_signed_integer();
    if (decoded && *decoded >= 0) {
      return static_cast<std::uint64_t>(*decoded);
    }
    if (!decoded) {
      return make_unexpected(archive_config_error(
          ParamArchiveErrorCode::kInvalidArchive, token, std::move(decoded.error())));
    }
  }
  return make_unexpected(invalid_archive(std::string{token}));
}

template <typename Integer>
[[nodiscard]] ParamArchiveResult<Integer> structural_integer(const config::ConfigView& value,
                                                             std::string_view token) {
  auto decoded = structural_unsigned(value, token);
  if (!decoded || *decoded > static_cast<std::uint64_t>(std::numeric_limits<Integer>::max())) {
    if (!decoded) {
      return make_unexpected(std::move(decoded.error()));
    }
    return make_unexpected(invalid_archive(std::string{token}));
  }
  return static_cast<Integer>(*decoded);
}

template <typename Digest>
[[nodiscard]] ParamArchiveResult<Digest> digest_value(const config::ConfigView& value,
                                                      std::string_view token) {
  auto text = string_value(value, token);
  if (!text) {
    return make_unexpected(std::move(text.error()));
  }
  auto decoded = parse_tagged_digest(*text);
  if (!decoded) {
    return make_unexpected(invalid_archive(std::string{token}));
  }
  Digest result;
  result.value = *decoded;
  return result;
}

[[nodiscard]] ParamArchiveResult<std::array<std::byte, kDigestBytes>> digest_bytes(
    const config::ConfigView& value, std::string_view token) {
  auto text = string_value(value, token);
  if (!text) {
    return make_unexpected(std::move(text.error()));
  }
  auto decoded = parse_tagged_digest(*text);
  if (!decoded) {
    return make_unexpected(invalid_archive(std::string{token}));
  }
  return *decoded;
}

[[nodiscard]] ParamArchiveResult<std::vector<std::string>> decode_segments(
    const config::ConfigView& value, bool may_be_empty, std::string_view token) {
  if (value.kind() != config::ConfigValueKind::kArray || value.size() > kMaxArchiveRecords ||
      (!may_be_empty && value.size() == 0U)) {
    return make_unexpected(invalid_archive(std::string{token}));
  }
  std::vector<std::string> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    auto item = value.try_element(index);
    if (!item) {
      return make_unexpected(archive_config_error(
          ParamArchiveErrorCode::kInvalidArchive, token, std::move(item.error())));
    }
    auto segment = string_value(*item, token);
    if (!segment || segment->empty()) {
      if (!segment) {
        return make_unexpected(std::move(segment.error()));
      }
      return make_unexpected(invalid_archive(std::string{token}));
    }
    result.push_back(std::move(*segment));
  }
  return result;
}

template <typename Path>
[[nodiscard]] ParamArchiveResult<Path> decode_path_object(const config::ConfigView& value,
                                                          std::string_view token) {
  static constexpr std::array<std::string_view, 1> kKeys{"segments"};
  auto valid = require_exact_keys(value, kKeys, token);
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }
  auto segments_view = member(value, "segments", token);
  if (!segments_view) {
    return make_unexpected(std::move(segments_view.error()));
  }
  auto segments = decode_segments(*segments_view, false, token);
  if (!segments) {
    return make_unexpected(std::move(segments.error()));
  }
  auto path = Path::from_segments(*segments);
  if (!path) {
    return make_unexpected(archive_error(
        ParamArchiveErrorCode::kInvalidArchive, std::string{token}, std::move(path.error())));
  }
  return std::move(*path);
}

struct ScalarTypeIdentity {
  ParamScalarKind kind{};
  ParamLogicalType logical_type{};
  std::uint8_t bit_width{};
};

[[nodiscard]] std::optional<ScalarTypeIdentity> scalar_type_identity(std::string_view id) noexcept {
  if (id == "holoscan.bool") {
    return ScalarTypeIdentity{ParamScalarKind::kBool, ParamLogicalType::kBool, 1U};
  }
  if (id == "holoscan.duration") {
    return ScalarTypeIdentity{
        ParamScalarKind::kDurationNanoseconds, ParamLogicalType::kDuration, 64U};
  }
  if (id == "holoscan.i8") {
    return ScalarTypeIdentity{
        ParamScalarKind::kSignedInteger, ParamLogicalType::kSignedInteger, 8U};
  }
  if (id == "holoscan.i16") {
    return ScalarTypeIdentity{
        ParamScalarKind::kSignedInteger, ParamLogicalType::kSignedInteger, 16U};
  }
  if (id == "holoscan.i32") {
    return ScalarTypeIdentity{
        ParamScalarKind::kSignedInteger, ParamLogicalType::kSignedInteger, 32U};
  }
  if (id == "holoscan.i64") {
    return ScalarTypeIdentity{
        ParamScalarKind::kSignedInteger, ParamLogicalType::kSignedInteger, 64U};
  }
  if (id == "holoscan.u8") {
    return ScalarTypeIdentity{
        ParamScalarKind::kUnsignedInteger, ParamLogicalType::kUnsignedInteger, 8U};
  }
  if (id == "holoscan.u16") {
    return ScalarTypeIdentity{
        ParamScalarKind::kUnsignedInteger, ParamLogicalType::kUnsignedInteger, 16U};
  }
  if (id == "holoscan.u32") {
    return ScalarTypeIdentity{
        ParamScalarKind::kUnsignedInteger, ParamLogicalType::kUnsignedInteger, 32U};
  }
  if (id == "holoscan.u64") {
    return ScalarTypeIdentity{
        ParamScalarKind::kUnsignedInteger, ParamLogicalType::kUnsignedInteger, 64U};
  }
  if (id == "holoscan.f32") {
    return ScalarTypeIdentity{
        ParamScalarKind::kFloatingPoint, ParamLogicalType::kFloatingPoint, 32U};
  }
  if (id == "holoscan.f64") {
    return ScalarTypeIdentity{
        ParamScalarKind::kFloatingPoint, ParamLogicalType::kFloatingPoint, 64U};
  }
  return std::nullopt;
}

[[nodiscard]] ParamLogicalType logical_type_from_id(std::string_view id) noexcept {
  if (const auto scalar = scalar_type_identity(id)) {
    return scalar->logical_type;
  }
  if (id == "holoscan.string") {
    return ParamLogicalType::kString;
  }
  if (id == "holoscan.sequence") {
    return ParamLogicalType::kSequence;
  }
  if (id == "holoscan.filesystem-path") {
    return ParamLogicalType::kFilesystemPath;
  }
  return ParamLogicalType::kObject;
}

[[nodiscard]] ParamArchiveResult<ParamScalarValue> decode_bool_scalar(
    const config::ConfigView& value, const ScalarTypeIdentity& type) {
  if (value.kind() != config::ConfigValueKind::kBool) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_VALUE_INVALID"));
  }
  auto decoded = value.try_as_bool();
  if (!decoded) {
    return make_unexpected(archive_config_error(ParamArchiveErrorCode::kInvalidArchive,
                                                "PARAM_ARCHIVE_VALUE_INVALID",
                                                std::move(decoded.error())));
  }
  return ParamScalarValue{
      .kind = ParamScalarKind::kBool, .bit_width = type.bit_width, .raw_bits = *decoded ? 1U : 0U};
}

[[nodiscard]] ParamArchiveResult<ParamScalarValue> decode_signed_scalar(
    const config::ConfigView& value, const ScalarTypeIdentity& type) {
  if (value.kind() != config::ConfigValueKind::kSignedInteger) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_VALUE_INVALID"));
  }
  auto decoded = value.try_as_signed_integer();
  if (!decoded) {
    return make_unexpected(archive_config_error(ParamArchiveErrorCode::kInvalidArchive,
                                                "PARAM_ARCHIVE_VALUE_INVALID",
                                                std::move(decoded.error())));
  }
  if (type.bit_width < 64U) {
    const auto minimum = -(std::int64_t{1} << (type.bit_width - 1U));
    const auto maximum = (std::int64_t{1} << (type.bit_width - 1U)) - 1;
    if (*decoded < minimum || *decoded > maximum) {
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_VALUE_RANGE"));
    }
  }
  const std::uint64_t mask = type.bit_width == 64U ? std::numeric_limits<std::uint64_t>::max()
                                                   : (std::uint64_t{1} << type.bit_width) - 1U;
  return ParamScalarValue{.kind = type.kind,
                          .bit_width = type.bit_width,
                          .raw_bits = std::bit_cast<std::uint64_t>(*decoded) & mask};
}

[[nodiscard]] ParamArchiveResult<ParamScalarValue> decode_unsigned_scalar(
    const config::ConfigView& value, const ScalarTypeIdentity& type) {
  if (value.kind() != config::ConfigValueKind::kUnsignedInteger) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_VALUE_INVALID"));
  }
  auto decoded = value.try_as_unsigned_integer();
  if (!decoded) {
    return make_unexpected(archive_config_error(ParamArchiveErrorCode::kInvalidArchive,
                                                "PARAM_ARCHIVE_VALUE_INVALID",
                                                std::move(decoded.error())));
  }
  if (type.bit_width < 64U && *decoded >= (std::uint64_t{1} << type.bit_width)) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_VALUE_RANGE"));
  }
  return ParamScalarValue{
      .kind = ParamScalarKind::kUnsignedInteger, .bit_width = type.bit_width, .raw_bits = *decoded};
}

[[nodiscard]] ParamArchiveResult<ParamScalarValue> decode_floating_scalar(
    const config::ConfigView& value, const ScalarTypeIdentity& type) {
  if (value.kind() != config::ConfigValueKind::kFloating) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_VALUE_INVALID"));
  }
  auto decoded = value.try_as_floating();
  if (!decoded) {
    return make_unexpected(archive_config_error(ParamArchiveErrorCode::kInvalidArchive,
                                                "PARAM_ARCHIVE_VALUE_INVALID",
                                                std::move(decoded.error())));
  }
  if (!std::isfinite(*decoded)) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_VALUE_INVALID"));
  }
  const double normalized = *decoded == 0.0 ? 0.0 : *decoded;
  if (type.bit_width == 32U) {
    const float narrowed = static_cast<float>(normalized);
    if (!std::isfinite(narrowed)) {
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_VALUE_RANGE"));
    }
    return ParamScalarValue{.kind = ParamScalarKind::kFloatingPoint,
                            .bit_width = 32U,
                            .raw_bits = std::bit_cast<std::uint32_t>(narrowed)};
  }
  return ParamScalarValue{.kind = ParamScalarKind::kFloatingPoint,
                          .bit_width = 64U,
                          .raw_bits = std::bit_cast<std::uint64_t>(normalized)};
}

[[nodiscard]] ParamArchiveResult<ParamScalarValue> decode_scalar(const config::ConfigView& value,
                                                                 std::string_view type_id) {
  const auto type = scalar_type_identity(type_id);
  if (!type) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_TYPE_INVALID"));
  }
  switch (type->kind) {
    case ParamScalarKind::kBool:
      return decode_bool_scalar(value, *type);
    case ParamScalarKind::kSignedInteger:
    case ParamScalarKind::kDurationNanoseconds:
      return decode_signed_scalar(value, *type);
    case ParamScalarKind::kUnsignedInteger:
      return decode_unsigned_scalar(value, *type);
    case ParamScalarKind::kFloatingPoint:
      return decode_floating_scalar(value, *type);
    case ParamScalarKind::kNone:
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_VALUE_UNSUPPORTED"));
  }
  return make_unexpected(invalid_archive("PARAM_ARCHIVE_VALUE_UNSUPPORTED"));
}

[[nodiscard]] ParamArchiveResult<ParamSnapshotCodecIdentity> decode_codec(
    const config::ConfigView& value) {
  auto availability_view = member(value, "availability", "PARAM_ARCHIVE_CODEC_INVALID");
  if (!availability_view) {
    return make_unexpected(std::move(availability_view.error()));
  }
  auto availability = string_value(*availability_view, "PARAM_ARCHIVE_CODEC_INVALID");
  if (!availability) {
    return make_unexpected(std::move(availability.error()));
  }
  if (*availability != "present") {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_CODEC_UNAVAILABLE"));
  }
  static constexpr std::array<std::string_view, 3> kKeys{"availability", "id", "version"};
  auto valid = require_exact_keys(value, kKeys, "PARAM_ARCHIVE_CODEC_INVALID");
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }
  auto id_view = member(value, "id", "PARAM_ARCHIVE_CODEC_INVALID");
  auto version_view = member(value, "version", "PARAM_ARCHIVE_CODEC_INVALID");
  if (!id_view) {
    return make_unexpected(std::move(id_view.error()));
  }
  if (!version_view) {
    return make_unexpected(std::move(version_view.error()));
  }
  auto id = string_value(*id_view, "PARAM_ARCHIVE_CODEC_INVALID");
  auto version = structural_integer<std::uint32_t>(*version_view, "PARAM_ARCHIVE_CODEC_INVALID");
  if (!id || id->empty() || !version || *version == 0U) {
    if (!id) {
      return make_unexpected(std::move(id.error()));
    }
    if (!version) {
      return make_unexpected(std::move(version.error()));
    }
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_CODEC_INVALID"));
  }
  return ParamSnapshotCodecIdentity{.id = std::move(*id), .version = *version};
}

[[nodiscard]] ParamArchiveResult<void> validate_unavailable_codec(const config::ConfigView& value) {
  static constexpr std::array<std::string_view, 2> kKeys{"availability", "reason"};
  auto valid = require_exact_keys(value, kKeys, "PARAM_ARCHIVE_CODEC_INVALID");
  if (!valid) {
    return valid;
  }
  auto availability_view = member(value, "availability", "PARAM_ARCHIVE_CODEC_INVALID");
  auto reason_view = member(value, "reason", "PARAM_ARCHIVE_CODEC_INVALID");
  if (!availability_view) {
    return make_unexpected(std::move(availability_view.error()));
  }
  if (!reason_view) {
    return make_unexpected(std::move(reason_view.error()));
  }
  auto availability = string_value(*availability_view, "PARAM_ARCHIVE_CODEC_INVALID");
  auto reason = string_value(*reason_view, "PARAM_ARCHIVE_CODEC_INVALID");
  if (!availability || !reason || *availability != "unavailable" ||
      *reason != "no-qualified-codec") {
    if (!availability) {
      return make_unexpected(std::move(availability.error()));
    }
    if (!reason) {
      return make_unexpected(std::move(reason.error()));
    }
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_CODEC_INVALID"));
  }
  return {};
}

// Keep this validator linear so each archive field retains its stage-specific error token.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] ParamArchiveResult<ParamSnapshotFieldRecord> decode_field(
    const config::ConfigView& value) {
  static constexpr std::array<std::string_view, 6> kKeys{
      "field_id", "path", "type", "codec", "projection", "payload"};
  auto valid = require_exact_keys(value, kKeys, "PARAM_ARCHIVE_FIELD_INVALID");
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }

  auto field_id_view = member(value, "field_id", "PARAM_ARCHIVE_FIELD_INVALID");
  auto path_view = member(value, "path", "PARAM_ARCHIVE_FIELD_INVALID");
  auto type_view = member(value, "type", "PARAM_ARCHIVE_FIELD_INVALID");
  auto codec_view = member(value, "codec", "PARAM_ARCHIVE_FIELD_INVALID");
  auto projection_view = member(value, "projection", "PARAM_ARCHIVE_FIELD_INVALID");
  auto payload_view = member(value, "payload", "PARAM_ARCHIVE_FIELD_INVALID");
  if (!field_id_view) {
    return make_unexpected(std::move(field_id_view.error()));
  }
  if (!path_view) {
    return make_unexpected(std::move(path_view.error()));
  }
  if (!type_view) {
    return make_unexpected(std::move(type_view.error()));
  }
  if (!codec_view) {
    return make_unexpected(std::move(codec_view.error()));
  }
  if (!projection_view) {
    return make_unexpected(std::move(projection_view.error()));
  }
  if (!payload_view) {
    return make_unexpected(std::move(payload_view.error()));
  }

  auto field_id = structural_integer<ParamFieldId>(*field_id_view, "PARAM_ARCHIVE_FIELD_INVALID");
  auto path = decode_path_object<ParamPath>(*path_view, "PARAM_ARCHIVE_FIELD_INVALID");
  if (!field_id || *field_id == 0U || !path) {
    if (!field_id) {
      return make_unexpected(std::move(field_id.error()));
    }
    if (!path) {
      return make_unexpected(std::move(path.error()));
    }
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_FIELD_INVALID"));
  }

  static constexpr std::array<std::string_view, 2> kTypeKeys{"id", "version"};
  valid = require_exact_keys(*type_view, kTypeKeys, "PARAM_ARCHIVE_TYPE_INVALID");
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }
  auto type_id_view = member(*type_view, "id", "PARAM_ARCHIVE_TYPE_INVALID");
  auto type_version_view = member(*type_view, "version", "PARAM_ARCHIVE_TYPE_INVALID");
  if (!type_id_view) {
    return make_unexpected(std::move(type_id_view.error()));
  }
  if (!type_version_view) {
    return make_unexpected(std::move(type_version_view.error()));
  }
  auto type_id = string_value(*type_id_view, "PARAM_ARCHIVE_TYPE_INVALID");
  auto type_version =
      structural_integer<std::uint32_t>(*type_version_view, "PARAM_ARCHIVE_TYPE_INVALID");
  if (!type_id || type_id->empty() || !type_version || *type_version != 1U) {
    if (!type_id) {
      return make_unexpected(std::move(type_id.error()));
    }
    if (!type_version) {
      return make_unexpected(std::move(type_version.error()));
    }
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_TYPE_INVALID"));
  }

  auto projection_text = string_value(*projection_view, "PARAM_ARCHIVE_PROJECTION_INVALID");
  if (!projection_text) {
    return make_unexpected(std::move(projection_text.error()));
  }
  auto projection = parse_projection(*projection_text);
  if (!projection) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_PROJECTION_INVALID"));
  }

  std::optional<ParamSnapshotCodecIdentity> codec;
  auto availability_view = member(*codec_view, "availability", "PARAM_ARCHIVE_CODEC_INVALID");
  if (!availability_view) {
    return make_unexpected(std::move(availability_view.error()));
  }
  auto availability = string_value(*availability_view, "PARAM_ARCHIVE_CODEC_INVALID");
  if (!availability) {
    return make_unexpected(std::move(availability.error()));
  }
  if (*availability == "present") {
    auto decoded = decode_codec(*codec_view);
    if (!decoded) {
      return make_unexpected(std::move(decoded.error()));
    }
    codec = std::move(*decoded);
  } else {
    auto unavailable = validate_unavailable_codec(*codec_view);
    if (!unavailable) {
      return make_unexpected(std::move(unavailable.error()));
    }
  }

  auto kind_view = member(*payload_view, "kind", "PARAM_ARCHIVE_PAYLOAD_INVALID");
  if (!kind_view) {
    return make_unexpected(std::move(kind_view.error()));
  }
  auto kind = string_value(*kind_view, "PARAM_ARCHIVE_PAYLOAD_INVALID");
  if (!kind) {
    return make_unexpected(std::move(kind.error()));
  }
  std::optional<ParamValue> projected_value;
  ParamLogicalType logical_type = ParamLogicalType::kObject;
  ParamFieldTypeShape shape{};
  if (*projection == ParamProjectionStatus::kIncluded) {
    static constexpr std::array<std::string_view, 2> kPayloadKeys{"kind", "value"};
    valid = require_exact_keys(*payload_view, kPayloadKeys, "PARAM_ARCHIVE_PAYLOAD_INVALID");
    if (!valid || *kind != "value" || !codec) {
      if (!valid) {
        return make_unexpected(std::move(valid.error()));
      }
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_PAYLOAD_INVALID"));
    }
    std::string expected_codec{"holoscan.config."};
    constexpr std::string_view kTypePrefix{"holoscan."};
    if (!type_id->starts_with(kTypePrefix)) {
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_TYPE_INVALID"));
    }
    expected_codec.append(type_id->substr(kTypePrefix.size()));
    if (codec->version != 1U || codec->id != expected_codec) {
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_CODEC_INVALID"));
    }
    auto scalar_view = member(*payload_view, "value", "PARAM_ARCHIVE_PAYLOAD_INVALID");
    if (!scalar_view) {
      return make_unexpected(std::move(scalar_view.error()));
    }
    auto scalar = decode_scalar(*scalar_view, *type_id);
    if (!scalar) {
      return make_unexpected(std::move(scalar.error()));
    }
    projected_value = holoscan::detail::ParamSnapshotArchiveAccess::value(*scalar);
    logical_type = logical_type_from_id(*type_id);
    shape.type = logical_type;
    shape.has_stable_shape = true;
    shape.bit_width = scalar->bit_width;
    if (logical_type == ParamLogicalType::kDuration) {
      shape.element_or_rep_type = ParamLogicalType::kSignedInteger;
      shape.element_or_rep_bit_width = scalar->bit_width;
      shape.duration_period_numerator = 1;
      shape.duration_period_denominator = 1'000'000'000;
    }
  } else {
    logical_type = logical_type_from_id(*type_id);
    shape.type = logical_type;
    if (const auto scalar_type = scalar_type_identity(*type_id)) {
      shape.has_stable_shape = true;
      shape.bit_width = scalar_type->bit_width;
      if (logical_type == ParamLogicalType::kDuration) {
        shape.element_or_rep_type = ParamLogicalType::kSignedInteger;
        shape.element_or_rep_bit_width = scalar_type->bit_width;
        shape.duration_period_numerator = 1;
        shape.duration_period_denominator = 1'000'000'000;
      }
    }
    static constexpr std::array<std::string_view, 2> kPayloadKeys{"kind", "reason"};
    valid = require_exact_keys(*payload_view, kPayloadKeys, "PARAM_ARCHIVE_PAYLOAD_INVALID");
    if (!valid || *kind != "no-value") {
      if (!valid) {
        return make_unexpected(std::move(valid.error()));
      }
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_PAYLOAD_INVALID"));
    }
    auto reason_view = member(*payload_view, "reason", "PARAM_ARCHIVE_PAYLOAD_INVALID");
    if (!reason_view) {
      return make_unexpected(std::move(reason_view.error()));
    }
    auto reason = string_value(*reason_view, "PARAM_ARCHIVE_PAYLOAD_INVALID");
    if (!reason || *reason != no_value_reason(*projection)) {
      if (!reason) {
        return make_unexpected(std::move(reason.error()));
      }
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_PAYLOAD_INVALID"));
    }
  }

  ParamSnapshotFieldRecord result;
  result.address.field = *field_id;
  result.path = std::move(*path);
  result.logical_type = logical_type;
  result.type = ParamSnapshotTypeIdentity{
      .id = std::move(*type_id), .version = *type_version, .shape = shape};
  result.codec = std::move(codec);
  result.projection = *projection;
  result.value = std::move(projected_value);
  return result;
}

[[nodiscard]] ParamArchiveResult<void> validate_unavailable(const config::ConfigView& value,
                                                            std::string_view reason,
                                                            std::string_view token) {
  static constexpr std::array<std::string_view, 2> kKeys{"availability", "reason"};
  auto valid = require_exact_keys(value, kKeys, token);
  if (!valid) {
    return valid;
  }
  auto availability_view = member(value, "availability", token);
  auto reason_view = member(value, "reason", token);
  if (!availability_view) {
    return make_unexpected(std::move(availability_view.error()));
  }
  if (!reason_view) {
    return make_unexpected(std::move(reason_view.error()));
  }
  auto availability = string_value(*availability_view, token);
  auto decoded_reason = string_value(*reason_view, token);
  if (!availability || !decoded_reason || *availability != "unavailable" ||
      *decoded_reason != reason) {
    if (!availability) {
      return make_unexpected(std::move(availability.error()));
    }
    if (!decoded_reason) {
      return make_unexpected(std::move(decoded_reason.error()));
    }
    return make_unexpected(invalid_archive(std::string{token}));
  }
  return {};
}

[[nodiscard]] ParamArchiveResult<config::ConfigView> present_value(const config::ConfigView& value,
                                                                   std::string_view token) {
  static constexpr std::array<std::string_view, 2> kKeys{"availability", "value"};
  auto valid = require_exact_keys(value, kKeys, token);
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }
  auto availability_view = member(value, "availability", token);
  if (!availability_view) {
    return make_unexpected(std::move(availability_view.error()));
  }
  auto availability = string_value(*availability_view, token);
  if (!availability || *availability != "present") {
    if (!availability) {
      return make_unexpected(std::move(availability.error()));
    }
    return make_unexpected(invalid_archive(std::string{token}));
  }
  return member(value, "value", token);
}

struct DecodedSchema {
  ParamSchemaAvailability availability{ParamSchemaAvailability::kOpaqueLocal};
  std::string id;
  ParamSchemaVersion version{};
  ParamSchemaHash structural{};
  ParamSchemaHash contract{};
  ParamSchemaHash metadata{};
  ParamSchemaCoverage coverage{ParamSchemaCoverage::kDeclaredFieldsOnly};
  ParamSchemaProof proof{ParamSchemaProof::kAuthorAsserted};
};

// Keep the opaque and declared schema grammars together to preserve their exact-key symmetry.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] ParamArchiveResult<DecodedSchema> decode_schema(const config::ConfigView& value) {
  static constexpr std::array<std::string_view, 6> kKeys{
      "availability", "id", "version", "coverage", "proof", "hashes"};
  auto valid = require_exact_keys(value, kKeys, "PARAM_ARCHIVE_SCHEMA_INVALID");
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }
  auto availability_view = member(value, "availability", "PARAM_ARCHIVE_SCHEMA_INVALID");
  auto id_view = member(value, "id", "PARAM_ARCHIVE_SCHEMA_INVALID");
  auto version_view = member(value, "version", "PARAM_ARCHIVE_SCHEMA_INVALID");
  auto coverage_view = member(value, "coverage", "PARAM_ARCHIVE_SCHEMA_INVALID");
  auto proof_view = member(value, "proof", "PARAM_ARCHIVE_SCHEMA_INVALID");
  auto hashes_view = member(value, "hashes", "PARAM_ARCHIVE_SCHEMA_INVALID");
  if (!availability_view) {
    return make_unexpected(std::move(availability_view.error()));
  }
  if (!id_view) {
    return make_unexpected(std::move(id_view.error()));
  }
  if (!version_view) {
    return make_unexpected(std::move(version_view.error()));
  }
  if (!coverage_view) {
    return make_unexpected(std::move(coverage_view.error()));
  }
  if (!proof_view) {
    return make_unexpected(std::move(proof_view.error()));
  }
  if (!hashes_view) {
    return make_unexpected(std::move(hashes_view.error()));
  }
  auto availability = string_value(*availability_view, "PARAM_ARCHIVE_SCHEMA_INVALID");
  auto coverage = string_value(*coverage_view, "PARAM_ARCHIVE_SCHEMA_INVALID");
  auto proof = string_value(*proof_view, "PARAM_ARCHIVE_SCHEMA_INVALID");
  if (!availability || !coverage || !proof) {
    if (!availability) {
      return make_unexpected(std::move(availability.error()));
    }
    if (!coverage) {
      return make_unexpected(std::move(coverage.error()));
    }
    return make_unexpected(std::move(proof.error()));
  }

  static constexpr std::array<std::string_view, 3> kHashKeys{"structural", "contract", "metadata"};
  valid = require_exact_keys(*hashes_view, kHashKeys, "PARAM_ARCHIVE_SCHEMA_HASH_INVALID");
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }
  auto structural_view = member(*hashes_view, "structural", "PARAM_ARCHIVE_SCHEMA_HASH_INVALID");
  auto contract_view = member(*hashes_view, "contract", "PARAM_ARCHIVE_SCHEMA_HASH_INVALID");
  auto metadata_view = member(*hashes_view, "metadata", "PARAM_ARCHIVE_SCHEMA_HASH_INVALID");
  if (!structural_view) {
    return make_unexpected(std::move(structural_view.error()));
  }
  if (!contract_view) {
    return make_unexpected(std::move(contract_view.error()));
  }
  if (!metadata_view) {
    return make_unexpected(std::move(metadata_view.error()));
  }

  DecodedSchema result;
  if (*availability == "opaque-local") {
    auto id_valid = validate_unavailable(*id_view, "opaque-schema", "PARAM_ARCHIVE_SCHEMA_INVALID");
    auto version_valid =
        validate_unavailable(*version_view, "opaque-schema", "PARAM_ARCHIVE_SCHEMA_INVALID");
    auto structural_valid = validate_unavailable(
        *structural_view, "opaque-schema", "PARAM_ARCHIVE_SCHEMA_HASH_INVALID");
    auto contract_valid =
        validate_unavailable(*contract_view, "opaque-schema", "PARAM_ARCHIVE_SCHEMA_HASH_INVALID");
    auto metadata_valid =
        validate_unavailable(*metadata_view, "opaque-schema", "PARAM_ARCHIVE_SCHEMA_HASH_INVALID");
    if (!id_valid) {
      return make_unexpected(std::move(id_valid.error()));
    }
    if (!version_valid) {
      return make_unexpected(std::move(version_valid.error()));
    }
    if (!structural_valid) {
      return make_unexpected(std::move(structural_valid.error()));
    }
    if (!contract_valid) {
      return make_unexpected(std::move(contract_valid.error()));
    }
    if (!metadata_valid) {
      return make_unexpected(std::move(metadata_valid.error()));
    }
    if (*coverage != "opaque" || *proof != "none") {
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_SCHEMA_INVALID"));
    }
    return result;
  }
  if (*availability != "declared") {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_SCHEMA_INVALID"));
  }

  auto id_value = present_value(*id_view, "PARAM_ARCHIVE_SCHEMA_INVALID");
  auto version_value = present_value(*version_view, "PARAM_ARCHIVE_SCHEMA_INVALID");
  auto structural_value = present_value(*structural_view, "PARAM_ARCHIVE_SCHEMA_HASH_INVALID");
  auto contract_value = present_value(*contract_view, "PARAM_ARCHIVE_SCHEMA_HASH_INVALID");
  auto metadata_value = present_value(*metadata_view, "PARAM_ARCHIVE_SCHEMA_HASH_INVALID");
  if (!id_value) {
    return make_unexpected(std::move(id_value.error()));
  }
  if (!version_value) {
    return make_unexpected(std::move(version_value.error()));
  }
  if (!structural_value) {
    return make_unexpected(std::move(structural_value.error()));
  }
  if (!contract_value) {
    return make_unexpected(std::move(contract_value.error()));
  }
  if (!metadata_value) {
    return make_unexpected(std::move(metadata_value.error()));
  }
  auto id = string_value(*id_value, "PARAM_ARCHIVE_SCHEMA_INVALID");
  auto version =
      structural_integer<ParamSchemaVersion>(*version_value, "PARAM_ARCHIVE_SCHEMA_INVALID");
  auto structural =
      digest_value<ParamSchemaHash>(*structural_value, "PARAM_ARCHIVE_SCHEMA_HASH_INVALID");
  auto contract =
      digest_value<ParamSchemaHash>(*contract_value, "PARAM_ARCHIVE_SCHEMA_HASH_INVALID");
  auto metadata =
      digest_value<ParamSchemaHash>(*metadata_value, "PARAM_ARCHIVE_SCHEMA_HASH_INVALID");
  if (!id || id->empty() || !version || *version == 0U || !structural || !contract || !metadata) {
    if (!id) {
      return make_unexpected(std::move(id.error()));
    }
    if (!version) {
      return make_unexpected(std::move(version.error()));
    }
    if (!structural) {
      return make_unexpected(std::move(structural.error()));
    }
    if (!contract) {
      return make_unexpected(std::move(contract.error()));
    }
    if (!metadata) {
      return make_unexpected(std::move(metadata.error()));
    }
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_SCHEMA_INVALID"));
  }
  result.availability = ParamSchemaAvailability::kDeclared;
  result.id = std::move(*id);
  result.version = *version;
  result.structural = *structural;
  result.contract = *contract;
  result.metadata = *metadata;
  if (*coverage == "complete") {
    result.coverage = ParamSchemaCoverage::kComplete;
  } else if (*coverage == "partial") {
    result.coverage = ParamSchemaCoverage::kDeclaredFieldsOnly;
  } else {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_SCHEMA_INVALID"));
  }
  if (*proof == "proven") {
    result.proof = ParamSchemaProof::kProven;
  } else if (*proof == "author-asserted") {
    result.proof = ParamSchemaProof::kAuthorAsserted;
  } else {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_SCHEMA_INVALID"));
  }
  return result;
}

// Keep operator member validation in wire order so diagnostics identify the first invalid member.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] ParamArchiveResult<ParamSnapshotOperatorRecord> decode_operator(
    const config::ConfigView& value, Profile envelope_profile) {
  static constexpr std::array<std::string_view, 9> kKeys{"path",
                                                         "schema",
                                                         "generation",
                                                         "sequence",
                                                         "state",
                                                         "value_coverage",
                                                         "filter",
                                                         "effective_profile",
                                                         "fields"};
  auto valid = require_exact_keys(value, kKeys, "PARAM_ARCHIVE_OPERATOR_INVALID");
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }
  auto path_view = member(value, "path", "PARAM_ARCHIVE_OPERATOR_INVALID");
  auto schema_view = member(value, "schema", "PARAM_ARCHIVE_OPERATOR_INVALID");
  auto generation_view = member(value, "generation", "PARAM_ARCHIVE_OPERATOR_INVALID");
  auto sequence_view = member(value, "sequence", "PARAM_ARCHIVE_OPERATOR_INVALID");
  auto state_view = member(value, "state", "PARAM_ARCHIVE_OPERATOR_INVALID");
  auto coverage_view = member(value, "value_coverage", "PARAM_ARCHIVE_OPERATOR_INVALID");
  auto filter_view = member(value, "filter", "PARAM_ARCHIVE_OPERATOR_INVALID");
  auto profile_view = member(value, "effective_profile", "PARAM_ARCHIVE_OPERATOR_INVALID");
  auto fields_view = member(value, "fields", "PARAM_ARCHIVE_OPERATOR_INVALID");
  if (!path_view) {
    return make_unexpected(std::move(path_view.error()));
  }
  if (!schema_view) {
    return make_unexpected(std::move(schema_view.error()));
  }
  if (!generation_view) {
    return make_unexpected(std::move(generation_view.error()));
  }
  if (!sequence_view) {
    return make_unexpected(std::move(sequence_view.error()));
  }
  if (!state_view) {
    return make_unexpected(std::move(state_view.error()));
  }
  if (!coverage_view) {
    return make_unexpected(std::move(coverage_view.error()));
  }
  if (!filter_view) {
    return make_unexpected(std::move(filter_view.error()));
  }
  if (!profile_view) {
    return make_unexpected(std::move(profile_view.error()));
  }
  if (!fields_view) {
    return make_unexpected(std::move(fields_view.error()));
  }

  auto path = decode_path_object<OperatorPath>(*path_view, "PARAM_ARCHIVE_OPERATOR_INVALID");
  auto schema = decode_schema(*schema_view);
  auto generation =
      structural_integer<ParamGeneration>(*generation_view, "PARAM_ARCHIVE_OPERATOR_INVALID");
  auto state = string_value(*state_view, "PARAM_ARCHIVE_OPERATOR_INVALID");
  auto coverage = string_value(*coverage_view, "PARAM_ARCHIVE_OPERATOR_INVALID");
  auto filter = string_value(*filter_view, "PARAM_ARCHIVE_OPERATOR_INVALID");
  auto profile_text = string_value(*profile_view, "PARAM_ARCHIVE_OPERATOR_INVALID");
  if (!path || !schema || !generation || !state || !coverage || !filter || !profile_text) {
    if (!path) {
      return make_unexpected(std::move(path.error()));
    }
    if (!schema) {
      return make_unexpected(std::move(schema.error()));
    }
    if (!generation) {
      return make_unexpected(std::move(generation.error()));
    }
    if (!state) {
      return make_unexpected(std::move(state.error()));
    }
    if (!coverage) {
      return make_unexpected(std::move(coverage.error()));
    }
    if (!filter) {
      return make_unexpected(std::move(filter.error()));
    }
    return make_unexpected(std::move(profile_text.error()));
  }
  const auto profile = parse_profile(*profile_text);
  if (!profile || *profile != envelope_profile || *filter != "policy-default" ||
      (*state != "published" && *state != "latest-accepted" && *state != "unavailable")) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_OPERATOR_INVALID"));
  }

  static constexpr std::array<std::string_view, 3> kSequenceKeys{
      "availability", "accepted", "published"};
  valid = require_exact_keys(*sequence_view, kSequenceKeys, "PARAM_ARCHIVE_SEQUENCE_INVALID");
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }
  auto sequence_availability_view =
      member(*sequence_view, "availability", "PARAM_ARCHIVE_SEQUENCE_INVALID");
  auto accepted_view = member(*sequence_view, "accepted", "PARAM_ARCHIVE_SEQUENCE_INVALID");
  auto published_view = member(*sequence_view, "published", "PARAM_ARCHIVE_SEQUENCE_INVALID");
  if (!sequence_availability_view) {
    return make_unexpected(std::move(sequence_availability_view.error()));
  }
  if (!accepted_view) {
    return make_unexpected(std::move(accepted_view.error()));
  }
  if (!published_view) {
    return make_unexpected(std::move(published_view.error()));
  }
  auto sequence_availability =
      string_value(*sequence_availability_view, "PARAM_ARCHIVE_SEQUENCE_INVALID");
  auto accepted =
      structural_integer<ParamUpdateSequence>(*accepted_view, "PARAM_ARCHIVE_SEQUENCE_INVALID");
  auto published =
      structural_integer<ParamUpdateSequence>(*published_view, "PARAM_ARCHIVE_SEQUENCE_INVALID");
  if (!sequence_availability || *sequence_availability != "present" || !accepted || !published) {
    if (!sequence_availability) {
      return make_unexpected(std::move(sequence_availability.error()));
    }
    if (!accepted) {
      return make_unexpected(std::move(accepted.error()));
    }
    if (!published) {
      return make_unexpected(std::move(published.error()));
    }
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_SEQUENCE_INVALID"));
  }

  ParamSnapshotValueCoverage value_coverage{ParamSnapshotValueCoverage::kUnavailable};
  if (*coverage == "unavailable") {
    value_coverage = ParamSnapshotValueCoverage::kUnavailable;
  } else if (*coverage == "policy-filtered") {
    value_coverage = ParamSnapshotValueCoverage::kPolicyFiltered;
  } else if (*coverage == "all-declared-fields") {
    value_coverage = ParamSnapshotValueCoverage::kAllDeclaredFields;
  } else {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_OPERATOR_INVALID"));
  }

  if (fields_view->kind() != config::ConfigValueKind::kArray ||
      fields_view->size() > kMaxArchiveRecords) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_FIELDS_INVALID"));
  }
  std::vector<ParamSnapshotFieldRecord> fields;
  fields.reserve(fields_view->size());
  ParamFieldId previous{};
  for (std::size_t index = 0; index < fields_view->size(); ++index) {
    auto field_view = fields_view->try_element(index);
    if (!field_view) {
      return make_unexpected(archive_config_error(ParamArchiveErrorCode::kInvalidArchive,
                                                  "PARAM_ARCHIVE_FIELDS_INVALID",
                                                  std::move(field_view.error())));
    }
    auto field = decode_field(*field_view);
    if (!field) {
      return make_unexpected(std::move(field.error()));
    }
    if (index != 0U && field->address.field <= previous) {
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_FIELDS_NOT_ORDERED"));
    }
    previous = field->address.field;
    fields.push_back(std::move(*field));
  }

  ParamSnapshotOperatorRecord result;
  result.path = std::move(*path);
  result.effective_profile = profile;
  result.schema_availability = schema->availability;
  result.schema_id = std::move(schema->id);
  result.schema_version = schema->version;
  result.structural_hash = schema->structural;
  result.contract_hash = schema->contract;
  result.metadata_hash = schema->metadata;
  result.schema_coverage = schema->coverage;
  result.schema_proof = schema->proof;
  result.generation = *generation;
  result.accepted_update_sequence = *accepted;
  result.published_update_sequence = *published;
  result.state = parse_state(*state);
  result.value_coverage = value_coverage;
  result.fields = std::move(fields);
  return result;
}

struct DecodedEnvelope {
  ParamSnapshot snapshot;
  config::SemanticDigest semantic_digest{};
};

// This is the top-level grammar transaction; retaining wire order preserves deterministic errors.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] ParamArchiveResult<DecodedEnvelope> decode_envelope(const config::ConfigView& root) {
  static constexpr std::array<std::string_view, 9> kKeys{"format",
                                                         "version",
                                                         "profile",
                                                         "source",
                                                         "scope",
                                                         "capture",
                                                         "operators",
                                                         "extensions",
                                                         "digest"};
  auto valid = require_exact_keys(root, kKeys, "PARAM_ARCHIVE_ENVELOPE_INVALID");
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }

  auto format_view = member(root, "format", "PARAM_ARCHIVE_ENVELOPE_INVALID");
  auto version_view = member(root, "version", "PARAM_ARCHIVE_ENVELOPE_INVALID");
  auto profile_view = member(root, "profile", "PARAM_ARCHIVE_ENVELOPE_INVALID");
  auto source_view = member(root, "source", "PARAM_ARCHIVE_ENVELOPE_INVALID");
  auto scope_view = member(root, "scope", "PARAM_ARCHIVE_ENVELOPE_INVALID");
  auto capture_view = member(root, "capture", "PARAM_ARCHIVE_ENVELOPE_INVALID");
  auto operators_view = member(root, "operators", "PARAM_ARCHIVE_ENVELOPE_INVALID");
  auto extensions_view = member(root, "extensions", "PARAM_ARCHIVE_ENVELOPE_INVALID");
  auto digest_view = member(root, "digest", "PARAM_ARCHIVE_ENVELOPE_INVALID");
  if (!format_view) {
    return make_unexpected(std::move(format_view.error()));
  }
  if (!version_view) {
    return make_unexpected(std::move(version_view.error()));
  }
  if (!profile_view) {
    return make_unexpected(std::move(profile_view.error()));
  }
  if (!source_view) {
    return make_unexpected(std::move(source_view.error()));
  }
  if (!scope_view) {
    return make_unexpected(std::move(scope_view.error()));
  }
  if (!capture_view) {
    return make_unexpected(std::move(capture_view.error()));
  }
  if (!operators_view) {
    return make_unexpected(std::move(operators_view.error()));
  }
  if (!extensions_view) {
    return make_unexpected(std::move(extensions_view.error()));
  }
  if (!digest_view) {
    return make_unexpected(std::move(digest_view.error()));
  }

  auto format = string_value(*format_view, "PARAM_ARCHIVE_FORMAT_INVALID");
  auto version = structural_unsigned(*version_view, "PARAM_ARCHIVE_VERSION_INVALID");
  auto profile_text = string_value(*profile_view, "PARAM_ARCHIVE_PROFILE_INVALID");
  if (!format || *format != kArchiveFormat) {
    if (!format) {
      return make_unexpected(std::move(format.error()));
    }
    return make_unexpected(archive_error(ParamArchiveErrorCode::kUnsupportedArchiveVersion,
                                         "PARAM_ARCHIVE_FORMAT_UNSUPPORTED"));
  }
  if (!version || *version != kArchiveVersion) {
    if (!version) {
      return make_unexpected(std::move(version.error()));
    }
    return make_unexpected(archive_error(ParamArchiveErrorCode::kUnsupportedArchiveVersion,
                                         "PARAM_ARCHIVE_VERSION_UNSUPPORTED"));
  }
  if (!profile_text) {
    return make_unexpected(std::move(profile_text.error()));
  }
  const auto profile = parse_profile(*profile_text);
  if (!profile) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_PROFILE_INVALID"));
  }
  if (extensions_view->kind() != config::ConfigValueKind::kObject ||
      extensions_view->size() != 0U) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_EXTENSION_UNSUPPORTED"));
  }
  auto expected_digest = digest_bytes(*digest_view, "PARAM_ARCHIVE_DIGEST_INVALID");
  if (!expected_digest) {
    return make_unexpected(std::move(expected_digest.error()));
  }

  static constexpr std::array<std::string_view, 3> kSourceKeys{"plan", "run", "catalog"};
  valid = require_exact_keys(*source_view, kSourceKeys, "PARAM_ARCHIVE_SOURCE_INVALID");
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }
  auto plan_view = member(*source_view, "plan", "PARAM_ARCHIVE_SOURCE_INVALID");
  auto run_view = member(*source_view, "run", "PARAM_ARCHIVE_SOURCE_INVALID");
  auto catalog_view = member(*source_view, "catalog", "PARAM_ARCHIVE_SOURCE_INVALID");
  if (!plan_view) {
    return make_unexpected(std::move(plan_view.error()));
  }
  if (!run_view) {
    return make_unexpected(std::move(run_view.error()));
  }
  if (!catalog_view) {
    return make_unexpected(std::move(catalog_view.error()));
  }
  auto plan = digest_value<PlanDigest>(*plan_view, "PARAM_ARCHIVE_SOURCE_INVALID");
  auto catalog = digest_value<ParamCatalogDigest>(*catalog_view, "PARAM_ARCHIVE_SOURCE_INVALID");
  if (!plan) {
    return make_unexpected(std::move(plan.error()));
  }
  if (!catalog) {
    return make_unexpected(std::move(catalog.error()));
  }
  std::optional<std::string> source_run;
  if (run_view->kind() == config::ConfigValueKind::kString) {
    auto run = string_value(*run_view, "PARAM_ARCHIVE_SOURCE_INVALID");
    if (!run || run->empty()) {
      if (!run) {
        return make_unexpected(std::move(run.error()));
      }
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_SOURCE_INVALID"));
    }
    source_run = std::move(*run);
  } else if (run_view->kind() != config::ConfigValueKind::kNull) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_SOURCE_INVALID"));
  }

  static constexpr std::array<std::string_view, 2> kScopeKeys{"kind", "segments"};
  valid = require_exact_keys(*scope_view, kScopeKeys, "PARAM_ARCHIVE_SCOPE_INVALID");
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }
  auto scope_kind_view = member(*scope_view, "kind", "PARAM_ARCHIVE_SCOPE_INVALID");
  auto scope_segments_view = member(*scope_view, "segments", "PARAM_ARCHIVE_SCOPE_INVALID");
  if (!scope_kind_view) {
    return make_unexpected(std::move(scope_kind_view.error()));
  }
  if (!scope_segments_view) {
    return make_unexpected(std::move(scope_segments_view.error()));
  }
  auto scope_kind = string_value(*scope_kind_view, "PARAM_ARCHIVE_SCOPE_INVALID");
  if (!scope_kind) {
    return make_unexpected(std::move(scope_kind.error()));
  }
  const bool root_scope = *scope_kind == "root";
  auto scope_segments =
      decode_segments(*scope_segments_view, root_scope, "PARAM_ARCHIVE_SCOPE_INVALID");
  if (!scope_segments || (root_scope && !scope_segments->empty()) ||
      (!root_scope && scope_segments->empty())) {
    if (!scope_segments) {
      return make_unexpected(std::move(scope_segments.error()));
    }
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_SCOPE_INVALID"));
  }
  ParamSnapshotScope scope;
  if (!root_scope) {
    if (*scope_kind != "graph" && *scope_kind != "operator") {
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_SCOPE_INVALID"));
    }
    auto path = OperatorPath::from_segments(*scope_segments);
    if (!path) {
      return make_unexpected(archive_error(ParamArchiveErrorCode::kInvalidArchive,
                                           "PARAM_ARCHIVE_SCOPE_INVALID",
                                           std::move(path.error())));
    }
    scope = holoscan::detail::ParamSnapshotArchiveAccess::scope(
        *scope_kind == "graph" ? ParamSnapshotScopeKind::kGraph : ParamSnapshotScopeKind::kOperator,
        std::move(*path));
  }

  static constexpr std::array<std::string_view, 3> kCaptureKeys{"view", "consistency", "frontier"};
  valid = require_exact_keys(*capture_view, kCaptureKeys, "PARAM_ARCHIVE_CAPTURE_INVALID");
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }
  auto capture_value_view = member(*capture_view, "view", "PARAM_ARCHIVE_CAPTURE_INVALID");
  auto consistency_view = member(*capture_view, "consistency", "PARAM_ARCHIVE_CAPTURE_INVALID");
  auto frontier_view = member(*capture_view, "frontier", "PARAM_ARCHIVE_CAPTURE_INVALID");
  if (!capture_value_view) {
    return make_unexpected(std::move(capture_value_view.error()));
  }
  if (!consistency_view) {
    return make_unexpected(std::move(consistency_view.error()));
  }
  if (!frontier_view) {
    return make_unexpected(std::move(frontier_view.error()));
  }
  auto capture_value = string_value(*capture_value_view, "PARAM_ARCHIVE_CAPTURE_INVALID");
  auto consistency = string_value(*consistency_view, "PARAM_ARCHIVE_CAPTURE_INVALID");
  if (!capture_value || !consistency) {
    if (!capture_value) {
      return make_unexpected(std::move(capture_value.error()));
    }
    return make_unexpected(std::move(consistency.error()));
  }
  ParamReadView read_view{ParamReadView::kPublished};
  if (*capture_value == "published") {
    read_view = ParamReadView::kPublished;
  } else if (*capture_value == "latest-accepted") {
    read_view = ParamReadView::kLatestAccepted;
  } else {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_CAPTURE_INVALID"));
  }
  const bool stopped = *consistency == "stopped-published";
  if (!stopped && *consistency != "live-per-operator") {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_CAPTURE_INVALID"));
  }
  if (stopped && read_view != ParamReadView::kPublished) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_CAPTURE_INVALID"));
  }
  static constexpr std::array<std::string_view, 3> kFrontierKeys{
      "availability", "accepted_through", "published_through"};
  valid = require_exact_keys(*frontier_view, kFrontierKeys, "PARAM_ARCHIVE_FRONTIER_INVALID");
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }
  auto frontier_availability_view =
      member(*frontier_view, "availability", "PARAM_ARCHIVE_FRONTIER_INVALID");
  auto accepted_view = member(*frontier_view, "accepted_through", "PARAM_ARCHIVE_FRONTIER_INVALID");
  auto published_view =
      member(*frontier_view, "published_through", "PARAM_ARCHIVE_FRONTIER_INVALID");
  if (!frontier_availability_view) {
    return make_unexpected(std::move(frontier_availability_view.error()));
  }
  if (!accepted_view) {
    return make_unexpected(std::move(accepted_view.error()));
  }
  if (!published_view) {
    return make_unexpected(std::move(published_view.error()));
  }
  auto frontier_availability =
      string_value(*frontier_availability_view, "PARAM_ARCHIVE_FRONTIER_INVALID");
  if (!frontier_availability) {
    return make_unexpected(std::move(frontier_availability.error()));
  }
  std::optional<ParamUpdateSequence> accepted;
  std::optional<ParamUpdateSequence> published;
  if (stopped) {
    if (*frontier_availability != "global") {
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_FRONTIER_INVALID"));
    }
    auto decoded_accepted =
        structural_integer<ParamUpdateSequence>(*accepted_view, "PARAM_ARCHIVE_FRONTIER_INVALID");
    auto decoded_published =
        structural_integer<ParamUpdateSequence>(*published_view, "PARAM_ARCHIVE_FRONTIER_INVALID");
    if (!decoded_accepted || !decoded_published) {
      if (!decoded_accepted) {
        return make_unexpected(std::move(decoded_accepted.error()));
      }
      return make_unexpected(std::move(decoded_published.error()));
    }
    accepted = *decoded_accepted;
    published = *decoded_published;
  } else {
    if (*frontier_availability != "per-operator") {
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_FRONTIER_INVALID"));
    }
    auto accepted_valid =
        validate_unavailable(*accepted_view, "per-operator", "PARAM_ARCHIVE_FRONTIER_INVALID");
    auto published_valid =
        validate_unavailable(*published_view, "per-operator", "PARAM_ARCHIVE_FRONTIER_INVALID");
    if (!accepted_valid) {
      return make_unexpected(std::move(accepted_valid.error()));
    }
    if (!published_valid) {
      return make_unexpected(std::move(published_valid.error()));
    }
  }

  if (operators_view->kind() != config::ConfigValueKind::kArray ||
      operators_view->size() > kMaxArchiveRecords) {
    return make_unexpected(invalid_archive("PARAM_ARCHIVE_OPERATORS_INVALID"));
  }
  std::vector<ParamSnapshotOperatorRecord> operators;
  operators.reserve(operators_view->size());
  for (std::size_t index = 0; index < operators_view->size(); ++index) {
    auto item_view = operators_view->try_element(index);
    if (!item_view) {
      return make_unexpected(archive_config_error(ParamArchiveErrorCode::kInvalidArchive,
                                                  "PARAM_ARCHIVE_OPERATORS_INVALID",
                                                  std::move(item_view.error())));
    }
    auto item = decode_operator(*item_view, *profile);
    if (!item) {
      return make_unexpected(std::move(item.error()));
    }
    if (index != 0U && !path_less(operators.back().path, item->path)) {
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_OPERATORS_NOT_ORDERED"));
    }
    operators.push_back(std::move(*item));
  }

  ParamSnapshot snapshot = holoscan::detail::ParamSnapshotArchiveAccess::snapshot(
      *plan,
      0U,
      0U,
      *catalog,
      read_view,
      stopped ? ParamCaptureConsistency::kRequireStopped
              : ParamCaptureConsistency::kLivePerOperator,
      std::move(scope),
      accepted,
      published,
      std::move(operators));
  holoscan::detail::ParamSnapshotArchiveAccess::set_archive_provenance(
      snapshot, *profile, std::move(source_run));
  auto rebuilt = build_archive(snapshot);
  if (!rebuilt) {
    return make_unexpected(std::move(rebuilt.error()));
  }
  auto actual_digest = semantic_digest(rebuilt->envelope);
  if (!actual_digest) {
    return make_unexpected(std::move(actual_digest.error()));
  }
  if (actual_digest->sha256 != *expected_digest) {
    return make_unexpected(
        archive_error(ParamArchiveErrorCode::kDigestMismatch, "PARAM_ARCHIVE_DIGEST_MISMATCH"));
  }
  return DecodedEnvelope{.snapshot = std::move(snapshot), .semantic_digest = *actual_digest};
}

}  // namespace

ParamArchiveError::ParamArchiveError(ParamArchiveErrorCode code, ParamToolError detail)
    : code_(code), detail_(std::move(detail)) {}

std::optional<ParamArchiveIdentity> archive_identity(const ParamSnapshot& snapshot) noexcept {
  const auto identity = holoscan::detail::ParamSnapshotArchiveAccess::archive_identity(snapshot);
  if (!identity) {
    return std::nullopt;
  }
  return ParamArchiveIdentity{
      .archive_digest = config::SemanticDigest{.canonical_version = identity->semantic_version,
                                               .sha256 = identity->semantic_sha256},
      .raw_source_digest =
          config::RawSourceDigest{.algorithm_version = identity->raw_source_version,
                                  .sha256 = identity->raw_source_sha256,
                                  .byte_size = identity->raw_source_bytes}};
}

ParamArchiveResult<ParamSaveResult> save(const ParamSnapshot& snapshot,
                                         const std::filesystem::path& path,
                                         ParamSaveOptions options) {
  if (path.empty() || options.safe_projection != ParamSafeProjection::kPersistedPublicOnly) {
    return make_unexpected(
        archive_error(ParamArchiveErrorCode::kProjectionFailure, "PARAM_ARCHIVE_OPTIONS_INVALID"));
  }
  try {
    ParamSaveResult report;
    config::SemanticDigest digest;
    auto envelope = complete_envelope(snapshot, report, digest);
    if (!envelope) {
      return make_unexpected(std::move(envelope.error()));
    }
    if (options.require_complete && !report.complete_for_normal_persistence) {
      return make_unexpected(
          archive_error(ParamArchiveErrorCode::kIncomplete, "PARAM_ARCHIVE_INCOMPLETE"));
    }
    std::string canonical;
    auto appended = append_json(*envelope, canonical, options.io.output_limits.max_output_bytes);
    if (!appended) {
      return make_unexpected(std::move(appended.error()));
    }
    auto document =
        config_io::detail::parse_generated_canonical_json(canonical, options.io.output_limits);
    if (!document) {
      return make_unexpected(archive_config_error(ParamArchiveErrorCode::kPersistenceFailure,
                                                  "PARAM_ARCHIVE_CANONICAL_PARSE_FAILED",
                                                  std::move(document.error())));
    }
    auto saved = config_io::try_save(*document, path, config_io::Format::kJson, options.io);
    if (!saved) {
      return make_unexpected(archive_config_error(ParamArchiveErrorCode::kPersistenceFailure,
                                                  "PARAM_ARCHIVE_SAVE_FAILED",
                                                  std::move(saved.error())));
    }
    report.written_source_digest = saved->written_source_digest;
    return report;
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        archive_error(ParamArchiveErrorCode::kPersistenceFailure, "PARAM_ARCHIVE_ALLOC_FAILED"));
  }
}

ParamArchiveResult<ParamSnapshot> load(const std::filesystem::path& path,
                                       ParamLoadOptions options) {
  if (path.empty()) {
    return make_unexpected(
        archive_error(ParamArchiveErrorCode::kInvalidArchive, "PARAM_ARCHIVE_PATH_INVALID"));
  }
  try {
    auto document = config_io::try_load(path, config_io::Format::kJson, options.parser);
    if (!document) {
      const ParamArchiveErrorCode code = archive_load_error_code(document.error().code());
      return make_unexpected(
          archive_config_error(code, "PARAM_ARCHIVE_LOAD_FAILED", std::move(document.error())));
    }
    auto decoded = decode_envelope(document->root());
    if (!decoded) {
      return make_unexpected(std::move(decoded.error()));
    }
    const auto raw = document->digest().source;
    if (!raw) {
      return make_unexpected(invalid_archive("PARAM_ARCHIVE_SOURCE_DIGEST_MISSING"));
    }
    holoscan::detail::ParamSnapshotArchiveAccess::set_archive_identity(
        decoded->snapshot,
        holoscan::detail::ParamSnapshotArchiveAccess::ArchiveIdentity{
            .semantic_version = decoded->semantic_digest.canonical_version,
            .semantic_sha256 = decoded->semantic_digest.sha256,
            .raw_source_version = raw->algorithm_version,
            .raw_source_sha256 = raw->sha256,
            .raw_source_bytes = raw->byte_size});
    return std::move(decoded->snapshot);
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        archive_error(ParamArchiveErrorCode::kPersistenceFailure, "PARAM_ARCHIVE_ALLOC_FAILED"));
  }
}

ParamRebindResult<ParamRebindProvenance> load_and_set_params(
    Graph& graph, const std::filesystem::path& path, const ParamRebindTargetScope& target_scope,
    ParamRebindOptions options) {
  auto snapshot = load(path);
  if (!snapshot) {
    return make_unexpected(persistence_rebind_error(snapshot.error()));
  }
  return set_params(graph, *snapshot, target_scope, options);
}

}  // namespace holoscan::params
