/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/channel/endpoint_catalog.hpp>

#include "catalog_diagnostic_detail.hpp"
#include "catalog_inline_payload_contract.hpp"
#include "channel_common.hpp"
#include "pod_codec.hpp"
#include "process_identity.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#endif

#include <holoscan/core/parameter_schema.hpp>

#include "core/canonical_text_encoding.hpp"
#include "core/domain/tensor_layout_validation.hpp"
#include "core/memory_kind_validation.hpp"
#include "endpoint_catalog_state.hpp"

namespace holoscan {
namespace {

using detail::append_json_string;
using detail::CatalogClockConversionRecord;
using detail::CatalogClockRecord;
using detail::ChannelEndpointRecord;
using detail::GeneratedEndpointNameRecord;
using GeneratedNameInfoView = std::optional<GeneratedEndpointNameInfo>;

constexpr std::string_view kCatalogFormat{"holoscan.channel-endpoints"};
constexpr std::uint64_t kCatalogVersion = 2U;
constexpr std::string_view kGeneratedAlgorithm{"holoscan.auto_endpoint_name"};
constexpr std::uint64_t kGeneratedAlgorithmVersion = 1U;

// These format limits bound cold parser memory and work before any
// catalog content can become normalized plan evidence. They are format safety
// limits, not HoloMQ or Runtime capacity semantics.
constexpr std::size_t kMaximumCatalogSourceBytes = 16U * 1024U * 1024U;
constexpr std::size_t kMaximumJsonDepth = 64U;
constexpr std::size_t kMaximumJsonNodes = 100'000U;
constexpr std::size_t kMaximumJsonObjectMembers = 4'096U;
constexpr std::size_t kMaximumJsonArrayElements = 65'536U;
constexpr std::size_t kMaximumCatalogClocks = 1'024U;
constexpr std::size_t kMaximumCatalogClockConversions = 4'096U;
constexpr std::size_t kMaximumPodFields = 1'024U;
constexpr std::size_t kMaximumTensorAxes = 32U;

[[nodiscard]] Error catalog_error(ErrorCode code, std::string_view token) {
  return Error{.code = code, .message = std::string(token)};
}

[[nodiscard]] Error catalog_resource_limit_error(std::string_view path, std::string_view resource,
                                                 std::size_t limit, std::size_t actual) {
  std::string message{"ENDPOINT_CATALOG_RESOURCE_LIMIT_EXCEEDED|path="};
  message += path;
  message += "|resource=";
  message += resource;
  message += "|limit=";
  message += std::to_string(limit);
  message += "|actual=";
  message += std::to_string(actual);
  return Error{.code = ErrorCode::kResourceExhausted, .message = std::move(message)};
}

[[nodiscard]] Error catalog_inline_capacity_error(
    const std::size_t endpoint_index, const std::string_view endpoint_name,
    const channel_internal::CatalogInlinePayloadContract& contract) {
  std::string message{"ENDPOINT_CATALOG_INLINE_CAPACITY_EXCEEDED|path=/endpoints/"};
  message += std::to_string(endpoint_index);
  message += "/payload|endpoint=";
  channel_internal::append_escaped_detail_value(message, endpoint_name);
  message += "|application_capacity=";
  message += std::to_string(contract.native_payload_capacity_bytes);
  message += "|framing_overhead=";
  message += std::to_string(contract.framing_overhead_bytes);
  message += "|required_frame=";
  message += std::to_string(contract.required_frame_bytes);
  message += "|maximum_frame=";
  message += std::to_string(channel_internal::kMaximumCatalogDataFrameBytes);
  return Error{.code = ErrorCode::kResourceExhausted, .message = std::move(message)};
}

[[nodiscard]] Error raw_clock_health_error(
    std::size_t endpoint_index, std::string_view endpoint_name, std::string_view capability,
    std::string_view requirement, std::string_view max_report_age, std::string_view direction) {
  std::string message{"ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID|path=/endpoints/"};
  message += std::to_string(endpoint_index);
  message += "/clock_health|endpoint=";
  channel_internal::append_escaped_detail_value(message, endpoint_name);
  message +=
      "|required=unavailable+not_required(age=absent,any_role)|status_reports+fresh_"
      "healthy_report(age=positive,external_publisher)|provided=capability=";
  channel_internal::append_escaped_detail_value(message, capability);
  message += ";requirement=";
  channel_internal::append_escaped_detail_value(message, requirement);
  message += ";max_report_age_ns=";
  channel_internal::append_escaped_detail_value(message, max_report_age);
  message += ";role=";
  message += direction == "import" ? "external_publisher" : "external_subscriber";
  return Error{.code = ErrorCode::kInvalidArgument, .message = std::move(message)};
}

template <typename Enum>
[[nodiscard]] constexpr bool closed_enum(Enum value, Enum last) noexcept {
  using Underlying = std::underlying_type_t<Enum>;
  return static_cast<Underlying>(value) <= static_cast<Underlying>(last);
}

[[nodiscard]] constexpr std::string_view queue_policy_name(QueuePolicy policy) noexcept {
  switch (policy) {
    case QueuePolicy::kBlocking:
      return "blocking";
    case QueuePolicy::kDropOldest:
      return "drop_oldest";
  }
  return "unsupported";
}

[[nodiscard]] constexpr bool supported_queue_policy(QueuePolicy policy) noexcept {
  return policy == QueuePolicy::kBlocking || policy == QueuePolicy::kDropOldest;
}

[[nodiscard]] bool decode_utf8(std::string_view text, std::size_t& offset,
                               std::uint32_t& code_point) noexcept {
  if (offset >= text.size()) {
    return false;
  }
  const auto lead = static_cast<std::uint8_t>(text[offset]);
  std::size_t count = 0U;
  if (lead <= 0x7FU) {
    count = 1U;
    code_point = lead;
  } else if (lead >= 0xC2U && lead <= 0xDFU) {
    count = 2U;
    code_point = lead & 0x1FU;
  } else if (lead >= 0xE0U && lead <= 0xEFU) {
    count = 3U;
    code_point = lead & 0x0FU;
  } else if (lead >= 0xF0U && lead <= 0xF4U) {
    count = 4U;
    code_point = lead & 0x07U;
  } else {
    return false;
  }
  if (count > text.size() - offset) {
    return false;
  }
  for (std::size_t index = 1U; index < count; ++index) {
    const auto continuation = static_cast<std::uint8_t>(text[offset + index]);
    if ((continuation & 0xC0U) != 0x80U) {
      return false;
    }
    code_point = (code_point << 6U) | (continuation & 0x3FU);
  }
  if ((count == 3U && code_point < 0x800U) || (count == 4U && code_point < 0x10000U) ||
      code_point > 0x10FFFFU || (code_point >= 0xD800U && code_point <= 0xDFFFU)) {
    return false;
  }
  offset += count;
  return true;
}

[[nodiscard]] bool valid_utf8(std::string_view text) noexcept {
  std::size_t offset = 0U;
  while (offset < text.size()) {
    std::uint32_t code_point{};
    if (!decode_utf8(text, offset, code_point)) {
      return false;
    }
  }
  return true;
}

void append_utf8(std::string& output, std::uint32_t code_point) {
  if (code_point <= 0x7FU) {
    output.push_back(static_cast<char>(code_point));
  } else if (code_point <= 0x7FFU) {
    output.push_back(static_cast<char>(0xC0U | (code_point >> 6U)));
    output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
  } else if (code_point <= 0xFFFFU) {
    output.push_back(static_cast<char>(0xE0U | (code_point >> 12U)));
    output.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
  } else {
    output.push_back(static_cast<char>(0xF0U | (code_point >> 18U)));
    output.push_back(static_cast<char>(0x80U | ((code_point >> 12U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
  }
}

struct JsonObjectMember;

struct JsonValue {
  enum class Kind : std::uint8_t { kNull, kBoolean, kInteger, kString, kArray, kObject };
  Kind kind{Kind::kNull};
  bool boolean{};
  std::string integer;
  std::string string;
  std::vector<JsonValue> array;
  std::vector<JsonObjectMember> object;
};

struct JsonObjectMember {
  std::string first;
  JsonValue second;
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view source) noexcept : source_(source) {}

  [[nodiscard]] expected<JsonValue, Error> parse() {
    skip_space();
    JsonValue value;
    if (!parse_value(value, 1U)) {
      return make_unexpected(catalog_error(
          error_code_,
          error_.empty() ? std::string_view{"ENDPOINT_CATALOG_PARSE_FAILED"} : error_));
    }
    skip_space();
    if (position_ != source_.size()) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_PARSE_FAILED"));
    }
    return value;
  }

 private:
  void skip_space() noexcept {
    while (position_ < source_.size() &&
           (source_[position_] == ' ' || source_[position_] == '\t' || source_[position_] == '\n' ||
            source_[position_] == '\r')) {
      ++position_;
    }
  }

  [[nodiscard]] bool consume(char expected) noexcept {
    if (position_ >= source_.size() || source_[position_] != expected) {
      return false;
    }
    ++position_;
    return true;
  }

  [[nodiscard]] bool parse_value(JsonValue& value, std::size_t depth) {
    if (depth > kMaximumJsonDepth) {
      return fail_resource("/", "nesting_depth", kMaximumJsonDepth, depth);
    }
    if (node_count_ >= kMaximumJsonNodes) {
      return fail_resource("/", "json_nodes", kMaximumJsonNodes, node_count_ + 1U);
    }
    ++node_count_;
    skip_space();
    if (position_ >= source_.size()) {
      return fail("ENDPOINT_CATALOG_PARSE_FAILED");
    }
    switch (source_[position_]) {
      case '{':
        return parse_object(value, depth);
      case '[':
        return parse_array(value, depth);
      case '"':
        value.kind = JsonValue::Kind::kString;
        return parse_string(value.string);
      case 't':
        value.kind = JsonValue::Kind::kBoolean;
        value.boolean = true;
        return consume_literal("true");
      case 'f':
        value.kind = JsonValue::Kind::kBoolean;
        value.boolean = false;
        return consume_literal("false");
      case 'n':
        value.kind = JsonValue::Kind::kNull;
        return consume_literal("null");
      default:
        value.kind = JsonValue::Kind::kInteger;
        return parse_integer(value.integer);
    }
  }

  [[nodiscard]] bool consume_literal(std::string_view literal) noexcept {
    if (source_.substr(position_, literal.size()) != literal) {
      return fail("ENDPOINT_CATALOG_PARSE_FAILED");
    }
    position_ += literal.size();
    return true;
  }

  [[nodiscard]] bool parse_object(JsonValue& value, std::size_t depth) {
    value.kind = JsonValue::Kind::kObject;
    if (!consume('{')) {
      return fail("ENDPOINT_CATALOG_PARSE_FAILED");
    }
    skip_space();
    if (consume('}')) {
      return true;
    }
    std::set<std::string> keys;
    while (true) {
      if (value.object.size() >= kMaximumJsonObjectMembers) {
        return fail_resource(
            "/", "object_members", kMaximumJsonObjectMembers, value.object.size() + 1U);
      }
      std::string key;
      if (!parse_string(key)) {
        return false;
      }
      if (!keys.insert(key).second) {
        return fail("ENDPOINT_CATALOG_DUPLICATE_KEY");
      }
      skip_space();
      if (!consume(':')) {
        return fail("ENDPOINT_CATALOG_PARSE_FAILED");
      }
      JsonValue child;
      if (!parse_value(child, depth + 1U)) {
        return false;
      }
      value.object.emplace_back(std::move(key), std::move(child));
      skip_space();
      if (consume('}')) {
        return true;
      }
      if (!consume(',')) {
        return fail("ENDPOINT_CATALOG_PARSE_FAILED");
      }
      skip_space();
    }
  }

  [[nodiscard]] bool parse_array(JsonValue& value, std::size_t depth) {
    value.kind = JsonValue::Kind::kArray;
    if (!consume('[')) {
      return fail("ENDPOINT_CATALOG_PARSE_FAILED");
    }
    skip_space();
    if (consume(']')) {
      return true;
    }
    while (true) {
      if (value.array.size() >= kMaximumJsonArrayElements) {
        return fail_resource(
            "/", "array_elements", kMaximumJsonArrayElements, value.array.size() + 1U);
      }
      JsonValue child;
      if (!parse_value(child, depth + 1U)) {
        return false;
      }
      value.array.push_back(std::move(child));
      skip_space();
      if (consume(']')) {
        return true;
      }
      if (!consume(',')) {
        return fail("ENDPOINT_CATALOG_PARSE_FAILED");
      }
      skip_space();
    }
  }

  [[nodiscard]] static int hex_digit(char value) noexcept {
    if (value >= '0' && value <= '9') {
      return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
      return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
      return value - 'A' + 10;
    }
    return -1;
  }

  [[nodiscard]] bool parse_hex_quad(std::uint32_t& value) noexcept {
    if (source_.size() - position_ < 4U) {
      return false;
    }
    value = 0U;
    for (std::size_t index = 0U; index < 4U; ++index) {
      const int digit = hex_digit(source_[position_++]);
      if (digit < 0) {
        return false;
      }
      value = (value << 4U) | static_cast<std::uint32_t>(digit);
    }
    return true;
  }

  [[nodiscard]] bool parse_unicode_escape(std::string& output) {
    std::uint32_t code_point{};
    if (!parse_hex_quad(code_point)) {
      return fail("ENDPOINT_CATALOG_PARSE_FAILED");
    }
    if (code_point >= 0xD800U && code_point <= 0xDBFFU) {
      if (source_.size() - position_ < 6U || source_[position_] != '\\' ||
          source_[position_ + 1U] != 'u') {
        return fail("ENDPOINT_CATALOG_PARSE_FAILED");
      }
      position_ += 2U;
      std::uint32_t low{};
      if (!parse_hex_quad(low) || low < 0xDC00U || low > 0xDFFFU) {
        return fail("ENDPOINT_CATALOG_PARSE_FAILED");
      }
      code_point = 0x10000U + ((code_point - 0xD800U) << 10U) + (low - 0xDC00U);
    } else if (code_point >= 0xDC00U && code_point <= 0xDFFFU) {
      return fail("ENDPOINT_CATALOG_PARSE_FAILED");
    }
    append_utf8(output, code_point);
    return true;
  }

  [[nodiscard]] bool parse_string(std::string& output) {
    skip_space();
    if (!consume('"')) {
      return fail("ENDPOINT_CATALOG_PARSE_FAILED");
    }
    while (position_ < source_.size()) {
      const char value = source_[position_++];
      if (value == '"') {
        return valid_utf8(output) || fail("ENDPOINT_CATALOG_PARSE_FAILED");
      }
      if (static_cast<std::uint8_t>(value) < 0x20U) {
        return fail("ENDPOINT_CATALOG_PARSE_FAILED");
      }
      if (value != '\\') {
        output.push_back(value);
        continue;
      }
      if (position_ >= source_.size()) {
        return fail("ENDPOINT_CATALOG_PARSE_FAILED");
      }
      switch (source_[position_++]) {
        case '"':
          output.push_back('"');
          break;
        case '\\':
          output.push_back('\\');
          break;
        case '/':
          output.push_back('/');
          break;
        case 'b':
          output.push_back('\b');
          break;
        case 'f':
          output.push_back('\f');
          break;
        case 'n':
          output.push_back('\n');
          break;
        case 'r':
          output.push_back('\r');
          break;
        case 't':
          output.push_back('\t');
          break;
        case 'u':
          if (!parse_unicode_escape(output)) {
            return false;
          }
          break;
        default:
          return fail("ENDPOINT_CATALOG_PARSE_FAILED");
      }
    }
    return fail("ENDPOINT_CATALOG_PARSE_FAILED");
  }

  [[nodiscard]] bool parse_integer(std::string& output) {
    const std::size_t first = position_;
    if (position_ < source_.size() && source_[position_] == '-') {
      ++position_;
    }
    if (position_ >= source_.size() || source_[position_] < '0' || source_[position_] > '9') {
      return fail("ENDPOINT_CATALOG_PARSE_FAILED");
    }
    if (source_[position_] == '0') {
      ++position_;
      if (position_ < source_.size() && source_[position_] >= '0' && source_[position_] <= '9') {
        return fail("ENDPOINT_CATALOG_PARSE_FAILED");
      }
    } else {
      while (position_ < source_.size() && source_[position_] >= '0' && source_[position_] <= '9') {
        ++position_;
      }
    }
    if (position_ < source_.size() &&
        (source_[position_] == '.' || source_[position_] == 'e' || source_[position_] == 'E')) {
      return fail("ENDPOINT_CATALOG_CANONICALIZATION_UNSUPPORTED");
    }
    output = std::string(source_.substr(first, position_ - first));
    return true;
  }

  [[nodiscard]] bool fail(std::string_view token) noexcept {
    if (error_.empty()) {
      error_ = token;
    }
    return false;
  }

  [[nodiscard]] bool fail_resource(std::string_view path, std::string_view resource,
                                   std::size_t limit, std::size_t actual) {
    if (error_.empty()) {
      Error error = catalog_resource_limit_error(path, resource, limit, actual);
      error_code_ = error.code;
      error_ = std::move(error.message);
    }
    return false;
  }

  std::string_view source_;
  std::size_t position_{};
  std::size_t node_count_{};
  ErrorCode error_code_{ErrorCode::kInvalidArgument};
  std::string error_;
};

[[nodiscard]] const JsonValue* json_field(const JsonValue& object, std::string_view name) noexcept {
  if (object.kind != JsonValue::Kind::kObject) {
    return nullptr;
  }
  const auto found = std::find_if(object.object.begin(),
                                  object.object.end(),
                                  [&](const auto& item) { return item.first == name; });
  return found == object.object.end() ? nullptr : &found->second;
}

[[nodiscard]] bool allowed_fields(const JsonValue& object,
                                  std::span<const std::string_view> allowed) noexcept {
  return object.kind == JsonValue::Kind::kObject &&
         std::all_of(object.object.begin(), object.object.end(), [&](const auto& item) {
           return std::find(allowed.begin(), allowed.end(), item.first) != allowed.end();
         });
}

[[nodiscard]] bool json_unsigned(const JsonValue& value, std::uint64_t& result) noexcept {
  if (value.kind != JsonValue::Kind::kInteger || value.integer.empty() ||
      value.integer.front() == '-') {
    return false;
  }
  const auto conversion =
      std::from_chars(value.integer.data(), value.integer.data() + value.integer.size(), result);
  return conversion.ec == std::errc() &&
         conversion.ptr == value.integer.data() + value.integer.size();
}

template <typename Byte, std::size_t N>
[[nodiscard]] bool parse_hex_bytes(std::string_view value, std::array<Byte, N>& result,
                                   bool sha256_prefix) noexcept {
  const std::string_view prefix = sha256_prefix ? std::string_view{"sha256:"} : std::string_view{};
  if (!value.starts_with(prefix) || value.size() != prefix.size() + (N * 2U)) {
    return false;
  }
  const auto digit = [](char character) noexcept -> int {
    if (character >= '0' && character <= '9') {
      return character - '0';
    }
    if (character >= 'a' && character <= 'f') {
      return character - 'a' + 10;
    }
    return -1;
  };
  for (std::size_t index = 0U; index < N; ++index) {
    const int high = digit(value[prefix.size() + (index * 2U)]);
    const int low = digit(value[prefix.size() + (index * 2U) + 1U]);
    if (high < 0 || low < 0) {
      return false;
    }
    const auto byte = static_cast<std::uint8_t>((high << 4U) | low);
    if constexpr (std::is_same_v<Byte, std::byte>) {
      result[index] = static_cast<std::byte>(byte);
    } else {
      result[index] = static_cast<Byte>(byte);
    }
  }
  return true;
}

[[nodiscard]] bool unicode_whitespace(std::uint32_t code_point) noexcept {
  return (code_point >= 0x0009U && code_point <= 0x000DU) || code_point == 0x0020U ||
         code_point == 0x0085U || code_point == 0x00A0U || code_point == 0x1680U ||
         (code_point >= 0x2000U && code_point <= 0x200AU) || code_point == 0x2028U ||
         code_point == 0x2029U || code_point == 0x202FU || code_point == 0x205FU ||
         code_point == 0x3000U;
}

[[nodiscard]] bool has_edge_whitespace(std::string_view value) noexcept {
  if (value.empty() || !valid_utf8(value)) {
    return false;
  }
  std::size_t offset = 0U;
  std::uint32_t first{};
  std::uint32_t last{};
  bool is_first = true;
  while (offset < value.size()) {
    std::uint32_t code_point{};
    if (!decode_utf8(value, offset, code_point)) {
      return false;
    }
    if (is_first) {
      first = code_point;
      is_first = false;
    }
    last = code_point;
  }
  return unicode_whitespace(first) || unicode_whitespace(last);
}

[[nodiscard]] bool endpoint_name_valid(std::string_view name) noexcept {
  if (name.empty() || name.size() > 255U || !valid_utf8(name) || has_edge_whitespace(name)) {
    return false;
  }
  std::size_t offset = 0U;
  while (offset < name.size()) {
    std::uint32_t code_point{};
    if (!decode_utf8(name, offset, code_point) || code_point <= 0x001FU ||
        (code_point >= 0x007FU && code_point <= 0x009FU)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool portable_identity_name_valid(std::string_view name) noexcept {
  if (name.empty() || name.size() > 255U || !valid_utf8(name)) {
    return false;
  }
  return std::none_of(name.begin(), name.end(), [](char character) {
    const auto byte = static_cast<std::uint8_t>(character);
    return byte == 0U || byte < 0x20U || byte == 0x7FU;
  });
}

[[nodiscard]] constexpr std::uint32_t pod_scalar_size(PodScalarKind kind) noexcept {
  switch (kind) {
    case PodScalarKind::kUInt8:
    case PodScalarKind::kInt8:
      return 1U;
    case PodScalarKind::kUInt16:
    case PodScalarKind::kInt16:
      return 2U;
    case PodScalarKind::kUInt32:
    case PodScalarKind::kInt32:
    case PodScalarKind::kFloat32:
      return 4U;
    case PodScalarKind::kUInt64:
    case PodScalarKind::kInt64:
    case PodScalarKind::kFloat64:
      return 8U;
  }
  return 0U;
}

[[nodiscard]] std::string_view pod_scalar_token(PodScalarKind kind) noexcept;
[[nodiscard]] std::array<std::byte, 32> hash_bytes(std::string_view bytes) noexcept;

template <typename Byte>
void append_hex_bytes(std::string& output, std::span<const Byte> bytes, bool sha256_prefix) {
  constexpr std::array<char, 16> kHex{
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  output.push_back('"');
  if (sha256_prefix) {
    output += "sha256:";
  }
  for (const Byte value : bytes) {
    const std::uint8_t byte = [&] {
      if constexpr (std::is_same_v<Byte, std::byte>) {
        return std::to_integer<std::uint8_t>(value);
      } else {
        return static_cast<std::uint8_t>(value);
      }
    }();
    output.push_back(kHex[byte >> 4U]);
    output.push_back(kHex[byte & 0x0FU]);
  }
  output.push_back('"');
}

[[nodiscard]] std::string_view scalar_identity_token(ScalarIdentityKind kind) noexcept {
  switch (kind) {
    case ScalarIdentityKind::kBoolean:
      return "bool";
    case ScalarIdentityKind::kUInt8:
      return "u8";
    case ScalarIdentityKind::kInt8:
      return "i8";
    case ScalarIdentityKind::kUInt16:
      return "u16";
    case ScalarIdentityKind::kInt16:
      return "i16";
    case ScalarIdentityKind::kUInt32:
      return "u32";
    case ScalarIdentityKind::kInt32:
      return "i32";
    case ScalarIdentityKind::kUInt64:
      return "u64";
    case ScalarIdentityKind::kInt64:
      return "i64";
    case ScalarIdentityKind::kFloat32:
      return "f32";
    case ScalarIdentityKind::kFloat64:
      return "f64";
    case ScalarIdentityKind::kLocalEnum:
      return "local_enum";
  }
  return {};
}

[[nodiscard]] std::string_view memory_conversion_token(
    std::optional<MemoryConversionPolicy> policy) noexcept {
  switch (policy.value_or(MemoryConversionPolicy::kDefault)) {
    case MemoryConversionPolicy::kDefault:
      return "default";
    case MemoryConversionPolicy::kDisable:
      return "disable";
    case MemoryConversionPolicy::kAllowIfSafe:
      return "allow_if_safe";
    case MemoryConversionPolicy::kRequire:
      return "require";
  }
  return {};
}

[[nodiscard]] std::optional<MemoryConversionPolicy> parse_memory_conversion_token(
    std::string_view token) noexcept {
  if (token == "default") {
    return MemoryConversionPolicy::kDefault;
  }
  if (token == "disable") {
    return MemoryConversionPolicy::kDisable;
  }
  if (token == "allow_if_safe") {
    return MemoryConversionPolicy::kAllowIfSafe;
  }
  if (token == "require") {
    return MemoryConversionPolicy::kRequire;
  }
  return std::nullopt;
}

[[nodiscard]] std::string_view memory_kind_token(MemoryKind kind) noexcept {
  switch (kind) {
    case MemoryKind::kUnknown:
      return "unknown";
    case MemoryKind::kHost:
      return "host";
    case MemoryKind::kPinnedHost:
      return "pinned_host";
    case MemoryKind::kCudaDevice:
      return "cuda_device";
    case MemoryKind::kCudaManaged:
      return "cuda_managed";
  }
  return {};
}

template <typename T>
void append_optional_unsigned(std::string& output, const std::optional<T>& value) {
  if (value) {
    output += std::to_string(*value);
  } else {
    output += "null";
  }
}

void append_pod_identity(std::string& output, const detail::PodCatalogIdentityRecord& pod) {
  output += "{\"name\":";
  append_json_string(output, pod.identity.canonical_name);
  output += ",\"version\":" + std::to_string(pod.identity.version);
  output += ",\"digest\":";
  append_hex_bytes(output, std::span<const std::uint8_t>{pod.digest_bytes}, true);
  output += ",\"key\":";
  append_hex_bytes(output, std::span<const std::uint8_t>{pod.key.bytes}, false);
  output += ",\"object_size\":" + std::to_string(pod.identity.object_size);
  output += ",\"object_alignment\":" + std::to_string(pod.identity.object_alignment);
  output += ",\"layout_format_version\":1,\"layout_digest\":";
  append_hex_bytes(output, std::span<const std::uint8_t>{pod.layout_digest_bytes}, true);
  output += ",\"integer_wire_endianness\":\"little\",\"floating_wire_format\":\"ieee754\"";
  output += ",\"fields\":[";
  for (std::size_t index = 0U; index < pod.identity.fields.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    const auto& field = pod.identity.fields[index];
    output += "{\"name\":";
    append_json_string(output, field.name);
    output += ",\"byte_offset\":" + std::to_string(field.byte_offset);
    output += ",\"element_count\":" + std::to_string(field.element_count);
    output += ",\"scalar\":";
    append_json_string(output, pod_scalar_token(field.scalar_kind));
    output.push_back('}');
  }
  output += "],\"compatibility_evidence\":[]}";
}

void append_schema_identity(std::string& output, const SchemaIdentityRecord& schema) {
  output += "{\"canonical_name\":";
  append_json_string(output, schema.canonical_name);
  output += ",\"compatibility_epoch\":" + std::to_string(schema.compatibility_epoch);
  output += ",\"family_id\":";
  append_hex_bytes(output, std::span<const std::byte>{schema.family_id.value}, false);
  output += ",\"file_identifier\":";
  append_json_string(
      output,
      std::string_view{schema.file_identifier.value.data(), schema.file_identifier.value.size()});
  output.push_back('}');
}

void append_tensor_contract(std::string& output, const TensorBoundaryContractRecord& tensor,
                            const bool include_memory_conversion) {
  output += "{\"role\":\"external_boundary\",\"representation\":{\"memory_kind\":";
  if (tensor.memory_kind) {
    append_json_string(output, memory_kind_token(*tensor.memory_kind));
  } else {
    output += "null";
  }
  output += ",\"dtype\":";
  if (tensor.dtype) {
    output += "{\"code\":" + std::to_string(tensor.dtype->code);
    output += ",\"bits\":" + std::to_string(tensor.dtype->bits);
    output += ",\"lanes\":" + std::to_string(tensor.dtype->lanes) + "}";
  } else {
    output += "null";
  }
  output += ",\"rank\":";
  append_optional_unsigned(output, tensor.rank);
  output += ",\"strides\":";
  append_json_string(output,
                     tensor.strides == TensorStridePolicy::kBoundedStrided ? "bounded_strided"
                                                                           : "contiguous_only");
  output += "},\"bounds\":{\"max_rank\":";
  append_optional_unsigned(output, tensor.max_rank);
  output += ",\"max_byte_span\":";
  append_optional_unsigned(output, tensor.max_byte_span);
  output += ",\"axes\":[";
  for (std::size_t index = 0U; index < tensor.axes.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    output += "{\"max_extent\":" + std::to_string(tensor.axes[index].max_extent);
    output +=
        ",\"max_abs_stride_bytes\":" + std::to_string(tensor.axes[index].max_abs_stride_bytes) +
        "}";
  }
  output += "]},\"transfer\":{\"foreign_emit_policy\":";
  switch (tensor.foreign_emit_policy) {
    case InterfaceTensorEmitPolicy::kRequireExportable:
      append_json_string(output, "require_exportable");
      break;
    case InterfaceTensorEmitPolicy::kCopyIfNeeded:
      append_json_string(output, "copy_if_needed");
      break;
    case InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded:
      append_json_string(output, "inline_small_host_then_copy_if_needed");
      break;
  }
  output += ",\"max_inline_bytes\":";
  append_optional_unsigned(output, tensor.max_inline_bytes);
  output += ",\"max_staging_byte_span\":";
  append_optional_unsigned(output, tensor.max_staging_byte_span);
  output.push_back('}');
  if (include_memory_conversion) {
    output += ",\"memory_conversion\":";
    if (tensor.memory_conversion) {
      append_json_string(output, memory_conversion_token(tensor.memory_conversion));
    } else {
      output += "null";
    }
  }
  output.push_back('}');
}

// Keep the closed payload-family serialization switch adjacent for auditability.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void append_payload(std::string& output, const PayloadIdentityRecord& payload_identity,
                    const std::optional<TensorBoundaryContractRecord>& tensor,
                    std::span<const DescriptorTensorTopicFieldRecord> tensor_fields,
                    const std::optional<SchemaIdentityRecord>& schema_identity,
                    const std::optional<detail::PodCatalogIdentityRecord>& pod_identity,
                    const std::uint64_t schema_writer_max_serialized_bytes) {
  output += "{\"family\":";
  switch (payload_identity.kind) {
    case PayloadIdentityKind::kScalar:
      append_json_string(output, "scalar");
      output += ",\"scalar\":";
      append_json_string(output,
                         payload_identity.scalar
                             ? scalar_identity_token(payload_identity.scalar->kind)
                             : std::string_view{});
      break;
    case PayloadIdentityKind::kTensor:
      append_json_string(output, "tensor");
      output += ",\"tensor\":";
      if (!tensor) {
        output += "null";
      } else {
        append_tensor_contract(output, *tensor, true);
      }
      break;
    case PayloadIdentityKind::kSchema:
      append_json_string(output, "schema");
      output +=
          ",\"transport_profile_tag\":" + std::to_string(payload_identity.transport_profile_tag);
      output +=
          ",\"writer_max_serialized_bytes\":" + std::to_string(schema_writer_max_serialized_bytes);
      if (payload_identity.payload_kind == PayloadKind::kSerializedBytes) {
        const auto capacity = channel_internal::catalog_inline_payload_contract(
            payload_identity, schema_writer_max_serialized_bytes);
        output += ",\"required_frame_bytes\":" +
                  std::to_string(capacity ? capacity->required_frame_bytes : 0U);
        output += ",\"selected_inline_bucket_bytes\":" +
                  std::to_string(capacity ? capacity->selected_bucket_bytes : 0U);
        output += ",\"bucket_usable_serialized_bytes\":" +
                  std::to_string(capacity ? capacity->bucket_usable_native_payload_bytes : 0U);
      } else {
        const auto selected_bucket =
            schema_writer_max_serialized_bytes <= std::numeric_limits<std::size_t>::max()
                ? channel_internal::inline_payload_bucket_size(
                      static_cast<std::size_t>(schema_writer_max_serialized_bytes))
                : std::nullopt;
        output += ",\"required_frame_bytes\":" + std::to_string(schema_writer_max_serialized_bytes);
        output +=
            ",\"selected_inline_bucket_bytes\":" + std::to_string(selected_bucket.value_or(0U));
        output +=
            ",\"bucket_usable_serialized_bytes\":" + std::to_string(selected_bucket.value_or(0U));
      }
      output += ",\"schema_identity\":";
      if (schema_identity) {
        append_schema_identity(output, *schema_identity);
      } else {
        output += "null";
      }
      if (!tensor_fields.empty()) {
        output += ",\"descriptor_tensor_fields\":[";
        for (std::size_t index = 0U; index < tensor_fields.size(); ++index) {
          if (index != 0U) {
            output.push_back(',');
          }
          output += "{\"field_id\":" + std::to_string(tensor_fields[index].field_id);
          output += ",\"tensor_index\":" + std::to_string(tensor_fields[index].tensor_index);
          output += ",\"contract\":";
          append_tensor_contract(output, tensor_fields[index].tensor, true);
          output.push_back('}');
        }
        output.push_back(']');
      }
      break;
    case PayloadIdentityKind::kPod:
      append_json_string(output, "pod");
      output += ",\"pod_identity\":";
      if (pod_identity) {
        append_pod_identity(output, *pod_identity);
      } else {
        output += "null";
      }
      break;
  }
  output.push_back('}');
}

void append_payload(std::string& output, const ChannelEndpointRecord& endpoint) {
  append_payload(output,
                 endpoint.payload_identity,
                 endpoint.tensor,
                 endpoint.tensor_fields,
                 endpoint.schema_identity,
                 endpoint.pod_identity,
                 endpoint.schema_writer_max_serialized_bytes);
}

void append_payload(std::string& output, const detail::ResolvedServiceStaticRequirements& service) {
  append_payload(output,
                 service.payload_identity,
                 service.tensor,
                 service.tensor_fields,
                 service.schema_identity,
                 service.pod_identity,
                 service.schema_writer_max_serialized_bytes);
}

[[nodiscard]] std::string service_binding(
    const detail::ResolvedServiceStaticRequirements& service) {
  std::string service_contract;
  service_contract += service.id;
  service_contract += "|";
  append_payload(service_contract, service);
  service_contract += "|";
  service_contract += queue_policy_name(service.queue_policy);
  service_contract += "|";
  service_contract += std::to_string(service.max_subscriber_depth);
  service_contract += "|";
  service_contract += std::to_string(service.max_subscriber_borrowed);
  service_contract += "|";
  service_contract += std::to_string(service.publisher_history_depth);
  service_contract += service.event_wake ? "|event" : "|no_event";
  const auto binding_digest = hash_bytes(service_contract);
  const std::string binding_sha =
      detail::lowercase_hex_text(binding_digest, detail::HexPrefix::kSha256);
  return "hsc5_local_v1_" + binding_sha.substr(7U, 16U);
}

[[nodiscard]] std::string service_binding(const ChannelEndpointRecord& endpoint) {
  return service_binding(endpoint.service_static);
}

void append_service(std::string& output, const ChannelEndpointRecord& endpoint) {
  const auto& service = endpoint.service_static;
  output += "{\"id\":";
  append_json_string(output, service.id);
  output += ",\"binding\":";
  append_json_string(output, service_binding(service));
  output += ",\"payload\":";
  append_payload(output, service);
  output += ",\"queue\":{\"policy\":\"";
  output += queue_policy_name(service.queue_policy);
  output += "\",\"max_subscriber_depth\":" + std::to_string(service.max_subscriber_depth);
  output += ",\"max_subscriber_borrowed\":" + std::to_string(service.max_subscriber_borrowed);
  output += ",\"publisher_history_depth\":" + std::to_string(service.publisher_history_depth) +
            "},\"event_wake\":";
  output += service.event_wake ? "true}" : "false}";
}

void append_endpoint(std::string& output, const ChannelEndpointRecord& endpoint) {
  output += "{\"name\":";
  append_json_string(output, endpoint.name);
  output += ",\"name_origin\":";
  append_json_string(
      output, endpoint.name_origin == EndpointNameOrigin::kGenerated ? "generated" : "explicit");
  output += ",\"generated\":";
  if (endpoint.generated_name) {
    output += "{\"algorithm\":";
    append_json_string(output, endpoint.generated_name->algorithm);
    output += ",\"version\":" + std::to_string(endpoint.generated_name->version) + "}";
  } else {
    output += "null";
  }
  output += ",\"uri\":";
  append_json_string(output, endpoint.uri);
  output += ",\"direction\":";
  append_json_string(output, endpoint.direction == TopicDirection::kImport ? "import" : "export");
  output += ",\"service\":";
  append_json_string(output, endpoint.service);
  output += ",\"memory_conversion\":";
  append_json_string(output, memory_conversion_token(endpoint.memory_conversion));
  output += ",\"publisher\":";
  if (endpoint.publisher_request && endpoint.publisher_request->max_loaned_samples) {
    output += "{\"max_loaned_samples\":" +
              std::to_string(*endpoint.publisher_request->max_loaned_samples) + "}";
  } else {
    output += "null";
  }
  output += ",\"subscriber\":";
  if (endpoint.subscriber_request) {
    output += "{\"queue_depth\":" + std::to_string(endpoint.subscriber_request->queue_depth);
    output += ",\"max_borrowed_samples\":" +
              std::to_string(endpoint.subscriber_request->max_borrowed_samples) + "}";
  } else {
    output += "null";
  }
  output += ",\"capture_time\":";
  if (endpoint.capture_time_clock) {
    output += "{\"required\":true,\"clock\":";
    append_json_string(output, endpoint.capture_time_clock->name());
    output.push_back('}');
  } else {
    output += "null";
  }
  output += ",\"clock_health\":{\"capability\":";
  const ClockHealthOptions& clock_health =
      endpoint.publisher_request ? endpoint.publisher_request->clock_health : ClockHealthOptions{};
  append_json_string(output,
                     clock_health.capability == ClockHealthCapability::kStatusReports
                         ? "status_reports"
                         : "unavailable");
  output += ",\"requirement\":";
  append_json_string(output,
                     clock_health.requirement == ClockHealthRequirement::kFreshHealthyReport
                         ? "fresh_healthy_report"
                         : "not_required");
  if (clock_health.max_report_age) {
    output += ",\"max_report_age_ns\":" + std::to_string(clock_health.max_report_age->count());
  }
  output += "},\"attach\":{\"peer\":";
  append_json_string(output,
                     endpoint.attach.peer == PeerRequirement::kRequired ? "required" : "optional");
  output += ",\"max_wait_ns\":" +
            std::to_string(endpoint.attach.max_wait.value_or(std::chrono::nanoseconds{0}).count()) +
            "}}";
}

void append_clock(std::string& output, const CatalogClockRecord& clock) {
  output += "{\"id\":" + std::to_string(clock.id) + ",\"name\":";
  append_json_string(output, clock.key.name());
  output += ",\"kind\":";
  append_json_string(output, clock.kind);
  output += ",\"unit\":";
  append_json_string(output, clock.unit);
  output += ",\"epoch\":";
  append_json_string(output, clock.epoch);
  output += ",\"monotonicity\":";
  append_json_string(output, clock.monotonicity);
  output += ",\"discipline\":{\"kind\":";
  append_json_string(output, clock.discipline_kind);
  output += ",\"identity\":";
  append_json_string(output, clock.discipline_identity);
  output += "}}";
}

void append_conversion(std::string& output, const CatalogClockConversionRecord& conversion) {
  output += "{\"from\":";
  append_json_string(output, conversion.from.name());
  output += ",\"to\":";
  append_json_string(output, conversion.to.name());
  output += ",\"scale_numerator\":";
  append_json_string(output, std::to_string(conversion.scale_numerator));
  output += ",\"scale_denominator\":";
  append_json_string(output, std::to_string(conversion.scale_denominator));
  output += ",\"offset_ns\":";
  append_json_string(output, std::to_string(conversion.offset_ns));
  output += ",\"rounding\":\"ties_to_even\",\"admitted_min\":";
  append_json_string(output, std::to_string(conversion.admitted_min));
  output += ",\"admitted_max\":";
  append_json_string(output, std::to_string(conversion.admitted_max));
  output += ",\"provenance\":";
  append_json_string(output, conversion.provenance);
  output.push_back('}');
}

[[nodiscard]] std::string catalog_semantic_json(
    std::string_view scope, std::span<const CatalogClockRecord> clocks,
    std::span<const CatalogClockConversionRecord> conversions,
    std::span<const ChannelEndpointRecord> endpoints) {
  std::string output{"{\"format\":\"holoscan.channel-endpoints\",\"version\":2,\"scope\":"};
  append_json_string(output, scope);
  output += ",\"clocks\":[";
  for (std::size_t index = 0U; index < clocks.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    append_clock(output, clocks[index]);
  }
  output += "],\"clock_conversions\":[";
  for (std::size_t index = 0U; index < conversions.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    append_conversion(output, conversions[index]);
  }
  output += "],\"services\":[";
  for (std::size_t index = 0U; index < endpoints.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    append_service(output, endpoints[index]);
  }
  output += "],\"endpoints\":[";
  for (std::size_t index = 0U; index < endpoints.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    append_endpoint(output, endpoints[index]);
  }
  output += "]}";
  return output;
}

[[nodiscard]] std::string catalog_json(std::string_view scope, std::string_view digest,
                                       std::span<const CatalogClockRecord> clocks,
                                       std::span<const CatalogClockConversionRecord> conversions,
                                       std::span<const ChannelEndpointRecord> endpoints) {
  std::string output{"{\"format\":\"holoscan.channel-endpoints\",\"version\":2,\"scope\":"};
  append_json_string(output, scope);
  output += ",\"digest\":";
  append_json_string(output, digest);
  output += ",\"clocks\":[";
  for (std::size_t index = 0U; index < clocks.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    append_clock(output, clocks[index]);
  }
  output += "],\"clock_conversions\":[";
  for (std::size_t index = 0U; index < conversions.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    append_conversion(output, conversions[index]);
  }
  output += "],\"services\":[";
  for (std::size_t index = 0U; index < endpoints.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    append_service(output, endpoints[index]);
  }
  output += "],\"endpoints\":[";
  for (std::size_t index = 0U; index < endpoints.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    append_endpoint(output, endpoints[index]);
  }
  output += "]}";
  return output;
}

[[nodiscard]] std::array<std::byte, 32> hash_bytes(std::string_view bytes) noexcept {
  detail::StableHashBuilder hash;
  for (const char byte : bytes) {
    hash.add_byte(static_cast<std::uint8_t>(byte));
  }
  return hash.finish().value;
}

[[nodiscard]] std::string_view pod_scalar_token(PodScalarKind kind) noexcept {
  switch (kind) {
    case PodScalarKind::kUInt8:
      return "u8";
    case PodScalarKind::kInt8:
      return "i8";
    case PodScalarKind::kUInt16:
      return "u16";
    case PodScalarKind::kInt16:
      return "i16";
    case PodScalarKind::kUInt32:
      return "u32";
    case PodScalarKind::kInt32:
      return "i32";
    case PodScalarKind::kUInt64:
      return "u64";
    case PodScalarKind::kInt64:
      return "i64";
    case PodScalarKind::kFloat32:
      return "f32";
    case PodScalarKind::kFloat64:
      return "f64";
  }
  return {};
}

[[nodiscard]] bool complete_schema_identity_valid(const SchemaIdentityRecord& identity) noexcept {
  return valid_schema_identity(identity);
}

[[nodiscard]] bool complete_pod_identity_valid(const PodIdentityRecord& identity) {
  return channel_internal::derive_pod_contract_evidence(identity).has_value();
}

[[nodiscard]] bool valid_catalog_tensor_contract(
    const TensorBoundaryContractRecord& tensor) noexcept {
  const bool bounds_valid =
      tensor.memory_kind && detail::valid_memory_kind_value(*tensor.memory_kind) &&
      (!tensor.dtype || detail::valid_tensor_dtype(*tensor.dtype)) && tensor.max_rank &&
      *tensor.max_rank > 0U && tensor.max_byte_span && *tensor.max_byte_span > 0U &&
      (!tensor.rank || *tensor.rank <= *tensor.max_rank) && tensor.axes.size() <= *tensor.max_rank;
  const bool stride_valid =
      tensor.strides == TensorStridePolicy::kContiguousOnly ||
      (tensor.strides == TensorStridePolicy::kBoundedStrided && tensor.max_rank &&
       tensor.axes.size() == *tensor.max_rank &&
       std::all_of(tensor.axes.begin(), tensor.axes.end(), [](const auto& axis) {
         return axis.max_abs_stride_bytes > 0U;
       }));
  const bool inline_policy =
      tensor.foreign_emit_policy == InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded;
  const bool emit_policy_valid =
      tensor.foreign_emit_policy == InterfaceTensorEmitPolicy::kRequireExportable ||
      tensor.foreign_emit_policy == InterfaceTensorEmitPolicy::kCopyIfNeeded || inline_policy;
  const bool inline_bound_valid = inline_policy
                                      ? tensor.max_inline_bytes && *tensor.max_inline_bytes > 0U
                                      : !tensor.max_inline_bytes;
  const bool copy_capable =
      tensor.foreign_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable;
  const bool staging_bound_valid =
      copy_capable ? tensor.max_staging_byte_span && *tensor.max_staging_byte_span > 0U
                   : !tensor.max_staging_byte_span;
  const bool conversion_valid =
      !tensor.memory_conversion ||
      closed_enum(*tensor.memory_conversion, MemoryConversionPolicy::kRequire);
  return bounds_valid && stride_valid && emit_policy_valid && inline_bound_valid &&
         staging_bound_valid && conversion_valid;
}

void finalize_payload_identity(ChannelEndpointRecord& endpoint) {
  if (endpoint.payload_identity.kind != PayloadIdentityKind::kPod ||
      !endpoint.payload_identity.pod) {
    endpoint.pod_identity.reset();
    return;
  }
  detail::PodCatalogIdentityRecord pod{.identity = *endpoint.payload_identity.pod};
  auto evidence = channel_internal::derive_pod_contract_evidence(pod.identity);
  if (!evidence) {
    endpoint.pod_identity.reset();
    return;
  }
  pod.digest_bytes = evidence->identity_digest;
  pod.key = evidence->key;
  pod.layout_digest_bytes = evidence->layout_digest;
  endpoint.pod_identity = std::move(pod);
  endpoint.pod_identity->rebuild_field_views();
}

void materialize_normalized_ownership(ChannelEndpointRecord& endpoint) {
  const auto inline_contract = channel_internal::catalog_inline_payload_contract(
      endpoint.payload_identity, endpoint.schema_writer_max_serialized_bytes);
  const bool descriptor_schema =
      endpoint.payload_identity.kind == PayloadIdentityKind::kSchema &&
      endpoint.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
  const auto descriptor_bucket =
      descriptor_schema &&
              endpoint.schema_writer_max_serialized_bytes <= std::numeric_limits<std::size_t>::max()
          ? channel_internal::inline_payload_bucket_size(
                static_cast<std::size_t>(endpoint.schema_writer_max_serialized_bytes))
          : std::nullopt;
  endpoint.service_static = detail::ResolvedServiceStaticRequirements{
      .id = endpoint.service,
      .payload_identity = endpoint.payload_identity,
      .schema_writer_max_serialized_bytes = endpoint.schema_writer_max_serialized_bytes,
      .required_data_frame_bytes =
          inline_contract ? inline_contract->required_frame_bytes
                          : (descriptor_schema ? endpoint.schema_writer_max_serialized_bytes : 0U),
      .selected_inline_bucket_bytes =
          inline_contract ? inline_contract->selected_bucket_bytes : descriptor_bucket.value_or(0U),
      .bucket_usable_serialized_bytes = inline_contract
                                            ? inline_contract->bucket_usable_native_payload_bytes
                                            : descriptor_bucket.value_or(0U),
      .pod_identity = endpoint.pod_identity,
      .schema_identity = endpoint.schema_identity,
      .tensor = endpoint.tensor,
      .tensor_fields = endpoint.tensor_fields,
      .queue_policy = endpoint.queue_policy,
      .max_subscriber_depth = endpoint.service_max_subscriber_depth,
      .max_subscriber_borrowed = endpoint.service_max_subscriber_borrowed,
      .publisher_history_depth = 0U,
      .event_wake = true,
  };
  endpoint.publisher_request =
      endpoint.direction == TopicDirection::kImport
          ? std::optional{detail::ResolvedPublisherEndpointRequest{
                .max_loaned_samples = endpoint.bounds.publisher
                                          ? endpoint.bounds.publisher->max_loaned_samples
                                          : std::nullopt,
                .capture_time_clock = endpoint.capture_time_clock,
                .clock_health = endpoint.clock_health,
            }}
          : std::nullopt;
  endpoint.subscriber_request =
      endpoint.direction == TopicDirection::kExport && endpoint.bounds.subscriber
          ? std::optional{detail::ResolvedSubscriberEndpointRequest{
                .queue_depth = endpoint.bounds.subscriber->queue_depth.value_or(1U),
                .max_borrowed_samples =
                    endpoint.bounds.subscriber->max_borrowed_samples.value_or(1U),
            }}
          : std::nullopt;
}

[[nodiscard]] bool canonical_sha256(std::string_view value) noexcept {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::all_of(value.begin() + 7, value.end(), [](char character) {
           return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
         });
}

// Catalog normalization is one transactional cross-record validation pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<ChannelEndpointCatalog, Error> normalize_catalog(
    std::string scope, std::vector<ChannelEndpointRecord> endpoints,
    std::vector<CatalogClockRecord> clocks = {},
    std::vector<CatalogClockConversionRecord> clock_conversions = {},
    std::optional<std::string_view> expected_digest = std::nullopt,
    std::optional<std::string_view> raw_source = std::nullopt,
    bool infer_standalone_clocks = false) {
  if (endpoints.size() > detail::kMaximumCatalogEndpoints) {
    return make_unexpected(catalog_resource_limit_error(
        "/endpoints", "normalized_endpoints", detail::kMaximumCatalogEndpoints, endpoints.size()));
  }
  if (clocks.size() > kMaximumCatalogClocks) {
    return make_unexpected(catalog_resource_limit_error(
        "/clocks", "normalized_clocks", kMaximumCatalogClocks, clocks.size()));
  }
  if (clock_conversions.size() > kMaximumCatalogClockConversions) {
    return make_unexpected(catalog_resource_limit_error("/clock_conversions",
                                                        "normalized_clock_conversions",
                                                        kMaximumCatalogClockConversions,
                                                        clock_conversions.size()));
  }
  if (scope.empty() || scope.size() > 255U || !valid_utf8(scope) || has_edge_whitespace(scope)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_SCOPE_INVALID"));
  }
  std::sort(endpoints.begin(), endpoints.end(), [](const auto& left, const auto& right) {
    return left.name < right.name;
  });
  for (const auto& endpoint : endpoints) {
    if (!endpoint.capture_time_clock || endpoint.capture_time_clock->is_default()) {
      continue;
    }
    const auto found = std::find_if(clocks.begin(), clocks.end(), [&](const auto& clock) {
      return clock.key == *endpoint.capture_time_clock;
    });
    if (found == clocks.end() && infer_standalone_clocks) {
      if (clocks.size() >= kMaximumCatalogClocks) {
        return make_unexpected(catalog_resource_limit_error(
            "/clocks", "normalized_clocks", kMaximumCatalogClocks, clocks.size() + 1U));
      }
      clocks.push_back(CatalogClockRecord{
          .key = *endpoint.capture_time_clock,
          .kind = "external",
          .unit = "nanoseconds",
          .epoch = "external",
          .monotonicity = "nondecreasing",
          .discipline_kind = "external",
          .discipline_identity = std::string(endpoint.capture_time_clock->name()),
      });
    }
  }
  std::sort(clocks.begin(), clocks.end(), [](const auto& left, const auto& right) {
    return left.key.name() < right.key.name();
  });
  for (std::size_t index = 0U; index < clocks.size(); ++index) {
    auto& clock = clocks[index];
    if (clock.key.is_default() || clock.key.name().empty() ||
        (index != 0U && clocks[index - 1U].key == clock.key)) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_CLOCK_INVALID"));
    }
    clock.id = static_cast<std::uint64_t>(index + 1U);
    if (clock.kind.empty() || clock.unit.empty() || clock.epoch.empty() ||
        clock.monotonicity.empty() || clock.discipline_kind.empty() ||
        clock.discipline_identity.empty()) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_CLOCK_INVALID"));
    }
  }
  std::sort(
      clock_conversions.begin(), clock_conversions.end(), [](const auto& left, const auto& right) {
        return std::tuple{left.from.name(), left.to.name()} <
               std::tuple{right.from.name(), right.to.name()};
      });
  for (std::size_t index = 0U; index < clock_conversions.size(); ++index) {
    const auto& conversion = clock_conversions[index];
    const auto known_clock = [&](const ClockKey& key) {
      return std::any_of(
          clocks.begin(), clocks.end(), [&](const auto& clock) { return clock.key == key; });
    };
    if (conversion.from == conversion.to || !known_clock(conversion.from) ||
        !known_clock(conversion.to) || conversion.scale_numerator <= 0 ||
        conversion.scale_denominator == 0U || conversion.admitted_min > conversion.admitted_max ||
        conversion.provenance.empty() ||
        (index != 0U && clock_conversions[index - 1U].from == conversion.from &&
         clock_conversions[index - 1U].to == conversion.to)) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_CLOCK_CONVERSION_INVALID"));
    }
  }
  using PodAuthorityKey = std::tuple<std::string_view, std::uint32_t>;
  std::map<PodAuthorityKey, const detail::PodCatalogIdentityRecord*> pods_by_authority;
  std::map<std::array<std::uint8_t, 16>, const detail::PodCatalogIdentityRecord*> pods_by_key;
  std::map<std::array<std::byte, 16>, const SchemaIdentityRecord*> schemas_by_family_id;
  for (std::size_t index = 0U; index < endpoints.size(); ++index) {
    auto& endpoint = endpoints[index];
    if (endpoint.payload_identity.pod &&
        endpoint.payload_identity.pod->fields.size() > kMaximumPodFields) {
      return make_unexpected(catalog_resource_limit_error(
          "/endpoints/" + std::to_string(index) + "/payload/pod_identity/fields",
          "normalized_pod_fields",
          kMaximumPodFields,
          endpoint.payload_identity.pod->fields.size()));
    }
    if (endpoint.tensor && endpoint.tensor->axes.size() > kMaximumTensorAxes) {
      return make_unexpected(catalog_resource_limit_error(
          "/endpoints/" + std::to_string(index) + "/payload/tensor/axes",
          "normalized_tensor_axes",
          kMaximumTensorAxes,
          endpoint.tensor->axes.size()));
    }
    if (endpoint.tensor_fields.size() > kMaximumTensorSchemaLoanFields) {
      return make_unexpected(catalog_resource_limit_error(
          "/endpoints/" + std::to_string(index) + "/payload/descriptor_tensor_fields",
          "normalized_descriptor_tensor_fields",
          kMaximumTensorSchemaLoanFields,
          endpoint.tensor_fields.size()));
    }
    for (std::size_t field_index = 0U; field_index < endpoint.tensor_fields.size(); ++field_index) {
      if (endpoint.tensor_fields[field_index].tensor.axes.size() <= kMaximumTensorAxes) {
        continue;
      }
      return make_unexpected(catalog_resource_limit_error(
          "/endpoints/" + std::to_string(index) + "/payload/descriptor_tensor_fields/" +
              std::to_string(field_index) + "/contract/bounds/axes",
          "normalized_tensor_axes",
          kMaximumTensorAxes,
          endpoint.tensor_fields[field_index].tensor.axes.size()));
    }
    if (!endpoint_name_valid(endpoint.name)) {
      return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_NAME_INVALID"));
    }
    if (!closed_enum(endpoint.direction, TopicDirection::kExport) ||
        !closed_enum(endpoint.name_origin, EndpointNameOrigin::kGenerated) ||
        !closed_enum(endpoint.attach.peer, PeerRequirement::kRequired) ||
        (endpoint.memory_conversion &&
         !closed_enum(*endpoint.memory_conversion, MemoryConversionPolicy::kRequire))) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CONTRACT_ENUM_INVALID"));
    }
    if (!endpoint.uri.starts_with("holomq://") || endpoint.uri.size() == 9U ||
        endpoint.uri.size() > 4096U || !valid_utf8(endpoint.uri) ||
        has_edge_whitespace(endpoint.uri) ||
        std::any_of(endpoint.uri.begin(), endpoint.uri.end(), [](char character) {
          const auto byte = static_cast<std::uint8_t>(character);
          return byte == 0U || byte < 0x20U || byte == 0x7FU;
        })) {
      return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_URI_INVALID"));
    }
    if (endpoint.service.empty()) {
      endpoint.service = endpoint.name + ".service";
    }
    if (endpoint.service.size() > 255U || !valid_utf8(endpoint.service) ||
        has_edge_whitespace(endpoint.service) ||
        channel_internal::is_ordered_terminal_service_id(endpoint.service)) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_SERVICE_INVALID"));
    }
    endpoint.memory_conversion =
        endpoint.memory_conversion.value_or(MemoryConversionPolicy::kDefault);
    endpoint.attach.max_wait = endpoint.attach.max_wait.value_or(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds{2}));
    constexpr std::uint32_t kMaximumAcceptedCount = std::numeric_limits<std::uint32_t>::max() - 1U;
    if (endpoint.direction == TopicDirection::kImport && endpoint.bounds.subscriber) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_DIRECTION_BOUNDS_INVALID"));
    }
    if (endpoint.direction == TopicDirection::kExport && endpoint.bounds.publisher) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_DIRECTION_BOUNDS_INVALID"));
    }
    if (endpoint.direction == TopicDirection::kImport && !endpoint.bounds.publisher) {
      endpoint.bounds.publisher = PublisherEndpointBounds{};
    }
    if (endpoint.direction == TopicDirection::kExport && !endpoint.bounds.subscriber) {
      endpoint.bounds.subscriber = SubscriberEndpointBounds{};
    }
    if (!endpoint.attach.max_wait || endpoint.attach.max_wait->count() <= 0) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_ATTACH_BOUND_INVALID"));
    }
    if (endpoint.bounds.publisher) {
      endpoint.bounds.publisher->max_loaned_samples =
          endpoint.bounds.publisher->max_loaned_samples.value_or(1U);
      if (*endpoint.bounds.publisher->max_loaned_samples == 0U ||
          *endpoint.bounds.publisher->max_loaned_samples > kMaximumAcceptedCount) {
        return make_unexpected(
            catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_BOUNDS_INVALID"));
      }
    }
    if (endpoint.bounds.subscriber) {
      endpoint.bounds.subscriber->queue_depth =
          endpoint.bounds.subscriber->queue_depth.value_or(1U);
      endpoint.bounds.subscriber->max_borrowed_samples =
          endpoint.bounds.subscriber->max_borrowed_samples.value_or(1U);
      if (*endpoint.bounds.subscriber->queue_depth == 0U ||
          *endpoint.bounds.subscriber->queue_depth > kMaximumAcceptedCount ||
          *endpoint.bounds.subscriber->max_borrowed_samples == 0U ||
          *endpoint.bounds.subscriber->max_borrowed_samples > kMaximumAcceptedCount) {
        return make_unexpected(
            catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_BOUNDS_INVALID"));
      }
    }
    if (endpoint.service_max_subscriber_depth == 0U) {
      endpoint.service_max_subscriber_depth =
          endpoint.bounds.subscriber ? *endpoint.bounds.subscriber->queue_depth : 1U;
    }
    if (endpoint.service_max_subscriber_borrowed == 0U) {
      endpoint.service_max_subscriber_borrowed =
          endpoint.bounds.subscriber ? *endpoint.bounds.subscriber->max_borrowed_samples : 1U;
    }
    if (endpoint.service_max_subscriber_depth > kMaximumAcceptedCount ||
        endpoint.service_max_subscriber_borrowed > kMaximumAcceptedCount ||
        (endpoint.bounds.subscriber &&
         (*endpoint.bounds.subscriber->queue_depth > endpoint.service_max_subscriber_depth ||
          *endpoint.bounds.subscriber->max_borrowed_samples >
              endpoint.service_max_subscriber_borrowed))) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_SERVICE_INVALID"));
    }
    if (!endpoint.schema_identity && endpoint.payload_identity.schema) {
      endpoint.schema_identity = endpoint.payload_identity.schema;
    }
    if (endpoint.schema_identity && !complete_schema_identity_valid(*endpoint.schema_identity)) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
    if (endpoint.payload_identity.kind == PayloadIdentityKind::kScalar) {
      if (endpoint.payload_identity.transport_profile_tag != 0U || endpoint.tensor ||
          !endpoint.payload_identity.scalar || endpoint.payload_identity.schema ||
          endpoint.payload_identity.pod || !endpoint.tensor_fields.empty() ||
          endpoint.payload_identity.scalar->kind == ScalarIdentityKind::kLocalEnum) {
        return make_unexpected(
            catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
      }
    } else if (endpoint.payload_identity.kind == PayloadIdentityKind::kTensor) {
      if (endpoint.payload_identity.transport_profile_tag != 0U ||
          endpoint.payload_identity.scalar || endpoint.payload_identity.schema ||
          endpoint.payload_identity.pod || !endpoint.tensor_fields.empty()) {
        return make_unexpected(
            catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
      }
      if (endpoint.tensor && !valid_catalog_tensor_contract(*endpoint.tensor)) {
        return make_unexpected(
            catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_TENSOR_POLICY_INVALID"));
      }
    } else if (endpoint.payload_identity.kind == PayloadIdentityKind::kSchema) {
      if (endpoint.tensor || endpoint.payload_identity.scalar ||
          !endpoint.payload_identity.schema || endpoint.payload_identity.pod ||
          endpoint.schema_writer_max_serialized_bytes == 0U ||
          endpoint.schema_writer_max_serialized_bytes > std::numeric_limits<std::uint32_t>::max() ||
          !complete_schema_identity_valid(*endpoint.payload_identity.schema) ||
          (endpoint.schema_identity &&
           (endpoint.schema_identity->canonical_name !=
                endpoint.payload_identity.schema->canonical_name ||
            endpoint.schema_identity->compatibility_epoch !=
                endpoint.payload_identity.schema->compatibility_epoch ||
            endpoint.schema_identity->family_id != endpoint.payload_identity.schema->family_id ||
            endpoint.schema_identity->file_identifier !=
                endpoint.payload_identity.schema->file_identifier))) {
        return make_unexpected(
            catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
      }
      const bool descriptor_schema =
          endpoint.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
      if (descriptor_schema != !endpoint.tensor_fields.empty() ||
          descriptor_schema != (endpoint.payload_identity.transport_profile_tag != 0U)) {
        return make_unexpected(
            catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_TENSOR_POLICY_INVALID"));
      }
      if (descriptor_schema &&
          (endpoint.schema_writer_max_serialized_bytes > std::numeric_limits<std::size_t>::max() ||
           !channel_internal::inline_payload_bucket_size(
               static_cast<std::size_t>(endpoint.schema_writer_max_serialized_bytes)))) {
        return make_unexpected(
            catalog_error(ErrorCode::kResourceExhausted, "ENDPOINT_PAYLOAD_CAPACITY_EXCEEDED"));
      }
      for (std::size_t field_index = 0U; field_index < endpoint.tensor_fields.size();
           ++field_index) {
        const auto& field = endpoint.tensor_fields[field_index];
        if (field.tensor_index != field_index || !valid_catalog_tensor_contract(field.tensor) ||
            (field_index != 0U &&
             endpoint.tensor_fields[field_index - 1U].field_id >= field.field_id)) {
          return make_unexpected(
              catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_TENSOR_POLICY_INVALID"));
        }
      }
    } else if (endpoint.payload_identity.kind == PayloadIdentityKind::kPod) {
      if (endpoint.payload_identity.transport_profile_tag != 0U || endpoint.tensor ||
          endpoint.payload_identity.scalar || endpoint.payload_identity.schema ||
          !endpoint.payload_identity.pod || !endpoint.tensor_fields.empty() ||
          !complete_pod_identity_valid(*endpoint.payload_identity.pod)) {
        return make_unexpected(
            catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
      }
    }
    if (endpoint.payload_identity.kind != PayloadIdentityKind::kSchema &&
        endpoint.schema_writer_max_serialized_bytes != 0U) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
    const auto inline_contract = channel_internal::catalog_inline_payload_contract(
        endpoint.payload_identity, endpoint.schema_writer_max_serialized_bytes);
    const bool ordinary_inline_payload =
        endpoint.payload_identity.kind == PayloadIdentityKind::kScalar ||
        endpoint.payload_identity.kind == PayloadIdentityKind::kPod ||
        (endpoint.payload_identity.kind == PayloadIdentityKind::kSchema &&
         endpoint.payload_identity.payload_kind == PayloadKind::kSerializedBytes);
    if (ordinary_inline_payload && !inline_contract) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
    if (inline_contract && inline_contract->selected_bucket_bytes == 0U) {
      return make_unexpected(catalog_inline_capacity_error(index, endpoint.name, *inline_contract));
    }
    finalize_payload_identity(endpoint);
    if (endpoint.pod_identity) {
      const auto& current = *endpoint.pod_identity;
      const auto incompatible = [&](const detail::PodCatalogIdentityRecord* prior) {
        return prior != nullptr && (current.digest_bytes != prior->digest_bytes ||
                                    current.layout_digest_bytes != prior->layout_digest_bytes);
      };
      const PodAuthorityKey authority{current.identity.canonical_name, current.identity.version};
      const auto authority_match = pods_by_authority.find(authority);
      const auto key_match = pods_by_key.find(current.key.bytes);
      if ((authority_match != pods_by_authority.end() && incompatible(authority_match->second)) ||
          (key_match != pods_by_key.end() && incompatible(key_match->second))) {
        return make_unexpected(
            catalog_error(ErrorCode::kIncompatible, "ENDPOINT_POD_IDENTITY_COLLISION"));
      }
      pods_by_authority.try_emplace(authority, &current);
      pods_by_key.try_emplace(current.key.bytes, &current);
    }
    if (endpoint.schema_identity) {
      const auto& current = *endpoint.schema_identity;
      const auto found = schemas_by_family_id.find(current.family_id.value);
      if (found != schemas_by_family_id.end() &&
          (current.canonical_name != found->second->canonical_name ||
           current.compatibility_epoch != found->second->compatibility_epoch ||
           current.file_identifier != found->second->file_identifier)) {
        return make_unexpected(
            catalog_error(ErrorCode::kIncompatible, "ENDPOINT_SCHEMA_IDENTITY_COLLISION"));
      }
      schemas_by_family_id.try_emplace(current.family_id.value, &current);
    }
    const bool ordinary_health =
        endpoint.clock_health.capability == ClockHealthCapability::kUnavailable &&
        endpoint.clock_health.requirement == ClockHealthRequirement::kNotRequired &&
        !endpoint.clock_health.max_report_age;
    const bool monitored_health =
        endpoint.direction == TopicDirection::kImport &&
        endpoint.clock_health.capability == ClockHealthCapability::kStatusReports &&
        endpoint.clock_health.requirement == ClockHealthRequirement::kFreshHealthyReport &&
        endpoint.clock_health.max_report_age && endpoint.clock_health.max_report_age->count() > 0;
    if (!ordinary_health && !monitored_health) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID"));
    }
    if (!supported_queue_policy(endpoint.queue_policy)) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_SERVICE_CONTRACT_INVALID"));
    }
    if (endpoint.capture_time_clock &&
        (endpoint.capture_time_clock->is_default() ||
         std::none_of(clocks.begin(), clocks.end(), [&](const auto& clock) {
           return clock.key == *endpoint.capture_time_clock;
         }))) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CAPTURE_CLOCK_INVALID"));
    }
    if (index != 0U && endpoints[index - 1U].name == endpoint.name) {
      return make_unexpected(catalog_error(ErrorCode::kDuplicateName, "ENDPOINT_NAME_DUPLICATE"));
    }
    const bool generated = endpoint.name_origin == EndpointNameOrigin::kGenerated;
    if (generated != endpoint.generated_name.has_value() ||
        (generated && (endpoint.generated_name->algorithm != kGeneratedAlgorithm ||
                       endpoint.generated_name->version != kGeneratedAlgorithmVersion ||
                       !endpoint.name.starts_with("__holoscan_auto_endpoint/v1/"))) ||
        (!generated && endpoint.name.starts_with("__holoscan_auto_endpoint/"))) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_NAME_GENERATED_METADATA_INVALID"));
    }
  }

  auto state = std::make_shared<detail::ChannelEndpointCatalogState>();
  state->scope = std::move(scope);
  state->clocks = std::move(clocks);
  state->clock_conversions = std::move(clock_conversions);
  state->endpoints = std::move(endpoints);
  for (auto& endpoint : state->endpoints) {
    materialize_normalized_ownership(endpoint);
  }
  const std::string semantic = catalog_semantic_json(
      state->scope, state->clocks, state->clock_conversions, state->endpoints);
  state->digest_bytes = hash_bytes(semantic);
  state->digest = detail::lowercase_hex_text(state->digest_bytes, detail::HexPrefix::kSha256);
  if (expected_digest &&
      (!canonical_sha256(*expected_digest) || *expected_digest != state->digest)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_DIGEST_MISMATCH"));
  }
  state->json = catalog_json(
      state->scope, state->digest, state->clocks, state->clock_conversions, state->endpoints);
  if (raw_source) {
    state->raw_source_sha256 =
        detail::lowercase_hex_text(hash_bytes(*raw_source), detail::HexPrefix::kSha256);
    state->raw_source_size_bytes = static_cast<std::uint64_t>(raw_source->size());
  }
  return detail::ChannelEndpointCatalogAccess::make(std::move(state));
}

[[nodiscard]] expected<std::optional<GeneratedEndpointNameRecord>, Error> parse_generated_name(
    const JsonValue& value) {
  if (value.kind == JsonValue::Kind::kNull) {
    return std::nullopt;
  }
  constexpr std::array<std::string_view, 2> kFields{"algorithm", "version"};
  if (!allowed_fields(value, kFields)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_NAME_GENERATED_METADATA_INVALID"));
  }
  const JsonValue* algorithm = json_field(value, "algorithm");
  const JsonValue* version = json_field(value, "version");
  std::uint64_t parsed_version{};
  if (algorithm == nullptr || algorithm->kind != JsonValue::Kind::kString || version == nullptr ||
      !json_unsigned(*version, parsed_version) ||
      parsed_version > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_NAME_GENERATED_METADATA_INVALID"));
  }
  return std::optional<GeneratedEndpointNameRecord>{GeneratedEndpointNameRecord{
      .algorithm = algorithm->string,
      .version = static_cast<std::uint32_t>(parsed_version),
  }};
}

[[nodiscard]] std::optional<ScalarIdentityKind> parse_scalar_identity(
    std::string_view token) noexcept {
  constexpr std::array<std::pair<std::string_view, ScalarIdentityKind>, 11> kTokens{{
      {"bool", ScalarIdentityKind::kBoolean},
      {"u8", ScalarIdentityKind::kUInt8},
      {"i8", ScalarIdentityKind::kInt8},
      {"u16", ScalarIdentityKind::kUInt16},
      {"i16", ScalarIdentityKind::kInt16},
      {"u32", ScalarIdentityKind::kUInt32},
      {"i32", ScalarIdentityKind::kInt32},
      {"u64", ScalarIdentityKind::kUInt64},
      {"i64", ScalarIdentityKind::kInt64},
      {"f32", ScalarIdentityKind::kFloat32},
      {"f64", ScalarIdentityKind::kFloat64},
  }};
  const auto found = std::find_if(
      kTokens.begin(), kTokens.end(), [&](const auto& entry) { return entry.first == token; });
  return found == kTokens.end() ? std::nullopt : std::optional{found->second};
}

[[nodiscard]] std::optional<MemoryKind> parse_memory_kind(std::string_view token) noexcept {
  constexpr std::array<std::pair<std::string_view, MemoryKind>, 5> kTokens{{
      {"unknown", MemoryKind::kUnknown},
      {"host", MemoryKind::kHost},
      {"pinned_host", MemoryKind::kPinnedHost},
      {"cuda_device", MemoryKind::kCudaDevice},
      {"cuda_managed", MemoryKind::kCudaManaged},
  }};
  const auto found = std::find_if(
      kTokens.begin(), kTokens.end(), [&](const auto& entry) { return entry.first == token; });
  return found == kTokens.end() ? std::nullopt : std::optional{found->second};
}

[[nodiscard]] std::optional<PodScalarKind> parse_pod_scalar(std::string_view token) noexcept {
  constexpr std::array<std::pair<std::string_view, PodScalarKind>, 10> kTokens{{
      {"u8", PodScalarKind::kUInt8},
      {"i8", PodScalarKind::kInt8},
      {"u16", PodScalarKind::kUInt16},
      {"i16", PodScalarKind::kInt16},
      {"u32", PodScalarKind::kUInt32},
      {"i32", PodScalarKind::kInt32},
      {"u64", PodScalarKind::kUInt64},
      {"i64", PodScalarKind::kInt64},
      {"f32", PodScalarKind::kFloat32},
      {"f64", PodScalarKind::kFloat64},
  }};
  const auto found = std::find_if(
      kTokens.begin(), kTokens.end(), [&](const auto& entry) { return entry.first == token; });
  return found == kTokens.end() ? std::nullopt : std::optional{found->second};
}

[[nodiscard]] expected<SchemaIdentityRecord, Error> parse_schema_identity(const JsonValue& value) {
  constexpr std::array<std::string_view, 4> kFields{
      "canonical_name", "compatibility_epoch", "family_id", "file_identifier"};
  if (!allowed_fields(value, kFields)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  const JsonValue* name = json_field(value, "canonical_name");
  const JsonValue* compatibility_epoch = json_field(value, "compatibility_epoch");
  const JsonValue* family_id = json_field(value, "family_id");
  const JsonValue* file_identifier = json_field(value, "file_identifier");
  std::uint64_t parsed_compatibility_epoch{};
  SchemaIdentityRecord result;
  if (name == nullptr || name->kind != JsonValue::Kind::kString || compatibility_epoch == nullptr ||
      !json_unsigned(*compatibility_epoch, parsed_compatibility_epoch) ||
      parsed_compatibility_epoch > std::numeric_limits<std::uint32_t>::max() ||
      family_id == nullptr || family_id->kind != JsonValue::Kind::kString ||
      file_identifier == nullptr || file_identifier->kind != JsonValue::Kind::kString ||
      file_identifier->string.size() != result.file_identifier.value.size() ||
      !parse_hex_bytes(family_id->string, result.family_id.value, false)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  result.canonical_name = name->string;
  result.compatibility_epoch = static_cast<std::uint32_t>(parsed_compatibility_epoch);
  std::copy(file_identifier->string.begin(),
            file_identifier->string.end(),
            result.file_identifier.value.begin());
  return result;
}

struct ParsedPodIdentity {
  PodIdentityRecord identity;
  std::array<std::uint8_t, 32> expected_digest{};
  TypeKey expected_key{};
  std::array<std::uint8_t, 32> expected_layout_digest{};
};

[[nodiscard]] expected<ParsedPodIdentity, Error> parse_pod_identity(const JsonValue& value) {
  constexpr std::array<std::string_view, 12> kFields{"name",
                                                     "version",
                                                     "digest",
                                                     "key",
                                                     "object_size",
                                                     "object_alignment",
                                                     "layout_format_version",
                                                     "layout_digest",
                                                     "integer_wire_endianness",
                                                     "floating_wire_format",
                                                     "fields",
                                                     "compatibility_evidence"};
  if (!allowed_fields(value, kFields)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  const JsonValue* name = json_field(value, "name");
  const JsonValue* version = json_field(value, "version");
  const JsonValue* digest = json_field(value, "digest");
  const JsonValue* key = json_field(value, "key");
  const JsonValue* object_size = json_field(value, "object_size");
  const JsonValue* object_alignment = json_field(value, "object_alignment");
  const JsonValue* layout_version = json_field(value, "layout_format_version");
  const JsonValue* layout_digest = json_field(value, "layout_digest");
  const JsonValue* integer_endianness = json_field(value, "integer_wire_endianness");
  const JsonValue* floating_format = json_field(value, "floating_wire_format");
  const JsonValue* fields = json_field(value, "fields");
  const JsonValue* evidence = json_field(value, "compatibility_evidence");
  std::uint64_t parsed_version{};
  std::uint64_t parsed_size{};
  std::uint64_t parsed_alignment{};
  std::uint64_t parsed_layout_version{};
  ParsedPodIdentity result;
  if (name == nullptr || name->kind != JsonValue::Kind::kString || version == nullptr ||
      !json_unsigned(*version, parsed_version) ||
      parsed_version > std::numeric_limits<std::uint32_t>::max() || digest == nullptr ||
      digest->kind != JsonValue::Kind::kString || key == nullptr ||
      key->kind != JsonValue::Kind::kString || object_size == nullptr ||
      !json_unsigned(*object_size, parsed_size) ||
      parsed_size > std::numeric_limits<std::uint32_t>::max() || object_alignment == nullptr ||
      !json_unsigned(*object_alignment, parsed_alignment) ||
      parsed_alignment > std::numeric_limits<std::uint32_t>::max() || layout_version == nullptr ||
      !json_unsigned(*layout_version, parsed_layout_version) || parsed_layout_version != 1U ||
      layout_digest == nullptr || layout_digest->kind != JsonValue::Kind::kString ||
      integer_endianness == nullptr || integer_endianness->kind != JsonValue::Kind::kString ||
      integer_endianness->string != "little" || floating_format == nullptr ||
      floating_format->kind != JsonValue::Kind::kString || floating_format->string != "ieee754" ||
      fields == nullptr || fields->kind != JsonValue::Kind::kArray || evidence == nullptr ||
      evidence->kind != JsonValue::Kind::kArray || !evidence->array.empty() ||
      !parse_hex_bytes(digest->string, result.expected_digest, true) ||
      !parse_hex_bytes(key->string, result.expected_key.bytes, false) ||
      !parse_hex_bytes(layout_digest->string, result.expected_layout_digest, true)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  if (fields->array.size() > kMaximumPodFields) {
    return make_unexpected(catalog_resource_limit_error("/services/*/payload/pod_identity/fields",
                                                        "normalized_pod_fields",
                                                        kMaximumPodFields,
                                                        fields->array.size()));
  }
  result.identity.canonical_name = name->string;
  result.identity.version = static_cast<std::uint32_t>(parsed_version);
  result.identity.object_size = static_cast<std::uint32_t>(parsed_size);
  result.identity.object_alignment = static_cast<std::uint32_t>(parsed_alignment);
  result.identity.fields.reserve(fields->array.size());
  for (const JsonValue& field : fields->array) {
    constexpr std::array<std::string_view, 4> kFieldFields{
        "name", "byte_offset", "element_count", "scalar"};
    const JsonValue* field_name = json_field(field, "name");
    const JsonValue* byte_offset = json_field(field, "byte_offset");
    const JsonValue* element_count = json_field(field, "element_count");
    const JsonValue* scalar = json_field(field, "scalar");
    std::uint64_t parsed_offset{};
    std::uint64_t parsed_count{};
    if (!allowed_fields(field, kFieldFields) || field_name == nullptr ||
        field_name->kind != JsonValue::Kind::kString || byte_offset == nullptr ||
        !json_unsigned(*byte_offset, parsed_offset) ||
        parsed_offset > std::numeric_limits<std::uint32_t>::max() || element_count == nullptr ||
        !json_unsigned(*element_count, parsed_count) ||
        parsed_count > std::numeric_limits<std::uint32_t>::max() || scalar == nullptr ||
        scalar->kind != JsonValue::Kind::kString) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
    const auto parsed_scalar = parse_pod_scalar(scalar->string);
    if (!parsed_scalar) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
    result.identity.fields.push_back(PodFieldIdentityRecord{
        .name = field_name->string,
        .byte_offset = static_cast<std::uint32_t>(parsed_offset),
        .scalar_kind = *parsed_scalar,
        .element_count = static_cast<std::uint32_t>(parsed_count),
    });
  }
  auto derived_evidence = channel_internal::derive_pod_contract_evidence(result.identity);
  if (!derived_evidence || derived_evidence->identity_digest != result.expected_digest ||
      derived_evidence->layout_digest != result.expected_layout_digest ||
      derived_evidence->key != result.expected_key) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  return result;
}

struct ParsedPayload {
  PayloadIdentityRecord identity;
  std::optional<TensorBoundaryContractRecord> tensor;
  std::vector<DescriptorTensorTopicFieldRecord> tensor_fields;
  std::uint64_t serialized_payload_capacity_bytes{};
  std::uint64_t required_frame_bytes{};
  std::uint64_t selected_inline_bucket_bytes{};
  std::uint64_t bucket_usable_serialized_bytes{};
};

[[nodiscard]] expected<TensorBoundaryContractRecord, Error> parse_tensor_contract(
    const JsonValue& tensor, const bool require_memory_conversion) {
  constexpr std::array<std::string_view, 5> kTensorFieldsWithConversion{
      "role", "representation", "bounds", "transfer", "memory_conversion"};
  constexpr std::array<std::string_view, 4> kTensorFieldsWithoutConversion{
      "role", "representation", "bounds", "transfer"};
  constexpr std::array<std::string_view, 4> kRepresentationFields{
      "memory_kind", "dtype", "rank", "strides"};
  constexpr std::array<std::string_view, 3> kBoundsFields{"max_rank", "max_byte_span", "axes"};
  constexpr std::array<std::string_view, 3> kTransferFields{
      "foreign_emit_policy", "max_inline_bytes", "max_staging_byte_span"};
  const JsonValue* role = json_field(tensor, "role");
  const JsonValue* representation = json_field(tensor, "representation");
  const JsonValue* bounds = json_field(tensor, "bounds");
  const JsonValue* transfer = json_field(tensor, "transfer");
  const JsonValue* conversion = json_field(tensor, "memory_conversion");
  const bool tensor_shape_valid =
      require_memory_conversion ? allowed_fields(tensor, kTensorFieldsWithConversion) &&
                                      tensor.object.size() == kTensorFieldsWithConversion.size()
                                : allowed_fields(tensor, kTensorFieldsWithoutConversion) &&
                                      tensor.object.size() == kTensorFieldsWithoutConversion.size();
  if (!tensor_shape_valid || role == nullptr || role->kind != JsonValue::Kind::kString ||
      role->string != "external_boundary" || representation == nullptr ||
      representation->kind != JsonValue::Kind::kObject ||
      !allowed_fields(*representation, kRepresentationFields) ||
      representation->object.size() != kRepresentationFields.size() || bounds == nullptr ||
      bounds->kind != JsonValue::Kind::kObject || !allowed_fields(*bounds, kBoundsFields) ||
      bounds->object.size() != kBoundsFields.size() || transfer == nullptr ||
      transfer->kind != JsonValue::Kind::kObject || !allowed_fields(*transfer, kTransferFields) ||
      transfer->object.size() != kTransferFields.size()) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  const JsonValue* memory_kind = json_field(*representation, "memory_kind");
  const JsonValue* dtype = json_field(*representation, "dtype");
  const JsonValue* rank = json_field(*representation, "rank");
  const JsonValue* stride = json_field(*representation, "strides");
  const JsonValue* max_rank = json_field(*bounds, "max_rank");
  const JsonValue* max_span = json_field(*bounds, "max_byte_span");
  const JsonValue* axes = json_field(*bounds, "axes");
  const JsonValue* emit = json_field(*transfer, "foreign_emit_policy");
  const JsonValue* inline_bytes = json_field(*transfer, "max_inline_bytes");
  const JsonValue* staging_bytes = json_field(*transfer, "max_staging_byte_span");
  if (memory_kind == nullptr || dtype == nullptr || rank == nullptr || stride == nullptr ||
      stride->kind != JsonValue::Kind::kString || max_rank == nullptr || max_span == nullptr ||
      axes == nullptr || axes->kind != JsonValue::Kind::kArray || emit == nullptr ||
      emit->kind != JsonValue::Kind::kString || inline_bytes == nullptr ||
      staging_bytes == nullptr || (require_memory_conversion && conversion == nullptr)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  if (axes->array.size() > kMaximumTensorAxes) {
    return make_unexpected(catalog_resource_limit_error("/services/*/payload/tensor/axes",
                                                        "normalized_tensor_axes",
                                                        kMaximumTensorAxes,
                                                        axes->array.size()));
  }
  TensorBoundaryContractRecord result;
  if (memory_kind->kind == JsonValue::Kind::kString) {
    result.memory_kind = parse_memory_kind(memory_kind->string);
    if (!result.memory_kind) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
  } else if (memory_kind->kind != JsonValue::Kind::kNull) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  if (dtype->kind == JsonValue::Kind::kObject) {
    constexpr std::array<std::string_view, 3> kDtypeFields{"code", "bits", "lanes"};
    const JsonValue* code = json_field(*dtype, "code");
    const JsonValue* bits = json_field(*dtype, "bits");
    const JsonValue* lanes = json_field(*dtype, "lanes");
    std::uint64_t parsed_code{};
    std::uint64_t parsed_bits{};
    std::uint64_t parsed_lanes{};
    if (!allowed_fields(*dtype, kDtypeFields) || dtype->object.size() != kDtypeFields.size() ||
        code == nullptr || !json_unsigned(*code, parsed_code) ||
        parsed_code > std::numeric_limits<std::uint8_t>::max() || bits == nullptr ||
        !json_unsigned(*bits, parsed_bits) ||
        parsed_bits > std::numeric_limits<std::uint8_t>::max() || lanes == nullptr ||
        !json_unsigned(*lanes, parsed_lanes) ||
        parsed_lanes > std::numeric_limits<std::uint16_t>::max()) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
    const DLDataType parsed_dtype{.code = static_cast<std::uint8_t>(parsed_code),
                                  .bits = static_cast<std::uint8_t>(parsed_bits),
                                  .lanes = static_cast<std::uint16_t>(parsed_lanes)};
    if (!detail::valid_tensor_dtype(parsed_dtype)) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
    result.dtype = parsed_dtype;
  } else if (dtype->kind != JsonValue::Kind::kNull) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  const auto parse_optional_unsigned =
      [](const JsonValue& field, std::uint64_t maximum, std::optional<std::uint64_t>& parsed) {
        if (field.kind == JsonValue::Kind::kNull) {
          parsed.reset();
          return true;
        }
        std::uint64_t value{};
        if (!json_unsigned(field, value) || value > maximum) {
          return false;
        }
        parsed = value;
        return true;
      };
  std::optional<std::uint64_t> parsed_rank;
  std::optional<std::uint64_t> parsed_max_rank;
  std::optional<std::uint64_t> parsed_span;
  std::optional<std::uint64_t> parsed_inline;
  std::optional<std::uint64_t> parsed_staging;
  if (!parse_optional_unsigned(*rank, std::numeric_limits<std::uint32_t>::max(), parsed_rank) ||
      !parse_optional_unsigned(
          *max_rank, std::numeric_limits<std::uint32_t>::max(), parsed_max_rank) ||
      !parse_optional_unsigned(*max_span, std::numeric_limits<std::uint64_t>::max(), parsed_span) ||
      !parse_optional_unsigned(
          *inline_bytes, std::numeric_limits<std::uint32_t>::max(), parsed_inline) ||
      !parse_optional_unsigned(
          *staging_bytes, std::numeric_limits<std::uint64_t>::max(), parsed_staging)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  if (parsed_rank) {
    result.rank = static_cast<std::uint32_t>(*parsed_rank);
  }
  if (parsed_max_rank) {
    result.max_rank = static_cast<std::uint32_t>(*parsed_max_rank);
  }
  result.max_byte_span = parsed_span;
  if (parsed_inline) {
    result.max_inline_bytes = static_cast<std::uint32_t>(*parsed_inline);
  }
  result.max_staging_byte_span = parsed_staging;
  result.axes.reserve(axes->array.size());
  for (const JsonValue& axis : axes->array) {
    constexpr std::array<std::string_view, 2> kAxisFields{"max_extent", "max_abs_stride_bytes"};
    const JsonValue* extent = json_field(axis, "max_extent");
    const JsonValue* stride_bytes = json_field(axis, "max_abs_stride_bytes");
    std::uint64_t parsed_extent{};
    std::uint64_t parsed_stride{};
    if (!allowed_fields(axis, kAxisFields) || axis.object.size() != kAxisFields.size() ||
        extent == nullptr || !json_unsigned(*extent, parsed_extent) || stride_bytes == nullptr ||
        !json_unsigned(*stride_bytes, parsed_stride)) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
    result.axes.push_back(
        TensorAxisBounds{.max_extent = parsed_extent, .max_abs_stride_bytes = parsed_stride});
  }
  if (stride->string == "contiguous_only") {
    result.strides = TensorStridePolicy::kContiguousOnly;
  } else if (stride->string == "bounded_strided") {
    result.strides = TensorStridePolicy::kBoundedStrided;
  } else {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  if (emit->string == "require_exportable") {
    result.foreign_emit_policy = InterfaceTensorEmitPolicy::kRequireExportable;
  } else if (emit->string == "copy_if_needed") {
    result.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
  } else if (emit->string == "inline_small_host_then_copy_if_needed") {
    result.foreign_emit_policy = InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded;
  } else {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  if (require_memory_conversion) {
    if (conversion->kind == JsonValue::Kind::kNull) {
      result.memory_conversion.reset();
    } else if (conversion->kind != JsonValue::Kind::kString) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    } else {
      result.memory_conversion = parse_memory_conversion_token(conversion->string);
      if (!result.memory_conversion) {
        return make_unexpected(
            catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
      }
    }
  }
  return result;
}

// Keep the closed payload grammar in one parser so bounds and allowed fields stay synchronized.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<ParsedPayload, Error> parse_payload(const JsonValue& value) {
  constexpr std::array<std::string_view, 11> kFields{"family",
                                                     "scalar",
                                                     "tensor",
                                                     "schema_identity",
                                                     "pod_identity",
                                                     "transport_profile_tag",
                                                     "writer_max_serialized_bytes",
                                                     "required_frame_bytes",
                                                     "selected_inline_bucket_bytes",
                                                     "bucket_usable_serialized_bytes",
                                                     "descriptor_tensor_fields"};
  if (!allowed_fields(value, kFields)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  const JsonValue* family = json_field(value, "family");
  if (family == nullptr || family->kind != JsonValue::Kind::kString) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  ParsedPayload result;
  if (family->string == "scalar") {
    const JsonValue* scalar = json_field(value, "scalar");
    if (value.object.size() != 2U || scalar == nullptr ||
        scalar->kind != JsonValue::Kind::kString) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
    const auto kind = parse_scalar_identity(scalar->string);
    if (!kind) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
    result.identity.kind = PayloadIdentityKind::kScalar;
    result.identity.payload_kind = PayloadKind::kInlineTyped;
    result.identity.scalar = ScalarIdentityRecord{.kind = *kind};
  } else if (family->string == "tensor") {
    const JsonValue* tensor = json_field(value, "tensor");
    if (value.object.size() != 2U || tensor == nullptr) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
    result.identity.kind = PayloadIdentityKind::kTensor;
    result.identity.payload_kind = PayloadKind::kDescriptorBacked;
    if (tensor->kind == JsonValue::Kind::kNull) {
      return result;
    }
    auto parsed_tensor = parse_tensor_contract(*tensor, true);
    if (!parsed_tensor) {
      return make_unexpected(std::move(parsed_tensor).error());
    }
    result.tensor = std::move(parsed_tensor).value();
  } else if (family->string == "schema") {
    const JsonValue* schema = json_field(value, "schema_identity");
    const JsonValue* profile_tag = json_field(value, "transport_profile_tag");
    const JsonValue* maximum = json_field(value, "writer_max_serialized_bytes");
    const JsonValue* required_frame = json_field(value, "required_frame_bytes");
    const JsonValue* selected_bucket = json_field(value, "selected_inline_bucket_bytes");
    const JsonValue* usable_serialized = json_field(value, "bucket_usable_serialized_bytes");
    const JsonValue* descriptor_fields = json_field(value, "descriptor_tensor_fields");
    std::uint64_t parsed_maximum{};
    std::uint64_t parsed_profile_tag{};
    const bool descriptor_schema = descriptor_fields != nullptr;
    std::uint64_t parsed_required_frame{};
    std::uint64_t parsed_selected_bucket{};
    std::uint64_t parsed_usable_serialized{};
    if (value.object.size() != (descriptor_schema ? 8U : 7U) || schema == nullptr ||
        profile_tag == nullptr || !json_unsigned(*profile_tag, parsed_profile_tag) ||
        parsed_profile_tag > std::numeric_limits<std::uint32_t>::max() ||
        descriptor_schema != (parsed_profile_tag != 0U) || maximum == nullptr ||
        !json_unsigned(*maximum, parsed_maximum) || parsed_maximum == 0U ||
        parsed_maximum > std::numeric_limits<std::uint32_t>::max() || required_frame == nullptr ||
        !json_unsigned(*required_frame, parsed_required_frame) || parsed_required_frame == 0U ||
        selected_bucket == nullptr || !json_unsigned(*selected_bucket, parsed_selected_bucket) ||
        parsed_selected_bucket == 0U || usable_serialized == nullptr ||
        !json_unsigned(*usable_serialized, parsed_usable_serialized) ||
        parsed_usable_serialized == 0U ||
        (descriptor_schema && (descriptor_fields->kind != JsonValue::Kind::kArray ||
                               descriptor_fields->array.empty()))) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
    auto parsed_schema = parse_schema_identity(*schema);
    if (!parsed_schema) {
      return make_unexpected(std::move(parsed_schema).error());
    }
    result.identity.kind = PayloadIdentityKind::kSchema;
    result.identity.payload_kind =
        descriptor_schema ? PayloadKind::kDescriptorBacked : PayloadKind::kSerializedBytes;
    result.identity.transport_profile_tag = static_cast<std::uint32_t>(parsed_profile_tag);
    result.identity.schema = std::move(parsed_schema).value();
    result.serialized_payload_capacity_bytes = parsed_maximum;
    result.required_frame_bytes = parsed_required_frame;
    result.selected_inline_bucket_bytes = parsed_selected_bucket;
    result.bucket_usable_serialized_bytes = parsed_usable_serialized;
    if (descriptor_schema) {
      result.tensor_fields.reserve(descriptor_fields->array.size());
      for (const JsonValue& field : descriptor_fields->array) {
        constexpr std::array<std::string_view, 3> kFieldFields{
            "field_id", "tensor_index", "contract"};
        const JsonValue* field_id = json_field(field, "field_id");
        const JsonValue* tensor_index = json_field(field, "tensor_index");
        const JsonValue* contract = json_field(field, "contract");
        std::uint64_t parsed_field_id{};
        std::uint64_t parsed_tensor_index{};
        if (!allowed_fields(field, kFieldFields) || field.object.size() != kFieldFields.size() ||
            field_id == nullptr || !json_unsigned(*field_id, parsed_field_id) ||
            parsed_field_id > std::numeric_limits<std::uint16_t>::max() ||
            tensor_index == nullptr || !json_unsigned(*tensor_index, parsed_tensor_index) ||
            parsed_tensor_index > std::numeric_limits<std::uint16_t>::max() ||
            contract == nullptr || contract->kind != JsonValue::Kind::kObject) {
          return make_unexpected(
              catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
        }
        auto parsed_contract = parse_tensor_contract(*contract, true);
        if (!parsed_contract) {
          return make_unexpected(std::move(parsed_contract).error());
        }
        result.tensor_fields.push_back(DescriptorTensorTopicFieldRecord{
            .field_id = static_cast<std::uint16_t>(parsed_field_id),
            .tensor_index = static_cast<std::uint16_t>(parsed_tensor_index),
            .tensor = std::move(parsed_contract).value(),
        });
      }
    }
  } else if (family->string == "pod") {
    const JsonValue* pod = json_field(value, "pod_identity");
    if (value.object.size() != 2U || pod == nullptr) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
    }
    auto parsed_pod = parse_pod_identity(*pod);
    if (!parsed_pod) {
      return make_unexpected(std::move(parsed_pod).error());
    }
    result.identity.kind = PayloadIdentityKind::kPod;
    result.identity.payload_kind = PayloadKind::kInlineTyped;
    result.identity.pod = std::move(parsed_pod).value().identity;
  } else {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_PAYLOAD_IDENTITY_INVALID"));
  }
  return result;
}

struct ParsedService {
  std::string id;
  std::string binding;
  ParsedPayload payload;
  QueuePolicy queue_policy{QueuePolicy::kBlocking};
  std::uint32_t max_subscriber_depth{};
  std::uint32_t max_subscriber_borrowed{};
};

[[nodiscard]] expected<ParsedService, Error> parse_service(const JsonValue& value) {
  constexpr std::array<std::string_view, 5> kFields{
      "id", "binding", "payload", "queue", "event_wake"};
  if (!allowed_fields(value, kFields)) {
    return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_SERVICE_INVALID"));
  }
  const JsonValue* id = json_field(value, "id");
  const JsonValue* binding = json_field(value, "binding");
  const JsonValue* payload = json_field(value, "payload");
  const JsonValue* queue = json_field(value, "queue");
  const JsonValue* event_wake = json_field(value, "event_wake");
  constexpr std::array<std::string_view, 4> kQueueFields{
      "policy", "max_subscriber_depth", "max_subscriber_borrowed", "publisher_history_depth"};
  if (id == nullptr || id->kind != JsonValue::Kind::kString || binding == nullptr ||
      binding->kind != JsonValue::Kind::kString || payload == nullptr || queue == nullptr ||
      !allowed_fields(*queue, kQueueFields) || event_wake == nullptr ||
      event_wake->kind != JsonValue::Kind::kBoolean || !event_wake->boolean) {
    return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_SERVICE_INVALID"));
  }
  const JsonValue* policy = json_field(*queue, "policy");
  const JsonValue* max_depth = json_field(*queue, "max_subscriber_depth");
  const JsonValue* max_borrowed = json_field(*queue, "max_subscriber_borrowed");
  const JsonValue* history = json_field(*queue, "publisher_history_depth");
  std::uint64_t parsed_depth{};
  std::uint64_t parsed_borrowed{};
  std::uint64_t parsed_history{};
  if (policy == nullptr || policy->kind != JsonValue::Kind::kString ||
      (policy->string != "blocking" && policy->string != "drop_oldest") || max_depth == nullptr ||
      !json_unsigned(*max_depth, parsed_depth) || parsed_depth == 0U ||
      parsed_depth > std::numeric_limits<std::uint32_t>::max() || max_borrowed == nullptr ||
      !json_unsigned(*max_borrowed, parsed_borrowed) || parsed_borrowed == 0U ||
      parsed_borrowed > std::numeric_limits<std::uint32_t>::max() || history == nullptr ||
      !json_unsigned(*history, parsed_history) || parsed_history != 0U) {
    return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_SERVICE_INVALID"));
  }
  auto parsed_payload = parse_payload(*payload);
  if (!parsed_payload) {
    return make_unexpected(std::move(parsed_payload).error());
  }
  QueuePolicy queue_policy = QueuePolicy::kBlocking;
  if (policy->string == "drop_oldest") {
    queue_policy = QueuePolicy::kDropOldest;
  }
  return ParsedService{
      .id = id->string,
      .binding = binding->string,
      .payload = std::move(parsed_payload).value(),
      .queue_policy = queue_policy,
      .max_subscriber_depth = static_cast<std::uint32_t>(parsed_depth),
      .max_subscriber_borrowed = static_cast<std::uint32_t>(parsed_borrowed),
  };
}

[[nodiscard]] expected<CatalogClockRecord, Error> parse_clock(const JsonValue& value) {
  constexpr std::array<std::string_view, 7> kFields{
      "id", "name", "kind", "unit", "epoch", "monotonicity", "discipline"};
  if (!allowed_fields(value, kFields)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_CLOCK_INVALID"));
  }
  const JsonValue* id = json_field(value, "id");
  const JsonValue* name = json_field(value, "name");
  const JsonValue* kind = json_field(value, "kind");
  const JsonValue* unit = json_field(value, "unit");
  const JsonValue* epoch = json_field(value, "epoch");
  const JsonValue* monotonicity = json_field(value, "monotonicity");
  const JsonValue* discipline = json_field(value, "discipline");
  constexpr std::array<std::string_view, 2> kDisciplineFields{"kind", "identity"};
  std::uint64_t parsed_id{};
  if (id == nullptr || !json_unsigned(*id, parsed_id) || parsed_id == 0U || name == nullptr ||
      name->kind != JsonValue::Kind::kString || kind == nullptr ||
      kind->kind != JsonValue::Kind::kString || unit == nullptr ||
      unit->kind != JsonValue::Kind::kString || epoch == nullptr ||
      epoch->kind != JsonValue::Kind::kString || monotonicity == nullptr ||
      monotonicity->kind != JsonValue::Kind::kString || discipline == nullptr ||
      !allowed_fields(*discipline, kDisciplineFields)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_CLOCK_INVALID"));
  }
  const JsonValue* discipline_kind = json_field(*discipline, "kind");
  const JsonValue* discipline_identity = json_field(*discipline, "identity");
  if (discipline_kind == nullptr || discipline_kind->kind != JsonValue::Kind::kString ||
      discipline_identity == nullptr || discipline_identity->kind != JsonValue::Kind::kString) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_CLOCK_INVALID"));
  }
  return CatalogClockRecord{
      .id = parsed_id,
      .key = ClockKey{name->string},
      .kind = kind->string,
      .unit = unit->string,
      .epoch = epoch->string,
      .monotonicity = monotonicity->string,
      .discipline_kind = discipline_kind->string,
      .discipline_identity = discipline_identity->string,
  };
}

[[nodiscard]] expected<CatalogClockConversionRecord, Error> parse_clock_conversion(
    const JsonValue& value) {
  constexpr std::array<std::string_view, 9> kFields{"from",
                                                    "to",
                                                    "scale_numerator",
                                                    "scale_denominator",
                                                    "offset_ns",
                                                    "rounding",
                                                    "admitted_min",
                                                    "admitted_max",
                                                    "provenance"};
  if (!allowed_fields(value, kFields)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_CLOCK_CONVERSION_INVALID"));
  }
  const JsonValue* from = json_field(value, "from");
  const JsonValue* to = json_field(value, "to");
  const JsonValue* numerator = json_field(value, "scale_numerator");
  const JsonValue* denominator = json_field(value, "scale_denominator");
  const JsonValue* offset = json_field(value, "offset_ns");
  const JsonValue* rounding = json_field(value, "rounding");
  const JsonValue* admitted_min = json_field(value, "admitted_min");
  const JsonValue* admitted_max = json_field(value, "admitted_max");
  const JsonValue* provenance = json_field(value, "provenance");
  std::int64_t parsed_numerator{};
  std::uint64_t parsed_denominator{};
  std::int64_t parsed_offset{};
  std::int64_t parsed_min{};
  std::int64_t parsed_max{};
  const auto parse_string_signed = [](const JsonValue* field, std::int64_t& result) noexcept {
    if (field == nullptr || field->kind != JsonValue::Kind::kString || field->string.empty()) {
      return false;
    }
    const auto conversion =
        std::from_chars(field->string.data(), field->string.data() + field->string.size(), result);
    return conversion.ec == std::errc() &&
           conversion.ptr == field->string.data() + field->string.size();
  };
  const auto parse_string_unsigned = [](const JsonValue* field, std::uint64_t& result) noexcept {
    if (field == nullptr || field->kind != JsonValue::Kind::kString || field->string.empty()) {
      return false;
    }
    const auto conversion =
        std::from_chars(field->string.data(), field->string.data() + field->string.size(), result);
    return conversion.ec == std::errc() &&
           conversion.ptr == field->string.data() + field->string.size();
  };
  if (from == nullptr || from->kind != JsonValue::Kind::kString || to == nullptr ||
      to->kind != JsonValue::Kind::kString || !parse_string_signed(numerator, parsed_numerator) ||
      !parse_string_unsigned(denominator, parsed_denominator) ||
      !parse_string_signed(offset, parsed_offset) || rounding == nullptr ||
      rounding->kind != JsonValue::Kind::kString || rounding->string != "ties_to_even" ||
      !parse_string_signed(admitted_min, parsed_min) ||
      !parse_string_signed(admitted_max, parsed_max) || provenance == nullptr ||
      provenance->kind != JsonValue::Kind::kString) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_CLOCK_CONVERSION_INVALID"));
  }
  return CatalogClockConversionRecord{
      .from = ClockKey{from->string},
      .to = ClockKey{to->string},
      .scale_numerator = parsed_numerator,
      .scale_denominator = parsed_denominator,
      .offset_ns = parsed_offset,
      .admitted_min = parsed_min,
      .admitted_max = parsed_max,
      .provenance = provenance->string,
  };
}

// Endpoint parsing validates the complete record before publishing any partial state.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<ChannelEndpointRecord, Error> parse_endpoint(
    const JsonValue& value, std::span<const ParsedService> services, std::size_t endpoint_index) {
  constexpr std::array<std::string_view, 12> kFields{"name",
                                                     "name_origin",
                                                     "generated",
                                                     "uri",
                                                     "direction",
                                                     "service",
                                                     "memory_conversion",
                                                     "publisher",
                                                     "subscriber",
                                                     "capture_time",
                                                     "clock_health",
                                                     "attach"};
  if (!allowed_fields(value, kFields)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_FORMAT_UNSUPPORTED"));
  }
  const JsonValue* direction = json_field(value, "direction");
  const JsonValue* generated = json_field(value, "generated");
  const JsonValue* name = json_field(value, "name");
  const JsonValue* name_origin = json_field(value, "name_origin");
  const JsonValue* uri = json_field(value, "uri");
  const JsonValue* service = json_field(value, "service");
  const JsonValue* memory_conversion = json_field(value, "memory_conversion");
  const JsonValue* publisher = json_field(value, "publisher");
  const JsonValue* subscriber = json_field(value, "subscriber");
  const JsonValue* capture_time = json_field(value, "capture_time");
  const JsonValue* clock_health = json_field(value, "clock_health");
  const JsonValue* attach = json_field(value, "attach");
  if (direction == nullptr || direction->kind != JsonValue::Kind::kString || generated == nullptr ||
      name == nullptr || name->kind != JsonValue::Kind::kString || name_origin == nullptr ||
      name_origin->kind != JsonValue::Kind::kString || uri == nullptr ||
      uri->kind != JsonValue::Kind::kString || service == nullptr ||
      service->kind != JsonValue::Kind::kString || memory_conversion == nullptr ||
      memory_conversion->kind != JsonValue::Kind::kString || publisher == nullptr ||
      subscriber == nullptr || capture_time == nullptr || clock_health == nullptr ||
      attach == nullptr) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_FORMAT_UNSUPPORTED"));
  }

  TopicDirection parsed_direction{};
  if (direction->string == "import") {
    parsed_direction = TopicDirection::kImport;
  } else if (direction->string == "export") {
    parsed_direction = TopicDirection::kExport;
  } else {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_DIRECTION_INVALID"));
  }
  EndpointNameOrigin parsed_origin{};
  if (name_origin->string == "explicit") {
    parsed_origin = EndpointNameOrigin::kExplicit;
  } else if (name_origin->string == "generated") {
    parsed_origin = EndpointNameOrigin::kGenerated;
  } else {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_NAME_GENERATED_METADATA_INVALID"));
  }
  auto parsed_generated = parse_generated_name(*generated);
  if (!parsed_generated) {
    return make_unexpected(std::move(parsed_generated).error());
  }
  const auto service_record = std::find_if(services.begin(), services.end(), [&](const auto& item) {
    return item.id == service->string;
  });
  if (service_record == services.end()) {
    return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_SERVICE_INVALID"));
  }

  const auto parsed_conversion = parse_memory_conversion_token(memory_conversion->string);
  if (!parsed_conversion) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_MEMORY_CONVERSION_INVALID"));
  }

  EndpointBounds bounds;
  if (publisher->kind != JsonValue::Kind::kNull) {
    constexpr std::array<std::string_view, 1> kPublisherFields{"max_loaned_samples"};
    const JsonValue* max_loaned = json_field(*publisher, "max_loaned_samples");
    std::uint64_t parsed_max{};
    if (!allowed_fields(*publisher, kPublisherFields) || max_loaned == nullptr ||
        !json_unsigned(*max_loaned, parsed_max) || parsed_max == 0U ||
        parsed_max > std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_BOUNDS_INVALID"));
    }
    bounds.publisher =
        PublisherEndpointBounds{.max_loaned_samples = static_cast<std::uint32_t>(parsed_max)};
  }
  if (subscriber->kind != JsonValue::Kind::kNull) {
    constexpr std::array<std::string_view, 2> kSubscriberFields{"queue_depth",
                                                                "max_borrowed_samples"};
    const JsonValue* depth = json_field(*subscriber, "queue_depth");
    const JsonValue* borrowed = json_field(*subscriber, "max_borrowed_samples");
    std::uint64_t parsed_depth{};
    std::uint64_t parsed_borrowed{};
    if (!allowed_fields(*subscriber, kSubscriberFields) || depth == nullptr ||
        !json_unsigned(*depth, parsed_depth) || parsed_depth == 0U ||
        parsed_depth > std::numeric_limits<std::uint32_t>::max() || borrowed == nullptr ||
        !json_unsigned(*borrowed, parsed_borrowed) || parsed_borrowed == 0U ||
        parsed_borrowed > std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_BOUNDS_INVALID"));
    }
    bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = static_cast<std::uint32_t>(parsed_depth),
        .max_borrowed_samples = static_cast<std::uint32_t>(parsed_borrowed),
    };
  }
  if ((parsed_direction == TopicDirection::kImport && bounds.subscriber) ||
      (parsed_direction == TopicDirection::kExport && bounds.publisher)) {
    return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_BOUNDS_INVALID"));
  }

  std::optional<ClockKey> capture_clock;
  if (capture_time->kind != JsonValue::Kind::kNull) {
    constexpr std::array<std::string_view, 2> kCaptureFields{"required", "clock"};
    const JsonValue* required = json_field(*capture_time, "required");
    const JsonValue* clock = json_field(*capture_time, "clock");
    if (!allowed_fields(*capture_time, kCaptureFields) || required == nullptr ||
        required->kind != JsonValue::Kind::kBoolean || !required->boolean || clock == nullptr ||
        clock->kind != JsonValue::Kind::kString) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CAPTURE_CLOCK_INVALID"));
    }
    capture_clock = ClockKey{clock->string};
  }

  constexpr std::array<std::string_view, 3> kHealthFields{
      "capability", "requirement", "max_report_age_ns"};
  const JsonValue* capability = clock_health->kind == JsonValue::Kind::kObject
                                    ? json_field(*clock_health, "capability")
                                    : nullptr;
  const JsonValue* requirement = clock_health->kind == JsonValue::Kind::kObject
                                     ? json_field(*clock_health, "requirement")
                                     : nullptr;
  const JsonValue* max_age = clock_health->kind == JsonValue::Kind::kObject
                                 ? json_field(*clock_health, "max_report_age_ns")
                                 : nullptr;
  const auto health_error = [&](std::string_view capability_value,
                                std::string_view requirement_value,
                                std::string_view max_age_value) {
    return raw_clock_health_error(endpoint_index,
                                  name->string,
                                  capability_value,
                                  requirement_value,
                                  max_age_value,
                                  direction->string);
  };
  ClockHealthOptions health;
  if (!allowed_fields(*clock_health, kHealthFields) || capability == nullptr ||
      capability->kind != JsonValue::Kind::kString || requirement == nullptr ||
      requirement->kind != JsonValue::Kind::kString) {
    return make_unexpected(
        health_error(capability != nullptr && capability->kind == JsonValue::Kind::kString
                         ? std::string_view{capability->string}
                         : "invalid_or_absent",
                     requirement != nullptr && requirement->kind == JsonValue::Kind::kString
                         ? std::string_view{requirement->string}
                         : "invalid_or_absent",
                     max_age == nullptr ? "absent" : "invalid"));
  }
  if (capability->string == "unavailable") {
    health.capability = ClockHealthCapability::kUnavailable;
  } else if (capability->string == "status_reports") {
    health.capability = ClockHealthCapability::kStatusReports;
  } else {
    return make_unexpected(health_error(
        capability->string, requirement->string, max_age == nullptr ? "absent" : "invalid"));
  }
  if (requirement->string == "not_required") {
    health.requirement = ClockHealthRequirement::kNotRequired;
  } else if (requirement->string == "fresh_healthy_report") {
    health.requirement = ClockHealthRequirement::kFreshHealthyReport;
  } else {
    return make_unexpected(health_error(
        capability->string, requirement->string, max_age == nullptr ? "absent" : "invalid"));
  }
  if (max_age != nullptr) {
    std::uint64_t parsed_age{};
    if (!json_unsigned(*max_age, parsed_age) ||
        parsed_age > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return make_unexpected(health_error(capability->string, requirement->string, "invalid"));
    }
    health.max_report_age = std::chrono::nanoseconds{static_cast<std::int64_t>(parsed_age)};
  }
  const bool ordinary_health = health.capability == ClockHealthCapability::kUnavailable &&
                               health.requirement == ClockHealthRequirement::kNotRequired &&
                               !health.max_report_age;
  const bool monitored_health = parsed_direction == TopicDirection::kImport &&
                                health.capability == ClockHealthCapability::kStatusReports &&
                                health.requirement == ClockHealthRequirement::kFreshHealthyReport &&
                                health.max_report_age && health.max_report_age->count() > 0;
  if (!ordinary_health && !monitored_health) {
    return make_unexpected(health_error(capability->string,
                                        requirement->string,
                                        health.max_report_age
                                            ? std::to_string(health.max_report_age->count())
                                            : std::string{"absent"}));
  }

  constexpr std::array<std::string_view, 2> kAttachFields{"peer", "max_wait_ns"};
  const JsonValue* peer = json_field(*attach, "peer");
  const JsonValue* max_wait = json_field(*attach, "max_wait_ns");
  std::uint64_t parsed_wait{};
  TopicAttachPolicy attach_policy;
  if (!allowed_fields(*attach, kAttachFields) || peer == nullptr ||
      peer->kind != JsonValue::Kind::kString || max_wait == nullptr ||
      !json_unsigned(*max_wait, parsed_wait) ||
      parsed_wait > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_ATTACH_POLICY_INVALID"));
  }
  if (peer->string == "optional") {
    attach_policy.peer = PeerRequirement::kOptional;
  } else if (peer->string == "required") {
    attach_policy.peer = PeerRequirement::kRequired;
  } else {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_ATTACH_POLICY_INVALID"));
  }
  attach_policy.max_wait = std::chrono::nanoseconds{static_cast<std::int64_t>(parsed_wait)};

  ChannelEndpointRecord result{
      .name = name->string,
      .uri = uri->string,
      .direction = parsed_direction,
      .name_origin = parsed_origin,
      .generated_name = std::move(parsed_generated).value(),
      .service = service->string,
      .queue_policy = service_record->queue_policy,
      .service_max_subscriber_depth = service_record->max_subscriber_depth,
      .service_max_subscriber_borrowed = service_record->max_subscriber_borrowed,
      .payload_identity = service_record->payload.identity,
      .schema_writer_max_serialized_bytes =
          service_record->payload.serialized_payload_capacity_bytes,
      .memory_conversion = parsed_conversion,
      .capture_time_clock = std::move(capture_clock),
      .clock_health = health,
      .attach = attach_policy,
      .tensor = service_record->payload.tensor,
      .tensor_fields = service_record->payload.tensor_fields,
      .bounds = bounds,
  };
  if (result.payload_identity.schema) {
    result.schema_identity = result.payload_identity.schema;
  }
  finalize_payload_identity(result);
  materialize_normalized_ownership(result);
  const bool subscriber_bounds_valid =
      !bounds.subscriber ||
      (bounds.subscriber->queue_depth && bounds.subscriber->max_borrowed_samples &&
       *bounds.subscriber->queue_depth <= service_record->max_subscriber_depth &&
       *bounds.subscriber->max_borrowed_samples <= service_record->max_subscriber_borrowed);
  const bool capacity_geometry_valid =
      result.payload_identity.kind != PayloadIdentityKind::kSchema ||
      (service_record->payload.required_frame_bytes ==
           result.service_static.required_data_frame_bytes &&
       service_record->payload.selected_inline_bucket_bytes ==
           result.service_static.selected_inline_bucket_bytes &&
       service_record->payload.bucket_usable_serialized_bytes ==
           result.service_static.bucket_usable_serialized_bytes);
  if (!subscriber_bounds_valid || !capacity_geometry_valid ||
      service_record->binding != service_binding(result)) {
    return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_SERVICE_INVALID"));
  }
  return result;
}

// Catalog parsing is a bounded, transactional parse-and-normalize operation.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<ChannelEndpointCatalog, Error> parse_catalog(std::string_view source) {
  if (source.size() > kMaximumCatalogSourceBytes) {
    return make_unexpected(catalog_resource_limit_error(
        "/", "source_bytes", kMaximumCatalogSourceBytes, source.size()));
  }
  if (!valid_utf8(source)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_PARSE_FAILED"));
  }
  auto parsed = JsonParser{source}.parse();
  if (!parsed) {
    return make_unexpected(std::move(parsed).error());
  }
  constexpr std::array<std::string_view, 8> kFields{"format",
                                                    "version",
                                                    "scope",
                                                    "digest",
                                                    "clocks",
                                                    "clock_conversions",
                                                    "services",
                                                    "endpoints"};
  if (!allowed_fields(*parsed, kFields)) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_FORMAT_UNSUPPORTED"));
  }
  const JsonValue* format = json_field(*parsed, "format");
  const JsonValue* version = json_field(*parsed, "version");
  const JsonValue* scope = json_field(*parsed, "scope");
  const JsonValue* digest = json_field(*parsed, "digest");
  const JsonValue* clocks = json_field(*parsed, "clocks");
  const JsonValue* conversions = json_field(*parsed, "clock_conversions");
  const JsonValue* services = json_field(*parsed, "services");
  const JsonValue* endpoints = json_field(*parsed, "endpoints");
  std::uint64_t parsed_version{};
  if (format == nullptr || format->kind != JsonValue::Kind::kString ||
      format->string != kCatalogFormat || version == nullptr ||
      !json_unsigned(*version, parsed_version) || parsed_version != kCatalogVersion ||
      scope == nullptr || scope->kind != JsonValue::Kind::kString || digest == nullptr ||
      digest->kind != JsonValue::Kind::kString || clocks == nullptr ||
      clocks->kind != JsonValue::Kind::kArray || conversions == nullptr ||
      conversions->kind != JsonValue::Kind::kArray || services == nullptr ||
      services->kind != JsonValue::Kind::kArray || endpoints == nullptr ||
      endpoints->kind != JsonValue::Kind::kArray) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_FORMAT_UNSUPPORTED"));
  }
  if (clocks->array.size() > kMaximumCatalogClocks) {
    return make_unexpected(catalog_resource_limit_error(
        "/clocks", "normalized_clocks", kMaximumCatalogClocks, clocks->array.size()));
  }
  if (conversions->array.size() > kMaximumCatalogClockConversions) {
    return make_unexpected(catalog_resource_limit_error("/clock_conversions",
                                                        "normalized_clock_conversions",
                                                        kMaximumCatalogClockConversions,
                                                        conversions->array.size()));
  }
  if (services->array.size() > detail::kMaximumCatalogEndpoints) {
    return make_unexpected(catalog_resource_limit_error("/services",
                                                        "normalized_services",
                                                        detail::kMaximumCatalogEndpoints,
                                                        services->array.size()));
  }
  if (endpoints->array.size() > detail::kMaximumCatalogEndpoints) {
    return make_unexpected(catalog_resource_limit_error("/endpoints",
                                                        "normalized_endpoints",
                                                        detail::kMaximumCatalogEndpoints,
                                                        endpoints->array.size()));
  }
  std::vector<CatalogClockRecord> clock_records;
  clock_records.reserve(clocks->array.size());
  for (const JsonValue& clock : clocks->array) {
    auto record = parse_clock(clock);
    if (!record) {
      return make_unexpected(std::move(record).error());
    }
    clock_records.push_back(std::move(record).value());
  }
  std::sort(clock_records.begin(), clock_records.end(), [](const auto& left, const auto& right) {
    return left.key.name() < right.key.name();
  });
  for (std::size_t index = 0U; index < clock_records.size(); ++index) {
    if (clock_records[index].id != index + 1U) {
      return make_unexpected(
          catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_CLOCK_INVALID"));
    }
  }

  std::vector<CatalogClockConversionRecord> conversion_records;
  conversion_records.reserve(conversions->array.size());
  for (const JsonValue& conversion : conversions->array) {
    auto record = parse_clock_conversion(conversion);
    if (!record) {
      return make_unexpected(std::move(record).error());
    }
    conversion_records.push_back(std::move(record).value());
  }

  std::vector<ParsedService> service_records;
  service_records.reserve(services->array.size());
  for (const JsonValue& service : services->array) {
    auto record = parse_service(service);
    if (!record) {
      return make_unexpected(std::move(record).error());
    }
    if (std::any_of(service_records.begin(), service_records.end(), [&](const auto& existing) {
          return existing.id == record->id;
        })) {
      return make_unexpected(
          catalog_error(ErrorCode::kDuplicateName, "ENDPOINT_SERVICE_DUPLICATE"));
    }
    service_records.push_back(std::move(record).value());
  }
  std::vector<ChannelEndpointRecord> records;
  records.reserve(endpoints->array.size());
  for (std::size_t endpoint_index = 0U; endpoint_index < endpoints->array.size();
       ++endpoint_index) {
    auto record = parse_endpoint(endpoints->array[endpoint_index], service_records, endpoint_index);
    if (!record) {
      return make_unexpected(std::move(record).error());
    }
    records.push_back(std::move(record).value());
  }
  if (service_records.size() != records.size() ||
      std::any_of(service_records.begin(), service_records.end(), [&](const auto& service) {
        return std::none_of(records.begin(), records.end(), [&](const auto& endpoint) {
          return endpoint.service == service.id;
        });
      })) {
    return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_SERVICE_INVALID"));
  }
  return normalize_catalog(scope->string,
                           std::move(records),
                           std::move(clock_records),
                           std::move(conversion_records),
                           digest->string,
                           source);
}

[[nodiscard]] std::filesystem::path temporary_candidate(const std::filesystem::path& destination) {
  static std::atomic<std::uint64_t> next_temporary{1U};
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path candidate = destination;
  candidate += ".tmp-" + std::to_string(channel_internal::current_process_identity()) + "-" +
               std::to_string(tick) + "-" +
               std::to_string(next_temporary.fetch_add(1U, std::memory_order_relaxed));
  return candidate;
}

enum class ExclusiveWriteResult : std::uint8_t { kSuccess, kCollision, kFailure };

enum class ExactFileMatch : std::uint8_t { kMatch, kMismatch, kFailure };

std::atomic<bool> oversize_next_staged_catalog_for_test{false};

// Compare in fixed chunks, then probe exactly one byte beyond the expected
// length so even a concurrently grown file consumes at most expected.size()+1.
[[nodiscard]] ExactFileMatch file_matches_exact_bytes(const std::filesystem::path& path,
                                                      std::string_view expected) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return ExactFileMatch::kFailure;
  }

  std::array<char, 4096U> buffer{};
  std::size_t offset = 0U;
  while (offset < expected.size()) {
    const std::size_t requested = std::min(buffer.size(), expected.size() - offset);
    stream.read(buffer.data(), static_cast<std::streamsize>(requested));
    const std::size_t bytes_read = static_cast<std::size_t>(stream.gcount());
    if (stream.bad() || (stream.fail() && !stream.eof())) {
      return ExactFileMatch::kFailure;
    }
    const std::string_view actual(buffer.data(), bytes_read);
    if (actual != expected.substr(offset, bytes_read)) {
      return ExactFileMatch::kMismatch;
    }
    offset += bytes_read;
    if (bytes_read != requested) {
      return ExactFileMatch::kMismatch;
    }
  }

  char extra_byte{};
  stream.read(&extra_byte, 1);
  if (stream.bad() || (stream.fail() && !stream.eof())) {
    return ExactFileMatch::kFailure;
  }
  return stream.gcount() == 0 ? ExactFileMatch::kMatch : ExactFileMatch::kMismatch;
}

[[nodiscard]] ExclusiveWriteResult write_exclusive_file(const std::filesystem::path& path,
                                                        std::string_view bytes) noexcept {
#if defined(_WIN32)
  HANDLE handle = CreateFileW(
      path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    return GetLastError() == ERROR_FILE_EXISTS || GetLastError() == ERROR_ALREADY_EXISTS
               ? ExclusiveWriteResult::kCollision
               : ExclusiveWriteResult::kFailure;
  }
  std::size_t offset = 0U;
  bool succeeded = true;
  while (offset < bytes.size()) {
    const DWORD remaining = static_cast<DWORD>(
        std::min<std::size_t>(bytes.size() - offset, static_cast<std::size_t>(MAXDWORD)));
    DWORD written = 0U;
    if (WriteFile(handle, bytes.data() + offset, remaining, &written, nullptr) == 0 ||
        written == 0U) {
      succeeded = false;
      break;
    }
    offset += written;
  }
  succeeded = succeeded && FlushFileBuffers(handle) != 0;
  succeeded = CloseHandle(handle) != 0 && succeeded;
  return succeeded ? ExclusiveWriteResult::kSuccess : ExclusiveWriteResult::kFailure;
#else
  const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (descriptor < 0) {
    return errno == EEXIST ? ExclusiveWriteResult::kCollision : ExclusiveWriteResult::kFailure;
  }
  std::size_t offset = 0U;
  bool succeeded = true;
  while (offset < bytes.size()) {
    const ssize_t written = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      succeeded = false;
      break;
    }
    offset += static_cast<std::size_t>(written);
  }
  succeeded = succeeded && ::fsync(descriptor) == 0;
  succeeded = ::close(descriptor) == 0 && succeeded;
  return succeeded ? ExclusiveWriteResult::kSuccess : ExclusiveWriteResult::kFailure;
#endif
}

[[nodiscard]] bool native_replace_file(const std::filesystem::path& source,
                                       const std::filesystem::path& destination) noexcept {
#if defined(_WIN32)
  return MoveFileExW(source.c_str(),
                     destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
  return ::rename(source.c_str(), destination.c_str()) == 0;
#endif
}

}  // namespace

void detail::EndpointCatalogTestAccess::oversize_next_staged_save() noexcept {
  oversize_next_staged_catalog_for_test.store(true, std::memory_order_relaxed);
}

void detail::EndpointCatalogTestAccess::reset_save_faults() noexcept {
  oversize_next_staged_catalog_for_test.store(false, std::memory_order_relaxed);
}

expected<ChannelEndpointCatalog, Error> ChannelEndpointCatalog::parse_json(
    std::string_view source) {
  return parse_catalog(source);
}

expected<ChannelEndpointCatalog, Error> ChannelEndpointCatalog::load_json(
    const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return make_unexpected(catalog_error(ErrorCode::kNotFound, "ENDPOINT_CATALOG_LOAD_FAILED"));
  }
  // Read at most limit+1 so a hostile or accidentally huge file cannot force
  // unbounded allocation before the schema-v2 ingress check runs.
  std::string source(kMaximumCatalogSourceBytes + 1U, '\0');
  stream.read(source.data(), static_cast<std::streamsize>(source.size()));
  const std::size_t bytes_read = static_cast<std::size_t>(stream.gcount());
  if (stream.bad()) {
    return make_unexpected(catalog_error(ErrorCode::kFailure, "ENDPOINT_CATALOG_LOAD_FAILED"));
  }
  source.resize(bytes_read);
  if (bytes_read > kMaximumCatalogSourceBytes) {
    return make_unexpected(
        catalog_resource_limit_error("/", "source_bytes", kMaximumCatalogSourceBytes, bytes_read));
  }
  return parse_catalog(source);
}

expected<void, Error> ChannelEndpointCatalog::save_json(const std::filesystem::path& path) const {
  if (state_ == nullptr || path.empty()) {
    return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CATALOG_EMPTY"));
  }
  const bool oversize_staged_for_test =
      oversize_next_staged_catalog_for_test.exchange(false, std::memory_order_relaxed);
  std::error_code filesystem_error;
  if (std::filesystem::exists(path, filesystem_error)) {
    if (filesystem_error) {
      return make_unexpected(catalog_error(ErrorCode::kFailure, "ENDPOINT_CATALOG_SAVE_FAILED"));
    }
    const ExactFileMatch existing_match = file_matches_exact_bytes(path, state_->json);
    if (existing_match == ExactFileMatch::kFailure) {
      return make_unexpected(catalog_error(ErrorCode::kFailure, "ENDPOINT_CATALOG_SAVE_FAILED"));
    }
    if (existing_match == ExactFileMatch::kMatch) {
      return {};
    }
  }

  constexpr std::size_t kMaximumTemporaryAttempts = 128U;
  std::filesystem::path temporary;
  for (std::size_t attempt = 0U; attempt < kMaximumTemporaryAttempts; ++attempt) {
    temporary = temporary_candidate(path);
    const ExclusiveWriteResult write_result = write_exclusive_file(temporary, state_->json);
    if (write_result == ExclusiveWriteResult::kCollision) {
      temporary.clear();
      continue;
    }
    if (write_result == ExclusiveWriteResult::kFailure) {
      std::filesystem::remove(temporary, filesystem_error);
      return make_unexpected(catalog_error(ErrorCode::kFailure, "ENDPOINT_CATALOG_SAVE_FAILED"));
    }
    break;
  }
  if (temporary.empty()) {
    return make_unexpected(catalog_error(ErrorCode::kFailure, "ENDPOINT_CATALOG_SAVE_FAILED"));
  }
  if (oversize_staged_for_test) {
    std::ofstream staged_mutation(temporary, std::ios::binary | std::ios::app);
    staged_mutation.put('\0');
    staged_mutation.close();
    if (!staged_mutation) {
      std::filesystem::remove(temporary, filesystem_error);
      return make_unexpected(catalog_error(ErrorCode::kFailure, "ENDPOINT_CATALOG_SAVE_FAILED"));
    }
  }
  const ExactFileMatch staged_match = file_matches_exact_bytes(temporary, state_->json);
  if (staged_match != ExactFileMatch::kMatch || !native_replace_file(temporary, path)) {
    std::filesystem::remove(temporary, filesystem_error);
    return make_unexpected(catalog_error(ErrorCode::kFailure, "ENDPOINT_CATALOG_SAVE_FAILED"));
  }
  return {};
}

std::string ChannelEndpointCatalog::json() const {
  return state_ != nullptr ? state_->json : std::string{};
}

std::string_view ChannelEndpointCatalog::digest() const noexcept {
  return state_ != nullptr ? std::string_view{state_->digest} : std::string_view{};
}

std::optional<std::string_view> ChannelEndpointCatalog::raw_source_sha256() const noexcept {
  if (state_ == nullptr) {
    return std::nullopt;
  }
  const auto& digest = state_->raw_source_sha256;
  if (!digest.has_value()) {
    return std::nullopt;
  }
  return std::string_view{digest.value()};
}

std::optional<std::uint64_t> ChannelEndpointCatalog::raw_source_size_bytes() const noexcept {
  return state_ == nullptr ? std::nullopt : state_->raw_source_size_bytes;
}

expected<ChannelEndpointCatalog::EndpointView, Error> ChannelEndpointCatalog::endpoint(
    std::string_view endpoint_name) const noexcept {
  if (state_ == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kNotFound});
  }

  const auto found = std::lower_bound(state_->endpoints.begin(),
                                      state_->endpoints.end(),
                                      endpoint_name,
                                      [](const ChannelEndpointRecord& endpoint,
                                         std::string_view name) { return endpoint.name < name; });
  if (found == state_->endpoints.end() || found->name != endpoint_name) {
    return make_unexpected(Error{.code = ErrorCode::kNotFound});
  }

  const auto index = static_cast<std::size_t>(found - state_->endpoints.begin());
  return EndpointView{state_.get(), index};
}

// Compatibility is a closed matrix over every endpoint contract field.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
ChannelEndpointCompatibilityReport ChannelEndpointCatalog::compatibility_with(
    const ChannelEndpointCatalog& candidate) const {
  auto report = std::make_shared<detail::ChannelEndpointCompatibilityReportState>();
  const auto append_diagnostic = [&](DiagnosticSeverity severity,
                                     std::string token,
                                     std::string service = {},
                                     std::string endpoint = {},
                                     std::string json_pointer = {},
                                     std::string required = {},
                                     std::string provided = {},
                                     std::optional<DiagnosticId> cause_id = std::nullopt) {
    if (severity == DiagnosticSeverity::kError) {
      report->compatible = false;
    }
    const DiagnosticId id{static_cast<std::uint32_t>(report->diagnostics.size() + 1U)};
    report->diagnostics.push_back(Diagnostic{
        .id = id,
        .severity = severity,
        .token = std::move(token),
        .service = std::move(service),
        .endpoint = std::move(endpoint),
        .json_pointer = std::move(json_pointer),
        .required = std::move(required),
        .provided = std::move(provided),
        .cause_id = cause_id,
    });
    return id;
  };
  const auto append_error = [&](std::string token,
                                std::string service = {},
                                std::string endpoint = {},
                                std::string json_pointer = {},
                                std::string required = {},
                                std::string provided = {},
                                std::optional<DiagnosticId> cause_id = std::nullopt) {
    return append_diagnostic(DiagnosticSeverity::kError,
                             std::move(token),
                             std::move(service),
                             std::move(endpoint),
                             std::move(json_pointer),
                             std::move(required),
                             std::move(provided),
                             cause_id);
  };
  struct PendingDiagnostic {
    DiagnosticSeverity severity{DiagnosticSeverity::kError};
    std::string token;
    std::string service;
    std::string endpoint;
    std::string json_pointer;
    std::string required;
    std::string provided;
    std::optional<std::size_t> service_cause_endpoint;
  };
  const auto flush_phase = [&](std::vector<PendingDiagnostic>& phase,
                               std::vector<std::optional<DiagnosticId>>* first_service_causes) {
    std::sort(phase.begin(), phase.end(), [](const auto& left, const auto& right) {
      return std::tie(left.service,
                      left.endpoint,
                      left.json_pointer,
                      left.token,
                      left.required,
                      left.provided) < std::tie(right.service,
                                                right.endpoint,
                                                right.json_pointer,
                                                right.token,
                                                right.required,
                                                right.provided);
    });
    for (auto& diagnostic : phase) {
      const DiagnosticId id = append_diagnostic(diagnostic.severity,
                                                std::move(diagnostic.token),
                                                std::move(diagnostic.service),
                                                std::move(diagnostic.endpoint),
                                                std::move(diagnostic.json_pointer),
                                                std::move(diagnostic.required),
                                                std::move(diagnostic.provided));
      if (first_service_causes != nullptr && diagnostic.service_cause_endpoint &&
          !(*first_service_causes)[*diagnostic.service_cause_endpoint]) {
        (*first_service_causes)[*diagnostic.service_cause_endpoint] = id;
      }
    }
    phase.clear();
  };
  if (state_ == nullptr || candidate.state_ == nullptr) {
    append_error("ENDPOINT_CATALOG_EMPTY",
                 {},
                 {},
                 "",
                 state_ == nullptr ? "absent" : "catalog",
                 candidate.state_ == nullptr ? "absent" : "catalog");
  } else {
    // Compatibility consumes validated immutable catalog objects. The first
    // phase therefore only has catalog-wide integrity/scope facts; malformed
    // format, duplicate-key, and digest errors are rejected by parse/load.
    if (state_->scope != candidate.state_->scope) {
      append_error(
          "ENDPOINT_SCOPE_MISMATCH", {}, {}, "/scope", state_->scope, candidate.state_->scope);
    }

    std::vector<std::size_t> required_order(state_->endpoints.size());
    for (std::size_t index = 0U; index < required_order.size(); ++index) {
      required_order[index] = index;
    }
    std::sort(
        required_order.begin(), required_order.end(), [&](std::size_t left, std::size_t right) {
          const auto& left_endpoint = state_->endpoints[left];
          const auto& right_endpoint = state_->endpoints[right];
          return std::tuple{left_endpoint.service_static.id, left_endpoint.name} <
                 std::tuple{right_endpoint.service_static.id, right_endpoint.name};
        });

    std::vector<std::optional<std::size_t>> matches(state_->endpoints.size());
    std::vector<bool> candidate_used(candidate.state_->endpoints.size());
    std::vector<std::optional<DiagnosticId>> first_service_cause(state_->endpoints.size());
    std::vector<PendingDiagnostic> lookup_diagnostics;

    // Phase 2: resolve stable names first. URI fallback is descriptive only and
    // must be unique; even a unique fallback retains an explicit pin mismatch.
    for (const std::size_t required_index : required_order) {
      const auto& required = state_->endpoints[required_index];
      const auto found =
          std::lower_bound(candidate.state_->endpoints.begin(),
                           candidate.state_->endpoints.end(),
                           required.name,
                           [](const ChannelEndpointRecord& endpoint, std::string_view name) {
                             return endpoint.name < name;
                           });
      if (found != candidate.state_->endpoints.end() && found->name == required.name) {
        const auto candidate_index =
            static_cast<std::size_t>(found - candidate.state_->endpoints.begin());
        matches[required_index] = candidate_index;
        candidate_used[candidate_index] = true;
        continue;
      }

      std::vector<std::size_t> uri_matches;
      for (std::size_t candidate_index = 0U; candidate_index < candidate.state_->endpoints.size();
           ++candidate_index) {
        if (!candidate_used[candidate_index] &&
            candidate.state_->endpoints[candidate_index].uri == required.uri) {
          uri_matches.push_back(candidate_index);
        }
      }
      const std::string pointer = "/endpoints/" + std::to_string(required_index);
      if (uri_matches.size() > 1U) {
        lookup_diagnostics.push_back(PendingDiagnostic{
            .token = "ENDPOINT_AMBIGUOUS",
            .service = required.service_static.id,
            .endpoint = required.name,
            .json_pointer = pointer,
            .required = required.name,
            .provided = std::to_string(uri_matches.size()) + " uri matches",
        });
      } else if (uri_matches.size() == 1U) {
        const std::size_t candidate_index = uri_matches.front();
        matches[required_index] = candidate_index;
        candidate_used[candidate_index] = true;
        lookup_diagnostics.push_back(PendingDiagnostic{
            .token = "ENDPOINT_NAME_GENERATED_PIN_MISMATCH",
            .service = required.service_static.id,
            .endpoint = required.name,
            .json_pointer = pointer + "/name",
            .required = required.name,
            .provided = candidate.state_->endpoints[candidate_index].name,
        });
      } else {
        lookup_diagnostics.push_back(PendingDiagnostic{
            .token = "ENDPOINT_NOT_FOUND",
            .service = required.service_static.id,
            .endpoint = required.name,
            .json_pointer = pointer,
            .required = required.name,
            .provided = "absent",
        });
      }
    }
    for (std::size_t candidate_index = 0U; candidate_index < candidate_used.size();
         ++candidate_index) {
      if (!candidate_used[candidate_index]) {
        const auto& provided = candidate.state_->endpoints[candidate_index];
        lookup_diagnostics.push_back(PendingDiagnostic{
            .token = "ENDPOINT_UNEXPECTED",
            .service = provided.service_static.id,
            .endpoint = provided.name,
            .json_pointer = "/endpoints/" + std::to_string(candidate_index),
            .required = "absent",
            .provided = provided.name,
        });
      }
    }
    flush_phase(lookup_diagnostics, nullptr);

    // Phase 3 begins with catalog-wide clock and exact affine-conversion facts.
    std::vector<PendingDiagnostic> service_diagnostics;
    std::vector<PendingDiagnostic> endpoint_diagnostics;
    for (std::size_t clock_index = 0U; clock_index < state_->clocks.size(); ++clock_index) {
      const auto& required_clock = state_->clocks[clock_index];
      const auto found = std::find_if(
          candidate.state_->clocks.begin(), candidate.state_->clocks.end(), [&](const auto& clock) {
            return clock.key == required_clock.key;
          });
      std::string required_json;
      std::string candidate_json;
      append_clock(required_json, required_clock);
      if (found == candidate.state_->clocks.end()) {
        service_diagnostics.push_back(PendingDiagnostic{
            .token = "ENDPOINT_CLOCK_DOMAIN_MISMATCH",
            .json_pointer = "/clocks/" + std::to_string(clock_index),
            .required = required_json,
            .provided = "absent",
        });
      } else {
        append_clock(candidate_json, *found);
        if (required_json != candidate_json) {
          service_diagnostics.push_back(PendingDiagnostic{
              .token = "ENDPOINT_CLOCK_DOMAIN_MISMATCH",
              .json_pointer = "/clocks/" + std::to_string(clock_index),
              .required = required_json,
              .provided = candidate_json,
          });
        }
      }
    }
    for (std::size_t conversion_index = 0U; conversion_index < state_->clock_conversions.size();
         ++conversion_index) {
      const auto& required_conversion = state_->clock_conversions[conversion_index];
      const auto found = std::find_if(candidate.state_->clock_conversions.begin(),
                                      candidate.state_->clock_conversions.end(),
                                      [&](const auto& conversion) {
                                        return conversion.from == required_conversion.from &&
                                               conversion.to == required_conversion.to;
                                      });
      std::string required_json;
      std::string candidate_json;
      append_conversion(required_json, required_conversion);
      if (found == candidate.state_->clock_conversions.end()) {
        service_diagnostics.push_back(PendingDiagnostic{
            .token = "ENDPOINT_CLOCK_CONVERSION_MISMATCH",
            .json_pointer = "/clock_conversions/" + std::to_string(conversion_index),
            .required = required_json,
            .provided = "absent",
        });
      } else {
        append_conversion(candidate_json, *found);
        if (required_json != candidate_json) {
          service_diagnostics.push_back(PendingDiagnostic{
              .token = "ENDPOINT_CLOCK_CONVERSION_MISMATCH",
              .json_pointer = "/clock_conversions/" + std::to_string(conversion_index),
              .required = required_json,
              .provided = candidate_json,
          });
        }
      }
    }

    for (const std::size_t endpoint_index : required_order) {
      if (!matches[endpoint_index]) {
        continue;
      }
      const auto& required = state_->endpoints[endpoint_index];
      const auto& provided = candidate.state_->endpoints[*matches[endpoint_index]];
      const auto append_service_error = [&](std::string token,
                                            std::string json_pointer,
                                            std::string required_value,
                                            std::string provided_value) {
        service_diagnostics.push_back(PendingDiagnostic{
            .token = std::move(token),
            .service = required.service_static.id,
            .endpoint = required.name,
            .json_pointer = std::move(json_pointer),
            .required = std::move(required_value),
            .provided = std::move(provided_value),
            .service_cause_endpoint = endpoint_index,
        });
      };
      const std::string service_pointer = "/services/" + std::to_string(endpoint_index);
      if (provided.service_static.id != required.service_static.id) {
        append_service_error("ENDPOINT_SERVICE_IDENTITY_MISMATCH",
                             service_pointer + "/id",
                             required.service_static.id,
                             provided.service_static.id);
      }
      if (provided.service_static.queue_policy != required.service_static.queue_policy) {
        append_service_error("ENDPOINT_SERVICE_QUEUE_POLICY_MISMATCH",
                             service_pointer + "/queue/policy",
                             std::string{queue_policy_name(required.service_static.queue_policy)},
                             std::string{queue_policy_name(provided.service_static.queue_policy)});
      }
      std::string required_payload;
      std::string candidate_payload;
      append_payload(required_payload, required.service_static);
      append_payload(candidate_payload, provided.service_static);
      if (required_payload != candidate_payload) {
        append_service_error(
            required.service_static.payload_identity.kind == PayloadIdentityKind::kSchema
                ? "ENDPOINT_SCHEMA_MISMATCH"
            : required.service_static.payload_identity.kind == PayloadIdentityKind::kTensor
                ? "ENDPOINT_TENSOR_POLICY_MISMATCH"
                : "ENDPOINT_TYPE_MISMATCH",
            service_pointer + "/payload",
            required_payload,
            candidate_payload);
      }
      if (provided.service_static.max_subscriber_depth <
          required.service_static.max_subscriber_depth) {
        append_service_error("ENDPOINT_SERVICE_MAX_QUEUE_INSUFFICIENT",
                             service_pointer + "/queue/max_subscriber_depth",
                             std::to_string(required.service_static.max_subscriber_depth),
                             std::to_string(provided.service_static.max_subscriber_depth));
      }
      if (provided.service_static.max_subscriber_borrowed <
          required.service_static.max_subscriber_borrowed) {
        append_service_error("ENDPOINT_SERVICE_MAX_BORROW_INSUFFICIENT",
                             service_pointer + "/queue/max_subscriber_borrowed",
                             std::to_string(required.service_static.max_subscriber_borrowed),
                             std::to_string(provided.service_static.max_subscriber_borrowed));
      }
      if (provided.service_static.publisher_history_depth !=
          required.service_static.publisher_history_depth) {
        append_service_error("ENDPOINT_SERVICE_HISTORY_MISMATCH",
                             service_pointer + "/queue/publisher_history_depth",
                             std::to_string(required.service_static.publisher_history_depth),
                             std::to_string(provided.service_static.publisher_history_depth));
      }
      if (provided.service_static.event_wake != required.service_static.event_wake) {
        append_service_error("ENDPOINT_SERVICE_READINESS_MISMATCH",
                             service_pointer + "/event_wake",
                             required.service_static.event_wake ? "true" : "false",
                             provided.service_static.event_wake ? "true" : "false");
      }

      // Phase 4: endpoint-local requests never participate in the opaque
      // symmetric service binding.
      const auto append_endpoint_error = [&](std::string token,
                                             std::string json_pointer,
                                             std::string required_value,
                                             std::string provided_value) {
        endpoint_diagnostics.push_back(PendingDiagnostic{
            .token = std::move(token),
            .service = required.service_static.id,
            .endpoint = required.name,
            .json_pointer = std::move(json_pointer),
            .required = std::move(required_value),
            .provided = std::move(provided_value),
        });
      };
      const std::string endpoint_pointer = "/endpoints/" + std::to_string(endpoint_index);
      if (provided.uri != required.uri) {
        append_endpoint_error(
            "ENDPOINT_URI_MISMATCH", endpoint_pointer + "/uri", required.uri, provided.uri);
      }
      if (provided.direction != required.direction) {
        append_endpoint_error("ENDPOINT_DIRECTION_MISMATCH",
                              endpoint_pointer + "/direction",
                              required.direction == TopicDirection::kImport ? "import" : "export",
                              provided.direction == TopicDirection::kImport ? "import" : "export");
      }
      const auto publisher_loan = [](const ChannelEndpointRecord& endpoint) {
        return endpoint.publisher_request && endpoint.publisher_request->max_loaned_samples
                   ? *endpoint.publisher_request->max_loaned_samples
                   : 1U;
      };
      if (publisher_loan(provided) < publisher_loan(required)) {
        append_endpoint_error("ENDPOINT_PUBLISHER_LOAN_LIMIT_MISMATCH",
                              endpoint_pointer + "/publisher/max_loaned_samples",
                              std::to_string(publisher_loan(required)),
                              std::to_string(publisher_loan(provided)));
      }
      if (required.subscriber_request && provided.subscriber_request &&
          provided.subscriber_request->queue_depth < required.subscriber_request->queue_depth) {
        append_endpoint_error("ENDPOINT_SUBSCRIBER_QUEUE_REQUEST_EXCEEDS_SERVICE",
                              endpoint_pointer + "/subscriber/queue_depth",
                              std::to_string(required.subscriber_request->queue_depth),
                              std::to_string(provided.subscriber_request->queue_depth));
      }
      if (required.subscriber_request && provided.subscriber_request &&
          provided.subscriber_request->max_borrowed_samples <
              required.subscriber_request->max_borrowed_samples) {
        append_endpoint_error("ENDPOINT_SUBSCRIBER_BORROW_REQUEST_EXCEEDS_SERVICE",
                              endpoint_pointer + "/subscriber/max_borrowed_samples",
                              std::to_string(required.subscriber_request->max_borrowed_samples),
                              std::to_string(provided.subscriber_request->max_borrowed_samples));
      }
      const auto capture_clock =
          [](const ChannelEndpointRecord& endpoint) -> const std::optional<ClockKey>& {
        return endpoint.capture_time_clock;
      };
      if (capture_clock(required) != capture_clock(provided)) {
        append_endpoint_error(
            "ENDPOINT_CLOCK_DOMAIN_MISMATCH",
            endpoint_pointer + "/capture_time/clock",
            capture_clock(required) ? std::string(capture_clock(required)->name()) : "null",
            capture_clock(provided) ? std::string(capture_clock(provided)->name()) : "null");
      }
      const auto endpoint_health =
          [](const ChannelEndpointRecord& endpoint) -> const ClockHealthOptions& {
        static const ClockHealthOptions kOrdinary;
        return endpoint.publisher_request ? endpoint.publisher_request->clock_health : kOrdinary;
      };
      if (endpoint_health(required).capability != endpoint_health(provided).capability) {
        append_endpoint_error(
            "ENDPOINT_CLOCK_HEALTH_CAPABILITY_MISMATCH",
            endpoint_pointer + "/clock_health/capability",
            endpoint_health(required).capability == ClockHealthCapability::kStatusReports
                ? "status_reports"
                : "unavailable",
            endpoint_health(provided).capability == ClockHealthCapability::kStatusReports
                ? "status_reports"
                : "unavailable");
      }
      if (endpoint_health(required).requirement != endpoint_health(provided).requirement ||
          endpoint_health(required).max_report_age != endpoint_health(provided).max_report_age) {
        const auto health_summary = [](const ClockHealthOptions& health) {
          std::string summary = health.requirement == ClockHealthRequirement::kFreshHealthyReport
                                    ? "fresh_healthy_report"
                                    : "not_required";
          summary += ":";
          summary +=
              health.max_report_age ? std::to_string(health.max_report_age->count()) : "null";
          return summary;
        };
        append_endpoint_error("ENDPOINT_CLOCK_HEALTH_REQUIREMENT_UNSATISFIED",
                              endpoint_pointer + "/clock_health/requirement",
                              health_summary(endpoint_health(required)),
                              health_summary(endpoint_health(provided)));
      }
      if (required.memory_conversion != provided.memory_conversion) {
        append_endpoint_error("ENDPOINT_TENSOR_POLICY_MISMATCH",
                              endpoint_pointer + "/memory_conversion",
                              std::string(memory_conversion_token(required.memory_conversion)),
                              std::string(memory_conversion_token(provided.memory_conversion)));
      }
      if (required.attach.peer != provided.attach.peer) {
        append_endpoint_error(
            "ENDPOINT_PEER_REQUIREMENT_MISMATCH",
            endpoint_pointer + "/attach/peer",
            required.attach.peer == PeerRequirement::kRequired ? "required" : "optional",
            provided.attach.peer == PeerRequirement::kRequired ? "required" : "optional");
      }
      const std::int64_t required_wait = required.attach.max_wait->count();
      const std::int64_t candidate_wait = provided.attach.max_wait->count();
      if (candidate_wait > required_wait) {
        append_endpoint_error("ENDPOINT_ATTACH_BOUND_EXCEEDED",
                              endpoint_pointer + "/attach/max_wait_ns",
                              std::to_string(required_wait),
                              std::to_string(candidate_wait));
      }
    }
    flush_phase(service_diagnostics, &first_service_cause);
    flush_phase(endpoint_diagnostics, nullptr);

    // Phase 5: derived bindings are consequences of service-static facts only.
    std::vector<PendingDiagnostic> binding_diagnostics;
    for (const std::size_t endpoint_index : required_order) {
      if (!matches[endpoint_index]) {
        continue;
      }
      const auto& required = state_->endpoints[endpoint_index];
      const auto& provided = candidate.state_->endpoints[*matches[endpoint_index]];
      if (service_binding(required) != service_binding(provided)) {
        binding_diagnostics.push_back(PendingDiagnostic{
            .token = "ENDPOINT_BINDING_MISMATCH",
            .service = required.service_static.id,
            .endpoint = required.name,
            .json_pointer = "/services/" + std::to_string(endpoint_index) + "/binding",
            .required = service_binding(required),
            .provided = service_binding(provided),
            .service_cause_endpoint = endpoint_index,
        });
      }
    }
    std::sort(binding_diagnostics.begin(),
              binding_diagnostics.end(),
              [](const auto& left, const auto& right) {
                return std::tie(left.service,
                                left.endpoint,
                                left.json_pointer,
                                left.token,
                                left.required,
                                left.provided) < std::tie(right.service,
                                                          right.endpoint,
                                                          right.json_pointer,
                                                          right.token,
                                                          right.required,
                                                          right.provided);
              });
    for (auto& diagnostic : binding_diagnostics) {
      const auto cause_id = diagnostic.service_cause_endpoint
                                ? first_service_cause[*diagnostic.service_cause_endpoint]
                                : std::nullopt;
      append_error(std::move(diagnostic.token),
                   std::move(diagnostic.service),
                   std::move(diagnostic.endpoint),
                   std::move(diagnostic.json_pointer),
                   std::move(diagnostic.required),
                   std::move(diagnostic.provided),
                   cause_id);
    }

    // Phase 6: source/name-origin provenance is observable but nonsemantic.
    std::vector<PendingDiagnostic> provenance_diagnostics;
    for (const std::size_t endpoint_index : required_order) {
      if (!matches[endpoint_index]) {
        continue;
      }
      const auto& required = state_->endpoints[endpoint_index];
      const auto& provided = candidate.state_->endpoints[*matches[endpoint_index]];
      const bool generated_provenance_equal =
          required.generated_name.has_value() == provided.generated_name.has_value() &&
          (!required.generated_name ||
           (required.generated_name->algorithm == provided.generated_name->algorithm &&
            required.generated_name->version == provided.generated_name->version));
      if (required.name_origin != provided.name_origin || !generated_provenance_equal) {
        provenance_diagnostics.push_back(PendingDiagnostic{
            .severity = DiagnosticSeverity::kWarning,
            .token = "ENDPOINT_NAME_PROVENANCE_CHANGED",
            .service = required.service_static.id,
            .endpoint = required.name,
            .json_pointer = "/endpoints/" + std::to_string(endpoint_index) + "/name_origin",
            .required =
                required.name_origin == EndpointNameOrigin::kGenerated ? "generated" : "explicit",
            .provided =
                provided.name_origin == EndpointNameOrigin::kGenerated ? "generated" : "explicit",
        });
      }
    }
    if (state_->digest == candidate.state_->digest &&
        (state_->raw_source_sha256 != candidate.state_->raw_source_sha256 ||
         state_->raw_source_size_bytes != candidate.state_->raw_source_size_bytes)) {
      provenance_diagnostics.push_back(PendingDiagnostic{
          .severity = DiagnosticSeverity::kWarning,
          .token = "ENDPOINT_SOURCE_PROVENANCE_CHANGED",
          .required =
              state_->raw_source_sha256.value_or("absent") + ":" +
              (state_->raw_source_size_bytes ? std::to_string(*state_->raw_source_size_bytes)
                                             : "absent"),
          .provided = candidate.state_->raw_source_sha256.value_or("absent") + ":" +
                      (candidate.state_->raw_source_size_bytes
                           ? std::to_string(*candidate.state_->raw_source_size_bytes)
                           : "absent"),
      });
    }
    flush_phase(provenance_diagnostics, nullptr);
  }
  report->json = "{\"compatible\":";
  report->json += report->compatible ? "true" : "false";
  report->json += ",\"diagnostics\":[";
  for (std::size_t index = 0U; index < report->diagnostics.size(); ++index) {
    if (index != 0U) {
      report->json.push_back(',');
    }
    const auto& diagnostic = report->diagnostics[index];
    report->json += "{\"id\":" + std::to_string(diagnostic.id.value) + ",\"code\":";
    append_json_string(report->json, diagnostic.token);
    report->json += diagnostic.severity == DiagnosticSeverity::kError ? ",\"severity\":\"error\""
                                                                      : ",\"severity\":\"warning\"";
    const auto append_optional_string = [&](std::string_view field, std::string_view value) {
      if (!value.empty()) {
        report->json += ",\"";
        report->json += field;
        report->json += "\":";
        append_json_string(report->json, value);
      }
    };
    append_optional_string("service", diagnostic.service);
    append_optional_string("endpoint", diagnostic.endpoint);
    if (!diagnostic.json_pointer.empty() || diagnostic.token == "ENDPOINT_CATALOG_EMPTY") {
      report->json += ",\"json_pointer\":";
      append_json_string(report->json, diagnostic.json_pointer);
    }
    append_optional_string("required", diagnostic.required);
    append_optional_string("provided", diagnostic.provided);
    if (diagnostic.cause_id) {
      report->json += ",\"cause_id\":" + std::to_string(diagnostic.cause_id->value);
    }
    report->json.push_back('}');
  }
  report->json += "]}";
  return detail::ChannelEndpointCompatibilityReportAccess::make(std::move(report));
}

bool ChannelEndpointCompatibilityReport::compatible() const noexcept {
  return state_ == nullptr || state_->compatible;
}

std::span<const Diagnostic> ChannelEndpointCompatibilityReport::diagnostics() const noexcept {
  return state_ == nullptr ? std::span<const Diagnostic>{}
                           : std::span<const Diagnostic>{state_->diagnostics};
}

std::string ChannelEndpointCompatibilityReport::json() const {
  return state_ == nullptr ? std::string{"{\"compatible\":true,\"diagnostics\":[]}"} : state_->json;
}

expected<ChannelEndpointCatalog, Error> detail::ChannelEndpointCatalogAccess::normalize(
    std::string scope, std::vector<detail::ChannelEndpointRecord> endpoints,
    std::vector<detail::CatalogClockRecord> clocks,
    std::vector<detail::CatalogClockConversionRecord> clock_conversions,
    bool infer_standalone_clocks) {
  return normalize_catalog(std::move(scope),
                           std::move(endpoints),
                           std::move(clocks),
                           std::move(clock_conversions),
                           std::nullopt,
                           std::nullopt,
                           infer_standalone_clocks);
}

std::size_t TypeKeyHash::operator()(const TypeKey& key) const noexcept {
  std::size_t result = sizeof(std::size_t) == 8U ? static_cast<std::size_t>(14695981039346656037ULL)
                                                 : static_cast<std::size_t>(2166136261U);
  const std::size_t prime = sizeof(std::size_t) == 8U ? static_cast<std::size_t>(1099511628211ULL)
                                                      : static_cast<std::size_t>(16777619U);
  for (const std::uint8_t byte : key.bytes) {
    result ^= static_cast<std::size_t>(byte);
    result *= prime;
  }
  return result;
}

std::string_view PodFieldLayoutView::name() const noexcept {
  return record_ == nullptr ? std::string_view{} : std::string_view{record_->name};
}

std::uint32_t PodFieldLayoutView::byte_offset() const noexcept {
  return record_ == nullptr ? 0U : record_->byte_offset;
}

std::uint32_t PodFieldLayoutView::element_count() const noexcept {
  return record_ == nullptr ? 0U : record_->element_count;
}

PodScalarKind PodFieldLayoutView::scalar_kind() const noexcept {
  return record_ == nullptr ? PodScalarKind::kUInt8 : record_->scalar_kind;
}

std::string_view PodTypeIdentityView::name() const noexcept {
  return record_ == nullptr ? std::string_view{}
                            : std::string_view{record_->identity.canonical_name};
}

std::uint32_t PodTypeIdentityView::version() const noexcept {
  return record_ == nullptr ? 0U : record_->identity.version;
}

const std::array<std::uint8_t, 32>& PodTypeIdentityView::digest_bytes() const noexcept {
  static constexpr std::array<std::uint8_t, 32> kEmpty{};
  return record_ == nullptr ? kEmpty : record_->digest_bytes;
}

TypeKey PodTypeIdentityView::key() const noexcept {
  return record_ == nullptr ? TypeKey{} : record_->key;
}

std::uint32_t PodTypeIdentityView::object_size() const noexcept {
  return record_ == nullptr ? 0U : record_->identity.object_size;
}

std::uint32_t PodTypeIdentityView::object_alignment() const noexcept {
  return record_ == nullptr ? 0U : record_->identity.object_alignment;
}

std::uint32_t PodTypeIdentityView::layout_format_version() const noexcept {
  return record_ == nullptr ? 0U : 1U;
}

const std::array<std::uint8_t, 32>& PodTypeIdentityView::layout_digest_bytes() const noexcept {
  static constexpr std::array<std::uint8_t, 32> kEmpty{};
  return record_ == nullptr ? kEmpty : record_->layout_digest_bytes;
}

std::span<const PodFieldLayoutView> PodTypeIdentityView::fields() const noexcept {
  return record_ == nullptr ? std::span<const PodFieldLayoutView>{}
                            : std::span<const PodFieldLayoutView>{record_->field_views};
}

std::string_view PodTypeIdentityView::integer_wire_endianness() const noexcept {
  return record_ == nullptr ? std::string_view{} : std::string_view{"little"};
}

std::string_view PodTypeIdentityView::floating_wire_format() const noexcept {
  return record_ == nullptr ? std::string_view{} : std::string_view{"ieee754"};
}

std::span<const CapabilityEvidenceView> PodTypeIdentityView::compatibility_evidence()
    const noexcept {
  return record_ == nullptr
             ? std::span<const CapabilityEvidenceView>{}
             : std::span<const CapabilityEvidenceView>{record_->compatibility_evidence};
}

std::string_view SchemaIdentityView::canonical_name() const noexcept {
  return record_ == nullptr ? std::string_view{} : std::string_view{record_->canonical_name};
}

std::uint32_t SchemaIdentityView::compatibility_epoch() const noexcept {
  return record_ == nullptr ? 0U : record_->compatibility_epoch;
}

SchemaId SchemaIdentityView::family_id() const noexcept {
  return record_ == nullptr ? SchemaId{} : record_->family_id;
}

FlatBufferFileIdentifier SchemaIdentityView::file_identifier() const noexcept {
  return record_ == nullptr ? FlatBufferFileIdentifier{} : record_->file_identifier;
}

const ClockKey& RequireCaptureTimeView::clock() const noexcept {
  static const ClockKey kDefault{};
  return clock_ == nullptr ? kDefault : *clock_;
}

ClockHealthCapability ClockHealthContractView::capability() const noexcept {
  return options_ == nullptr ? ClockHealthCapability::kUnavailable : options_->capability;
}

ClockHealthRequirement ClockHealthContractView::requirement() const noexcept {
  return options_ == nullptr ? ClockHealthRequirement::kNotRequired : options_->requirement;
}

std::optional<std::chrono::nanoseconds> ClockHealthContractView::max_report_age() const noexcept {
  return options_ == nullptr ? std::nullopt : options_->max_report_age;
}

std::optional<MemoryKind> TensorBoundaryContractView::memory_kind() const noexcept {
  return record_ == nullptr ? std::nullopt : record_->memory_kind;
}

std::optional<DLDataType> TensorBoundaryContractView::dtype() const noexcept {
  return record_ == nullptr ? std::nullopt : record_->dtype;
}

std::optional<std::uint32_t> TensorBoundaryContractView::rank() const noexcept {
  return record_ == nullptr ? std::nullopt : record_->rank;
}

std::optional<std::uint32_t> TensorBoundaryContractView::max_rank() const noexcept {
  return record_ == nullptr ? std::nullopt : record_->max_rank;
}

std::optional<std::uint64_t> TensorBoundaryContractView::max_byte_span() const noexcept {
  return record_ == nullptr ? std::nullopt : record_->max_byte_span;
}

std::span<const TensorAxisBounds> TensorBoundaryContractView::axes() const noexcept {
  return record_ == nullptr ? std::span<const TensorAxisBounds>{}
                            : std::span<const TensorAxisBounds>{record_->axes};
}

TensorStridePolicy TensorBoundaryContractView::stride_policy() const noexcept {
  return record_ == nullptr ? TensorStridePolicy::kContiguousOnly : record_->strides;
}

InterfaceTensorEmitPolicy TensorBoundaryContractView::foreign_emit_policy() const noexcept {
  return record_ == nullptr ? InterfaceTensorEmitPolicy::kRequireExportable
                            : record_->foreign_emit_policy;
}

std::optional<std::uint32_t> TensorBoundaryContractView::max_inline_bytes() const noexcept {
  return record_ == nullptr ? std::nullopt : record_->max_inline_bytes;
}

std::optional<std::uint64_t> TensorBoundaryContractView::max_staging_byte_span() const noexcept {
  return record_ == nullptr ? std::nullopt : record_->max_staging_byte_span;
}

std::string detail::ChannelEndpointCatalogAccess::backend_service_binding(
    ChannelEndpointCatalog::EndpointView endpoint) {
  const ChannelEndpointRecord* endpoint_record = record(endpoint);
  return endpoint_record == nullptr ? std::string{} : service_binding(*endpoint_record);
}

ClockId detail::ChannelEndpointCatalogAccess::capture_clock_id(
    ChannelEndpointCatalog::EndpointView endpoint) noexcept {
  const ChannelEndpointRecord* endpoint_record = record(endpoint);
  if (endpoint_record == nullptr || !endpoint_record->capture_time_clock) {
    return {};
  }
  const auto found = std::find_if(endpoint.state_->clocks.begin(),
                                  endpoint.state_->clocks.end(),
                                  [&](const CatalogClockRecord& clock) {
                                    return clock.key == *endpoint_record->capture_time_clock;
                                  });
  return found == endpoint.state_->clocks.end() ? ClockId{} : ClockId{found->id};
}

std::string_view ChannelEndpointCatalog::scope() const noexcept {
  return state_ == nullptr ? std::string_view{} : std::string_view{state_->scope};
}

std::string_view ChannelEndpointCatalog::EndpointView::name() const noexcept {
  return *this ? std::string_view{state_->endpoints[index_].name} : std::string_view{};
}

std::string_view ChannelEndpointCatalog::EndpointView::uri() const noexcept {
  return *this ? std::string_view{state_->endpoints[index_].uri} : std::string_view{};
}

std::string_view ChannelEndpointCatalog::EndpointView::service() const noexcept {
  return *this ? std::string_view{state_->endpoints[index_].service} : std::string_view{};
}

TopicDirection ChannelEndpointCatalog::EndpointView::direction() const noexcept {
  return *this ? state_->endpoints[index_].direction : TopicDirection::kImport;
}

EndpointNameOrigin ChannelEndpointCatalog::EndpointView::name_origin() const noexcept {
  return *this ? state_->endpoints[index_].name_origin : EndpointNameOrigin::kExplicit;
}

GeneratedNameInfoView ChannelEndpointCatalog::EndpointView::generated_name() const noexcept {
  if (!*this) {
    return std::nullopt;
  }
  const auto& generated = state_->endpoints[index_].generated_name;
  if (!generated.has_value()) {
    return std::nullopt;
  }
  return GeneratedEndpointNameInfo{
      .algorithm = generated->algorithm,
      .version = generated->version,
  };
}

std::optional<PodTypeIdentityView> ChannelEndpointCatalog::EndpointView::pod_type_identity()
    const noexcept {
  if (!*this) {
    return std::nullopt;
  }
  const auto& pod = state_->endpoints[index_].service_static.pod_identity;
  if (!pod.has_value()) {
    return std::nullopt;
  }
  return detail::ChannelEndpointCatalogAccess::pod_view(&pod.value());
}

std::optional<SchemaIdentityView> ChannelEndpointCatalog::EndpointView::schema_identity()
    const noexcept {
  if (!*this) {
    return std::nullopt;
  }
  const auto& schema = state_->endpoints[index_].service_static.schema_identity;
  if (!schema.has_value()) {
    return std::nullopt;
  }
  return detail::ChannelEndpointCatalogAccess::schema_view(&schema.value());
}

std::optional<std::uint64_t>
ChannelEndpointCatalog::EndpointView::schema_writer_max_serialized_bytes() const noexcept {
  if (!*this) {
    return std::nullopt;
  }
  const auto& service = state_->endpoints[index_].service_static;
  if (service.payload_identity.kind != PayloadIdentityKind::kSchema ||
      service.schema_writer_max_serialized_bytes == 0U) {
    return std::nullopt;
  }
  return service.schema_writer_max_serialized_bytes;
}

std::optional<std::uint64_t> ChannelEndpointCatalog::EndpointView::schema_required_frame_bytes()
    const noexcept {
  if (!*this) {
    return std::nullopt;
  }
  const auto& service = state_->endpoints[index_].service_static;
  return service.payload_identity.kind == PayloadIdentityKind::kSchema &&
                 service.required_data_frame_bytes != 0U
             ? std::optional{service.required_data_frame_bytes}
             : std::nullopt;
}

std::optional<std::uint64_t>
ChannelEndpointCatalog::EndpointView::schema_selected_inline_bucket_bytes() const noexcept {
  if (!*this) {
    return std::nullopt;
  }
  const auto& service = state_->endpoints[index_].service_static;
  return service.payload_identity.kind == PayloadIdentityKind::kSchema &&
                 service.selected_inline_bucket_bytes != 0U
             ? std::optional{service.selected_inline_bucket_bytes}
             : std::nullopt;
}

std::optional<std::uint64_t>
ChannelEndpointCatalog::EndpointView::schema_bucket_usable_serialized_bytes() const noexcept {
  if (!*this) {
    return std::nullopt;
  }
  const auto& service = state_->endpoints[index_].service_static;
  return service.payload_identity.kind == PayloadIdentityKind::kSchema &&
                 service.bucket_usable_serialized_bytes != 0U
             ? std::optional{service.bucket_usable_serialized_bytes}
             : std::nullopt;
}

std::optional<std::uint32_t> ChannelEndpointCatalog::EndpointView::schema_transport_profile_tag()
    const noexcept {
  const auto& payload = state_->endpoints[index_].service_static.payload_identity;
  if (payload.kind != PayloadIdentityKind::kSchema) {
    return std::nullopt;
  }
  return payload.transport_profile_tag;
}

std::optional<RequireCaptureTimeView> ChannelEndpointCatalog::EndpointView::capture_time()
    const noexcept {
  if (!*this) {
    return std::nullopt;
  }
  const auto& capture_time = state_->endpoints[index_].capture_time_clock;
  if (!capture_time.has_value()) {
    return std::nullopt;
  }
  return detail::ChannelEndpointCatalogAccess::capture_time_view(&capture_time.value());
}

ClockHealthContractView ChannelEndpointCatalog::EndpointView::clock_health() const noexcept {
  if (!*this) {
    return detail::ChannelEndpointCatalogAccess::clock_health_view(nullptr);
  }
  const auto& publisher = state_->endpoints[index_].publisher_request;
  return detail::ChannelEndpointCatalogAccess::clock_health_view(
      publisher.has_value() ? &publisher->clock_health : nullptr);
}

MemoryConversionPolicy ChannelEndpointCatalog::EndpointView::memory_conversion() const noexcept {
  return *this ? state_->endpoints[index_].memory_conversion.value_or(
                     MemoryConversionPolicy::kDefault)
               : MemoryConversionPolicy::kDefault;
}

TopicAttachPolicy ChannelEndpointCatalog::EndpointView::attach_policy() const noexcept {
  return *this ? state_->endpoints[index_].attach : TopicAttachPolicy{};
}

EndpointBounds ChannelEndpointCatalog::EndpointView::bounds() const noexcept {
  if (!*this) {
    return {};
  }
  EndpointBounds result;
  const auto& endpoint = state_->endpoints[index_];
  if (endpoint.publisher_request && endpoint.publisher_request->max_loaned_samples) {
    result.publisher = PublisherEndpointBounds{.max_loaned_samples =
                                                   endpoint.publisher_request->max_loaned_samples};
  }
  if (endpoint.subscriber_request) {
    result.subscriber = SubscriberEndpointBounds{
        .queue_depth = endpoint.subscriber_request->queue_depth,
        .max_borrowed_samples = endpoint.subscriber_request->max_borrowed_samples,
    };
  }
  return result;
}

std::optional<TensorBoundaryContractView>
ChannelEndpointCatalog::EndpointView::tensor_requirements() const noexcept {
  if (!*this) {
    return std::nullopt;
  }
  const auto& tensor = state_->endpoints[index_].service_static.tensor;
  if (!tensor.has_value()) {
    return std::nullopt;
  }
  return detail::ChannelEndpointCatalogAccess::tensor_requirements_view(&tensor.value());
}

ChannelEndpointCatalog::EndpointView::operator bool() const noexcept {
  return state_ != nullptr && index_ < state_->endpoints.size();
}

}  // namespace holoscan
