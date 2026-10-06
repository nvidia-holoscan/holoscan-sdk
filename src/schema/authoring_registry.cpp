/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/authoring/register_payload.hpp>
#include <holoscan/schema/authoring_registry.hpp>
#include <holoscan/schema/sdk_schemas.hpp>

namespace holoscan::authoring {
namespace {

expected<const ForeignSchemaValue::Object*, Error> built_in_schema_fields(
    const ForeignSchemaValue& value, std::span<const std::string_view> known) {
  const auto* fields = value.get<ForeignSchemaValue::Object>();
  if (fields == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "PYTHON_SCHEMA_PAYLOAD_TYPE_MISMATCH"});
  }
  for (const auto& [name, unused] : *fields) {
    (void)unused;
    if (std::find(known.begin(), known.end(), name) == known.end()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "PYTHON_SCHEMA_FIELD_UNKNOWN"});
    }
  }
  return fields;
}

}  // namespace

template <>
struct PythonSchemaNativeTraits<schema::StringT> {
  static expected<ForeignSchemaValue, Error> to_foreign(const schema::StringT& value) {
    ForeignSchemaValue::Object fields;
    fields.emplace("data", ForeignSchemaValue::string(value.data));
    return ForeignSchemaValue::object(std::move(fields));
  }

  static expected<schema::StringT, Error> from_foreign(const ForeignSchemaValue& value) {
    constexpr std::array<std::string_view, 1U> kKnown{"data"};
    auto fields = built_in_schema_fields(value, kKnown);
    if (!fields) {
      return make_unexpected(std::move(fields).error());
    }
    schema::StringT result;
    if (const auto* field = value.field("data")) {
      auto converted = detail::foreign_schema_scalar<std::string>(*field);
      if (!converted) {
        return make_unexpected(std::move(converted).error());
      }
      result.data = std::move(*converted);
    }
    return result;
  }
};

template <>
struct PythonSchemaNativeTraits<schema::TimeT> {
  static expected<ForeignSchemaValue, Error> to_foreign(const schema::TimeT& value) {
    ForeignSchemaValue::Object fields;
    fields.emplace("time_ns", ForeignSchemaValue::signed_integer(value.time_ns));
    fields.emplace("acqtime_ns", ForeignSchemaValue::signed_integer(value.acqtime_ns));
    return ForeignSchemaValue::object(std::move(fields));
  }

  static expected<schema::TimeT, Error> from_foreign(const ForeignSchemaValue& value) {
    constexpr std::array<std::string_view, 2U> kKnown{"time_ns", "acqtime_ns"};
    auto fields = built_in_schema_fields(value, kKnown);
    if (!fields) {
      return make_unexpected(std::move(fields).error());
    }
    schema::TimeT result;
    if (const auto* field = value.field("time_ns")) {
      auto converted = detail::foreign_schema_scalar<std::int64_t>(*field);
      if (!converted) {
        return make_unexpected(std::move(converted).error());
      }
      result.time_ns = *converted;
    }
    if (const auto* field = value.field("acqtime_ns")) {
      auto converted = detail::foreign_schema_scalar<std::int64_t>(*field);
      if (!converted) {
        return make_unexpected(std::move(converted).error());
      }
      result.acqtime_ns = *converted;
    }
    return result;
  }
};

}  // namespace holoscan::authoring

namespace holoscan::schema {

bool register_sdk_schema_payloads() noexcept {
  static const bool registered = []() noexcept {
    constexpr SchemaIdentity header_identity = schema_identity<HeaderT>();
    constexpr SchemaIdentity string_identity = schema_identity<StringT>();
    constexpr SchemaIdentity tensor_packet_identity = schema_identity<TensorPacketT>();
    constexpr SchemaIdentity time_identity = schema_identity<TimeT>();
    const bool header_registered =
        authoring::detail::register_payload_automatically<HeaderT>(header_identity.canonical_name);
    const bool string_registered =
        authoring::detail::register_payload_automatically<StringT>(string_identity.canonical_name);
    const bool tensor_packet_registered =
        authoring::detail::register_payload_automatically<TensorPacketT>(
            tensor_packet_identity.canonical_name);
    const bool time_registered =
        authoring::detail::register_payload_automatically<TimeT>(time_identity.canonical_name);
    return header_registered && string_registered && tensor_packet_registered && time_registered;
  }();
  return registered;
}

namespace {

[[maybe_unused]] const bool kSdkSchemaPayloadsRegistered = register_sdk_schema_payloads();

}  // namespace
}  // namespace holoscan::schema
