/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/channel/endpoint_catalog_builder.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "endpoint_catalog_state.hpp"

namespace holoscan {
namespace {

[[nodiscard]] Error builder_error(ErrorCode code, std::string_view token) {
  return Error{.code = code, .message = std::string(token)};
}

[[nodiscard]] Error endpoint_limit_error(std::size_t actual) {
  std::string message{
      "ENDPOINT_CATALOG_RESOURCE_LIMIT_EXCEEDED|path=/endpoints|resource=normalized_endpoints|"
      "limit="};
  message += std::to_string(detail::kMaximumCatalogEndpoints);
  message += "|actual=";
  message += std::to_string(actual);
  return Error{.code = ErrorCode::kResourceExhausted, .message = std::move(message)};
}

[[nodiscard]] std::string escape_json_pointer_segment(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const char character : value) {
    if (character == '~') {
      result += "~0";
    } else if (character == '/') {
      result += "~1";
    } else {
      result.push_back(character);
    }
  }
  return result;
}

[[nodiscard]] std::string clock_health_capability_summary(ClockHealthCapability capability) {
  switch (capability) {
    case ClockHealthCapability::kUnavailable:
      return "unavailable";
    case ClockHealthCapability::kStatusReports:
      return "status_reports";
  }
  return "unknown(" + std::to_string(static_cast<std::uint32_t>(capability)) + ")";
}

[[nodiscard]] std::string clock_health_requirement_summary(ClockHealthRequirement requirement) {
  switch (requirement) {
    case ClockHealthRequirement::kNotRequired:
      return "not_required";
    case ClockHealthRequirement::kFreshHealthyReport:
      return "fresh_healthy_report";
  }
  return "unknown(" + std::to_string(static_cast<std::uint32_t>(requirement)) + ")";
}

[[nodiscard]] Error invalid_clock_health_error(std::string_view endpoint_name,
                                               const ClockHealthOptions& health,
                                               TopicDirection direction) {
  std::string message{"ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID|path=/builder/endpoints/"};
  message += escape_json_pointer_segment(endpoint_name);
  message +=
      "/clock_health|required=unavailable+not_required(age=absent,any_role)|status_"
      "reports+fresh_healthy_report(age=positive,external_publisher)|provided=capability=";
  message += clock_health_capability_summary(health.capability);
  message += ";requirement=";
  message += clock_health_requirement_summary(health.requirement);
  message += ";max_report_age_ns=";
  message += health.max_report_age ? std::to_string(health.max_report_age->count()) : "absent";
  message += ";role=";
  message += direction == TopicDirection::kImport ? "external_publisher" : "external_subscriber";
  return Error{.code = ErrorCode::kInvalidArgument, .message = std::move(message)};
}

template <typename Enum>
[[nodiscard]] constexpr bool closed_enum(Enum value, Enum last) noexcept {
  using Underlying = std::underlying_type_t<Enum>;
  static_assert(std::is_unsigned_v<Underlying>);
  return static_cast<Underlying>(value) <= static_cast<Underlying>(last);
}

[[nodiscard]] bool schema_equal(const SchemaIdentityRecord& left,
                                const SchemaIdentityRecord& right) noexcept {
  return left.canonical_name == right.canonical_name && left.version == right.version &&
         left.schema_sha256 == right.schema_sha256 && left.id == right.id;
}

[[nodiscard]] SchemaIdentityRecord copy_schema(const SchemaIdentity& schema) {
  return SchemaIdentityRecord{
      .canonical_name = std::string(schema.canonical_name),
      .version = schema.version,
      .schema_sha256 = schema.schema_sha256,
      .id = schema.id,
  };
}

[[nodiscard]] expected<std::optional<TensorBoundaryContractRecord>, Error> copy_tensor_requirements(
    const TopicOptions& options, const PayloadIdentityRecord& identity) {
  if (identity.kind != PayloadIdentityKind::kTensor) {
    if (options.tensor) {
      return make_unexpected(
          builder_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_OPTIONS_INVALID"));
    }
    return std::nullopt;
  }
  if (!options.tensor) {
    return std::nullopt;
  }
  TensorBoundaryContract boundary = *options.tensor;
  if (!boundary.memory_conversion) {
    boundary.memory_conversion = options.memory_conversion;
  } else if (options.memory_conversion && boundary.memory_conversion != options.memory_conversion) {
    return make_unexpected(
        builder_error(ErrorCode::kIncompatible, "GRAPH_TENSOR_MEMORY_POLICY_CONFLICT"));
  }
  auto normalized = detail::normalize_tensor_boundary_contract(boundary);
  if (!normalized) {
    return make_unexpected(std::move(normalized.error()));
  }
  const auto& tensor = *normalized;
  const auto& transfer = tensor.transfer.value_or(TensorTransferPolicy{});
  TensorBoundaryContractRecord result{
      .memory_kind = tensor.representation.memory_kind,
      .dtype = tensor.representation.dtype,
      .rank = tensor.representation.rank,
      .max_rank = tensor.max_rank,
      .max_byte_span = tensor.max_byte_span,
      .axes = tensor.axes,
      .strides = *tensor.representation.strides,
      .foreign_emit_policy = transfer.foreign_emit_policy,
      .max_inline_bytes =
          transfer.max_inline_bytes
              ? std::optional{static_cast<std::uint32_t>(*transfer.max_inline_bytes)}
              : std::nullopt,
      .max_staging_byte_span = transfer.max_staging_byte_span,
      .memory_conversion = tensor.memory_conversion,
      .origins = tensor.origins,
  };
  return std::optional<TensorBoundaryContractRecord>{std::move(result)};
}

[[nodiscard]] expected<void, Error> validate_clock_health(std::string_view endpoint_name,
                                                          const ClockHealthOptions& health,
                                                          TopicDirection direction) {
  if (!closed_enum(health.capability, ClockHealthCapability::kStatusReports) ||
      !closed_enum(health.requirement, ClockHealthRequirement::kFreshHealthyReport)) {
    return make_unexpected(invalid_clock_health_error(endpoint_name, health, direction));
  }
  const bool ordinary = health.capability == ClockHealthCapability::kUnavailable &&
                        health.requirement == ClockHealthRequirement::kNotRequired &&
                        !health.max_report_age;
  const bool monitored = direction == TopicDirection::kImport &&
                         health.capability == ClockHealthCapability::kStatusReports &&
                         health.requirement == ClockHealthRequirement::kFreshHealthyReport &&
                         health.max_report_age && health.max_report_age->count() > 0;
  if (!ordinary && !monitored) {
    return make_unexpected(invalid_clock_health_error(endpoint_name, health, direction));
  }
  return {};
}

// Keep the complete cross-field option validation in one atomic staging step.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> apply_options(detail::ChannelEndpointRecord& record,
                                                  const TopicOptions& options) {
  if (!closed_enum(options.attach.peer, PeerRequirement::kRequired) ||
      (options.attach.max_wait && options.attach.max_wait->count() <= 0) ||
      (options.memory_conversion &&
       !closed_enum(*options.memory_conversion, MemoryConversionPolicy::kRequire))) {
    return make_unexpected(builder_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_ENUM_INVALID"));
  }
  if (options.capture_time && !options.capture_time->clock.is_default() &&
      options.capture_time->clock.name().empty()) {
    return make_unexpected(
        builder_error(ErrorCode::kInvalidArgument, "GRAPH_CAPTURE_CLOCK_INVALID"));
  }
  if ((record.direction == TopicDirection::kImport && options.bounds.subscriber) ||
      (record.direction == TopicDirection::kExport && options.bounds.publisher) ||
      (options.bounds.publisher && options.bounds.subscriber)) {
    return make_unexpected(
        builder_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_DIRECTION_BOUNDS_INVALID"));
  }
  if (options.bounds.publisher && options.bounds.publisher->max_loaned_samples &&
      *options.bounds.publisher->max_loaned_samples == 0U) {
    return make_unexpected(
        builder_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_PUBLISHER_BOUND_INVALID"));
  }
  if (options.bounds.subscriber &&
      ((options.bounds.subscriber->queue_depth && *options.bounds.subscriber->queue_depth == 0U) ||
       (options.bounds.subscriber->max_borrowed_samples &&
        *options.bounds.subscriber->max_borrowed_samples == 0U))) {
    return make_unexpected(
        builder_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_SUBSCRIBER_BOUND_INVALID"));
  }
  if (options.memory_conversion && record.payload_identity.kind != PayloadIdentityKind::kTensor) {
    return make_unexpected(
        builder_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_MEMORY_CONVERSION_IRRELEVANT"));
  }

  std::optional<SchemaIdentityRecord> schema;
  if (options.schema) {
    schema = copy_schema(*options.schema);
  }
  if (record.payload_identity.kind == PayloadIdentityKind::kSchema) {
    const auto& payload_schema = record.payload_identity.schema;
    if (!payload_schema.has_value()) {
      return make_unexpected(
          builder_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_SCHEMA_INVALID"));
    }
    if (schema && !schema_equal(*schema, payload_schema.value())) {
      return make_unexpected(
          builder_error(ErrorCode::kIncompatible, "GRAPH_TOPIC_SCHEMA_MISMATCH"));
    }
    schema = payload_schema;
  } else if (record.payload_identity.kind != PayloadIdentityKind::kTensor && schema) {
    return make_unexpected(
        builder_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_SCHEMA_IRRELEVANT"));
  }

  auto tensor = copy_tensor_requirements(options, record.payload_identity);
  if (!tensor) {
    return make_unexpected(std::move(tensor).error());
  }
  const auto memory_conversion =
      options.memory_conversion
          ? options.memory_conversion
          : (options.tensor ? options.tensor->memory_conversion : std::nullopt);
  const ClockHealthOptions health = options.clock_health.value_or(ClockHealthOptions{});
  auto valid_health = validate_clock_health(record.name, health, record.direction);
  if (!valid_health) {
    return valid_health;
  }

  record.schema_identity = std::move(schema);
  record.memory_conversion = memory_conversion;
  record.capture_time_clock =
      options.capture_time ? std::optional<ClockKey>{options.capture_time->clock} : std::nullopt;
  record.clock_health = health;
  record.attach = options.attach;
  record.tensor = std::move(tensor).value();
  record.bounds = options.bounds;
  return {};
}

}  // namespace

struct ChannelEndpointCatalog::Builder::State {
  std::string deployment_scope;
  std::vector<detail::ChannelEndpointRecord> endpoints;
  std::map<std::string, std::size_t, std::less<>> endpoint_by_name;
  std::vector<detail::CatalogClockRecord> clocks;
  std::vector<detail::CatalogClockConversionRecord> clock_conversions;
};

ChannelEndpointCatalog::Builder::Builder(std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}

ChannelEndpointCatalog::Builder::~Builder() = default;
ChannelEndpointCatalog::Builder::Builder(Builder&& other) noexcept = default;
ChannelEndpointCatalog::Builder& ChannelEndpointCatalog::Builder::operator=(
    Builder&& other) noexcept = default;

expected<void, Error> ChannelEndpointCatalog::Builder::add_topic(
    TopicDirection direction, std::string_view endpoint_name, std::string_view uri,
    const TopicOptions& options, PayloadIdentityRecord payload_identity,
    const std::uint64_t serialized_payload_capacity_bytes) {
  if (state_ == nullptr) {
    return make_unexpected(builder_error(ErrorCode::kClosed, "ENDPOINT_CATALOG_BUILDER_CONSUMED"));
  }
  if (options.endpoint_name && *options.endpoint_name != endpoint_name) {
    return make_unexpected(builder_error(ErrorCode::kInvalidArgument, "ENDPOINT_NAME_MISMATCH"));
  }
  if (state_->endpoint_by_name.contains(endpoint_name)) {
    return make_unexpected(builder_error(ErrorCode::kDuplicateName, "ENDPOINT_NAME_DUPLICATE"));
  }
  if (state_->endpoints.size() >= detail::kMaximumCatalogEndpoints) {
    return make_unexpected(endpoint_limit_error(state_->endpoints.size() + 1U));
  }

  detail::ChannelEndpointRecord record{
      .name = std::string(endpoint_name),
      .uri = std::string(uri),
      .direction = direction,
      .name_origin = EndpointNameOrigin::kExplicit,
      .payload_identity = std::move(payload_identity),
      .serialized_payload_capacity_bytes = serialized_payload_capacity_bytes,
  };
  auto applied = apply_options(record, options);
  if (!applied) {
    return applied;
  }

  auto validated = detail::ChannelEndpointCatalogAccess::normalize(
      state_->deployment_scope, {record}, state_->clocks, state_->clock_conversions, true);
  if (!validated) {
    return make_unexpected(std::move(validated).error());
  }
  const std::size_t index = state_->endpoints.size();
  static_assert(std::is_nothrow_move_constructible_v<detail::ChannelEndpointRecord>);
  state_->endpoints.push_back(std::move(record));
  bool unique = false;
  try {
    unique = state_->endpoint_by_name.emplace(state_->endpoints.back().name, index).second;
  } catch (...) {
    state_->endpoints.pop_back();
    throw;
  }
  if (!unique) {
    state_->endpoints.pop_back();
    return make_unexpected(builder_error(ErrorCode::kDuplicateName, "ENDPOINT_NAME_DUPLICATE"));
  }
  return {};
}

expected<void, Error> ChannelEndpointCatalog::Builder::set_topic_options(
    std::string_view endpoint_name, const TopicOptions& options) {
  if (state_ == nullptr) {
    return make_unexpected(builder_error(ErrorCode::kClosed, "ENDPOINT_CATALOG_BUILDER_CONSUMED"));
  }
  if (options.endpoint_name && *options.endpoint_name != endpoint_name) {
    return make_unexpected(builder_error(ErrorCode::kInvalidArgument, "ENDPOINT_NAME_MISMATCH"));
  }
  const auto indexed = state_->endpoint_by_name.find(endpoint_name);
  if (indexed == state_->endpoint_by_name.end()) {
    return make_unexpected(builder_error(ErrorCode::kNotFound, "ENDPOINT_NOT_FOUND"));
  }
  auto found = state_->endpoints.begin() + static_cast<std::ptrdiff_t>(indexed->second);

  detail::ChannelEndpointRecord replacement = *found;
  auto applied = apply_options(replacement, options);
  if (!applied) {
    return applied;
  }
  auto validated = detail::ChannelEndpointCatalogAccess::normalize(
      state_->deployment_scope, {replacement}, state_->clocks, state_->clock_conversions, true);
  if (!validated) {
    return make_unexpected(std::move(validated).error());
  }
  *found = std::move(replacement);
  return {};
}

expected<ChannelEndpointCatalog, Error> ChannelEndpointCatalog::Builder::build() && {
  if (state_ == nullptr) {
    return make_unexpected(builder_error(ErrorCode::kClosed, "ENDPOINT_CATALOG_BUILDER_CONSUMED"));
  }
  std::unique_ptr<State> draft = std::move(state_);
  return detail::ChannelEndpointCatalogAccess::normalize(std::move(draft->deployment_scope),
                                                         std::move(draft->endpoints),
                                                         std::move(draft->clocks),
                                                         std::move(draft->clock_conversions),
                                                         true);
}

ChannelEndpointCatalog::Builder ChannelEndpointCatalog::builder(std::string_view deployment_scope) {
  auto state = std::make_unique<Builder::State>();
  state->deployment_scope = std::string(deployment_scope);
  return Builder{std::move(state)};
}

ChannelEndpointCatalog::Builder ChannelEndpointCatalog::edit() const {
  auto state = std::make_unique<Builder::State>();
  if (state_) {
    state->deployment_scope = state_->scope;
    state->endpoints = state_->endpoints;
    state->clocks = state_->clocks;
    state->clock_conversions = state_->clock_conversions;
    for (std::size_t index = 0U; index < state->endpoints.size(); ++index) {
      state->endpoint_by_name.emplace(state->endpoints[index].name, index);
    }
  }
  return Builder{std::move(state)};
}

}  // namespace holoscan
