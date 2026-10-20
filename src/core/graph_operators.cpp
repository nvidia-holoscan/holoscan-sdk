/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Graph Operator registration: port and lifecycle metadata validation, Operator lookup, and
// Operator parameters.

#include <holoscan/core/graph.hpp>
#include <holoscan/logger/logger.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <deque>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "diagnostic_context.hpp"
#include "domain/tensor_layout_validation.hpp"
#include "graph_internal.hpp"
#include "graph_mutation_rollback.hpp"
#include "graph_state.hpp"
#include "operator_spec_compiler_access.hpp"
#include "parameter_input_internal.hpp"
#include "plan_record_contracts.hpp"
#include "port_configuration_internal.hpp"
#include "tensor_field_configuration_internal.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] std::string_view graph_operator_port_type_mismatch_reason(
    const PortStorage& port, PortDirection direction, const PayloadIdentityRecord& payload_identity,
    const void* payload_local_type_token, const std::type_info* payload_local_type_info) {
  // Local enums retain their exact local-token requirement, including when RTTI is supplied.
  if (payload_local_type_token == nullptr &&
      ((payload_identity.scalar &&
        payload_identity.scalar->kind == ScalarIdentityKind::kLocalEnum) ||
       (port.record.payload_identity.scalar &&
        port.record.payload_identity.scalar->kind == ScalarIdentityKind::kLocalEnum))) {
    return "process-local enums require typed C++ local Graph authoring";
  }
  if (port.record.direction != direction) {
    return "port direction differs";
  }
  if (!payload_identity_exactly_equal(port.record.payload_identity, payload_identity)) {
    return "semantic payload identity differs";
  }
  // Tensor and admitted arithmetic scalars are closed SDK types. When exact native resolution
  // is requested, schema/POD types need a matching token or RTTI witness before a typed cast.
  // The legacy erased overload retains its semantic-only authoring behavior.
  const bool canonical_builtin =
      payload_identity.kind == PayloadIdentityKind::kTensor ||
      (payload_identity.kind == PayloadIdentityKind::kScalar && payload_identity.scalar &&
       payload_identity.scalar->kind != ScalarIdentityKind::kLocalEnum);
  const bool exact_requested =
      payload_local_type_token != nullptr || payload_local_type_info != nullptr;
  const bool same_token =
      payload_local_type_token != nullptr && port.local_type_token == payload_local_type_token;
  const bool rtti_eligible = payload_identity.kind == PayloadIdentityKind::kSchema ||
                             payload_identity.kind == PayloadIdentityKind::kPod;
  // Compare type_info values, never their addresses or diagnostic names. This follows the
  // existing Core parameter-type contract for ODR-conforming types across compatible DSOs.
  const bool same_rtti = !same_token && rtti_eligible && payload_local_type_info != nullptr &&
                         port.local_type_info != nullptr &&
                         *payload_local_type_info == *port.local_type_info;
  if (exact_requested && !canonical_builtin && !same_token && !same_rtti) {
    return "native payload type differs or exact native proof is unavailable";
  }
  return {};
}

[[nodiscard]] Error tensor_field_graph_error(ErrorCode code, std::string_view token,
                                             const TensorFieldConfigurationRecord& field) {
  std::string message(token);
  message.append(" (field_id=");
  message.append(std::to_string(field.field_id));
  message.append(", tensor_index=");
  message.append(std::to_string(field.tensor_index));
  message.push_back(')');
  return Error{code, std::move(message)};
}

[[nodiscard]] expected<void, Error> run_setup(OperatorBase& prototype, OperatorSpec& spec) {
  try {
    prototype.setup(spec);
  } catch (const RuntimeError& error) {
    graph_log_noexcept(
        [&] { HOLOSCAN_LOG_DEBUG("Operator setup RuntimeError detail: {}", error.what()); });
    return make_unexpected(graph_factory_error("GRAPH_OPERATOR_SETUP_RUNTIME_ERROR"));
  } catch (const std::exception& exception) {
    graph_log_noexcept(
        [&] { HOLOSCAN_LOG_DEBUG("Operator setup exception detail: {}", exception.what()); });
    return make_unexpected(graph_factory_error("GRAPH_OPERATOR_SETUP_FAILED"));
  } catch (...) {
    return make_unexpected(graph_factory_error("GRAPH_OPERATOR_SETUP_FAILED"));
  }
  if (OperatorSpecCompilerAccess::declaration_error(spec).code != ErrorCode::kSuccess) {
    return make_unexpected(OperatorSpecCompilerAccess::declaration_error(spec));
  }
  return {};
}

[[nodiscard]] bool valid_tensor_field_payload_options(
    const TensorFieldTransferRecord& options) noexcept {
  const bool inline_policy = options.interface_tensor_emit_policy ==
                             InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded;
  return closed_enum(options.interface_tensor_emit_policy) &&
         options.max_inline_bytes.value_or(1U) != 0U &&
         options.max_tensor_byte_span.value_or(1U) != 0U &&
         inline_policy == options.max_inline_bytes.has_value();
}

// Field selector, schema, allocation, and transfer checks jointly define one authored field.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> validate_one_tensor_field_metadata(
    const OperatorSpecCompilerAccess::PortMetadata& metadata,
    const AuthoredTensorFieldConfiguration& authored, std::vector<std::uint16_t>& field_ids,
    std::vector<std::uint16_t>& tensor_indices) {
  const auto& field = authored.configuration;
  if (authored.schema_type_token != metadata.local_type_token) {
    return make_unexpected(tensor_field_graph_error(
        ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_SCHEMA_MISMATCH", field));
  }
  if (!authored.selector_valid || field.tensor_index >= metadata.descriptor_tensor_field_count) {
    return make_unexpected(
        tensor_field_graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_UNKNOWN", field));
  }
  if (std::find(field_ids.begin(), field_ids.end(), field.field_id) != field_ids.end() ||
      std::find(tensor_indices.begin(), tensor_indices.end(), field.tensor_index) !=
          tensor_indices.end()) {
    return make_unexpected(
        tensor_field_graph_error(ErrorCode::kDuplicateName, "GRAPH_TENSOR_FIELD_DUPLICATE", field));
  }
  field_ids.push_back(field.field_id);
  tensor_indices.push_back(field.tensor_index);

  if ((field.memory_conversion && !closed_enum(*field.memory_conversion)) ||
      (field.tensor_layout && field.tensor_layout->memory_kind &&
       !closed_enum(*field.tensor_layout->memory_kind)) ||
      (field.payload_options && !valid_tensor_field_payload_options(*field.payload_options))) {
    return make_unexpected(tensor_field_graph_error(
        ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_ENUM_INVALID", field));
  }
  if (metadata.descriptor_tensor_field_count == 1U && field.payload_options &&
      field.payload_options->max_inline_bytes &&
      *field.payload_options->max_inline_bytes > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(tensor_field_graph_error(
        ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_ENUM_INVALID", field));
  }
  if (field.tensor_layout && field.tensor_layout->dtype &&
      !valid_tensor_dtype(*field.tensor_layout->dtype)) {
    return make_unexpected(tensor_field_graph_error(
        ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_DTYPE_INVALID", field));
  }
  if (field.tensor_allocation &&
      (field.tensor_allocation->max_rank == 0U || field.tensor_allocation->max_byte_span == 0U)) {
    if (field.tensor_allocation->max_rank == 0U && field.tensor_layout &&
        !field.tensor_layout->rank) {
      return make_unexpected(tensor_field_graph_error(
          ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_RANK_BOUND_UNRESOLVED", field));
    }
    return make_unexpected(tensor_field_graph_error(
        ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_ALLOCATION_INVALID", field));
  }
  if (field.tensor_byte_span_expression &&
      (metadata.direction != PortDirection::kOutput || !field.tensor_contract ||
       field.tensor_contract->storage != TensorOutputStorage::kRuntimePool ||
       field.tensor_contract->max_byte_span || field.tensor_allocation)) {
    return make_unexpected(tensor_field_graph_error(
        ErrorCode::kInvalidArgument, "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID", field));
  }
  if (field.tensor_memory_kind_expression &&
      (metadata.direction != PortDirection::kOutput || !field.tensor_contract ||
       field.tensor_contract->storage != TensorOutputStorage::kRuntimePool ||
       field.tensor_contract->representation.memory_kind ||
       (field.tensor_layout && field.tensor_layout->memory_kind))) {
    return make_unexpected(tensor_field_graph_error(
        ErrorCode::kInvalidArgument, "GRAPH_TENSOR_MEMORY_KIND_DERIVATION_INVALID", field));
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_tensor_field_metadata(
    const OperatorSpecCompilerAccess::PortMetadata& metadata) {
  if (metadata.descriptor_tensor_field_count == 0U) {
    if (!metadata.tensor_fields.empty()) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_SCHEMA_INVALID"));
    }
    return {};
  }
  if (metadata.descriptor_tensor_field_count > 1U &&
      (metadata.tensor_layout || metadata.memory_conversion || metadata.tensor_allocation ||
       metadata.payload_options || metadata.tensor_byte_span_expression ||
       metadata.tensor_memory_kind_expression)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_SCOPE_REQUIRED"));
  }
  if (metadata.descriptor_tensor_field_count == 1U && !metadata.tensor_fields.empty() &&
      (metadata.tensor_layout || metadata.memory_conversion || metadata.tensor_allocation ||
       metadata.payload_options || metadata.tensor_byte_span_expression ||
       metadata.tensor_memory_kind_expression)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_SCOPE_CONFLICT"));
  }

  std::vector<std::uint16_t> field_ids;
  std::vector<std::uint16_t> tensor_indices;
  field_ids.reserve(metadata.tensor_fields.size());
  tensor_indices.reserve(metadata.tensor_fields.size());
  for (const auto& authored : metadata.tensor_fields) {
    auto valid = validate_one_tensor_field_metadata(metadata, authored, field_ids, tensor_indices);
    if (!valid) {
      return valid;
    }
  }
  return {};
}

[[nodiscard]] bool valid_normalized_rate(const Rate& rate) noexcept {
  return rate.numerator_hz != 0U && rate.denominator_hz != 0U &&
         std::gcd(rate.numerator_hz, rate.denominator_hz) == 1U;
}

[[nodiscard]] bool valid_normalized_rate_ratio(const RateRatio& ratio) noexcept {
  return ratio.numerator != 0U && ratio.denominator != 0U &&
         std::gcd(ratio.numerator, ratio.denominator) == 1U;
}

[[nodiscard]] bool valid_rate_binding_stage(LifecycleStage stage) noexcept {
  return stage == LifecycleStage::kConfigure || stage == LifecycleStage::kDiscover ||
         stage == LifecycleStage::kWarmUp || stage == LifecycleStage::kArm;
}

[[nodiscard]] expected<void, Error> validate_rate_metadata(
    const OperatorSpecCompilerAccess::PortMetadata& metadata) {
  if (metadata.rate_error.code != ErrorCode::kSuccess) {
    return make_unexpected(metadata.rate_error);
  }
  if (!closed_enum(metadata.rate_kind) ||
      (metadata.nominal_rate && !valid_normalized_rate(*metadata.nominal_rate)) ||
      (metadata.expected_rate && !valid_normalized_rate(*metadata.expected_rate)) ||
      (metadata.follows_ratio && !valid_normalized_rate_ratio(*metadata.follows_ratio)) ||
      (metadata.max_nominal_rate && !valid_normalized_rate(*metadata.max_nominal_rate)) ||
      (metadata.lifecycle_binding_stage &&
       !valid_rate_binding_stage(*metadata.lifecycle_binding_stage))) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "PORT_RATE_INVALID"));
  }

  const bool input = metadata.direction == PortDirection::kInput;
  const bool no_output_fields =
      !metadata.nominal_rate && metadata.follows_input_declaration == nullptr &&
      !metadata.follows_ratio && !metadata.lifecycle_binding_stage && !metadata.max_nominal_rate;
  switch (metadata.rate_kind) {
    case PortRateKind::kUnspecified:
      if (metadata.expected_rate || !no_output_fields) {
        break;
      }
      return {};
    case PortRateKind::kExpectedInput:
      if (input && metadata.expected_rate && no_output_fields) {
        return {};
      }
      break;
    case PortRateKind::kFixedNominal:
      if (!input && metadata.nominal_rate && !metadata.expected_rate &&
          metadata.follows_input_declaration == nullptr && !metadata.follows_ratio &&
          !metadata.lifecycle_binding_stage && !metadata.max_nominal_rate) {
        return {};
      }
      break;
    case PortRateKind::kFollowsInput:
      if (!input && !metadata.nominal_rate && !metadata.expected_rate &&
          metadata.follows_input_declaration != nullptr && metadata.follows_ratio &&
          !metadata.lifecycle_binding_stage && !metadata.max_nominal_rate) {
        return {};
      }
      break;
    case PortRateKind::kLifecycleNominal:
      if (!input && !metadata.nominal_rate && !metadata.expected_rate &&
          metadata.follows_input_declaration == nullptr && !metadata.follows_ratio &&
          metadata.lifecycle_binding_stage &&
          valid_rate_binding_stage(*metadata.lifecycle_binding_stage)) {
        return {};
      }
      break;
  }
  return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "PORT_RATE_CONFLICT"));
}

[[nodiscard]] expected<std::optional<std::size_t>, Error> validate_follows_reference(
    const OperatorSpec& spec, const OperatorSpecCompilerAccess::PortMetadata& metadata) {
  if (metadata.rate_kind != PortRateKind::kFollowsInput) {
    return std::optional<std::size_t>{};
  }
  for (std::size_t index = 0U; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    const auto candidate = OperatorSpecCompilerAccess::port(spec, index);
    if (candidate.declaration == metadata.follows_input_declaration &&
        candidate.direction == PortDirection::kInput) {
      return std::optional<std::size_t>{index};
    }
  }
  return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "PORT_RATE_REFERENCE_INVALID"));
}

// Port metadata validation is kept as one ordered fail-closed funnel so each declaration is
// rejected before any Graph-owned state is committed.
[[nodiscard]] expected<void, Error>
validate_port_metadata(  // NOLINT(readability-function-cognitive-complexity)
    const OperatorSpecCompilerAccess::PortMetadata& metadata) {
  auto rate = validate_rate_metadata(metadata);
  if (!rate) {
    return rate;
  }
  if (metadata.tensor_contract_error.code != ErrorCode::kSuccess) {
    return make_unexpected(metadata.tensor_contract_error);
  }
  auto tensor_fields = validate_tensor_field_metadata(metadata);
  if (!tensor_fields)
    return tensor_fields;
  auto name = validate_name(metadata.name, "GRAPH_PORT_NAME_INVALID");
  if (!name)
    return name;
  auto identity = validate_payload_identity(metadata.payload_identity);
  if (!identity)
    return identity;
  const bool schema = metadata.payload_identity.kind == PayloadIdentityKind::kSchema;
  const bool py_object = is_py_object_payload_identity(metadata.payload_identity);
  const bool schema_capacity_valid =
      (schema && !py_object) == (metadata.serialized_payload_capacity_bytes != 0U);
  const bool schema_runtime_valid =
      !schema ||
      (metadata.runtime_type_ops != nullptr &&
       ((py_object &&
         metadata.runtime_type_ops->codec_family ==
             RuntimePayloadCodecFamily::kProcessLocalObject &&
         metadata.runtime_type_ops->wire_capacity == 0U) ||
        (!py_object &&
         (metadata.runtime_type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema ||
          metadata.runtime_type_ops->codec_family ==
              RuntimePayloadCodecFamily::kDescriptorTensorSchema) &&
         metadata.runtime_type_ops->wire_capacity == metadata.serialized_payload_capacity_bytes)));
  if (!schema_capacity_valid || !schema_runtime_valid) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_SCHEMA_CAPACITY_INVALID"));
  }
  if ((metadata.direction == PortDirection::kInput && metadata.queue_depth == 0U) ||
      (metadata.direction == PortDirection::kOutput && metadata.max_emits_per_compute == 0U)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_BOUND_INVALID"));
  }
  if ((metadata.memory_conversion && !closed_enum(*metadata.memory_conversion)) ||
      (metadata.tensor_layout && metadata.tensor_layout->memory_kind &&
       !closed_enum(*metadata.tensor_layout->memory_kind)) ||
      (metadata.payload_options &&
       !closed_enum(metadata.payload_options->interface_tensor_emit_policy))) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_ENUM_INVALID"));
  }
  if (metadata.tensor_layout && metadata.tensor_layout->dtype &&
      !valid_tensor_dtype(*metadata.tensor_layout->dtype)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_DTYPE_INVALID"));
  }
  if (!metadata.diagnostic_alias.empty()) {
    auto alias = validate_name(metadata.diagnostic_alias, "GRAPH_PORT_ALIAS_INVALID");
    if (!alias)
      return alias;
  }
  if (metadata.tensor_allocation && (metadata.tensor_allocation->max_rank == 0U ||
                                     metadata.tensor_allocation->max_byte_span == 0U)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_ALLOCATION_INVALID"));
  }
  if (metadata.tensor_byte_span_expression &&
      (metadata.direction != PortDirection::kOutput || !metadata.tensor ||
       metadata.payload_identity.kind != PayloadIdentityKind::kTensor ||
       !metadata.tensor_contract ||
       metadata.tensor_contract->storage != TensorOutputStorage::kRuntimePool ||
       metadata.tensor_contract->max_byte_span || metadata.tensor_allocation)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID"));
  }
  if (metadata.tensor_memory_kind_expression &&
      (metadata.direction != PortDirection::kOutput || !metadata.tensor ||
       metadata.payload_identity.kind != PayloadIdentityKind::kTensor ||
       !metadata.tensor_contract ||
       metadata.tensor_contract->storage != TensorOutputStorage::kRuntimePool ||
       metadata.tensor_contract->representation.memory_kind ||
       (metadata.tensor_layout && metadata.tensor_layout->memory_kind))) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_MEMORY_KIND_DERIVATION_INVALID"));
  }
  if (metadata.payload_options) {
    auto options = validate_payload_options(
        *metadata.payload_options, metadata.payload_identity, metadata.tensor_layout);
    if (!options)
      return options;
  }
  return {};
}

}  // namespace

[[nodiscard]] bool schema_equal(const std::optional<SchemaIdentityRecord>& lhs,
                                const std::optional<SchemaIdentityRecord>& rhs) {
  if (lhs.has_value() != rhs.has_value())
    return false;
  return !lhs || schema_identity_equal(*lhs, *rhs);
}

namespace {

[[nodiscard]] bool payload_options_equal(const std::optional<PayloadOptionsRecord>& lhs,
                                         const std::optional<PayloadOptionsRecord>& rhs) {
  if (lhs.has_value() != rhs.has_value())
    return false;
  return !lhs || (schema_equal(lhs->schema, rhs->schema) &&
                  lhs->max_inline_bytes == rhs->max_inline_bytes &&
                  lhs->max_tensor_byte_span == rhs->max_tensor_byte_span &&
                  lhs->interface_tensor_emit_policy == rhs->interface_tensor_emit_policy);
}

[[nodiscard]] bool supports_generic_output_loan(
    const OperatorSpecCompilerAccess::PortMetadata& metadata) noexcept {
  return metadata.direction == PortDirection::kOutput && metadata.runtime_type_ops != nullptr &&
         metadata.runtime_type_ops->generic_loan_supported &&
         (!metadata.payload_identity.scalar ||
          metadata.payload_identity.scalar->kind != ScalarIdentityKind::kLocalEnum);
}

[[nodiscard]] bool port_setup_equal(const PortStorage& existing,
                                    const OperatorSpecCompilerAccess::PortMetadata& candidate,
                                    std::optional<GraphPortId> candidate_follows_input) {
  const auto candidate_setup = setup_port_configuration(candidate, candidate_follows_input);
  const auto& setup = existing.record.setup;
  return existing.record.direction == candidate.direction &&
         existing.record.optional == candidate.optional && existing.record.name == candidate.name &&
         existing.local_type_token == candidate.local_type_token &&
         (existing.local_type_info == candidate.local_type_info ||
          (existing.local_type_info != nullptr && candidate.local_type_info != nullptr &&
           *existing.local_type_info == *candidate.local_type_info)) &&
         payload_identity_exactly_equal(existing.record.payload_identity,
                                        candidate.payload_identity) &&
         existing.record.serialized_payload_capacity_bytes ==
             candidate.serialized_payload_capacity_bytes &&
         existing.record.generic_loan_supported == supports_generic_output_loan(candidate) &&
         existing.record.generic_loan_slot_size_bytes ==
             (supports_generic_output_loan(candidate)
                  ? static_cast<std::uint64_t>(
                        candidate.runtime_type_ops->generic_loan_slot_size_bytes)
                  : 0U) &&
         existing.record.descriptor_tensor_field_count == candidate.descriptor_tensor_field_count &&
         setup.queue_depth == candidate_setup.queue_depth &&
         setup.max_emits_per_compute == candidate_setup.max_emits_per_compute &&
         setup.late_joiner_history_depth == candidate_setup.late_joiner_history_depth &&
         setup.rate_kind == candidate_setup.rate_kind &&
         setup.declared_nominal_rate == candidate_setup.declared_nominal_rate &&
         setup.declared_expected_rate == candidate_setup.declared_expected_rate &&
         setup.follows_input == candidate_setup.follows_input &&
         setup.follows_ratio == candidate_setup.follows_ratio &&
         setup.lifecycle_binding_stage == candidate_setup.lifecycle_binding_stage &&
         setup.declared_max_nominal_rate == candidate_setup.declared_max_nominal_rate &&
         setup.diagnostic_alias == candidate_setup.diagnostic_alias &&
         tensor_port_layout_exactly_equal(setup.tensor_layout, candidate_setup.tensor_layout) &&
         setup.tensor_contract.has_value() == candidate_setup.tensor_contract.has_value() &&
         (!setup.tensor_contract ||
          tensor_contract_equal(*setup.tensor_contract, *candidate_setup.tensor_contract)) &&
         setup.memory_conversion == candidate_setup.memory_conversion &&
         setup.tensor_allocation.has_value() == candidate_setup.tensor_allocation.has_value() &&
         (!setup.tensor_allocation ||
          (setup.tensor_allocation->max_rank == candidate_setup.tensor_allocation->max_rank &&
           setup.tensor_allocation->max_byte_span ==
               candidate_setup.tensor_allocation->max_byte_span)) &&
         setup.tensor_byte_span_expression == candidate_setup.tensor_byte_span_expression &&
         setup.tensor_memory_kind_expression == candidate_setup.tensor_memory_kind_expression &&
         payload_options_equal(setup.payload_options, candidate_setup.payload_options) &&
         setup.accepts_projected_tensor == candidate_setup.accepts_projected_tensor &&
         tensor_field_configuration_equal(setup.tensor_fields, candidate_setup.tensor_fields);
}

[[nodiscard]] bool clock_key_equal(const ClockKey& lhs, const ClockKey& rhs) {
  return lhs == rhs;
}

[[nodiscard]] bool source_equal(const ReadinessSourceDeclarationRecord& existing,
                                const OperatorSpecCompilerAccess::SourceMetadata& candidate) {
  return existing.kind == candidate.kind && existing.name == candidate.name &&
         existing.requested_capacity == candidate.capacity &&
         existing.requested_sender_reference_capacity == candidate.sender_reference_capacity &&
         clock_key_equal(existing.requested_clock, candidate.clock) &&
         existing.wake_requirement == (candidate.kind == ReadinessSourceKind::kSpin
                                           ? ReadinessWakeRequirement::kBusyPollRequired
                                           : ReadinessWakeRequirement::kDeploymentSelected);
}

[[nodiscard]] bool lifecycle_callback_equal(
    const LifecycleCallbackAuthoringRecord& existing,
    const OperatorSpecCompilerAccess::LifecycleCallbackMetadata& candidate) noexcept {
  return existing.stage == candidate.stage && existing.required == candidate.required &&
         existing.method == candidate.method &&
         existing.origin == LifecycleCallbackOrigin::kAdvancedMember;
}

[[nodiscard]] constexpr LifecycleMethodId lifecycle_method_id(LifecycleStage stage) noexcept {
  return static_cast<LifecycleMethodId>(static_cast<std::uint8_t>(stage) + 1U);
}

[[nodiscard]] bool lifecycle_callback_equal_legacy(const LifecycleCallbackAuthoringRecord& existing,
                                                   LifecycleStage stage,
                                                   LifecycleCallbackOrigin origin) noexcept {
  return existing.stage == stage && existing.required &&
         existing.method == lifecycle_method_id(stage) && existing.origin == origin;
}

[[nodiscard]] bool has_lifecycle_callback(const OperatorSpec& spec, LifecycleStage stage) noexcept {
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::lifecycle_callback_count(spec);
       ++index) {
    if (OperatorSpecCompilerAccess::lifecycle_callback(spec, index).stage == stage)
      return true;
  }
  return false;
}

[[nodiscard]] bool lifecycle_fact_equal(
    const LifecycleFactAuthoringRecord& existing,
    const OperatorSpecCompilerAccess::LifecycleFactMetadata& candidate) noexcept {
  static_assert(
      static_cast<std::uint8_t>(LifecycleFactKind::kProduces) ==
      static_cast<std::uint8_t>(OperatorSpecCompilerAccess::LifecycleFactKind::kProduces));
  static_assert(
      static_cast<std::uint8_t>(LifecycleFactKind::kRequires) ==
      static_cast<std::uint8_t>(OperatorSpecCompilerAccess::LifecycleFactKind::kRequires));
  const auto candidate_kind = static_cast<LifecycleFactKind>(candidate.kind);
  return existing.stage == candidate.stage && existing.kind == candidate_kind &&
         existing.stable_name == candidate.stable_name;
}

[[nodiscard]] expected<void, Error> validate_lifecycle_metadata(const OperatorSpec& spec,
                                                                const void* operator_type_token) {
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::lifecycle_callback_count(spec);
       ++index) {
    const auto callback = OperatorSpecCompilerAccess::lifecycle_callback(spec, index);
    if (!closed_enum(callback.stage)) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_STAGE_INVALID"));
    }
    if (callback.declaring_type_token != operator_type_token) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_LIFECYCLE_CALLBACK_OWNER_MISMATCH"));
    }
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::lifecycle_fact_count(spec);
       ++index) {
    const auto fact = OperatorSpecCompilerAccess::lifecycle_fact(spec, index);
    if (!closed_enum(fact.stage)) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_STAGE_INVALID"));
    }
    if (!closed_enum(fact.kind)) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_FACT_KIND_INVALID"));
    }
    auto fact_name = validate_name(fact.stable_name, "GRAPH_LIFECYCLE_KEY_NAME_INVALID");
    if (!fact_name)
      return make_unexpected(std::move(fact_name.error()));
  }
  return {};
}

}  // namespace

// Registration intentionally keeps validation, ownership transfer, and rollback-sensitive state
// publication in one ordered transaction.
expected<OperatorRegistration, Error>
graph_register_operator(  // NOLINT(readability-function-cognitive-complexity)
    const std::shared_ptr<GraphState>& state, std::string_view name,
    std::shared_ptr<OperatorFactoryBase> factory, std::unique_ptr<OperatorBase> prototype,
    const void* operator_type_token, const InitialOperatorParameters* initial_params) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  auto valid_name = validate_name(name, "GRAPH_OPERATOR_NAME_INVALID");
  if (!valid_name)
    return make_unexpected(std::move(valid_name.error()));
  if (!factory || !prototype || operator_type_token == nullptr) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_OPERATOR_FACTORY_INVALID"));
  }
  const OperatorFactoryBase::LifecycleHookSet lifecycle_hooks = factory->lifecycle_hooks();

  OperatorSpec spec;
  auto setup = run_setup(*prototype, spec);
  if (!setup) {
    setup.error().message +=
        fmt::format("; operator={} callback=setup", diagnostic_display_text(name));
    return make_unexpected(std::move(setup.error()));
  }
  std::vector<std::optional<std::size_t>> follows_port_indices;
  follows_port_indices.reserve(OperatorSpecCompilerAccess::port_count(spec));
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    const auto metadata = OperatorSpecCompilerAccess::port(spec, index);
    auto valid = validate_port_metadata(metadata);
    if (!valid)
      return make_unexpected(std::move(valid.error()));
    auto follows = validate_follows_reference(spec, metadata);
    if (!follows) {
      return make_unexpected(std::move(follows.error()));
    }
    follows_port_indices.push_back(*follows);
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::source_count(spec); ++index) {
    const auto source = OperatorSpecCompilerAccess::source(spec, index);
    auto source_name = validate_name(source.name, "GRAPH_SOURCE_NAME_INVALID");
    if (!source_name)
      return make_unexpected(std::move(source_name.error()));
    if (!closed_enum(source.kind)) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_SOURCE_KIND_INVALID"));
    }
    const bool single =
        source.kind == ReadinessSourceKind::kReadable || source.kind == ReadinessSourceKind::kSpin;
    if ((single && source.capacity != 1U) || (!single && source.capacity == 0U) ||
        (source.kind == ReadinessSourceKind::kNotification &&
         source.sender_reference_capacity == 0U)) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_SOURCE_BOUND_INVALID"));
    }
  }
  auto valid_lifecycle = validate_lifecycle_metadata(spec, operator_type_token);
  if (!valid_lifecycle)
    return make_unexpected(std::move(valid_lifecycle.error()));
  if ((lifecycle_hooks.start && has_lifecycle_callback(spec, LifecycleStage::kStart)) ||
      (lifecycle_hooks.stop && has_lifecycle_callback(spec, LifecycleStage::kStop))) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_LIFECYCLE_CALLBACK_DUPLICATE"));
  }
  std::unique_lock lock(state->mutex_);
  if (state->vertex_names_.contains(std::string(name))) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_OPERATOR_NAME_DUPLICATE"));
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    const auto identity = OperatorSpecCompilerAccess::port(spec, index).payload_identity;
    auto catalog = validate_pod_catalog(*state, identity);
    if (!catalog)
      return make_unexpected(std::move(catalog.error()));
    if (identity.pod) {
      const auto& candidate = identity.pod.value();
      for (std::size_t prior = 0; prior < index; ++prior) {
        const auto prior_identity = OperatorSpecCompilerAccess::port(spec, prior).payload_identity;
        if (prior_identity.pod && prior_identity.pod->canonical_name == candidate.canonical_name &&
            prior_identity.pod->version == candidate.version &&
            !payload_identity_exactly_equal(prior_identity, identity)) {
          return make_unexpected(
              graph_error(ErrorCode::kIncompatible, "GRAPH_POD_IDENTITY_CONFLICT"));
        }
      }
    }
  }

  std::unordered_set<const void*> candidate_declarations;
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    const auto metadata = OperatorSpecCompilerAccess::port(spec, index);
    if (!candidate_declarations.emplace(metadata.declaration).second) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_DECLARATION_DUPLICATE"));
    }
    if (metadata.direction == PortDirection::kInput) {
      const auto& input = *static_cast<const InputBase*>(metadata.declaration);
      if (GraphPortOwnershipAccess::owner(input) != nullptr) {
        return make_unexpected(
            graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_ALREADY_OWNED"));
      }
    } else {
      const auto& output = *static_cast<const OutputBase*>(metadata.declaration);
      if (GraphPortOwnershipAccess::owner(output) != nullptr) {
        return make_unexpected(
            graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_ALREADY_OWNED"));
      }
    }
  }

  OperatorStorage storage;
  GraphMutationRollback rollback(*state);
  const std::size_t operator_index = state->operators_.size();
  const VertexId vertex = state->next_vertex_id();
  const OperatorHostMetadata host_metadata = prototype->host_metadata();
  if (host_metadata.kind != OperatorHostKind::kNative &&
      host_metadata.kind != OperatorHostKind::kForeign) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_OPERATOR_HOST_METADATA_INVALID"));
  }
  if (!host_metadata.serialization_domain.empty() &&
      !valid_operator_serialization_domain(host_metadata.serialization_domain)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_OPERATOR_HOST_METADATA_INVALID"));
  }
  storage = OperatorStorage{
      .name = std::string(name),
      .handle_identity = std::make_shared<std::byte>(),
      .operator_type_token = operator_type_token,
      .factory = std::move(factory),
      .prototype = std::shared_ptr<OperatorBase>(std::move(prototype)),
      .vertex = vertex,
  };
  storage.ports.reserve(OperatorSpecCompilerAccess::port_count(spec));
  std::vector<GraphPortId> port_ids;
  port_ids.reserve(OperatorSpecCompilerAccess::port_count(spec));
  for (std::size_t index = 0U; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    port_ids.push_back(state->next_port_id());
  }

  state->vertices_.push_back(VertexRecord{
      .id = vertex,
      .graph = state->graph_record_.id,
      .kind = VertexKind::kOperator,
      .name = std::string(name),
      .operator_host = host_metadata.kind,
      .serialization_domain = std::string(host_metadata.serialization_domain),
      .cuda_graph_capture_eligible = OperatorSpecCompilerAccess::cuda_graph_capture_eligible(spec),
  });
  GraphMutationRollback::checkpoint(*state);
  state->vertex_names_.emplace(name);
  GraphMutationRollback::checkpoint(*state);
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    const auto metadata = OperatorSpecCompilerAccess::port(spec, index);
    const GraphPortId port_id = port_ids[index];
    const std::optional<GraphPortId> follows_input =
        follows_port_indices[index]
            ? std::optional<GraphPortId>{port_ids[*follows_port_indices[index]]}
            : std::nullopt;
    PortStorage port{
        .record =
            PortRecord{
                .id = port_id,
                .owner = vertex,
                .owner_kind = PortOwnerKind::kOperator,
                .direction = metadata.direction,
                .optional = metadata.optional,
                .name = std::string(metadata.name),
                .payload_identity = metadata.payload_identity,
                .serialized_payload_capacity_bytes = metadata.serialized_payload_capacity_bytes,
                .generic_loan_supported = supports_generic_output_loan(metadata),
                .generic_loan_slot_size_bytes =
                    supports_generic_output_loan(metadata)
                        ? static_cast<std::uint64_t>(
                              metadata.runtime_type_ops->generic_loan_slot_size_bytes)
                        : 0U,
                .descriptor_tensor_field_count = metadata.descriptor_tensor_field_count,
                .setup = setup_port_configuration(metadata, follows_input),
            },
        .handle_identity = std::make_shared<std::byte>(),
        .declaration = metadata.declaration,
        .local_type_token = metadata.local_type_token,
        .operator_index = operator_index,
        .setup_tensor_allocation = metadata.tensor_allocation,
        .local_type_info = metadata.local_type_info,
    };
    state->ports_.push_back(std::move(port));
    GraphMutationRollback::checkpoint(*state);
    state->port_by_declaration_.emplace(metadata.declaration, port_id);
    GraphMutationRollback::checkpoint(*state);
    storage.ports.push_back(port_id);
    if (metadata.direction == PortDirection::kInput) {
      GraphPortOwnershipAccess::claim(*static_cast<const InputBase*>(metadata.declaration),
                                      state.get());
    } else {
      GraphPortOwnershipAccess::claim(*static_cast<const OutputBase*>(metadata.declaration),
                                      state.get());
    }
    GraphMutationRollback::checkpoint(*state);
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    const auto identity = OperatorSpecCompilerAccess::port(spec, index).payload_identity;
    record_pod_identity(*state, identity);
    if (identity.pod)
      GraphMutationRollback::checkpoint(*state);
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::source_count(spec); ++index) {
    const auto source = OperatorSpecCompilerAccess::source(spec, index);
    state->readiness_sources_.push_back(ReadinessSourceDeclarationRecord{
        .owner = vertex,
        .declaration_ordinal = static_cast<std::uint32_t>(index),
        .kind = source.kind,
        .name = std::string(source.name),
        .requested_capacity = source.capacity,
        .requested_sender_reference_capacity = source.kind == ReadinessSourceKind::kNotification
                                                   ? source.sender_reference_capacity
                                                   : 0U,
        .requested_clock = source.clock,
        .wake_requirement = source.kind == ReadinessSourceKind::kSpin
                                ? ReadinessWakeRequirement::kBusyPollRequired
                                : ReadinessWakeRequirement::kDeploymentSelected,
        .referenced = false,
    });
    GraphMutationRollback::checkpoint(*state);
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::lifecycle_callback_count(spec);
       ++index) {
    const auto callback = OperatorSpecCompilerAccess::lifecycle_callback(spec, index);
    state->lifecycle_callbacks_.push_back(LifecycleCallbackAuthoringRecord{
        .owner = vertex,
        .stage = callback.stage,
        .required = callback.required,
        .method = callback.method,
        .origin = LifecycleCallbackOrigin::kAdvancedMember,
    });
    GraphMutationRollback::checkpoint(*state);
  }
  if (lifecycle_hooks.start) {
    state->lifecycle_callbacks_.push_back(LifecycleCallbackAuthoringRecord{
        .owner = vertex,
        .stage = LifecycleStage::kStart,
        .required = true,
        .method = lifecycle_method_id(LifecycleStage::kStart),
        .origin = LifecycleCallbackOrigin::kLegacyStart,
    });
    GraphMutationRollback::checkpoint(*state);
  }
  if (lifecycle_hooks.stop) {
    state->lifecycle_callbacks_.push_back(LifecycleCallbackAuthoringRecord{
        .owner = vertex,
        .stage = LifecycleStage::kStop,
        .required = true,
        .method = lifecycle_method_id(LifecycleStage::kStop),
        .origin = LifecycleCallbackOrigin::kLegacyStop,
    });
    GraphMutationRollback::checkpoint(*state);
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::lifecycle_fact_count(spec);
       ++index) {
    const auto fact = OperatorSpecCompilerAccess::lifecycle_fact(spec, index);
    state->lifecycle_facts_.push_back(LifecycleFactAuthoringRecord{
        .owner = vertex,
        .stage = fact.stage,
        .kind = static_cast<LifecycleFactKind>(fact.kind),
        .stable_name = std::string(fact.stable_name),
    });
    GraphMutationRollback::checkpoint(*state);
  }
  state->operators_.push_back(std::move(storage));
  state->vertex_adjacency_.emplace_back();
  state->cycle_visit_marks_.push_back(0U);
  if (initial_params != nullptr) {
    lock.unlock();
    auto initialized = [&]() {
      try {
        return graph_set_operator_params(state,
                                         operator_index,
                                         initial_params->value,
                                         operator_type_token,
                                         state->operators_[operator_index].handle_identity.get(),
                                         initial_params->local_type_token,
                                         initial_params->schema,
                                         nullptr);
      } catch (...) {
        lock.lock();
        throw;
      }
    }();
    lock.lock();
    if (!initialized) {
      return make_unexpected(std::move(initialized.error()));
    }
  }
  state->mark_mutation();
  GraphMutationRollback::checkpoint(*state);
  rollback.commit();
  graph_log_noexcept([&] { HOLOSCAN_LOG_TRACE("Registered Graph operator '{}'", name); });
  return OperatorRegistration{
      .index = operator_index,
      .handle_identity = state->operators_[operator_index].handle_identity,
  };
}

const OperatorBase* graph_operator_prototype(const std::shared_ptr<GraphState>& state,
                                             std::size_t index, const void* operator_type_token,
                                             const void* handle_identity) noexcept {
  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size())
    return nullptr;
  const auto& op = state->operators_[index];
  if (op.operator_type_token != operator_type_token ||
      op.handle_identity.get() != handle_identity) {
    return nullptr;
  }
  return op.prototype.get();
}

const void* graph_operator_port_declaration(const std::shared_ptr<GraphState>& state,
                                            std::size_t index, const void* handle_identity,
                                            std::string_view name) noexcept {
  if (!state)
    return nullptr;
  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size() ||
      state->operators_[index].handle_identity.get() != handle_identity) {
    return nullptr;
  }
  const auto& op = state->operators_[index];
  for (const GraphPortId port_id : op.ports) {
    const PortStorage* port = find_port(*state, port_id);
    if (port != nullptr && port->record.name == name)
      return port->declaration;
  }
  return nullptr;
}

expected<std::vector<PortRecord>, Error> graph_operator_ports(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity) {
  if (!state) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size() ||
      state->operators_[index].operator_type_token != operator_type_token ||
      state->operators_[index].handle_identity.get() != handle_identity) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  std::vector<PortRecord> result;
  result.reserve(state->operators_[index].ports.size());
  for (const GraphPortId port_id : state->operators_[index].ports) {
    const PortStorage* port = find_port(*state, port_id);
    if (port == nullptr) {
      return make_unexpected(graph_error(ErrorCode::kFailure, "GRAPH_PORT_RECORD_MISSING"));
    }
    result.push_back(port->record);
  }
  return result;
}

expected<const void*, Error> graph_resolve_operator_port(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity, std::string_view name, PortDirection direction,
    const PayloadIdentityRecord& payload_identity) {
  return graph_resolve_operator_port(state,
                                     index,
                                     operator_type_token,
                                     handle_identity,
                                     name,
                                     direction,
                                     payload_identity,
                                     nullptr);
}

expected<const void*, Error> graph_resolve_operator_port(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity, std::string_view name, PortDirection direction,
    const PayloadIdentityRecord& payload_identity, const void* payload_local_type_token) {
  return graph_resolve_operator_port(state,
                                     index,
                                     operator_type_token,
                                     handle_identity,
                                     name,
                                     direction,
                                     payload_identity,
                                     payload_local_type_token,
                                     nullptr);
}

expected<const void*, Error> graph_resolve_operator_port(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity, std::string_view name, PortDirection direction,
    const PayloadIdentityRecord& payload_identity, const void* payload_local_type_token,
    const std::type_info* payload_local_type_info) {
  if (!state) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size() ||
      state->operators_[index].operator_type_token != operator_type_token ||
      state->operators_[index].handle_identity.get() != handle_identity) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const auto port_error = [&](ErrorCode code,
                              std::string_view token,
                              std::string_view reason,
                              const PortStorage* port = nullptr) {
    Error error{code};
    try {
      error.message = fmt::format(
          "{}; operator='{}' port='{}'; {}; requested={} {} native='{}'",
          token,
          state->operators_[index].name,
          name,
          reason,
          direction == PortDirection::kInput ? "input" : "output",
          graph_payload_identity_name(payload_identity),
          payload_local_type_info == nullptr ? "unavailable" : payload_local_type_info->name());
      if (port != nullptr) {
        error.message += fmt::format(
            "; declared={} {} native='{}'",
            port->record.direction == PortDirection::kInput ? "input" : "output",
            graph_payload_identity_name(port->record.payload_identity),
            port->local_type_info == nullptr ? "unavailable" : port->local_type_info->name());
      }
    } catch (...) {
      // Preserve rejection even when supplemental cold diagnostics cannot be allocated.
      error.message.clear();
    }
    return make_unexpected(std::move(error));
  };
  for (const GraphPortId port_id : state->operators_[index].ports) {
    const PortStorage* port = find_port(*state, port_id);
    if (port == nullptr || port->record.name != name) {
      continue;
    }
    const auto mismatch_reason = graph_operator_port_type_mismatch_reason(
        *port, direction, payload_identity, payload_local_type_token, payload_local_type_info);
    if (!mismatch_reason.empty()) {
      return port_error(
          ErrorCode::kIncompatible, "GRAPH_PORT_TYPE_MISMATCH", mismatch_reason, port);
    }
    if (port->declaration == nullptr) {
      return port_error(ErrorCode::kInvalidArgument,
                        "GRAPH_PORT_DECLARATION_INVALID",
                        "port declaration is unavailable",
                        port);
    }
    return port->declaration;
  }
  return port_error(ErrorCode::kNotFound, "GRAPH_PORT_NOT_FOUND", "named port does not exist");
}

expected<PortRecord, Error> graph_operator_port_record(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity, std::string_view name, PortDirection direction) {
  if (!state) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size() ||
      state->operators_[index].operator_type_token != operator_type_token ||
      state->operators_[index].handle_identity.get() != handle_identity) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  for (const GraphPortId id : state->operators_[index].ports) {
    const auto* port = find_port(*state, id);
    if (port != nullptr && port->record.name == name) {
      if (port->record.direction != direction) {
        return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_PORT_TYPE_MISMATCH"));
      }
      return port->record;
    }
  }
  return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_PORT_NOT_FOUND"));
}

bool graph_operator_valid(const std::shared_ptr<GraphState>& state, std::size_t index,
                          const void* operator_type_token, const void* handle_identity) noexcept {
  return graph_operator_prototype(state, index, operator_type_token, handle_identity) != nullptr;
}

std::string_view graph_operator_name(const std::shared_ptr<GraphState>& state, std::size_t index,
                                     const void* handle_identity) noexcept {
  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size() ||
      state->operators_[index].handle_identity.get() != handle_identity) {
    return {};
  }
  return state->operators_[index].name;
}

expected<std::string, Error> graph_operator_path(const std::shared_ptr<GraphState>& state,
                                                 std::size_t index, const void* handle_identity) {
  if (!state) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (state->owner_ == nullptr || index >= state->operators_.size() ||
      state->operators_[index].handle_identity.get() != handle_identity) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const VertexId vertex_id = state->operators_[index].vertex;
  const auto vertex = std::ranges::find(state->vertices_, vertex_id, &VertexRecord::id);
  if (vertex == state->vertices_.end()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  return authored_operator_path(*state, *vertex);
}

expected<OperatorRegistration, Error> graph_find_operator(const std::shared_ptr<GraphState>& state,
                                                          std::string_view name) noexcept {
  if (!state) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (state->owner_ == nullptr) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const auto found = std::ranges::find(state->operators_, name, &OperatorStorage::name);
  if (found == state->operators_.end()) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_OPERATOR_NOT_FOUND"));
  }
  return OperatorRegistration{
      .index = static_cast<std::size_t>(std::distance(state->operators_.begin(), found)),
      .handle_identity = found->handle_identity,
  };
}

OperatorRef graph_operator_ref(const std::shared_ptr<GraphState>& state, std::size_t index,
                               const void* handle_identity) noexcept {
  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size() ||
      state->operators_[index].handle_identity.get() != handle_identity) {
    return {};
  }
  return state->operators_[index].plan_ref;
}

expected<GraphOperatorLocator, Error> graph_operator_locator(
    const std::shared_ptr<GraphState>& state, std::size_t index,
    const void* handle_identity) noexcept {
  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size() ||
      state->operators_[index].handle_identity.get() != handle_identity) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return GraphOperatorLocator{
      .graph = state->instance_id_,
      .generation = state->generation_,
      .root_operator_index = index,
  };
}

expected<VertexId, Error> graph_operator_vertex(const std::shared_ptr<GraphState>& state,
                                                std::size_t index, const void* operator_type_token,
                                                const void* handle_identity) {
  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size() ||
      state->operators_[index].operator_type_token != operator_type_token ||
      state->operators_[index].handle_identity.get() != handle_identity) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "PARAM_CONFIG_HANDLE_INVALID"));
  }
  return state->operators_[index].vertex;
}

expected<GraphOperatorParameterTarget, Error> graph_operator_parameter_target(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity) {
  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "PARAM_CONFIG_HANDLE_INVALID"));
  }
  const auto& op = state->operators_[index];
  if (op.operator_type_token != operator_type_token ||
      op.handle_identity.get() != handle_identity || !op.factory || !op.prototype || !op.vertex) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "PARAM_CONFIG_HANDLE_INVALID"));
  }
  const ParamTypeDescriptor& type = op.factory->parameter_type();
  const ParamSchemaDescriptor* const schema = op.factory->parameter_schema();
  if (type.local_type_token == nullptr || (schema != nullptr && schema->local_type != &type)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "PARAM_CONFIG_HANDLE_INVALID"));
  }
  return GraphOperatorParameterTarget{
      .vertex = op.vertex,
      .schema = schema,
  };
}

namespace {

[[nodiscard]] bool parameter_schema_stable_equal(const ParamSchemaDescriptor* left,
                                                 const ParamSchemaDescriptor* right) noexcept {
  if (left == nullptr || right == nullptr) {
    return left == right;
  }
  return left->stable_id == right->stable_id && left->version == right->version &&
         left->structural_hash == right->structural_hash &&
         left->contract_hash == right->contract_hash;
}

[[nodiscard]] bool parameter_type_compatible(
    const void* caller_type_token, const ParamSchemaDescriptor* caller_schema,
    const ParamTypeDescriptor& defining_type,
    const ParamSchemaDescriptor* defining_schema) noexcept {
  if (caller_type_token == nullptr || defining_type.local_type_token == nullptr) {
    return false;
  }
  if (caller_schema == nullptr || defining_schema == nullptr) {
    return caller_schema == nullptr && defining_schema == nullptr &&
           caller_type_token == defining_type.local_type_token;
  }
  return caller_schema->local_type != nullptr &&
         caller_schema->local_type->local_type_token == caller_type_token &&
         parameter_schema_stable_equal(caller_schema, defining_schema);
}

}  // namespace

// Parameter replacement validates a complete candidate and its port/source invariants before
// atomically transferring ownership into the existing logical Operator identity.
expected<void, Error>
graph_set_operator_params(  // NOLINT(readability-function-cognitive-complexity)
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* params,
    const void* expected_operator_type_token, const void* handle_identity,
    const void* caller_parameter_type_token, const ParamSchemaDescriptor* caller_schema,
    std::shared_ptr<const ParameterInputEvidenceState> evidence) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));

  std::shared_ptr<OperatorFactoryBase> factory;
  VertexId vertex;
  const void* defining_operator_type_token{};
  {
    const std::lock_guard lock(state->mutex_);
    if (index >= state->operators_.size()) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
    }
    if (state->operators_[index].operator_type_token != expected_operator_type_token ||
        state->operators_[index].handle_identity.get() != handle_identity) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
    }
    factory = state->operators_[index].factory;
    vertex = state->operators_[index].vertex;
    defining_operator_type_token = state->operators_[index].operator_type_token;
  }
  if (!factory) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const ParamTypeDescriptor& defining_type = factory->parameter_type();
  const ParamSchemaDescriptor* const defining_schema = factory->parameter_schema();
  if (defining_type.local_type_token == nullptr ||
      (defining_schema != nullptr && defining_schema->local_type != &defining_type) ||
      !parameter_type_compatible(
          caller_parameter_type_token, caller_schema, defining_type, defining_schema)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARAMETER_TYPE_MISMATCH"));
  }
  auto updated_factory = factory->clone_with_params(params, defining_type.local_type_token);
  if (!updated_factory) {
    return make_unexpected(std::move(updated_factory.error()));
  }
  const ParamTypeDescriptor& updated_type = updated_factory.value()->parameter_type();
  const ParamSchemaDescriptor* const updated_schema = updated_factory.value()->parameter_schema();
  if (updated_type.local_type_token != defining_type.local_type_token ||
      (updated_schema != nullptr && updated_schema->local_type != &updated_type) ||
      !parameter_schema_stable_equal(defining_schema, updated_schema)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARAMETER_TYPE_MISMATCH"));
  }
  if (evidence) {
    if (updated_schema == nullptr || evidence->local_type_token == nullptr ||
        evidence->schema_id != updated_schema->stable_id ||
        evidence->schema_version != updated_schema->version ||
        evidence->structural_hash != updated_schema->structural_hash ||
        evidence->contract_hash != updated_schema->contract_hash) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_INPUT_EVIDENCE_MISMATCH"));
    }
    auto normalized_evidence = std::make_shared<ParameterInputEvidenceState>(*evidence);
    normalized_evidence->local_type_token = defining_type.local_type_token;
    evidence = std::move(normalized_evidence);
  }
  auto candidate = updated_factory.value()->create(nullptr);
  if (!candidate)
    return make_unexpected(std::move(candidate.error()));
  OperatorSpec spec;
  auto setup = run_setup(*candidate.value(), spec);
  if (!setup)
    return setup;
  auto valid_lifecycle = validate_lifecycle_metadata(spec, defining_operator_type_token);
  if (!valid_lifecycle)
    return make_unexpected(std::move(valid_lifecycle.error()));
  const OperatorFactoryBase::LifecycleHookSet lifecycle_hooks =
      updated_factory.value()->lifecycle_hooks();
  if ((lifecycle_hooks.start && has_lifecycle_callback(spec, LifecycleStage::kStart)) ||
      (lifecycle_hooks.stop && has_lifecycle_callback(spec, LifecycleStage::kStop))) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_LIFECYCLE_CALLBACK_DUPLICATE"));
  }

  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size() || state->operators_[index].factory != factory ||
      state->operators_[index].operator_type_token != expected_operator_type_token ||
      state->operators_[index].handle_identity.get() != handle_identity) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_OPERATOR_CHANGED"));
  }
  auto& op = state->operators_[index];
  const auto vertex_record = std::find_if(
      state->vertices_.begin(), state->vertices_.end(), [vertex](const auto& candidate) {
        return candidate.id == vertex;
      });
  if (vertex_record == state->vertices_.end())
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_OPERATOR_CHANGED"));
  if (vertex_record->cuda_graph_capture_eligible !=
      OperatorSpecCompilerAccess::cuda_graph_capture_eligible(spec)) {
    return make_unexpected(
        graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_OPERATOR_SPEC_CHANGED"));
  }
  if (OperatorSpecCompilerAccess::port_count(spec) != op.ports.size()) {
    return make_unexpected(
        graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_PORT_SHAPE_CHANGED"));
  }
  std::vector<const void*> declarations;
  declarations.reserve(op.ports.size());
  std::unordered_map<const void*, GraphPortId> replay_port_ids;
  replay_port_ids.reserve(op.ports.size());
  for (std::size_t port_index = 0; port_index < op.ports.size(); ++port_index) {
    const auto metadata = OperatorSpecCompilerAccess::port(spec, port_index);
    if (!replay_port_ids.emplace(metadata.declaration, op.ports[port_index]).second) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_DECLARATION_DUPLICATE"));
    }
  }
  for (std::size_t port_index = 0; port_index < op.ports.size(); ++port_index) {
    const auto metadata = OperatorSpecCompilerAccess::port(spec, port_index);
    PortStorage* existing = find_port(*state, op.ports[port_index]);
    std::optional<GraphPortId> follows_input;
    if (metadata.follows_input_declaration != nullptr) {
      const auto found = replay_port_ids.find(metadata.follows_input_declaration);
      if (found == replay_port_ids.end()) {
        return make_unexpected(
            graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_PORT_SHAPE_CHANGED"));
      }
      follows_input = found->second;
    }
    if (existing == nullptr || !port_setup_equal(*existing, metadata, follows_input)) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_PORT_SHAPE_CHANGED"));
    }
    const void* owner =
        metadata.direction == PortDirection::kInput
            ? GraphPortOwnershipAccess::owner(*static_cast<const InputBase*>(metadata.declaration))
            : GraphPortOwnershipAccess::owner(
                  *static_cast<const OutputBase*>(metadata.declaration));
    if (owner != nullptr && owner != state.get()) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_ALREADY_OWNED"));
    }
    declarations.push_back(metadata.declaration);
  }
  std::vector<ReadinessSourceDeclarationRecord*> existing_sources;
  for (auto& source : state->readiness_sources_) {
    if (source.owner == vertex)
      existing_sources.push_back(&source);
  }
  if (OperatorSpecCompilerAccess::source_count(spec) != existing_sources.size()) {
    return make_unexpected(
        graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_SOURCE_SHAPE_CHANGED"));
  }
  for (std::size_t source_index = 0; source_index < existing_sources.size(); ++source_index) {
    if (!source_equal(*existing_sources[source_index],
                      OperatorSpecCompilerAccess::source(spec, source_index))) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_SOURCE_SHAPE_CHANGED"));
    }
  }
  std::vector<const LifecycleCallbackAuthoringRecord*> existing_callbacks;
  for (const auto& callback : state->lifecycle_callbacks_) {
    if (callback.owner == vertex)
      existing_callbacks.push_back(&callback);
  }
  const std::size_t expected_callback_count =
      OperatorSpecCompilerAccess::lifecycle_callback_count(spec) +
      static_cast<std::size_t>(lifecycle_hooks.start) +
      static_cast<std::size_t>(lifecycle_hooks.stop);
  if (expected_callback_count != existing_callbacks.size()) {
    return make_unexpected(
        graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_LIFECYCLE_SHAPE_CHANGED"));
  }
  for (std::size_t callback_index = 0; callback_index < existing_callbacks.size();
       ++callback_index) {
    bool matches = false;
    const std::size_t advanced_count = OperatorSpecCompilerAccess::lifecycle_callback_count(spec);
    if (callback_index < advanced_count) {
      matches = lifecycle_callback_equal(
          *existing_callbacks[callback_index],
          OperatorSpecCompilerAccess::lifecycle_callback(spec, callback_index));
    } else {
      std::size_t legacy_index = callback_index - advanced_count;
      if (lifecycle_hooks.start) {
        if (legacy_index == 0U) {
          matches = lifecycle_callback_equal_legacy(*existing_callbacks[callback_index],
                                                    LifecycleStage::kStart,
                                                    LifecycleCallbackOrigin::kLegacyStart);
        } else {
          --legacy_index;
        }
      }
      if (!matches && lifecycle_hooks.stop && legacy_index == 0U) {
        matches = lifecycle_callback_equal_legacy(*existing_callbacks[callback_index],
                                                  LifecycleStage::kStop,
                                                  LifecycleCallbackOrigin::kLegacyStop);
      }
    }
    if (!matches) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_LIFECYCLE_SHAPE_CHANGED"));
    }
  }
  std::vector<const LifecycleFactAuthoringRecord*> existing_facts;
  for (const auto& fact : state->lifecycle_facts_) {
    if (fact.owner == vertex)
      existing_facts.push_back(&fact);
  }
  if (OperatorSpecCompilerAccess::lifecycle_fact_count(spec) != existing_facts.size()) {
    return make_unexpected(
        graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_LIFECYCLE_SHAPE_CHANGED"));
  }
  for (std::size_t fact_index = 0; fact_index < existing_facts.size(); ++fact_index) {
    if (!lifecycle_fact_equal(*existing_facts[fact_index],
                              OperatorSpecCompilerAccess::lifecycle_fact(spec, fact_index))) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_LIFECYCLE_SHAPE_CHANGED"));
    }
  }

  auto replacement_map = state->port_by_declaration_;
  for (const GraphPortId port_id : op.ports) {
    PortStorage* existing = find_port(*state, port_id);
    replacement_map.erase(existing->declaration);
  }
  for (std::size_t port_index = 0; port_index < op.ports.size(); ++port_index) {
    const auto [unused, inserted] =
        replacement_map.emplace(declarations[port_index], op.ports[port_index]);
    (void)unused;
    if (!inserted) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_DECLARATION_DUPLICATE"));
    }
  }
  // Constructing shared ownership may allocate its control block. Complete that fallible work
  // before replacing declarations or factories so the publication sequence below is no-throw.
  std::shared_ptr<OperatorBase> replacement_prototype{std::move(candidate.value())};
  using DeclarationMap = decltype(state->port_by_declaration_);
  static_assert(noexcept(std::declval<DeclarationMap&>().swap(std::declval<DeclarationMap&>())));
  static_assert(std::is_nothrow_move_assignable_v<std::shared_ptr<OperatorFactoryBase>>);
  static_assert(
      std::is_nothrow_move_assignable_v<std::shared_ptr<const ParameterInputEvidenceState>>);
  static_assert(std::is_nothrow_move_assignable_v<std::shared_ptr<OperatorBase>>);
  GraphMutationRollback::checkpoint(*state);
  for (std::size_t port_index = 0; port_index < op.ports.size(); ++port_index) {
    const auto metadata = OperatorSpecCompilerAccess::port(spec, port_index);
    PortStorage* existing = find_port(*state, op.ports[port_index]);
    existing->declaration = declarations[port_index];
    existing->local_type_info = metadata.local_type_info;
    if (metadata.direction == PortDirection::kInput) {
      GraphPortOwnershipAccess::claim(*static_cast<const InputBase*>(metadata.declaration),
                                      state.get());
    } else {
      GraphPortOwnershipAccess::claim(*static_cast<const OutputBase*>(metadata.declaration),
                                      state.get());
    }
  }
  state->port_by_declaration_.swap(replacement_map);
  op.factory = std::move(updated_factory.value());
  op.prototype = std::move(replacement_prototype);
  op.parameter_input_evidence = std::move(evidence);
  state->mark_mutation();
  return {};
}

}  // namespace holoscan::detail
