/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parameter_plan_lowering.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/parameter_snapshot.hpp>

#include "../core/operator_spec_compiler_access.hpp"
#include "../core/parameter_catalog_internal.hpp"
#include "../core/parameter_input_internal.hpp"
#include "../core/plan_record_contracts.hpp"
#include "../core/port_configuration_internal.hpp"
#include "../core/tensor_field_configuration_internal.hpp"
#include "detail/checked_arithmetic.hpp"
#include "lifecycle_plan_lowering.hpp"
#include "profile_admission.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

constexpr std::uint32_t kRuntimeEventsPerAcceptedUpdate = 2U;
constexpr std::uint32_t kLifecycleEventsPerAcceptedUpdate = 8U;

[[nodiscard]] std::vector<ParamEnumChoiceRecord> own_enum_choices(const ParamEnumChoices& choices) {
  std::vector<ParamEnumChoiceRecord> result;
  result.reserve(choices.count);
  for (std::size_t index = 0; index < choices.count; ++index) {
    const ParamEnumChoice& choice = choices.values[index];
    result.push_back(
        ParamEnumChoiceRecord{.name = std::string(choice.name), .value = choice.value});
  }
  return result;
}

void append_diagnostic(ParameterPlanLowering& result, std::string_view token,
                       DiagnosticSource source = {}) {
  const auto duplicate = std::find_if(
      result.diagnostics.begin(), result.diagnostics.end(), [&](const auto& diagnostic) {
        return diagnostic.token == token && diagnostic.source.vertex == source.vertex;
      });
  if (duplicate == result.diagnostics.end()) {
    result.diagnostics.push_back(
        ParameterPlanDiagnostic{.token = std::string(token), .source = source});
  }
}

[[nodiscard]] bool supported_scalar_type(ParamLogicalType type) noexcept {
  switch (type) {
    case ParamLogicalType::kBool:
    case ParamLogicalType::kSignedInteger:
    case ParamLogicalType::kUnsignedInteger:
    case ParamLogicalType::kFloatingPoint:
    case ParamLogicalType::kDuration:
      return true;
    case ParamLogicalType::kString:
    case ParamLogicalType::kSequence:
    case ParamLogicalType::kFilesystemPath:
    case ParamLogicalType::kObject:
      return false;
  }
  return false;
}

[[nodiscard]] bool supported_scalar_field(const ParamFieldDescriptor& field) noexcept {
  if (field.type_shape.optional || !supported_scalar_type(field.logical_type) ||
      !field.type_shape.has_stable_shape) {
    return false;
  }
  if (field.logical_type != ParamLogicalType::kDuration) {
    return true;
  }
  return (field.type_shape.element_or_rep_type == ParamLogicalType::kSignedInteger ||
          field.type_shape.element_or_rep_type == ParamLogicalType::kUnsignedInteger) &&
         field.type_shape.element_or_rep_bit_width > 0U &&
         field.type_shape.element_or_rep_bit_width <= 64U;
}

[[nodiscard]] bool read_scalar(const ParamFieldDescriptor& field, const void* aggregate,
                               ParamScalarValue& result) noexcept {
  return field.local_operations.read_member_as_scalar != nullptr &&
         field.local_operations.read_member_as_scalar(aggregate, result) &&
         canonicalize_parameter_scalar(result);
}

[[nodiscard]] ParamValueRouteDisposition classify_value_route(
    const ParamFieldDescriptor& field) noexcept {
  return supported_scalar_field(field) ? ParamValueRouteDisposition::kPortableScalar
                                       : ParamValueRouteDisposition::kLocalTypedOnly;
}

[[nodiscard]] bool profile_allows_active_mutation(std::optional<Profile> profile) noexcept {
  return profile && parameter_mutation_profile_disposition(*profile) ==
                        ParameterMutationProfileDisposition::kRuntimeAndLifecycle;
}

[[nodiscard]] ParameterFieldUpdateDisposition classify_field(const ParamFieldDescriptor& field,
                                                             bool runtime_requested,
                                                             bool lifecycle_requested,
                                                             bool profile_supported) noexcept {
  switch (field.mutability) {
    case ParamMutability::kCompileTime:
      return ParameterFieldUpdateDisposition::kCompileTime;
    case ParamMutability::kRecompile:
      return ParameterFieldUpdateDisposition::kRecompileRequired;
    case ParamMutability::kRuntime:
      if (!runtime_requested) {
        return ParameterFieldUpdateDisposition::kPolicyDisabled;
      }
      break;
    case ParamMutability::kLifecycle:
      if (!lifecycle_requested) {
        return ParameterFieldUpdateDisposition::kLifecycleDeferred;
      }
      break;
  }
  if (!profile_supported) {
    return ParameterFieldUpdateDisposition::kProfileDisallowed;
  }
  if (!supported_scalar_field(field)) {
    return ParameterFieldUpdateDisposition::kUnsupportedLogicalType;
  }
  if (field.local_operations.replace_member_from_staged == nullptr) {
    return ParameterFieldUpdateDisposition::kReplacementUnavailable;
  }
  return ParameterFieldUpdateDisposition::kAdmitted;
}

[[nodiscard]] bool supports_fixed_scalar_generation(const ParamTypeDescriptor& type,
                                                    const ParamSchemaDescriptor* schema) noexcept {
  if (schema == nullptr || schema->coverage != ParamSchemaCoverage::kComplete ||
      !type.copy_construct_is_nothrow || !type.trivially_copyable || !type.destroy_is_nothrow ||
      type.object_alignment == 0U || (type.object_alignment & (type.object_alignment - 1U)) != 0U) {
    return false;
  }
  return std::all_of(schema->fields.begin(), schema->fields.end(), [](const auto& field) {
    return supported_scalar_field(field) && !field.storage.owns_dynamic_storage &&
           field.storage.fixed_object_extent && field.storage.supports_bounded_materialization;
  });
}

[[nodiscard]] bool lifecycle_capacity_satisfies(
    const LifecycleTransactionOptions& configured,
    const LifecycleTransactionCapacityValues& required) noexcept {
  return configured.max_pending_requests >= required.max_pending_requests &&
         configured.max_pending_transactions >= required.max_pending_transactions &&
         configured.max_receipts >= required.max_receipts &&
         configured.max_fenced_operators >= required.max_fenced_operators &&
         configured.max_candidate_generations >= required.max_candidate_generations &&
         configured.max_events >= required.max_events &&
         configured.max_audit_records >= required.max_audit_records &&
         configured.max_quiesce_records >= required.max_quiesce_records &&
         configured.max_rollback_records >= required.max_rollback_records;
}

[[nodiscard]] bool aligned_stride(const ParamTypeDescriptor& type, std::uint64_t& result) noexcept {
  const auto size = static_cast<std::uint64_t>(type.object_size);
  const auto alignment = static_cast<std::uint64_t>(type.object_alignment);
  if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
    return false;
  }
  std::uint64_t with_padding{};
  if (!checked_add(size, alignment - 1U, with_padding)) {
    return false;
  }
  result = (with_padding / alignment) * alignment;
  return true;
}

[[nodiscard]] bool profile_allows_local_typed_materialization(Profile profile) noexcept {
  return profile == Profile::kStandard || profile == Profile::kResearch ||
         profile == Profile::kRobotics;
}

[[nodiscard]] bool profile_requires_complete_proven_schema(Profile profile) noexcept {
  return profile == Profile::kRealTime || profile == Profile::kRegulated;
}

[[nodiscard]] ParameterSchemaAdmissionDisposition schema_disposition(
    bool no_params, const ParamSchemaDescriptor* schema, std::optional<Profile> profile,
    bool require_proven) noexcept {
  if (!profile) {
    return ParameterSchemaAdmissionDisposition::kRejectedByProfile;
  }
  const bool complete_proven =
      no_params || (schema != nullptr && schema->coverage == ParamSchemaCoverage::kComplete &&
                    schema->proof == ParamSchemaProof::kProven);
  if ((profile_requires_complete_proven_schema(*profile) || require_proven) && !complete_proven) {
    return ParameterSchemaAdmissionDisposition::kRejectedByProfile;
  }
  if (no_params) {
    return ParameterSchemaAdmissionDisposition::kCompleteProven;
  }
  if (schema == nullptr) {
    return ParameterSchemaAdmissionDisposition::kOpaqueLocal;
  }
  if (schema->coverage != ParamSchemaCoverage::kComplete) {
    return ParameterSchemaAdmissionDisposition::kDeclaredPartial;
  }
  return schema->proof == ParamSchemaProof::kProven
             ? ParameterSchemaAdmissionDisposition::kCompleteProven
             : ParameterSchemaAdmissionDisposition::kCompleteAuthorAsserted;
}

[[nodiscard]] ParameterMaterializationStrategy materialization_strategy(
    bool no_params, bool fixed_scalar, bool owns_dynamic_storage,
    ParameterSchemaAdmissionDisposition disposition, std::optional<Profile> profile,
    ParamStorageMode requested) noexcept {
  if (no_params) {
    return ParameterMaterializationStrategy::kStaticNoParams;
  }
  if (disposition == ParameterSchemaAdmissionDisposition::kRejectedByProfile || !profile) {
    return ParameterMaterializationStrategy::kRejected;
  }
  if (fixed_scalar) {
    return ParameterMaterializationStrategy::kFixedObject;
  }
  if (*profile == Profile::kRobotics && owns_dynamic_storage) {
    return ParameterMaterializationStrategy::kRejected;
  }
  if (requested == ParamStorageMode::kPreallocatedBounded ||
      requested == ParamStorageMode::kInlineOnly ||
      !profile_allows_local_typed_materialization(*profile)) {
    return ParameterMaterializationStrategy::kRejected;
  }
  if (requested == ParamStorageMode::kDynamicColdPath ||
      requested == ParamStorageMode::kFreezeBeforeArm ||
      requested == ParamStorageMode::kProfileDefault) {
    return ParameterMaterializationStrategy::kColdOwnedTyped;
  }
  return ParameterMaterializationStrategy::kRejected;
}

struct OperatorPathInfo {
  OperatorPath path;
  std::string rendered;
};

[[nodiscard]] std::optional<OperatorPathInfo> operator_path(const GraphSnapshotData& snapshot,
                                                            VertexId owner) {
  const VertexRecord* vertex = find_vertex(snapshot, owner);
  if (vertex == nullptr || vertex->kind != VertexKind::kOperator) {
    return std::nullopt;
  }
  const GraphRecord* graph = find_graph(snapshot, vertex->graph);
  if (graph == nullptr) {
    return std::nullopt;
  }
  std::vector<std::string> segments;
  if (graph->path.has_value()) {
    const auto& graph_path = graph->path.value();  // NOLINT(bugprone-unchecked-optional-access)
    for (const auto& segment : graph_path.segments()) {
      segments.push_back(segment);
    }
  }
  segments.push_back(vertex->name);
  auto path = OperatorPath::from_segments(segments);
  if (!path) {
    return std::nullopt;
  }
  auto rendered = canonical_operator_path(snapshot, *vertex);
  if (!rendered) {
    return std::nullopt;
  }
  return OperatorPathInfo{.path = std::move(path).value(), .rendered = std::move(*rendered)};
}

[[nodiscard]] ParameterGenerationZeroOrigin generation_zero_origin(
    ParameterContributionOrigin origin) noexcept {
  switch (origin) {
    case ParameterContributionOrigin::kCaptured:
      return ParameterGenerationZeroOrigin::kInputCaptured;
    case ParameterContributionOrigin::kAlias:
      return ParameterGenerationZeroOrigin::kInputAlias;
    case ParameterContributionOrigin::kMigration:
      return ParameterGenerationZeroOrigin::kInputMigration;
  }
  return ParameterGenerationZeroOrigin::kInputCaptured;
}

[[nodiscard]] bool payload_options_equal(const std::optional<PayloadOptionsRecord>& lhs,
                                         const std::optional<PayloadOptionsRecord>& rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  return !lhs || (schema_identity_exactly_equal(lhs->schema, rhs->schema) &&
                  lhs->max_inline_bytes == rhs->max_inline_bytes &&
                  lhs->max_tensor_byte_span == rhs->max_tensor_byte_span &&
                  lhs->interface_tensor_emit_policy == rhs->interface_tensor_emit_policy);
}

[[nodiscard]] bool port_setup_equal(const PortRecord& existing,
                                    const OperatorSpecCompilerAccess::PortMetadata& candidate,
                                    std::optional<GraphPortId> follows_input = std::nullopt) {
  const PortConfigurationRecord replayed = setup_port_configuration(candidate, follows_input);
  return existing.owner_kind == PortOwnerKind::kOperator &&
         existing.direction == candidate.direction && existing.name == candidate.name &&
         existing.optional == candidate.optional &&
         payload_identity_exactly_equal(existing.payload_identity, candidate.payload_identity) &&
         existing.descriptor_tensor_field_count == candidate.descriptor_tensor_field_count &&
         existing.setup.queue_depth == replayed.queue_depth &&
         existing.setup.max_emits_per_compute == replayed.max_emits_per_compute &&
         existing.setup.late_joiner_history_depth == replayed.late_joiner_history_depth &&
         existing.setup.rate_kind == replayed.rate_kind &&
         existing.setup.declared_nominal_rate == replayed.declared_nominal_rate &&
         existing.setup.declared_expected_rate == replayed.declared_expected_rate &&
         existing.setup.follows_input == replayed.follows_input &&
         existing.setup.follows_ratio == replayed.follows_ratio &&
         existing.setup.lifecycle_binding_stage == replayed.lifecycle_binding_stage &&
         existing.setup.declared_max_nominal_rate == replayed.declared_max_nominal_rate &&
         existing.setup.diagnostic_alias == replayed.diagnostic_alias &&
         tensor_port_layout_exactly_equal(existing.setup.tensor_layout, replayed.tensor_layout) &&
         existing.setup.tensor_contract.has_value() == replayed.tensor_contract.has_value() &&
         (!existing.setup.tensor_contract ||
          tensor_contract_equal(*existing.setup.tensor_contract, *replayed.tensor_contract)) &&
         existing.setup.memory_conversion == replayed.memory_conversion &&
         existing.setup.tensor_allocation.has_value() == replayed.tensor_allocation.has_value() &&
         (!existing.setup.tensor_allocation ||
          (existing.setup.tensor_allocation->max_rank == replayed.tensor_allocation->max_rank &&
           existing.setup.tensor_allocation->max_byte_span ==
               replayed.tensor_allocation->max_byte_span)) &&
         existing.setup.tensor_byte_span_expression == replayed.tensor_byte_span_expression &&
         existing.setup.tensor_memory_kind_expression == replayed.tensor_memory_kind_expression &&
         payload_options_equal(existing.setup.payload_options, replayed.payload_options) &&
         tensor_field_configuration_equal(existing.setup.tensor_fields, replayed.tensor_fields);
}

[[nodiscard]] bool replayed_ports_match(const GraphSnapshotData& snapshot, VertexId owner,
                                        const OperatorSpec& spec) {
  std::vector<const PortRecord*> expected;
  for (const auto& port : snapshot.ports) {
    if (port.owner == owner && port.owner_kind == PortOwnerKind::kOperator) {
      expected.push_back(&port);
    }
  }
  if (expected.size() != OperatorSpecCompilerAccess::port_count(spec)) {
    return false;
  }
  std::sort(expected.begin(), expected.end(), [](const auto* lhs, const auto* rhs) {
    return std::tuple{lhs->direction, lhs->name} < std::tuple{rhs->direction, rhs->name};
  });
  std::vector<OperatorSpecCompilerAccess::PortMetadata> replayed;
  replayed.reserve(expected.size());
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    replayed.push_back(OperatorSpecCompilerAccess::port(spec, index));
  }
  std::sort(replayed.begin(), replayed.end(), [](const auto& lhs, const auto& rhs) {
    return std::tuple{lhs.direction, lhs.name} < std::tuple{rhs.direction, rhs.name};
  });
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    std::optional<GraphPortId> follows_input;
    if (replayed[index].follows_input_declaration != nullptr) {
      const auto replayed_input =
          std::find_if(replayed.begin(), replayed.end(), [&](const auto& row) {
            return row.declaration == replayed[index].follows_input_declaration &&
                   row.direction == PortDirection::kInput;
          });
      if (replayed_input == replayed.end()) {
        return false;
      }
      const auto expected_input =
          std::find_if(expected.begin(), expected.end(), [&](const auto* row) {
            return row->direction == PortDirection::kInput && row->name == replayed_input->name;
          });
      if (expected_input == expected.end()) {
        return false;
      }
      follows_input = (*expected_input)->id;
    }
    if ((index != 0U && replayed[index - 1U].direction == replayed[index].direction &&
         replayed[index - 1U].name == replayed[index].name) ||
        !port_setup_equal(*expected[index], replayed[index], follows_input)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool replayed_sources_match(const GraphSnapshotData& snapshot, VertexId owner,
                                          const OperatorSpec& spec) {
  std::vector<const ReadinessSourceDeclarationRecord*> expected;
  for (const auto& source : snapshot.readiness_sources) {
    if (source.owner == owner) {
      expected.push_back(&source);
    }
  }
  std::sort(expected.begin(), expected.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->declaration_ordinal < rhs->declaration_ordinal;
  });
  if (expected.size() != OperatorSpecCompilerAccess::source_count(spec)) {
    return false;
  }
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    const auto replayed = OperatorSpecCompilerAccess::source(spec, index);
    const ReadinessWakeRequirement wake = replayed.kind == ReadinessSourceKind::kSpin
                                              ? ReadinessWakeRequirement::kBusyPollRequired
                                              : ReadinessWakeRequirement::kDeploymentSelected;
    if (expected[index]->declaration_ordinal != index || expected[index]->kind != replayed.kind ||
        expected[index]->name != replayed.name ||
        expected[index]->requested_capacity != replayed.capacity ||
        expected[index]->requested_sender_reference_capacity !=
            replayed.sender_reference_capacity ||
        expected[index]->requested_clock != replayed.clock ||
        expected[index]->wake_requirement != wake) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr LifecycleMethodId lifecycle_method_id(LifecycleStage stage) noexcept {
  return static_cast<LifecycleMethodId>(static_cast<std::uint8_t>(stage) + 1U);
}

[[nodiscard]] bool replayed_lifecycle_matches(const GraphSnapshotData& snapshot, VertexId owner,
                                              const OperatorSpec& spec,
                                              OperatorFactoryBase::LifecycleHookSet hooks) {
  using CallbackKey = std::tuple<LifecycleStage, bool, LifecycleMethodId, LifecycleCallbackOrigin>;
  std::vector<CallbackKey> expected_callbacks;
  for (const auto& callback : snapshot.lifecycle_callbacks) {
    if (callback.owner == owner) {
      expected_callbacks.emplace_back(
          callback.stage, callback.required, callback.method, callback.origin);
    }
  }
  std::vector<CallbackKey> replayed_callbacks;
  replayed_callbacks.reserve(OperatorSpecCompilerAccess::lifecycle_callback_count(spec) +
                             static_cast<std::size_t>(hooks.start) +
                             static_cast<std::size_t>(hooks.stop));
  for (std::size_t index = 0U; index < OperatorSpecCompilerAccess::lifecycle_callback_count(spec);
       ++index) {
    const auto callback = OperatorSpecCompilerAccess::lifecycle_callback(spec, index);
    replayed_callbacks.emplace_back(callback.stage,
                                    callback.required,
                                    callback.method,
                                    LifecycleCallbackOrigin::kAdvancedMember);
  }
  if (hooks.start) {
    replayed_callbacks.emplace_back(LifecycleStage::kStart,
                                    true,
                                    lifecycle_method_id(LifecycleStage::kStart),
                                    LifecycleCallbackOrigin::kLegacyStart);
  }
  if (hooks.stop) {
    replayed_callbacks.emplace_back(LifecycleStage::kStop,
                                    true,
                                    lifecycle_method_id(LifecycleStage::kStop),
                                    LifecycleCallbackOrigin::kLegacyStop);
  }
  std::sort(expected_callbacks.begin(), expected_callbacks.end());
  std::sort(replayed_callbacks.begin(), replayed_callbacks.end());
  if (expected_callbacks != replayed_callbacks) {
    return false;
  }

  using FactKey = std::tuple<LifecycleStage, LifecycleFactKind, std::string_view>;
  std::vector<FactKey> expected_facts;
  for (const auto& fact : snapshot.lifecycle_facts) {
    if (fact.owner == owner) {
      expected_facts.emplace_back(fact.stage, fact.kind, fact.stable_name);
    }
  }
  std::vector<FactKey> replayed_facts;
  replayed_facts.reserve(OperatorSpecCompilerAccess::lifecycle_fact_count(spec));
  for (std::size_t index = 0U; index < OperatorSpecCompilerAccess::lifecycle_fact_count(spec);
       ++index) {
    const auto fact = OperatorSpecCompilerAccess::lifecycle_fact(spec, index);
    const LifecycleFactKind kind =
        fact.kind == OperatorSpecCompilerAccess::LifecycleFactKind::kProduces
            ? LifecycleFactKind::kProduces
            : LifecycleFactKind::kRequires;
    replayed_facts.emplace_back(fact.stage, kind, fact.stable_name);
  }
  std::sort(expected_facts.begin(), expected_facts.end());
  std::sort(replayed_facts.begin(), replayed_facts.end());
  return expected_facts == replayed_facts;
}

[[nodiscard]] bool replayed_declarations_match(const GraphSnapshotData& snapshot, VertexId owner,
                                               const OperatorFactoryBase& factory) {
  const VertexRecord* vertex = find_vertex(snapshot, owner);
  if (vertex == nullptr || vertex->kind != VertexKind::kOperator) {
    return false;
  }
  auto created = factory.create(nullptr);
  if (!created || !created.value()) {
    return false;
  }
  OperatorSpec spec;
  try {
    created.value()->setup(spec);
  } catch (...) {
    return false;
  }
  if (OperatorSpecCompilerAccess::declaration_error(spec).code != ErrorCode::kSuccess) {
    return false;
  }
  return vertex->cuda_graph_capture_eligible ==
             OperatorSpecCompilerAccess::cuda_graph_capture_eligible(spec) &&
         replayed_ports_match(snapshot, owner, spec) &&
         replayed_sources_match(snapshot, owner, spec) &&
         replayed_lifecycle_matches(snapshot, owner, spec, factory.lifecycle_hooks());
}

[[nodiscard]] bool valid_config_source(const ParamConfigFieldProvenance& source,
                                       std::string_view target_path,
                                       std::string_view target_schema_id,
                                       ParamSchemaVersion target_schema_version,
                                       ParamFieldId target_field_id, ParamSchemaHash target_full_id,
                                       std::size_t expected_ordinal) noexcept {
  return source.document.semantic_version != 0U &&
         (!source.document.has_raw_source || source.document.raw_algorithm_version != 0U) &&
         (!source.document.has_base_raw_source ||
          source.document.base_raw_algorithm_version != 0U) &&
         !source.source_path.empty() && source.source_path.size() <= 512U &&
         source.source_ordinal == expected_ordinal && source.target_operator_path == target_path &&
         source.target_schema_id == target_schema_id &&
         source.target_schema_version == target_schema_version &&
         source.target_field_id == target_field_id && source.target_full_id == target_full_id;
}

[[nodiscard]] bool valid_config_source(const ParamConfigFieldProvenance& source,
                                       const ParameterTransactionOperatorState& target,
                                       const ParameterTransactionFieldState& contribution,
                                       std::size_t expected_ordinal) noexcept {
  const bool origin_matches = (contribution.origin == ParamFieldRebindOrigin::kCaptured &&
                               source.origin == ParamFieldRebindOrigin::kCaptured) ||
                              (contribution.origin == ParamFieldRebindOrigin::kAlias &&
                               source.origin == ParamFieldRebindOrigin::kAlias);
  // Attachment clears the transaction copy; compilation later stamps the final path.
  return origin_matches && valid_config_source(source,
                                               std::string_view{},
                                               target.schema_id,
                                               target.schema_version,
                                               contribution.field,
                                               contribution.full_id,
                                               expected_ordinal);
}

[[nodiscard]] bool collect_local_config_sources(
    const ParameterInputEvidenceState& evidence, const ParamTypeDescriptor& type,
    const ParamSchemaDescriptor& schema, std::string_view operator_path,
    std::vector<std::optional<ParamConfigFieldProvenance>>& sources) {
  if (evidence.local_type_token != type.local_type_token ||
      evidence.schema_id != schema.stable_id || evidence.schema_version != schema.version ||
      evidence.structural_hash != schema.structural_hash ||
      evidence.contract_hash != schema.contract_hash || evidence.config_fields.empty() ||
      evidence.config_fields.size() > 65'536U) {
    return false;
  }
  sources.resize(schema.fields.size());
  for (std::size_t source_index = 0; source_index < evidence.config_fields.size(); ++source_index) {
    const auto& source = evidence.config_fields[source_index];
    if (source.document.semantic_version == 0U ||
        (source.document.has_raw_source && source.document.raw_algorithm_version == 0U) ||
        (source.document.has_base_raw_source && source.document.base_raw_algorithm_version == 0U) ||
        source.source_path.empty() || source.source_path.size() > 512U ||
        source.source_ordinal != source_index || !source.target_operator_path.empty() ||
        source.target_schema_id != schema.stable_id ||
        source.target_schema_version != schema.version ||
        (source.origin != ParamFieldRebindOrigin::kCaptured &&
         source.origin != ParamFieldRebindOrigin::kAlias)) {
      return false;
    }
    const auto field = std::ranges::find_if(schema.fields, [&](const auto& candidate) {
      return candidate.id == source.target_field_id && candidate.full_id == source.target_full_id;
    });
    if (field == schema.fields.end()) {
      return false;
    }
    const std::size_t field_index = static_cast<std::size_t>(field - schema.fields.begin());
    if (sources[field_index]) {
      return false;
    }
    sources[field_index] = source;
    sources[field_index]->target_operator_path = operator_path;
  }
  return true;
}

class MutableAggregateCopy {
 public:
  MutableAggregateCopy(const ParamTypeDescriptor& type, const void* source, bool use_defaults)
      : type_(&type) {
    value_ = ::operator new(type.object_size, std::align_val_t{type.object_alignment});
    try {
      if (use_defaults) {
        type.default_construct(value_);
      } else {
        type.copy_construct(value_, source);
      }
    } catch (...) {
      ::operator delete(value_, std::align_val_t{type.object_alignment});
      value_ = nullptr;
      throw;
    }
  }
  MutableAggregateCopy(const MutableAggregateCopy&) = delete;
  MutableAggregateCopy& operator=(const MutableAggregateCopy&) = delete;
  ~MutableAggregateCopy() {
    if (value_ != nullptr) {
      type_->destroy(value_);
      ::operator delete(value_, std::align_val_t{type_->object_alignment});
    }
  }
  [[nodiscard]] void* get() noexcept { return value_; }

 private:
  const ParamTypeDescriptor* type_{};
  void* value_{};
};

class StagedFieldCopy {
 public:
  StagedFieldCopy(const ParamFieldLocalOperations& operations, const void* staged)
      : operations_(&operations) {
    value_ = ::operator new(operations.object_size, std::align_val_t{operations.object_alignment});
    try {
      operations.copy_construct_from_staged(value_, staged);
      constructed_ = true;
    } catch (...) {
      ::operator delete(value_, std::align_val_t{operations.object_alignment});
      value_ = nullptr;
      throw;
    }
  }
  StagedFieldCopy(const StagedFieldCopy&) = delete;
  StagedFieldCopy& operator=(const StagedFieldCopy&) = delete;
  ~StagedFieldCopy() {
    if (constructed_) {
      operations_->destroy_staged(value_);
    }
    if (value_ != nullptr) {
      ::operator delete(value_, std::align_val_t{operations_->object_alignment});
    }
  }
  [[nodiscard]] void* get() noexcept { return value_; }

 private:
  const ParamFieldLocalOperations* operations_{};
  void* value_{};
  bool constructed_{};
};

[[nodiscard]] bool transaction_scope_targets(const GraphSnapshotData& snapshot,
                                             const ParameterTransactionState& transaction,
                                             const ExecutableFactoryInventory& factories,
                                             std::vector<VertexId>& selected) {
  const auto& scope = transaction.scope;
  const bool graph_scope = scope.kind == ParamRebindTargetScopeKind::kWholeGraph ||
                           scope.kind == ParamRebindTargetScopeKind::kGraph;
  if (graph_scope != scope.graph.has_value() || (graph_scope && !scope.operators.empty()) ||
      (!graph_scope && scope.operators.empty())) {
    return false;
  }
  selected.reserve(factories.operators.size());
  for (const auto& executable : factories.operators) {
    const auto vertex = std::ranges::find(snapshot.vertices, executable.vertex, &VertexRecord::id);
    if (vertex == snapshot.vertices.end())
      return false;
    const bool included = graph_scope ? is_descendant_or_self(snapshot, vertex->graph, *scope.graph)
                                      : std::ranges::find(scope.operators, executable.vertex) !=
                                            scope.operators.end();
    if (included)
      selected.push_back(executable.vertex);
  }
  return !selected.empty();
}

// Apply one graph-owned sparse transaction over the final authored factory baselines.
[[nodiscard]] bool apply_parameter_transaction(
    const GraphSnapshotData& snapshot, ExecutableFactoryInventory& factories,
    std::span<const std::optional<OperatorPathInfo>> paths,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    std::vector<std::vector<ParameterContributionRecord>>& accepted_contributions,
    ParameterPlanLowering& result) {
  const auto& transaction = factories.parameter_transaction;
  if (!transaction) {
    return true;
  }
  const bool sparse = transaction->provenance.mode == ParamRebindMode::kSparseContributions;
  const bool complete = transaction->provenance.mode == ParamRebindMode::kCompleteObject;
  const bool config_source = transaction->source == ParameterTransactionSource::kConfig;
  const bool snapshot_source = transaction->source == ParameterTransactionSource::kSnapshot;
  if ((!sparse && !complete) || transaction->operators.empty() ||
      (config_source && (!sparse || !transaction->provenance.fields.empty())) ||
      (snapshot_source && !transaction->provenance.config_fields.empty())) {
    append_diagnostic(result, "PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE");
    return false;
  }

  std::vector<VertexId> selected_scope_targets;
  if (snapshot_source &&
      !transaction_scope_targets(snapshot, *transaction, factories, selected_scope_targets)) {
    append_diagnostic(result, "PARAMETER_REBIND_SCOPE_INVALID");
    return false;
  }
  if (snapshot_source) {
    const bool staged_outside_scope =
        std::ranges::any_of(transaction->operators, [&](const auto& op) {
          return std::ranges::find(selected_scope_targets, op.target) ==
                 selected_scope_targets.end();
        });
    const bool incomplete =
        complete &&
        (selected_scope_targets.size() != transaction->operators.size() ||
         std::ranges::any_of(selected_scope_targets, [&](VertexId target) {
           return std::ranges::find(
                      transaction->operators, target, &ParameterTransactionOperatorState::target) ==
                  transaction->operators.end();
         }));
    if (staged_outside_scope || incomplete) {
      append_diagnostic(
          result,
          incomplete ? "PARAMETER_REBIND_SCOPE_INCOMPLETE" : "PARAMETER_REBIND_SCOPE_INVALID");
      return false;
    }
  }

  bool valid = true;
  std::vector<bool> accepted_config_ordinals(transaction->provenance.config_fields.size());
  std::size_t accepted_config_ordinal_count{};
  std::vector<bool> accepted_rebind_ordinals(transaction->provenance.fields.size());
  std::size_t accepted_rebind_ordinal_count{};
  std::vector<std::shared_ptr<OperatorFactoryBase>> replacements(factories.operators.size());
  std::vector<std::vector<ParameterContributionRecord>> contributions(factories.operators.size());
  std::vector<VertexId> seen_targets;
  seen_targets.reserve(transaction->operators.size());
  std::size_t next_config_provenance{};
  std::size_t next_rebind_provenance{};
  for (const ParameterTransactionOperatorState& staged : transaction->operators) {
    const auto executable = std::ranges::find(
        factories.operators, staged.target, &ExecutableOperatorFactoryRecord::vertex);
    if (std::ranges::find(seen_targets, staged.target) != seen_targets.end()) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_DUPLICATE_TARGET");
      valid = false;
      continue;
    }
    seen_targets.push_back(staged.target);
    if (executable == factories.operators.end()) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_TARGET_UNKNOWN");
      valid = false;
      continue;
    }
    const std::size_t index = static_cast<std::size_t>(executable - factories.operators.begin());
    const DiagnosticSource source{.vertex = executable->vertex};
    if (!executable->factory || !paths[index]) {
      append_diagnostic(result, "PARAMETER_FACTORY_INVENTORY_INVALID", source);
      valid = false;
      continue;
    }
    if (snapshot_source &&
        (!staged.required_source_profile ||
         effective_profile_for_vertex(effective_vertex_profiles, executable->vertex) !=
             staged.required_source_profile)) {
      append_diagnostic(result, "PARAMETER_REBIND_PROFILE_MISMATCH", source);
      valid = false;
      continue;
    }
    if (replacements[index]) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_DUPLICATE_TARGET", source);
      valid = false;
      continue;
    }
    const ParamTypeDescriptor& type = executable->factory->parameter_type();
    const ParamSchemaDescriptor* const schema = executable->factory->parameter_schema();
    if (schema == nullptr || staged.local_type_token != type.local_type_token ||
        schema->local_type != &type || staged.schema_id != schema->stable_id ||
        staged.schema_version != schema->version ||
        staged.structural_hash != schema->structural_hash ||
        staged.contract_hash != schema->contract_hash) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_SCHEMA_MISMATCH", source);
      valid = false;
      continue;
    }
    if (type.validate_aggregate == nullptr) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_VALUE_INVALID", source);
      valid = false;
      continue;
    }
    const std::size_t provenance_size = config_source ? transaction->provenance.config_fields.size()
                                                      : transaction->provenance.fields.size();
    std::size_t& next_provenance = config_source ? next_config_provenance : next_rebind_provenance;
    // Producers emit one contiguous provenance range per target in transaction operator order.
    if (next_provenance > provenance_size || staged.provenance_begin != next_provenance ||
        staged.provenance_count > provenance_size - next_provenance) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH", source);
      valid = false;
      continue;
    }
    const std::size_t provenance_end = next_provenance + staged.provenance_count;
    next_provenance = provenance_end;

    try {
      MutableAggregateCopy candidate{
          type, executable->factory->initial_parameter_value(), complete};
      std::vector<ParamFieldId> seen_fields;
      seen_fields.reserve(staged.fields.size());
      bool fields_valid = true;
      for (const ParameterTransactionFieldState& contribution : staged.fields) {
        const auto field = std::ranges::find_if(schema->fields, [&](const auto& descriptor) {
          return descriptor.id == contribution.field && descriptor.full_id == contribution.full_id;
        });
        const bool duplicate =
            std::ranges::find(seen_fields, contribution.field) != seen_fields.end();
        if (field == schema->fields.end() || duplicate ||
            (config_source && contribution.origin == ParamFieldRebindOrigin::kAlias &&
             field->alias.empty())) {
          append_diagnostic(result, "PARAMETER_OVERRIDE_FIELD_INVALID", source);
          valid = false;
          fields_valid = false;
          break;
        }
        const ParamFieldLocalOperations& operations = field->local_operations;
        if (contribution.config_provenance_index.has_value() ==
            contribution.rebind_provenance_index.has_value()) {
          append_diagnostic(result, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH", source);
          valid = false;
          fields_valid = false;
          break;
        }
        if (contribution.config_provenance_index) {
          const std::size_t source_index = *contribution.config_provenance_index;
          if (source_index < staged.provenance_begin || source_index >= provenance_end ||
              source_index >= transaction->provenance.config_fields.size() ||
              accepted_config_ordinals[source_index] ||
              !valid_config_source(transaction->provenance.config_fields[source_index],
                                   staged,
                                   contribution,
                                   source_index)) {
            append_diagnostic(result, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH", source);
            valid = false;
            fields_valid = false;
            break;
          }
          accepted_config_ordinals[source_index] = true;
          ++accepted_config_ordinal_count;
        } else {
          const std::size_t source_index = *contribution.rebind_provenance_index;
          if (source_index < staged.provenance_begin || source_index >= provenance_end ||
              source_index >= transaction->provenance.fields.size() ||
              accepted_rebind_ordinals[source_index]) {
            append_diagnostic(result, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH", source);
            valid = false;
            fields_valid = false;
            break;
          }
          const ParamRebindFieldProvenance& evidence = transaction->provenance.fields[source_index];
          if (!evidence.target_operator_path.empty() ||
              evidence.target_schema_id != staged.schema_id ||
              evidence.target_schema_version != staged.schema_version ||
              evidence.target_field_id != contribution.field ||
              evidence.target_full_id != contribution.full_id ||
              evidence.origin != contribution.origin ||
              evidence.disposition != ParamFieldRebindDisposition::kContributed) {
            append_diagnostic(result, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH", source);
            valid = false;
            fields_valid = false;
            break;
          }
          accepted_rebind_ordinals[source_index] = true;
          ++accepted_rebind_ordinal_count;
        }
        if (contribution.scalar_value) {
          if (contribution.local_value || contribution.local_value_digest != nullptr ||
              operations.replace_member_from_scalar == nullptr ||
              contribution.staged_value_digest !=
                  parameter_transaction_scalar_digest(*contribution.scalar_value) ||
              !operations.replace_member_from_scalar(candidate.get(), *contribution.scalar_value)) {
            append_diagnostic(result, "PARAMETER_OVERRIDE_VALUE_INVALID", source);
            valid = false;
            fields_valid = false;
            break;
          }
        } else {
          ParamSchemaHash observed_digest{};
          if (contribution.staged_type_token != operations.local_field_type_token ||
              !contribution.local_value || contribution.local_value_digest == nullptr ||
              operations.copy_construct_from_staged == nullptr ||
              operations.destroy_staged == nullptr ||
              operations.replace_member_from_staged == nullptr || operations.object_size == 0U ||
              operations.object_alignment == 0U ||
              !contribution.local_value_digest(contribution.local_value.get(), observed_digest) ||
              observed_digest != contribution.staged_value_digest) {
            append_diagnostic(result, "PARAMETER_OVERRIDE_VALUE_INVALID", source);
            valid = false;
            fields_valid = false;
            break;
          }
          StagedFieldCopy field_value{operations, contribution.local_value.get()};
          operations.replace_member_from_staged(candidate.get(), field_value.get());
        }
        seen_fields.push_back(contribution.field);
        contributions[index].push_back(ParameterContributionRecord{
            .field = contribution.field,
            .full_id = contribution.full_id,
            .origin = contribution.origin == ParamFieldRebindOrigin::kAlias
                          ? ParameterContributionOrigin::kAlias
                          : ParameterContributionOrigin::kCaptured,
            .config_provenance_index = contribution.config_provenance_index,
        });
      }
      if (!fields_valid) {
        continue;
      }
      if (!type.validate_aggregate(candidate.get()).ok()) {
        append_diagnostic(result, "PARAMETER_OVERRIDE_VALUE_INVALID", source);
        valid = false;
        continue;
      }
      auto cloned = executable->factory->clone_with_params(candidate.get(), type.local_type_token);
      if (!cloned || !cloned.value()) {
        append_diagnostic(result, "PARAMETER_OVERRIDE_VALUE_INVALID", source);
        valid = false;
        continue;
      }
      if (!replayed_declarations_match(snapshot, executable->vertex, *cloned.value())) {
        append_diagnostic(result, "PARAMETER_OVERRIDE_DECLARATION_MISMATCH", source);
        valid = false;
        continue;
      }
      replacements[index] = std::move(cloned).value();
    } catch (...) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_VALUE_INVALID", source);
      valid = false;
    }
  }
  const std::size_t expected_rebind_contributions =
      static_cast<std::size_t>(std::ranges::count(transaction->provenance.fields,
                                                  ParamFieldRebindDisposition::kContributed,
                                                  &ParamRebindFieldProvenance::disposition));
  if (!valid || next_config_provenance != transaction->provenance.config_fields.size() ||
      next_rebind_provenance != transaction->provenance.fields.size() ||
      accepted_config_ordinal_count != transaction->provenance.config_fields.size() ||
      accepted_rebind_ordinal_count != expected_rebind_contributions) {
    if (valid) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH");
    }
    return false;
  }
  for (std::size_t index = 0; index < replacements.size(); ++index) {
    if (replacements[index]) {
      factories.operators[index].factory = std::move(replacements[index]);
    }
  }
  accepted_contributions = std::move(contributions);
  return true;
}

void rebuild_parameter_catalog(ParameterPlanLowering& result) {
  const auto operator_records = result.catalog.operators();
  const auto field_records = result.catalog.fields();
  if (operator_records.empty() && field_records.empty()) {
    return;
  }

  std::vector<OperatorParamRecord> operators{operator_records.begin(), operator_records.end()};
  std::vector<ParamFieldCatalogRecord> fields{field_records.begin(), field_records.end()};
  for (auto& catalog_field : fields) {
    const auto plan_field =
        std::find_if(result.fields.begin(), result.fields.end(), [&](const auto& candidate) {
          return candidate.owner == catalog_field.address.operator_instance &&
                 candidate.field == catalog_field.address.field &&
                 candidate.full_id == catalog_field.full_id;
        });
    if (plan_field != result.fields.end()) {
      catalog_field.update_disposition = plan_field->update_disposition;
    }
  }
  const ParamCatalogDigest digest = parameter_catalog_digest(operators, fields);
  result.catalog = ParameterCatalogAccess::make(digest, std::move(operators), std::move(fields));
}

void disable_admitted_fields(ParameterPlanLowering& result) {
  for (auto& field : result.fields) {
    if (field.update_disposition == ParameterFieldUpdateDisposition::kAdmitted) {
      field.update_disposition = ParameterFieldUpdateDisposition::kCapacityUnavailable;
    }
  }
  for (auto& op : result.operators) {
    op.runtime_updates_enabled = false;
    op.lifecycle_updates_enabled = false;
  }
  result.lifecycle_mutable_fields.clear();
  result.capacity.enabled_operator_count = 0U;
  result.capacity.compute_ingress_capacity = 0U;
  result.capacity.max_compute_patches_per_request = 0U;
  result.capacity.parameter_arena_bytes = 0U;
  result.capacity.max_arena_alignment = 0U;
  result.capacity.runtime_updates_enabled = false;
  result.capacity.lifecycle_updates_enabled = false;
  rebuild_parameter_catalog(result);
}

}  // namespace

// Keep Parameter admission and all derived record families in one lowering transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
ParameterPlanLowering lower_parameter_plan(
    const GraphSnapshotData& snapshot, ExecutableFactoryInventory& factories,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    PlanGenerationId plan_generation, const ParameterCompileOptions& options,
    const std::optional<LifecycleTransactionOptions>& lifecycle_options) {
  ParameterPlanLowering result;
  result.operators.reserve(factories.operators.size());

  bool valid = true;
  const bool runtime_requested =
      options.runtime_updates == ParamRuntimeUpdatePolicy::kEnabledIfAdmitted ||
      options.runtime_updates == ParamRuntimeUpdatePolicy::kRequirePreallocated;
  const bool runtime_disabled =
      options.runtime_updates == ParamRuntimeUpdatePolicy::kProfileDefault ||
      options.runtime_updates == ParamRuntimeUpdatePolicy::kDisabled;
  if (!runtime_requested && !runtime_disabled) {
    append_diagnostic(result, "PARAMETER_POLICY_INVALID");
    valid = false;
  }
  result.capacity.resolved_update_policy =
      runtime_requested ? options.runtime_updates : ParamRuntimeUpdatePolicy::kDisabled;

  if (options.exhaustion != ParamExhaustionPolicy::kRejectUpdate) {
    append_diagnostic(result, "PARAMETER_POLICY_INVALID");
    valid = false;
  }
  if (runtime_requested && options.storage != ParamStorageMode::kProfileDefault &&
      options.storage != ParamStorageMode::kPreallocatedBounded &&
      options.storage != ParamStorageMode::kInlineOnly) {
    append_diagnostic(result, "PARAMETER_UPDATE_STORAGE_UNSUPPORTED");
    valid = false;
  }

  if (factories.operators.size() > std::numeric_limits<OperatorInstanceId>::max()) {
    append_diagnostic(result, "PARAMETER_FACTORY_INVENTORY_INVALID");
    valid = false;
  }

  std::vector<std::optional<OperatorPathInfo>> operator_paths;
  std::vector<ParameterGenerationZeroOrigin> operator_baseline_origins;
  std::vector<ParameterGenerationZeroOrigin> field_baseline_origins;
  std::vector<std::vector<std::optional<ParamConfigFieldProvenance>>> local_config_sources(
      factories.operators.size());
  operator_paths.reserve(factories.operators.size());
  operator_baseline_origins.reserve(factories.operators.size());
  field_baseline_origins.reserve(factories.operators.size());
  for (const auto& executable : factories.operators) {
    operator_paths.push_back(operator_path(snapshot, executable.vertex));
    if (!executable.factory) {
      operator_baseline_origins.push_back(ParameterGenerationZeroOrigin::kDefault);
      field_baseline_origins.push_back(ParameterGenerationZeroOrigin::kDefault);
      continue;
    }
    operator_baseline_origins.push_back(executable.factory->has_explicit_parameter_baseline()
                                            ? ParameterGenerationZeroOrigin::kTypedBaseline
                                            : ParameterGenerationZeroOrigin::kDefault);
    field_baseline_origins.push_back(executable.parameter_input_evidence
                                         ? (executable.parameter_input_evidence->explicit_baseline
                                                ? ParameterGenerationZeroOrigin::kTypedBaseline
                                                : ParameterGenerationZeroOrigin::kDefault)
                                         : (executable.factory->has_explicit_parameter_baseline()
                                                ? ParameterGenerationZeroOrigin::kTypedBaseline
                                                : ParameterGenerationZeroOrigin::kDefault));
  }
  std::vector<std::vector<ParameterContributionRecord>> accepted_contributions(
      factories.operators.size());
  if (!apply_parameter_transaction(snapshot,
                                   factories,
                                   operator_paths,
                                   effective_vertex_profiles,
                                   accepted_contributions,
                                   result)) {
    valid = false;
  }

  std::vector<OperatorParamRecord> catalog_operators;
  std::vector<ParamFieldCatalogRecord> catalog_fields;
  catalog_operators.reserve(factories.operators.size());
  std::vector<ParameterInstanceProjectionIdentity> instance_identities;

  std::uint32_t runtime_operator_count{};
  std::uint32_t lifecycle_operator_count{};
  std::uint32_t max_runtime_fields{};
  for (std::size_t factory_index = 0; factory_index < factories.operators.size(); ++factory_index) {
    const auto& executable = factories.operators[factory_index];
    const DiagnosticSource source{.vertex = executable.vertex};
    const OperatorInstanceId expected_id = static_cast<OperatorInstanceId>(factory_index + 1U);
    if (!executable.factory || executable.operator_id != expected_id ||
        executable.vertex.value == 0U || executable.vertex.value > snapshot.vertices.size()) {
      append_diagnostic(result, "PARAMETER_FACTORY_INVENTORY_INVALID", source);
      valid = false;
      continue;
    }

    const OperatorRef owner{
        .id = executable.operator_id,
        .plan_generation = plan_generation,
        .run_generation = 0U,
    };
    const auto profile = effective_profile_for_vertex(effective_vertex_profiles, executable.vertex);
    const bool profile_supported = profile_allows_active_mutation(profile);
    if (runtime_requested && !profile_supported) {
      append_diagnostic(result, "PARAMETER_UPDATE_PROFILE_UNSUPPORTED", source);
      valid = false;
    }

    const ParamTypeDescriptor& type = executable.factory->parameter_type();
    const ParamSchemaDescriptor* const schema = executable.factory->parameter_schema();
    const bool no_params = executable.factory->has_no_params();
    const auto& path = operator_paths[factory_index];
    const bool complete_transaction_target =
        factories.parameter_transaction &&
        factories.parameter_transaction->provenance.mode == ParamRebindMode::kCompleteObject &&
        std::any_of(factories.parameter_transaction->operators.begin(),
                    factories.parameter_transaction->operators.end(),
                    [&](const auto& target) { return target.target == executable.vertex; });
    const auto schema_admission =
        schema_disposition(no_params, schema, profile, options.require_proven_schema_completeness);
    const bool fixed_scalar_generation = supports_fixed_scalar_generation(type, schema);
    const auto materialization = materialization_strategy(no_params,
                                                          fixed_scalar_generation,
                                                          type.storage.owns_dynamic_storage,
                                                          schema_admission,
                                                          profile,
                                                          options.storage);
    OperatorParameterPlanRecord op{
        .owner = owner,
        .operator_path = path ? path->rendered : std::string{},
        .effective_profile = profile,
        .initial_generation = 0U,
        .no_params = no_params,
        .schema_availability = schema == nullptr ? ParamSchemaAvailability::kOpaqueLocal
                                                 : ParamSchemaAvailability::kDeclared,
        .schema_disposition = schema_admission,
        .materialization = materialization,
        .object_size = static_cast<std::uint64_t>(type.object_size),
        .object_alignment = static_cast<std::uint64_t>(type.object_alignment),
        .storage = type.storage,
    };
    if (!path || op.operator_path.empty() || (schema != nullptr && schema->local_type != &type)) {
      append_diagnostic(result, "PARAMETER_FACTORY_INVENTORY_INVALID", source);
      valid = false;
    }
    if (schema != nullptr) {
      op.schema_id = std::string(schema->stable_id);
      op.schema_version = schema->version;
      op.structural_hash = schema->structural_hash;
      op.contract_hash = schema->contract_hash;
      op.metadata_hash = schema->metadata_hash;
      op.schema_coverage = schema->coverage;
      op.schema_proof = schema->proof;
    }
    if (executable.parameter_input_evidence &&
        (schema == nullptr || !path ||
         !collect_local_config_sources(*executable.parameter_input_evidence,
                                       type,
                                       *schema,
                                       path->rendered,
                                       local_config_sources[factory_index]))) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH", source);
      valid = false;
    }
    op.fixed_scalar_generation_strategy = fixed_scalar_generation;

    const bool explicit_policies_required =
        options.require_explicit_field_policies || (profile && *profile == Profile::kRegulated);
    if (schema != nullptr && explicit_policies_required &&
        std::any_of(schema->fields.begin(), schema->fields.end(), [](const auto& field) {
          return !field.policy_evidence.complete();
        })) {
      append_diagnostic(result, "PARAMETER_EXPLICIT_FIELD_POLICY_UNPROVEN", source);
      valid = false;
    }

    if (schema_admission == ParameterSchemaAdmissionDisposition::kRejectedByProfile) {
      append_diagnostic(result, "PARAMETER_SCHEMA_INCOMPLETE", source);
      valid = false;
    }
    if (materialization == ParameterMaterializationStrategy::kRejected) {
      append_diagnostic(result, "PARAMETER_UPDATE_STORAGE_UNSUPPORTED", source);
      valid = false;
    }

    const auto final_digest = parameter_value_digest(
        type, schema, executable.factory->initial_parameter_value(), no_params);
    if (type.validate_aggregate == nullptr ||
        !type.validate_aggregate(executable.factory->initial_parameter_value()).ok()) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_VALUE_INVALID", source);
      valid = false;
    }
    result.generation_zero_operators.push_back(ParameterGenerationZeroOperatorRecord{
        .owner = owner,
        .baseline_origin = operator_baseline_origins[factory_index],
        .value_digest = final_digest,
    });
    if (path) {
      const auto duplicate_path =
          std::find_if(catalog_operators.begin(), catalog_operators.end(), [&](const auto& item) {
            return item.path == path->path;
          });
      if (duplicate_path != catalog_operators.end()) {
        append_diagnostic(result, "PARAMETER_INSTANCE_ID_COLLISION", source);
        valid = false;
      }
      catalog_operators.push_back(OperatorParamRecord{
          .operator_instance = owner,
          .path = path->path,
          .effective_profile = profile,
          .schema_availability = op.schema_availability,
          .schema_disposition = schema_admission,
          .materialization = materialization,
          .schema_id = op.schema_id,
          .schema_version = op.schema_version,
          .structural_hash = op.structural_hash,
          .contract_hash = op.contract_hash,
          .metadata_hash = op.metadata_hash,
          .schema_coverage = op.schema_coverage,
          .schema_proof = op.schema_proof,
      });
    }

    const std::size_t field_begin = result.fields.size();
    if (schema != nullptr) {
      std::vector<ParameterFieldProjectionIdentity> identities;
      identities.reserve(schema->fields.size());
      for (std::size_t descriptor_index = 0; descriptor_index < schema->fields.size();
           ++descriptor_index) {
        const ParamFieldDescriptor& descriptor = schema->fields[descriptor_index];
        const ParameterProjectionCollision field_collision =
            classify_parameter_field_projection(identities, descriptor.id, descriptor.full_id);
        if (field_collision != ParameterProjectionCollision::kNone) {
          append_diagnostic(result, parameter_projection_collision_token(field_collision), source);
          valid = false;
        } else {
          identities.emplace_back(descriptor.id, descriptor.full_id);
        }
        const ParamSchemaHash instance_identity =
            path ? parameter_instance_identity(path->path, descriptor.full_id) : ParamSchemaHash{};
        const ParamInstanceId instance = compact_parameter_instance_id(instance_identity);
        const ParameterProjectionCollision instance_collision =
            classify_parameter_instance_projection(
                instance_identities, instance, instance_identity);
        if (instance_collision != ParameterProjectionCollision::kNone) {
          append_diagnostic(
              result, parameter_projection_collision_token(instance_collision), source);
          valid = false;
        } else {
          instance_identities.emplace_back(instance, instance_identity);
        }
        const auto disposition = classify_field(
            descriptor, runtime_requested, lifecycle_options.has_value(), profile_supported);
        const ParamValueRouteDisposition value_route = classify_value_route(descriptor);
        const auto projection = classify_param_projection(
            make_document_field(descriptor),
            ParamProjectionContext{
                .profile_allows = materialization != ParameterMaterializationStrategy::kRejected,
                .codec_available = value_route == ParamValueRouteDisposition::kPortableScalar,
            });
        const bool admitted = disposition == ParameterFieldUpdateDisposition::kAdmitted;
        op.runtime_updates_enabled |=
            admitted && descriptor.mutability == ParamMutability::kRuntime;
        op.lifecycle_updates_enabled |=
            admitted && descriptor.mutability == ParamMutability::kLifecycle;
        result.fields.push_back(ParameterFieldPlanRecord{
            .owner = owner,
            .instance = instance,
            .instance_identity_digest = instance_identity,
            .field = descriptor.id,
            .full_id = descriptor.full_id,
            .canonical_name = std::string(descriptor.canonical_name),
            .logical_type = descriptor.logical_type,
            .type_shape = descriptor.type_shape,
            .mutability = descriptor.mutability,
            .earliest_invalidated_stage = descriptor.earliest_invalidated_stage,
            .persistence = descriptor.persistence,
            .visibility = descriptor.visibility,
            .external_access = descriptor.external_access,
            .sensitivity = descriptor.sensitivity,
            .required_on_load = descriptor.required_on_load,
            .alias = std::string(descriptor.alias),
            .validator_id = std::string(descriptor.validator_id),
            .constraints = descriptor.constraints,
            .enum_choices = own_enum_choices(descriptor.enum_choices),
            .string_encoding = descriptor.string_encoding,
            .element_string_encoding = descriptor.element_string_encoding,
            .value_route = value_route,
            .storage = descriptor.storage,
            .materialization = materialization,
            .projection = projection,
            .update_disposition = disposition,
            .adapter_eligible = projection == params::ParamProjectionStatus::kIncluded &&
                                descriptor.external_access != ParamExternalAccess::kNone &&
                                value_route == ParamValueRouteDisposition::kPortableScalar,
        });

        ParameterGenerationZeroOrigin field_origin = complete_transaction_target
                                                         ? ParameterGenerationZeroOrigin::kDefault
                                                         : field_baseline_origins[factory_index];
        const auto contribution =
            std::find_if(accepted_contributions[factory_index].begin(),
                         accepted_contributions[factory_index].end(),
                         [&](const auto& item) {
                           return item.field == descriptor.id && item.full_id == descriptor.full_id;
                         });
        if (contribution != accepted_contributions[factory_index].end()) {
          field_origin = generation_zero_origin(contribution->origin);
        }
        std::optional<ParamConfigFieldProvenance> config_source;
        if (contribution != accepted_contributions[factory_index].end() &&
            contribution->config_provenance_index && factories.parameter_transaction) {
          const auto& provenance = factories.parameter_transaction->provenance.config_fields;
          const std::size_t source_index = *contribution->config_provenance_index;
          if (source_index < provenance.size()) {
            config_source = provenance[source_index];
            if (factories.parameter_transaction && path) {
              config_source->target_operator_path = path->rendered;
            }
          }
        } else if (contribution == accepted_contributions[factory_index].end() &&
                   descriptor_index < local_config_sources[factory_index].size() &&
                   local_config_sources[factory_index][descriptor_index]) {
          config_source = local_config_sources[factory_index][descriptor_index];
          field_origin = config_source->origin == ParamFieldRebindOrigin::kAlias
                             ? ParameterGenerationZeroOrigin::kInputAlias
                             : ParameterGenerationZeroOrigin::kInputCaptured;
        }
        result.generation_zero_fields.push_back(ParameterGenerationZeroFieldRecord{
            .instance = instance,
            .field = descriptor.id,
            .full_id = descriptor.full_id,
            .origin = field_origin,
            .transaction_contributed = contribution != accepted_contributions[factory_index].end(),
            .value_digest = parameter_field_value_digest(
                *schema, descriptor, executable.factory->initial_parameter_value()),
            .config_source = std::move(config_source),
        });

        auto param_path = ParamPath::parse(descriptor.canonical_name);
        if (!param_path) {
          append_diagnostic(result, "PARAMETER_OVERRIDE_FIELD_INVALID", source);
          valid = false;
        } else if (descriptor.visibility != ParamVisibility::kInternal) {
          catalog_fields.push_back(ParamFieldCatalogRecord{
              .instance = instance,
              .instance_identity_digest = instance_identity,
              .address = ParamAddress{.operator_instance = owner, .field = descriptor.id},
              .full_id = descriptor.full_id,
              .path = std::move(param_path).value(),
              .logical_type = descriptor.logical_type,
              .type_shape = descriptor.type_shape,
              .mutability = descriptor.mutability,
              .earliest_invalidated_stage = descriptor.earliest_invalidated_stage,
              .persistence = descriptor.persistence,
              .visibility = descriptor.visibility,
              .external_access = descriptor.external_access,
              .sensitivity = descriptor.sensitivity,
              .required_on_load = descriptor.required_on_load,
              .alias = std::string(descriptor.alias),
              .validator_id = std::string(descriptor.validator_id),
              .constraints = descriptor.constraints,
              .enum_choices = own_enum_choices(descriptor.enum_choices),
              .string_encoding = descriptor.string_encoding,
              .element_string_encoding = descriptor.element_string_encoding,
              .value_route = value_route,
              .storage = descriptor.storage,
              .materialization = materialization,
              .projection = projection,
              .update_disposition = disposition,
              .adapter_eligible = projection == params::ParamProjectionStatus::kIncluded &&
                                  descriptor.external_access != ParamExternalAccess::kNone &&
                                  value_route == ParamValueRouteDisposition::kPortableScalar,
          });
        }
      }
    }

    if ((op.runtime_updates_enabled || op.lifecycle_updates_enabled) &&
        !op.fixed_scalar_generation_strategy) {
      for (std::size_t index = field_begin; index < result.fields.size(); ++index) {
        if (result.fields[index].update_disposition == ParameterFieldUpdateDisposition::kAdmitted) {
          result.fields[index].update_disposition =
              ParameterFieldUpdateDisposition::kCapacityUnavailable;
        }
      }
      op.runtime_updates_enabled = false;
      op.lifecycle_updates_enabled = false;
      append_diagnostic(result, "PARAMETER_UPDATE_STORAGE_UNSUPPORTED", source);
      valid = false;
    }

    if (op.runtime_updates_enabled) {
      ++runtime_operator_count;
      const auto runtime_field_count = std::count_if(
          result.fields.begin() + static_cast<std::ptrdiff_t>(field_begin),
          result.fields.end(),
          [](const auto& field) {
            return field.update_disposition == ParameterFieldUpdateDisposition::kAdmitted &&
                   field.mutability == ParamMutability::kRuntime;
          });
      std::uint32_t runtime_fields{};
      if (!checked_size32(static_cast<std::size_t>(runtime_field_count), runtime_fields)) {
        append_diagnostic(result, "PARAMETER_UPDATE_CAPACITY_INSUFFICIENT", source);
        valid = false;
      }
      max_runtime_fields = std::max(max_runtime_fields, runtime_fields);
    }
    if (op.lifecycle_updates_enabled) {
      ++lifecycle_operator_count;
      for (std::size_t index = field_begin; index < result.fields.size(); ++index) {
        const auto& field = result.fields[index];
        if (field.update_disposition == ParameterFieldUpdateDisposition::kAdmitted &&
            field.mutability == ParamMutability::kLifecycle) {
          result.lifecycle_mutable_fields.push_back(LifecycleMutableFieldPlanRecord{
              .owner = owner,
              .field = field.field,
              .from_stage = field.earliest_invalidated_stage,
          });
        }
      }
    }
    result.operators.push_back(std::move(op));
  }

  std::sort(result.fields.begin(), result.fields.end(), [](const auto& lhs, const auto& rhs) {
    return std::tuple{lhs.owner.id, lhs.field} < std::tuple{rhs.owner.id, rhs.field};
  });
  std::sort(result.generation_zero_fields.begin(),
            result.generation_zero_fields.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.instance, lhs.field} < std::tuple{rhs.instance, rhs.field};
            });
  std::sort(catalog_fields.begin(), catalog_fields.end(), [](const auto& lhs, const auto& rhs) {
    return std::tuple{lhs.address.operator_instance.id, lhs.address.field} <
           std::tuple{rhs.address.operator_instance.id, rhs.address.field};
  });
  // An operator with both runtime- and lifecycle-mutable fields owns one
  // parameter arena and one catalog entry. Count it once even though each
  // update path contributes distinct queue and evidence obligations.
  const auto enabled_operator_total =
      std::count_if(result.operators.begin(), result.operators.end(), [](const auto& op) {
        return op.runtime_updates_enabled || op.lifecycle_updates_enabled;
      });
  std::uint32_t enabled_operator_count{};
  if (!checked_size32(static_cast<std::size_t>(enabled_operator_total), enabled_operator_count)) {
    append_diagnostic(result, "PARAMETER_UPDATE_CAPACITY_INSUFFICIENT");
    valid = false;
  }
  const std::uint32_t pending = options.max_pending_updates_per_operator;
  std::uint32_t runtime_slots{};
  std::uint32_t lifecycle_slots{};
  if (runtime_operator_count != 0U && (!checked_add(pending, 2U, runtime_slots) || pending == 0U)) {
    append_diagnostic(result, "PARAMETER_UPDATE_CAPACITY_INSUFFICIENT");
    valid = false;
  }
  if (lifecycle_operator_count != 0U &&
      (!checked_add(pending, 4U, lifecycle_slots) || pending == 0U)) {
    append_diagnostic(result, "PARAMETER_UPDATE_CAPACITY_INSUFFICIENT");
    valid = false;
  }
  const std::uint32_t minimum_slots = std::max(runtime_slots, lifecycle_slots);
  const std::uint32_t slots = options.generation_slots_per_operator == 0U
                                  ? minimum_slots
                                  : options.generation_slots_per_operator;
  if (minimum_slots != 0U && slots < minimum_slots) {
    append_diagnostic(result, "PARAMETER_UPDATE_CAPACITY_INSUFFICIENT");
    valid = false;
  }

  std::uint32_t runtime_request_capacity{};
  std::uint32_t lifecycle_request_capacity{};
  std::uint32_t lifecycle_receipt_capacity{};
  std::uint32_t required_audit_capacity{};
  std::uint32_t runtime_event_capacity{};
  std::uint32_t lifecycle_event_capacity{};
  std::uint32_t required_event_capacity{};
  if (!checked_multiply(runtime_operator_count, pending, runtime_request_capacity) ||
      !checked_multiply(lifecycle_operator_count, pending, lifecycle_request_capacity) ||
      !checked_multiply(lifecycle_request_capacity, 2U, lifecycle_receipt_capacity) ||
      !checked_add(runtime_request_capacity, lifecycle_receipt_capacity, required_audit_capacity) ||
      !checked_multiply(
          runtime_request_capacity, kRuntimeEventsPerAcceptedUpdate, runtime_event_capacity) ||
      !checked_multiply(lifecycle_receipt_capacity,
                        kLifecycleEventsPerAcceptedUpdate,
                        lifecycle_event_capacity) ||
      !checked_add(runtime_event_capacity, lifecycle_event_capacity, required_event_capacity)) {
    append_diagnostic(result, "PARAMETER_UPDATE_CAPACITY_INSUFFICIENT");
    valid = false;
  }
  const std::uint32_t audit =
      options.audit_capacity == 0U ? required_audit_capacity : options.audit_capacity;
  if (audit < required_audit_capacity ||
      (enabled_operator_count != 0U && options.max_event_cursors == 0U)) {
    append_diagnostic(result, "PARAMETER_UPDATE_CAPACITY_INSUFFICIENT");
    valid = false;
  }

  std::uint64_t arena_bytes{};
  std::uint64_t max_alignment{};
  for (const auto& op : result.operators) {
    if (!op.runtime_updates_enabled && !op.lifecycle_updates_enabled) {
      continue;
    }
    const auto factory =
        std::find_if(factories.operators.begin(), factories.operators.end(), [&](const auto& item) {
          return item.operator_id == op.owner.id;
        });
    if (factory == factories.operators.end() || !factory->factory) {
      append_diagnostic(result, "PARAMETER_FACTORY_INVENTORY_INVALID");
      valid = false;
      continue;
    }
    const ParamTypeDescriptor& type = factory->factory->parameter_type();
    std::uint64_t stride{};
    std::uint64_t operator_bytes{};
    std::uint64_t next_arena{};
    if (!aligned_stride(type, stride) || !checked_multiply(stride, slots, operator_bytes) ||
        !checked_add(arena_bytes, operator_bytes, next_arena)) {
      append_diagnostic(result,
                        "PARAMETER_UPDATE_CAPACITY_INSUFFICIENT",
                        DiagnosticSource{.vertex = factory->vertex});
      valid = false;
      continue;
    }
    arena_bytes = next_arena;
    max_alignment = std::max(max_alignment, static_cast<std::uint64_t>(type.object_alignment));
  }
  if (options.max_parameter_arena_bytes &&
      arena_bytes > static_cast<std::uint64_t>(*options.max_parameter_arena_bytes)) {
    append_diagnostic(result, "PARAMETER_UPDATE_CAPACITY_INSUFFICIENT");
    valid = false;
  }

  std::uint32_t compute_ingress{};
  if (runtime_operator_count != 0U &&
      !checked_multiply(runtime_operator_count, pending, compute_ingress)) {
    append_diagnostic(result, "PARAMETER_UPDATE_CAPACITY_INSUFFICIENT");
    valid = false;
  }

  LifecycleTransactionCapacityValues required_lifecycle_capacity{};
  if (lifecycle_operator_count != 0U) {
    // Queue, receipt, candidate, event, and audit bounds are intrinsic to
    // accepted Parameter submissions. Receipt-lifetime bounds retain one
    // superseded predecessor per pending update until explicit retirement.
    // Fence, quiesce, and rollback bounds depend on lifecycle dependency
    // closure expansion and are finalized only after the first lifecycle
    // lowering pass.
    required_lifecycle_capacity = {
        .max_pending_requests = lifecycle_request_capacity,
        .max_pending_transactions = lifecycle_request_capacity,
        .max_receipts = lifecycle_receipt_capacity,
        .max_fenced_operators = 0U,
        .max_candidate_generations = lifecycle_request_capacity,
        .max_events = lifecycle_event_capacity,
        .max_audit_records = lifecycle_receipt_capacity,
        .max_quiesce_records = 0U,
        .max_rollback_records = 0U,
    };
  }

  result.capacity.arena_strategy = ParameterArenaStrategy::kSeparatePerOperator;
  result.capacity.enabled_operator_count = enabled_operator_count;
  // A generation-zero-only plan retains the Parameter catalog and static
  // bindings but owns no update queue, generation arena, observer ledger, or
  // cursor storage. Keeping every inactive capacity at zero lets runtime
  // materialization distinguish that valid mode from a malformed active plan.
  result.capacity.max_pending_updates_per_operator = enabled_operator_count == 0U ? 0U : pending;
  result.capacity.generation_slots_per_operator = enabled_operator_count == 0U ? 0U : slots;
  result.capacity.compute_ingress_capacity = compute_ingress;
  result.capacity.max_compute_patches_per_request = max_runtime_fields;
  result.capacity.audit_capacity = enabled_operator_count == 0U ? 0U : audit;
  result.capacity.event_capacity = required_event_capacity;
  result.capacity.max_event_cursors = enabled_operator_count == 0U ? 0U : options.max_event_cursors;
  result.capacity.required_lifecycle_capacity = required_lifecycle_capacity;
  result.capacity.parameter_arena_bytes = arena_bytes;
  result.capacity.max_arena_alignment = max_alignment;
  result.capacity.runtime_updates_enabled = runtime_operator_count != 0U;
  result.capacity.lifecycle_updates_enabled = lifecycle_operator_count != 0U;

  if (!valid) {
    disable_admitted_fields(result);
  }
  for (auto& catalog_field : catalog_fields) {
    const auto plan_field =
        std::find_if(result.fields.begin(), result.fields.end(), [&](const auto& candidate) {
          return candidate.owner == catalog_field.address.operator_instance &&
                 candidate.field == catalog_field.address.field &&
                 candidate.full_id == catalog_field.full_id;
        });
    if (plan_field != result.fields.end()) {
      catalog_field.update_disposition = plan_field->update_disposition;
    }
  }
  const ParamCatalogDigest catalog_digest =
      parameter_catalog_digest(catalog_operators, catalog_fields);
  result.catalog = ParameterCatalogAccess::make(
      catalog_digest, std::move(catalog_operators), std::move(catalog_fields));
  return result;
}

ParameterCatalogView rebind_parameter_catalog_plan_generation(const ParameterCatalogView& catalog,
                                                              PlanGenerationId plan_generation) {
  std::vector<OperatorParamRecord> operators{catalog.operators().begin(),
                                             catalog.operators().end()};
  for (auto& record : operators) {
    record.operator_instance.plan_generation = plan_generation;
  }
  std::vector<ParamFieldCatalogRecord> fields{catalog.fields().begin(), catalog.fields().end()};
  for (auto& record : fields) {
    record.address.operator_instance.plan_generation = plan_generation;
  }
  const ParamCatalogDigest digest = parameter_catalog_digest(operators, fields);
  return ParameterCatalogAccess::make(digest, std::move(operators), std::move(fields));
}

void finalize_parameter_lifecycle_capacity(
    ParameterPlanLowering& result, const LifecyclePlanLowering& lifecycle,
    const std::optional<LifecycleTransactionOptions>& lifecycle_options) {
  if (result.lifecycle_mutable_fields.empty()) {
    return;
  }

  bool valid = true;
  std::vector<OperatorInstanceId> closure_members;
  std::vector<OperatorInstanceId> candidate_owners;
  for (const auto& field : result.lifecycle_mutable_fields) {
    if (!is_lifecycle_forward_stage(field.from_stage)) {
      append_diagnostic(result, "PARAMETER_LIFECYCLE_CAPACITY_INSUFFICIENT");
      valid = false;
      continue;
    }
    candidate_owners.push_back(field.owner.id);
    const auto closure =
        std::find_if(lifecycle.restart_closures.begin(),
                     lifecycle.restart_closures.end(),
                     [&](const auto& item) { return item.target.id == field.owner.id; });
    if (closure == lifecycle.restart_closures.end()) {
      append_diagnostic(result, "PARAMETER_LIFECYCLE_CAPACITY_INSUFFICIENT", DiagnosticSource{});
      valid = false;
      continue;
    }
    for (const auto& member : closure->closure) {
      closure_members.push_back(member.id);
    }
  }
  std::sort(closure_members.begin(), closure_members.end());
  closure_members.erase(std::unique(closure_members.begin(), closure_members.end()),
                        closure_members.end());
  std::sort(candidate_owners.begin(), candidate_owners.end());
  candidate_owners.erase(std::unique(candidate_owners.begin(), candidate_owners.end()),
                         candidate_owners.end());

  std::uint32_t fenced{};
  std::uint32_t active_and_successor_members{};
  if (!checked_size32(closure_members.size(), fenced) ||
      !checked_multiply(fenced, 2U, active_and_successor_members)) {
    valid = false;
  }

  auto& required = result.capacity.required_lifecycle_capacity;
  std::uint32_t candidate_owner_count{};
  std::uint32_t active_and_successor_candidates{};
  if (!checked_size32(candidate_owners.size(), candidate_owner_count) ||
      !checked_multiply(candidate_owner_count, 2U, active_and_successor_candidates)) {
    valid = false;
  }
  required.max_candidate_generations =
      std::max(required.max_candidate_generations, active_and_successor_candidates);
  required.max_fenced_operators = fenced;
  required.max_quiesce_records = active_and_successor_members;
  required.max_rollback_records = active_and_successor_members;
  if (!valid || !lifecycle_options || !lifecycle_capacity_satisfies(*lifecycle_options, required)) {
    append_diagnostic(result, "PARAMETER_LIFECYCLE_CAPACITY_INSUFFICIENT");
    disable_admitted_fields(result);
  }
}

}  // namespace holoscan::detail
