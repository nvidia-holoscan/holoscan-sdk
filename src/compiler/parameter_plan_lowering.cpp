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
#include "../core/parameter_override_set_internal.hpp"
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
  if (!supported_scalar_type(field.logical_type) || !field.type_shape.has_stable_shape) {
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
    ParamOverrideContributionOrigin origin) noexcept {
  switch (origin) {
    case ParamOverrideContributionOrigin::kCaptured:
      return ParameterGenerationZeroOrigin::kOverrideCaptured;
    case ParamOverrideContributionOrigin::kAlias:
      return ParameterGenerationZeroOrigin::kOverrideAlias;
    case ParamOverrideContributionOrigin::kMigration:
      return ParameterGenerationZeroOrigin::kOverrideMigration;
  }
  return ParameterGenerationZeroOrigin::kOverrideCaptured;
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
  auto created = factory.create();
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
  return replayed_ports_match(snapshot, owner, spec) &&
         replayed_sources_match(snapshot, owner, spec) &&
         replayed_lifecycle_matches(snapshot, owner, spec, factory.lifecycle_hooks());
}

[[nodiscard]] ParameterProtectedValueDigest override_value_digest(
    const ParamTypeDescriptor& type, const ParamSchemaDescriptor* schema, const void* value,
    bool no_params, bool (*tooling_digest)(const void*, ParamSchemaHash&) noexcept) {
  ParameterProtectedValueDigest result = parameter_value_digest(type, schema, value, no_params);
  if (result.available() || tooling_digest == nullptr) {
    return result;
  }
  ParamSchemaHash digest{};
  if (!tooling_digest(value, digest)) {
    return result;
  }
  return {.disposition = ParameterValueDigestDisposition::kToolingCodec, .digest = digest};
}

[[nodiscard]] bool field_values_equal(const ParamFieldDescriptor& field, const void* lhs,
                                      const void* rhs) noexcept {
  const auto& operations = field.local_operations;
  if (operations.copy_construct_from_member == nullptr || operations.destroy_staged == nullptr ||
      operations.semantic_equal_to_staged == nullptr || operations.object_size == 0U ||
      operations.object_alignment == 0U) {
    return false;
  }
  void* staging = nullptr;
  bool constructed = false;
  try {
    staging = ::operator new(operations.object_size, std::align_val_t{operations.object_alignment});
    operations.copy_construct_from_member(staging, lhs);
    constructed = true;
    const bool equal = operations.semantic_equal_to_staged(rhs, staging);
    operations.destroy_staged(staging);
    ::operator delete(staging, std::align_val_t{operations.object_alignment});
    return equal;
  } catch (...) {
    if (constructed) {
      operations.destroy_staged(staging);
    }
    if (staging != nullptr) {
      ::operator delete(staging, std::align_val_t{operations.object_alignment});
    }
    return false;
  }
}

[[nodiscard]] bool valid_config_source(const ParamConfigFieldProvenance& source,
                                       const ParamOverrideOperatorState& target,
                                       const ParamOverrideFieldOriginRecord& contribution,
                                       std::size_t expected_ordinal) noexcept {
  const bool origin_matches = (contribution.origin == ParamOverrideContributionOrigin::kCaptured &&
                               source.origin == ParamFieldRebindOrigin::kCaptured) ||
                              (contribution.origin == ParamOverrideContributionOrigin::kAlias &&
                               source.origin == ParamFieldRebindOrigin::kAlias);
  return source.document.semantic_version != 0U &&
         (!source.document.has_raw_source || source.document.raw_algorithm_version != 0U) &&
         !source.source_path.empty() && source.source_path.size() <= 512U &&
         source.source_ordinal == expected_ordinal &&
         source.target_operator_path == target.operator_path &&
         source.target_schema_id == target.schema_id &&
         source.target_schema_version == target.schema_version &&
         source.target_field_id == contribution.field &&
         source.target_full_id == contribution.full_id && origin_matches;
}

[[nodiscard]] bool selected_rebind_target_path(const ParamRebindProvenance& provenance,
                                               std::string_view candidate) noexcept {
  switch (provenance.target_scope_kind) {
    case ParamRebindTargetScopeKind::kWholeGraph:
      return true;
    case ParamRebindTargetScopeKind::kGraph:
      return candidate.size() > provenance.target_scope_paths.front().size() &&
             candidate.starts_with(provenance.target_scope_paths.front()) &&
             candidate[provenance.target_scope_paths.front().size()] == '/';
    case ParamRebindTargetScopeKind::kOperator:
      return candidate == provenance.target_scope_paths.front();
    case ParamRebindTargetScopeKind::kOperatorSet:
      return std::ranges::binary_search(provenance.target_scope_paths, candidate);
  }
  return false;
}

[[nodiscard]] bool valid_rebind_target_scope_shape(
    const ParamRebindProvenance& provenance) noexcept {
  const auto& paths = provenance.target_scope_paths;
  switch (provenance.target_scope_kind) {
    case ParamRebindTargetScopeKind::kWholeGraph:
      return paths.empty();
    case ParamRebindTargetScopeKind::kGraph:
    case ParamRebindTargetScopeKind::kOperator:
      return paths.size() == 1U && !paths.front().empty() && paths.front().size() <= 512U;
    case ParamRebindTargetScopeKind::kOperatorSet:
      return !paths.empty() && paths.size() <= 65'536U && std::ranges::is_sorted(paths) &&
             std::ranges::adjacent_find(paths) == paths.end() &&
             std::ranges::all_of(paths, [](const std::string& path) {
               return !path.empty() && path.size() <= 512U;
             });
  }
  return false;
}

[[nodiscard]] bool valid_rebind_target_selection(
    const ParamRebindProvenance& provenance, std::span<const std::optional<OperatorPathInfo>> paths,
    const ExecutableFactoryInventory& factories) {
  if (!valid_rebind_target_scope_shape(provenance) || provenance.target_operators.empty() ||
      provenance.target_operators.size() > 65'536U ||
      !std::ranges::is_sorted(
          provenance.target_operators, {}, &ParamRebindTargetOperatorProvenance::operator_path) ||
      std::ranges::adjacent_find(
          provenance.target_operators, {}, &ParamRebindTargetOperatorProvenance::operator_path) !=
          provenance.target_operators.end() ||
      provenance.target_selection_digest !=
          ParamOverrideSetAccess::target_selection_digest(provenance.target_scope_kind,
                                                          provenance.target_scope_paths,
                                                          provenance.target_operators) ||
      paths.size() != factories.operators.size()) {
    return false;
  }

  std::vector<ParamRebindTargetOperatorProvenance> expected;
  expected.reserve(provenance.target_operators.size());
  for (std::size_t index = 0; index < paths.size(); ++index) {
    const auto& path = paths[index];
    if (!path.has_value()) {
      continue;
    }
    const auto& path_info = path.value();
    if (!selected_rebind_target_path(provenance, path_info.rendered)) {
      continue;
    }
    const auto& executable = factories.operators[index];
    const ParamSchemaDescriptor* const schema =
        executable.factory == nullptr ? nullptr : executable.factory->parameter_schema();
    if (schema == nullptr || schema->stable_id.empty() || schema->stable_id.size() > 256U ||
        schema->version == 0U) {
      return false;
    }
    expected.push_back(ParamRebindTargetOperatorProvenance{
        .operator_path = path_info.rendered,
        .schema_id = std::string{schema->stable_id},
        .schema_version = schema->version,
        .structural_hash = schema->structural_hash,
        .contract_hash = schema->contract_hash,
    });
  }
  std::ranges::sort(expected, {}, &ParamRebindTargetOperatorProvenance::operator_path);
  return expected == provenance.target_operators;
}

struct OverridePathIndex {
  std::string_view path;
  std::size_t index{};
};

[[nodiscard]] std::vector<OverridePathIndex> make_factory_path_index(
    std::span<const std::optional<OperatorPathInfo>> paths) {
  std::vector<OverridePathIndex> result;
  result.reserve(paths.size());
  for (std::size_t index = 0; index < paths.size(); ++index) {
    const auto& path = paths[index];
    if (path.has_value()) {
      // The same local optional is checked immediately before this cold index projection.
      result.push_back(OverridePathIndex{
          .path = path.value().rendered,  // NOLINT(bugprone-unchecked-optional-access)
          .index = index});
    }
  }
  std::ranges::sort(result, [](const auto& lhs, const auto& rhs) { return lhs.path < rhs.path; });
  return result;
}

[[nodiscard]] const OverridePathIndex* find_factory_path(std::span<const OverridePathIndex> index,
                                                         std::string_view path) noexcept {
  const auto found = std::lower_bound(
      index.begin(), index.end(), path, [](const OverridePathIndex& item, std::string_view key) {
        return item.path < key;
      });
  return found != index.end() && found->path == path ? &*found : nullptr;
}

[[nodiscard]] std::vector<bool> duplicate_override_targets(
    std::span<const ParamOverrideOperatorState> operators) {
  std::vector<OverridePathIndex> ordered;
  ordered.reserve(operators.size());
  for (std::size_t index = 0; index < operators.size(); ++index) {
    ordered.push_back(OverridePathIndex{.path = operators[index].operator_path, .index = index});
  }
  std::ranges::sort(ordered, [](const auto& lhs, const auto& rhs) {
    return lhs.path < rhs.path || (lhs.path == rhs.path && lhs.index < rhs.index);
  });
  std::vector<bool> duplicate(operators.size());
  for (std::size_t index = 1; index < ordered.size(); ++index) {
    if (ordered[index - 1U].path == ordered[index].path) {
      duplicate[ordered[index].index] = true;
    }
  }
  return duplicate;
}

// Override replay is transactional across every affected Operator factory.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool apply_parameter_overrides(
    const GraphSnapshotData& snapshot, ExecutableFactoryInventory& factories,
    std::span<const std::optional<OperatorPathInfo>> paths,
    std::span<const ParameterProtectedValueDigest> baseline_digests,
    ParamSchemaHash admission_context_digest,
    const std::optional<ParamOverrideSet>& parameter_overrides,
    std::vector<std::vector<ParamOverrideFieldOriginRecord>>& accepted_contributions,
    ParameterPlanLowering& result) {
  if (!parameter_overrides) {
    return true;
  }
  const ParamOverrideSetState& state = ParamOverrideSetAccess::state(*parameter_overrides);
  if (state.representation_version != 3U) {
    append_diagnostic(result, "PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE");
    return false;
  }
  if (!state.target_graph || *state.target_graph != snapshot.identity ||
      state.target_identity != snapshot.identity.hash.value) {
    append_diagnostic(result, "PARAMETER_OVERRIDE_TARGET_MISMATCH");
    return false;
  }
  const bool exact_context = state.context_binding == ParamOverrideContextBinding::kExact;
  const bool compile_context = state.context_binding == ParamOverrideContextBinding::kCompile;
  if ((!exact_context && !compile_context) ||
      state.provenance.context_binding != state.context_binding) {
    append_diagnostic(result, "PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE");
    return false;
  }
  const ParamSchemaHash empty_context_digest{};
  if (compile_context) {
    if (state.admission_context_digest != empty_context_digest ||
        state.provenance.admission_context_digest != empty_context_digest) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE");
      return false;
    }
  } else {
    if (state.admission_context_digest == empty_context_digest ||
        state.provenance.admission_context_digest == empty_context_digest) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE");
      return false;
    }
    if (state.admission_context_digest != admission_context_digest) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_ADMISSION_CONTEXT_MISMATCH");
      return false;
    }
  }
  if (state.semantic_digest !=
      ParamOverrideSetAccess::recompute_semantic_digest(*parameter_overrides)) {
    append_diagnostic(result, "PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE");
    return false;
  }
  if (state.provenance.target != snapshot.identity ||
      (exact_context && state.provenance.admission_context_digest != admission_context_digest)) {
    append_diagnostic(result, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH");
    return false;
  }
  const ParamSchemaHash empty_target_selection_digest = {};
  const bool has_rebind_target_selection =
      state.provenance.target_selection_digest != empty_target_selection_digest ||
      !state.provenance.target_scope_paths.empty() || !state.provenance.target_operators.empty() ||
      !state.provenance.fields.empty() ||
      state.provenance.target_scope_kind != ParamRebindTargetScopeKind::kWholeGraph;
  const PlanDigest empty_plan_digest;
  const ParamCatalogDigest empty_catalog_digest;
  const bool has_snapshot_source =
      state.provenance.source_archive_semantic_version != 0U ||
      state.provenance.source_archive_digest != std::array<std::byte, 32>{} ||
      state.provenance.source_plan_digest != empty_plan_digest ||
      state.provenance.source_plan_generation != 0U ||
      state.provenance.source_run_generation != 0U ||
      state.provenance.source_catalog_digest != empty_catalog_digest ||
      state.provenance.source_scope_kind != ParamSnapshotScopeKind::kRoot ||
      !state.provenance.source_scope_path.empty();
  const bool has_config_source = !state.provenance.config_fields.empty();
  if (exact_context && has_config_source) {
    append_diagnostic(result, "PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE");
    return false;
  }
  if (compile_context &&
      (state.provenance.mode != ParamRebindMode::kSparseContributions || !has_config_source ||
       has_snapshot_source || !state.provenance.migrations.empty() ||
       !state.provenance.operator_aliases.empty() || !state.provenance.field_aliases.empty() ||
       !state.provenance.fields.empty() || has_rebind_target_selection)) {
    append_diagnostic(result, "PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE");
    return false;
  }
  if (has_rebind_target_selection &&
      !valid_rebind_target_selection(state.provenance, paths, factories)) {
    append_diagnostic(result, "PARAMETER_OVERRIDE_TARGET_SELECTION_MISMATCH");
    return false;
  }

  bool valid = true;
  std::vector<bool> accepted_config_ordinals(state.provenance.config_fields.size());
  std::size_t accepted_config_ordinal_count{};
  const std::vector<OverridePathIndex> factory_path_index = make_factory_path_index(paths);
  const std::vector<bool> duplicate_targets = duplicate_override_targets(state.operators);
  std::vector<std::shared_ptr<OperatorFactoryBase>> replacements(factories.operators.size());
  std::vector<std::vector<ParamOverrideFieldOriginRecord>> contributions(
      factories.operators.size());
  for (std::size_t override_index = 0; override_index < state.operators.size(); ++override_index) {
    const ParamOverrideOperatorState& override_record = state.operators[override_index];
    if (duplicate_targets[override_index]) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_DUPLICATE_TARGET");
      valid = false;
      continue;
    }
    const OverridePathIndex* const target =
        find_factory_path(factory_path_index, override_record.operator_path);
    if (target == nullptr) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_TARGET_UNKNOWN");
      valid = false;
      continue;
    }
    const std::size_t index = target->index;
    const auto& executable = factories.operators[index];
    const DiagnosticSource source{.vertex = executable.vertex};
    if (!executable.factory || replacements[index]) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_DUPLICATE_TARGET", source);
      valid = false;
      continue;
    }
    const ParamTypeDescriptor& type = executable.factory->parameter_type();
    const ParamSchemaDescriptor* const schema = executable.factory->parameter_schema();
    if (schema == nullptr || override_record.local_type != &type || schema->local_type != &type ||
        override_record.schema_id != schema->stable_id ||
        override_record.schema_version != schema->version ||
        override_record.structural_hash != schema->structural_hash ||
        override_record.contract_hash != schema->contract_hash) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_SCHEMA_MISMATCH", source);
      valid = false;
      continue;
    }
    ParameterProtectedValueDigest baseline_digest = baseline_digests[index];
    if (!baseline_digest.available()) {
      baseline_digest = override_value_digest(type,
                                              schema,
                                              executable.factory->initial_parameter_value(),
                                              executable.factory->has_no_params(),
                                              override_record.local_value_digest);
    }
    if (!baseline_digest.available() ||
        override_record.baseline_value_digest != baseline_digest.digest) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_BASELINE_MISMATCH", source);
      valid = false;
      continue;
    }
    if (!override_record.resolved_value || type.validate_aggregate == nullptr ||
        !type.validate_aggregate(override_record.resolved_value.get()).ok()) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_VALUE_INVALID", source);
      valid = false;
      continue;
    }
    const ParameterProtectedValueDigest resolved_digest =
        override_value_digest(type,
                              schema,
                              override_record.resolved_value.get(),
                              executable.factory->has_no_params(),
                              override_record.local_value_digest);
    if (!resolved_digest.available() ||
        override_record.resolved_value_digest != resolved_digest.digest) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_VALUE_INVALID", source);
      valid = false;
      continue;
    }

    std::vector<ParamFieldId> contributed_fields;
    contributed_fields.reserve(override_record.contributions.size());
    bool fields_valid = true;
    for (const ParamOverrideFieldOriginRecord& contribution : override_record.contributions) {
      const auto field = std::find_if(schema->fields.begin(),
                                      schema->fields.end(),
                                      [&](const ParamFieldDescriptor& descriptor) {
                                        return descriptor.id == contribution.field &&
                                               descriptor.full_id == contribution.full_id;
                                      });
      if (field == schema->fields.end() ||
          std::find(contributed_fields.begin(), contributed_fields.end(), contribution.field) !=
              contributed_fields.end() ||
          (contribution.origin == ParamOverrideContributionOrigin::kAlias &&
           field->alias.empty())) {
        append_diagnostic(result, "PARAMETER_OVERRIDE_FIELD_INVALID", source);
        valid = false;
        fields_valid = false;
        break;
      }
      if (compile_context && !contribution.config_provenance_index) {
        append_diagnostic(result, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH", source);
        valid = false;
        fields_valid = false;
        break;
      }
      if (contribution.config_provenance_index) {
        const std::size_t source_index = *contribution.config_provenance_index;
        if (source_index >= state.provenance.config_fields.size() ||
            accepted_config_ordinals[source_index] ||
            !valid_config_source(state.provenance.config_fields[source_index],
                                 override_record,
                                 contribution,
                                 source_index)) {
          append_diagnostic(result, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH", source);
          valid = false;
          fields_valid = false;
          break;
        }
        accepted_config_ordinals[source_index] = true;
        ++accepted_config_ordinal_count;
      }
      contributed_fields.push_back(contribution.field);
    }
    if (!fields_valid) {
      continue;
    }

    if (!override_record.field_replacements_only) {
      const void* const baseline = executable.factory->initial_parameter_value();
      for (const ParamFieldDescriptor& field : schema->fields) {
        ParamScalarValue before;
        ParamScalarValue after;
        const bool contributed =
            std::find(contributed_fields.begin(), contributed_fields.end(), field.id) !=
            contributed_fields.end();
        if (read_scalar(field, baseline, before) &&
            read_scalar(field, override_record.resolved_value.get(), after)) {
          if (!contributed && before != after) {
            append_diagnostic(result, "PARAMETER_OVERRIDE_UNDECLARED_CHANGE", source);
            valid = false;
            fields_valid = false;
            break;
          }
        } else if (!contributed &&
                   !field_values_equal(field, baseline, override_record.resolved_value.get())) {
          append_diagnostic(result, "PARAMETER_OVERRIDE_UNDECLARED_CHANGE", source);
          valid = false;
          fields_valid = false;
          break;
        }
      }
    }
    if (!fields_valid) {
      continue;
    }
    auto cloned = executable.factory->clone_with_params(override_record.resolved_value.get(),
                                                        type.local_type_token);
    if (!cloned || !cloned.value()) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_VALUE_INVALID", source);
      valid = false;
      continue;
    }
    if (!replayed_declarations_match(snapshot, executable.vertex, *cloned.value())) {
      append_diagnostic(result, "PARAMETER_OVERRIDE_DECLARATION_MISMATCH", source);
      valid = false;
      continue;
    }
    replacements[index] = std::move(cloned).value();
    contributions[index] = override_record.contributions;
    std::sort(contributions[index].begin(),
              contributions[index].end(),
              [](const auto& lhs, const auto& rhs) { return lhs.field < rhs.field; });
  }
  if (!valid) {
    return false;
  }
  if (accepted_config_ordinal_count != state.provenance.config_fields.size()) {
    append_diagnostic(result, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH");
    return false;
  }
  for (std::size_t index = 0U; index < replacements.size(); ++index) {
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
    const std::optional<LifecycleTransactionOptions>& lifecycle_options,
    ParamSchemaHash admission_context_digest,
    const std::optional<ParamOverrideSet>& parameter_overrides) {
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
  std::vector<ParameterGenerationZeroOrigin> baseline_origins;
  std::vector<ParameterProtectedValueDigest> baseline_digests;
  operator_paths.reserve(factories.operators.size());
  baseline_origins.reserve(factories.operators.size());
  baseline_digests.reserve(factories.operators.size());
  for (const auto& executable : factories.operators) {
    operator_paths.push_back(operator_path(snapshot, executable.vertex));
    if (!executable.factory) {
      baseline_origins.push_back(ParameterGenerationZeroOrigin::kDefault);
      baseline_digests.push_back({});
      continue;
    }
    baseline_origins.push_back(executable.factory->has_explicit_parameter_baseline()
                                   ? ParameterGenerationZeroOrigin::kTypedBaseline
                                   : ParameterGenerationZeroOrigin::kDefault);
    baseline_digests.push_back(parameter_value_digest(executable.factory->parameter_type(),
                                                      executable.factory->parameter_schema(),
                                                      executable.factory->initial_parameter_value(),
                                                      executable.factory->has_no_params()));
  }
  std::vector<std::vector<ParamOverrideFieldOriginRecord>> accepted_contributions(
      factories.operators.size());
  if (!apply_parameter_overrides(snapshot,
                                 factories,
                                 operator_paths,
                                 baseline_digests,
                                 admission_context_digest,
                                 parameter_overrides,
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
        .baseline_origin = baseline_origins[factory_index],
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
      for (const ParamFieldDescriptor& descriptor : schema->fields) {
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

        ParameterGenerationZeroOrigin field_origin = baseline_origins[factory_index];
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
            contribution->config_provenance_index && parameter_overrides) {
          const auto& provenance =
              ParamOverrideSetAccess::state(*parameter_overrides).provenance.config_fields;
          const std::size_t source_index = *contribution->config_provenance_index;
          if (source_index < provenance.size()) {
            config_source = provenance[source_index];
          }
        }
        result.generation_zero_fields.push_back(ParameterGenerationZeroFieldRecord{
            .instance = instance,
            .field = descriptor.id,
            .full_id = descriptor.full_id,
            .origin = field_origin,
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
