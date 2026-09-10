/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_operator_reconstruction.hpp"

#include "runtime_operator_plan_validation.hpp"
#include "runtime_operator_replay_validation.hpp"
#include "runtime_work_inventory.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/profiler/nvtx3.hpp>

#include "../channel/endpoint_catalog_state.hpp"
#include "../compiler/closed_island_activation_plan.hpp"
#include "../compiler/execution_plan_lookup.hpp"
#include "../compiler/inline_partition_execution_plan.hpp"
#include "../compiler/partition_runner_plan.hpp"
#include "../core/diagnostic_registry.hpp"
#include "../core/lifecycle_capacity_derivation.hpp"
#include "../core/operator_runtime_access.hpp"
#include "../core/operator_spec_compiler_access.hpp"
#include "../core/plan_record_contracts.hpp"
#include "../core/readiness_source_internal.hpp"
#include "../core/source_id_internal.hpp"
#include "../core/temporal_contract_internal.hpp"
#include "../threading/thread_placement_contracts.hpp"
#include "dispatcher.hpp"
#include "operator_profile_labels.hpp"
#include "parameter_control_state.hpp"
#include "runtime_clock_reconstruction.hpp"
#include "runtime_worker.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] expected<RuntimeOperatorInstance, Error> reconstruct_operator(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record,
    std::uint32_t factory_slot, RuntimeReplayMismatchDetail* replay_mismatch) {
  if (record.factory == nullptr || record.operator_id == 0U) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_FACTORY_INVALID"));
  }
  // Construction, setup, and contract replay all execute user code on the run-owned worker.
  // Owning control calls could otherwise wait for the coordinator that this reconstruction path
  // has not yet attached, so expose only callback-safe borrowed seams throughout reconstruction.
  const ParamGeneralControlContextScope general_control_scope;
  auto created = record.factory->create();
  if (!created) {
    return make_unexpected(operator_reconstruction_error(created.error().code,
                                                         "RUNTIME_OPERATOR_RECONSTRUCTION_FAILED"));
  }
  if (created.value() == nullptr) {
    return make_unexpected(operator_reconstruction_error(ErrorCode::kFailure,
                                                         "RUNTIME_OPERATOR_RECONSTRUCTION_FAILED"));
  }
  auto reconstructed_operator = std::move(created).value();

  auto replay = std::make_unique<OperatorSpec>();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    reconstructed_operator->setup(*replay);
    if (auto validated = validate_replayed_operator(
            plan, record, *reconstructed_operator, *replay, factory_slot, replay_mismatch);
        !validated) {
      return make_unexpected(std::move(validated.error()));
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_SETUP_FAILED"));
  }
#endif

  const ExecutionPartitionRecord* const partition =
      find_execution_partition_by_member(plan, record.vertex);
  if (partition == nullptr) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_INVENTORY_MISMATCH"));
  }

  // Operator Centric Logging, Lookup for graph vertex name for vars op_name and profile_domain in
  // RuntimeOperatorInstance Iterate through graph vertices (operators) to find one that matches
  // current op ID and then store to vertex
  const auto vertex = std::find_if(plan.graph.vertices().begin(),
                                   plan.graph.vertices().end(),
                                   [&](const auto& row) { return row.id == record.vertex; });
  // Get name of vertex
  const std::string op_name =
      vertex != plan.graph.vertices().end() ? vertex->name : std::string{"unknown-operator"};

  std::string profile_name = operator_profile_escape_component(op_name);
  if (vertex != plan.graph.vertices().end()) {
    profile_name = operator_profile_operator_label(plan, vertex->graph, op_name);
  }

  const std::string compute_nvtx_label = profile_name + ":compute";

  auto tags =
      std::make_unique<TagAllocator>(1U, plan.executable_factories.test_initial_tag_sequence);
  const TemporalTriggerPlanRecord* const temporal_trigger =
      find_temporal_trigger_by_owner(plan, record.operator_id);
  const PredicateTriggerPlanRecord* const predicate_trigger =
      temporal_trigger != nullptr && temporal_trigger->kind == TemporalRecordTriggerKind::kPredicate
          ? find_predicate_trigger_by_owner(plan, temporal_trigger->owner)
          : nullptr;
  RuntimeOperatorInstance instance{
      .vertex = record.vertex,
      .operator_id = record.operator_id,
      .partition = partition->id,
      .execution_lane = partition->execution_lane,
      .op = std::move(reconstructed_operator),
      .replay_spec = std::move(replay),
      .tags = std::move(tags),
      .temporal_trigger = temporal_trigger,
      .predicate_trigger = predicate_trigger,
      .compute_nvtx_message = nvtx3::message{nvtxDomainRegisterStringA(
          holoscan::profiler::operator_profile_domain(), compute_nvtx_label.c_str())},
  };
  return instance;
}

[[nodiscard]] expected<void, Error> build_operator_indexes(
    const ExecutionPlanState& plan, ProcessGroupId process_group,
    RuntimeOperatorReconstruction& reconstruction) {
  constexpr std::size_t kNoReceiver = std::numeric_limits<std::size_t>::max();
  const auto vertices = plan.graph.vertices();
  reconstruction.receiver_by_vertex.assign(vertices.size() + 1U, kNoReceiver);
  reconstruction.receiver_by_operator_id.assign(plan.executable_factories.operators.size() + 1U,
                                                kNoReceiver);

  for (std::size_t receiver = 0U; receiver < reconstruction.operators.size(); ++receiver) {
    const RuntimeOperatorInstance& instance = reconstruction.operators[receiver];
    if (!instance.vertex || instance.vertex.value >= reconstruction.receiver_by_vertex.size() ||
        instance.operator_id == 0U ||
        static_cast<std::size_t>(instance.operator_id) >=
            reconstruction.receiver_by_operator_id.size()) {
      return make_unexpected(operator_reconstruction_error(ErrorCode::kFailure,
                                                           "RUNTIME_OPERATOR_INVENTORY_MISMATCH"));
    }
    const VertexRecord& vertex = vertices[static_cast<std::size_t>(instance.vertex.value - 1U)];
    const ExecutableOperatorFactoryRecord& factory =
        plan.executable_factories.operators[static_cast<std::size_t>(instance.operator_id - 1U)];
    const std::size_t vertex_slot = static_cast<std::size_t>(instance.vertex.value);
    const std::size_t operator_slot = static_cast<std::size_t>(instance.operator_id);
    if (vertex.id != instance.vertex || vertex.kind != VertexKind::kOperator ||
        factory.vertex != instance.vertex || factory.operator_id != instance.operator_id ||
        (process_group && !worker_owns_vertex(plan, process_group, instance.vertex)) ||
        reconstruction.receiver_by_vertex[vertex_slot] != kNoReceiver ||
        reconstruction.receiver_by_operator_id[operator_slot] != kNoReceiver) {
      return make_unexpected(operator_reconstruction_error(ErrorCode::kFailure,
                                                           "RUNTIME_OPERATOR_INVENTORY_MISMATCH"));
    }
    reconstruction.receiver_by_vertex[vertex_slot] = receiver;
    reconstruction.receiver_by_operator_id[operator_slot] = receiver;
  }

  for (const ExecutableOperatorFactoryRecord& factory : plan.executable_factories.operators) {
    if (process_group && !worker_owns_vertex(plan, process_group, factory.vertex)) {
      continue;
    }
    if (!factory.vertex || factory.vertex.value >= reconstruction.receiver_by_vertex.size() ||
        factory.operator_id == 0U ||
        static_cast<std::size_t>(factory.operator_id) >=
            reconstruction.receiver_by_operator_id.size()) {
      return make_unexpected(operator_reconstruction_error(ErrorCode::kFailure,
                                                           "RUNTIME_OPERATOR_INVENTORY_MISMATCH"));
    }
    const std::size_t vertex_receiver =
        reconstruction.receiver_by_vertex[static_cast<std::size_t>(factory.vertex.value)];
    const std::size_t operator_receiver =
        reconstruction.receiver_by_operator_id[static_cast<std::size_t>(factory.operator_id)];
    if (vertex_receiver == kNoReceiver || vertex_receiver != operator_receiver) {
      return make_unexpected(operator_reconstruction_error(ErrorCode::kFailure,
                                                           "RUNTIME_OPERATOR_INVENTORY_MISMATCH"));
    }
  }
  return {};
}

}  // namespace

// Reconstruction validates every frozen Operator inventory before publishing any run-owned state,
// then builds Operators, indexes, and Parameter bindings in dependency order.
// NOLINTBEGIN(readability-function-cognitive-complexity)
expected<RuntimeOperatorReconstruction, Error> reconstruct_runtime_operators(
    const ExecutionPlanState& plan, RunGenerationId run_generation, ProcessGroupId process_group,
    RuntimeReplayMismatchDetail* replay_mismatch) {
  if (replay_mismatch != nullptr) {
    *replay_mismatch = RuntimeReplayMismatchDetail{};
  }
  if (run_generation == 0U) {
    return make_unexpected(operator_reconstruction_error(ErrorCode::kInvalidArgument,
                                                         "RUNTIME_RUN_GENERATION_INVALID"));
  }
  if (process_group && std::none_of(plan.vertex_process_placements.begin(),
                                    plan.vertex_process_placements.end(),
                                    [&](const auto& placement) {
                                      return placement.process_group == process_group;
                                    })) {
    // This remains an Error-only defense in depth for callers below the RunSession boundary.
    // start_run_session() owns the registered, structured startup diagnostic for an invalid
    // explicit process group before executor construction can begin.
    return make_unexpected(operator_reconstruction_error(
        ErrorCode::kInvalidArgument, "RUNTIME_PROCESS_GROUP_SELECTION_INVALID"));
  }
  const auto planned_operator_count = static_cast<std::size_t>(std::count_if(
      plan.graph.vertices().begin(), plan.graph.vertices().end(), [](const auto& vertex) {
        return vertex.kind == VertexKind::kOperator;
      }));
  if (!lifecycle_plan_well_formed(plan, planned_operator_count)) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_LIFECYCLE_PLAN_INVALID"));
  }

  // Resolve and match the executing package before reconstructing an Operator.
  // Operator reconstruction replays setup(), so performing this cold gate first
  // guarantees that no user callback observes an incompatible package.
  const RuntimePackageCapabilityManifest& generated = runtime_package_capability_manifest();
  const std::string canonical = runtime_package_capability_canonical_json_v1(generated, true);
  auto package = resolve_runtime_package_capability_provider(
      RuntimePackageCapabilityProviderInput{.embedded_bytes = canonical});
  if (!package) {
    return make_unexpected(std::move(package.error()));
  }
  auto matched = match_runtime_package_capabilities(plan.profile_manifest,
                                                    package.value(),
                                                    RuntimePackageCapabilityMatchContext{
                                                        .startup_attempt = 1U,
                                                        .plan_generation = plan.plan_generation,
                                                        .run_generation = run_generation,
                                                    });
  if (!matched) {
    return make_unexpected(std::move(matched.error()));
  }
  if (!matched->compatible()) {
    return make_unexpected(Error{
        ErrorCode::kIncompatible,
        std::string{runtime_package_capability_match_token(matched->result)},
    });
  }
  // Parameter owner-local inventories are consumed as contiguous spans during reconstruction.
  // Authenticate their structural shape before validating runtime backends so a malformed frozen
  // inventory cannot be masked by an unrelated unavailable readiness or placement binding.
  if (plan.operator_parameters.size() != planned_operator_count ||
      !plan.parameter_runtime_capacity.has_value() ||
      !parameter_owner_inventories_well_formed(plan, planned_operator_count)) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_PARAMETER_INVENTORY_MISMATCH"));
  }
  const ParameterRuntimeCapacityPlanRecord parameter_runtime_capacity =
      plan.parameter_runtime_capacity.value_or(ParameterRuntimeCapacityPlanRecord{});
  if (auto supported = validate_foundation_plan(plan, planned_operator_count); !supported) {
    return make_unexpected(std::move(supported.error()));
  }
  if (plan.executable_factories.operators.size() != planned_operator_count) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_INVENTORY_MISMATCH"));
  }
  // Authenticate the compiler-derived partition runner only after the executable inventory has
  // retained its more specific diagnostic, but before reconstructing any Operator or invoking any
  // lifecycle callback. The runner record is a cold GraphExec foundation fact; Runtime must never
  // rebuild or silently repair a mutated copy.
  if (!valid_partition_runner_plan(plan)) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
  auto clock_inventory = validate_runtime_clock_reconstruction(plan);
  if (!clock_inventory) {
    return make_unexpected(std::move(clock_inventory.error()));
  }
  RuntimeOperatorReconstruction result;
  result.package_capability_report = std::move(matched).value();

  // A compiler-accepted plan without callback owners still passes exact package compatibility
  // and frozen-plan validation. This includes interface-only topology plans: imported and
  // exported topics remain visible to the execution cover without requiring an Operator replay.
  // Once those cold gates succeed, construct no package-backed Operator or clock that the run can
  // never observe.
  if (!has_plan_callback_owners(plan)) {
    if (auto indexed = build_operator_indexes(plan, process_group, result); !indexed) {
      return make_unexpected(std::move(indexed.error()));
    }
    auto parameter_updates = ParameterUpdateCoordinator::create(
        std::span<const ParameterRuntimeOperatorBinding>{}, parameter_runtime_capacity);
    if (!parameter_updates) {
      return make_unexpected(operator_reconstruction_error(
          parameter_updates.error().code, "RUNTIME_PARAMETER_BINDING_UNAVAILABLE"));
    }
    result.parameter_updates = std::move(parameter_updates).value();
    return result;
  }

  result.operators.reserve(plan.executable_factories.operators.size());
  for (std::size_t index = 0U; index < plan.executable_factories.operators.size(); ++index) {
    const auto& record = plan.executable_factories.operators[index];
    if (process_group && !worker_owns_vertex(plan, process_group, record.vertex)) {
      continue;
    }
    auto reconstructed =
        reconstruct_operator(plan, record, static_cast<std::uint32_t>(index + 1U), replay_mismatch);
    if (!reconstructed) {
      return make_unexpected(std::move(reconstructed.error()));
    }
    result.operators.push_back(std::move(reconstructed).value());
  }
  if (auto indexed = build_operator_indexes(plan, process_group, result); !indexed) {
    return make_unexpected(std::move(indexed.error()));
  }
  std::vector<ParameterRuntimeOperatorBinding> parameter_bindings;
  parameter_bindings.reserve(result.operators.size());
  for (RuntimeOperatorInstance& instance : result.operators) {
    const OperatorRef plan_owner{instance.operator_id, plan.plan_generation, 0U};
    const OperatorParameterPlanRecord* const parameter_plan =
        find_operator_parameters_by_owner(plan, plan_owner);
    if (parameter_plan == nullptr) {
      return make_unexpected(operator_reconstruction_error(ErrorCode::kFailure,
                                                           "RUNTIME_PARAMETER_INVENTORY_MISMATCH"));
    }
    const ParamTypeDescriptor& parameter_type = OperatorRuntimeAccess::parameter_type(*instance.op);
    instance.static_parameter_header = PinnedParameterHeader{
        .value = OperatorRuntimeAccess::parameter_address(*instance.op),
        .local_type_token = parameter_type.local_type_token,
        .generation = 0U,
        .schema_hash = parameter_plan->contract_hash,
        .slot = 0U,
    };
    instance.parameter_derived_terms = parameter_derived_terms_by_owner(plan, plan_owner);
    instance.runtime_parameter_updates_enabled = parameter_plan->runtime_updates_enabled;
    instance.lifecycle_parameter_updates_enabled = parameter_plan->lifecycle_updates_enabled;
    auto operator_path = planned_operator_path(plan, instance.vertex);
    if (!operator_path) {
      return make_unexpected(std::move(operator_path.error()));
    }
    parameter_bindings.push_back(ParameterRuntimeOperatorBinding{
        .run_bound_owner =
            OperatorRef{
                .id = instance.operator_id,
                .plan_generation = plan.plan_generation,
                .run_generation = run_generation,
            },
        .operator_path = std::move(operator_path).value(),
        .plan_digest = plan.digest,
        .plan = parameter_plan,
        .fields = parameter_fields_by_owner(plan, plan_owner),
        .derived_terms = instance.parameter_derived_terms,
        .type = &parameter_type,
        .schema = OperatorRuntimeAccess::parameter_schema(*instance.op),
        .initial_value = OperatorRuntimeAccess::parameter_address(*instance.op),
    });
  }
  auto parameter_updates = ParameterUpdateCoordinator::create(
      parameter_bindings,
      parameter_runtime_capacity,
      static_cast<std::uint32_t>(plan.operator_parameters.size()));
  if (!parameter_updates) {
    return make_unexpected(operator_reconstruction_error(parameter_updates.error().code,
                                                         "RUNTIME_PARAMETER_BINDING_UNAVAILABLE"));
  }
  result.parameter_updates = std::move(parameter_updates).value();

  // Generation-zero-only values move into coordinator-owned immutable storage so terminal
  // Parameter handles remain valid after GraphExec teardown. Rebind the cached compute header
  // once here so compute and lifecycle callbacks observe that same object without hot-path pins.
  for (RuntimeOperatorInstance& instance : result.operators) {
    if (instance.runtime_parameter_updates_enabled ||
        instance.lifecycle_parameter_updates_enabled) {
      continue;
    }
    const OperatorRef owner{
        .id = instance.operator_id,
        .plan_generation = plan.plan_generation,
        .run_generation = run_generation,
    };
    auto published = result.parameter_updates->pin_published(owner);
    if (!published) {
      return make_unexpected(operator_reconstruction_error(
          ErrorCode::kFailure, "RUNTIME_PARAMETER_BINDING_UNAVAILABLE"));
    }
    const PinnedParameterHeader* const header = published->header();
    if (header == nullptr || header->generation != 0U) {
      result.parameter_updates->release(*published);
      return make_unexpected(operator_reconstruction_error(
          ErrorCode::kFailure, "RUNTIME_PARAMETER_BINDING_UNAVAILABLE"));
    }
    instance.static_parameter_header = *header;
    result.parameter_updates->release(*published);
  }
  return result;
}
// NOLINTEND(readability-function-cognitive-complexity)

}  // namespace holoscan::detail
