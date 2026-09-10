/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_operator_plan_validation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "../compiler/execution_plan_state.hpp"
#include "../core/lifecycle_capacity_derivation.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] bool plan_ref_matches_legacy_operator(const OperatorRef& ref,
                                                    std::size_t operator_index,
                                                    PlanGenerationId plan_generation) noexcept {
  return ref.id == static_cast<OperatorInstanceId>(operator_index + 1U) &&
         ref.plan_generation == plan_generation && ref.run_generation == 0U;
}

[[nodiscard]] bool valid_plan_operator_ref(const OperatorRef& ref, std::size_t operator_count,
                                           PlanGenerationId plan_generation) noexcept {
  return ref.plan_bound() && ref.plan_generation == plan_generation && ref.id != 0U &&
         ref.id <= operator_count;
}

[[nodiscard]] RequiredLifecycleCapacity expected_lifecycle_required_capacity(
    const ExecutionPlanState& plan, const LifecycleTransactionCapacityValues& configured) {
  const LifecycleTransactionCapacityValues parameter_required =
      plan.parameter_runtime_capacity ? plan.parameter_runtime_capacity->required_lifecycle_capacity
                                      : LifecycleTransactionCapacityValues{};
  std::vector<std::uint8_t> membership_marks(plan.lifecycle_restart_closures.size());
  return derive_required_lifecycle_capacity(
      configured,
      admitted_lifecycle_restart_capacity(plan.lifecycle_restart_targets,
                                          plan.lifecycle_restart_closures,
                                          plan.plan_generation,
                                          membership_marks),
      parameter_required);
}

[[nodiscard]] bool lifecycle_checkpoints_well_formed(const ExecutionPlanState& plan,
                                                     std::size_t operator_count) {
  constexpr std::size_t kLifecycleStageCount =
      static_cast<std::size_t>(LifecycleStage::kRelease) + 1U;
  if (plan.lifecycle_checkpoints.size() % kLifecycleStageCount != 0U ||
      plan.lifecycle_checkpoints.size() / kLifecycleStageCount != operator_count) {
    return false;
  }

  for (std::size_t stage_index = 0U; stage_index < kLifecycleStageCount; ++stage_index) {
    std::vector<std::uint8_t> seen(operator_count, 0U);
    std::uint32_t previous_level{};
    OperatorInstanceId previous_operator{};
    for (std::size_t operator_index = 0U; operator_index < operator_count; ++operator_index) {
      const auto& checkpoint =
          plan.lifecycle_checkpoints[stage_index * operator_count + operator_index];
      if (!valid_plan_operator_ref(
              checkpoint.operator_instance, operator_count, plan.plan_generation) ||
          checkpoint.stage != static_cast<LifecycleStage>(stage_index) ||
          (checkpoint.required && !checkpoint.callback_present)) {
        return false;
      }
      const std::size_t identity_index = checkpoint.operator_instance.id - 1U;
      if (seen[identity_index] != 0U ||
          (operator_index == 0U && checkpoint.stable_topological_level != 0U) ||
          checkpoint.stable_topological_level < previous_level ||
          checkpoint.stable_topological_level > previous_level + 1U ||
          (checkpoint.stable_topological_level == previous_level && operator_index != 0U &&
           checkpoint.operator_instance.id <= previous_operator)) {
        return false;
      }
      seen[identity_index] = 1U;
      previous_level = checkpoint.stable_topological_level;
      previous_operator = checkpoint.operator_instance.id;
    }
  }
  return true;
}

[[nodiscard]] bool lifecycle_targets_well_formed(const ExecutionPlanState& plan,
                                                 std::size_t operator_count) noexcept {
  if (plan.lifecycle_restart_targets.size() != operator_count) {
    return false;
  }
  for (std::size_t operator_index = 0U; operator_index < operator_count; ++operator_index) {
    const auto& target = plan.lifecycle_restart_targets[operator_index];
    if (!plan_ref_matches_legacy_operator(target.target, operator_index, plan.plan_generation)) {
      return false;
    }
    if (plan.lifecycle_transaction_capacity.has_value()) {
      if (target.disposition == LifecycleRestartTargetDisposition::kDisabled) {
        return false;
      }
    } else if (target.disposition != LifecycleRestartTargetDisposition::kDisabled ||
               target.self_allowed || target.external_control_allowed ||
               target.parameter_origin_allowed || target.allowed_from_stage_mask != 0U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool lifecycle_dependencies_well_formed(const ExecutionPlanState& plan,
                                                      std::size_t operator_count) noexcept {
  return std::all_of(
      plan.lifecycle_dependencies.begin(),
      plan.lifecycle_dependencies.end(),
      [&](const auto& dependency) {
        const bool key_shape = dependency.kind == LifecycleEdgeKind::kBefore
                                   ? dependency.dependency_key.empty()
                                   : dependency.kind == LifecycleEdgeKind::kDependencyKey &&
                                         !dependency.dependency_key.empty();
        return valid_plan_operator_ref(dependency.before, operator_count, plan.plan_generation) &&
               valid_plan_operator_ref(dependency.after, operator_count, plan.plan_generation) &&
               dependency.before.id != dependency.after.id &&
               is_valid_lifecycle_stage(dependency.stage) && key_shape;
      });
}

[[nodiscard]] bool lifecycle_permissions_well_formed(const ExecutionPlanState& plan,
                                                     std::size_t operator_count) noexcept {
  if (!plan.lifecycle_transaction_capacity && !plan.lifecycle_restart_permissions.empty()) {
    return false;
  }

  OperatorInstanceId previous_requester{};
  OperatorInstanceId previous_target{};
  for (const auto& permission : plan.lifecycle_restart_permissions) {
    if (!valid_plan_operator_ref(permission.requester, operator_count, plan.plan_generation) ||
        !valid_plan_operator_ref(permission.target, operator_count, plan.plan_generation) ||
        permission.requester.id == permission.target.id ||
        std::tuple{permission.requester.id, permission.target.id} <=
            std::tuple{previous_requester, previous_target}) {
      return false;
    }
    previous_requester = permission.requester.id;
    previous_target = permission.target.id;
  }
  return true;
}

[[nodiscard]] bool lifecycle_mutable_fields_well_formed(const ExecutionPlanState& plan,
                                                        std::size_t operator_count) noexcept {
  if (!plan.lifecycle_transaction_capacity && !plan.lifecycle_mutable_fields.empty()) {
    return false;
  }

  OperatorInstanceId previous_owner{};
  ParamFieldId previous_field{};
  for (const auto& field : plan.lifecycle_mutable_fields) {
    if (!valid_plan_operator_ref(field.owner, operator_count, plan.plan_generation) ||
        field.field == 0U || !is_lifecycle_forward_stage(field.from_stage) ||
        std::tuple{field.owner.id, field.field} <= std::tuple{previous_owner, previous_field}) {
      return false;
    }
    const auto planned_field = std::find_if(
        plan.parameter_fields.begin(), plan.parameter_fields.end(), [&](const auto& row) {
          return row.owner == field.owner && row.field == field.field;
        });
    if (planned_field == plan.parameter_fields.end() ||
        planned_field->mutability != ParamMutability::kLifecycle) {
      return false;
    }
    previous_owner = field.owner.id;
    previous_field = field.field;
  }
  return true;
}

// Keep closure membership, dependency containment, and per-stage widening checks in one
// fail-closed validator because they jointly authenticate the frozen restart transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool lifecycle_closures_well_formed(const ExecutionPlanState& plan,
                                                  std::size_t operator_count) noexcept {
  constexpr std::uint8_t kAllRestartStagesMask = 0x3fU;
  if (plan.lifecycle_restart_closures.size() != operator_count) {
    return false;
  }

  for (std::size_t target_index = 0U; target_index < operator_count; ++target_index) {
    const auto& closure = plan.lifecycle_restart_closures[target_index];
    if (!plan_ref_matches_legacy_operator(closure.target, target_index, plan.plan_generation) ||
        closure.closure.empty() || closure.valid_requested_stage_mask != kAllRestartStagesMask ||
        closure.required_fenced_operators != closure.closure.size()) {
      return false;
    }

    OperatorInstanceId previous{};
    bool contains_target = false;
    for (const OperatorRef member : closure.closure) {
      if (!valid_plan_operator_ref(member, operator_count, plan.plan_generation) ||
          member.id <= previous) {
        return false;
      }
      contains_target = contains_target || member == closure.target;
      previous = member.id;
    }
    if (!contains_target) {
      return false;
    }
    for (const auto& dependency : plan.lifecycle_dependencies) {
      const bool contains_before =
          std::binary_search(closure.closure.begin(),
                             closure.closure.end(),
                             dependency.before,
                             [](auto lhs, auto rhs) { return lhs.id < rhs.id; });
      const bool contains_after = std::binary_search(
          closure.closure.begin(), closure.closure.end(), dependency.after, [](auto lhs, auto rhs) {
            return lhs.id < rhs.id;
          });
      if (contains_before && !contains_after) {
        return false;
      }
    }

    std::uint32_t maximum_quiesce{};
    std::uint32_t maximum_rollback{};
    for (std::size_t stage_index = 0U; stage_index < closure.stage_resolutions.size();
         ++stage_index) {
      const auto& resolution = closure.stage_resolutions[stage_index];
      const LifecycleStage requested = static_cast<LifecycleStage>(stage_index);
      if (resolution.requested_stage != requested ||
          !is_lifecycle_forward_stage(resolution.effective_stage) ||
          static_cast<std::uint8_t>(resolution.effective_stage) > stage_index ||
          (resolution.effective_stage == requested) !=
              (resolution.reason == LifecycleRestartWideningReason::kNone) ||
          (resolution.effective_stage != requested &&
           resolution.reason != LifecycleRestartWideningReason::kDependencyStage) ||
          resolution.required_structural_event_records != kLifecycleStructuralEventsPerRequest) {
        return false;
      }
      maximum_quiesce = std::max(maximum_quiesce, resolution.required_quiesce_records);
      maximum_rollback = std::max(maximum_rollback, resolution.required_rollback_records);
    }
    if (closure.required_quiesce_records != maximum_quiesce ||
        closure.required_rollback_records != maximum_rollback) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool lifecycle_capacity_well_formed(const ExecutionPlanState& plan) {
  if (!plan.lifecycle_transaction_capacity) {
    return true;
  }
  const auto& capacity = *plan.lifecycle_transaction_capacity;
  const RequiredLifecycleCapacity required =
      expected_lifecycle_required_capacity(plan, capacity.configured);
  if (!lifecycle_capacity_values_positive(capacity.configured) || required.arithmetic_overflow ||
      required.invalid_input ||
      !lifecycle_capacity_values_equal(capacity.required_minimum, required.values) ||
      !lifecycle_capacity_values_satisfy(capacity.configured, capacity.required_minimum) ||
      capacity.derivation_reason != kCanonicalGraphLifecycleCapacityDerivationReason ||
      capacity.resolved_parameter_update_policy == ParamRuntimeUpdatePolicy::kProfileDefault) {
    return false;
  }

  if (!plan.parameter_runtime_capacity) {
    return capacity.resolved_parameter_update_policy == ParamRuntimeUpdatePolicy::kDisabled &&
           !capacity.parameter_origin_enabled &&
           capacity.required_generation_slots_per_operator == 0U &&
           capacity.required_parameter_arena_bytes == 0U &&
           capacity.required_parameter_audit_records == 0U;
  }
  const auto& parameters = *plan.parameter_runtime_capacity;
  return capacity.resolved_parameter_update_policy == parameters.resolved_update_policy &&
         capacity.parameter_origin_enabled == parameters.lifecycle_updates_enabled &&
         capacity.required_generation_slots_per_operator ==
             parameters.generation_slots_per_operator &&
         capacity.required_parameter_arena_bytes == parameters.parameter_arena_bytes &&
         capacity.required_parameter_audit_records == parameters.audit_capacity &&
         lifecycle_capacity_values_satisfy(capacity.required_minimum,
                                           parameters.required_lifecycle_capacity);
}

[[nodiscard]] bool lifecycle_target_authority_well_formed(const ExecutionPlanState& plan) noexcept {
  constexpr std::uint8_t kAllRestartStagesMask = 0x3fU;
  for (const auto& target : plan.lifecycle_restart_targets) {
    const bool has_cross_permission =
        std::any_of(plan.lifecycle_restart_permissions.begin(),
                    plan.lifecycle_restart_permissions.end(),
                    [&](const auto& permission) { return permission.target == target.target; });
    const bool has_mutable_field =
        std::any_of(plan.lifecycle_mutable_fields.begin(),
                    plan.lifecycle_mutable_fields.end(),
                    [&](const auto& field) { return field.owner == target.target; });
    const bool has_authority = target.self_allowed || target.external_control_allowed ||
                               target.parameter_origin_allowed || has_cross_permission;

    if (!plan.lifecycle_transaction_capacity) {
      if (target.disposition != LifecycleRestartTargetDisposition::kDisabled || has_authority ||
          target.allowed_from_stage_mask != 0U) {
        return false;
      }
      continue;
    }
    if (target.disposition == LifecycleRestartTargetDisposition::kNotRequested) {
      if (has_authority || target.allowed_from_stage_mask != 0U) {
        return false;
      }
      continue;
    }
    if (target.disposition != LifecycleRestartTargetDisposition::kAdmitted || !has_authority ||
        target.allowed_from_stage_mask == 0U ||
        (target.allowed_from_stage_mask & ~kAllRestartStagesMask) != 0U ||
        target.parameter_origin_allowed != has_mutable_field) {
      return false;
    }
  }
  return true;
}

template <typename Record>
[[nodiscard]] bool parameter_owner_groups_well_formed(std::span<const Record> records,
                                                      std::size_t operator_count,
                                                      PlanGenerationId plan_generation) {
  std::vector<bool> closed_owner(operator_count + 1U);
  OperatorInstanceId active_owner{};
  for (const Record& record : records) {
    const OperatorRef owner = record.owner;
    if (!owner.plan_bound() || owner.plan_generation != plan_generation || owner.id == 0U ||
        owner.id > operator_count) {
      return false;
    }
    if (owner.id == active_owner) {
      continue;
    }
    if (closed_owner[owner.id]) {
      return false;
    }
    if (active_owner != 0U) {
      closed_owner[active_owner] = true;
    }
    active_owner = owner.id;
  }
  return true;
}

}  // namespace

[[nodiscard]] bool lifecycle_plan_well_formed(const ExecutionPlanState& plan,
                                              std::size_t operator_count) {
  if (!lifecycle_checkpoints_well_formed(plan, operator_count) ||
      !lifecycle_targets_well_formed(plan, operator_count) ||
      !lifecycle_dependencies_well_formed(plan, operator_count) ||
      !lifecycle_permissions_well_formed(plan, operator_count) ||
      !lifecycle_mutable_fields_well_formed(plan, operator_count) ||
      !lifecycle_closures_well_formed(plan, operator_count) ||
      !lifecycle_capacity_well_formed(plan) || !lifecycle_target_authority_well_formed(plan)) {
    return false;
  }
  return true;
}

[[nodiscard]] expected<std::vector<std::string>, Error> planned_operator_path(
    const ExecutionPlanState& plan, VertexId owner) {
  const auto vertex = std::find_if(plan.graph.vertices().begin(),
                                   plan.graph.vertices().end(),
                                   [owner](const auto& row) { return row.id == owner; });
  if (vertex == plan.graph.vertices().end() || vertex->name.empty()) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_PARAMETER_INVENTORY_MISMATCH"));
  }
  const auto graph = std::find_if(plan.graph.graphs().begin(),
                                  plan.graph.graphs().end(),
                                  [&](const auto& row) { return row.id == vertex->graph; });
  if (graph == plan.graph.graphs().end()) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_PARAMETER_INVENTORY_MISMATCH"));
  }
  std::vector<std::string> path;
  if (graph->path.has_value()) {
    // The optional belongs to the immutable snapshot and cannot change between this guard and use.
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    const GraphInstancePath& graph_path = graph->path.value();
    path.assign(graph_path.segments().begin(), graph_path.segments().end());
    // NOLINTEND(bugprone-unchecked-optional-access)
  }
  path.push_back(vertex->name);
  return path;
}

bool parameter_owner_inventories_well_formed(const ExecutionPlanState& plan,
                                             std::size_t operator_count) {
  return parameter_owner_groups_well_formed(
             std::span{plan.parameter_fields}, operator_count, plan.plan_generation) &&
         parameter_owner_groups_well_formed(
             std::span{plan.parameter_derived_terms}, operator_count, plan.plan_generation);
}

}  // namespace holoscan::detail
