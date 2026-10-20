/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lifecycle_plan_lowering.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../core/diagnostic_registry.hpp"
#include "../core/lifecycle_capacity_derivation.hpp"
#include "../core/lifecycle_graph_lowering.hpp"
#include "../core/lifecycle_kernel.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

constexpr std::uint8_t kAllForwardStagesMask = 0x3FU;

[[nodiscard]] constexpr ReasonId reason_id(StructuralDiagnosticId id) noexcept {
  return static_cast<ReasonId>(id);
}

void append_diagnostic(LifecyclePlanLowering& result, std::string_view token,
                       DiagnosticSource source = {}, std::string json_pointer = {},
                       std::string required = {}, std::string provided = {}) {
  static_cast<void>(
      append_unique_plan_lowering_diagnostic(result.diagnostics,
                                             LifecyclePlanDiagnostic{
                                                 .token = std::string(token),
                                                 .source = source,
                                                 .json_pointer = std::move(json_pointer),
                                                 .required = std::move(required),
                                                 .provided = std::move(provided),
                                             }));
  result.accepted = false;
}

[[nodiscard]] OperatorRef operator_ref(LifecycleNodeId node_id,
                                       PlanGenerationId plan_generation) noexcept {
  return OperatorRef{
      .id = node_id,
      .plan_generation = plan_generation,
      .run_generation = 0U,
  };
}

[[nodiscard]] bool profile_supports_active_transactions(Profile profile) noexcept {
  switch (profile) {
    case Profile::kStandard:
    case Profile::kRobotics:
    case Profile::kResearch:
      return true;
    case Profile::kRealTime:
    case Profile::kRegulated:
      return false;
  }
  return false;
}

[[nodiscard]] bool all_operator_profiles_support_active_transactions(
    const GraphSnapshotData& snapshot,
    std::span<const std::optional<Profile>> effective_vertex_profiles) noexcept {
  return std::all_of(snapshot.vertices.begin(), snapshot.vertices.end(), [&](const auto& vertex) {
    if (vertex.kind != VertexKind::kOperator) {
      return true;
    }
    const auto profile = effective_profile_for_vertex(effective_vertex_profiles, vertex.id);
    return profile.has_value() && profile_supports_active_transactions(*profile);
  });
}

[[nodiscard]] bool restart_origin_valid(LifecycleRestartOrigin origin) noexcept {
  switch (origin) {
    case LifecycleRestartOrigin::kSelf:
    case LifecycleRestartOrigin::kExternal:
    case LifecycleRestartOrigin::kOperatorRequest:
      return true;
  }
  return false;
}

[[nodiscard]] std::string_view restart_origin_name(LifecycleRestartOrigin origin) noexcept {
  switch (origin) {
    case LifecycleRestartOrigin::kSelf:
      return "self";
    case LifecycleRestartOrigin::kExternal:
      return "external";
    case LifecycleRestartOrigin::kOperatorRequest:
      return "operator_request";
  }
  return "invalid";
}

[[nodiscard]] std::string_view profile_name(Profile profile) noexcept {
  switch (profile) {
    case Profile::kStandard:
      return "standard";
    case Profile::kRobotics:
      return "robotics";
    case Profile::kRealTime:
      return "real_time";
    case Profile::kRegulated:
      return "regulated";
    case Profile::kResearch:
      return "research";
  }
  return "invalid";
}

}  // namespace

std::string_view lifecycle_plan_diagnostic_token(LifecycleKernelErrorCode code) noexcept {
  switch (code) {
    case LifecycleKernelErrorCode::kMissingProducer:
      return "LIFECYCLE_DEPENDENCY_MISSING";
    case LifecycleKernelErrorCode::kDuplicateProducer:
      return "LIFECYCLE_DEPENDENCY_DUPLICATE";
    case LifecycleKernelErrorCode::kCrossScopeDependency:
      return "LIFECYCLE_DEPENDENCY_SCOPE_INVALID";
    case LifecycleKernelErrorCode::kCrossStageDependency:
      return "LIFECYCLE_DEPENDENCY_STAGE_INVALID";
    case LifecycleKernelErrorCode::kCycle:
      return "LIFECYCLE_DEPENDENCY_CYCLE";
    case LifecycleKernelErrorCode::kDuplicateCallback:
      return "LIFECYCLE_CALLBACK_DUPLICATE";
    case LifecycleKernelErrorCode::kAmbiguousNodeIdentity:
      return "LIFECYCLE_NODE_IDENTITY_AMBIGUOUS";
    case LifecycleKernelErrorCode::kInvalidStage:
    case LifecycleKernelErrorCode::kCapacityExceeded:
    case LifecycleKernelErrorCode::kInvalidScope:
    case LifecycleKernelErrorCode::kInvalidNode:
    case LifecycleKernelErrorCode::kDuplicateNode:
    case LifecycleKernelErrorCode::kInvalidCallback:
    case LifecycleKernelErrorCode::kInvalidDependencyKey:
      return "LIFECYCLE_STAGE_GRAPH_INVALID";
  }
  return "LIFECYCLE_STAGE_GRAPH_INVALID";
}

namespace {

void append_lifecycle_graph_diagnostic(LifecyclePlanLowering& result,
                                       const LifecycleKernelError& error) {
  append_diagnostic(result,
                    lifecycle_plan_diagnostic_token(error.code),
                    {},
                    "/lifecycle/stage_graph",
                    error.stable_name,
                    std::to_string(static_cast<std::uint8_t>(error.code)));
}

[[nodiscard]] std::string effective_scope_id(const GraphSnapshotData& snapshot,
                                             VertexId vertex_id) {
  const auto vertex = std::find_if(
      snapshot.vertices.begin(), snapshot.vertices.end(), [vertex_id](const auto& record) {
        return record.id == vertex_id;
      });
  if (vertex == snapshot.vertices.end()) {
    return {};
  }
  return "graph:" + std::to_string(vertex->graph.value) +
         "/operator:" + std::to_string(vertex->id.value);
}

[[nodiscard]] LifecycleTransactionCapacityValues configured_capacity(
    const LifecycleTransactionOptions& options) noexcept {
  return {
      .max_pending_requests = options.max_pending_requests,
      .max_pending_transactions = options.max_pending_transactions,
      .max_receipts = options.max_receipts,
      .max_fenced_operators = options.max_fenced_operators,
      .max_candidate_generations = options.max_candidate_generations,
      .max_events = options.max_events,
      .max_audit_records = options.max_audit_records,
      .max_quiesce_records = options.max_quiesce_records,
      .max_rollback_records = options.max_rollback_records,
  };
}

[[nodiscard]] bool all_capacity_fields_positive(
    const LifecycleTransactionCapacityValues& values) noexcept {
  return values.max_pending_requests > 0U && values.max_pending_transactions > 0U &&
         values.max_receipts > 0U && values.max_fenced_operators > 0U &&
         values.max_candidate_generations > 0U && values.max_events > 0U &&
         values.max_audit_records > 0U && values.max_quiesce_records > 0U &&
         values.max_rollback_records > 0U;
}

// Closure publication validates and emits one complete restart-authority record family.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void append_restart_closures(
    const std::unordered_map<std::uint64_t, LifecycleNodeId>& nodes,
    std::span<const LifecycleDependencyPlanRecord> dependencies,
    std::span<const LifecycleRestartClosureDependency> closure_dependencies,
    std::span<const LifecycleCheckpointPlanRecord> checkpoints, PlanGenerationId plan_generation,
    LifecyclePlanLowering& result) {
  result.restart_closures.reserve(nodes.size());
  std::vector<LifecycleNodeId> ordered_nodes;
  ordered_nodes.reserve(nodes.size());
  for (const auto& [vertex, node] : nodes) {
    static_cast<void>(vertex);
    ordered_nodes.push_back(node);
  }
  std::sort(ordered_nodes.begin(), ordered_nodes.end());

  std::vector<std::pair<LifecycleNodeId, LifecycleNodeId>> closure_edges;
  closure_edges.reserve(closure_dependencies.size());
  for (const auto& dependency : closure_dependencies) {
    const auto source = nodes.find(dependency.source_owner.value);
    const auto dependent = nodes.find(dependency.dependent_owner.value);
    if (source != nodes.end() && dependent != nodes.end() && source->second != dependent->second) {
      closure_edges.emplace_back(source->second, dependent->second);
    }
  }
  std::sort(closure_edges.begin(), closure_edges.end());
  closure_edges.erase(std::unique(closure_edges.begin(), closure_edges.end()), closure_edges.end());

  for (const LifecycleNodeId target : ordered_nodes) {
    std::vector<LifecycleNodeId> closure{target};
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto& dependency : dependencies) {
        if (std::find(closure.begin(), closure.end(), dependency.before.id) != closure.end() &&
            std::find(closure.begin(), closure.end(), dependency.after.id) == closure.end()) {
          closure.push_back(dependency.after.id);
          changed = true;
        }
      }
      for (const auto& [source, dependent] : closure_edges) {
        if (std::find(closure.begin(), closure.end(), source) != closure.end() &&
            std::find(closure.begin(), closure.end(), dependent) == closure.end()) {
          closure.push_back(dependent);
          changed = true;
        }
      }
    }
    std::sort(closure.begin(), closure.end());

    LifecycleRestartClosurePlanRecord record{
        .target = operator_ref(target, plan_generation),
        .valid_requested_stage_mask = kAllForwardStagesMask,
        .required_fenced_operators = static_cast<std::uint32_t>(closure.size()),
    };
    record.closure.reserve(closure.size());
    for (const LifecycleNodeId member : closure) {
      record.closure.push_back(operator_ref(member, plan_generation));
    }
    for (std::uint8_t requested = 0U;
         requested <= static_cast<std::uint8_t>(LifecycleStage::kStart);
         ++requested) {
      LifecycleStage effective = static_cast<LifecycleStage>(requested);
      for (const auto& dependency : dependencies) {
        if (!is_lifecycle_forward_stage(dependency.stage) ||
            std::find(closure.begin(), closure.end(), dependency.before.id) == closure.end() ||
            std::find(closure.begin(), closure.end(), dependency.after.id) == closure.end()) {
          continue;
        }
        if (static_cast<std::uint8_t>(dependency.stage) < static_cast<std::uint8_t>(effective)) {
          effective = dependency.stage;
        }
      }
      record.stage_resolutions[requested] = {
          .requested_stage = static_cast<LifecycleStage>(requested),
          .effective_stage = effective,
          .reason = effective == static_cast<LifecycleStage>(requested)
                        ? LifecycleRestartWideningReason::kNone
                        : LifecycleRestartWideningReason::kDependencyStage,
      };
      auto& resolution = record.stage_resolutions[requested];
      resolution.required_structural_event_records = kLifecycleStructuralEventsPerRequest;
      for (const LifecycleNodeId member : closure) {
        bool member_requires_quiesce_record = false;
        for (const auto& checkpoint : checkpoints) {
          if (checkpoint.operator_instance.id != member || !checkpoint.callback_present) {
            continue;
          }
          const bool forward_work =
              is_lifecycle_forward_stage(checkpoint.stage) &&
              static_cast<std::uint8_t>(checkpoint.stage) >= static_cast<std::uint8_t>(effective);
          const bool teardown_work = checkpoint.stage == LifecycleStage::kStop ||
                                     checkpoint.stage == LifecycleStage::kRelease;
          if (forward_work) {
            ++resolution.required_rollback_records;
          }
          member_requires_quiesce_record =
              member_requires_quiesce_record || forward_work || teardown_work;
        }
        if (member_requires_quiesce_record) {
          ++resolution.required_quiesce_records;
        }
      }
      record.required_quiesce_records =
          std::max(record.required_quiesce_records, resolution.required_quiesce_records);
      record.required_rollback_records =
          std::max(record.required_rollback_records, resolution.required_rollback_records);
    }
    result.restart_closures.push_back(std::move(record));
  }
}

void reject_admitted_restart_targets(LifecyclePlanLowering& result) noexcept {
  for (auto& target : result.restart_targets) {
    if (target.disposition == LifecycleRestartTargetDisposition::kAdmitted) {
      target.disposition = LifecycleRestartTargetDisposition::kRejected;
      target.disposition_reason =
          reason_id(StructuralDiagnosticId::kLifecycleRestartTargetUnsupported);
    }
  }
}

[[nodiscard]] bool capacity_satisfies(const LifecycleTransactionCapacityValues& configured,
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

struct LifecycleCapacityField {
  std::string_view name;
  std::uint32_t LifecycleTransactionCapacityValues::* value;
};

inline constexpr std::array kLifecycleCapacityFields{
    LifecycleCapacityField{"max_pending_requests",
                           &LifecycleTransactionCapacityValues::max_pending_requests},
    LifecycleCapacityField{"max_pending_transactions",
                           &LifecycleTransactionCapacityValues::max_pending_transactions},
    LifecycleCapacityField{"max_receipts", &LifecycleTransactionCapacityValues::max_receipts},
    LifecycleCapacityField{"max_fenced_operators",
                           &LifecycleTransactionCapacityValues::max_fenced_operators},
    LifecycleCapacityField{"max_candidate_generations",
                           &LifecycleTransactionCapacityValues::max_candidate_generations},
    LifecycleCapacityField{"max_events", &LifecycleTransactionCapacityValues::max_events},
    LifecycleCapacityField{"max_audit_records",
                           &LifecycleTransactionCapacityValues::max_audit_records},
    LifecycleCapacityField{"max_quiesce_records",
                           &LifecycleTransactionCapacityValues::max_quiesce_records},
    LifecycleCapacityField{"max_rollback_records",
                           &LifecycleTransactionCapacityValues::max_rollback_records},
};

void append_capacity_diagnostics(LifecyclePlanLowering& result,
                                 const LifecycleTransactionCapacityValues& configured,
                                 const LifecycleTransactionCapacityValues& required) {
  for (const auto& field : kLifecycleCapacityFields) {
    const std::uint32_t provided = configured.*field.value;
    const std::uint32_t minimum = required.*field.value;
    const std::string pointer = "/lifecycle_transactions/" + std::string(field.name);
    if (provided == 0U) {
      append_diagnostic(result,
                        "LIFECYCLE_TRANSACTION_CAPACITY_INVALID",
                        {},
                        pointer,
                        "positive",
                        std::to_string(provided));
    } else if (provided < minimum) {
      append_diagnostic(result,
                        "LIFECYCLE_TRANSACTION_CAPACITY_INSUFFICIENT",
                        {},
                        pointer,
                        std::to_string(minimum),
                        std::to_string(provided));
    }
  }
}

struct RestartAuthority {
  bool requested{};
  bool self{};
  bool external{};
  bool cross_target{};
  bool parameter{};
};

using RestartAuthorityMap = std::unordered_map<LifecycleNodeId, RestartAuthority>;

void append_restart_target_diagnostic(LifecyclePlanLowering& result,
                                      const LifecycleRestartPermissionAuthoringRecord& permission,
                                      std::string_view reason) {
  append_diagnostic(
      result,
      "LIFECYCLE_RESTART_TARGET_UNSUPPORTED",
      DiagnosticSource{.vertex = permission.target},
      "/lifecycle/restart_permissions",
      "valid_origin_requester_target_authority",
      std::string(restart_origin_name(permission.origin)) + ":" + std::string(reason));
}

[[nodiscard]] const LifecycleLogicalCallbackRecord* find_callback(const LifecycleStageGraph& graph,
                                                                  LifecycleNodeId node_id,
                                                                  LifecycleStage stage) noexcept {
  const auto callback = std::find_if(
      graph.callbacks.begin(), graph.callbacks.end(), [node_id, stage](const auto& record) {
        return record.node_id == node_id && record.stage == stage;
      });
  return callback == graph.callbacks.end() ? nullptr : &*callback;
}

void append_checkpoints(const LifecycleStageGraph& graph, PlanGenerationId plan_generation,
                        LifecyclePlanLowering& result) {
  result.checkpoints.reserve(graph.nodes.size() * graph.stages.size());
  for (const auto& stage_plan : graph.stages) {
    for (std::size_t level = 0U; level < stage_plan.levels.size(); ++level) {
      for (const LifecycleNodeId node_id : stage_plan.levels[level].node_ids) {
        const auto* callback = find_callback(graph, node_id, stage_plan.stage);
        result.checkpoints.push_back({
            .operator_instance = operator_ref(node_id, plan_generation),
            .stage = stage_plan.stage,
            .callback_present = callback != nullptr,
            .required = callback != nullptr && callback->required,
            .stable_topological_level = static_cast<std::uint32_t>(level),
        });
      }
    }
  }
}

void append_dependencies(const LifecycleStageGraphInput& input, PlanGenerationId plan_generation,
                         LifecyclePlanLowering& result) {
  result.dependencies.reserve(input.direct_edges.size() + input.requirements.size());
  for (const auto& edge : input.direct_edges) {
    result.dependencies.push_back({
        .before = operator_ref(edge.before_node_id, plan_generation),
        .after = operator_ref(edge.after_node_id, plan_generation),
        .stage = edge.before_stage,
        .kind = LifecycleEdgeKind::kBefore,
        .dependency_key = {},
    });
  }
  for (const auto& requirement : input.requirements) {
    const auto producer =
        std::find_if(input.producers.begin(), input.producers.end(), [&](const auto& candidate) {
          return candidate.key.scope_id == requirement.key.scope_id &&
                 candidate.key.stage == requirement.key.stage &&
                 candidate.key.stable_name == requirement.key.stable_name;
        });
    if (producer == input.producers.end()) {
      continue;  // The lifecycle normalizer has already rejected this case.
    }
    result.dependencies.push_back({
        .before = operator_ref(producer->node_id, plan_generation),
        .after = operator_ref(requirement.node_id, plan_generation),
        .stage = requirement.declared_stage,
        .kind = LifecycleEdgeKind::kDependencyKey,
        .dependency_key = requirement.key.stable_name,
    });
  }
  std::sort(
      result.dependencies.begin(), result.dependencies.end(), [](const auto& lhs, const auto& rhs) {
        return std::tuple{lhs.stage, lhs.before.id, lhs.after.id, lhs.kind, lhs.dependency_key} <
               std::tuple{rhs.stage, rhs.before.id, rhs.after.id, rhs.kind, rhs.dependency_key};
      });
}

[[nodiscard]] RestartAuthorityMap make_restart_authority(
    const std::unordered_map<std::uint64_t, LifecycleNodeId>& nodes) {
  RestartAuthorityMap authority;
  authority.reserve(nodes.size());
  for (const auto& [vertex, node] : nodes) {
    static_cast<void>(vertex);
    authority.emplace(node, RestartAuthority{});
  }
  return authority;
}

[[nodiscard]] bool collect_restart_permissions(
    const GraphSnapshotData& snapshot,
    const std::unordered_map<std::uint64_t, LifecycleNodeId>& nodes,
    PlanGenerationId plan_generation, RestartAuthorityMap& authority,
    std::vector<LifecycleRestartPermissionPlanRecord>& cross_permissions,
    LifecyclePlanLowering& result) {
  bool permissions_valid = true;
  for (const auto& permission : snapshot.lifecycle_restart_permissions) {
    const auto target = nodes.find(permission.target.value);
    if (target == nodes.end()) {
      append_restart_target_diagnostic(result, permission, "target_unresolved");
      permissions_valid = false;
      continue;
    }
    auto& target_authority = authority.at(target->second);
    // Once a target resolves, retain the fact that authoring requested it even
    // if origin or requester validation later rejects the permission. This is
    // what distinguishes an inspectable rejected target from kNotRequested.
    target_authority.requested = true;
    const bool requester_shape_valid =
        (permission.origin == LifecycleRestartOrigin::kOperatorRequest) ==
        permission.requester.has_value();
    if (!restart_origin_valid(permission.origin) || !requester_shape_valid) {
      append_restart_target_diagnostic(result, permission, "origin_shape_invalid");
      permissions_valid = false;
      continue;
    }
    switch (permission.origin) {
      case LifecycleRestartOrigin::kSelf:
        target_authority.self = true;
        break;
      case LifecycleRestartOrigin::kExternal:
        target_authority.external = true;
        break;
      case LifecycleRestartOrigin::kOperatorRequest: {
        if (!permission.requester) {
          append_restart_target_diagnostic(result, permission, "requester_missing");
          permissions_valid = false;
          break;
        }
        const auto requester = nodes.find(permission.requester->value);
        if (requester == nodes.end() || requester->second == target->second) {
          append_restart_target_diagnostic(result, permission, "requester_invalid");
          permissions_valid = false;
          break;
        }
        target_authority.cross_target = true;
        cross_permissions.push_back({
            .requester = operator_ref(requester->second, plan_generation),
            .target = operator_ref(target->second, plan_generation),
        });
        break;
      }
    }
  }
  return permissions_valid;
}

void append_restart_targets(const GraphSnapshotData& snapshot,
                            const std::unordered_map<std::uint64_t, LifecycleNodeId>& nodes,
                            const RestartAuthorityMap& authority,
                            std::span<const std::optional<Profile>> effective_vertex_profiles,
                            bool enabled, bool admission_valid, PlanGenerationId plan_generation,
                            LifecyclePlanLowering& result) {
  result.restart_targets.reserve(nodes.size());
  for (const auto& vertex : snapshot.vertices) {
    if (vertex.kind != VertexKind::kOperator) {
      continue;
    }
    const LifecycleNodeId node_id = nodes.at(vertex.id.value);
    const RestartAuthority& requested = authority.at(node_id);
    const bool has_requested_origin = requested.requested;
    const std::optional<Profile> effective_profile =
        effective_profile_for_vertex(effective_vertex_profiles, vertex.id);
    const bool profile_supported =
        effective_profile.has_value() && profile_supports_active_transactions(*effective_profile);
    const bool origin_admitted = enabled && profile_supported && admission_valid;

    LifecycleRestartTargetPlanRecord target{
        .target = operator_ref(node_id, plan_generation),
        .effective_scope_id = effective_scope_id(snapshot, vertex.id),
        .effective_profile = effective_profile,
    };
    if (!enabled) {
      target.disposition = LifecycleRestartTargetDisposition::kDisabled;
      target.disposition_reason = reason_id(StructuralDiagnosticId::kLifecycleTransactionsDisabled);
    } else if (!has_requested_origin) {
      target.disposition = LifecycleRestartTargetDisposition::kNotRequested;
    } else if (!origin_admitted) {
      target.disposition = LifecycleRestartTargetDisposition::kRejected;
      target.disposition_reason = reason_id(
          profile_supported ? StructuralDiagnosticId::kLifecycleRestartTargetUnsupported
                            : StructuralDiagnosticId::kLifecycleTransactionUnsupportedProfile);
    } else {
      target.self_allowed = requested.self;
      target.external_control_allowed = requested.external;
      target.parameter_origin_allowed = requested.parameter;
      target.allowed_from_stage_mask = kAllForwardStagesMask;
      target.disposition = LifecycleRestartTargetDisposition::kAdmitted;
    }
    result.restart_targets.push_back(std::move(target));
  }
}

void append_restart_records(const GraphSnapshotData& snapshot,
                            const std::unordered_map<std::uint64_t, LifecycleNodeId>& nodes,
                            std::span<const std::optional<Profile>> effective_vertex_profiles,
                            std::span<const LifecycleMutableFieldPlanRecord> mutable_fields,
                            const std::optional<LifecycleTransactionOptions>& options,
                            PlanGenerationId plan_generation, bool admission_valid,
                            LifecyclePlanLowering& result) {
  auto authority = make_restart_authority(nodes);
  for (const auto& field : mutable_fields) {
    if (field.owner.id == 0U) {
      append_diagnostic(result,
                        "LIFECYCLE_RESTART_TARGET_UNSUPPORTED",
                        {},
                        "/lifecycle_mutable_fields/owner",
                        "nonzero_plan_bound_operator",
                        "0");
      admission_valid = false;
      continue;
    }
    const LifecycleNodeId node = static_cast<LifecycleNodeId>(field.owner.id);
    const auto found = authority.find(node);
    if (found == authority.end()) {
      append_diagnostic(result,
                        "LIFECYCLE_RESTART_TARGET_UNSUPPORTED",
                        DiagnosticSource{.vertex = VertexId{field.owner.id}},
                        "/lifecycle_mutable_fields/owner",
                        "operator_in_current_plan",
                        std::to_string(field.owner.id));
      admission_valid = false;
      continue;
    }
    found->second.requested = true;
    found->second.parameter = true;
  }
  std::vector<LifecycleRestartPermissionPlanRecord> cross_permissions;
  cross_permissions.reserve(snapshot.lifecycle_restart_permissions.size());
  const bool permissions_valid = collect_restart_permissions(
      snapshot, nodes, plan_generation, authority, cross_permissions, result);
  const bool enabled = options.has_value();
  admission_valid = admission_valid && permissions_valid;
  if (enabled) {
    std::sort(
        cross_permissions.begin(), cross_permissions.end(), [](const auto& lhs, const auto& rhs) {
          return std::tuple{lhs.requester.id, lhs.target.id} <
                 std::tuple{rhs.requester.id, rhs.target.id};
        });
    result.restart_permissions = std::move(cross_permissions);
  }
  append_restart_targets(snapshot,
                         nodes,
                         authority,
                         effective_vertex_profiles,
                         enabled,
                         admission_valid,
                         plan_generation,
                         result);
}

}  // namespace

// Keep lifecycle admission, capacity, and diagnostic publication in one transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
LifecyclePlanLowering lower_lifecycle_plan(
    const GraphSnapshotData& snapshot,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    std::span<const LifecycleMutableFieldPlanRecord> lifecycle_mutable_fields,
    const std::optional<ParameterRuntimeCapacityPlanRecord>& parameter_capacity,
    const std::optional<LifecycleTransactionOptions>& transaction_options,
    PlanGenerationId plan_generation,
    std::span<const LifecycleRestartClosureDependency> closure_dependencies) {
  LifecyclePlanLowering result;
  result.mutable_fields.reserve(lifecycle_mutable_fields.size());
  for (const auto& field : lifecycle_mutable_fields) {
    if (!is_lifecycle_forward_stage(field.from_stage)) {
      append_diagnostic(result,
                        "LIFECYCLE_RESTART_STAGE_MASK_INVALID",
                        DiagnosticSource{.vertex = VertexId{field.owner.id}},
                        "/lifecycle_mutable_fields/from_stage",
                        "0x3f",
                        std::to_string(static_cast<std::uint8_t>(field.from_stage)));
      continue;
    }
    result.mutable_fields.push_back(field);
  }
  if (plan_generation == 0U) {
    append_diagnostic(
        result, "LIFECYCLE_RESTART_TARGET_UNSUPPORTED", {}, "/plan_generation", "nonzero", "0");
  }

  std::unordered_map<std::uint64_t, LifecycleNodeId> nodes;
  nodes.reserve(snapshot.vertices.size());
  for (const auto& vertex : snapshot.vertices) {
    if (vertex.kind == VertexKind::kOperator) {
      if (nodes.size() >= std::numeric_limits<LifecycleNodeId>::max()) {
        append_diagnostic(result,
                          "LIFECYCLE_RESTART_TARGET_UNSUPPORTED",
                          DiagnosticSource{.vertex = vertex.id},
                          "/lifecycle/operators",
                          "dense_id_capacity",
                          "exhausted");
        break;
      }
      nodes.emplace(vertex.id.value, static_cast<LifecycleNodeId>(nodes.size() + 1U));
    }
  }

  // An empty Graph has no lifecycle work. The private runtime normalizer
  // deliberately requires at least one node, so compilation handles this
  // valid identity case without invoking it.
  if (!nodes.empty()) {
    auto lowered = lower_lifecycle_stage_graph_input(snapshot);
    if (!lowered) {
      append_lifecycle_graph_diagnostic(result, lowered.error());
    } else {
      auto graph = build_lifecycle_stage_graph(lowered->input, lowered->minimum_capacity);
      if (!graph) {
        append_lifecycle_graph_diagnostic(result, graph.error());
      } else {
        // REQ: COMPILE-008
        append_checkpoints(*graph, plan_generation, result);
        append_dependencies(lowered->input, plan_generation, result);
      }
    }
  }

  append_restart_closures(nodes,
                          result.dependencies,
                          closure_dependencies,
                          result.checkpoints,
                          plan_generation,
                          result);

  // Resolve target authority before deriving capacity. Rejected and
  // nonrequested targets cannot contribute runtime work, while admitted
  // dependency members remain part of their target's closure.
  append_restart_records(snapshot,
                         nodes,
                         effective_vertex_profiles,
                         result.mutable_fields,
                         transaction_options,
                         plan_generation,
                         result.accepted,
                         result);
  std::vector<std::uint8_t> restart_capacity_membership(result.restart_closures.size());
  const LifecycleRestartCapacityMaxima restart_capacity =
      admitted_lifecycle_restart_capacity(result.restart_targets,
                                          result.restart_closures,
                                          plan_generation,
                                          restart_capacity_membership);

  bool capacity_valid = true;
  if (transaction_options) {
    const auto configured = configured_capacity(*transaction_options);
    const auto required_result = derive_required_lifecycle_capacity(
        configured,
        restart_capacity,
        parameter_capacity ? parameter_capacity->required_lifecycle_capacity
                           : LifecycleTransactionCapacityValues{});
    const auto& required = required_result.values;
    const bool configured_valid = all_capacity_fields_positive(configured);
    const bool configured_sufficient = !required_result.arithmetic_overflow &&
                                       !required_result.invalid_input &&
                                       capacity_satisfies(configured, required);
    result.transaction_capacity = LifecycleTransactionCapacityPlanRecord{
        .configured = configured,
        .required_minimum = required,
        .resolved_parameter_update_policy = parameter_capacity
                                                ? parameter_capacity->resolved_update_policy
                                                : ParamRuntimeUpdatePolicy::kDisabled,
        .parameter_origin_enabled = parameter_capacity &&
                                    parameter_capacity->lifecycle_updates_enabled &&
                                    configured_valid && configured_sufficient,
        .required_generation_slots_per_operator =
            parameter_capacity ? parameter_capacity->generation_slots_per_operator : 0U,
        .required_parameter_arena_bytes =
            parameter_capacity ? parameter_capacity->parameter_arena_bytes : 0U,
        .required_parameter_audit_records =
            parameter_capacity ? parameter_capacity->audit_capacity : 0U,
        .derivation_reason = kCanonicalGraphLifecycleCapacityDerivationReason,
    };
    if (required_result.arithmetic_overflow) {
      append_diagnostic(result,
                        "LIFECYCLE_TRANSACTION_CAPACITY_OVERFLOW",
                        {},
                        "/lifecycle_transactions",
                        "representable uint32 derived minima",
                        "overflow");
      capacity_valid = false;
    } else if (required_result.invalid_input) {
      append_diagnostic(result,
                        "LIFECYCLE_STAGE_GRAPH_INVALID",
                        {},
                        "/lifecycle_transactions/capacity_derivation",
                        "canonical dense restart target and closure records",
                        "invalid");
      capacity_valid = false;
    } else if (!configured_valid || !configured_sufficient) {
      append_capacity_diagnostics(result, configured, required);
      capacity_valid = false;
    }
    if (!capacity_valid) {
      reject_admitted_restart_targets(result);
    }
    if (!all_operator_profiles_support_active_transactions(snapshot, effective_vertex_profiles)) {
      for (const auto& vertex : snapshot.vertices) {
        if (vertex.kind != VertexKind::kOperator) {
          continue;
        }
        const auto profile = effective_profile_for_vertex(effective_vertex_profiles, vertex.id);
        if (!profile.has_value() || !profile_supports_active_transactions(*profile)) {
          append_diagnostic(result,
                            "LIFECYCLE_TRANSACTION_UNSUPPORTED_PROFILE",
                            DiagnosticSource{.vertex = vertex.id},
                            "/lifecycle_transactions/profile",
                            "active_restart_admitted",
                            profile ? std::string(profile_name(*profile)) : "unresolved");
        }
      }
    }
  }
  return result;
}

}  // namespace holoscan::detail
