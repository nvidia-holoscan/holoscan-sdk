/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_plan.hpp>

#include "../core/graph_snapshot.hpp"
#include "../core/lifecycle_kernel.hpp"

namespace holoscan::detail {

/// Bounded diagnostic evidence produced while lowering lifecycle declarations.
struct LifecyclePlanDiagnostic {
  std::string token;
  DiagnosticSource source{};
  std::string json_pointer;
  std::string required;
  std::string provided;
};

/// Pointer-free lifecycle portion of one accepted or rejected execution plan.
struct LifecyclePlanLowering {
  bool accepted{true};
  std::vector<LifecyclePlanDiagnostic> diagnostics;
  std::vector<LifecycleCheckpointPlanRecord> checkpoints;
  std::vector<LifecycleDependencyPlanRecord> dependencies;
  std::vector<LifecycleRestartPermissionPlanRecord> restart_permissions;
  std::vector<LifecycleRestartTargetPlanRecord> restart_targets;
  std::vector<LifecycleRestartClosurePlanRecord> restart_closures;
  std::vector<LifecycleMutableFieldPlanRecord> mutable_fields;
  std::optional<LifecycleTransactionCapacityPlanRecord> transaction_capacity;
};

/// Return the stable compiler diagnostic token for one lifecycle-kernel rejection.
///
/// The Graph snapshot intentionally cannot encode several malformed private
/// kernel shapes (for example, a dependency whose declaration stage differs
/// from its key stage). Keeping this mapping independently testable preserves
/// the compile diagnostic contract for those defensive paths without adding
/// redundant or contradictory fields to the canonical Graph snapshot.
[[nodiscard]] std::string_view lifecycle_plan_diagnostic_token(
    LifecycleKernelErrorCode code) noexcept;

/// Lower canonical Graph lifecycle declarations into immutable plan records.
///
/// The snapshot must be the canonical compiler handoff for one plan. Operator
/// identities are assigned densely in canonical Operator-vertex order and are
/// stamped with `plan_generation`; every resulting reference is plan-bound.
/// `effective_vertex_profiles` must use the same canonical VertexId-indexed
/// layout as the snapshot. Target admission and evidence use each entry rather
/// than substituting the root profile for tighter regions.
// REQ: COMPILE-008
[[nodiscard]] LifecyclePlanLowering lower_lifecycle_plan(
    const GraphSnapshotData& snapshot,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    std::span<const LifecycleMutableFieldPlanRecord> lifecycle_mutable_fields,
    const std::optional<ParameterRuntimeCapacityPlanRecord>& parameter_capacity,
    const std::optional<LifecycleTransactionOptions>& transaction_options,
    PlanGenerationId plan_generation);

}  // namespace holoscan::detail
