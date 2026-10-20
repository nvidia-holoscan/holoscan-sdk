/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_plan.hpp>

#include "core/graph_snapshot.hpp"
#include "plan_lowering_diagnostic.hpp"

namespace holoscan::detail {

/// Classify one read-only platform placement probe.
enum class PlacementPlatformProbeDisposition : std::uint8_t {
  kPassed = 0,
  kUnavailable = 1,
  kFailed = 2,
};

/// Hold bounded evidence returned by an injected read-only platform probe.
struct PlacementPlatformProbeResult {
  PlacementPlatformProbeDisposition disposition{PlacementPlatformProbeDisposition::kFailed};
  bool irq_isolation_proven{false};
  std::string evidence_id;
  CPUAffinity effective_affinity;
  std::optional<Error> error;
};

/// Read platform placement capability without applying process or thread state.
class PlacementPlatformProbe {
 public:
  virtual ~PlacementPlatformProbe() = default;

  /// Inspect one normalized placement request without mutating OS state.
  [[nodiscard]] virtual PlacementPlatformProbeResult inspect(
      const ThreadPlacement& placement) const = 0;
};

/// Hold normalized thread-placement evidence and bounded diagnostics.
struct DeploymentPlanLowering {
  std::vector<PlacementProbePlanRecord> placement_probes;
  std::vector<PartitionPlacementPlanRecord> placements;
  std::vector<SupportThreadAffinityPlanRecord> support_thread_affinities;
  std::vector<ExecutionLanePlanRecord> lanes;
  std::vector<std::uint32_t> lane_by_vertex;
  std::vector<PlanLoweringDiagnostic> errors;
  std::vector<std::string> warning_tokens;
};

/// Normalize deployment bindings without applying any process/thread policy.
///
/// Exact manual-partition names are resolved to immutable accepted partitions.
/// Runtime materialization consumes these records without traversing Graph
/// metadata or re-resolving partition strings.
[[nodiscard]] DeploymentPlanLowering lower_deployment_plan(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    std::span<ExecutionPartitionRecord> partitions);

/// Normalize deployment bindings using resolved profile scopes and the system probe.
///
/// Each entry in `effective_profiles` is indexed by `VertexId::value - 1`.
/// Missing or invalid entries reject referenced placements rather than silently
/// assuming the graph-level default profile.
[[nodiscard]] DeploymentPlanLowering lower_deployment_plan(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    std::span<const std::optional<Profile>> effective_profiles,
    std::span<ExecutionPartitionRecord> partitions);

/// Normalize deployment bindings using an injected read-only platform probe.
///
/// This overload is the deterministic compiler/test seam. Probe observations
/// become immutable plan evidence; the lowerer never invokes placement apply.
[[nodiscard]] DeploymentPlanLowering lower_deployment_plan(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    std::span<const std::optional<Profile>> effective_profiles,
    std::span<ExecutionPartitionRecord> partitions, const PlacementPlatformProbe& platform_probe);

}  // namespace holoscan::detail
