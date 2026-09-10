/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "deployment_plan_lowering.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>

#include "../core/diagnostic_registry.hpp"
#include "../threading/linux_thread_controller.hpp"
#include "detail/canonical_utils.hpp"
#include "execution_plan_contracts.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] bool placement_less(const ThreadPlacement& lhs, const ThreadPlacement& rhs) {
  const std::vector<std::uint32_t> empty;
  const auto& lhs_cores = lhs.affinity ? lhs.affinity->cores : empty;
  const auto& rhs_cores = rhs.affinity ? rhs.affinity->cores : empty;
  return std::tuple{lhs.affinity.has_value(),
                    lhs_cores,
                    lhs.scheduling.policy,
                    lhs.scheduling.priority,
                    lhs.scheduling.deadline.has_value(),
                    lhs.scheduling.deadline ? lhs.scheduling.deadline->runtime_ns : 0U,
                    lhs.scheduling.deadline ? lhs.scheduling.deadline->deadline_ns : 0U,
                    lhs.scheduling.deadline ? lhs.scheduling.deadline->period_ns : 0U,
                    lhs.wake_mode,
                    lhs.irq_isolation,
                    lhs.probe_failure_policy} <
         std::tuple{rhs.affinity.has_value(),
                    rhs_cores,
                    rhs.scheduling.policy,
                    rhs.scheduling.priority,
                    rhs.scheduling.deadline.has_value(),
                    rhs.scheduling.deadline ? rhs.scheduling.deadline->runtime_ns : 0U,
                    rhs.scheduling.deadline ? rhs.scheduling.deadline->deadline_ns : 0U,
                    rhs.scheduling.deadline ? rhs.scheduling.deadline->period_ns : 0U,
                    rhs.wake_mode,
                    rhs.irq_isolation,
                    rhs.probe_failure_policy};
}

void append_probe(std::vector<PlacementProbePlanRecord>& probes, ExecutionPartitionId partition,
                  std::string probe_id, PlacementProbeStatus status, std::string evidence_id = {},
                  DiagnosticId reason = {}) {
  probes.push_back(PlacementProbePlanRecord{
      .partition = partition,
      .probe_id = std::move(probe_id),
      .status = status,
      .evidence_id = std::move(evidence_id),
      .reason = reason,
  });
}

[[nodiscard]] std::optional<Profile> effective_partition_profile(
    std::span<const VertexId> operators,
    std::span<const std::optional<Profile>> effective_profiles) noexcept {
  std::optional<Profile> result;
  std::uint8_t result_rank = 0U;
  for (const VertexId owner : operators) {
    const auto profile = effective_profile_for_vertex(effective_profiles, owner);
    const auto rank = profile ? profile_semantic_rank(*profile) : std::nullopt;
    if (!profile || !rank) {
      return std::nullopt;
    }
    if (!result || *rank > result_rank) {
      result = profile;
      result_rank = *rank;
    }
  }
  return result;
}

[[nodiscard]] bool profile_permits_probe_fallback(Profile profile) noexcept {
  return profile == Profile::kStandard || profile == Profile::kResearch;
}

[[nodiscard]] PlacementProbeStatus plan_probe_status(
    PlacementPlatformProbeDisposition disposition) noexcept {
  return disposition == PlacementPlatformProbeDisposition::kPassed
             ? PlacementProbeStatus::kPassed
             : PlacementProbeStatus::kRejected;
}

class SystemPlacementPlatformProbe final : public PlacementPlatformProbe {
 public:
  [[nodiscard]] PlacementPlatformProbeResult inspect(
      const ThreadPlacement& placement) const override {
    auto observation = LinuxThreadController::probe(placement);
    if (!observation) {
      return PlacementPlatformProbeResult{
          .disposition = PlacementPlatformProbeDisposition::kFailed,
          .evidence_id = "holoscan.compile.thread_placement.probe_failed/v1",
      };
    }
    const bool available =
        observation->unavailable_cores.empty() && observation->can_apply_realtime;
    return PlacementPlatformProbeResult{
        .disposition = available ? PlacementPlatformProbeDisposition::kPassed
                                 : PlacementPlatformProbeDisposition::kUnavailable,
        // LinuxThreadController proves effective affinity and scheduling
        // capability only. IRQ isolation needs separate deployment evidence.
        .irq_isolation_proven = false,
        .evidence_id = available ? "holoscan.platform.linux.thread_placement/v1"
                                 : "holoscan.platform.linux.thread_placement.unavailable/v1",
    };
  }
};

[[nodiscard]] const PlacementPlatformProbe& system_placement_probe() {
  static const SystemPlacementPlatformProbe probe;
  return probe;
}

void reject_placement(PartitionPlacementPlanRecord& placement, DiagnosticId reason,
                      DeploymentPlanLowering& result, std::string_view token) {
  placement.accepted.reset();
  placement.disposition = PlanDisposition::kRejected;
  placement.reason = reason;
  result.error_tokens.emplace_back(token);
}

}  // namespace

DeploymentPlanLowering lower_deployment_plan(const GraphSnapshotData& snapshot,
                                             const DeploymentBinding& deployment,
                                             std::span<ExecutionPartitionRecord> partitions) {
  const std::vector<std::optional<Profile>> unresolved(snapshot.vertices.size());
  return lower_deployment_plan(
      snapshot, deployment, unresolved, partitions, system_placement_probe());
}

DeploymentPlanLowering lower_deployment_plan(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    std::span<const std::optional<Profile>> effective_profiles,
    std::span<ExecutionPartitionRecord> partitions) {
  return lower_deployment_plan(
      snapshot, deployment, effective_profiles, partitions, system_placement_probe());
}

// Keep admission, diagnostic, and canonical placement publication transactional.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
DeploymentPlanLowering lower_deployment_plan(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    std::span<const std::optional<Profile>> effective_profiles,
    std::span<ExecutionPartitionRecord> partitions, const PlacementPlatformProbe& platform_probe) {
  constexpr std::string_view kUnknownOrDuplicate = "PARTITION_PLACEMENT_UNKNOWN_OR_DUPLICATE";
  constexpr std::string_view kRejected = "THREAD_PLACEMENT_REJECTED";

  DeploymentPlanLowering result;
  result.lane_by_vertex.resize(snapshot.vertices.size(), 0U);
  result.lanes.reserve(partitions.size());
  for (std::size_t partition_index = 0U; partition_index < partitions.size(); ++partition_index) {
    auto& partition = partitions[partition_index];
    partition.execution_lane = static_cast<std::uint32_t>(partition_index);
    if (!partition.id) {
      result.error_tokens.emplace_back(kUnknownOrDuplicate);
      continue;
    }
    for (const VertexId owner : partition.members) {
      if (!owner || owner.value > result.lane_by_vertex.size()) {
        result.error_tokens.emplace_back(kUnknownOrDuplicate);
        continue;
      }
      result.lane_by_vertex[static_cast<std::size_t>(owner.value - 1U)] = partition.execution_lane;
    }
    result.lanes.push_back(ExecutionLanePlanRecord{
        .execution_lane = partition.execution_lane,
        .partition = partition.id,
        .accepted_placement = ThreadPlacement{},
    });
  }

  result.placements.reserve(deployment.thread_placements().size());
  for (const auto& authored : deployment.thread_placements()) {
    const auto matching_partition =
        std::find_if(partitions.begin(), partitions.end(), [&](const auto& candidate) {
          return candidate.authored_name && *candidate.authored_name == authored.partition_name;
        });
    PartitionPlacementPlanRecord record{
        .partition = matching_partition == partitions.end() ? ExecutionPartitionId{}
                                                            : matching_partition->id,
        .authored_name = authored.partition_name,
        .execution_lane =
            matching_partition == partitions.end() ? 0U : matching_partition->execution_lane,
        .requested = authored.placement,
    };
    if (record.requested.affinity) {
      record.requested.affinity->normalize();
    }
    if (!bounded_identity_valid(record.authored_name, 255U) ||
        matching_partition == partitions.end()) {
      record.disposition = PlanDisposition::kRejected;
      record.reason = diagnostic_id(kUnknownOrDuplicate);
    } else if (!record.requested.validate()) {
      record.disposition = PlanDisposition::kRejected;
      record.reason = diagnostic_id(kRejected);
      result.error_tokens.emplace_back(kRejected);
    }
    result.placements.push_back(std::move(record));
  }
  std::sort(
      result.placements.begin(), result.placements.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.authored_name != rhs.authored_name
                   ? lhs.authored_name < rhs.authored_name
                   : placement_less(lhs.requested, rhs.requested);
      });
  for (std::size_t index = 0U; index < result.placements.size();) {
    std::size_t end = index + 1U;
    while (end < result.placements.size() &&
           result.placements[end].authored_name == result.placements[index].authored_name) {
      ++end;
    }
    if (end - index > 1U) {
      for (std::size_t duplicate = index; duplicate < end; ++duplicate) {
        result.placements[duplicate].accepted.reset();
        result.placements[duplicate].disposition = PlanDisposition::kRejected;
        result.placements[duplicate].reason = diagnostic_id(kUnknownOrDuplicate);
      }
      result.error_tokens.emplace_back(kUnknownOrDuplicate);
    } else if (result.placements[index].reason == diagnostic_id(kUnknownOrDuplicate)) {
      result.error_tokens.emplace_back(kUnknownOrDuplicate);
    }
    index = end;
  }

  for (auto& placement : result.placements) {
    if (placement.reason) {
      continue;
    }
    const auto partition =
        std::find_if(partitions.begin(), partitions.end(), [&](const auto& candidate) {
          return candidate.id == placement.partition;
        });
    if (partition == partitions.end()) {
      reject_placement(placement, diagnostic_id(kUnknownOrDuplicate), result, kUnknownOrDuplicate);
      continue;
    }

    append_probe(result.placement_probes,
                 placement.partition,
                 "holoscan.placement.syntax/v1",
                 PlacementProbeStatus::kPassed,
                 "holoscan.compile.thread_placement.validation/v1");
    const auto partition_profile =
        effective_partition_profile(partition->members, effective_profiles);
    if (!partition_profile) {
      append_probe(result.placement_probes,
                   placement.partition,
                   "holoscan.placement.profile_scope/v1",
                   PlacementProbeStatus::kRejected,
                   {},
                   diagnostic_id(kRejected));
      reject_placement(placement, diagnostic_id(kRejected), result, kRejected);
      continue;
    }
    append_probe(result.placement_probes,
                 placement.partition,
                 "holoscan.placement.profile_scope/v1",
                 PlacementProbeStatus::kPassed,
                 "holoscan.compile.profile_scope_resolution/v1");

    PlacementPlatformProbeResult observation = platform_probe.inspect(placement.requested);
    if (observation.evidence_id.empty()) {
      observation.disposition = PlacementPlatformProbeDisposition::kFailed;
      observation.evidence_id = "holoscan.compile.thread_placement.probe_evidence_missing/v1";
    }
    append_probe(result.placement_probes,
                 placement.partition,
                 "holoscan.placement.platform/v1",
                 plan_probe_status(observation.disposition),
                 observation.evidence_id,
                 observation.disposition == PlacementPlatformProbeDisposition::kPassed
                     ? DiagnosticId{}
                     : diagnostic_id(kRejected));

    bool accepted = observation.disposition == PlacementPlatformProbeDisposition::kPassed;
    ThreadPlacement accepted_placement = placement.requested;
    if (!accepted) {
      append_probe(result.placement_probes,
                   placement.partition,
                   "holoscan.placement.irq_isolation/v1",
                   placement.requested.irq_isolation == IRQIsolation::kNotRequested
                       ? PlacementProbeStatus::kNotRequested
                       : PlacementProbeStatus::kRejected,
                   placement.requested.irq_isolation == IRQIsolation::kNotRequested
                       ? std::string{}
                       : observation.evidence_id,
                   placement.requested.irq_isolation == IRQIsolation::kNotRequested
                       ? DiagnosticId{}
                       : diagnostic_id(kRejected));
      reject_placement(placement, diagnostic_id(kRejected), result, kRejected);
    } else if (placement.requested.irq_isolation == IRQIsolation::kNotRequested) {
      append_probe(result.placement_probes,
                   placement.partition,
                   "holoscan.placement.irq_isolation/v1",
                   PlacementProbeStatus::kNotRequested);
    } else if (observation.irq_isolation_proven) {
      append_probe(result.placement_probes,
                   placement.partition,
                   "holoscan.placement.irq_isolation/v1",
                   PlacementProbeStatus::kPassed,
                   observation.evidence_id);
    } else if (placement.requested.irq_isolation == IRQIsolation::kPreferIsolated &&
               placement.requested.probe_failure_policy ==
                   ProbeFailurePolicy::kWarnAndRecordFallback &&
               profile_permits_probe_fallback(*partition_profile)) {
      accepted_placement.irq_isolation = IRQIsolation::kNotRequested;
      append_probe(result.placement_probes,
                   placement.partition,
                   "holoscan.placement.irq_isolation/v1",
                   PlacementProbeStatus::kFallbackRecorded,
                   observation.evidence_id);
    } else {
      append_probe(result.placement_probes,
                   placement.partition,
                   "holoscan.placement.irq_isolation/v1",
                   PlacementProbeStatus::kRejected,
                   {},
                   diagnostic_id(kRejected));
      accepted = false;
      reject_placement(placement, diagnostic_id(kRejected), result, kRejected);
    }

    if (accepted) {
      placement.accepted = accepted_placement;
      placement.disposition = PlanDisposition::kAccepted;
      placement.reason = {};
      append_probe(result.placement_probes,
                   placement.partition,
                   "holoscan.placement.apply_readback/v1",
                   PlacementProbeStatus::kRuntimeRequired,
                   "holoscan.runtime.thread_placement.apply_readback/v1");
      const auto lane =
          std::find_if(result.lanes.begin(), result.lanes.end(), [&](const auto& candidate) {
            return candidate.partition == placement.partition;
          });
      if (lane != result.lanes.end()) {
        lane->accepted_placement = accepted_placement;
      }
    }
  }

  std::sort(result.lanes.begin(), result.lanes.end(), [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.execution_lane, lhs.partition.value) <
           std::tie(rhs.execution_lane, rhs.partition.value);
  });
  canonicalize_tokens(result.error_tokens);
  canonicalize_tokens(result.warning_tokens);
  return result;
}

}  // namespace holoscan::detail
