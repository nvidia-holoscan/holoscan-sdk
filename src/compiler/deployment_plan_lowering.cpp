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
#include <utility>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>

#include "../core/diagnostic_registry.hpp"
#include "../threading/linux_thread_controller.hpp"
#include "detail/canonical_utils.hpp"
#include "execution_plan_contracts.hpp"
#include "snapshot_lookup.hpp"
#include "support_thread_affinity_contracts.hpp"

namespace holoscan::detail {
namespace {

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
          .error = std::move(observation).error(),
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
        .effective_affinity = observation->effective_affinity,
    };
  }
};

[[nodiscard]] const PlacementPlatformProbe& system_placement_probe() {
  static const SystemPlacementPlatformProbe probe;
  return probe;
}

void append_error(DeploymentPlanLowering& result, std::string_view token,
                  std::optional<DiagnosticDetail> detail = std::nullopt,
                  std::string json_pointer = {}) {
  result.errors.push_back(PlanLoweringDiagnostic{
      .token = std::string(token),
      .json_pointer = std::move(json_pointer),
      .detail = std::move(detail),
  });
}

[[nodiscard]] std::string support_thread_diagnostic_pointer(SupportThreadRole role) {
  std::string pointer{"/deployment/support_thread_affinities/"};
  if (valid_support_thread_role(role)) {
    pointer += support_thread_role_name(role);
  } else {
    pointer += "unknown-" + std::to_string(static_cast<std::uint8_t>(role));
  }
  return pointer;
}

[[nodiscard]] std::string support_thread_request_summary(SupportThreadRole role,
                                                         const ThreadPlacement& requested) {
  return "role=" + std::string(support_thread_role_name(role)) + ",requested=" +
         (requested.affinity ? support_thread_affinity_summary(*requested.affinity) : "<absent>");
}

[[nodiscard]] DiagnosticDetail support_thread_failure_detail(SupportThreadRole role, Error error,
                                                             std::string field,
                                                             std::string provided,
                                                             std::string required) {
  DiagnosticDetail detail{
      .operation = DiagnosticOperation::kCompile,
      .error = std::move(error),
      .operator_path = "support-thread/" + std::string(support_thread_role_name(role)),
  };
  detail.mismatches.push_back(DiagnosticMismatch{
      .field = std::move(field),
      .provided = std::move(provided),
      .required = std::move(required),
  });
  return detail;
}

void reject_placement(PartitionPlacementPlanRecord& placement, DiagnosticId reason,
                      DeploymentPlanLowering& result, std::string_view token) {
  placement.accepted.reset();
  placement.disposition = PlanDisposition::kRejected;
  placement.reason = reason;
  append_error(result, token);
}

void reject_support_thread_affinity(SupportThreadAffinityPlanRecord& placement, DiagnosticId reason,
                                    DeploymentPlanLowering& result, std::string_view token,
                                    DiagnosticDetail detail) {
  placement.accepted.reset();
  placement.disposition = PlanDisposition::kRejected;
  placement.reason = reason;
  append_error(result, token, std::move(detail), support_thread_diagnostic_pointer(placement.role));
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
  constexpr std::string_view kSupportUnknownOrDuplicate =
      "SUPPORT_THREAD_PLACEMENT_UNKNOWN_OR_DUPLICATE";
  constexpr std::string_view kRejected = "THREAD_PLACEMENT_REJECTED";

  DeploymentPlanLowering result;
  result.support_thread_affinities.reserve(deployment.support_thread_affinities().size());
  for (const auto& authored : deployment.support_thread_affinities()) {
    SupportThreadAffinityPlanRecord record{
        .role = authored.role,
        .requested = authored.placement,
    };
    if (record.requested.affinity) {
      record.requested.affinity->normalize();
    }
    if (!valid_support_thread_role(record.role)) {
      record.reason = diagnostic_id(kSupportUnknownOrDuplicate);
    } else if (!affinity_only_support_placement(record.requested) || !record.requested.validate()) {
      record.reason = diagnostic_id(kRejected);
    }
    result.support_thread_affinities.push_back(std::move(record));
  }
  std::sort(result.support_thread_affinities.begin(),
            result.support_thread_affinities.end(),
            [](const auto& lhs, const auto& rhs) {
              return lhs.role != rhs.role
                         ? static_cast<std::uint8_t>(lhs.role) < static_cast<std::uint8_t>(rhs.role)
                         : thread_placement_less(lhs.requested, rhs.requested);
            });
  for (std::size_t index = 0U; index < result.support_thread_affinities.size();) {
    std::size_t end = index + 1U;
    while (end < result.support_thread_affinities.size() &&
           result.support_thread_affinities[end].role ==
               result.support_thread_affinities[index].role) {
      ++end;
    }
    if (end - index > 1U) {
      for (std::size_t duplicate = index; duplicate < end; ++duplicate) {
        auto& record = result.support_thread_affinities[duplicate];
        record.accepted.reset();
        record.disposition = PlanDisposition::kRejected;
        record.reason = diagnostic_id(kSupportUnknownOrDuplicate);
      }
      const auto role = result.support_thread_affinities[index].role;
      append_error(
          result,
          kSupportUnknownOrDuplicate,
          support_thread_failure_detail(
              role,
              Error{ErrorCode::kInvalidArgument, "duplicate Runtime support-thread affinity role"},
              "support_thread_role",
              "role=" + std::string(support_thread_role_name(role)) +
                  ",bindings=" + std::to_string(end - index),
              "at most one affinity binding per support-thread role"),
          support_thread_diagnostic_pointer(role));
    } else if (result.support_thread_affinities[index].reason ==
               diagnostic_id(kSupportUnknownOrDuplicate)) {
      const auto& record = result.support_thread_affinities[index];
      append_error(
          result,
          kSupportUnknownOrDuplicate,
          support_thread_failure_detail(
              record.role,
              Error{ErrorCode::kInvalidArgument, "unknown Runtime support-thread affinity role"},
              "support_thread_role",
              "value=" + std::to_string(static_cast<std::uint8_t>(record.role)),
              "controller or channel_pump"),
          support_thread_diagnostic_pointer(record.role));
    } else if (result.support_thread_affinities[index].reason == diagnostic_id(kRejected)) {
      const auto& record = result.support_thread_affinities[index];
      const auto validated = record.requested.validate();
      append_error(
          result,
          kRejected,
          support_thread_failure_detail(
              record.role,
              validated ? Error{ErrorCode::kInvalidArgument,
                                "Runtime support-thread placement must contain affinity only"}
                        : validated.error(),
              "support_thread_affinity",
              support_thread_request_summary(record.role, record.requested),
              "a nonempty affinity-only placement"),
          support_thread_diagnostic_pointer(record.role));
    }
    index = end;
  }
  for (auto& placement : result.support_thread_affinities) {
    if (placement.reason) {
      continue;
    }
    PlacementPlatformProbeResult observation = platform_probe.inspect(placement.requested);
    if (observation.evidence_id.empty()) {
      observation.disposition = PlacementPlatformProbeDisposition::kFailed;
    }
    if (observation.disposition != PlacementPlatformProbeDisposition::kPassed) {
      std::string provided = support_thread_request_summary(placement.role, placement.requested);
      if (!observation.effective_affinity.cores.empty()) {
        provided += ",effective=" + support_thread_affinity_summary(observation.effective_affinity);
      }
      if (!observation.evidence_id.empty()) {
        provided += ",probe=" + observation.evidence_id;
      }
      reject_support_thread_affinity(
          placement,
          diagnostic_id(kRejected),
          result,
          kRejected,
          support_thread_failure_detail(
              placement.role,
              observation.error.value_or(
                  Error{ErrorCode::kIncompatible,
                        "requested affinity is unavailable to the compiling thread"}),
              "support_thread_affinity",
              std::move(provided),
              "requested CPUs must be in the compiling thread's effective affinity"));
      continue;
    }
    placement.accepted = placement.requested;
    placement.disposition = PlanDisposition::kAccepted;
    placement.reason = {};
  }

  result.lane_by_vertex.resize(snapshot.vertices.size(), 0U);
  result.lanes.reserve(partitions.size());
  for (std::size_t partition_index = 0U; partition_index < partitions.size(); ++partition_index) {
    auto& partition = partitions[partition_index];
    partition.execution_lane = static_cast<std::uint32_t>(partition_index);
    if (!partition.id) {
      append_error(result, kUnknownOrDuplicate);
      continue;
    }
    for (const VertexId owner : partition.members) {
      if (!owner || owner.value > result.lane_by_vertex.size()) {
        append_error(result, kUnknownOrDuplicate);
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
      append_error(result, kRejected);
    }
    result.placements.push_back(std::move(record));
  }
  std::sort(
      result.placements.begin(), result.placements.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.authored_name != rhs.authored_name
                   ? lhs.authored_name < rhs.authored_name
                   : thread_placement_less(lhs.requested, rhs.requested);
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
      append_error(result, kUnknownOrDuplicate);
    } else if (result.placements[index].reason == diagnostic_id(kUnknownOrDuplicate)) {
      append_error(result, kUnknownOrDuplicate);
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
  canonicalize_plan_lowering_diagnostics(result.errors);
  canonicalize_tokens(result.warning_tokens);
  return result;
}

}  // namespace holoscan::detail
