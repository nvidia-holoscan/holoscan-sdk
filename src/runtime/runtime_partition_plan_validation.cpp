/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_partition_plan_validation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "../compiler/execution_plan_lookup.hpp"
#include "../compiler/execution_plan_state.hpp"
#include "../compiler/partition_authority_validation.hpp"
#include "../compiler/partition_runner_plan.hpp"
#include "../compiler/support_thread_affinity_contracts.hpp"
#include "../threading/thread_placement_contracts.hpp"
#include "runtime_work_inventory.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] bool scheduler_plan_supported(const ExecutionPlanState& plan, bool zero_work) {
  const SchedulerPlanRecord& scheduler = plan.scheduler_plan;
  if (scheduler.ready_policy.id != "holoscan.ready.deterministic_serial" ||
      scheduler.ready_policy.version != 1U || scheduler.reason) {
    return false;
  }
  if (zero_work) {
    return scheduler.execution_lane_count == 0U && scheduler.max_concurrent_activations == 0U &&
           scheduler.worker_count == 0U && scheduler.intra_path_parallelism == 0U &&
           plan.execution_lanes.empty() && plan.placement_probes.empty() &&
           plan.partition_placements.empty();
  }
  const std::uint32_t callback_owner_count =
      static_cast<std::uint32_t>(plan.partition_runners.size());
  return scheduler.execution_lane_count != 0U &&
         scheduler.max_concurrent_activations == callback_owner_count &&
         scheduler.activation_overlap == (callback_owner_count > 1U) &&
         scheduler.worker_count == callback_owner_count &&
         scheduler.intra_path_parallelism == (callback_owner_count == 0U ? 0U : 1U) &&
         plan.execution_lanes.size() == scheduler.execution_lane_count;
}

[[nodiscard]] std::optional<std::uint8_t> placement_probe_bit(std::string_view probe_id) noexcept {
  if (probe_id == "holoscan.placement.syntax/v1") {
    return 1U << 0U;
  }
  if (probe_id == "holoscan.placement.profile_scope/v1") {
    return 1U << 1U;
  }
  if (probe_id == "holoscan.placement.platform/v1") {
    return 1U << 2U;
  }
  if (probe_id == "holoscan.placement.irq_isolation/v1") {
    return 1U << 3U;
  }
  if (probe_id == "holoscan.placement.apply_readback/v1") {
    return 1U << 4U;
  }
  return std::nullopt;
}

[[nodiscard]] bool placement_probe_status_supported(const PlacementProbePlanRecord& probe,
                                                    std::uint8_t bit) noexcept {
  if (probe.reason) {
    return false;
  }
  switch (bit) {
    case 1U << 0U:
    case 1U << 1U:
    case 1U << 2U:
      return probe.status == PlacementProbeStatus::kPassed && !probe.evidence_id.empty();
    case 1U << 3U:
      return probe.status == PlacementProbeStatus::kNotRequested ||
             ((probe.status == PlacementProbeStatus::kPassed ||
               probe.status == PlacementProbeStatus::kFallbackRecorded) &&
              !probe.evidence_id.empty());
    case 1U << 4U:
      return probe.status == PlacementProbeStatus::kRuntimeRequired && !probe.evidence_id.empty();
    default:
      return false;
  }
}

[[nodiscard]] std::optional<std::size_t> dense_partition_index(
    const ExecutionPlanState& plan, const ExecutionPartitionId partition) noexcept {
  if (!partition || partition.value > plan.execution_partitions.size()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(partition.value - 1U);
}

// Validate lane, placement, probe, and affinity inventories as one exact topology relation.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool partition_placement_relations_supported(const ExecutionPlanState& plan) {
  constexpr std::uint8_t kCompleteProbeMask = (1U << 5U) - 1U;
  const ThreadPlacement default_placement;
  std::vector<bool> lane_seen(plan.execution_partitions.size());
  for (std::size_t lane_index = 0U; lane_index < plan.execution_lanes.size(); ++lane_index) {
    const auto& lane = plan.execution_lanes[lane_index];
    const auto partition_index = dense_partition_index(plan, lane.partition);
    const auto* partition =
        partition_index ? find_execution_partition_by_id(plan, lane.partition) : nullptr;
    if (!partition_index || lane.execution_lane != lane_index || partition == nullptr ||
        partition->execution_lane != lane.execution_lane || lane_seen[*partition_index]) {
      return false;
    }
    lane_seen[*partition_index] = true;
  }
  if (std::find(lane_seen.begin(), lane_seen.end(), false) != lane_seen.end()) {
    return false;
  }

  std::vector<bool> placement_seen(plan.execution_partitions.size());
  for (const auto& placement : plan.partition_placements) {
    const auto partition_index = dense_partition_index(plan, placement.partition);
    const auto* partition =
        partition_index ? find_execution_partition_by_id(plan, placement.partition) : nullptr;
    if (!partition_index || partition == nullptr || !partition->authored_name ||
        *partition->authored_name != placement.authored_name ||
        partition->execution_lane != placement.execution_lane ||
        placement.execution_lane >= plan.execution_lanes.size() ||
        plan.execution_lanes[placement.execution_lane].partition != placement.partition ||
        placement_seen[*partition_index] || placement.disposition != PlanDisposition::kAccepted ||
        placement.reason || !placement.accepted || !placement.requested.validate() ||
        !placement.accepted->validate() ||
        !thread_placement_equal(
            *placement.accepted,
            plan.execution_lanes[placement.execution_lane].accepted_placement)) {
      return false;
    }
    placement_seen[*partition_index] = true;
  }
  for (std::size_t partition_index = 0U; partition_index < placement_seen.size();
       ++partition_index) {
    const auto& partition = plan.execution_partitions[partition_index];
    if (!placement_seen[partition_index] &&
        !thread_placement_equal(plan.execution_lanes[partition.execution_lane].accepted_placement,
                                default_placement)) {
      return false;
    }
  }

  std::vector<std::uint8_t> probe_masks(plan.execution_partitions.size());
  std::vector<bool> green_context_probe_seen(plan.execution_partitions.size());
  for (const auto& probe : plan.placement_probes) {
    const auto partition_index = dense_partition_index(plan, probe.partition);
    const auto* partition =
        partition_index ? find_execution_partition_by_id(plan, probe.partition) : nullptr;
    if (probe.probe_id == "cuda.green_context.split") {
      if (!partition_index || partition == nullptr ||
          (probe.status != PlacementProbeStatus::kPassed &&
           probe.status != PlacementProbeStatus::kRuntimeRequired) ||
          probe.reason || probe.evidence_id.empty() || green_context_probe_seen[*partition_index]) {
        return false;
      }
      green_context_probe_seen[*partition_index] = true;
      continue;
    }
    const auto bit = placement_probe_bit(probe.probe_id);
    if (!partition_index || partition == nullptr || !placement_seen[*partition_index] || !bit ||
        !placement_probe_status_supported(probe, *bit)) {
      return false;
    }
    auto& mask = probe_masks[*partition_index];
    if ((mask & *bit) != 0U) {
      return false;
    }
    mask = static_cast<std::uint8_t>(mask | *bit);
  }
  for (std::size_t partition_index = 0U; partition_index < placement_seen.size();
       ++partition_index) {
    const std::uint8_t expected = placement_seen[partition_index] ? kCompleteProbeMask : 0U;
    if (probe_masks[partition_index] != expected) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool partition_records_supported(const ExecutionPlanState& plan) {
  if (!valid_accepted_partition_authority(plan)) {
    return false;
  }

  const bool zero_work = !has_structural_runtime_topology(plan);
  if (!scheduler_plan_supported(plan, zero_work)) {
    return false;
  }
  if (!support_thread_affinity_records_valid(plan.support_thread_affinities, true)) {
    return false;
  }
  if (find_inline_partition_controller_affinity_conflict(plan)) {
    return false;
  }
  if (zero_work) {
    return true;
  }
  if (std::any_of(plan.execution_partitions.begin(),
                  plan.execution_partitions.end(),
                  [&](const auto& partition) {
                    return partition.execution_lane >= plan.scheduler_plan.execution_lane_count;
                  })) {
    return false;
  }
  return partition_placement_relations_supported(plan);
}

}  // namespace holoscan::detail
