/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include <holoscan/core/execution_plan.hpp>

#include "../compiler/execution_plan_state.hpp"

namespace holoscan::detail {

/// Mutate frozen partition and zero-work plan facts for source-tree Runtime tests.
class RuntimePartitionTestPlanAccess {
 public:
  /// @brief Identify one deliberately malformed partition/scheduler record used by startup tests.
  enum class PartitionMutation {
    kAcceptedAlgorithm,
    kAcceptedAlgorithmVersion,
    kRequestedAlgorithm,
    kAlgorithmMaturity,
    kInterfaceVersion,
    kDisposition,
    kSelectionSource,
    kPartitioningMode,
    kAuthoredPartitionCount,
    kAutomaticPartitionCount,
    kReadyPolicy,
    kPartitionId,
    kPartitionLane,
    kPartitionAuthoredName,
    kPartitionOperatorPath,
    kPartitionOperatorPathOrder,
    kPartitionOperatorPathIndex,
    kPartitionMemberMissing,
    kPartitionMemberDuplicate,
    kPartitionMemberOrder,
    kGraphInterfaceMemberAndDecision,
    kDecisionMissing,
    kDecisionPartitionMismatch,
    kDecisionOrder,
    kScheduleEdgeOrder,
    kConstrainedModeManual,
    kLanePartitionMismatch,
    kPlacementOrphan,
    kPlacementDuplicate,
    kPlacementPartitionMismatch,
    kProbeOrphan,
    kProbeDuplicate,
    kProbePartitionMismatch,
    kSupportAffinityDuplicate,
    kSupportAffinityUnknownRole,
    kSupportAffinityMalformedRequest,
    kInlineControllerAffinityConflict,
    kPartitionRunnerOrdinal,
  };

  /// @brief Identify one malformed zero-work scheduler, placement, or Runtime inventory fact.
  enum class ZeroWorkMutation {
    kWorkerCount,
    kExecutionLaneCount,
    kMaxConcurrentActivations,
    kIntraPathParallelism,
    kExecutionLane,
    kOrphanPartition,
    kPartitionDecision,
    kScheduleEdge,
    kPlacementProbe,
    kPartitionPlacement,
    kOperatorFactory,
    kMaterializationObligation,
  };

  /// @brief Copy one injected plan and corrupt exactly one frozen partition/runtime invariant.
  [[nodiscard]] static ExecutionPlan mutate_partition(const ExecutionPlan& plan,
                                                      PartitionMutation mutation) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    const auto unmatched_partition = [&]() {
      ExecutionPartitionId candidate{1U};
      while (std::any_of(state->execution_partitions.begin(),
                         state->execution_partitions.end(),
                         [&](const auto& partition) { return partition.id == candidate; })) {
        if (candidate.value == std::numeric_limits<std::uint32_t>::max()) {
          throw std::logic_error{"no unmatched execution partition id remains"};
        }
        ++candidate.value;
      }
      return candidate;
    };
    switch (mutation) {
      case PartitionMutation::kAcceptedAlgorithm:
        state->partitioning_plan.accepted_algorithm = AlgorithmKey{"unsupported", 1U};
        break;
      case PartitionMutation::kAcceptedAlgorithmVersion:
        if (auto& algorithm = state->partitioning_plan.accepted_algorithm; algorithm) {
          ++algorithm->version;
        }
        break;
      case PartitionMutation::kRequestedAlgorithm:
        state->partitioning_plan.requested_algorithm = AlgorithmKey{"unsupported", 1U};
        break;
      case PartitionMutation::kAlgorithmMaturity:
        state->partitioning_plan.maturity = AlgorithmMaturity::kExperimental;
        break;
      case PartitionMutation::kInterfaceVersion:
        ++state->partitioning_plan.interface_version;
        break;
      case PartitionMutation::kDisposition:
        state->partitioning_plan.disposition = PlanDisposition::kRejected;
        break;
      case PartitionMutation::kSelectionSource:
        state->partitioning_plan.selection_source = PlanSelectionSource::kExplicitFallback;
        break;
      case PartitionMutation::kPartitioningMode:
        if (state->partitioning_plan.mode == PartitioningMode::kAutomatic) {
          return {};
        }
        state->partitioning_plan.mode = PartitioningMode::kAutomatic;
        break;
      case PartitionMutation::kAuthoredPartitionCount:
        ++state->partitioning_plan.authored_partition_count;
        break;
      case PartitionMutation::kAutomaticPartitionCount:
        ++state->partitioning_plan.automatic_partition_count;
        break;
      case PartitionMutation::kReadyPolicy:
        state->scheduler_plan.ready_policy.version = 2U;
        break;
      case PartitionMutation::kPartitionId:
        state->execution_partitions.front().id.value += 1U;
        break;
      case PartitionMutation::kPartitionLane:
        state->execution_partitions.front().execution_lane =
            state->scheduler_plan.execution_lane_count;
        break;
      case PartitionMutation::kPartitionAuthoredName:
        if (state->execution_partitions.empty() ||
            !state->execution_partitions.front().authored_name.has_value()) {
          return {};
        }
        state->execution_partitions.front().authored_name.reset();
        break;
      case PartitionMutation::kPartitionOperatorPath:
        if (state->execution_partitions.empty() ||
            state->execution_partitions.front().operator_paths.empty()) {
          return {};
        }
        state->execution_partitions.front().operator_paths.front() += "-mutated";
        break;
      case PartitionMutation::kPartitionOperatorPathOrder: {
        const auto partition = std::find_if(
            state->execution_partitions.begin(),
            state->execution_partitions.end(),
            [](const auto& candidate) { return candidate.operator_paths.size() >= 2U; });
        if (partition == state->execution_partitions.end()) {
          return {};
        }
        std::reverse(partition->operator_paths.begin(), partition->operator_paths.end());
        break;
      }
      case PartitionMutation::kPartitionOperatorPathIndex:
        if (state->execution_partition_index_by_operator_path.empty()) {
          return {};
        }
        state->execution_partition_index_by_operator_path.front().table_index =
            std::numeric_limits<std::uint32_t>::max();
        break;
      case PartitionMutation::kPartitionMemberMissing:
        state->execution_partitions.front().members.pop_back();
        break;
      case PartitionMutation::kPartitionMemberDuplicate:
        state->execution_partitions.front().members.push_back(
            state->execution_partitions.front().members.front());
        break;
      case PartitionMutation::kPartitionMemberOrder: {
        const auto partition = std::find_if(state->execution_partitions.begin(),
                                            state->execution_partitions.end(),
                                            [](const auto& candidate) {
                                              return candidate.members.size() >= 2U &&
                                                     candidate.operator_paths.size() >= 2U;
                                            });
        if (partition == state->execution_partitions.end()) {
          return {};
        }
        std::reverse(partition->members.begin(), partition->members.end());
        break;
      }
      case PartitionMutation::kGraphInterfaceMemberAndDecision: {
        const auto graph_interface = std::find_if(
            state->graph.vertices().begin(), state->graph.vertices().end(), [](const auto& vertex) {
              return vertex.kind == VertexKind::kGraphInterface;
            });
        if (graph_interface == state->graph.vertices().end() ||
            state->execution_partitions.empty()) {
          return {};
        }
        auto& partition = state->execution_partitions.front();
        partition.members.insert(std::lower_bound(partition.members.begin(),
                                                  partition.members.end(),
                                                  graph_interface->id.value,
                                                  [](VertexId member, std::uint64_t value) {
                                                    return member.value < value;
                                                  }),
                                 graph_interface->id);
        state->partition_decisions.push_back(PartitionDecisionRecord{
            .vertex = graph_interface->id,
            .partition = partition.id,
            .decision = PartitionDecisionKind::kJoin,
        });
        std::sort(state->partition_decisions.begin(),
                  state->partition_decisions.end(),
                  [](const auto& left, const auto& right) {
                    return left.vertex.value < right.vertex.value;
                  });
        break;
      }
      case PartitionMutation::kDecisionMissing:
        if (state->partition_decisions.empty()) {
          return {};
        }
        state->partition_decisions.pop_back();
        break;
      case PartitionMutation::kDecisionPartitionMismatch:
        if (state->partition_decisions.empty()) {
          return {};
        }
        state->partition_decisions.front().partition = ExecutionPartitionId{};
        break;
      case PartitionMutation::kDecisionOrder:
        if (state->partition_decisions.size() < 2U) {
          return {};
        }
        std::swap(state->partition_decisions[0], state->partition_decisions[1]);
        break;
      case PartitionMutation::kScheduleEdgeOrder:
        if (state->schedule_edges.size() < 2U) {
          return {};
        }
        std::swap(state->schedule_edges[0], state->schedule_edges[1]);
        break;
      case PartitionMutation::kConstrainedModeManual:
        if (state->partitioning_plan.mode != PartitioningMode::kConstrainedAutomatic ||
            state->partitioning_plan.automatic_partition_count != 0U) {
          return {};
        }
        state->partitioning_plan.mode = PartitioningMode::kManual;
        break;
      case PartitionMutation::kLanePartitionMismatch:
        state->execution_lanes.front().partition = unmatched_partition();
        break;
      case PartitionMutation::kPlacementOrphan:
        state->partition_placements.emplace_back();
        break;
      case PartitionMutation::kPlacementDuplicate:
        if (state->partition_placements.empty()) {
          throw std::logic_error{"partition placement mutation fixture is empty"};
        }
        state->partition_placements.push_back(state->partition_placements.front());
        break;
      case PartitionMutation::kPlacementPartitionMismatch:
        if (state->partition_placements.empty()) {
          throw std::logic_error{"partition placement mutation fixture is empty"};
        }
        state->partition_placements.front().partition = unmatched_partition();
        break;
      case PartitionMutation::kProbeOrphan:
        state->placement_probes.emplace_back();
        break;
      case PartitionMutation::kProbeDuplicate:
        if (state->placement_probes.empty()) {
          throw std::logic_error{"partition placement probe mutation fixture is empty"};
        }
        state->placement_probes.push_back(state->placement_probes.front());
        break;
      case PartitionMutation::kProbePartitionMismatch:
        if (state->placement_probes.empty()) {
          throw std::logic_error{"partition placement probe mutation fixture is empty"};
        }
        state->placement_probes.front().partition = unmatched_partition();
        break;
      case PartitionMutation::kSupportAffinityDuplicate:
        if (state->support_thread_affinities.empty()) {
          return {};
        }
        state->support_thread_affinities.push_back(state->support_thread_affinities.front());
        break;
      case PartitionMutation::kSupportAffinityUnknownRole:
        if (state->support_thread_affinities.empty()) {
          return {};
        }
        state->support_thread_affinities.front().role =
            static_cast<SupportThreadRole>(std::numeric_limits<std::uint8_t>::max());
        break;
      case PartitionMutation::kSupportAffinityMalformedRequest:
        if (state->support_thread_affinities.empty()) {
          return {};
        }
        state->support_thread_affinities.front().requested.affinity = CPUAffinity{};
        break;
      case PartitionMutation::kInlineControllerAffinityConflict: {
        const auto controller = std::ranges::find(state->support_thread_affinities,
                                                  SupportThreadRole::kController,
                                                  &SupportThreadAffinityPlanRecord::role);
        if (controller == state->support_thread_affinities.end() ||
            state->execution_lanes.size() != 1U || state->partition_placements.size() != 1U) {
          return {};
        }
        const auto& controller_placement = controller->accepted;
        auto& partition_placement = state->partition_placements.front().accepted;
        if (!controller_placement.has_value() || !partition_placement.has_value()) {
          return {};
        }
        const auto& affinity = controller_placement->affinity;
        if (!affinity.has_value()) {
          return {};
        }
        const std::uint32_t controller_core = affinity->cores.front();
        const std::uint32_t conflicting_core = controller_core == 0U ? 1U : 0U;
        state->execution_lanes.front().accepted_placement.affinity = CPUAffinity{conflicting_core};
        state->partition_placements.front().requested.affinity = CPUAffinity{conflicting_core};
        partition_placement->affinity = CPUAffinity{conflicting_core};
        break;
      }
      case PartitionMutation::kPartitionRunnerOrdinal:
        ++state->partition_runners.front().ordinal;
        break;
    }
    return ExecutionPlanAccess::make(std::move(state));
  }

  /// @brief Copy one accepted empty-Graph plan and corrupt one zero-work invariant.
  [[nodiscard]] static ExecutionPlan mutate_zero_work(const ExecutionPlan& plan,
                                                      ZeroWorkMutation mutation) {
    const auto source = ExecutionPlanAccess::retain(plan);
    if (!source) {
      return {};
    }
    auto state = std::make_shared<ExecutionPlanState>(*source);
    switch (mutation) {
      case ZeroWorkMutation::kWorkerCount:
        state->scheduler_plan.worker_count = 1U;
        break;
      case ZeroWorkMutation::kExecutionLaneCount:
        state->scheduler_plan.execution_lane_count = 1U;
        break;
      case ZeroWorkMutation::kMaxConcurrentActivations:
        state->scheduler_plan.max_concurrent_activations = 1U;
        break;
      case ZeroWorkMutation::kIntraPathParallelism:
        state->scheduler_plan.intra_path_parallelism = 1U;
        break;
      case ZeroWorkMutation::kExecutionLane:
        state->execution_lanes.push_back(ExecutionLanePlanRecord{.execution_lane = 0U});
        break;
      case ZeroWorkMutation::kOrphanPartition:
        state->execution_partitions.push_back(ExecutionPartitionRecord{
            .id = ExecutionPartitionId{1U},
            .members = {VertexId{1U}},
            .execution_lane = 0U,
        });
        break;
      case ZeroWorkMutation::kPartitionDecision:
        state->partition_decisions.emplace_back();
        break;
      case ZeroWorkMutation::kScheduleEdge:
        state->schedule_edges.emplace_back();
        break;
      case ZeroWorkMutation::kPlacementProbe:
        state->placement_probes.emplace_back();
        break;
      case ZeroWorkMutation::kPartitionPlacement:
        state->partition_placements.emplace_back();
        break;
      case ZeroWorkMutation::kOperatorFactory:
        state->executable_factories.operators.emplace_back();
        break;
      case ZeroWorkMutation::kMaterializationObligation:
        state->profile_manifest.materialization_obligations.emplace_back();
        break;
    }
    return ExecutionPlanAccess::make(std::move(state));
  }
};

}  // namespace holoscan::detail
