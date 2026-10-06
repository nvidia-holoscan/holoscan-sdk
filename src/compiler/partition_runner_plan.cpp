/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/partition_runner_plan.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "compiler/execution_plan_lookup.hpp"

namespace holoscan::detail {
namespace {

struct RunnerMemberIdentity {
  VertexId vertex{};
  OperatorInstanceId operator_id{};
  std::optional<CUDAGraphRegionPlanRef> fused_cuda_graph_region;
};

[[nodiscard]] const PortPlanRecord* dense_port(const ExecutionPlanState& state,
                                               PortId id) noexcept {
  if (!id || id.value() > state.ports.size()) {
    return nullptr;
  }
  const auto& port = state.ports[static_cast<std::size_t>(id.value() - 1U)];
  return port.id == id ? &port : nullptr;
}

[[nodiscard]] std::optional<std::size_t> member_index(
    const std::vector<RunnerMemberIdentity>& members, VertexId vertex) noexcept {
  for (std::size_t index = 0U; index < members.size(); ++index) {
    if (members[index].vertex == vertex) {
      return index;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ProcessGroupId> exact_process_group(
    const ExecutionPlanState& state, const ExecutionPartitionRecord& partition) noexcept {
  std::optional<ProcessGroupId> result;
  for (const VertexId member : partition.members) {
    const VertexProcessPlacementPlanRecord* placement = nullptr;
    for (const auto& candidate : state.vertex_process_placements) {
      if (candidate.vertex != member) {
        continue;
      }
      if (placement != nullptr) {
        return std::nullopt;
      }
      placement = &candidate;
    }
    if (placement == nullptr || !placement->process_group ||
        (result && *result != placement->process_group)) {
      return std::nullopt;
    }
    result = placement->process_group;
  }
  return result;
}

[[nodiscard]] const ExecutionLanePlanRecord* exact_lane(
    const ExecutionPlanState& state, const ExecutionPartitionRecord& partition) noexcept {
  const ExecutionLanePlanRecord* result = nullptr;
  for (const auto& candidate : state.execution_lanes) {
    if (candidate.partition != partition.id ||
        candidate.execution_lane != partition.execution_lane) {
      continue;
    }
    if (result != nullptr) {
      return nullptr;
    }
    result = &candidate;
  }
  return result;
}

[[nodiscard]] const TemporalTriggerPlanRecord* exact_trigger(const ExecutionPlanState& state,
                                                             OperatorInstanceId owner) noexcept {
  const TemporalTriggerPlanRecord* result = nullptr;
  for (const auto& candidate : state.temporal_triggers) {
    if (candidate.owner.id != owner) {
      continue;
    }
    if (result != nullptr) {
      return nullptr;
    }
    result = &candidate;
  }
  return result;
}

void canonicalize_edges(std::vector<EdgeId>& edges) {
  std::sort(
      edges.begin(), edges.end(), [](EdgeId lhs, EdgeId rhs) { return lhs.value < rhs.value; });
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
}

// Keep runner eligibility checks and the resulting canonical record in one derivation pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::optional<PartitionRunnerPlanRecord> derive_runner(
    const ExecutionPlanState& state, const ExecutionPartitionRecord& partition,
    std::uint32_t ordinal) {
  const auto process_group = exact_process_group(state, partition);
  const auto* lane = exact_lane(state, partition);
  if (!partition.id || !process_group || lane == nullptr) {
    return std::nullopt;
  }

  // The region source becomes the fused callback owner. Other capture-member factories remain
  // available for startup capture but are omitted from steady-state callback dispatch.
  std::vector<RunnerMemberIdentity> members;
  for (const VertexId vertex : partition.members) {
    const auto* fused = find_unique_cuda_graph_execution_region_by_member(state, vertex);
    if (fused != nullptr && vertex != fused->fused_operator) {
      continue;
    }
    const auto begin = members.size();
    for (const auto& candidate : state.executable_factories.operators) {
      if (candidate.vertex == vertex) {
        members.push_back(RunnerMemberIdentity{
            .vertex = candidate.vertex,
            .operator_id = candidate.operator_id,
            .fused_cuda_graph_region = fused != nullptr
                                           ? std::optional{CUDAGraphRegionPlanRef{
                                                 .graph = fused->graph, .region = fused->region}}
                                           : std::nullopt,
        });
      }
    }
    if (members.size() > begin + 1U) {
      return std::nullopt;
    }
  }

  std::vector<std::uint32_t> indegree(members.size(), 0U);
  std::vector<std::vector<std::size_t>> successors(members.size());
  PartitionRunnerPlanRecord result{
      .ordinal = ordinal,
      .partition = partition.id,
      .execution_lane = partition.execution_lane,
      .process_group = *process_group,
  };
  for (const auto& edge : state.payload_edges) {
    // Internal region edges are executed inside the captured CUDA Graph. Boundary owners are
    // rebound to the fused source identity so ordinary neighbors see one operator.
    if (is_cuda_graph_internal_execution_edge(state, edge.edge)) {
      continue;
    }
    const auto* source = dense_port(state, edge.source);
    const auto* target = dense_port(state, edge.target);
    if (source == nullptr || target == nullptr) {
      return std::nullopt;
    }
    const auto* source_region =
        find_unique_cuda_graph_execution_region_by_member(state, source->owner);
    const auto* target_region =
        find_unique_cuda_graph_execution_region_by_member(state, target->owner);
    const VertexId source_owner =
        source_region != nullptr ? source_region->fused_operator : source->owner;
    const VertexId target_owner =
        target_region != nullptr ? target_region->fused_operator : target->owner;
    const auto source_index = member_index(members, source_owner);
    const auto target_index = member_index(members, target_owner);
    if (source_index && target_index) {
      if (*source_index == *target_index) {
        return std::nullopt;
      }
      ++indegree[*target_index];
      successors[*source_index].push_back(*target_index);
      result.internal_edges.push_back(edge.edge);
    } else if (source_index) {
      result.outgoing_boundary_edges.push_back(edge.edge);
    } else if (target_index) {
      result.incoming_boundary_edges.push_back(edge.edge);
    }
  }

  std::vector<std::size_t> ready;
  std::vector<bool> topology_roots(members.size(), false);
  for (std::size_t index = 0U; index < members.size(); ++index) {
    if (indegree[index] == 0U) {
      ready.push_back(index);
      topology_roots[index] = true;
    }
  }
  if (!members.empty() && ready.empty()) {
    return std::nullopt;
  }
  result.operators.reserve(members.size());
  while (!ready.empty()) {
    const auto selected =
        std::min_element(ready.begin(), ready.end(), [&](std::size_t lhs, std::size_t rhs) {
          return members[lhs].vertex.value < members[rhs].vertex.value;
        });
    const std::size_t current = *selected;
    ready.erase(selected);
    const auto* trigger = exact_trigger(state, members[current].operator_id);
    result.operators.push_back(PartitionRunnerOperatorPlanRecord{
        .vertex = members[current].vertex,
        .operator_id = members[current].operator_id,
        .fused_cuda_graph_region = members[current].fused_cuda_graph_region,
        .readiness_source = trigger != nullptr ? trigger->readiness_source : SourceId{},
        .trigger_kind = trigger != nullptr ? std::optional<TemporalRecordTriggerKind>{trigger->kind}
                                           : std::nullopt,
        .topology_root = topology_roots[current],
    });
    for (const std::size_t successor : successors[current]) {
      if (--indegree[successor] == 0U) {
        ready.push_back(successor);
      }
    }
  }
  if (result.operators.size() != members.size()) {
    return std::nullopt;
  }

  canonicalize_edges(result.internal_edges);
  canonicalize_edges(result.incoming_boundary_edges);
  canonicalize_edges(result.outgoing_boundary_edges);
  return result;
}

}  // namespace

std::optional<std::vector<PartitionRunnerPlanRecord>> derive_partition_runner_plan(
    const ExecutionPlanState& state) {
  if (!state.admitted) {
    return std::vector<PartitionRunnerPlanRecord>{};
  }
  if (state.execution_partitions.empty()) {
    if (!state.executable_factories.operators.empty()) {
      return std::nullopt;
    }
    return std::vector<PartitionRunnerPlanRecord>{};
  }

  std::vector<PartitionRunnerPlanRecord> result;
  result.reserve(state.execution_partitions.size());
  for (const ExecutionPartitionRecord& partition : state.execution_partitions) {
    auto runner = derive_runner(state, partition, static_cast<std::uint32_t>(result.size()));
    if (!runner) {
      return std::nullopt;
    }
    // Imported/exported topics are executable partitioning vertices, but they do not reconstruct
    // an Operator callback owner. Preserve their public partition/lane and the corresponding
    // boundary edge on the adjacent callback runner without manufacturing an empty Runtime owner.
    if (runner->operators.empty()) {
      continue;
    }
    result.push_back(std::move(*runner));
  }
  return result;
}

std::optional<InlinePartitionControllerAffinityConflict>
find_inline_partition_controller_affinity_conflict(const ExecutionPlanState& state) {
  const auto controller = std::ranges::find(state.support_thread_affinities,
                                            SupportThreadRole::kController,
                                            &SupportThreadAffinityPlanRecord::role);
  if (controller == state.support_thread_affinities.end() || !controller->accepted ||
      !controller->accepted->affinity) {
    return std::nullopt;
  }

  for (const PartitionRunnerPlanRecord& runner : state.partition_runners) {
    if (std::ranges::count(state.partition_runners,
                           runner.process_group,
                           &PartitionRunnerPlanRecord::process_group) != 1) {
      continue;
    }
    const auto lane = std::ranges::find_if(state.execution_lanes, [&](const auto& candidate) {
      return candidate.partition == runner.partition &&
             candidate.execution_lane == runner.execution_lane;
    });
    if (lane == state.execution_lanes.end() || !lane->accepted_placement.affinity ||
        lane->accepted_placement.affinity->cores == controller->accepted->affinity->cores) {
      continue;
    }
    return InlinePartitionControllerAffinityConflict{
        .process_group = runner.process_group,
        .partition = runner.partition,
        .execution_lane = runner.execution_lane,
        .controller_affinity = *controller->accepted->affinity,
        .partition_affinity = *lane->accepted_placement.affinity,
    };
  }
  return std::nullopt;
}

bool valid_partition_runner_plan(const ExecutionPlanState& state) {
  const auto expected = derive_partition_runner_plan(state);
  if (!expected || state.partition_runners != *expected) {
    return false;
  }
  if (!state.admitted) {
    return true;
  }
  if (find_inline_partition_controller_affinity_conflict(state)) {
    return false;
  }
  const std::uint32_t callback_owner_count = static_cast<std::uint32_t>(expected->size());
  if ((expected->empty() != state.executable_factories.operators.empty()) ||
      state.scheduler_plan.execution_lane_count != state.execution_lanes.size() ||
      state.scheduler_plan.max_concurrent_activations != callback_owner_count ||
      state.scheduler_plan.activation_overlap != (callback_owner_count > 1U) ||
      state.scheduler_plan.worker_count != callback_owner_count ||
      state.scheduler_plan.intra_path_parallelism != (callback_owner_count == 0U ? 0U : 1U)) {
    return false;
  }
  std::vector<std::uint8_t> covered(state.executable_factories.operators.size(), 0U);
  std::vector<std::uint8_t> lanes(state.execution_lanes.size(), 0U);
  for (std::size_t runner_index = 0U; runner_index < expected->size(); ++runner_index) {
    const PartitionRunnerPlanRecord& runner = (*expected)[runner_index];
    if (runner.ordinal != runner_index || !runner.partition || runner.operators.empty() ||
        runner.execution_lane >= lanes.size() || lanes[runner.execution_lane] != 0U) {
      return false;
    }
    lanes[runner.execution_lane] = 1U;
    for (const PartitionRunnerOperatorPlanRecord& member : runner.operators) {
      const auto factory = std::find_if(state.executable_factories.operators.begin(),
                                        state.executable_factories.operators.end(),
                                        [&](const ExecutableOperatorFactoryRecord& candidate) {
                                          return candidate.vertex == member.vertex &&
                                                 candidate.operator_id == member.operator_id;
                                        });
      if (factory == state.executable_factories.operators.end()) {
        return false;
      }
      const auto cover_factory = [&](const ExecutableOperatorFactoryRecord& candidate) {
        const std::size_t factory_index =
            static_cast<std::size_t>(&candidate - state.executable_factories.operators.data());
        if (covered[factory_index] != 0U) {
          return false;
        }
        covered[factory_index] = 1U;
        return true;
      };
      if (!member.fused_cuda_graph_region) {
        if (!cover_factory(*factory)) {
          return false;
        }
        continue;
      }
      // One fused runner member accounts for every original factory used during startup capture.
      const auto* capture_plan = find_unique_cuda_graph_capture_plan_by_region(
          state, member.fused_cuda_graph_region->graph, member.fused_cuda_graph_region->region);
      if (capture_plan == nullptr || capture_plan->members.empty() ||
          capture_plan->members.front().vertex != member.vertex) {
        return false;
      }
      for (const auto& captured : capture_plan->members) {
        const auto captured_factory = std::find_if(
            state.executable_factories.operators.begin(),
            state.executable_factories.operators.end(),
            [&](const auto& candidate) { return candidate.vertex == captured.vertex; });
        if (captured_factory == state.executable_factories.operators.end() ||
            !cover_factory(*captured_factory)) {
          return false;
        }
      }
    }
  }
  return std::ranges::all_of(covered, [](std::uint8_t value) { return value == 1U; });
}

}  // namespace holoscan::detail
