/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "partition_cover_lowering.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../core/diagnostic_registry.hpp"
#include "partition_plan_lowering.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

using partitioning::EdgeIndex;
using partitioning::VertexIndex;

[[nodiscard]] bool has_local_operator_predecessor(
    const GraphSnapshotData& snapshot, VertexId target, std::optional<std::size_t> partition_index,
    std::span<const std::optional<std::size_t>> partition_by_vertex) {
  for (const auto& edge : snapshot.edges) {
    const PortRecord* const source =
        find_port(snapshot, resolve_interface_port(snapshot, edge.source));
    const PortRecord* const destination =
        find_port(snapshot, resolve_interface_port(snapshot, edge.target));
    if (source == nullptr || destination == nullptr || destination->owner != target) {
      continue;
    }
    const VertexRecord* const predecessor = find_vertex(snapshot, source->owner);
    if (predecessor == nullptr || predecessor->kind != VertexKind::kOperator) {
      continue;
    }
    if (!partition_index.has_value() ||
        (predecessor->id.value <= partition_by_vertex.size() &&
         partition_by_vertex[static_cast<std::size_t>(predecessor->id.value - 1U)] ==
             partition_index)) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] std::size_t independent_execution_root_count(
    const GraphSnapshotData& snapshot, const ExecutionPlanState& plan,
    std::optional<std::size_t> partition_index,
    std::span<const std::optional<std::size_t>> partition_by_vertex) {
  std::size_t roots = 0U;
  for (const auto& vertex : snapshot.vertices) {
    if (vertex.kind != VertexKind::kOperator) {
      continue;
    }
    if (partition_index.has_value()) {
      if (vertex.id.value > partition_by_vertex.size() ||
          partition_by_vertex[static_cast<std::size_t>(vertex.id.value - 1U)] != partition_index) {
        continue;
      }
    }
    if (has_independent_temporal_source(plan, vertex.id) ||
        !has_local_operator_predecessor(
            snapshot, vertex.id, partition_index, partition_by_vertex)) {
      ++roots;
    }
  }
  // External endpoint vertices represent ingress for their adjacent Operator;
  // they are not independent callback owners. Count the receiving Operator
  // once regardless of its number of imported inputs. A contract-selected
  // physical or clock source remains an independent root even when it also
  // has an upstream Operator input; declaration-only readiness sources do not.
  return roots;
}

// Validate the authored cover as one fail-closed transaction so every rejection shares the same
// precedence and canonical assignment state.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> validate_manual_operator_cover(
    const GraphSnapshotData& snapshot,
    std::vector<std::optional<std::size_t>>& partition_by_vertex) {
  const GraphRecord* const root = find_root_graph(snapshot);
  const bool include_all =
      snapshot.partitions.size() == 1U && snapshot.partitions.front().include_all;
  if (include_all) {
    const auto& partition = snapshot.partitions.front();
    if (root == nullptr || partition.graph != root->id || !partition.members.empty()) {
      return partition_lowering_error(ErrorCode::kInvalidArgument,
                                      "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT");
    }
    partition_by_vertex.resize(snapshot.vertices.size());
    for (const auto& vertex : snapshot.vertices) {
      if (partition_executable(vertex.kind)) {
        partition_by_vertex[static_cast<std::size_t>(vertex.id.value - 1U)] = 0U;
      }
    }
    return {};
  }
  if (snapshot.graphs.size() != 1U ||
      std::any_of(snapshot.graphs.begin(), snapshot.graphs.end(), [](const auto& graph) {
        return graph.parent.has_value();
      })) {
    return partition_lowering_error(ErrorCode::kNotSupported,
                                    "PARTITION_MANUAL_EMBEDDED_GRAPH_UNSUPPORTED");
  }
  const GraphId root_id = snapshot.graphs.front().id;
  partition_by_vertex.resize(snapshot.vertices.size());
  for (std::size_t partition_index = 0U; partition_index < snapshot.partitions.size();
       ++partition_index) {
    const auto& partition = snapshot.partitions[partition_index];
    if (partition.graph != root_id || partition.include_all || partition.members.empty()) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    for (const VertexId member : partition.members) {
      if (!member || member.value > snapshot.vertices.size()) {
        return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      const VertexRecord* const vertex = find_vertex(snapshot, member);
      auto& assigned = partition_by_vertex[static_cast<std::size_t>(member.value - 1U)];
      if (vertex == nullptr || vertex->kind != VertexKind::kOperator || vertex->graph != root_id ||
          assigned.has_value()) {
        return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      assigned = partition_index;
    }
  }
  for (const auto& vertex : snapshot.vertices) {
    if (vertex.kind == VertexKind::kOperator &&
        (!vertex.id || vertex.id.value > partition_by_vertex.size() ||
         !partition_by_vertex[static_cast<std::size_t>(vertex.id.value - 1U)].has_value())) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
  }
  return {};
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> attach_synthetic_vertices(
    const GraphSnapshotData& snapshot, const PartitionProblemStorage& storage,
    std::vector<std::optional<std::size_t>>& partition_by_vertex,
    std::vector<std::vector<VertexId>>& members_by_partition) {
  for (const auto& vertex : snapshot.vertices) {
    if (!partition_executable(vertex.kind) || vertex.kind == VertexKind::kOperator) {
      continue;
    }
    const auto found = storage.executable_index.find(vertex.id.value);
    if (found == storage.executable_index.end()) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    std::vector<std::size_t> adjacent_partitions;
    const auto append_neighbor = [&](VertexIndex neighbor) {
      const VertexId neighbor_id = storage.vertices[neighbor].id;
      const VertexRecord* const record = find_vertex(snapshot, neighbor_id);
      if (record == nullptr || record->kind != VertexKind::kOperator || !neighbor_id ||
          neighbor_id.value > partition_by_vertex.size()) {
        return;
      }
      const auto& partition = partition_by_vertex[static_cast<std::size_t>(neighbor_id.value - 1U)];
      if (partition.has_value()) {
        adjacent_partitions.push_back(*partition);
      }
    };
    const auto& view = storage.vertices[found->second];
    for (const EdgeIndex edge : view.incoming_edges) {
      if (edge >= storage.edges.size()) {
        return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      append_neighbor(storage.edges[edge].source);
    }
    for (const EdgeIndex edge : view.outgoing_edges) {
      if (edge >= storage.edges.size()) {
        return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      append_neighbor(storage.edges[edge].target);
    }
    std::sort(adjacent_partitions.begin(), adjacent_partitions.end());
    adjacent_partitions.erase(std::unique(adjacent_partitions.begin(), adjacent_partitions.end()),
                              adjacent_partitions.end());
    if (adjacent_partitions.size() != 1U) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    const std::size_t selected = adjacent_partitions.front();
    if (!vertex.id || vertex.id.value > partition_by_vertex.size() ||
        selected >= members_by_partition.size()) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    partition_by_vertex[static_cast<std::size_t>(vertex.id.value - 1U)] = selected;
    members_by_partition[selected].push_back(vertex.id);
  }
  return {};
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<partitioning::PartitioningPassResult, Error> build_manual_partition_cover(
    const GraphSnapshotData& snapshot, const PartitionProblemStorage& storage,
    std::span<const std::uint32_t> process_group_by_vertex) {
  const bool include_all =
      snapshot.partitions.size() == 1U && snapshot.partitions.front().include_all;
  std::vector<std::optional<std::size_t>> partition_by_vertex;
  auto cover = validate_manual_operator_cover(snapshot, partition_by_vertex);
  if (!cover) {
    return make_unexpected(std::move(cover.error()));
  }

  std::vector<std::vector<VertexId>> members_by_partition(snapshot.partitions.size());
  for (std::size_t index = 0U; index < snapshot.partitions.size(); ++index) {
    if (snapshot.partitions[index].include_all) {
      for (const auto& vertex : snapshot.vertices) {
        if (partition_executable(vertex.kind)) {
          members_by_partition[index].push_back(vertex.id);
        }
      }
    } else {
      members_by_partition[index] = snapshot.partitions[index].members;
    }
    std::optional<std::uint32_t> process_group;
    for (const VertexId member : members_by_partition[index]) {
      if (!member || member.value > process_group_by_vertex.size()) {
        return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      const std::uint32_t candidate =
          process_group_by_vertex[static_cast<std::size_t>(member.value - 1U)];
      if (candidate == 0U || (process_group.has_value() && *process_group != candidate)) {
        return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      process_group = candidate;
    }
  }
  if (!include_all) {
    auto synthetics =
        attach_synthetic_vertices(snapshot, storage, partition_by_vertex, members_by_partition);
    if (!synthetics) {
      return make_unexpected(std::move(synthetics.error()));
    }
  }
  struct ManualCandidate {
    const PartitionRecord* authored{};
    std::vector<VertexId> members;
  };
  std::vector<ManualCandidate> candidates;
  candidates.reserve(snapshot.partitions.size());
  for (std::size_t index = 0U; index < snapshot.partitions.size(); ++index) {
    auto members = std::move(members_by_partition[index]);
    std::sort(members.begin(), members.end(), [](VertexId lhs, VertexId rhs) {
      return lhs.value < rhs.value;
    });
    candidates.push_back(ManualCandidate{&snapshot.partitions[index], std::move(members)});
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
    return std::lexicographical_compare(
        lhs.members.begin(),
        lhs.members.end(),
        rhs.members.begin(),
        rhs.members.end(),
        [](VertexId left, VertexId right) { return left.value < right.value; });
  });

  partitioning::ValidatedPartitionPlan plan;
  plan.algorithm = partitioning::serial_baseline_algorithm();
  plan.partitions.reserve(candidates.size());
  for (std::size_t index = 0U; index < candidates.size(); ++index) {
    const auto& candidate = candidates[index];
    if (!candidate.authored->id ||
        candidate.authored->id.value > std::numeric_limits<std::uint32_t>::max()) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    const std::uint32_t canonical_id = static_cast<std::uint32_t>(index + 1U);
    plan.partitions.push_back(partitioning::CanonicalPartition{
        canonical_id,
        candidate.members,
        partitioning::AuthoredPartitionConstraintId{
            static_cast<std::uint32_t>(candidate.authored->id.value)},
    });
    const auto& members = plan.partitions.back().members;
    for (std::size_t member_index = 0U; member_index < members.size(); ++member_index) {
      const VertexRecord* const member = find_vertex(snapshot, members[member_index]);
      const bool authored = member != nullptr && member->kind == VertexKind::kOperator;
      plan.decisions.push_back(partitioning::CanonicalPartitionDecision{
          members[member_index],
          canonical_id,
          authored
              ? PartitionDecisionKind::kAuthored
              : (member_index == 0U ? PartitionDecisionKind::kStart : PartitionDecisionKind::kJoin),
          authored ? std::vector{PartitionDecisionReason::kAuthoredPartitionBoundary}
                   : std::vector<PartitionDecisionReason>{},
      });
    }
  }
  std::sort(plan.decisions.begin(), plan.decisions.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.vertex.value < rhs.vertex.value;
  });
  return partitioning::PartitioningPassResult{
      .requested_algorithm = partitioning::serial_baseline_algorithm(),
      .maturity = AlgorithmMaturity::kProduction,
      .selection_source = PlanSelectionSource::kCoreDefault,
      .experimental_warning_required = false,
      .plan = std::move(plan),
  };
}

// Synthetic attachment must inspect the complete constrained cover before committing membership.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> attach_constrained_partition_synthetics(
    const GraphSnapshotData& snapshot, const PartitionProblemStorage& storage,
    std::span<const std::uint32_t> process_group_by_vertex,
    partitioning::ValidatedPartitionPlan& plan) {
  if (storage.authored_constraints.empty()) {
    return {};
  }
  if (process_group_by_vertex.size() != snapshot.vertices.size()) {
    return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
  }
  std::unordered_map<std::uint64_t, std::size_t> partition_by_vertex;
  for (std::size_t index = 0U; index < plan.partitions.size(); ++index) {
    for (const VertexId member : plan.partitions[index].members) {
      partition_by_vertex.emplace(member.value, index);
    }
  }
  for (const auto& vertex : snapshot.vertices) {
    if (!partition_executable(vertex.kind) || vertex.kind == VertexKind::kOperator) {
      continue;
    }
    const auto executable_view = storage.executable_index.find(vertex.id.value);
    const auto current_partition = partition_by_vertex.find(vertex.id.value);
    if (executable_view == storage.executable_index.end() ||
        current_partition == partition_by_vertex.end()) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    std::vector<std::size_t> adjacent_partitions;
    const auto append_neighbor = [&](VertexIndex neighbor) {
      const auto found = partition_by_vertex.find(storage.vertices[neighbor].id.value);
      if (found != partition_by_vertex.end()) {
        adjacent_partitions.push_back(found->second);
      }
    };
    const auto& view = storage.vertices[executable_view->second];
    for (const EdgeIndex edge : view.incoming_edges) {
      append_neighbor(storage.edges[edge].source);
    }
    for (const EdgeIndex edge : view.outgoing_edges) {
      append_neighbor(storage.edges[edge].target);
    }
    std::sort(adjacent_partitions.begin(), adjacent_partitions.end());
    adjacent_partitions.erase(std::unique(adjacent_partitions.begin(), adjacent_partitions.end()),
                              adjacent_partitions.end());
    if (adjacent_partitions.size() != 1U ||
        adjacent_partitions.front() == current_partition->second ||
        !plan.partitions[adjacent_partitions.front()].authored_constraint.has_value()) {
      continue;
    }
    if (!vertex.id || vertex.id.value > process_group_by_vertex.size()) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    const std::uint32_t endpoint_process_group =
        process_group_by_vertex[static_cast<std::size_t>(vertex.id.value - 1U)];
    std::optional<std::uint32_t> anchor_process_group;
    for (const VertexId member : plan.partitions[adjacent_partitions.front()].members) {
      if (!member || member.value > process_group_by_vertex.size()) {
        return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      const std::uint32_t candidate =
          process_group_by_vertex[static_cast<std::size_t>(member.value - 1U)];
      if (candidate == 0U ||
          (anchor_process_group.has_value() && *anchor_process_group != candidate)) {
        return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      anchor_process_group = candidate;
    }
    if (endpoint_process_group == 0U || !anchor_process_group.has_value()) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    if (endpoint_process_group != *anchor_process_group) {
      const auto decision =
          std::find_if(plan.decisions.begin(), plan.decisions.end(), [&](const auto& candidate) {
            return candidate.vertex == vertex.id;
          });
      if (decision == plan.decisions.end()) {
        return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      // An imported endpoint is a topology source, so its process boundary is carried by its
      // outgoing edge and would otherwise be recorded only on the authored neighbor whose
      // decision is already fixed. Preserve both mandatory boundary facts on the retained
      // endpoint-only decision before publishing the final canonical cover.
      decision->decision = PartitionDecisionKind::kBoundarySplit;
      decision->reasons.push_back(PartitionDecisionReason::kProcessBoundary);
      decision->reasons.push_back(PartitionDecisionReason::kExternalEndpointBoundary);
      std::sort(decision->reasons.begin(), decision->reasons.end());
      decision->reasons.erase(std::unique(decision->reasons.begin(), decision->reasons.end()),
                              decision->reasons.end());
      continue;
    }
    auto& old_members = plan.partitions[current_partition->second].members;
    old_members.erase(std::remove(old_members.begin(), old_members.end(), vertex.id),
                      old_members.end());
    auto& anchor_members = plan.partitions[adjacent_partitions.front()].members;
    anchor_members.push_back(vertex.id);
    std::sort(anchor_members.begin(), anchor_members.end(), [](VertexId lhs, VertexId rhs) {
      return lhs.value < rhs.value;
    });
    for (auto& decision : plan.decisions) {
      if (decision.vertex == vertex.id) {
        decision.canonical_partition_id = plan.partitions[adjacent_partitions.front()].canonical_id;
        decision.decision = PartitionDecisionKind::kJoin;
        decision.reasons.clear();
        break;
      }
    }
  }

  plan.partitions.erase(
      std::remove_if(plan.partitions.begin(),
                     plan.partitions.end(),
                     [](const auto& partition) { return partition.members.empty(); }),
      plan.partitions.end());
  std::sort(plan.partitions.begin(), plan.partitions.end(), [](const auto& lhs, const auto& rhs) {
    return std::lexicographical_compare(
        lhs.members.begin(),
        lhs.members.end(),
        rhs.members.begin(),
        rhs.members.end(),
        [](VertexId left, VertexId right) { return left.value < right.value; });
  });
  std::unordered_map<std::uint32_t, std::uint32_t> new_id_by_old;
  for (std::size_t index = 0U; index < plan.partitions.size(); ++index) {
    const std::uint32_t new_id = static_cast<std::uint32_t>(index + 1U);
    new_id_by_old.emplace(plan.partitions[index].canonical_id, new_id);
    plan.partitions[index].canonical_id = new_id;
  }
  for (auto& decision : plan.decisions) {
    const auto remapped = new_id_by_old.find(decision.canonical_partition_id);
    if (remapped == new_id_by_old.end()) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    decision.canonical_partition_id = remapped->second;
  }
  return {};
}

// Validate the complete accepted cover and its derived decisions as one canonical authority.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> validate_final_accepted_partition_cover(
    const GraphSnapshotData& snapshot, std::span<const std::uint32_t> process_group_by_vertex,
    const partitioning::ValidatedPartitionPlan& plan) {
  const std::string_view constraint_token = snapshot.automatic_partition_remaining
                                                ? "PARTITION_PROPOSAL_CONSTRAINT_VIOLATION"
                                                : "PARTITION_COVER_INVALID";
  if (plan.partitions.empty()) {
    const bool zero_work =
        snapshot.partitions.empty() &&
        std::none_of(snapshot.vertices.begin(), snapshot.vertices.end(), [](const auto& vertex) {
          return partition_executable(vertex.kind);
        });
    if (zero_work && plan.decisions.empty()) {
      return {};
    }
    return partition_lowering_error(ErrorCode::kInvalidArgument, constraint_token);
  }
  std::vector<std::optional<std::uint32_t>> partition_by_vertex(snapshot.vertices.size());
  if (process_group_by_vertex.size() != snapshot.vertices.size()) {
    return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
  }
  std::vector<std::uint64_t> authored_ids;
  authored_ids.reserve(snapshot.partitions.size());
  for (std::size_t index = 0U; index < plan.partitions.size(); ++index) {
    const auto& partition = plan.partitions[index];
    if (partition.canonical_id != index + 1U || partition.members.empty() ||
        !std::is_sorted(partition.members.begin(),
                        partition.members.end(),
                        [](VertexId lhs, VertexId rhs) { return lhs.value < rhs.value; })) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    if (index > 0U && !std::lexicographical_compare(
                          plan.partitions[index - 1U].members.begin(),
                          plan.partitions[index - 1U].members.end(),
                          partition.members.begin(),
                          partition.members.end(),
                          [](VertexId lhs, VertexId rhs) { return lhs.value < rhs.value; })) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    std::optional<std::uint32_t> partition_process_group;
    for (const VertexId member : partition.members) {
      if (!member || member.value > partition_by_vertex.size()) {
        return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      const VertexRecord* const vertex = find_vertex(snapshot, member);
      auto& assigned = partition_by_vertex[static_cast<std::size_t>(member.value - 1U)];
      if (vertex == nullptr || !partition_executable(vertex->kind) || assigned.has_value()) {
        return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      const std::uint32_t process_group =
          process_group_by_vertex[static_cast<std::size_t>(member.value - 1U)];
      if (process_group == 0U ||
          (partition_process_group.has_value() && *partition_process_group != process_group)) {
        return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      partition_process_group = process_group;
      assigned = partition.canonical_id;
    }
    if (!partition.authored_constraint.has_value()) {
      continue;
    }
    const auto authored = std::find_if(
        snapshot.partitions.begin(), snapshot.partitions.end(), [&](const auto& candidate) {
          return candidate.id.value == partition.authored_constraint->value;
        });
    if (authored == snapshot.partitions.end()) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, constraint_token);
    }
    authored_ids.push_back(authored->id.value);
    std::vector<VertexId> expected = authored->members;
    if (authored->include_all) {
      expected.clear();
      for (const auto& vertex : snapshot.vertices) {
        if (partition_executable(vertex.kind)) {
          expected.push_back(vertex.id);
        }
      }
    }
    std::sort(expected.begin(), expected.end(), [](VertexId lhs, VertexId rhs) {
      return lhs.value < rhs.value;
    });
    std::vector<VertexId> actual;
    for (const VertexId member : partition.members) {
      const VertexRecord* const vertex = find_vertex(snapshot, member);
      if (vertex != nullptr && (authored->include_all || vertex->kind == VertexKind::kOperator)) {
        actual.push_back(member);
      }
    }
    if (actual != expected) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, constraint_token);
    }
  }
  for (const auto& vertex : snapshot.vertices) {
    if (partition_executable(vertex.kind) &&
        (!vertex.id || vertex.id.value > partition_by_vertex.size() ||
         !partition_by_vertex[static_cast<std::size_t>(vertex.id.value - 1U)].has_value())) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
  }
  std::sort(authored_ids.begin(), authored_ids.end());
  if (authored_ids.size() != snapshot.partitions.size() ||
      std::adjacent_find(authored_ids.begin(), authored_ids.end()) != authored_ids.end()) {
    return partition_lowering_error(ErrorCode::kInvalidArgument, constraint_token);
  }
  std::vector<bool> decision_by_vertex(snapshot.vertices.size(), false);
  for (const auto& decision : plan.decisions) {
    if (!decision.vertex || decision.vertex.value > decision_by_vertex.size() ||
        decision.canonical_partition_id == 0U ||
        decision.canonical_partition_id > plan.partitions.size()) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    const std::size_t vertex_index = static_cast<std::size_t>(decision.vertex.value - 1U);
    if (decision_by_vertex[vertex_index] ||
        partition_by_vertex[vertex_index] != decision.canonical_partition_id) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    decision_by_vertex[vertex_index] = true;
  }
  for (const auto& vertex : snapshot.vertices) {
    if (partition_executable(vertex.kind) &&
        !decision_by_vertex[static_cast<std::size_t>(vertex.id.value - 1U)]) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
  }
  return {};
}

expected<void, Error> validate_partition_execution_roots(const GraphSnapshotData& snapshot,
                                                         ExecutionPlanState& output) {
  const auto publish_failure = [&](Error error) -> unexpected<Error> {
    output.partitioning_plan.accepted_algorithm.reset();
    output.partitioning_plan.disposition = PlanDisposition::kRejected;
    output.partitioning_plan.reason = diagnostic_id(error.message);
    return make_unexpected(std::move(error));
  };
  if (output.partitioning_plan.disposition != PlanDisposition::kAccepted) {
    return {};
  }

  std::vector<std::optional<std::size_t>> partition_by_vertex(snapshot.vertices.size());
  for (std::size_t partition_index = 0U; partition_index < output.execution_partitions.size();
       ++partition_index) {
    for (const VertexId member : output.execution_partitions[partition_index].members) {
      if (!member || member.value > partition_by_vertex.size()) {
        return publish_failure(Error{ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID"});
      }
      const std::size_t member_index = static_cast<std::size_t>(member.value - 1U);
      if (partition_by_vertex[member_index].has_value()) {
        // A corrupted/private plan must never silently reinterpret a vertex as
        // belonging to whichever duplicate partition happened to be visited last.
        return publish_failure(Error{ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID"});
      }
      partition_by_vertex[member_index] = partition_index;
    }
  }

  if (snapshot.partitions.empty() ||
      (snapshot.partitions.size() == 1U && snapshot.partitions.front().include_all)) {
    return {};
  }

  for (std::size_t partition_index = 0U; partition_index < output.execution_partitions.size();
       ++partition_index) {
    if (!output.execution_partitions[partition_index].authored_name.has_value()) {
      continue;
    }
    if (independent_execution_root_count(snapshot, output, partition_index, partition_by_vertex) >
        1U) {
      return publish_failure(Error{ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID"});
    }
  }
  return {};
}

}  // namespace holoscan::detail
