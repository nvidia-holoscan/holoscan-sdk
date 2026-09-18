/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "partition_authority_validation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "authored_partition_membership.hpp"
#include "core/graph_compiler_access.hpp"
#include "execution_plan_state.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

/// Carry the exact public lookup authority derived from a validated dense cover.
struct AcceptedPartitionProjection {
  bool zero_work{};
  std::vector<std::uint32_t> partition_by_vertex;
  std::vector<std::uint32_t> expected_by_id;
  std::vector<ExecutionPartitionStringLookupEntry> expected_by_name;
  std::vector<ExecutionPartitionStringLookupEntry> expected_by_path;
};

[[nodiscard]] constexpr bool executable_vertex(VertexKind kind) noexcept {
  return kind == VertexKind::kOperator || kind == VertexKind::kImportedTopic ||
         kind == VertexKind::kExportedTopic;
}

[[nodiscard]] bool accepted_partition_algorithm_supported(
    const PartitioningPlanRecord& partitioning) noexcept {
  if (partitioning.disposition != PlanDisposition::kAccepted ||
      partitioning.interface_version != 2U || !partitioning.requested_algorithm.has_value() ||
      !partitioning.accepted_algorithm.has_value() ||
      partitioning.requested_algorithm != partitioning.accepted_algorithm || partitioning.reason ||
      partitioning.selection_source != PlanSelectionSource::kCoreDefault ||
      partitioning.maturity != AlgorithmMaturity::kProduction) {
    return false;
  }

  const AlgorithmKey& accepted = *partitioning.accepted_algorithm;
  const bool serial_baseline =
      accepted.id == "holoscan.partition.serial_baseline" && accepted.version == 1U;
  const bool source_chain =
      accepted.id == "holoscan.partition.source_chain" && accepted.version == 1U;
  switch (partitioning.mode) {
    case PartitioningMode::kAutomatic:
      return source_chain && partitioning.authored_partition_count == 0U;
    case PartitioningMode::kConstrainedAutomatic:
      return source_chain && partitioning.authored_partition_count != 0U;
    case PartitioningMode::kManual:
      return serial_baseline && partitioning.authored_partition_count != 0U &&
             partitioning.automatic_partition_count == 0U;
  }
  return false;
}

[[nodiscard]] constexpr bool valid_partition_decision_kind(
    PartitionDecisionKind decision) noexcept {
  switch (decision) {
    case PartitionDecisionKind::kStart:
    case PartitionDecisionKind::kJoin:
    case PartitionDecisionKind::kBoundarySplit:
    case PartitionDecisionKind::kFanInSplit:
    case PartitionDecisionKind::kFanOutLegSplit:
    case PartitionDecisionKind::kCollapseRejected:
    case PartitionDecisionKind::kAuthored:
      return true;
  }
  return false;
}

[[nodiscard]] constexpr bool valid_partition_decision_reason(
    PartitionDecisionReason reason) noexcept {
  switch (reason) {
    case PartitionDecisionReason::kProcessBoundary:
    case PartitionDecisionReason::kExternalEndpointBoundary:
    case PartitionDecisionReason::kDeviceTransferIncompatibility:
    case PartitionDecisionReason::kRegionBackendAdmission:
    case PartitionDecisionReason::kExplicitThreadPlacement:
    case PartitionDecisionReason::kLifecycleIsolation:
    case PartitionDecisionReason::kReadinessIsolation:
    case PartitionDecisionReason::kIndependentReadinessRoot:
    case PartitionDecisionReason::kSourceRoot:
    case PartitionDecisionReason::kFanIn:
    case PartitionDecisionReason::kFanOutLeg:
    case PartitionDecisionReason::kCollapseRejected:
    case PartitionDecisionReason::kLinearChainJoin:
    case PartitionDecisionReason::kAuthoredPartitionBoundary:
      return true;
  }
  return false;
}

template <typename Values>
[[nodiscard]] bool has_duplicates(Values values) {
  std::sort(values.begin(), values.end());
  return std::adjacent_find(values.begin(), values.end()) != values.end();
}

[[nodiscard]] const PortRecord* find_graph_port(std::span<const PortRecord> ports,
                                                GraphPortId port) noexcept {
  const auto found =
      std::find_if(ports.begin(), ports.end(), [&](const auto& item) { return item.id == port; });
  return found == ports.end() ? nullptr : &*found;
}

[[nodiscard]] VertexId resolve_edge_owner(const GraphView& graph, GraphPortId port) {
  const auto ports = graph.ports();
  const auto bindings = graph.interface_bindings();
  for (std::size_t depth = 0U; depth <= bindings.size(); ++depth) {
    const PortRecord* const record = find_graph_port(ports, port);
    if (record == nullptr || record->owner_kind != PortOwnerKind::kGraphInterface) {
      return record == nullptr ? VertexId{} : record->owner;
    }
    const auto binding = std::find_if(bindings.begin(), bindings.end(), [&](const auto& item) {
      return item.public_port == port;
    });
    if (binding == bindings.end()) {
      return {};
    }
    port = binding->internal_port;
  }
  return {};
}

[[nodiscard]] bool empty_partition_authority_valid(const ExecutionPlanState& state) {
  return state.execution_partitions.empty() && state.partition_decisions.empty() &&
         state.schedule_edges.empty() &&
         state.partitioning_plan.mode == PartitioningMode::kAutomatic &&
         state.partitioning_plan.authored_partition_count == 0U &&
         state.partitioning_plan.automatic_partition_count == 0U &&
         state.execution_partition_index_by_id.empty() &&
         state.execution_partition_index_by_operator_id.empty() &&
         state.execution_partition_index_by_root_operator.empty() &&
         state.execution_partition_index_by_authored_name.empty() &&
         state.execution_partition_index_by_operator_path.empty();
}

// Authenticate selection provenance against the immutable authoring snapshot, including omitted
// claims. Flat partitions keep their existing authority contract.
[[nodiscard]] bool atomic_partition_selection_valid(const ExecutionPlanState& state,
                                                    const ExecutionPartitionRecord& partition) {
  const auto declarations = state.graph.partitions();
  const auto authored =
      std::find_if(declarations.begin(), declarations.end(), [&](const auto& item) {
        return partition.authored_name && item.name == *partition.authored_name;
      });
  if (partition.graph_members.empty() &&
      (authored == declarations.end() || authored->graph_members.empty())) {
    return true;
  }
  if (authored == declarations.end() || partition.graph_members != authored->graph_members) {
    return false;
  }
  const auto snapshot = GraphCompilerAccess::retain_snapshot(state.graph);
  if (!snapshot)
    return false;
  const auto expected = resolve_authored_partition_members(*snapshot, *authored);
  if (!expected)
    return false;
  std::vector<VertexId> actual;
  for (const VertexId member : partition.members) {
    const auto* vertex = find_vertex(*snapshot, member);
    if (vertex != nullptr && vertex->kind == VertexKind::kOperator)
      actual.push_back(member);
  }
  return actual == *expected;
}

// Validate the public cover and all derived decision relations as one fail-closed authority.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::optional<AcceptedPartitionProjection> validate_partition_cover_and_decisions(
    const ExecutionPlanState& state) {
  // Every frozen atomic declaration must survive in the accepted inventory. This forward join
  // also detects a removed claim combined with a renamed/removed accepted authored name.
  for (const auto& authored : state.graph.partitions()) {
    if (!authored.graph_members.empty() &&
        std::count_if(
            state.execution_partitions.begin(),
            state.execution_partitions.end(),
            [&](const auto& partition) { return partition.authored_name == authored.name; }) != 1) {
      return std::nullopt;
    }
  }
  const auto vertices = state.graph.vertices();
  std::vector<const VertexRecord*> executable_vertices;
  executable_vertices.reserve(vertices.size());
  for (const auto& vertex : vertices) {
    if (executable_vertex(vertex.kind)) {
      executable_vertices.push_back(&vertex);
    }
  }
  if (state.partition_decisions.size() != executable_vertices.size()) {
    return std::nullopt;
  }
  if (executable_vertices.empty()) {
    if (!empty_partition_authority_valid(state)) {
      return std::nullopt;
    }
    return AcceptedPartitionProjection{.zero_work = true};
  }
  if (state.execution_partitions.empty() ||
      state.execution_partitions.size() > executable_vertices.size()) {
    return std::nullopt;
  }

  AcceptedPartitionProjection projection;
  projection.partition_by_vertex.assign(vertices.size(), 0U);
  projection.expected_by_id.assign(state.execution_partitions.size(), 0U);
  std::vector<std::string_view> authored_names;
  std::vector<std::string_view> operator_paths;
  std::uint32_t authored_count = 0U;
  std::uint32_t automatic_count = 0U;
  for (std::size_t partition_index = 0U; partition_index < state.execution_partitions.size();
       ++partition_index) {
    const std::uint32_t table_index = static_cast<std::uint32_t>(partition_index + 1U);
    const auto& partition = state.execution_partitions[partition_index];
    if (partition.id.value != table_index || partition.members.empty() ||
        !atomic_partition_selection_valid(state, partition)) {
      return std::nullopt;
    }
    projection.expected_by_id[partition_index] = table_index;
    if (partition.authored_name) {
      if (partition.authored_name->empty()) {
        return std::nullopt;
      }
      ++authored_count;
      authored_names.push_back(*partition.authored_name);
      projection.expected_by_name.push_back({*partition.authored_name, table_index});
    } else {
      ++automatic_count;
    }

    VertexId previous_member;
    std::size_t operator_path_index = 0U;
    for (const VertexId member : partition.members) {
      if (!member || member.value > vertices.size() ||
          (previous_member && member.value <= previous_member.value)) {
        return std::nullopt;
      }
      previous_member = member;
      const std::size_t vertex_index = static_cast<std::size_t>(member.value - 1U);
      const VertexRecord& vertex = vertices[vertex_index];
      if (vertex.id != member || !executable_vertex(vertex.kind) ||
          projection.partition_by_vertex[vertex_index] != 0U) {
        return std::nullopt;
      }
      projection.partition_by_vertex[vertex_index] = table_index;
      if (vertex.kind != VertexKind::kOperator) {
        continue;
      }
      const auto expected_path = canonical_operator_path(state.graph.graphs(), vertex);
      if (!expected_path || operator_path_index >= partition.operator_paths.size() ||
          partition.operator_paths[operator_path_index] != *expected_path) {
        return std::nullopt;
      }
      operator_paths.push_back(partition.operator_paths[operator_path_index]);
      projection.expected_by_path.push_back(
          {partition.operator_paths[operator_path_index], table_index});
      ++operator_path_index;
    }
    if (operator_path_index != partition.operator_paths.size()) {
      return std::nullopt;
    }
  }

  if (has_duplicates(authored_names) || has_duplicates(operator_paths) ||
      authored_count != state.partitioning_plan.authored_partition_count ||
      automatic_count != state.partitioning_plan.automatic_partition_count ||
      authored_count + automatic_count != state.execution_partitions.size()) {
    return std::nullopt;
  }

  for (std::size_t executable_index = 0U; executable_index < executable_vertices.size();
       ++executable_index) {
    const VertexRecord& vertex = *executable_vertices[executable_index];
    if (!vertex.id || vertex.id.value > projection.partition_by_vertex.size()) {
      return std::nullopt;
    }
    const std::size_t vertex_index = static_cast<std::size_t>(vertex.id.value - 1U);
    const auto& decision = state.partition_decisions[executable_index];
    if (projection.partition_by_vertex[vertex_index] == 0U || decision.vertex != vertex.id ||
        decision.partition.value != projection.partition_by_vertex[vertex_index] ||
        !valid_partition_decision_kind(decision.decision) ||
        !std::all_of(
            decision.reasons.begin(), decision.reasons.end(), valid_partition_decision_reason) ||
        !std::is_sorted(decision.reasons.begin(), decision.reasons.end()) ||
        std::adjacent_find(decision.reasons.begin(), decision.reasons.end()) !=
            decision.reasons.end()) {
      return std::nullopt;
    }
  }
  for (std::size_t vertex_index = 0U; vertex_index < vertices.size(); ++vertex_index) {
    if (executable_vertex(vertices[vertex_index].kind) !=
        (projection.partition_by_vertex[vertex_index] != 0U)) {
      return std::nullopt;
    }
  }

  const auto sort_entries = [](auto& entries) {
    std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
      return left.key < right.key;
    });
  };
  sort_entries(projection.expected_by_name);
  sort_entries(projection.expected_by_path);
  return projection;
}

[[nodiscard]] bool partition_lookup_indices_valid(const ExecutionPlanState& state,
                                                  const AcceptedPartitionProjection& projection) {
  if (state.execution_partition_index_by_id != projection.expected_by_id ||
      state.execution_partition_index_by_authored_name != projection.expected_by_name ||
      state.execution_partition_index_by_operator_path != projection.expected_by_path) {
    return false;
  }

  const auto vertices = state.graph.vertices();
  std::vector<std::uint32_t> expected_by_operator(state.executable_factories.operators.size(), 0U);
  std::size_t root_operator_count = 0U;
  for (const auto& factory : state.executable_factories.operators) {
    if (factory.root_operator_index) {
      root_operator_count = std::max(root_operator_count, *factory.root_operator_index + 1U);
    }
  }
  std::vector<std::uint32_t> expected_by_root(root_operator_count, 0U);
  for (const auto& factory : state.executable_factories.operators) {
    if (factory.operator_id == 0U || factory.operator_id > expected_by_operator.size() ||
        !factory.vertex || factory.vertex.value > projection.partition_by_vertex.size()) {
      return false;
    }
    const std::size_t vertex_index = static_cast<std::size_t>(factory.vertex.value - 1U);
    if (vertices[vertex_index].kind != VertexKind::kOperator) {
      return false;
    }
    const std::uint32_t table_index = projection.partition_by_vertex[vertex_index];
    auto& operator_entry = expected_by_operator[static_cast<std::size_t>(factory.operator_id - 1U)];
    if (table_index == 0U || operator_entry != 0U) {
      return false;
    }
    operator_entry = table_index;
    if (factory.root_operator_index) {
      auto& root_entry = expected_by_root[*factory.root_operator_index];
      if (root_entry != 0U) {
        return false;
      }
      root_entry = table_index;
    }
  }
  return state.execution_partition_index_by_operator_id == expected_by_operator &&
         state.execution_partition_index_by_root_operator == expected_by_root;
}

[[nodiscard]] bool partition_schedule_edges_valid(
    const ExecutionPlanState& state, std::span<const std::uint32_t> partition_by_vertex) {
  std::vector<ScheduleEdgeRecord> expected_schedule_edges;
  for (const auto& edge : state.graph.edges()) {
    const VertexId source_owner = resolve_edge_owner(state.graph, edge.source);
    const VertexId destination_owner = resolve_edge_owner(state.graph, edge.target);
    if (!source_owner || source_owner.value > partition_by_vertex.size() || !destination_owner ||
        destination_owner.value > partition_by_vertex.size()) {
      continue;
    }
    const ExecutionPartitionId source{
        partition_by_vertex[static_cast<std::size_t>(source_owner.value - 1U)]};
    const ExecutionPartitionId destination{
        partition_by_vertex[static_cast<std::size_t>(destination_owner.value - 1U)]};
    if (source && destination && source != destination) {
      expected_schedule_edges.push_back({edge.id, source, destination});
    }
  }
  if (state.schedule_edges.size() != expected_schedule_edges.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < expected_schedule_edges.size(); ++index) {
    if (state.schedule_edges[index].edge != expected_schedule_edges[index].edge ||
        state.schedule_edges[index].source != expected_schedule_edges[index].source ||
        state.schedule_edges[index].destination != expected_schedule_edges[index].destination) {
      return false;
    }
  }
  return true;
}

}  // namespace

// Validate the public cover and all derived lookup/crossing-edge relations as one authority.
// Runtime-only scheduler, placement, runner, and resource relationships are validated separately.
bool valid_accepted_partition_authority(const ExecutionPlanState& state) {
  if (!accepted_partition_algorithm_supported(state.partitioning_plan)) {
    return false;
  }
  const auto projection = validate_partition_cover_and_decisions(state);
  if (!projection) {
    return false;
  }
  if (projection->zero_work) {
    return true;
  }
  return partition_lookup_indices_valid(state, *projection) &&
         partition_schedule_edges_valid(state, projection->partition_by_vertex);
}

}  // namespace holoscan::detail
