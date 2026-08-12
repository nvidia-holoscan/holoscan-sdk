/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "partition_plan_lowering.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>

#include "../core/diagnostic_registry.hpp"
#include "detail/snapshot_lookup.hpp"
#include "partitioning/partition_planner_catalog.hpp"
#include "partitioning/partitioning_pass.hpp"

namespace holoscan::detail {
namespace {

using partitioning::EdgeIndex;
using partitioning::ExecutionDomainKey;
using partitioning::PartitionBoundary;
using partitioning::PartitionBoundaryMask;
using partitioning::PartitionEdgeView;
using partitioning::PartitionProblem;
using partitioning::PartitionVertexView;
using partitioning::VertexIndex;

struct PartitionProblemStorage {
  std::vector<std::vector<EdgeIndex>> incoming;
  std::vector<std::vector<EdgeIndex>> outgoing;
  std::vector<std::vector<RegionId>> source_regions;
  std::vector<PartitionVertexView> vertices;
  std::vector<PartitionEdgeView> edges;
  std::unordered_map<std::uint64_t, VertexIndex> executable_index;
};

[[nodiscard]] unexpected<Error> partition_error(ErrorCode code, std::string_view token) {
  return make_unexpected(Error{code, std::string(token)});
}

[[nodiscard]] bool executable(VertexKind kind) noexcept {
  return kind == VertexKind::kOperator || kind == VertexKind::kImportedTopic ||
         kind == VertexKind::kExportedTopic;
}

[[nodiscard]] std::vector<RegionId> regions_for_vertex(const GraphSnapshotData& snapshot,
                                                       const VertexRecord& vertex) {
  std::vector<RegionId> result;
  for (const auto& region : snapshot.regions) {
    const bool direct_member =
        std::find(region.members.begin(), region.members.end(), vertex.id) != region.members.end();
    const bool embedded_member =
        std::any_of(region.graph_members.begin(), region.graph_members.end(), [&](GraphId graph) {
          return is_descendant_or_self(snapshot, vertex.graph, graph);
        });
    if (direct_member || embedded_member) {
      result.push_back(region.id);
    }
  }
  return result;
}

[[nodiscard]] expected<void, Error> append_executable_vertices(const GraphSnapshotData& snapshot,
                                                               PartitionProblemStorage& storage) {
  for (const auto& vertex : snapshot.vertices) {
    if (!executable(vertex.kind)) {
      continue;
    }
    if (storage.vertices.size() >= std::numeric_limits<VertexIndex>::max()) {
      return partition_error(ErrorCode::kResourceExhausted, "PARTITION_LIMIT_EXCEEDED");
    }
    const auto index = static_cast<VertexIndex>(storage.vertices.size());
    storage.executable_index.emplace(vertex.id.value, index);
    storage.incoming.emplace_back();
    storage.outgoing.emplace_back();
    storage.source_regions.push_back(regions_for_vertex(snapshot, vertex));
    storage.vertices.emplace_back();
  }
  return {};
}

[[nodiscard]] expected<void, Error> append_problem_edge(
    const GraphSnapshotData& snapshot, const EdgeRecord& edge,
    std::span<const std::uint32_t> process_group_by_vertex, PartitionProblemStorage& storage) {
  const GraphPortId resolved_source = resolve_interface_port(snapshot, edge.source);
  const GraphPortId resolved_target = resolve_interface_port(snapshot, edge.target);
  const PortRecord* const source_port = find_port(snapshot, resolved_source);
  const PortRecord* const target_port = find_port(snapshot, resolved_target);
  if (source_port == nullptr || target_port == nullptr) {
    return partition_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
  }
  const auto source = storage.executable_index.find(source_port->owner.value);
  const auto target = storage.executable_index.find(target_port->owner.value);
  if (source == storage.executable_index.end() || target == storage.executable_index.end() ||
      storage.edges.size() >= std::numeric_limits<EdgeIndex>::max()) {
    return partition_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
  }

  PartitionBoundaryMask boundaries;
  const VertexRecord* const source_vertex = find_vertex(snapshot, source_port->owner);
  const VertexRecord* const target_vertex = find_vertex(snapshot, target_port->owner);
  if (source_vertex == nullptr || target_vertex == nullptr) {
    return partition_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
  }
  if (source_vertex->kind != VertexKind::kOperator ||
      target_vertex->kind != VertexKind::kOperator) {
    boundaries.add(PartitionBoundary::kExternalEndpoint);
  }
  if (source_vertex->id.value > process_group_by_vertex.size() ||
      target_vertex->id.value > process_group_by_vertex.size()) {
    return partition_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
  }
  if (process_group_by_vertex[static_cast<std::size_t>(source_vertex->id.value - 1U)] !=
      process_group_by_vertex[static_cast<std::size_t>(target_vertex->id.value - 1U)]) {
    boundaries.add(PartitionBoundary::kProcess);
  }
  const auto edge_index = static_cast<EdgeIndex>(storage.edges.size());
  storage.outgoing[source->second].push_back(edge_index);
  storage.incoming[target->second].push_back(edge_index);
  storage.edges.push_back(PartitionEdgeView{
      edge.id,
      source->second,
      target->second,
      boundaries,
      boundaries.empty(),
      true,
  });
  return {};
}

[[nodiscard]] expected<void, Error> append_problem_edges(
    const GraphSnapshotData& snapshot, std::span<const std::uint32_t> process_group_by_vertex,
    PartitionProblemStorage& storage) {
  for (const auto& edge : snapshot.edges) {
    auto appended = append_problem_edge(snapshot, edge, process_group_by_vertex, storage);
    if (!appended) {
      return make_unexpected(std::move(appended.error()));
    }
  }
  return {};
}

void populate_vertex_views(const GraphSnapshotData& snapshot,
                           std::span<const std::uint32_t> process_group_by_vertex,
                           PartitionProblemStorage& storage) {
  using DomainKey = std::tuple<std::uint32_t, std::uint64_t>;
  std::map<DomainKey, std::uint64_t> domains;
  for (const auto& vertex : snapshot.vertices) {
    if (!executable(vertex.kind)) {
      continue;
    }
    const bool external = vertex.kind != VertexKind::kOperator;
    const std::uint64_t isolated_vertex = external ? vertex.id.value : 0U;
    domains.try_emplace(DomainKey{
        process_group_by_vertex[static_cast<std::size_t>(vertex.id.value - 1U)], isolated_vertex});
  }
  std::uint64_t next_domain = 1U;
  for (auto& [unused, domain] : domains) {
    (void)unused;
    domain = next_domain++;
  }
  for (const auto& vertex : snapshot.vertices) {
    const auto found = storage.executable_index.find(vertex.id.value);
    if (found == storage.executable_index.end()) {
      continue;
    }
    const VertexIndex index = found->second;
    const bool external = vertex.kind != VertexKind::kOperator;
    PartitionBoundaryMask boundary_before;
    if (external) {
      boundary_before.add(PartitionBoundary::kExternalEndpoint);
    }
    const std::uint64_t isolated_vertex = external ? vertex.id.value : 0U;
    const DomainKey domain_key{
        process_group_by_vertex[static_cast<std::size_t>(vertex.id.value - 1U)], isolated_vertex};
    const std::uint64_t execution_domain = domains.at(domain_key);
    storage.vertices[index] = PartitionVertexView{
        vertex.id,
        index,
        storage.incoming[index],
        storage.outgoing[index],
        storage.incoming[index].empty(),
        ExecutionDomainKey{execution_domain},
        boundary_before,
        storage.source_regions[index],
    };
  }
}

[[nodiscard]] expected<PartitionProblemStorage, Error> make_problem_storage(
    const GraphSnapshotData& snapshot, std::span<const std::uint32_t> process_group_by_vertex) {
  PartitionProblemStorage storage;
  if (process_group_by_vertex.size() != snapshot.vertices.size()) {
    return partition_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
  }
  auto vertices = append_executable_vertices(snapshot, storage);
  if (!vertices) {
    return make_unexpected(std::move(vertices.error()));
  }
  auto edges = append_problem_edges(snapshot, process_group_by_vertex, storage);
  if (!edges) {
    return make_unexpected(std::move(edges.error()));
  }
  populate_vertex_views(snapshot, process_group_by_vertex, storage);
  return storage;
}

[[nodiscard]] PlanDigest digest_partition_config(
    Profile profile, std::span<const std::optional<Profile>> effective_profiles,
    std::optional<std::string_view> environment,
    std::span<const PartitionRecord> authored_partitions) {
  StableHashBuilder hash;
  constexpr std::string_view domain = "holoscan.partition.config.v1";
  for (const char character : domain) {
    hash.add_byte(static_cast<std::uint8_t>(character));
  }
  hash.add(static_cast<std::uint8_t>(profile));
  hash.add(environment.has_value());
  if (environment) {
    for (const char character : *environment) {
      hash.add_byte(static_cast<std::uint8_t>(character));
    }
  }
  hash.add(static_cast<std::uint64_t>(effective_profiles.size()));
  for (const auto effective : effective_profiles) {
    hash.add(effective.has_value());
    if (effective) {
      hash.add(static_cast<std::uint8_t>(*effective));
    }
  }
  hash.add(static_cast<std::uint64_t>(authored_partitions.size()));
  for (const auto& partition : authored_partitions) {
    hash.add(partition.id.value);
    hash.add(partition.graph.value);
    hash.add(static_cast<std::uint64_t>(partition.name.size()));
    for (const char character : partition.name) {
      hash.add_byte(static_cast<std::uint8_t>(character));
    }
    hash.add(static_cast<std::uint64_t>(partition.members.size()));
    for (const VertexId member : partition.members) {
      hash.add(member.value);
    }
  }
  return PlanDigest{hash.finish().value};
}

void initialize_partition_attempt(Profile root_profile,
                                  std::span<const std::optional<Profile>> effective_profiles,
                                  std::optional<std::string_view> environment,
                                  std::span<const PartitionRecord> authored_partitions,
                                  ExecutionPlanState& output) {
  output.partitioning_plan = PartitioningPlanRecord{
      .requested_algorithm = partitioning::serial_baseline_algorithm(),
      .accepted_algorithm = std::nullopt,
      .interface_version = partitioning::kPartitionPlannerInterfaceVersion,
      .maturity = AlgorithmMaturity::kProduction,
      .disposition = PlanDisposition::kRejected,
      .selection_source = PlanSelectionSource::kCoreDefault,
      .config_digest = digest_partition_config(
          root_profile, effective_profiles, environment, authored_partitions),
      .reason = {},
  };
  if (!environment) {
    return;
  }

  output.partitioning_plan.selection_source = PlanSelectionSource::kEnvironmentOverride;
  if (*environment == "serial_baseline/v1") {
    return;
  }
  if (*environment == "source_chain/v1") {
    output.partitioning_plan.requested_algorithm = partitioning::source_chain_algorithm();
    output.partitioning_plan.maturity = AlgorithmMaturity::kExperimental;
    return;
  }
  // Invalid syntax has provenance but no fabricated requested algorithm.
  output.partitioning_plan.requested_algorithm.reset();
  output.partitioning_plan.maturity.reset();
}

[[nodiscard]] unexpected<Error> publish_partition_failure(Error error, ExecutionPlanState& output) {
  output.partitioning_plan.accepted_algorithm.reset();
  output.partitioning_plan.disposition = PlanDisposition::kRejected;
  output.partitioning_plan.reason = diagnostic_id(error.message);
  return make_unexpected(std::move(error));
}

void append_schedule_edges(const GraphSnapshotData& snapshot, ExecutionPlanState& output) {
  std::unordered_map<std::uint64_t, ExecutionPartitionId> partition_by_vertex;
  for (const auto& partition : output.execution_partitions) {
    for (const VertexId vertex : partition.members) {
      partition_by_vertex.emplace(vertex.value, partition.id);
    }
  }
  for (const auto& edge : snapshot.edges) {
    const PortRecord* const source =
        find_port(snapshot, resolve_interface_port(snapshot, edge.source));
    const PortRecord* const target =
        find_port(snapshot, resolve_interface_port(snapshot, edge.target));
    if (source == nullptr || target == nullptr) {
      continue;
    }
    const auto source_partition = partition_by_vertex.find(source->owner.value);
    const auto target_partition = partition_by_vertex.find(target->owner.value);
    if (source_partition != partition_by_vertex.end() &&
        target_partition != partition_by_vertex.end() &&
        source_partition->second != target_partition->second) {
      output.schedule_edges.push_back(
          ScheduleEdgeRecord{edge.id, source_partition->second, target_partition->second});
    }
  }
}

[[nodiscard]] bool has_independent_temporal_source(const ExecutionPlanState& plan, VertexId owner) {
  const auto executable =
      std::find_if(plan.executable_factories.operators.begin(),
                   plan.executable_factories.operators.end(),
                   [&](const auto& candidate) { return candidate.vertex == owner; });
  if (executable == plan.executable_factories.operators.end()) {
    return false;
  }
  return std::any_of(
      plan.temporal_triggers.begin(), plan.temporal_triggers.end(), [&](const auto& trigger) {
        if (trigger.owner.id != executable->operator_id) {
          return false;
        }
        switch (trigger.kind) {
          case TemporalRecordTriggerKind::kClock:
          case TemporalRecordTriggerKind::kReadable:
          case TemporalRecordTriggerKind::kNotified:
          case TemporalRecordTriggerKind::kDoorbell:
          case TemporalRecordTriggerKind::kSpin:
            return true;
          case TemporalRecordTriggerKind::kPredicate:
            return !trigger.trigger_input && trigger.trigger_clock.valid();
          case TemporalRecordTriggerKind::kAnyInput:
          case TemporalRecordTriggerKind::kEach:
          case TemporalRecordTriggerKind::kAligned:
          case TemporalRecordTriggerKind::kWindow:
            return false;
        }
        return false;
      });
}

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

[[nodiscard]] expected<void, Error> validate_manual_operator_cover(
    const GraphSnapshotData& snapshot, const PartitionProblemStorage& storage,
    std::vector<std::optional<std::size_t>>& partition_by_vertex) {
  if (snapshot.graphs.size() != 1U ||
      std::any_of(snapshot.graphs.begin(), snapshot.graphs.end(), [](const auto& graph) {
        return graph.parent.has_value();
      })) {
    return partition_error(ErrorCode::kNotSupported, "PARTITION_MANUAL_EMBEDDED_GRAPH_UNSUPPORTED");
  }
  const GraphId root = snapshot.graphs.front().id;
  partition_by_vertex.resize(snapshot.vertices.size());
  for (std::size_t partition_index = 0U; partition_index < snapshot.partitions.size();
       ++partition_index) {
    const auto& partition = snapshot.partitions[partition_index];
    if (partition.graph != root || partition.members.empty()) {
      return partition_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    for (const VertexId member : partition.members) {
      if (!member || member.value > snapshot.vertices.size()) {
        return partition_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      const VertexRecord* const vertex = find_vertex(snapshot, member);
      auto& assigned = partition_by_vertex[static_cast<std::size_t>(member.value - 1U)];
      if (vertex == nullptr || vertex->kind != VertexKind::kOperator || vertex->graph != root ||
          assigned.has_value()) {
        return partition_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      assigned = partition_index;
    }
  }
  for (const auto& vertex : snapshot.vertices) {
    if (vertex.kind == VertexKind::kOperator &&
        (!vertex.id || vertex.id.value > partition_by_vertex.size() ||
         !partition_by_vertex[static_cast<std::size_t>(vertex.id.value - 1U)].has_value())) {
      return partition_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
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
    if (!executable(vertex.kind) || vertex.kind == VertexKind::kOperator) {
      continue;
    }
    const auto found = storage.executable_index.find(vertex.id.value);
    if (found == storage.executable_index.end()) {
      return partition_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
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
        return partition_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      append_neighbor(storage.edges[edge].source);
    }
    for (const EdgeIndex edge : view.outgoing_edges) {
      if (edge >= storage.edges.size()) {
        return partition_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      append_neighbor(storage.edges[edge].target);
    }
    std::sort(adjacent_partitions.begin(), adjacent_partitions.end());
    adjacent_partitions.erase(std::unique(adjacent_partitions.begin(), adjacent_partitions.end()),
                              adjacent_partitions.end());
    if (adjacent_partitions.size() != 1U) {
      return partition_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    const std::size_t selected = adjacent_partitions.front();
    if (!vertex.id || vertex.id.value > partition_by_vertex.size() ||
        selected >= members_by_partition.size()) {
      return partition_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    partition_by_vertex[static_cast<std::size_t>(vertex.id.value - 1U)] = selected;
    members_by_partition[selected].push_back(vertex.id);
  }
  return {};
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> publish_manual_partition_cover(
    const GraphSnapshotData& snapshot, const PartitionProblemStorage& storage,
    std::span<const std::uint32_t> lane_by_vertex,
    std::span<const std::uint32_t> process_group_by_vertex, ExecutionPlanState& output) {
  std::vector<std::optional<std::size_t>> partition_by_vertex;
  auto cover = validate_manual_operator_cover(snapshot, storage, partition_by_vertex);
  if (!cover) {
    return make_unexpected(std::move(cover.error()));
  }

  std::vector<std::vector<VertexId>> members_by_partition(snapshot.partitions.size());
  for (std::size_t index = 0U; index < snapshot.partitions.size(); ++index) {
    members_by_partition[index] = snapshot.partitions[index].members;
    std::optional<std::uint32_t> process_group;
    for (const VertexId member : members_by_partition[index]) {
      if (!member || member.value > process_group_by_vertex.size()) {
        return partition_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      const std::uint32_t candidate =
          process_group_by_vertex[static_cast<std::size_t>(member.value - 1U)];
      if (candidate == 0U || (process_group.has_value() && *process_group != candidate)) {
        return partition_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      process_group = candidate;
    }
  }
  auto synthetics =
      attach_synthetic_vertices(snapshot, storage, partition_by_vertex, members_by_partition);
  if (!synthetics) {
    return make_unexpected(std::move(synthetics.error()));
  }
  // Keep this output equivalent to an accepted algorithmic-plan result: the
  // cover is disjoint and complete, member order is canonical ascending
  // VertexId order, and every execution partition receives one lane fact.
  // Current public compilation supplies zero lane placeholders here; deployment
  // lowers each authored partition to its final lane afterward. Retaining the
  // equality check below makes a future cold caller that supplies preassigned
  // lanes fail closed when one manual partition spans incompatible lanes.
  output.partitioning_plan.accepted_algorithm = partitioning::serial_baseline_algorithm();
  output.partitioning_plan.maturity = AlgorithmMaturity::kProduction;
  output.partitioning_plan.disposition = PlanDisposition::kAccepted;
  output.partitioning_plan.selection_source = PlanSelectionSource::kCoreDefault;
  output.partitioning_plan.reason = {};
  output.execution_partitions.reserve(snapshot.partitions.size());
  for (std::size_t index = 0U; index < snapshot.partitions.size(); ++index) {
    auto& members = members_by_partition[index];
    std::sort(members.begin(), members.end(), [](VertexId lhs, VertexId rhs) {
      return lhs.value < rhs.value;
    });
    std::optional<std::uint32_t> lane;
    for (const VertexId member : members) {
      if (!member || member.value > lane_by_vertex.size()) {
        return partition_error(ErrorCode::kInvalidArgument,
                               "PARTITION_RESOURCE_REALIZATION_FAILED");
      }
      const std::uint32_t candidate = lane_by_vertex[static_cast<std::size_t>(member.value - 1U)];
      if (lane.value_or(candidate) != candidate) {
        return partition_error(ErrorCode::kIncompatible, "PARTITION_RESOURCE_REALIZATION_FAILED");
      }
      lane = candidate;
    }
    const ExecutionPartitionId id{static_cast<std::uint32_t>(index + 1U)};
    output.execution_partitions.push_back(ExecutionPartitionRecord{
        .id = id,
        .authored_name = snapshot.partitions[index].name,
        .members = members,
        .execution_lane = lane.value_or(0U),
    });
    for (std::size_t member_index = 0U; member_index < members.size(); ++member_index) {
      output.partition_decisions.push_back(PartitionDecisionRecord{
          .vertex = members[member_index],
          .partition = id,
          .decision =
              member_index == 0U ? PartitionDecisionKind::kStart : PartitionDecisionKind::kJoin,
          .reasons = {},
      });
    }
  }
  std::sort(output.partition_decisions.begin(),
            output.partition_decisions.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.vertex.value < rhs.vertex.value; });
  append_schedule_edges(snapshot, output);
  return {};
}

[[nodiscard]] expected<void, Error> publish_validated_plan(
    const partitioning::PartitioningPassResult& result, const GraphSnapshotData& snapshot,
    std::span<const std::uint32_t> lane_by_vertex, ExecutionPlanState& output) {
  output.partitioning_plan = PartitioningPlanRecord{
      .requested_algorithm = result.requested_algorithm,
      .accepted_algorithm = result.plan.algorithm,
      .interface_version = partitioning::kPartitionPlannerInterfaceVersion,
      .maturity = result.maturity,
      .disposition = PlanDisposition::kAccepted,
      .selection_source = result.selection_source,
      .config_digest = {},
      .reason = {},
  };

  output.execution_partitions.reserve(result.plan.partitions.size());
  for (const auto& partition : result.plan.partitions) {
    std::optional<std::uint32_t> lane;
    for (const VertexId member : partition.members) {
      if (!member || member.value > lane_by_vertex.size()) {
        return partition_error(ErrorCode::kInvalidArgument,
                               "PARTITION_RESOURCE_REALIZATION_FAILED");
      }
      const std::uint32_t candidate = lane_by_vertex[static_cast<std::size_t>(member.value - 1U)];
      if (lane && *lane != candidate) {
        return partition_error(ErrorCode::kIncompatible, "PARTITION_RESOURCE_REALIZATION_FAILED");
      }
      lane = candidate;
    }
    output.execution_partitions.push_back(ExecutionPartitionRecord{
        .id = ExecutionPartitionId{partition.canonical_id},
        .members = partition.members,
        .execution_lane = lane.value_or(0U),
    });
  }
  output.partition_decisions.reserve(result.plan.decisions.size());
  for (const auto& decision : result.plan.decisions) {
    output.partition_decisions.push_back(PartitionDecisionRecord{
        .vertex = decision.vertex,
        .partition = ExecutionPartitionId{decision.canonical_partition_id},
        .decision = decision.decision,
        .reasons = decision.reasons,
    });
  }

  append_schedule_edges(snapshot, output);
  return {};
}

}  // namespace

expected<void, Error> lower_partition_plan(
    const GraphSnapshotData& snapshot, Profile root_profile,
    std::span<const std::optional<Profile>> effective_profiles,
    std::optional<std::string_view> captured_environment_value,
    std::span<const std::uint32_t> lane_by_vertex,
    std::span<const std::uint32_t> process_group_by_vertex, ExecutionPlanState& output) {
  // REQ: COMPILE-004, COMPILE-005, COMPILE-006
  // VertexId is a dense one-based index in a canonical snapshot. Both the
  // ordinary compile path and parameter rebind therefore provide one lane
  // placeholder per snapshot vertex before this lowering path indexes it.
  if (lane_by_vertex.size() != snapshot.vertices.size()) {
    return publish_partition_failure(
        Error{ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID"}, output);
  }
  initialize_partition_attempt(
      root_profile, effective_profiles, captured_environment_value, snapshot.partitions, output);
  const bool manual_cover = !snapshot.partitions.empty();
  if (manual_cover && captured_environment_value.has_value()) {
    return publish_partition_failure(
        Error{ErrorCode::kInvalidArgument, "PARTITION_SELECTION_CONFLICT"}, output);
  }
  // Implicit/manual partitioning is finalized after temporal replay. Declaration-only
  // readiness sources must remain inert here; otherwise an unused source silently manufactures a
  // partition boundary before the compiler knows which source the Operator contract selected.
  auto storage_result = make_problem_storage(snapshot, process_group_by_vertex);
  if (!storage_result) {
    return publish_partition_failure(std::move(storage_result).error(), output);
  }
  auto storage = std::move(storage_result).value();
  const PartitionProblem problem{
      storage.vertices,
      storage.edges,
      static_cast<std::uint32_t>(storage.vertices.size()),
      partitioning::kAllPartitionPlannerCapabilities,
  };

  if (manual_cover) {
    auto published = publish_manual_partition_cover(
        snapshot, storage, lane_by_vertex, process_group_by_vertex, output);
    if (!published) {
      return publish_partition_failure(std::move(published).error(), output);
    }
  } else {
    auto catalog = partitioning::PartitionPlannerCatalog::create_ea1_builtins();
    if (!catalog) {
      return publish_partition_failure(std::move(catalog).error(), output);
    }
    const partitioning::PartitioningPass pass{catalog.value()};
    auto result = pass.run(
        partitioning::PartitionSelectionInput{
            .captured_environment_value = captured_environment_value,
            .compiler_service_request = std::nullopt,
            .test_override_request = std::nullopt,
            .root_profile = root_profile,
            .effective_profiles = effective_profiles,
            .effective_regions = output.profile_manifest.effective_regions,
        },
        problem);
    if (!result) {
      return publish_partition_failure(std::move(result).error(), output);
    }

    auto published = publish_validated_plan(result.value(), snapshot, lane_by_vertex, output);
    if (!published) {
      return publish_partition_failure(std::move(published).error(), output);
    }
  }
  output.partitioning_plan.config_digest = digest_partition_config(
      root_profile, effective_profiles, captured_environment_value, snapshot.partitions);
  const auto max_lane = std::max_element(
      output.execution_partitions.begin(),
      output.execution_partitions.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.execution_lane < rhs.execution_lane; });
  const std::uint32_t lane_count =
      max_lane == output.execution_partitions.end() ? 0U : max_lane->execution_lane + 1U;
  output.scheduler_plan = SchedulerPlanRecord{
      .ready_policy = AlgorithmKey{"holoscan.ready.deterministic_serial", 1U},
      .execution_lane_count = lane_count,
      .max_concurrent_activations = output.execution_partitions.empty() ? 0U : 1U,
      .activation_overlap = false,
      .worker_count = lane_count,
      .intra_path_parallelism = output.execution_partitions.empty() ? 0U : 1U,
      .reason = {},
  };
  return {};
}

expected<void, Error> validate_partition_execution_roots(const GraphSnapshotData& snapshot,
                                                         ExecutionPlanState& output) {
  if (output.partitioning_plan.disposition != PlanDisposition::kAccepted) {
    return {};
  }

  std::vector<std::optional<std::size_t>> partition_by_vertex(snapshot.vertices.size());
  for (std::size_t partition_index = 0U; partition_index < output.execution_partitions.size();
       ++partition_index) {
    for (const VertexId member : output.execution_partitions[partition_index].members) {
      if (!member || member.value > partition_by_vertex.size()) {
        return publish_partition_failure(
            Error{ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID"}, output);
      }
      const std::size_t member_index = static_cast<std::size_t>(member.value - 1U);
      if (partition_by_vertex[member_index].has_value()) {
        // A corrupted/private plan must never silently reinterpret a vertex as
        // belonging to whichever duplicate partition happened to be visited last.
        return publish_partition_failure(
            Error{ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID"}, output);
      }
      partition_by_vertex[member_index] = partition_index;
    }
  }

  if (snapshot.partitions.empty()) {
    if (independent_execution_root_count(snapshot, output, std::nullopt, partition_by_vertex) >
        1U) {
      return publish_partition_failure(
          Error{ErrorCode::kInvalidArgument, "PARTITION_MANUAL_AUTHORING_REQUIRED"}, output);
    }
    return {};
  }

  for (std::size_t partition_index = 0U; partition_index < output.execution_partitions.size();
       ++partition_index) {
    if (independent_execution_root_count(snapshot, output, partition_index, partition_by_vertex) >
        1U) {
      return publish_partition_failure(
          Error{ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID"}, output);
    }
  }
  return {};
}

}  // namespace holoscan::detail
