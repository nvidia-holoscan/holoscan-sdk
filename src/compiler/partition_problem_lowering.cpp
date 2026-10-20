/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "partition_problem_lowering.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "authored_partition_membership.hpp"
#include "runtime_package_capabilities.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

using partitioning::EdgeIndex;
using partitioning::ExecutionDomainKey;
using partitioning::PartitionBoundary;
using partitioning::PartitionBoundaryMask;
using partitioning::PartitionEdgeView;
using partitioning::PartitionVertexView;
using partitioning::VertexIndex;

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
    if (!partition_executable(vertex.kind)) {
      continue;
    }
    if (storage.vertices.size() >= std::numeric_limits<VertexIndex>::max()) {
      return partition_lowering_error(ErrorCode::kResourceExhausted, "PARTITION_LIMIT_EXCEEDED");
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
    return partition_lowering_error(ErrorCode::kInvalidArgument,
                                    "PARTITION_PROPOSAL_CONTRACT_INVALID");
  }
  const auto source = storage.executable_index.find(source_port->owner.value);
  const auto target = storage.executable_index.find(target_port->owner.value);
  if (source == storage.executable_index.end() || target == storage.executable_index.end() ||
      storage.edges.size() >= std::numeric_limits<EdgeIndex>::max()) {
    return partition_lowering_error(ErrorCode::kInvalidArgument,
                                    "PARTITION_PROPOSAL_CONTRACT_INVALID");
  }

  PartitionBoundaryMask boundaries;
  const VertexRecord* const source_vertex = find_vertex(snapshot, source_port->owner);
  const VertexRecord* const target_vertex = find_vertex(snapshot, target_port->owner);
  if (source_vertex == nullptr || target_vertex == nullptr) {
    return partition_lowering_error(ErrorCode::kInvalidArgument,
                                    "PARTITION_PROPOSAL_CONTRACT_INVALID");
  }
  if (source_vertex->kind != VertexKind::kOperator ||
      target_vertex->kind != VertexKind::kOperator) {
    boundaries.add(PartitionBoundary::kExternalEndpoint);
  }
  if (source_vertex->id.value > process_group_by_vertex.size() ||
      target_vertex->id.value > process_group_by_vertex.size()) {
    return partition_lowering_error(ErrorCode::kInvalidArgument,
                                    "PARTITION_PROPOSAL_CONTRACT_INVALID");
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

// Normalize every authored and inferred device obligation before partition proposal construction.
[[nodiscard]] expected<std::vector<std::vector<std::int32_t>>, Error>
normalized_device_obligations_by_vertex(  // NOLINT(readability-function-cognitive-complexity)
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    std::span<const TensorDeviceBindingRecord> tensor_device_bindings) {
  std::vector<std::vector<std::int32_t>> devices(snapshot.vertices.size());
  const auto assign = [&](VertexId owner, DeviceId device) {
    if (!owner || owner.value > devices.size() || !device.valid()) {
      return;
    }
    devices[static_cast<std::size_t>(owner.value - 1U)].push_back(device.value);
  };

  // Index resolved concrete consumers once. A bound nested output can appear on a parent edge
  // through one or more public Graph-interface aliases; both endpoints are resolved before the
  // exact output obligation is propagated to its consumer Operator.
  std::vector<std::vector<VertexId>> consumers_by_output(snapshot.ports.size());
  for (const auto& edge : snapshot.edges) {
    const GraphPortId source_id = resolve_interface_port(snapshot, edge.source);
    const GraphPortId target_id = resolve_interface_port(snapshot, edge.target);
    const PortRecord* const source = find_port(snapshot, source_id);
    const PortRecord* const target = find_port(snapshot, target_id);
    if (source == nullptr || target == nullptr || !source_id ||
        source_id.value > consumers_by_output.size()) {
      return partition_lowering_error(ErrorCode::kInvalidArgument,
                                      "PARTITION_PROPOSAL_CONTRACT_INVALID");
    }
    consumers_by_output[static_cast<std::size_t>(source_id.value - 1U)].push_back(target->owner);
  }
  for (const auto& binding : tensor_device_bindings) {
    const PortRecord* const port = find_port(snapshot, GraphPortId{binding.port.value()});
    const bool tensor_bearing =
        port != nullptr &&
        (port->payload_identity.kind == PayloadIdentityKind::kTensor ||
         (port->payload_identity.kind == PayloadIdentityKind::kSchema &&
          port->payload_identity.payload_kind == PayloadKind::kDescriptorBacked));
    if (port == nullptr || port->owner_kind != PortOwnerKind::kOperator ||
        port->direction != PortDirection::kOutput || !binding.device.valid() || !tensor_bearing) {
      continue;
    }
    assign(port->owner, binding.device);
    for (const VertexId consumer :
         consumers_by_output[static_cast<std::size_t>(port->id.value - 1U)]) {
      assign(consumer, binding.device);
    }
  }
  std::unordered_map<std::string_view, const PartitionRecord*> partitions_by_name;
  partitions_by_name.reserve(snapshot.partitions.size());
  for (const auto& partition : snapshot.partitions) {
    partitions_by_name.emplace(partition.name, &partition);
  }
  for (const auto& binding : deployment.cuda_green_contexts()) {
    const auto partition = partitions_by_name.find(binding.partition_name);
    if (partition == partitions_by_name.end()) {
      continue;
    }
    const auto members = resolve_authored_partition_members(snapshot, *partition->second);
    // Membership errors are reported by cover/anchor validation before any plan is accepted.
    if (!members)
      continue;
    for (const VertexId member : *members) {
      assign(member, binding.placement.device);
    }
  }

  for (auto& obligations : devices) {
    std::sort(obligations.begin(), obligations.end());
    obligations.erase(std::unique(obligations.begin(), obligations.end()), obligations.end());
  }
  return devices;
}

[[nodiscard]] bool compatible_device_obligations(std::span<const std::int32_t> left,
                                                 std::span<const std::int32_t> right) noexcept {
  // An empty set carries no device constraint. A nonempty set is compatible only when both
  // vertices name the same one exact device; a multi-device set remains an explicit conflict.
  if (left.size() > 1U || right.size() > 1U) {
    return false;
  }
  return left.empty() || right.empty() || left.front() == right.front();
}

void populate_vertex_views(const GraphSnapshotData& snapshot,
                           std::span<const std::uint32_t> process_group_by_vertex,
                           const ExecutionPlanState& plan, PartitionProblemStorage& storage) {
  using DomainKey = std::pair<std::uint32_t, std::uint64_t>;
  using SerializationKey = std::pair<std::uint32_t, std::string_view>;
  std::map<DomainKey, std::uint64_t> execution_domains;
  std::map<SerializationKey, std::uint64_t> serialization_domains;
  for (const auto& vertex : snapshot.vertices) {
    if (!partition_executable(vertex.kind)) {
      continue;
    }
    const bool external = vertex.kind != VertexKind::kOperator;
    const std::uint64_t isolated_vertex = external ? vertex.id.value : 0U;
    const std::uint32_t process_group =
        process_group_by_vertex[static_cast<std::size_t>(vertex.id.value - 1U)];
    execution_domains.try_emplace(DomainKey{process_group, isolated_vertex});
    if (!vertex.serialization_domain.empty()) {
      serialization_domains.try_emplace(
          SerializationKey{process_group, vertex.serialization_domain});
    }
  }
  std::uint64_t next_domain = 1U;
  for (auto& [unused, domain] : execution_domains) {
    (void)unused;
    domain = next_domain++;
  }
  next_domain = 1U;
  for (auto& [unused, domain] : serialization_domains) {
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
    const bool independent_temporal_source = has_independent_temporal_source(plan, vertex.id);
    if (independent_temporal_source && !storage.incoming[index].empty()) {
      boundary_before.add(PartitionBoundary::kReadiness);
    }
    const std::uint64_t isolated_vertex = external ? vertex.id.value : 0U;
    const std::uint32_t process_group =
        process_group_by_vertex[static_cast<std::size_t>(vertex.id.value - 1U)];
    const DomainKey domain_key{process_group, isolated_vertex};
    const std::uint64_t execution_domain = execution_domains.at(domain_key);
    const std::uint64_t serialization_domain =
        vertex.serialization_domain.empty() ? 0U
                                            : serialization_domains.at(SerializationKey{
                                                  process_group, vertex.serialization_domain});
    storage.vertices[index] = PartitionVertexView{
        vertex.id,
        index,
        storage.incoming[index],
        storage.outgoing[index],
        storage.incoming[index].empty() || independent_temporal_source,
        ExecutionDomainKey{execution_domain},
        partitioning::SerializationDomainKey{serialization_domain},
        storage.device_obligations[static_cast<std::size_t>(vertex.id.value - 1U)],
        boundary_before,
        storage.source_regions[index],
    };
  }
  for (auto& edge : storage.edges) {
    if (!compatible_device_obligations(storage.vertices[edge.source].device_obligations,
                                       storage.vertices[edge.target].device_obligations)) {
      edge.mandatory_boundaries.add(PartitionBoundary::kDeviceTransfer);
      edge.same_partition_collapse_legal = false;
    }
  }
}

}  // namespace

expected<PartitionProblemStorage, Error> make_partition_problem_storage(
    const GraphSnapshotData& snapshot, std::span<const std::uint32_t> process_group_by_vertex,
    const DeploymentBinding& deployment,
    std::span<const TensorDeviceBindingRecord> tensor_device_bindings,
    const ExecutionPlanState& plan) {
  PartitionProblemStorage storage;
  if (process_group_by_vertex.size() != snapshot.vertices.size()) {
    return partition_lowering_error(ErrorCode::kInvalidArgument,
                                    "PARTITION_PROPOSAL_CONTRACT_INVALID");
  }
  auto vertices = append_executable_vertices(snapshot, storage);
  if (!vertices) {
    return make_unexpected(std::move(vertices.error()));
  }
  auto edges = append_problem_edges(snapshot, process_group_by_vertex, storage);
  if (!edges) {
    return make_unexpected(std::move(edges.error()));
  }
  auto devices =
      normalized_device_obligations_by_vertex(snapshot, deployment, tensor_device_bindings);
  if (!devices) {
    return make_unexpected(std::move(devices.error()));
  }
  storage.device_obligations = std::move(devices).value();
  populate_vertex_views(snapshot, process_group_by_vertex, plan, storage);
  return storage;
}

expected<void, Error> append_authored_partition_constraints(const GraphSnapshotData& snapshot,
                                                            PartitionProblemStorage& storage) {
  const GraphRecord* const root = find_root_graph(snapshot);
  if (root == nullptr || snapshot.partitions.empty()) {
    return partition_lowering_error(ErrorCode::kInvalidArgument,
                                    "PARTITION_PROPOSAL_CONTRACT_INVALID");
  }
  storage.authored_constraint_members.reserve(snapshot.partitions.size());
  for (const auto& partition : snapshot.partitions) {
    if (partition.include_all) {
      return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    auto resolved = resolve_authored_partition_members(snapshot, partition);
    if (!resolved)
      return make_unexpected(std::move(resolved.error()));
    std::vector<VertexIndex> members;
    members.reserve(resolved->size());
    for (const VertexId member : *resolved) {
      const auto found = storage.executable_index.find(member.value);
      if (found == storage.executable_index.end()) {
        return partition_lowering_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
      }
      members.push_back(found->second);
    }
    storage.authored_constraint_members.push_back(std::move(members));
  }
  storage.authored_constraints.reserve(snapshot.partitions.size());
  for (std::size_t index = 0U; index < snapshot.partitions.size(); ++index) {
    if (!snapshot.partitions[index].id ||
        snapshot.partitions[index].id.value > std::numeric_limits<std::uint32_t>::max()) {
      return partition_lowering_error(ErrorCode::kInvalidArgument,
                                      "PARTITION_PROPOSAL_CONTRACT_INVALID");
    }
    storage.authored_constraints.push_back(partitioning::AuthoredPartitionConstraintView{
        partitioning::AuthoredPartitionConstraintId{
            static_cast<std::uint32_t>(snapshot.partitions[index].id.value)},
        storage.authored_constraint_members[index],
    });
  }
  return {};
}

bool has_independent_temporal_source(const ExecutionPlanState& plan, VertexId owner) {
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

}  // namespace holoscan::detail
