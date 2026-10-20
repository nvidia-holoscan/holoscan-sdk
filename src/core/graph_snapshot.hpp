/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/core/graph.hpp>

namespace holoscan::detail {

/// Identify how one pointer-free callback declaration enters the lifecycle plan.
enum class LifecycleCallbackOrigin : std::uint8_t {
  kAdvancedMember,
  kLegacyStart,
  kLegacyStop,
};

/// Describe one sparse lifecycle callback without retaining a receiver or function address.
struct LifecycleCallbackAuthoringRecord {
  VertexId owner{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  bool required{};
  LifecycleMethodId method{};
  LifecycleCallbackOrigin origin{LifecycleCallbackOrigin::kAdvancedMember};
};

enum class LifecycleFactKind : std::uint8_t {
  kProduces,
  kRequires,
};

/// Describe one deep-owned Operator-local lifecycle ordering fact.
struct LifecycleFactAuthoringRecord {
  VertexId owner{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  LifecycleFactKind kind{LifecycleFactKind::kProduces};
  std::string stable_name;
};

/// Preserve compiler-visible capability evidence for one cold clock declaration.
///
/// These facts are derived once from the concrete clock instance validated by
/// Graph authoring. They grant no live-clock authority and remain private to
/// snapshot/plan lowering.
struct ClockCapabilityAuthoringRecord {
  GraphId graph{};
  ClockKey key;
  bool can_read_now{};
  bool can_wait_until{};
  bool steady{};
  bool independent_progress{};
  bool finite_deadline{};
  bool external_driver_admitted{};
  std::string evidence_id;
};

/// Deep-owned immutable data behind one GraphView.
struct GraphSnapshotData {
  GraphGeneration generation{};
  GraphAuthoringIdentity identity{};
  std::vector<GraphRecord> graphs;
  std::vector<VertexRecord> vertices;
  std::vector<PortRecord> ports;
  std::vector<EdgeRecord> edges;
  std::vector<InterfaceBindingRecord> interface_bindings;
  std::vector<ExternalTopicRecord> external_topics;
  std::vector<ReadinessSourceDeclarationRecord> readiness_sources;
  std::vector<RegionRecord> regions;
  std::vector<PartitionRecord> partitions;
  bool automatic_partition_remaining{};
  std::vector<ClockDeclarationRecord> clocks;
  std::vector<ClockCapabilityAuthoringRecord> clock_capabilities;
  std::vector<ClockImportAuthoringRecord> clock_imports;
  std::vector<EmbeddedClockBindingAuthoringRecord> clock_bindings;
  std::vector<ClockConversionAuthoringRecord> clock_conversions;
  std::vector<LifecycleDependencyAuthoringRecord> lifecycle_dependencies;
  std::vector<LifecycleCallbackAuthoringRecord> lifecycle_callbacks;
  std::vector<LifecycleFactAuthoringRecord> lifecycle_facts;
  std::vector<LifecycleRestartPermissionAuthoringRecord> lifecycle_restart_permissions;
};

/// Return a pointer-free compiler handoff whose local identities are derived
/// from canonical instance paths and declaration facts rather than authoring
/// insertion order. Public authoring handles continue to use the live state's
/// IDs; this normalized copy is the sole input intended for plan lowering.
[[nodiscard]] inline GraphSnapshotData
canonicalize_graph_snapshot(  // NOLINT(readability-function-cognitive-complexity)
    GraphSnapshotData snapshot) {
  using IdMap = std::unordered_map<std::uint64_t, std::uint64_t>;
  const auto graph_path = [](const GraphRecord& graph) {
    return graph.path ? std::vector<std::string>{graph.path->segments().begin(),
                                                 graph.path->segments().end()}
                      : std::vector<std::string>{};
  };

  std::sort(snapshot.graphs.begin(), snapshot.graphs.end(), [&](const auto& lhs, const auto& rhs) {
    return std::tuple{graph_path(lhs), lhs.name} < std::tuple{graph_path(rhs), rhs.name};
  });
  IdMap graph_ids;
  for (std::size_t index = 0; index < snapshot.graphs.size(); ++index) {
    graph_ids.emplace(snapshot.graphs[index].id.value, index + 1U);
  }
  for (std::size_t index = 0; index < snapshot.graphs.size(); ++index) {
    auto& graph = snapshot.graphs[index];
    if (graph.parent)
      graph.parent = GraphId{graph_ids.at(graph.parent->value)};
    graph.id = GraphId{index + 1U};
  }

  std::unordered_map<std::uint64_t, std::uint32_t> boundary_ordinals;
  for (const auto& topic : snapshot.external_topics)
    boundary_ordinals.emplace(topic.vertex.value, topic.declaration_ordinal);
  const auto boundary_ordinal = [&](VertexId vertex) {
    const auto found = boundary_ordinals.find(vertex.value);
    return found == boundary_ordinals.end() ? std::numeric_limits<std::uint32_t>::max()
                                            : found->second;
  };
  std::sort(
      snapshot.vertices.begin(), snapshot.vertices.end(), [&](const auto& lhs, const auto& rhs) {
        return std::tuple{
                   graph_ids.at(lhs.graph.value), lhs.kind, lhs.name, boundary_ordinal(lhs.id)} <
               std::tuple{
                   graph_ids.at(rhs.graph.value), rhs.kind, rhs.name, boundary_ordinal(rhs.id)};
      });
  IdMap vertex_ids;
  for (std::size_t index = 0; index < snapshot.vertices.size(); ++index)
    vertex_ids.emplace(snapshot.vertices[index].id.value, index + 1U);
  for (std::size_t index = 0; index < snapshot.vertices.size(); ++index) {
    auto& vertex = snapshot.vertices[index];
    vertex.graph = GraphId{graph_ids.at(vertex.graph.value)};
    vertex.id = VertexId{index + 1U};
  }

  std::sort(snapshot.ports.begin(), snapshot.ports.end(), [&](const auto& lhs, const auto& rhs) {
    return std::tuple{vertex_ids.at(lhs.owner.value), lhs.direction, lhs.name, lhs.owner_kind} <
           std::tuple{vertex_ids.at(rhs.owner.value), rhs.direction, rhs.name, rhs.owner_kind};
  });
  IdMap port_ids;
  for (std::size_t index = 0; index < snapshot.ports.size(); ++index)
    port_ids.emplace(snapshot.ports[index].id.value, index + 1U);
  for (std::size_t index = 0; index < snapshot.ports.size(); ++index) {
    auto& port = snapshot.ports[index];
    if (port.setup.follows_input) {
      port.setup.follows_input = GraphPortId{port_ids.at(port.setup.follows_input->value)};
    }
    if (port.graph_override.follows_input) {
      port.graph_override.follows_input =
          GraphPortId{port_ids.at(port.graph_override.follows_input->value)};
    }
    port.owner = VertexId{vertex_ids.at(port.owner.value)};
    port.id = GraphPortId{index + 1U};
  }

  for (auto& edge : snapshot.edges) {
    edge.source = GraphPortId{port_ids.at(edge.source.value)};
    edge.target = GraphPortId{port_ids.at(edge.target.value)};
  }
  std::sort(snapshot.edges.begin(), snapshot.edges.end(), [](const auto& lhs, const auto& rhs) {
    return std::tuple{lhs.source.value, lhs.target.value} <
           std::tuple{rhs.source.value, rhs.target.value};
  });
  for (std::size_t index = 0; index < snapshot.edges.size(); ++index)
    snapshot.edges[index].id = EdgeId{index + 1U};

  for (auto& binding : snapshot.interface_bindings) {
    binding.public_port = GraphPortId{port_ids.at(binding.public_port.value)};
    binding.internal_port = GraphPortId{port_ids.at(binding.internal_port.value)};
  }
  std::sort(snapshot.interface_bindings.begin(),
            snapshot.interface_bindings.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.public_port.value, lhs.internal_port.value, lhs.direction} <
                     std::tuple{rhs.public_port.value, rhs.internal_port.value, rhs.direction};
            });

  for (auto& topic : snapshot.external_topics) {
    topic.vertex = VertexId{vertex_ids.at(topic.vertex.value)};
    topic.graph_port = GraphPortId{port_ids.at(topic.graph_port.value)};
  }
  std::sort(snapshot.external_topics.begin(),
            snapshot.external_topics.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.vertex.value, lhs.declaration_ordinal, lhs.imported, lhs.uri} <
                     std::tuple{rhs.vertex.value, rhs.declaration_ordinal, rhs.imported, rhs.uri};
            });
  for (std::size_t index = 0; index < snapshot.external_topics.size(); ++index)
    snapshot.external_topics[index].id = ExternalTopicId{index + 1U};

  for (auto& source : snapshot.readiness_sources)
    source.owner = VertexId{vertex_ids.at(source.owner.value)};
  std::sort(snapshot.readiness_sources.begin(),
            snapshot.readiness_sources.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.owner.value, lhs.declaration_ordinal, lhs.kind, lhs.name} <
                     std::tuple{rhs.owner.value, rhs.declaration_ordinal, rhs.kind, rhs.name};
            });

  for (auto& region : snapshot.regions) {
    region.graph = GraphId{graph_ids.at(region.graph.value)};
    for (auto& member : region.members)
      member = VertexId{vertex_ids.at(member.value)};
    for (auto& member : region.graph_members)
      member = GraphId{graph_ids.at(member.value)};
    std::sort(region.members.begin(), region.members.end(), [](VertexId lhs, VertexId rhs) {
      return lhs.value < rhs.value;
    });
    std::sort(region.graph_members.begin(),
              region.graph_members.end(),
              [](GraphId lhs, GraphId rhs) { return lhs.value < rhs.value; });
  }
  std::sort(snapshot.regions.begin(), snapshot.regions.end(), [](const auto& lhs, const auto& rhs) {
    return std::tuple{lhs.graph.value, lhs.name} < std::tuple{rhs.graph.value, rhs.name};
  });
  for (std::size_t index = 0; index < snapshot.regions.size(); ++index)
    snapshot.regions[index].id = RegionId{index + 1U};

  for (auto& partition : snapshot.partitions) {
    partition.graph = GraphId{graph_ids.at(partition.graph.value)};
    for (auto& member : partition.graph_members)
      member = GraphId{graph_ids.at(member.value)};
    std::sort(partition.graph_members.begin(),
              partition.graph_members.end(),
              [](GraphId lhs, GraphId rhs) { return lhs.value < rhs.value; });
    for (auto& member : partition.members)
      member = VertexId{vertex_ids.at(member.value)};
    std::sort(partition.members.begin(), partition.members.end(), [](VertexId lhs, VertexId rhs) {
      return lhs.value < rhs.value;
    });
  }
  std::sort(
      snapshot.partitions.begin(), snapshot.partitions.end(), [](const auto& lhs, const auto& rhs) {
        return std::tuple{lhs.graph.value, lhs.name} < std::tuple{rhs.graph.value, rhs.name};
      });
  for (std::size_t index = 0; index < snapshot.partitions.size(); ++index)
    snapshot.partitions[index].id = PartitionId{index + 1U};

  for (auto& clock : snapshot.clocks)
    clock.graph = GraphId{graph_ids.at(clock.graph.value)};
  std::sort(snapshot.clocks.begin(), snapshot.clocks.end(), [](const auto& lhs, const auto& rhs) {
    return std::tuple{lhs.graph.value, lhs.key.name()} <
           std::tuple{rhs.graph.value, rhs.key.name()};
  });
  for (auto& capability : snapshot.clock_capabilities)
    capability.graph = GraphId{graph_ids.at(capability.graph.value)};
  std::sort(snapshot.clock_capabilities.begin(),
            snapshot.clock_capabilities.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.graph.value, lhs.key.name()} <
                     std::tuple{rhs.graph.value, rhs.key.name()};
            });
  for (auto& clock_import : snapshot.clock_imports)
    clock_import.graph = GraphId{graph_ids.at(clock_import.graph.value)};
  std::sort(snapshot.clock_imports.begin(),
            snapshot.clock_imports.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.graph.value, lhs.key.name()} <
                     std::tuple{rhs.graph.value, rhs.key.name()};
            });
  for (auto& binding : snapshot.clock_bindings) {
    binding.child_graph = GraphId{graph_ids.at(binding.child_graph.value)};
    binding.parent_graph = GraphId{graph_ids.at(binding.parent_graph.value)};
  }
  std::sort(snapshot.clock_bindings.begin(),
            snapshot.clock_bindings.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.child_graph.value,
                                lhs.child_clock.name(),
                                lhs.parent_graph.value,
                                lhs.parent_clock.name()} < std::tuple{rhs.child_graph.value,
                                                                      rhs.child_clock.name(),
                                                                      rhs.parent_graph.value,
                                                                      rhs.parent_clock.name()};
            });
  for (auto& conversion : snapshot.clock_conversions)
    conversion.graph = GraphId{graph_ids.at(conversion.graph.value)};
  std::sort(snapshot.clock_conversions.begin(),
            snapshot.clock_conversions.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.graph.value, lhs.from.name(), lhs.to.name()} <
                     std::tuple{rhs.graph.value, rhs.from.name(), rhs.to.name()};
            });

  for (auto& dependency : snapshot.lifecycle_dependencies) {
    dependency.graph = GraphId{graph_ids.at(dependency.graph.value)};
    dependency.before = VertexId{vertex_ids.at(dependency.before.value)};
    dependency.after = VertexId{vertex_ids.at(dependency.after.value)};
  }
  std::sort(snapshot.lifecycle_dependencies.begin(),
            snapshot.lifecycle_dependencies.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.graph.value, lhs.stage, lhs.before.value, lhs.after.value} <
                     std::tuple{rhs.graph.value, rhs.stage, rhs.before.value, rhs.after.value};
            });
  for (auto& callback : snapshot.lifecycle_callbacks)
    callback.owner = VertexId{vertex_ids.at(callback.owner.value)};
  std::sort(snapshot.lifecycle_callbacks.begin(),
            snapshot.lifecycle_callbacks.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.owner.value, lhs.stage, lhs.required, lhs.method, lhs.origin} <
                     std::tuple{rhs.owner.value, rhs.stage, rhs.required, rhs.method, rhs.origin};
            });
  for (auto& fact : snapshot.lifecycle_facts)
    fact.owner = VertexId{vertex_ids.at(fact.owner.value)};
  std::sort(snapshot.lifecycle_facts.begin(),
            snapshot.lifecycle_facts.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.owner.value, lhs.stage, lhs.kind, lhs.stable_name} <
                     std::tuple{rhs.owner.value, rhs.stage, rhs.kind, rhs.stable_name};
            });
  for (auto& permission : snapshot.lifecycle_restart_permissions) {
    if (permission.requester)
      permission.requester = VertexId{vertex_ids.at(permission.requester->value)};
    permission.target = VertexId{vertex_ids.at(permission.target.value)};
  }
  std::sort(snapshot.lifecycle_restart_permissions.begin(),
            snapshot.lifecycle_restart_permissions.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.origin,
                                lhs.requester.has_value(),
                                lhs.requester ? lhs.requester->value : 0U,
                                lhs.target.value} <
                     std::tuple{rhs.origin,
                                rhs.requester.has_value(),
                                rhs.requester ? rhs.requester->value : 0U,
                                rhs.target.value};
            });
  return snapshot;
}

}  // namespace holoscan::detail
