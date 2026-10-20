/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "../core/canonical_text_encoding.hpp"
#include "core/graph_snapshot.hpp"

namespace holoscan::detail {
namespace snapshot_lookup_detail {

template <typename Record, typename Id>
[[nodiscard]] inline const Record* find_dense_record(const std::vector<Record>& records,
                                                     Id id) noexcept {
  if (!id || id.value > records.size()) {
    return nullptr;
  }
  const auto& record = records[static_cast<std::size_t>(id.value - 1U)];
  return record.id == id ? &record : nullptr;
}

}  // namespace snapshot_lookup_detail

[[nodiscard]] inline const GraphRecord* find_graph(const GraphSnapshotData& snapshot,
                                                   GraphId id) noexcept {
  return snapshot_lookup_detail::find_dense_record(snapshot.graphs, id);
}

[[nodiscard]] inline const VertexRecord* find_vertex(const GraphSnapshotData& snapshot,
                                                     VertexId id) noexcept {
  return snapshot_lookup_detail::find_dense_record(snapshot.vertices, id);
}

[[nodiscard]] inline const PortRecord* find_port(const GraphSnapshotData& snapshot,
                                                 GraphPortId id) noexcept {
  return snapshot_lookup_detail::find_dense_record(snapshot.ports, id);
}

[[nodiscard]] inline const EdgeRecord* find_edge(const GraphSnapshotData& snapshot,
                                                 EdgeId id) noexcept {
  return snapshot_lookup_detail::find_dense_record(snapshot.edges, id);
}

/// Return the first root Graph, or null when the snapshot has no parentless Graph.
[[nodiscard]] inline const GraphRecord* find_root_graph(
    const GraphSnapshotData& snapshot) noexcept {
  const auto root = std::find_if(snapshot.graphs.begin(),
                                 snapshot.graphs.end(),
                                 [](const auto& graph) { return !graph.parent.has_value(); });
  return root == snapshot.graphs.end() ? nullptr : &*root;
}

[[nodiscard]] inline std::string canonical_graph_path(const GraphRecord& graph) {
  if (!graph.path) {
    return "_root";
  }
  std::string result;
  for (const auto& segment : graph.path->segments()) {
    result.push_back('/');
    result += segment;
  }
  return result.empty() ? std::string{"_root"} : result;
}

[[nodiscard]] inline std::string canonical_vertex_path(const GraphSnapshotData& snapshot,
                                                       const VertexRecord& vertex) {
  const auto* graph = find_graph(snapshot, vertex.graph);
  if (graph == nullptr || !graph->path) {
    return vertex.name;
  }
  std::string result = canonical_graph_path(*graph);
  result.push_back('/');
  result += vertex.name;
  return result;
}

/// Render an Operator identity with the public OperatorPath grammar: decoded Graph and Operator
/// name segments are percent-encoded independently, joined by '/', and never prefixed by '/'.
[[nodiscard]] inline std::optional<std::string> canonical_operator_path(
    const GraphRecord& graph, const VertexRecord& vertex) {
  if (graph.id != vertex.graph || vertex.kind != VertexKind::kOperator || vertex.name.empty()) {
    return std::nullopt;
  }
  std::string result;
  if (graph.path.has_value()) {
    const auto& graph_path = graph.path.value();  // NOLINT(bugprone-unchecked-optional-access)
    for (const auto& segment : graph_path.segments()) {
      append_percent_encoded_uri_path_segment(result, segment);
    }
  }
  append_percent_encoded_uri_path_segment(result, vertex.name);
  return result;
}

[[nodiscard]] inline std::optional<std::string> canonical_operator_path(
    std::span<const GraphRecord> graphs, const VertexRecord& vertex) {
  const auto graph = std::find_if(graphs.begin(), graphs.end(), [&](const GraphRecord& candidate) {
    return candidate.id == vertex.graph;
  });
  return graph == graphs.end() ? std::nullopt : canonical_operator_path(*graph, vertex);
}

[[nodiscard]] inline std::optional<std::string> canonical_operator_path(
    const GraphSnapshotData& snapshot, const VertexRecord& vertex) {
  const auto* graph = find_graph(snapshot, vertex.graph);
  return graph == nullptr ? std::nullopt : canonical_operator_path(*graph, vertex);
}

/// Name a port for diagnostics as `<operator path>.<port>`, or by its local name when its owner
/// is not an Operator.
[[nodiscard]] inline std::string diagnostic_port_label(const GraphSnapshotData& snapshot,
                                                       GraphPortId id) {
  const auto* port = find_port(snapshot, id);
  if (port == nullptr) {
    return "port#" + std::to_string(id.value);
  }
  const auto* owner = find_vertex(snapshot, port->owner);
  const auto operator_path =
      owner == nullptr ? std::nullopt : canonical_operator_path(snapshot, *owner);
  return operator_path ? *operator_path + "." + port->name : port->name;
}

[[nodiscard]] inline GraphPortId resolve_interface_port(const GraphSnapshotData& snapshot,
                                                        GraphPortId port) noexcept {
  // Interface bindings are same-direction aliases. Resolve repeatedly because
  // a reusable Graph may be embedded through more than one interface level.
  for (std::size_t depth = 0U; depth <= snapshot.interface_bindings.size(); ++depth) {
    const auto found = std::lower_bound(
        snapshot.interface_bindings.begin(),
        snapshot.interface_bindings.end(),
        port.value,
        [](const auto& binding, auto value) { return binding.public_port.value < value; });
    if (found == snapshot.interface_bindings.end() || found->public_port != port) {
      return port;
    }
    port = found->internal_port;
  }
  return {};
}

[[nodiscard]] inline std::vector<GraphPortId> interface_alias_component(
    const GraphSnapshotData& snapshot, GraphPortId initial) {
  std::vector<GraphPortId> component{initial};
  for (std::size_t cursor = 0U; cursor < component.size(); ++cursor) {
    for (const auto& binding : snapshot.interface_bindings) {
      GraphPortId adjacent{};
      if (binding.public_port == component[cursor]) {
        adjacent = binding.internal_port;
      } else if (binding.internal_port == component[cursor]) {
        adjacent = binding.public_port;
      }
      if (adjacent && std::find(component.begin(), component.end(), adjacent) == component.end()) {
        component.push_back(adjacent);
      }
    }
  }
  return component;
}

[[nodiscard]] inline std::optional<Profile> effective_profile_for_vertex(
    std::span<const std::optional<Profile>> effective_profiles, VertexId vertex) noexcept {
  if (!vertex || vertex.value > effective_profiles.size()) {
    return std::nullopt;
  }
  return effective_profiles[static_cast<std::size_t>(vertex.value - 1U)];
}

[[nodiscard]] inline bool is_descendant_or_self(const GraphSnapshotData& snapshot,
                                                GraphId candidate, GraphId ancestor) noexcept {
  // Composition normally guarantees a tree. The bound makes malformed parent
  // cycles fail closed instead of turning a cold validation into an unbounded walk.
  for (std::size_t depth = 0U; depth <= snapshot.graphs.size(); ++depth) {
    if (candidate == ancestor) {
      return true;
    }
    const GraphRecord* const graph = find_graph(snapshot, candidate);
    if (graph == nullptr || !graph->parent) {
      return false;
    }
    candidate = *graph->parent;
  }
  return false;
}

}  // namespace holoscan::detail
