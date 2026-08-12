/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

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
