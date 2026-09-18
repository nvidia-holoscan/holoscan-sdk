/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "snapshot_lookup.hpp"

namespace holoscan::detail {

/// Resolve one direct child subtree, rejecting every internal partition declaration.
[[nodiscard]] inline expected<std::vector<VertexId>, Error> resolve_atomic_instance_members(
    const GraphSnapshotData& snapshot, GraphId root, GraphId selected) {
  const GraphRecord* const graph = find_graph(snapshot, selected);
  if (graph == nullptr || graph->parent != root) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_GRAPH_INVALID"});
  }
  if (std::ranges::any_of(snapshot.partitions, [&](const auto& declaration) {
        return is_descendant_or_self(snapshot, declaration.graph, selected);
      })) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_INTERNAL_DECLARATION"});
  }
  std::vector<VertexId> members;
  for (const auto& vertex : snapshot.vertices) {
    if (vertex.kind == VertexKind::kOperator &&
        is_descendant_or_self(snapshot, vertex.graph, selected)) {
      members.push_back(vertex.id);
    }
  }
  if (members.empty()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_GRAPH_EMPTY"});
  }
  return members;
}

/// Resolve one exact authored selection for admission and frozen-plan authority validation.
/// Ordinary selections contain only Operators; include_all also owns executable endpoints.
[[nodiscard]] inline expected<std::vector<VertexId>, Error> resolve_authored_partition_members(
    const GraphSnapshotData& snapshot, const PartitionRecord& partition) {
  const auto invalid = [](const char* token) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, token});
  };
  const GraphRecord* const root = find_root_graph(snapshot);
  if (root == nullptr || partition.graph != root->id) {
    return invalid("PARTITION_COVER_INVALID");
  }
  std::vector<VertexId> members;
  if (partition.include_all) {
    if (snapshot.partitions.size() != 1U || snapshot.automatic_partition_remaining ||
        !partition.members.empty() || !partition.graph_members.empty()) {
      return invalid("GRAPH_PARTITION_INCLUDE_ALL_CONFLICT");
    }
    for (const auto& vertex : snapshot.vertices) {
      if (vertex.kind == VertexKind::kOperator || vertex.kind == VertexKind::kImportedTopic ||
          vertex.kind == VertexKind::kExportedTopic) {
        members.push_back(vertex.id);
      }
    }
    return members;
  }
  for (const VertexId member : partition.members) {
    const VertexRecord* const vertex = find_vertex(snapshot, member);
    if (vertex == nullptr || vertex->kind != VertexKind::kOperator || vertex->graph != root->id) {
      return invalid("PARTITION_COVER_INVALID");
    }
    members.push_back(member);
  }
  GraphId previous;
  for (const GraphId selected : partition.graph_members) {
    if (previous && selected.value <= previous.value) {
      return invalid("GRAPH_PARTITION_GRAPH_INVALID");
    }
    previous = selected;
    const auto resolved = resolve_atomic_instance_members(snapshot, root->id, selected);
    if (!resolved)
      return make_unexpected(resolved.error());
    members.insert(members.end(), resolved->begin(), resolved->end());
  }
  std::sort(members.begin(), members.end(), [](VertexId lhs, VertexId rhs) {
    return lhs.value < rhs.value;
  });
  if (members.empty() || std::adjacent_find(members.begin(), members.end()) != members.end()) {
    return invalid("PARTITION_COVER_INVALID");
  }
  return members;
}

}  // namespace holoscan::detail
