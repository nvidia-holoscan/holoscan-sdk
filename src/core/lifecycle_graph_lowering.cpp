/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lifecycle_graph_lowering.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace holoscan::detail {
namespace {

[[nodiscard]] LifecycleKernelError lowering_error(LifecycleKernelErrorCode code,
                                                  LifecycleScopeId scope_id = {},
                                                  LifecycleNodeId node_id = {},
                                                  LifecycleStage stage = LifecycleStage::kConfigure,
                                                  std::string stable_name = {}) {
  return {
      .code = code,
      .scope_id = scope_id,
      .node_id = node_id,
      .stage = stage,
      .stable_name = std::move(stable_name),
  };
}

[[nodiscard]] LifecycleLogicalCallbackOrigin callback_origin(
    LifecycleCallbackOrigin origin) noexcept {
  switch (origin) {
    case LifecycleCallbackOrigin::kAdvancedMember:
      return LifecycleLogicalCallbackOrigin::kAdvancedMember;
    case LifecycleCallbackOrigin::kLegacyStart:
      return LifecycleLogicalCallbackOrigin::kLegacyStart;
    case LifecycleCallbackOrigin::kLegacyStop:
      return LifecycleLogicalCallbackOrigin::kLegacyStop;
  }
  return LifecycleLogicalCallbackOrigin::kAdvancedMember;
}

}  // namespace

// Lower all lifecycle scopes and dependencies as one validated graph input.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<LifecycleGraphLowering, LifecycleKernelError> lower_lifecycle_stage_graph_input(
    const GraphSnapshotData& snapshot) {
  LifecycleGraphLowering result;
  result.input.scopes.reserve(snapshot.graphs.size());
  for (const auto& graph : snapshot.graphs) {
    if (graph.id.value == 0U) {
      return make_unexpected(lowering_error(LifecycleKernelErrorCode::kInvalidScope));
    }
    result.input.scopes.push_back({
        .scope_id = graph.id.value,
        .parent_scope_id =
            graph.parent ? std::optional<LifecycleScopeId>{graph.parent->value} : std::nullopt,
    });
  }

  std::unordered_map<std::uint64_t, LifecycleNodeId> nodes_by_vertex;
  nodes_by_vertex.reserve(snapshot.vertices.size());
  for (const auto& vertex : snapshot.vertices) {
    if (vertex.kind != VertexKind::kOperator)
      continue;
    if (result.input.nodes.size() >= std::numeric_limits<LifecycleNodeId>::max()) {
      return make_unexpected(lowering_error(LifecycleKernelErrorCode::kCapacityExceeded));
    }
    const LifecycleNodeId node_id = static_cast<LifecycleNodeId>(result.input.nodes.size() + 1U);
    if (!nodes_by_vertex.emplace(vertex.id.value, node_id).second) {
      return make_unexpected(
          lowering_error(LifecycleKernelErrorCode::kDuplicateNode, vertex.graph.value, node_id));
    }
    result.input.nodes.push_back({
        .node_id = node_id,
        .scope_id = vertex.graph.value,
        .stable_name = vertex.name,
        .stable_order = node_id,
    });
  }
  if (result.input.nodes.empty()) {
    return make_unexpected(lowering_error(LifecycleKernelErrorCode::kInvalidNode));
  }

  const auto node_for =
      [&](VertexId vertex,
          LifecycleStage stage) -> expected<LifecycleNodeId, LifecycleKernelError> {
    const auto found = nodes_by_vertex.find(vertex.value);
    if (found == nodes_by_vertex.end()) {
      return make_unexpected(lowering_error(LifecycleKernelErrorCode::kInvalidNode, {}, {}, stage));
    }
    return found->second;
  };

  result.input.callbacks.reserve(snapshot.lifecycle_callbacks.size());
  for (const auto& callback : snapshot.lifecycle_callbacks) {
    auto node = node_for(callback.owner, callback.stage);
    if (!node)
      return make_unexpected(std::move(node.error()));
    result.input.callbacks.push_back({
        .node_id = *node,
        .stage = callback.stage,
        .required = callback.required,
        .method = callback.method,
        .origin = callback_origin(callback.origin),
    });
  }

  for (const auto& fact : snapshot.lifecycle_facts) {
    auto node = node_for(fact.owner, fact.stage);
    if (!node)
      return make_unexpected(std::move(node.error()));
    const auto owner = std::find_if(snapshot.vertices.begin(),
                                    snapshot.vertices.end(),
                                    [&](const auto& vertex) { return vertex.id == fact.owner; });
    if (owner == snapshot.vertices.end() || owner->kind != VertexKind::kOperator) {
      return make_unexpected(
          lowering_error(LifecycleKernelErrorCode::kInvalidNode, {}, *node, fact.stage));
    }
    LifecycleLogicalDependencyKey key{
        .scope_id = owner->graph.value,
        .stage = fact.stage,
        .stable_name = fact.stable_name,
    };
    if (fact.kind == LifecycleFactKind::kProduces) {
      result.input.producers.push_back({
          .node_id = *node,
          .declared_stage = fact.stage,
          .key = std::move(key),
      });
    } else {
      result.input.requirements.push_back({
          .node_id = *node,
          .declared_stage = fact.stage,
          .key = std::move(key),
      });
    }
  }

  result.input.direct_edges.reserve(snapshot.lifecycle_dependencies.size());
  for (const auto& dependency : snapshot.lifecycle_dependencies) {
    auto before = node_for(dependency.before, dependency.stage);
    if (!before)
      return make_unexpected(std::move(before.error()));
    auto after = node_for(dependency.after, dependency.stage);
    if (!after)
      return make_unexpected(std::move(after.error()));
    result.input.direct_edges.push_back({
        .declaration_scope_id = dependency.graph.value,
        .before_node_id = *before,
        .before_stage = dependency.stage,
        .after_node_id = *after,
        .after_stage = dependency.stage,
    });
  }

  if (result.input.nodes.size() >
      std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(6U)) {
    return make_unexpected(lowering_error(LifecycleKernelErrorCode::kCapacityExceeded));
  }
  result.minimum_capacity = {
      .max_scopes = result.input.scopes.size(),
      .max_nodes = result.input.nodes.size(),
      .max_callbacks = result.input.callbacks.size(),
      .max_dependency_records = result.input.producers.size() + result.input.requirements.size(),
      .max_direct_edges = result.input.direct_edges.size(),
      .max_completions = result.input.nodes.size() * 6U,
  };
  return result;
}

}  // namespace holoscan::detail
