/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "holoscan/core/graphs/flow_graph.hpp"

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "holoscan/core/errors.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"

namespace holoscan {

// Explicit instantiation
//   for OperatorFlowGraph
template class FlowGraph<OperatorNodeType, OperatorEdgeDataElementType>;
//   for FragmentFlowGraph
template class FlowGraph<FragmentNodeType, FragmentEdgeDataElementType>;

template <typename NodeT, typename EdgeDataElementT>
void FlowGraph<NodeT, EdgeDataElementT>::add_node(const NodeT& node) {
  if (succ_.find(node) == succ_.end()) {
    if (!node) {
      HOLOSCAN_LOG_ERROR("Calling add_node() with nullptr");
      return;
    }
    // If there is already a node with the same name, it will raise an error.
    if (name_map_.find(node->name()) != name_map_.end()) {
      HOLOSCAN_LOG_ERROR("Calling add_node() with a node ('{}') that has a duplicate name",
                         node->name());
      throw RuntimeError(ErrorCode::kDuplicateName);
    }

    succ_[node] =
        std::map<NodeType, EdgeDataType, NodeTypeCompare>(NodeTypeCompare(&ordered_nodes_));
    pred_[node] =
        std::map<NodeType, EdgeDataType, NodeTypeCompare>(NodeTypeCompare(&ordered_nodes_));
    ordered_nodes_.push_back(node);
    name_map_[node->name()] = node;
  }
}

template <typename NodeT, typename EdgeDataElementT>
void FlowGraph<NodeT, EdgeDataElementT>::add_flow(const NodeType& node_u, const NodeType& node_v,
                                                  const EdgeDataType& port_map) {
  // Note: add_node does nothing if the node was already added
  add_node(node_u);
  add_node(node_v);
  auto it_edgedata = succ_[node_u].find(node_v);
  if (it_edgedata != succ_[node_u].end()) {
    const auto& datadict = it_edgedata->second;
    if (port_map) {
      for (auto& [key, value] : *port_map) {
        if (datadict->count(key)) {
          // merge any new connections with pre-existing ones
          datadict->at(key).merge(value);
        } else {
          datadict->insert({key, value});
        }
      }
    }
    succ_[node_u][node_v] = datadict;
    pred_[node_v][node_u] = datadict;
  } else {
    auto datadict = std::make_shared<EdgeDataElementType>();

    if (port_map) {
      for (auto& [key, value] : *port_map) { datadict->insert({key, value}); }
    }
    succ_[node_u][node_v] = datadict;
    pred_[node_v][node_u] = datadict;
  }
}

/// Remove a node and all its edges from the graph
template <typename NodeT, typename EdgeDataElementT>
void FlowGraph<NodeT, EdgeDataElementT>::remove_node(const NodeType& node) {
  auto it_prev = pred_.find(node);
  if (it_prev == pred_.end()) {
    HOLOSCAN_LOG_WARN("Node with name '{}' not found in graph: no node was removed.", node->name());
    return;
  }

  // remove node from the predecessors map of its successor nodes
  for (auto& [s_node, _] : succ_[node]) {
    auto& pred_map = pred_[s_node];
    if (pred_map.find(node) != pred_map.end()) { pred_map.erase(node); }
  }

  // remove node from the successors map of its predecessor nodes
  for (auto& [p_node, _] : pred_[node]) {
    auto& succ_map = succ_[p_node];
    if (succ_map.find(node) != succ_map.end()) { succ_map.erase(node); }
  }

  // Remove the node itself
  succ_.erase(node);
  pred_.erase(node);
  auto it = std::find(ordered_nodes_.begin(), ordered_nodes_.end(), node);
  if (it != ordered_nodes_.end()) { ordered_nodes_.erase(it); }
  name_map_.erase(node->name());
}

template <typename NodeT, typename EdgeDataElementT>
std::optional<typename FlowGraph<NodeT, EdgeDataElementT>::EdgeDataType>
FlowGraph<NodeT, EdgeDataElementT>::get_port_map(const NodeType& node_u,
                                                 const NodeType& node_v) const {
  auto it_u = succ_.find(node_u);
  if (it_u == succ_.end()) { return std::nullopt; }
  auto it_v = it_u->second.find(node_v);
  if (it_v == it_u->second.end()) { return std::nullopt; }
  return it_v->second;
}

template <typename NodeT, typename EdgeDataElementT>
bool FlowGraph<NodeT, EdgeDataElementT>::is_root(const NodeType& node) const {
  if (!node) {
    HOLOSCAN_LOG_WARN("Calling is_root() with nullptr");
    return false;
  }
  auto it_pred = pred_.find(node);
  if (it_pred == pred_.end()) {
    HOLOSCAN_LOG_WARN("Node with name '{}' not found in graph: cannot determine if it is a root",
                      node->name());
    return false;
  }
  if (it_pred->second.empty()) { return true; }

  return false;
}

template <typename NodeT, typename EdgeDataElementT>
bool FlowGraph<NodeT, EdgeDataElementT>::is_leaf(const NodeType& node) const {
  if (!node) {
    HOLOSCAN_LOG_WARN("Calling is_leaf() with nullptr");
    return false;
  }
  auto it_succ = succ_.find(node);
  if (it_succ == succ_.end()) {
    HOLOSCAN_LOG_WARN("Node with name '{}' not found in graph: cannot determine if it is a leaf",
                      node->name());
    return false;
  }
  if (it_succ->second.empty()) { return true; }
  return false;
}

template <typename NodeT, typename EdgeDataElementT>
std::vector<typename FlowGraph<NodeT, EdgeDataElementT>::NodeType>
FlowGraph<NodeT, EdgeDataElementT>::has_cycle() const {
  std::vector<NodeType> cyclic_roots;

  // List of visited nodes across DFS from multiple roots
  std::unordered_set<NodeType> global_visited;

  // Do an iterative DFS from all the root nodes
  auto root_nodes = get_root_nodes();

  if (root_nodes.size() == 0) {
    // There is no implicit root. Therefore, we need to start from somewhere.
    // Start from the first added node which is user-defined root.
    // FIXME Currently, this function is not supported for a disconnected graph.
    root_nodes.push_back(ordered_nodes_.front());
  }

  for (const auto& node : root_nodes) {
    std::unordered_set<NodeType> current_visited;
    std::vector<NodeType> stack;
    stack.push_back(node);

    while (!stack.empty()) {
      auto node = stack.back();
      stack.pop_back();

      current_visited.insert(node);

      // This node has already been visited in a previous DFS traversal
      if (global_visited.find(node) != global_visited.end()) { continue; }

      global_visited.insert(node);

      auto succ_it = succ_.find(node);
      if (succ_it != succ_.end()) {
        for (const auto& [node_next, _] : succ_it->second) {
          if (current_visited.find(node_next) != current_visited.end()) {
            // The currently visited set of nodes already includes the successor node
            // Therefore, it must be a cycle
            cyclic_roots.push_back(node_next);
            continue;  // skip adding the node to the stack
          }
          stack.push_back(node_next);
        }
      }
    }
  }
  return cyclic_roots;
}

template <typename NodeT, typename EdgeDataElementT>
std::vector<typename FlowGraph<NodeT, EdgeDataElementT>::NodeType>
FlowGraph<NodeT, EdgeDataElementT>::get_root_nodes() const {
  std::vector<NodeType> roots;
  for (const auto& node : ordered_nodes_) {
    if (is_root(node)) { roots.push_back(node); }
  }
  return roots;
}

template <typename NodeT, typename EdgeDataElementT>
std::vector<typename FlowGraph<NodeT, EdgeDataElementT>::NodeType>
FlowGraph<NodeT, EdgeDataElementT>::get_nodes() const {
  std::vector<NodeType> nodes;
  nodes.reserve(ordered_nodes_.size());  // pre-allocate memory
  for (const auto& node : ordered_nodes_) { nodes.push_back(node); }
  return nodes;
}

template <typename NodeT, typename EdgeDataElementT>
std::vector<typename FlowGraph<NodeT, EdgeDataElementT>::NodeType>
FlowGraph<NodeT, EdgeDataElementT>::get_next_nodes(const NodeType& node) const {
  std::vector<NodeType> nodes;
  auto it_succ = succ_.find(node);
  if (it_succ == succ_.end()) { return nodes; }
  nodes.reserve(it_succ->second.size());  // pre-allocate memory
  for (const auto& [node_next, _] : it_succ->second) { nodes.push_back(node_next); }
  return nodes;
}

template <typename NodeT, typename EdgeDataElementT>
std::vector<typename FlowGraph<NodeT, EdgeDataElementT>::NodeType>
FlowGraph<NodeT, EdgeDataElementT>::get_previous_nodes(const NodeType& node) const {
  std::vector<NodeType> nodes;
  auto it_prev = pred_.find(node);
  if (it_prev == pred_.end()) { return nodes; }
  nodes.reserve(it_prev->second.size());
  for (const auto& [node_prev, _] : it_prev->second) { nodes.push_back(node_prev); }
  return nodes;
}

template <typename NodeT, typename EdgeDataElementT>
typename FlowGraph<NodeT, EdgeDataElementT>::NodeType FlowGraph<NodeT, EdgeDataElementT>::find_node(
    const NodePredicate& pred) const {
  for (const auto& [node, _] : succ_) {
    if (pred(node)) { return node; }
  }
  return nullptr;
}

template <typename NodeT, typename EdgeDataElementT>
typename FlowGraph<NodeT, EdgeDataElementT>::NodeType FlowGraph<NodeT, EdgeDataElementT>::find_node(
    const NodeType& node) const {
  auto it_prev = pred_.find(node);
  if (it_prev == pred_.end()) { return nullptr; }
  return it_prev->first;
}

template <typename NodeT, typename EdgeDataElementT>
typename FlowGraph<NodeT, EdgeDataElementT>::NodeType FlowGraph<NodeT, EdgeDataElementT>::find_node(
    const std::string& name) const {
  auto it = name_map_.find(name);
  if (it == name_map_.end()) { return nullptr; }
  return it->second;
}

}  // namespace holoscan
