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

#ifndef HOLOSCAN_CORE_GRAPHS_FLOW_GRAPH_HPP
#define HOLOSCAN_CORE_GRAPHS_FLOW_GRAPH_HPP

#include <functional>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

#include "../graph.hpp"

namespace holoscan {

// Forward declarations
template <typename NodeT, typename EdgeDataElementT>
class FlowGraph;

// Graph type aliases
//   for operator graph
using OperatorFlowGraph = FlowGraph<OperatorNodeType, OperatorEdgeDataElementType>;
//   for fragment graph
using FragmentFlowGraph = FlowGraph<FragmentNodeType, FragmentEdgeDataElementType>;

template <typename NodeT = OperatorNodeType,
          typename EdgeDataElementT = OperatorEdgeDataElementType>
class FlowGraph : public Graph<NodeT, EdgeDataElementT> {
 public:
  using NodeType = NodeT;
  using NodePredicate = std::function<bool(const NodeType&)>;
  using EdgeDataElementType = EdgeDataElementT;
  using EdgeDataType = std::shared_ptr<EdgeDataElementType>;

  // Custom comparator for NodeType that orders by insertion order
  struct NodeTypeCompare {
    const std::list<NodeType>* ordered_nodes;

    // Default constructor required by std::map
    NodeTypeCompare() : ordered_nodes(nullptr) {}

    explicit NodeTypeCompare(const std::list<NodeType>* nodes) : ordered_nodes(nodes) {}

    bool operator()(const NodeType& lhs, const NodeType& rhs) const {
      // If ordered_nodes is null, fall back to name comparison
      if (!ordered_nodes) { return lhs->name() < rhs->name(); }
      // Find the positions in ordered_nodes_
      auto lhs_it = std::find(ordered_nodes->begin(), ordered_nodes->end(), lhs);
      auto rhs_it = std::find(ordered_nodes->begin(), ordered_nodes->end(), rhs);
      // Compare positions
      return std::distance(ordered_nodes->begin(), lhs_it) <
             std::distance(ordered_nodes->begin(), rhs_it);
    }
  };

  using Graph<NodeT, EdgeDataElementT>::Graph;
  ~FlowGraph() override = default;

  void add_node(const NodeType& node) override;
  void add_flow(const NodeType& node_u, const NodeType& node_v,
                const EdgeDataType& port_map) override;

  std::optional<EdgeDataType> get_port_map(const NodeType& node_u,
                                           const NodeType& node_v) const override;

  bool is_root(const NodeType& node) const override;

  bool is_user_defined_root(const NodeType& node) const override {
    return get_nodes().empty() ? false : get_nodes()[0] == node;
  }

  bool is_leaf(const NodeType& node) const override;

  std::vector<NodeType> has_cycle() const override;

  /**
   * @brief Get all root nodes.
   *
   * The nodes are returned in the order they were added to the graph.
   *
   * @return A vector of all root nodes.
   */
  std::vector<NodeType> get_root_nodes() const override;

  /**
   * @brief Get all nodes.
   *
   * The nodes are returned in the order they were added to the graph.
   *
   * @return A vector of all nodes.
   */
  std::vector<NodeType> get_nodes() const override;

  /**
   * @brief Get all nodes immediately downstream of a given node.
   *
   * The nodes are returned in the order in which they were added to the graph.
   *
   * @return A vector of all next nodes.
   */
  std::vector<NodeType> get_next_nodes(const NodeType& node) const override;

  /**
   * @brief Get all nodes immediately upstream of a given node.
   *
   * The nodes are returned in the order in which they were added to the graph.
   *
   * @param node The node to get the upstream nodes of.
   * @return A vector of all previous nodes.
   */
  std::vector<NodeType> get_previous_nodes(const NodeType& node) const override;

  NodeType find_node(const NodePredicate& pred) const override;

  NodeType find_node(const NodeType& node) const override;

  NodeType find_node(const std::string& name) const override;

  void remove_node(const NodeType& node) override;

 private:
  // Use std::map values so that nodes returned by get_root_nodes() and get_next_nodes()
  // are in a deterministic order (by insertion order).
  std::unordered_map<NodeType, std::map<NodeType, EdgeDataType, NodeTypeCompare>> succ_;
  std::unordered_map<NodeType, std::map<NodeType, EdgeDataType, NodeTypeCompare>> pred_;

  std::list<NodeType> ordered_nodes_;  ///< Nodes in the order they were added to the graph.
  std::unordered_map<std::string, NodeType> name_map_;  ///< Map from node name to node.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_GRAPHS_FLOW_GRAPH_HPP */
