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
#include <utility>
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
      // If ordered_nodes is null or empty, fall back to name comparison
      if (!ordered_nodes || ordered_nodes->empty()) {
        return lhs->name() < rhs->name();
      }
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

  bool is_user_defined_root(const NodeType& node) const override;

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

  /**
   * @brief Get the outdegree of a given node of a given port.
   *
   * @param node The node to get the outdegree of.
   * @param port_name The name of the port in the given node.
   * @return The outdegree of the given node of the given port.
   */
  size_t get_outdegree(const NodeType& node, const std::string& port_name) const override;

  /**
   * @brief Get the indegree of a given node of a given port.
   *
   * @param node The node to get the indegree of.
   * @param port_name The name of the port in the given node.
   * @return The indegree of the given node of the given port.
   */
  size_t get_indegree(const NodeType& node, const std::string& port_name) const override;

  /**
   * @brief Get port connectivity maps for the graph.
   *
   * Returns two maps that describe the connectivity between input and output ports:
   * 1. Input-to-Output map: Keys are input port unique IDs,
   *    values are vectors of output port unique IDs that connect to this input port.
   * 2. Output-to-Input map: Keys are output port unique IDs, values are vectors of input port
   *    unique IDs that this output port connects to.
   *
   * For multi-receiver ports, each individual port (e.g., "in:0", "in:1", "in:2") is listed as
   * a separate key.
   *
   * For `OperatorFlowGraph` the unique ID has format:
   *     "<fragment_name>.<operator_name>.<port_name>"
   *     (or just <operator_name>.<port_name> if no fragment name was assigned).
   *
   * For `FragmentGraph` the unique ID has format: "<fragment_name>.<port_name>".
   *
   * @return A pair containing (input_to_output_map, output_to_input_map)
   */
  std::pair<std::map<std::string, std::vector<std::string>>,
            std::map<std::string, std::vector<std::string>>>
  get_port_connectivity_maps() const override;

  /**
   * @brief Get a YAML formatted description of the port connectivity maps.
   *
   * Returns a string containing the port connectivity information in YAML format.
   * The output includes both input-to-output and output-to-input mappings.
   *
   * Example output for the following computation graph
   *
   * tx -> mx ---- rx1
   *          \___ rx2
   *
   * (assume each operator has one input port named 'in' and one output port named 'out')
   *
   * ```yaml
   * input_to_output:
   *   mx.in:
   *     - tx.out
   *   rx1.in:
   *     - mx.out
   *   rx2.in:
   *     - mx.out
   * output_to_input:
   *   tx.out:
   *     - mx.in
   *   mx.out:
   *     - rx1.in
   *     - rx2.in
   * ```
   *
   * @return A YAML formatted string describing the port connectivity
   */
  std::string port_map_description() const override;

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

  // Cache for expensive cycle detection
  mutable std::optional<std::vector<NodeType>> cached_cyclic_roots_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_GRAPHS_FLOW_GRAPH_HPP */
