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

#ifndef HOLOSCAN_CORE_GRAPH_HPP
#define HOLOSCAN_CORE_GRAPH_HPP

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "./common.hpp"
namespace holoscan {

// Forward declarations
class Operator;

// Graph type aliases
//   for operator graph
using OperatorNodeType = std::shared_ptr<Operator>;
using OperatorEdgeDataElementType =
    std::unordered_map<std::string, std::set<std::string, std::less<>>>;
using OperatorGraph = Graph<OperatorNodeType, OperatorEdgeDataElementType>;

//   for fragment graph
using FragmentNodeType = std::shared_ptr<Fragment>;
using FragmentEdgeDataElementType =
    std::unordered_map<std::string, std::set<std::string, std::less<>>>;
using FragmentGraph = Graph<FragmentNodeType, FragmentEdgeDataElementType>;

/**
 * @brief Abstract base class for all graphs.
 */
template <typename NodeT = OperatorNodeType,
          typename EdgeDataElementT = OperatorEdgeDataElementType>
class Graph {
 public:
  using NodeType = NodeT;
  using NodePredicate = std::function<bool(const NodeT&)>;
  using EdgeDataElementType = EdgeDataElementT;
  using EdgeDataType = std::shared_ptr<EdgeDataElementT>;

  Graph() = default;
  virtual ~Graph() = default;

  // Delete the copy constructor and assignment operator to prevent copying.
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  /**
   * @brief Add the node to the graph.
   *
   * @param node The node to add.
   */
  virtual void add_node(const NodeT& node) = 0;
  /**
   * @brief Add an edge to the graph.
   *
   * @param node_u A source node.
   * @param node_v A destination node.
   * @param port_map A map from the source node's port name to the destination node's port
   * name(s).
   */
  virtual void add_flow(const NodeType& node_u, const NodeType& node_v,
                        const EdgeDataType& port_map) = 0;

  /**
   * @brief Get a mapping from the source node's port name to the destination node's port
   * name(s).
   *
   * @param node_u A source node.
   * @param node_v A destination node.
   * @return A map from the source node's port name to the destination node's port name(s).
   */
  virtual std::optional<EdgeDataType> get_port_map(const NodeType& node_u,
                                                   const NodeType& node_v) const = 0;

  /**
   * @brief Check if the graph is empty.
   *
   * @return true if the graph is empty. Otherwise, false.
   */
  virtual bool is_empty() const {
    return !find_node([](const NodeType&) { return true; });
  }

  /**
   * @brief Check if the node is a root node.
   *
   * @param node A node in the graph.
   * @return true if the node is a root node.
   */
  virtual bool is_root(const NodeType& node) const = 0;

  /**
   * @brief Check if the node is a user-defined root node. A user-defined root is the first node
   * that is added to the graph.
   *
   * @param node A node in the graph.
   * @return true if the node is a user-defined root node.
   */
  virtual bool is_user_defined_root(const NodeType& node) const = 0;

  /**
   * @brief Check if the node is a leaf node.
   *
   * @param node A node in the graph.
   * @return true if the node is a leaf node.
   */
  virtual bool is_leaf(const NodeType& node) const = 0;

  /**
   * @brief Returns a vector of root nodes of the cycles if the graph has cycle(s). Otherwise, an
   * empty vector is returned.
   *
   * @return Returns a vector of root nodes of cycles.
   */
  virtual std::vector<NodeType> has_cycle() const = 0;

  /**
   * @brief Get all root nodes.
   *
   * @return A vector of root nodes.
   */
  virtual std::vector<NodeType> get_root_nodes() const = 0;

  /**
   * @brief Get all nodes.
   *
   * The order of the nodes is not guaranteed.
   *
   * @return A vector of all nodes.
   */
  virtual std::vector<NodeType> get_nodes() const = 0;

  /**
   * @brief Get the next nodes of the given node.
   *
   * @param node A node in the graph.
   * @return A vector of next nodes.
   */
  virtual std::vector<NodeType> get_next_nodes(const NodeType& node) const = 0;

  /**
   * @brief Find a node in the graph that satisfies the given predicate.
   * @param pred A predicate.
   * @return The node if found, otherwise nullptr.
   */
  virtual NodeType find_node(const NodePredicate& pred) const = 0;

  /**
   * @brief Find a node in the graph that is equal to the given node.
   *
   * @param node The node to find.
   * @return The node in the graph if found, otherwise nullptr.
   */
  virtual NodeType find_node(const NodeType& node) const = 0;

  /**
   * @brief Find a node in the graph whose name is equal to the given name.
   *
   * @param name The name to find.
   * @return The node in the graph if found, otherwise nullptr.
   */
  virtual NodeType find_node(const std::string& name) const = 0;

  /**
   * @brief Get the previous nodes of the given node.
   *
   * @param node A node in the graph.
   * @return A vector of previous nodes.
   */
  virtual std::vector<NodeType> get_previous_nodes(const NodeType& node) const = 0;

  /**
   * @brief Set the context.
   *
   * @param context The context.
   */
  virtual void context(void* context) { context_ = context; }
  /**
   * @brief Get the context.
   *
   * @return The context.
   */
  virtual void* context() const { return context_; }

  /**
   * @brief Remove a node (and all its edges) from the graph.
   *
   * @param node The node to remove.
   */
  virtual void remove_node(const NodeType& node) = 0;

 protected:
  void* context_ = nullptr;  ///< The context.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_GRAPH_HPP */
