/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_GRAPHS_GRAPH_HPP
#define HOLOSCAN_CORE_GRAPHS_GRAPH_HPP

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

/**
 * @brief Abstract base class for all graphs.
 */
class Graph {
 public:
  using NodeType = std::shared_ptr<Operator>;
  using EdgeDataElementType = std::unordered_map<std::string, std::set<std::string, std::less<>>>;
  using EdgeDataType = std::shared_ptr<EdgeDataElementType>;

  Graph() = default;
  virtual ~Graph() = default;

  /**
   * @brief Add a node to the graph.
   *
   * @param op The node to add.
   */
  virtual void add_operator(const NodeType& op) = 0;
  /**
   * @brief Add an edge to the graph.
   *
   * @param op_u A source operator.
   * @param op_v A destination operator.
   * @param port_map A map from the source operator's port name to the destination operator's port
   * name(s).
   */
  virtual void add_flow(const NodeType& op_u, const NodeType& op_v,
                        const EdgeDataType& port_map) = 0;

  /**
   * @brief Get a mapping from the source operator's port name to the destination operator's port
   * name(s).
   *
   * @param op_u A source operator.
   * @param op_v A destination operator.
   * @return A map from the source operator's port name to the destination operator's port name(s).
   */
  virtual std::optional<EdgeDataType> get_port_map(const NodeType& op_u, const NodeType& op_v) = 0;

  /**
   * @brief Check if the operator is a root operator.
   *
   * @param op A node in the graph.
   * @return true if the operator is a root operator.
   */
  virtual bool is_root(const NodeType& op) = 0;

  /**
   * @brief Check if the operator is a leaf operator.
   *
   * @param op A node in the graph.
   * @return true if the operator is a leaf operator.
   */
  virtual bool is_leaf(const NodeType& op) = 0;

  /**
   * @brief Get all root operators.
   *
   * @return A vector of root operators.
   */
  virtual std::vector<NodeType> get_root_operators() = 0;

  /**
   * @brief Get all operators.
   *
   * @return A vector of all operators.
   */
  virtual std::vector<NodeType> get_operators() = 0;

  /**
   * @brief Get the next operators of the given operator.
   *
   * @param op A node in the graph.
   * @return A vector of next operators.
   */
  virtual std::vector<NodeType> get_next_operators(const NodeType& op) = 0;

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
  virtual void* context() { return context_; }

 protected:
  void* context_ = nullptr;  ///< The context.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_GRAPHS_GRAPH_HPP */
