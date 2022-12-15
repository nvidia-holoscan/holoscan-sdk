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

#ifndef HOLOSCAN_CORE_GRAPHS_FLOW_GRAPH_HPP
#define HOLOSCAN_CORE_GRAPHS_FLOW_GRAPH_HPP

#include <vector>
#include <unordered_map>

#include "../graph.hpp"

namespace holoscan {

class FlowGraph : public Graph {
 public:
  using Graph::Graph;
  ~FlowGraph() override = default;

  void add_operator(const NodeType& op) override;
  void add_flow(const NodeType& op_u, const NodeType& op_v, const EdgeDataType& port_map) override;

  std::optional<EdgeDataType> get_port_map(const NodeType& op_u, const NodeType& op_v) override;

  bool is_root(const NodeType& op) override;

  bool is_leaf(const NodeType& op) override;

  std::vector<NodeType> get_root_operators() override;

  std::vector<NodeType> get_operators() override;

  std::vector<NodeType> get_next_operators(const NodeType& op) override;

 private:
  std::unordered_map<NodeType, std::unordered_map<NodeType, EdgeDataType>> succ_;
  std::unordered_map<NodeType, std::unordered_map<NodeType, EdgeDataType>> pred_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_GRAPHS_FLOW_GRAPH_HPP */
