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

#include "holoscan/core/graphs/flow_graph.hpp"

namespace holoscan {

using NodeType = std::shared_ptr<Operator>;
using EdgeDataElementType = std::unordered_map<std::string, std::set<std::string, std::less<>>>;
using EdgeDataType = std::shared_ptr<EdgeDataElementType>;

void FlowGraph::add_operator(const NodeType& op) {
  (void)op;
  if (succ_.find(op) == succ_.end()) {
    if (!op) {
      HOLOSCAN_LOG_ERROR("Calling add_operator() with nullptr");
      return;
    }
    succ_[op] = std::unordered_map<NodeType, EdgeDataType>();
    pred_[op] = std::unordered_map<NodeType, EdgeDataType>();
  }
}

void FlowGraph::add_flow(const NodeType& op_u, const NodeType& op_v, const EdgeDataType& port_map) {
  if (succ_.find(op_u) == succ_.end()) {
    if (!op_u) {
      HOLOSCAN_LOG_ERROR("Calling add_flow() with nullptr (op_u is nullptr)");
      return;
    }
    succ_[op_u] = std::unordered_map<NodeType, EdgeDataType>();
    pred_[op_u] = std::unordered_map<NodeType, EdgeDataType>();
  }
  if (succ_.find(op_v) == succ_.end()) {
    if (!op_v) {
      HOLOSCAN_LOG_ERROR("Calling add_flow() with nullptr (op_v is nullptr)");
      return;
    }
    succ_[op_v] = std::unordered_map<NodeType, EdgeDataType>();
    pred_[op_v] = std::unordered_map<NodeType, EdgeDataType>();
  }

  auto it_edgedata = succ_[op_u].find(op_v);
  if (it_edgedata != succ_[op_u].end()) {
    const auto& datadict = it_edgedata->second;
    if (port_map) {
      for (auto& [key, value] : *port_map) { datadict->insert({key, value}); }
    }
    succ_[op_u][op_v] = datadict;
    pred_[op_v][op_u] = datadict;
  } else {
    auto datadict = std::make_shared<EdgeDataElementType>();

    if (port_map) {
      for (auto& [key, value] : *port_map) { datadict->insert({key, value}); }
    }
    succ_[op_u][op_v] = datadict;
    pred_[op_v][op_u] = datadict;
  }
}

std::optional<EdgeDataType> FlowGraph::get_port_map(const NodeType& op_u, const NodeType& op_v) {
  auto it_u = succ_.find(op_u);
  if (it_u == succ_.end()) { return std::nullopt; }
  auto it_v = it_u->second.find(op_v);
  if (it_v == it_u->second.end()) { return std::nullopt; }
  return it_v->second;
}

bool FlowGraph::is_root(const NodeType& op) {
  if (auto it_pred = pred_.find(op); it_pred->second.empty()) { return true; }

  return false;
}

bool FlowGraph::is_leaf(const NodeType& op) {
  if (auto it_succ = succ_.find(op); it_succ->second.empty()) { return true; }
  return false;
}

std::vector<NodeType> FlowGraph::get_root_operators() {
  std::vector<NodeType> roots;
  for (const auto& [op, _] : pred_) {
    if (is_root(op)) { roots.push_back(op); }
  }
  return roots;
}

std::vector<NodeType> FlowGraph::get_operators() {
  std::vector<NodeType> ops;
  for (const auto& [op, _] : succ_) { ops.push_back(op); }
  return ops;
}

std::vector<NodeType> FlowGraph::get_next_operators(const NodeType& op) {
  std::vector<NodeType> ops;
  auto it_succ = succ_.find(op);
  if (it_succ == succ_.end()) { return ops; }
  for (const auto& [op_next, _] : it_succ->second) { ops.push_back(op_next); }
  return ops;
}

}  // namespace holoscan
