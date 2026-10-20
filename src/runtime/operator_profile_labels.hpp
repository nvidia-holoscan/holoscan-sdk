/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <string>
#include <string_view>

#include "../compiler/execution_plan_state.hpp"

namespace holoscan::detail {

[[nodiscard]] inline std::string operator_profile_escape_component(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const char ch : value) {
    if (ch == '\\' || ch == '.' || ch == ':' || ch == '/') {
      result.push_back('\\');
    }
    result.push_back(ch);
  }
  return result;
}

// Operator Centric Logging labels use the complete root-to-leaf graph instance path so
// reused nested graph instances remain distinguishable. Return the trailing-slash subgraph
// prefix, or an empty string when the owning scope is the root graph.
[[nodiscard]] inline std::string operator_profile_subgraph_prefix(const ExecutionPlanState& plan,
                                                                  GraphId graph_id) {
  const auto graph = std::find_if(plan.graph.graphs().begin(),
                                  plan.graph.graphs().end(),
                                  [graph_id](const auto& row) { return row.id == graph_id; });
  if (graph == plan.graph.graphs().end() || !graph->path.has_value()) {
    return {};
  }

  const auto& path = graph->path;
  std::string result;
  // The graph lookup and presence guard above prove this borrowed optional is engaged.
  for (const auto& segment :
       path.value().segments()) {  // NOLINT(bugprone-unchecked-optional-access)
    result += operator_profile_escape_component(segment);
    result.push_back('/');
  }
  return result;
}

[[nodiscard]] inline std::string operator_profile_operator_label(const ExecutionPlanState& plan,
                                                                 GraphId graph_id,
                                                                 std::string_view operator_name) {
  std::string result = operator_profile_subgraph_prefix(plan, graph_id);
  result += operator_profile_escape_component(operator_name);
  return result;
}

[[nodiscard]] inline std::string operator_profile_port_label(const ExecutionPlanState& plan,
                                                             GraphId graph_id,
                                                             std::string_view operator_name,
                                                             std::string_view port_name) {
  std::string result = operator_profile_operator_label(plan, graph_id, operator_name);
  result.push_back('.');
  result += operator_profile_escape_component(port_name);
  return result;
}

}  // namespace holoscan::detail
