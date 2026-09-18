/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace holoscan::detail {

/// @brief Current authored identity and declared parameter type for one Operator.
struct ParameterAuthoringTarget {
  VertexId vertex{};
  GraphId graph{};
  std::string path;
  const ParamSchemaDescriptor* schema{};
  const ParamTypeDescriptor* type{};
};

struct ParameterAuthoringGraph {
  GraphId graph{};
  std::string path;
};

struct ParameterAuthoringInventory {
  GraphId root_graph{};
  std::vector<ParameterAuthoringGraph> graphs;
  std::vector<ParameterAuthoringTarget> targets;
};

}  // namespace holoscan::detail
