/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/register_graph.hpp>

struct OpaqueGraphParams {
  int value{};
};

// Exercise the opaque-Params rejection in AUTHORING_COMPILE_FAIL.graph_opaque_params.
class OpaqueParamsGraph final : public holoscan::Graph {
 public:
  using Params = OpaqueGraphParams;

  explicit OpaqueParamsGraph(const Params&) noexcept {}
};

// REQ: AUTHOR-020
HOLOSCAN_REGISTER_GRAPH(OpaqueParamsGraph, "holoscan.test.OpaqueParamsGraph");

int main() {
  return 0;
}
