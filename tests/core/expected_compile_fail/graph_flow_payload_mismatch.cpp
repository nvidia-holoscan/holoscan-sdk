/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <holoscan/core/graph.hpp>

// REQ: GRAPH-006
int main() {
  holoscan::Graph graph;
  holoscan::Output<std::uint32_t> output{"output"};
  holoscan::Input<std::uint64_t> input{"input"};
  (void)graph.add_flow(output, input);
}
