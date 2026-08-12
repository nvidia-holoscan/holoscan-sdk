/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <holoscan/core/graph.hpp>

// REQ: GRAPH-008
int main() {
  holoscan::Graph graph;
  holoscan::Output<std::uint32_t> output{"output"};
  graph.configure(output).tensor_allocation(
      holoscan::TensorAllocationBounds{.max_rank = 1, .max_byte_span = 4});
}
