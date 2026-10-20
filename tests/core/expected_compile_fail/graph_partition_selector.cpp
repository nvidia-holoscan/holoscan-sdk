/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/graph.hpp>

// REQ: GRAPH-001
// REQ: COMPILE-068
int main() {
  holoscan::Graph graph;
  graph.set_partition_algorithm("source_chain");
}
