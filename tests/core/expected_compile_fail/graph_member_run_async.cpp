/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/graph.hpp>

// REQ: GRAPH-001
int main() {
  holoscan::Graph graph;
  (void)graph.run_async();
}
