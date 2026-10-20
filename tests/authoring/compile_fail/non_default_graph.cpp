/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/register_graph.hpp>

// Exercise constructor-shape rejection in AUTHORING_COMPILE_FAIL.non_default_graph.
class NonDefaultGraph final : public holoscan::Graph {
 public:
  explicit NonDefaultGraph(int value) noexcept : value_(value) {}

 private:
  int value_{};
};

// REQ: AUTHOR-020
HOLOSCAN_REGISTER_GRAPH(NonDefaultGraph, "holoscan.test.NonDefaultGraph");

int main() {
  return 0;
}
