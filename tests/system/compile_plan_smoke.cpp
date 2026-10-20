/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke: holoscan::compile() accepts a minimal single-vertex graph and
// produces a non-empty JSON execution plan with at least one execution
// partition; the resulting GraphView contains the expected vertex.

// Topology: graph → SmokeOperator("source")  (one vertex, no edges)

#include <cstdint>
#include <iostream>

#include <holoscan/core/compile.hpp>

namespace {

// Minimal output-only operator; used as the sole vertex to keep the graph
// simple enough to test compilation in isolation.
class SmokeOperator final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
  holoscan::Output<std::uint32_t> output;
};

}  // namespace

// REQ: COMPILE-001
int main() {
  holoscan::Graph graph{"compile-plan-smoke"};
  (void)graph.op<SmokeOperator>("source");
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok() || plan.json().empty() || plan.graph().vertices().size() != 1U ||
      plan.execution_partitions().empty()) {
    // Exit 1: compilation failed, JSON empty, vertex missing, or no execution partitions
    return 1;
  }
  std::cout << plan.json() << '\n';
  return 0;
}
