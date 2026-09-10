/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke: Graph::view() produces a consistent vertex/edge snapshot for both a
// directly-composed graph and a Graph subclass that re-exports ports.  Also
// verifies at compile-time that the base Graph type does not expose run-time
// APIs (compile/run/run_async).

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <holoscan/core/graph.hpp>

// Topology (direct):   source → target   (+1 imported HoloMQ topic vertex)
// Topology (reusable): [exported input] → worker → [exported output]

namespace {

// REQ: GRAPH-001
// REQ: GRAPH-002
// REQ: GRAPH-006
// REQ: GRAPH-007
// REQ: GRAPH-009
// REQ: GRAPH-011
// Minimal operator with one typed input and one typed output; used as a vertex
// inside both the direct graph and the ReusableSmokeGraph wrapper.
class SmokeOperator final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> input;
  holoscan::Output<std::uint32_t> output;
};

// A Graph subclass that wraps one SmokeOperator and re-exports its ports as
// the graph's own interface; tests that interface bindings appear in the view.
class ReusableSmokeGraph final : public holoscan::Graph {
 public:
  holoscan::Input<std::uint32_t> input{"input"};
  holoscan::Output<std::uint32_t> output{"output"};

 protected:
  void compose() override {
    auto worker = op<SmokeOperator>("worker");
    export_input(input, worker->input);
    export_output(output, worker->output);
  }
};

template <typename T>
concept HasCompileMember = requires(T& value) { value.compile(); };
template <typename T>
concept HasRunMember = requires(T& value) { value.run(); };
template <typename T>
concept HasRunAsyncMember = requires(T& value) { value.run_async(); };

static_assert(!HasCompileMember<holoscan::Graph>);
static_assert(!HasRunMember<holoscan::Graph>);
static_assert(!HasRunAsyncMember<holoscan::Graph>);

bool has_nonzero_byte(const auto& bytes) {
  return std::any_of(
      bytes.begin(), bytes.end(), [](std::byte byte) { return byte != std::byte{}; });
}

}  // namespace

int main() {
  holoscan::Graph direct{"direct"};
  auto source = direct.op<SmokeOperator>("source");
  auto target = direct.op<SmokeOperator>("target");
  (void)direct.add_flow(source->output, target->input);
  (void)direct.import_topic<std::uint32_t>("holomq://holoscan-graph-smoke");
  const holoscan::GraphView direct_view = direct.view();
  if (direct_view.vertices().size() != 3U || direct_view.edges().size() != 1U ||
      direct_view.identity().version != direct_view.generation() ||
      !has_nonzero_byte(direct_view.identity().graph.value) ||
      !has_nonzero_byte(direct_view.identity().hash.value)) {
    // Exit 1: direct graph has wrong vertex/edge count, or identity fields are zeroed
    return 1;
  }

  ReusableSmokeGraph reusable;
  const holoscan::GraphView reusable_view = reusable.view();
  if (reusable_view.vertices().size() != 3U || reusable_view.interface_bindings().size() != 2U) {
    // Exit 2: reusable graph has wrong vertex count or missing interface bindings
    return 2;
  }
  return 0;
}
