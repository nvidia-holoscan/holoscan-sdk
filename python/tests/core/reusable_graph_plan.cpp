/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "pipeline_test_operators.hpp"

namespace {

using holoscan::python::test::CounterSource;
using holoscan::python::test::CounterSourceParams;
using holoscan::python::test::DoubleIt;
using holoscan::python::test::ScalePairGraph;
using holoscan::python::test::ScalePairGraphParams;
using holoscan::python::test::Sink;
using holoscan::python::test::WrappedScaleGraph;
using holoscan::python::test::WrappedScaleGraphParams;

// Re-export a configurable chain of DoubleIt operators for independent C++ parity plans.
// Representative test: test_reusable_graph_plan_matches_independent_cpp_authoring.
class DoublerGraph final : public holoscan::Graph {
 public:
  explicit DoublerGraph(std::uint32_t copies) : copies_(copies) {}

  [[nodiscard]] holoscan::ClockKey clock_import() const noexcept { return clock_import_; }

  holoscan::Input<std::uint32_t> input{"in"};
  holoscan::Output<std::uint32_t> output{"out"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    std::vector<holoscan::OperatorHandle<DoubleIt>> members;
    members.reserve(copies_);
    for (std::uint32_t index = 0; index < copies_; ++index) {
      members.push_back(op<DoubleIt>("doubler-" + std::to_string(index)));
      if (index == 0U) {
        export_input(input, members.back()->in);
      } else {
        add_flow(members[index - 1U]->out, members.back()->in);
      }
    }
    export_output(output, members.back()->out);
  }

 private:
  std::uint32_t copies_{};
  holoscan::ClockKey clock_import_;
};

// Re-export a nested DoublerGraph to exercise reusable-graph composition and clock binding.
// Representative test: test_reusable_graph_plan_matches_independent_cpp_authoring.
class WrappedDoublerGraph final : public holoscan::Graph {
 public:
  explicit WrappedDoublerGraph(std::uint32_t copies) : copies_(copies) {}

  [[nodiscard]] holoscan::ClockKey clock_import() const noexcept { return clock_import_; }

  holoscan::Input<std::uint32_t> input{"in"};
  holoscan::Output<std::uint32_t> output{"out"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto inner = embed<DoublerGraph>("inner", copies_);
    bind_clock(inner, inner->clock_import(), clock_import_);
    export_input(input, inner->input);
    export_output(output, inner->output);
  }

 private:
  std::uint32_t copies_{};
  holoscan::ClockKey clock_import_;
};

//   registered-reusable-graphs-parity
//   |-- source --+--> direct[DoublerGraph] ------------> direct-sink
//   |            `--> wrapped[WrappedDoublerGraph] ----> wrapped-sink
//   `-- runtime-clock --> direct, wrapped
holoscan::ExecutionPlan reusable_graph_plan(bool bind_direct_clock) {
  holoscan::Graph graph{"reusable-graphs-parity"};
  auto source = graph.op<CounterSource>("source");
  source.set_params(CounterSourceParams{.limit = 4U});
  const auto direct = graph.embed<DoublerGraph>("direct", 1U);
  const auto wrapped = graph.embed<WrappedDoublerGraph>("wrapped", 2U);
  const auto direct_sink = graph.op<Sink>("direct-sink");
  const auto wrapped_sink = graph.op<Sink>("wrapped-sink");
  graph.add_flow(source->out, direct->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(source->out, wrapped->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(direct->output, direct_sink->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(wrapped->output, wrapped_sink->in, holoscan::ConnectionOptions{.queue_depth = 4U});

  const holoscan::ClockKey runtime_clock =
      graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  if (bind_direct_clock) {
    graph.bind_clock(direct, direct->clock_import(), runtime_clock);
  }
  graph.bind_clock(wrapped, wrapped->clock_import(), runtime_clock);
  graph.region("stage").add(direct).add(wrapped).constrain(
      holoscan::RegionConstraint::bounded_memory());
  return holoscan::compile(graph);
}

//   registered-reusable-graphs-parity
//   |-- source --+--> direct[ScalePairGraph] ----------> direct-sink
//   |            `--> wrapped[WrappedScaleGraph] ------> wrapped-sink
//   `-- runtime-clock --> direct, wrapped
holoscan::ExecutionPlan registered_reusable_graph_plan() {
  holoscan::Graph graph{"registered-reusable-graphs-parity"};
  auto source = graph.op<CounterSource>("source");
  source.set_params(CounterSourceParams{.limit = 4U});
  const auto direct = graph.embed<ScalePairGraph>("direct", ScalePairGraphParams{.factor = 3U});
  const auto wrapped =
      graph.embed<WrappedScaleGraph>("wrapped", WrappedScaleGraphParams{.factor = 5U});
  const auto direct_sink = graph.op<Sink>("direct-sink");
  const auto wrapped_sink = graph.op<Sink>("wrapped-sink");
  graph.add_flow(source->out, direct->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(source->out, wrapped->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(direct->output, direct_sink->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(wrapped->output, wrapped_sink->in, holoscan::ConnectionOptions{.queue_depth = 4U});

  const holoscan::ClockKey runtime_clock =
      graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.bind_clock(direct, direct->clock_import(), runtime_clock);
  graph.bind_clock(wrapped, wrapped->clock_import(), runtime_clock);
  graph.region("stage").add(direct).add(wrapped).constrain(
      holoscan::RegionConstraint::bounded_memory());
  return holoscan::compile(graph);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: python_core_reusable_graph_plan <bound|unbound|registered>\n";
    return 64;
  }
  const std::string_view mode{argv[1]};
  if (mode == "registered") {
    std::cout << registered_reusable_graph_plan().json();
    return 0;
  }
  if (mode != "bound" && mode != "unbound") {
    std::cerr << "unknown mode\n";
    return 64;
  }
  std::cout << reusable_graph_plan(mode == "bound").json();
  return 0;
}
