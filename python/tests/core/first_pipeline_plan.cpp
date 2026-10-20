/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "pipeline_test_operators.hpp"

int main() {
  using holoscan::python::test::CounterSource;
  using holoscan::python::test::CounterSourceParams;
  using holoscan::python::test::DoubleIt;
  using holoscan::python::test::Sink;

  // Key setup: Author the independent C++ plan compared by the Python plan/session tests.
  //
  //   source --uint32--> doubler --uint32--> sink
  holoscan::Graph graph{"first-pipeline"};
  auto source = graph.op<CounterSource>("source");
  source.set_params(CounterSourceParams{.limit = 10U});
  const auto doubler = graph.op<DoubleIt>("doubler");
  const auto sink = graph.op<Sink>("sink");
  graph.add_flow(source->out, doubler->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(doubler->out, sink->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  // Key assertion: Emit the accepted canonical plan bytes for the Python parity assertion.
  std::cout << plan.json();
  return 0;
}
