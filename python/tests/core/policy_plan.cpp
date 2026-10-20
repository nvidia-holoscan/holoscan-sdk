/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "pipeline_test_operators.hpp"

namespace {

using holoscan::python::test::CounterSource;
using holoscan::python::test::CounterSourceParams;
using holoscan::python::test::DoubleIt;
using holoscan::python::test::Sink;

// Holds the three handles authored for every mode consumed by
// test_thread_placement_plan_matches_independent_cpp_authoring and its policy peers.
struct Pipeline {
  holoscan::OperatorHandle<CounterSource> source;
  holoscan::OperatorHandle<DoubleIt> doubler;
  holoscan::OperatorHandle<Sink> sink;
};

Pipeline add_pipeline(holoscan::Graph& graph) {
  //   source --uint32--> doubler --uint32--> sink
  auto source = graph.op<CounterSource>("source");
  source.set_params(CounterSourceParams{.limit = 4U});
  const auto doubler = graph.op<DoubleIt>("doubler");
  const auto sink = graph.op<Sink>("sink");
  graph.add_flow(source->out, doubler->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(doubler->out, sink->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  return Pipeline{source, doubler, sink};
}

holoscan::ExecutionPlan thread_placement_plan() {
  holoscan::Graph graph{"thread-placement"};
  const Pipeline pipeline = add_pipeline(graph);
  graph.partition("worker").add(pipeline.source, pipeline.doubler, pipeline.sink);
  holoscan::CompileOptions options;
  options.deployment.bind_thread_placement(
      "worker", holoscan::ThreadPlacement{.wake_mode = holoscan::WakeMode::kSleepingWaitset});
  return holoscan::compile(graph, std::move(options));
}

holoscan::ExecutionPlan support_thread_affinity_plan(std::uint32_t controller_cpu,
                                                     std::uint32_t pump_cpu) {
  holoscan::Graph graph{"support-thread-affinity"};
  (void)add_pipeline(graph);
  holoscan::CompileOptions options;
  options.deployment
      .bind_support_thread_affinity(holoscan::SupportThreadRole::kController,
                                    holoscan::CPUAffinity{controller_cpu})
      .bind_support_thread_affinity(holoscan::SupportThreadRole::kChannelPump,
                                    holoscan::CPUAffinity{pump_cpu});
  return holoscan::compile(graph, std::move(options));
}

std::optional<std::uint32_t> parse_cpu_id(std::string_view text) {
  std::uint32_t value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

holoscan::ExecutionPlan green_context_plan(const holoscan::CudaGreenContextCapability& capability) {
  holoscan::Graph graph{"green-context"};
  const Pipeline pipeline = add_pipeline(graph);
  graph.partition("critical").add(pipeline.source);
  graph.partition("background").add(pipeline.doubler, pipeline.sink);
  holoscan::CompileOptions options;
  options.deployment.bind_cuda_green_context("critical",
                                             holoscan::CudaGreenContextPlacement{
                                                 .device = capability.device,
                                                 .sm_count = capability.structured_sm_quantum,
                                             });
  return holoscan::compile(graph, std::move(options));
}

holoscan::ExecutionPlan automatic_partition_plan() {
  holoscan::Graph graph{"automatic-partitioning"};
  const Pipeline pipeline = add_pipeline(graph);
  graph.partition("anchor").add(pipeline.source);
  graph.auto_partition_remaining();
  return holoscan::compile(graph);
}

holoscan::ExecutionPlan include_all_plan() {
  holoscan::Graph graph{"include-all"};
  (void)add_pipeline(graph);
  graph.partition("whole-application").include_all();
  return holoscan::compile(graph);
}

holoscan::ExecutionPlan policy_plan() {
  using namespace std::chrono_literals;
  holoscan::Graph graph{"graph-policies"};
  const Pipeline pipeline = add_pipeline(graph);
  graph.region("critical")
      .add(pipeline.source)
      .add(pipeline.doubler)
      .constrain(holoscan::RegionConstraint::static_topology())
      .constrain(holoscan::RegionConstraint::bounded_memory())
      .constrain(holoscan::RegionConstraint::device_compatible_ops())
      .constrain(holoscan::RegionConstraint::latency_budget(250us))
      .constrain(holoscan::RegionConstraint::cuda_graph())
      .profile(holoscan::Profile::kRobotics);
  auto lifecycle = graph.lifecycle();
  lifecycle.before(pipeline.source, pipeline.doubler, holoscan::LifecycleStage::kConfigure)
      .allow_self_restart(pipeline.source)
      .allow_external_restart(pipeline.doubler)
      .allow_restart_request(pipeline.source, pipeline.doubler);
  return holoscan::compile(graph);
}

holoscan::ExecutionPlan partition_diagnostic_plan() {
  holoscan::Graph graph{"partition-diagnostic"};
  const Pipeline pipeline = add_pipeline(graph);
  graph.partition("incomplete").add(pipeline.source);
  return holoscan::compile(graph);
}

holoscan::ExecutionPlan region_diagnostic_plan() {
  holoscan::Graph graph{"region-diagnostic"};
  const Pipeline pipeline = add_pipeline(graph);
  graph.region("looser").add(pipeline.source).profile(holoscan::Profile::kResearch);
  return holoscan::compile(graph);
}

holoscan::ExecutionPlan tensor_diagnostic_plan() {
  holoscan::Graph graph{"tensor-device-diagnostic"};
  (void)add_pipeline(graph);
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "out",
      .device = holoscan::DeviceId{0},
  });
  return holoscan::compile(graph, std::move(options));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: python_core_policy_plan <mode> [controller-cpu pump-cpu]\n";
    return 64;
  }
  // Key setup: Select the independent C++ policy graph requested by the Python parity test.
  const std::string_view mode{argv[1]};
  if (mode != "support-thread" && argc != 2) {
    return 64;
  }
  holoscan::ExecutionPlan plan;
  if (mode == "thread") {
    plan = thread_placement_plan();
  } else if (mode == "support-thread") {
    if (argc != 4) {
      return 64;
    }
    const auto controller_cpu = parse_cpu_id(argv[2]);
    const auto pump_cpu = parse_cpu_id(argv[3]);
    if (!controller_cpu || !pump_cpu) {
      return 64;
    }
    plan = support_thread_affinity_plan(*controller_cpu, *pump_cpu);
  } else if (mode == "green") {
    const auto capability = holoscan::query_cuda_green_context_capability(holoscan::DeviceId{0});
    if (!capability || capability->structured_sm_quantum == 0U ||
        capability->max_structured_context_count < 2U) {
      return 77;
    }
    plan = green_context_plan(*capability);
  } else if (mode == "automatic") {
    plan = automatic_partition_plan();
  } else if (mode == "include-all") {
    plan = include_all_plan();
  } else if (mode == "policy") {
    plan = policy_plan();
  } else if (mode == "partition-diagnostic") {
    plan = partition_diagnostic_plan();
  } else if (mode == "region-diagnostic") {
    plan = region_diagnostic_plan();
  } else if (mode == "tensor-diagnostic") {
    plan = tensor_diagnostic_plan();
  } else {
    std::cerr << "unknown mode\n";
    return 64;
  }
  // Key assertion: Emit its canonical plan bytes for the corresponding Python assertion.
  std::cout << plan.json();
  return 0;
}
