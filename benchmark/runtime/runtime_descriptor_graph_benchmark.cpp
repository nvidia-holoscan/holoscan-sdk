/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark/benchmark.h>
#include <algorithm>
#include <memory>

#include "support/borrowed_descriptor_graph.hpp"

namespace holoscan::test::borrowed_descriptor {
namespace {

// REQ: OPERATOR-056
// REQ: SENSOR-IO-011
void BM_DescriptorGraph(benchmark::State& state) {
  const auto mode = static_cast<Mode>(state.range(0));
  const bool transport = state.range(1) != 0;
  auto run = [&](bool count) -> std::shared_ptr<Probe> {
    auto probe = std::make_shared<Probe>(512);
    probe->count_allocations = count;
    const auto plan = make_plan(probe, mode, transport);
    if (!plan.ok()) {
      state.SkipWithError("descriptor graph compilation failed");
      return {};
    }
    auto session = run_async(plan);
    const bool complete = probe->wait();
    session.request_stop();
    session.wait();
    if (!complete || !probe->valid.load() || session.termination() != RunTermination::kStopped) {
      state.SkipWithError("descriptor graph failed correctness or bounded progress");
      return {};
    }
    return probe;
  };
  // Worker-thread C++ new counts are collected in a separate, untimed graph run.
  auto counts = run(true);
  if (!counts)
    return;
  if (mode != Mode::kOwning &&
      (counts->receive_allocations != 0 || counts->metadata_allocations != 0)) {
    state.SkipWithError("borrowed descriptor handling allocated on worker lane");
    return;
  }
  const double calls = static_cast<double>(counts->calls);
  state.counters["receive_allocations_per_hop"] = counts->receive_allocations / calls;
  state.counters["metadata_allocations_per_hop"] = counts->metadata_allocations / calls;
  state.counters["emit_allocations_per_hop"] = counts->emit_allocations / calls;
  state.counters["callback_allocations_per_hop"] = counts->callback_allocations / calls;
  for (auto _ : state) {
    auto timed = run(false);
    if (!timed)
      break;
    // Manual time is mean transform compute duration, not graph setup/teardown or throughput.
    state.SetIterationTime(static_cast<double>(timed->callback_ns) / calls / 1e9);
    std::sort(timed->latency_ns.begin(), timed->latency_ns.end());
    state.counters["source_to_sink_p50_ns"] = timed->latency_ns[timed->limit / 2];
    state.counters["source_to_sink_p95_ns"] = timed->latency_ns[timed->limit * 95 / 100];
    state.counters["source_to_sink_p99_ns"] = timed->latency_ns[timed->limit * 99 / 100];
    state.counters["messages_per_run"] = static_cast<double>(timed->limit);
  }
}

BENCHMARK(BM_DescriptorGraph)
    ->Args({0, 0})
    ->Args({1, 0})
    ->Args({2, 0})
    ->Args({0, 1})
    ->Args({1, 1})
    ->Args({2, 1})
    ->ArgNames({"mode", "transport"})
    ->UseManualTime()
    ->Iterations(1)
    ->Repetitions(3);

}  // namespace
}  // namespace holoscan::test::borrowed_descriptor
