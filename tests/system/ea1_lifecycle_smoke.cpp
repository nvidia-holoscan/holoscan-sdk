/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke: the lifecycle kernel executes startup and shutdown for a two-node
// dependency chain across all eight lifecycle stages, proving that
// build_lifecycle_stage_graph() produces a correct stage-ordering graph and
// that the execute_lifecycle_{startup,shutdown}() functions fire callbacks in
// stable_order sequence (not insertion order).

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/lifecycle_kernel.hpp"

namespace {

using holoscan::LifecycleStage;
using holoscan::LifecycleStatus;
using holoscan::LifecycleTransitionPhase;
using holoscan::detail::LifecycleCallbackBinding;
using holoscan::detail::LifecycleExecutionDisposition;
using holoscan::detail::LifecycleKernelCapacity;
using holoscan::detail::LifecycleNodeId;
using holoscan::detail::LifecycleStageGraphInput;

[[nodiscard]] constexpr holoscan::LifecycleMethodId method_for(LifecycleStage stage) noexcept {
  return static_cast<holoscan::LifecycleMethodId>(static_cast<std::uint8_t>(stage) + 1U);
}

// One callback invocation recorded by the probe: which node fired, at which
// lifecycle stage, and in which transition phase (startup vs. shutdown).
struct TraceEntry {
  LifecycleNodeId node{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  LifecycleTransitionPhase phase{LifecycleTransitionPhase::kNone};

  friend bool operator==(const TraceEntry&, const TraceEntry&) = default;
};

// Fixed-capacity log of TraceEntries.  Capacity (16) comfortably covers the
// 8 callbacks that startup+shutdown together produce for this two-node graph.
struct Trace {
  std::array<TraceEntry, 16U> entries{};
  std::size_t size{};
};

// Callback context: identifies which node owns this probe and where to append
// recorded entries.  Passed as void* self to the lifecycle callback.
struct Probe {
  LifecycleNodeId node{};
  Trace* trace{};
};

[[nodiscard]] LifecycleStatus record(void* self, LifecycleStage stage,
                                     LifecycleTransitionPhase phase) noexcept {
  auto& probe = *static_cast<Probe*>(self);
  if (probe.trace == nullptr || probe.trace->size == probe.trace->entries.size()) {
    return LifecycleStatus::kFatalFailure;
  }
  probe.trace->entries[probe.trace->size++] = {
      .node = probe.node,
      .stage = stage,
      .phase = phase,
  };
  return LifecycleStatus::kOk;
}

// Builds the LifecycleStageGraphInput for a two-node chain.  Nodes are
// deliberately inserted out of stable_order to verify that the kernel
// resolves canonical ordering rather than iterating by insertion order.
// Each stage has a direct edge: first → second.
[[nodiscard]] LifecycleStageGraphInput make_input() {
  LifecycleStageGraphInput input;
  input.scopes = {{.scope_id = 1U}};
  // Deliberately author the nodes out of order so the trace proves canonical
  // stage ordering rather than insertion-order traversal.
  input.nodes = {
      {.node_id = 2U, .scope_id = 1U, .stable_name = "second", .stable_order = 2U},
      {.node_id = 1U, .scope_id = 1U, .stable_name = "first", .stable_order = 1U},
  };
  input.callbacks = {
      {.node_id = 1U,
       .stage = LifecycleStage::kConfigure,
       .required = true,
       .method = method_for(LifecycleStage::kConfigure)},
      {.node_id = 2U,
       .stage = LifecycleStage::kAllocate,
       .required = true,
       .method = method_for(LifecycleStage::kAllocate)},
      {.node_id = 1U,
       .stage = LifecycleStage::kStart,
       .required = true,
       .method = method_for(LifecycleStage::kStart)},
      {.node_id = 2U,
       .stage = LifecycleStage::kStart,
       .required = true,
       .method = method_for(LifecycleStage::kStart)},
      {.node_id = 1U,
       .stage = LifecycleStage::kStop,
       .required = true,
       .method = method_for(LifecycleStage::kStop)},
      {.node_id = 2U,
       .stage = LifecycleStage::kStop,
       .required = true,
       .method = method_for(LifecycleStage::kStop)},
      {.node_id = 1U,
       .stage = LifecycleStage::kRelease,
       .required = true,
       .method = method_for(LifecycleStage::kRelease)},
      {.node_id = 2U,
       .stage = LifecycleStage::kRelease,
       .required = true,
       .method = method_for(LifecycleStage::kRelease)},
  };
  for (std::uint8_t stage = 0U; stage < 8U; ++stage) {
    const auto value = static_cast<LifecycleStage>(stage);
    input.direct_edges.push_back({.declaration_scope_id = 1U,
                                  .before_node_id = 1U,
                                  .before_stage = value,
                                  .after_node_id = 2U,
                                  .after_stage = value});
  }
  return input;
}

// Wires each node/stage pair to the shared record() callback via its Probe.
// Note: node 1 has kConfigure but not kAllocate; node 2 has kAllocate but not
// kConfigure — asymmetric bindings are intentional to exercise sparse callback maps.
[[nodiscard]] std::vector<LifecycleCallbackBinding> make_bindings(Probe& first, Probe& second) {
  return {
      {.node_id = 1U, .stage = LifecycleStage::kConfigure, .self = &first, .callback = &record},
      {.node_id = 2U, .stage = LifecycleStage::kAllocate, .self = &second, .callback = &record},
      {.node_id = 1U, .stage = LifecycleStage::kStart, .self = &first, .callback = &record},
      {.node_id = 2U, .stage = LifecycleStage::kStart, .self = &second, .callback = &record},
      {.node_id = 1U, .stage = LifecycleStage::kStop, .self = &first, .callback = &record},
      {.node_id = 2U, .stage = LifecycleStage::kStop, .self = &second, .callback = &record},
      {.node_id = 1U, .stage = LifecycleStage::kRelease, .self = &first, .callback = &record},
      {.node_id = 2U, .stage = LifecycleStage::kRelease, .self = &second, .callback = &record},
  };
}

}  // namespace

// REQ: LIFECYCLE-001
// REQ: LIFECYCLE-003
// REQ: LIFECYCLE-004
int main() {
  // [scope 1]  node 1 (first, stable_order=1) ──► node 2 (second, stable_order=2)
  //            for every lifecycle stage (kConfigure … kRelease)
  //
  // Expected startup trace:  first.kConfigure, second.kAllocate,
  //                          first.kStart, second.kStart
  // Expected shutdown trace: first.kStop, second.kStop,
  //                          first.kRelease, second.kRelease
  const auto input = make_input();
  const LifecycleKernelCapacity capacity{
      .max_scopes = 1U,
      .max_nodes = 2U,
      .max_callbacks = 8U,
      .max_dependency_records = 0U,
      .max_direct_edges = 8U,
      .max_completions = 12U,
  };
  const auto graph = holoscan::detail::build_lifecycle_stage_graph(input, capacity);
  if (!graph) {
    return 1;  // exit 1: lifecycle stage graph must build successfully from the input
  }

  Trace trace;
  Probe first{.node = 1U, .trace = &trace};
  Probe second{.node = 2U, .trace = &trace};
  const auto bindings = make_bindings(first, second);
  auto state = holoscan::detail::make_lifecycle_kernel_state(*graph);
  const auto startup = holoscan::detail::execute_lifecycle_startup(*graph, bindings, state);
  if (startup.disposition != LifecycleExecutionDisposition::kCompleted || !startup.steady_state ||
      state.completed.size() != 12U) {
    // Exit 2: startup must complete, reach steady state, and record 12 internal completions
    return 2;
  }
  const auto shutdown = holoscan::detail::execute_lifecycle_shutdown(*graph, bindings, state);
  if (shutdown.disposition != LifecycleExecutionDisposition::kCompleted || shutdown.steady_state ||
      trace.size != 8U) {
    // Exit 3: shutdown must complete, must NOT reach steady state, and all 8 callbacks must fire
    return 3;
  }

  const std::array expected_startup{
      TraceEntry{.node = 1U,
                 .stage = LifecycleStage::kConfigure,
                 .phase = LifecycleTransitionPhase::kStartupForward},
      TraceEntry{.node = 2U,
                 .stage = LifecycleStage::kAllocate,
                 .phase = LifecycleTransitionPhase::kStartupForward},
      TraceEntry{.node = 1U,
                 .stage = LifecycleStage::kStart,
                 .phase = LifecycleTransitionPhase::kStartupForward},
      TraceEntry{.node = 2U,
                 .stage = LifecycleStage::kStart,
                 .phase = LifecycleTransitionPhase::kStartupForward},
  };
  std::size_t index = 0U;
  for (const auto& expected : expected_startup) {
    if (trace.entries[index++] != expected) {
      return 4;  // exit 4: startup callbacks must fire in canonical stage order
    }
  }
  for (const LifecycleStage stage : {LifecycleStage::kStop, LifecycleStage::kRelease}) {
    for (const LifecycleNodeId node : {1U, 2U}) {
      if (trace.entries[index++] !=
          TraceEntry{
              .node = node, .stage = stage, .phase = LifecycleTransitionPhase::kShutdownTeardown}) {
        return 5;  // exit 5: shutdown callbacks must fire in teardown order (Stop then Release)
      }
    }
  }
  return index == trace.size ? 0 : 6;  // exit 6: no extra trace entries beyond what was expected
}
