/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <core/graph_compiler_access.hpp>
#include <core/lifecycle_context_internal.hpp>
#include <core/lifecycle_context_test_access.hpp>
#include <core/lifecycle_graph_lowering.hpp>
#include <core/operator_spec_test_access.hpp>

namespace holoscan {
namespace {

inline constexpr LifecycleDependencyKey kScopedReady{"test.scoped.ready"};

// Operator that conditionally declares (producer=true) or requires (consumer=true) a kScopedReady
// lifecycle dependency during setup(). Controls scope-level dependency edges in lowering tests.
// Representative test: EmbeddedFactScopesRemainIndependent.
class ScopedLifecycleOperator final : public Operator<> {
 public:
  explicit ScopedLifecycleOperator(bool producer = false, bool consumer = false)
      : producer_(producer), consumer_(consumer) {}

  void setup(OperatorSpec& spec) override {
    if (producer_)
      spec.lifecycle().produces(LifecycleStage::kConfigure, kScopedReady);
    if (consumer_)
      spec.lifecycle().requires_dependency(LifecycleStage::kConfigure, kScopedReady);
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }

 private:
  bool producer_{};
  bool consumer_{};
};

// Single-node graph wrapping one ScopedLifecycleOperator; exposes node() for use as a handle in
// parent-scope ordering or cross-embed dependency scenarios.
// Representative test: ParentScopeMayOrderDescendantOperators.
class OneNodeGraph final : public Graph {
 public:
  explicit OneNodeGraph(bool producer = false, bool consumer = false)
      : producer_(producer), consumer_(consumer) {}

  [[nodiscard]] const OperatorHandle<ScopedLifecycleOperator>& node() const {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return node_.value();
  }

 protected:
  void compose() override { node_ = op<ScopedLifecycleOperator>("node", producer_, consumer_); }

 private:
  bool producer_{};
  bool consumer_{};
  std::optional<OperatorHandle<ScopedLifecycleOperator>> node_;
};

// Two-node graph with exactly one kScopedReady producer and one consumer, satisfying their
// dependency within the same embed scope. Each embedded instance resolves its fact independently.
// Representative test: EmbeddedFactScopesRemainIndependent.
class PairFactGraph final : public Graph {
 protected:
  void compose() override {
    (void)op<ScopedLifecycleOperator>("producer", true, false);
    (void)op<ScopedLifecycleOperator>("consumer", false, true);
  }
};

// Records a single lifecycle callback invocation: receiver id, lifecycle stage, and transition
// phase. Entries are compared element-wise against an expected sequence in the execution test.
struct ExecutionTraceEntry {
  std::uint8_t receiver{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  LifecycleTransitionPhase phase{LifecycleTransitionPhase::kNone};

  friend bool operator==(const ExecutionTraceEntry&, const ExecutionTraceEntry&) = default;
};

// Fixed-capacity log of up to 8 ExecutionTraceEntry records; populated during lifecycle kernel
// execution and inspected for correct stage ordering and phase assignment.
struct ExecutionTrace {
  std::array<ExecutionTraceEntry, 8U> entries{};
  std::size_t size{};
};

// Operator that registers callbacks for all four lifecycle stages and records each invocation into
// a shared ExecutionTrace via bind_trace(). Used to verify that the lowered stage graph drives
// callbacks in the correct topological and phase order.
// Representative test: PublicGraphLowersAndExecutesAgainstFreshCallbackReceivers.
class ExecutableLifecycleOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kConfigure, &ExecutableLifecycleOperator::configure)
        .stage(LifecycleStage::kStart, &ExecutableLifecycleOperator::start_stage)
        .stage(LifecycleStage::kStop, &ExecutableLifecycleOperator::stop_stage)
        .stage(LifecycleStage::kRelease, &ExecutableLifecycleOperator::release);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  LifecycleStatus configure(LifecycleContext& context) noexcept {
    return record(LifecycleStage::kConfigure, context);
  }
  LifecycleStatus start_stage(LifecycleContext& context) noexcept {
    return record(LifecycleStage::kStart, context);
  }
  LifecycleStatus stop_stage(LifecycleContext& context) noexcept {
    return record(LifecycleStage::kStop, context);
  }
  LifecycleStatus release(LifecycleContext& context) noexcept {
    return record(LifecycleStage::kRelease, context);
  }

  void bind_trace(std::uint8_t receiver, ExecutionTrace& trace) noexcept {
    receiver_ = receiver;
    trace_ = &trace;
  }

 private:
  LifecycleStatus record(LifecycleStage stage, LifecycleContext& context) noexcept {
    if (trace_ == nullptr || trace_->size == trace_->entries.size()) {
      return LifecycleStatus::kFatalFailure;
    }
    trace_->entries[trace_->size++] = {
        .receiver = receiver_,
        .stage = stage,
        .phase = context.transition_phase(),
    };
    return LifecycleStatus::kOk;
  }

  std::uint8_t receiver_{};
  ExecutionTrace* trace_{};
};

// Owns a fresh ExecutableLifecycleOperator and its OperatorSpec, bound to a trace and receiver id.
// Lets tests exercise lifecycle callbacks against operators not part of graph.compose().
struct ReboundLifecycleReceiver {
  explicit ReboundLifecycleReceiver(std::uint8_t receiver, ExecutionTrace& trace) {
    op.bind_trace(receiver, trace);
    op.setup(spec);
  }

  ExecutableLifecycleOperator op;
  OperatorSpec spec;
};

// Trampoline state passed to the lifecycle kernel: points to a ReboundLifecycleReceiver and the
// index of the specific callback within its OperatorSpec, enabling per-stage dispatch.
struct ReboundLifecycleCallback {
  ReboundLifecycleReceiver* receiver{};
  std::size_t callback_index{};
};

[[nodiscard]] LifecycleStatus invoke_rebound_lifecycle_callback(
    void* self, LifecycleStage, LifecycleTransitionPhase phase) noexcept {
  auto& binding = *static_cast<ReboundLifecycleCallback*>(self);
  detail::LifecycleContextState context_state{.phase = phase};
  LifecycleContext context = detail::LifecycleContextTestAccess::context(context_state);
  return detail::OperatorSpecTestAccess::invoke_lifecycle_callback(
      binding.receiver->spec, binding.callback_index, &binding.receiver->op, context);
}

// Convenience wrapper: snapshots the graph, lowers it to LifecycleGraphLowering, and asserts
// success. Avoids duplicating the snapshot+lower sequence in each test.
void lower(Graph& graph, detail::LifecycleGraphLowering& result) {
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_TRUE(snapshot);
  auto lowered = detail::lower_lifecycle_stage_graph_input(*snapshot);
  ASSERT_TRUE(lowered.has_value());
  result = std::move(lowered).value();
}

// REQ: LIFECYCLE-003
TEST(LifecycleGraphLowering, CanonicalSnapshotProducesDeterministicInput) {
  // Topology (insertion order is reversed in 'reverse'; canonical output must match 'forward'):
  //   a --(kStart)--> b
  Graph forward;
  auto forward_a = forward.op<ScopedLifecycleOperator>("a");
  auto forward_b = forward.op<ScopedLifecycleOperator>("b");
  forward.lifecycle().before(forward_a, forward_b, LifecycleStage::kStart);

  Graph reverse;
  auto reverse_b = reverse.op<ScopedLifecycleOperator>("b");
  auto reverse_a = reverse.op<ScopedLifecycleOperator>("a");
  reverse.lifecycle().before(reverse_a, reverse_b, LifecycleStage::kStart);

  detail::LifecycleGraphLowering lhs;
  detail::LifecycleGraphLowering rhs;
  lower(forward, lhs);
  lower(reverse, rhs);
  ASSERT_EQ(lhs.input.nodes.size(), 2U);
  ASSERT_EQ(rhs.input.nodes.size(), 2U);
  EXPECT_EQ(lhs.input.nodes[0].stable_name, rhs.input.nodes[0].stable_name);
  EXPECT_EQ(lhs.input.nodes[1].stable_name, rhs.input.nodes[1].stable_name);
  ASSERT_EQ(lhs.input.direct_edges.size(), 1U);
  ASSERT_EQ(rhs.input.direct_edges.size(), 1U);
  EXPECT_EQ(lhs.input.direct_edges[0].declaration_scope_id,
            rhs.input.direct_edges[0].declaration_scope_id);
  EXPECT_EQ(lhs.input.direct_edges[0].before_node_id, rhs.input.direct_edges[0].before_node_id);
  EXPECT_EQ(lhs.input.direct_edges[0].after_node_id, rhs.input.direct_edges[0].after_node_id);
}

// REQ: LIFECYCLE-003
TEST(LifecycleGraphLowering, EmbeddedFactScopesRemainIndependent) {
  // Valid topology (two independent embedded PairFactGraphs -- scopes never cross):
  //   [left scope: producer --> consumer]   [right scope: producer --> consumer]
  //
  // Invalid topology (kScopedReady spans two separate embed scopes -- no shared producer):
  //   [embed "producer": node(kScopedReady producer)]   [embed "consumer": node(consumer)]
  //   -> build_lifecycle_stage_graph returns kMissingProducer
  Graph valid;
  (void)valid.embed<PairFactGraph>("left");
  (void)valid.embed<PairFactGraph>("right");
  detail::LifecycleGraphLowering independent;
  lower(valid, independent);
  ASSERT_EQ(independent.input.producers.size(), 2U);
  ASSERT_EQ(independent.input.requirements.size(), 2U);
  EXPECT_NE(independent.input.producers[0].key.scope_id,
            independent.input.producers[1].key.scope_id);
  EXPECT_TRUE(detail::build_lifecycle_stage_graph(independent.input, independent.minimum_capacity));

  Graph invalid;
  (void)invalid.embed<OneNodeGraph>("producer", true, false);
  (void)invalid.embed<OneNodeGraph>("consumer", false, true);
  detail::LifecycleGraphLowering isolated;
  lower(invalid, isolated);
  auto rejected = detail::build_lifecycle_stage_graph(isolated.input, isolated.minimum_capacity);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, detail::LifecycleKernelErrorCode::kMissingProducer);
}

// REQ: LIFECYCLE-003
TEST(LifecycleGraphLowering, ParentScopeMayOrderDescendantOperators) {
  // Topology (parent graph orders two embedded single-node subgraphs for kStart):
  //   left.node --(parent-scope kStart)--> right.node
  //   (imported topic is non-operator and is ignored by lifecycle lowering)
  Graph parent;
  auto left = parent.embed<OneNodeGraph>("left");
  auto right = parent.embed<OneNodeGraph>("right");
  parent.lifecycle().before(left->node(), right->node(), LifecycleStage::kStart);
  (void)parent.import_topic<std::uint32_t>("holomq://lifecycle-boundary");

  detail::LifecycleGraphLowering lowered;
  lower(parent, lowered);
  ASSERT_EQ(lowered.input.nodes.size(), 2U);
  ASSERT_EQ(lowered.input.direct_edges.size(), 1U);
  EXPECT_TRUE(detail::build_lifecycle_stage_graph(lowered.input, lowered.minimum_capacity));

  const auto& edge = lowered.input.direct_edges[0];
  const auto before =
      std::find_if(lowered.input.nodes.begin(), lowered.input.nodes.end(), [&](const auto& node) {
        return node.node_id == edge.before_node_id;
      });
  ASSERT_NE(before, lowered.input.nodes.end());
  lowered.input.direct_edges[0].declaration_scope_id = before->scope_id;
  auto rejected = detail::build_lifecycle_stage_graph(lowered.input, lowered.minimum_capacity);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, detail::LifecycleKernelErrorCode::kCrossScopeDependency);
}

// REQ: LIFECYCLE-001
// REQ: LIFECYCLE-002
// REQ: LIFECYCLE-003
// REQ: LIFECYCLE-004
TEST(LifecycleGraphLowering, PublicGraphLowersAndExecutesAgainstFreshCallbackReceivers) {
  // Topology:
  //   first --(kConfigure, kStart, kStop, kRelease)--> second
  Graph graph;
  auto second = graph.op<ExecutableLifecycleOperator>("second");
  auto first = graph.op<ExecutableLifecycleOperator>("first");
  for (const LifecycleStage stage : {LifecycleStage::kConfigure,
                                     LifecycleStage::kStart,
                                     LifecycleStage::kStop,
                                     LifecycleStage::kRelease}) {
    graph.lifecycle().before(first, second, stage);
  }

  // Key setup: two ExecutableLifecycleOperators in dependency order (first before second across
  //   all four stages); the stage graph is compiled from the lowered input; fresh
  //   ReboundLifecycleReceivers supply callbacks without a live operator runtime.
  detail::LifecycleGraphLowering lowered;
  lower(graph, lowered);
  const auto stage_graph =
      detail::build_lifecycle_stage_graph(lowered.input, lowered.minimum_capacity);
  ASSERT_TRUE(stage_graph);

  const auto node_for = [&](std::string_view name) {
    const auto node = std::find_if(
        lowered.input.nodes.begin(), lowered.input.nodes.end(), [&](const auto& candidate) {
          return candidate.stable_name == name;
        });
    EXPECT_NE(node, lowered.input.nodes.end());
    return node == lowered.input.nodes.end() ? detail::LifecycleNodeId{} : node->node_id;
  };

  ExecutionTrace trace;
  ReboundLifecycleReceiver first_receiver{1U, trace};
  ReboundLifecycleReceiver second_receiver{2U, trace};
  const std::array receiver_nodes{
      std::pair{&first_receiver, node_for("first")},
      std::pair{&second_receiver, node_for("second")},
  };
  std::array<ReboundLifecycleCallback, 8U> callback_states{};
  std::vector<detail::LifecycleCallbackBinding> bindings;
  bindings.reserve(callback_states.size());
  std::size_t callback_state_index = 0U;
  for (const auto& [receiver, node_id] : receiver_nodes) {
    const std::size_t callback_count =
        detail::OperatorSpecTestAccess::lifecycle_callback_count(receiver->spec);
    ASSERT_EQ(callback_count, 4U);
    for (std::size_t callback_index = 0U; callback_index < callback_count; ++callback_index) {
      const auto metadata =
          detail::OperatorSpecTestAccess::lifecycle_callback(receiver->spec, callback_index);
      const auto logical = std::find_if(
          stage_graph->callbacks.begin(), stage_graph->callbacks.end(), [&](const auto& candidate) {
            return candidate.node_id == node_id && candidate.stage == metadata.stage;
          });
      ASSERT_NE(logical, stage_graph->callbacks.end());
      EXPECT_EQ(logical->method, metadata.method);
      callback_states[callback_state_index] = {
          .receiver = receiver,
          .callback_index = callback_index,
      };
      bindings.push_back({.node_id = node_id,
                          .stage = metadata.stage,
                          .self = &callback_states[callback_state_index],
                          .callback = &invoke_rebound_lifecycle_callback});
      ++callback_state_index;
    }
  }
  ASSERT_EQ(callback_state_index, callback_states.size());

  // Key assertion: startup drives kConfigure then kStart in dependency order (1 before 2);
  //   shutdown drives kStop then kRelease in the same order; all 8 trace entries match exactly.
  auto state = detail::make_lifecycle_kernel_state(*stage_graph);
  const auto startup = detail::execute_lifecycle_startup(*stage_graph, bindings, state);
  ASSERT_EQ(startup.disposition, detail::LifecycleExecutionDisposition::kCompleted);
  const auto shutdown = detail::execute_lifecycle_shutdown(*stage_graph, bindings, state);
  ASSERT_EQ(shutdown.disposition, detail::LifecycleExecutionDisposition::kCompleted);

  const std::array expected{
      ExecutionTraceEntry{
          1U, LifecycleStage::kConfigure, LifecycleTransitionPhase::kStartupForward},
      ExecutionTraceEntry{
          2U, LifecycleStage::kConfigure, LifecycleTransitionPhase::kStartupForward},
      ExecutionTraceEntry{1U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward},
      ExecutionTraceEntry{2U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward},
      ExecutionTraceEntry{1U, LifecycleStage::kStop, LifecycleTransitionPhase::kShutdownTeardown},
      ExecutionTraceEntry{2U, LifecycleStage::kStop, LifecycleTransitionPhase::kShutdownTeardown},
      ExecutionTraceEntry{
          1U, LifecycleStage::kRelease, LifecycleTransitionPhase::kShutdownTeardown},
      ExecutionTraceEntry{
          2U, LifecycleStage::kRelease, LifecycleTransitionPhase::kShutdownTeardown},
  };
  EXPECT_EQ(trace.size, expected.size());
  EXPECT_EQ(trace.entries, expected);
}

}  // namespace
}  // namespace holoscan
