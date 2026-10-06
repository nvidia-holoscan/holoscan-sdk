/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/lifecycle_kernel.hpp"

namespace holoscan::test {
namespace {

using detail::LifecycleCallbackBinding;
using detail::LifecycleExecutionDisposition;
using detail::LifecycleKernelCapacity;
using detail::LifecycleNodeId;
using detail::LifecycleStageGraphInput;

constexpr std::size_t kTraceCapacity = 256U;

// Records a single lifecycle callback invocation for ordered-sequence assertions.
// Supports equality comparison so expect_same_trace can compare full sequences.
struct TraceEntry {
  LifecycleNodeId node_id{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  LifecycleTransitionPhase phase{LifecycleTransitionPhase::kNone};

  friend bool operator==(const TraceEntry&, const TraceEntry&) = default;
};

// Fixed-capacity ordered log of TraceEntry records accumulated during a test run.
// Tests call clear() between startup and subsequent lifecycle operations.
struct CallbackTrace {
  std::array<TraceEntry, kTraceCapacity> entries{};
  std::size_t size{};

  void clear() noexcept { size = 0U; }
};

// Specifies one injected failure for a CallbackProbe. When enabled, the first call
// matching (stage, phase) after skipping matching_calls_to_skip occurrences returns status.
struct FailureRule {
  bool enabled{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  LifecycleTransitionPhase phase{LifecycleTransitionPhase::kNone};
  LifecycleStatus status{LifecycleStatus::kFatalFailure};
  std::size_t matching_calls_to_skip{};
};

// Test double for a single lifecycle node: appends every callback invocation to trace
// and optionally injects up to three failures at configurable (stage, phase) points.
struct CallbackProbe {
  LifecycleNodeId node_id{};
  CallbackTrace* trace{};
  std::array<FailureRule, 3> failures{};
};

// Captures raw LifecycleLogRecord events emitted by the executor so tests can assert
// log severity, operation fields, and phase tags without affecting control flow.
struct LifecycleLogTrace {
  std::array<detail::LifecycleLogRecord, 64U> records{};
  std::size_t size{};

  void clear() noexcept { size = 0U; }
};

// Maps a LifecycleStage ordinal to a required LifecycleMethodId for the test graph.
// Uses stage_index+1 to avoid the reserved zero method ID.
[[nodiscard]] constexpr LifecycleMethodId method_for(LifecycleStage stage) noexcept {
  return static_cast<LifecycleMethodId>(static_cast<std::uint8_t>(stage) + 1U);
}

// Log sink that appends each record to a LifecycleLogTrace for content assertions.
void record_lifecycle_log(void* self, const detail::LifecycleLogRecord& record) {
  auto& trace = *static_cast<LifecycleLogTrace*>(self);
  if (trace.size < trace.records.size()) {
    trace.records[trace.size++] = record;
  }
}

// Log sink that throws an int, confirming the executor swallows log-sink exceptions
// without altering the result or kernel state (used in LoggingEmits... test).
void throw_from_lifecycle_log(void*, const detail::LifecycleLogRecord&) {
  throw 1;
}

// LifecycleCallback implementation: appends a TraceEntry then checks FailureRule slots.
// Used as the callback field in every test LifecycleCallbackBinding.
[[nodiscard]] LifecycleStatus record_callback(void* self, LifecycleStage stage,
                                              LifecycleTransitionPhase phase) noexcept {
  auto& probe = *static_cast<CallbackProbe*>(self);
  if (probe.trace->size < probe.trace->entries.size()) {
    probe.trace->entries[probe.trace->size++] = {
        .node_id = probe.node_id,
        .stage = stage,
        .phase = phase,
    };
  }
  for (auto& failure : probe.failures) {
    if (failure.enabled && failure.stage == stage && failure.phase == phase) {
      if (failure.matching_calls_to_skip != 0U) {
        --failure.matching_calls_to_skip;
        continue;
      }
      return failure.status;
    }
  }
  return LifecycleStatus::kOk;
}

// Derives a tight LifecycleKernelCapacity from graph input with no slack.
// max_completions = nodes * 6 covers all stages traversed twice in the worst-case restart.
[[nodiscard]] LifecycleKernelCapacity capacity_for(const LifecycleStageGraphInput& input) {
  return LifecycleKernelCapacity{
      .max_scopes = input.scopes.size(),
      .max_nodes = input.nodes.size(),
      .max_callbacks = input.callbacks.size(),
      .max_dependency_records = input.producers.size() + input.requirements.size(),
      .max_direct_edges = input.direct_edges.size(),
      .max_completions = input.nodes.size() * 6U,
  };
}

// Builds the standard two-node lifecycle graph used by ordering and shutdown tests.
//   scope 9
//   ├── node 1 "first"  (stable_order=1)
//   └── node 2 "second" (stable_order=2)
// Representative tests: RunsStableStagesAndRollsBackReverseCompletedOrder,
//                       ShutdownUsesAuthoredTeardownDependenciesAndIsIdempotent.
[[nodiscard]] LifecycleStageGraphInput two_node_execution_input() {
  LifecycleStageGraphInput input;
  input.scopes = {{.scope_id = 9U}};
  input.nodes = {
      {.node_id = 2U, .scope_id = 9U, .stable_name = "second", .stable_order = 2U},
      {.node_id = 1U, .scope_id = 9U, .stable_name = "first", .stable_order = 1U},
  };
  for (std::uint8_t stage = 0U; stage < 8U; ++stage) {
    const auto lifecycle_stage = static_cast<LifecycleStage>(stage);
    input.callbacks.push_back({.node_id = 1U,
                               .stage = lifecycle_stage,
                               .required = true,
                               .method = method_for(lifecycle_stage)});
    input.callbacks.push_back({.node_id = 2U,
                               .stage = lifecycle_stage,
                               .required = true,
                               .method = method_for(lifecycle_stage)});
  }
  return input;
}

// Builds the standard single-node lifecycle graph used by restart and boundary tests.
//   scope 1
//   └── node 1 "operator" (stable_order=1)
// Representative tests: RestartSuffixesMatchExactRequestedStageTable,
//                       FailedRestartRestoresOldSuffixAndPreservesPrimaryFailure.
[[nodiscard]] LifecycleStageGraphInput one_node_execution_input() {
  LifecycleStageGraphInput input;
  input.scopes = {{.scope_id = 1U}};
  input.nodes = {
      {.node_id = 1U, .scope_id = 1U, .stable_name = "operator", .stable_order = 1U},
  };
  for (std::uint8_t stage = 0U; stage < 8U; ++stage) {
    const auto lifecycle_stage = static_cast<LifecycleStage>(stage);
    input.callbacks.push_back({.node_id = 1U,
                               .stage = lifecycle_stage,
                               .required = true,
                               .method = method_for(lifecycle_stage)});
  }
  return input;
}

// Wires both probes into a 16-entry binding list (8 stages × 2 nodes) for the two-node graph.
[[nodiscard]] std::vector<LifecycleCallbackBinding> two_node_bindings(CallbackProbe& first,
                                                                      CallbackProbe& second) {
  std::vector<LifecycleCallbackBinding> bindings;
  bindings.reserve(16U);
  for (std::uint8_t stage = 0U; stage < 8U; ++stage) {
    bindings.push_back({.node_id = 1U,
                        .stage = static_cast<LifecycleStage>(stage),
                        .self = &first,
                        .callback = &record_callback});
    bindings.push_back({.node_id = 2U,
                        .stage = static_cast<LifecycleStage>(stage),
                        .self = &second,
                        .callback = &record_callback});
  }
  return bindings;
}

// Wires the given probe into an 8-entry binding list for the single-node graph.
[[nodiscard]] std::vector<LifecycleCallbackBinding> one_node_bindings(CallbackProbe& probe) {
  std::vector<LifecycleCallbackBinding> bindings;
  bindings.reserve(8U);
  for (std::uint8_t stage = 0U; stage < 8U; ++stage) {
    bindings.push_back({.node_id = 1U,
                        .stage = static_cast<LifecycleStage>(stage),
                        .self = &probe,
                        .callback = &record_callback});
  }
  return bindings;
}

// Checks one TraceEntry by index, emitting node_id/stage/phase labels on failure.
void expect_trace_entry(const CallbackTrace& trace, std::size_t index, LifecycleNodeId node_id,
                        LifecycleStage stage, LifecycleTransitionPhase phase) {
  ASSERT_LT(index, trace.size);
  EXPECT_EQ(trace.entries[index].node_id, node_id);
  EXPECT_EQ(trace.entries[index].stage, stage);
  EXPECT_EQ(trace.entries[index].phase, phase);
}

// Asserts two CallbackTrace objects have equal size and matching entries in the same order.
void expect_same_trace(const CallbackTrace& lhs, const CallbackTrace& rhs) {
  ASSERT_EQ(lhs.size, rhs.size);
  for (std::size_t index = 0U; index < lhs.size; ++index) {
    EXPECT_EQ(lhs.entries[index], rhs.entries[index]);
  }
}

// Asserts all fields of two LifecycleExecutionFailure structs match, including the
// present flag so an absent failure is not silently accepted as equal to a present one.
void expect_same_failure(const detail::LifecycleExecutionFailure& lhs,
                         const detail::LifecycleExecutionFailure& rhs) {
  EXPECT_EQ(lhs.present, rhs.present);
  EXPECT_EQ(lhs.node_id, rhs.node_id);
  EXPECT_EQ(lhs.stage, rhs.stage);
  EXPECT_EQ(lhs.phase, rhs.phase);
  EXPECT_EQ(lhs.status, rhs.status);
}

// Asserts disposition, all four failure slots, and steady_state of two execution results match.
void expect_same_result(const detail::LifecycleExecutionResult& lhs,
                        const detail::LifecycleExecutionResult& rhs) {
  EXPECT_EQ(lhs.disposition, rhs.disposition);
  expect_same_failure(lhs.primary_failure, rhs.primary_failure);
  expect_same_failure(lhs.rollback_failure, rhs.rollback_failure);
  expect_same_failure(lhs.restoration_failure, rhs.restoration_failure);
  expect_same_failure(lhs.restoration_rollback_failure, rhs.restoration_rollback_failure);
  EXPECT_EQ(lhs.steady_state, rhs.steady_state);
}

// Asserts observable LifecycleKernelState fields match, including the completed list entries.
void expect_same_kernel_state(const detail::LifecycleKernelState& lhs,
                              const detail::LifecycleKernelState& rhs) {
  EXPECT_EQ(lhs.steady_state, rhs.steady_state);
  EXPECT_EQ(lhs.completion_capacity, rhs.completion_capacity);
  ASSERT_EQ(lhs.completed.size(), rhs.completed.size());
  for (std::size_t index = 0U; index < lhs.completed.size(); ++index) {
    EXPECT_EQ(lhs.completed[index].node_id, rhs.completed[index].node_id);
    EXPECT_EQ(lhs.completed[index].stage, rhs.completed[index].stage);
  }
  EXPECT_EQ(lhs.rollback_visited, rhs.rollback_visited);
}

// Verifies every callback-related log record in trace carries the expected operation tag.
void expect_callback_log_operation(const LifecycleLogTrace& trace,
                                   detail::LifecycleLogOperation operation) {
  bool saw_callback = false;
  for (std::size_t index = 0U; index < trace.size; ++index) {
    const auto& record = trace.records[index];
    if (record.kind == detail::LifecycleLogEventKind::kCallbackBegin ||
        record.kind == detail::LifecycleLogEventKind::kCallbackEnd) {
      saw_callback = true;
      EXPECT_EQ(record.operation, operation);
    }
  }
  EXPECT_TRUE(saw_callback);
}

// Returns true if any callback log record in trace carries the given transition phase.
[[nodiscard]] bool callback_log_contains_phase(const LifecycleLogTrace& trace,
                                               LifecycleTransitionPhase phase) {
  return std::any_of(trace.records.begin(),
                     trace.records.begin() + static_cast<std::ptrdiff_t>(trace.size),
                     [phase](const auto& record) {
                       return (record.kind == detail::LifecycleLogEventKind::kCallbackBegin ||
                               record.kind == detail::LifecycleLogEventKind::kCallbackEnd) &&
                              record.phase == phase;
                     });
}

// REQ: LIFECYCLE-001
// REQ: LIFECYCLE-004
TEST(LifecycleExecutor, RunsStableStagesAndRollsBackReverseCompletedOrder) {
  const auto input = two_node_execution_input();
  const auto graph = detail::build_lifecycle_stage_graph(input, capacity_for(input));
  ASSERT_TRUE(graph.has_value());

  // Two-node topology (startup order: first → second at each stage):
  //   scope 9
  //   ├── node 1 "first"  (stable_order=1) ─ runs each stage first
  //   └── node 2 "second" (stable_order=2) ─ fails kStart; rollback reverses completion order
  // Key setup: second fails kStart (fatal); its kRelease rollback then fails (retryable).
  CallbackTrace trace;
  CallbackProbe first{.node_id = 1U, .trace = &trace};
  CallbackProbe second{.node_id = 2U, .trace = &trace};
  second.failures = {
      FailureRule{.enabled = true,
                  .stage = LifecycleStage::kStart,
                  .phase = LifecycleTransitionPhase::kStartupForward,
                  .status = LifecycleStatus::kFatalFailure},
      FailureRule{.enabled = true,
                  .stage = LifecycleStage::kRelease,
                  .phase = LifecycleTransitionPhase::kForwardRollback,
                  .status = LifecycleStatus::kRetryableFailure},
  };
  const auto bindings = two_node_bindings(first, second);
  auto state = detail::make_lifecycle_kernel_state(*graph);
  const auto completion_storage = state.completed.capacity();
  const auto rollback_storage = state.rollback_visited.capacity();
  const auto result = detail::execute_lifecycle_startup(*graph, bindings, state);

  EXPECT_EQ(result.disposition, LifecycleExecutionDisposition::kRollbackFailed);
  ASSERT_TRUE(result.primary_failure.present);
  EXPECT_EQ(result.primary_failure.node_id, 2U);
  EXPECT_EQ(result.primary_failure.stage, LifecycleStage::kStart);
  EXPECT_EQ(result.primary_failure.status, LifecycleStatus::kFatalFailure);
  ASSERT_TRUE(result.rollback_failure.present);
  EXPECT_EQ(result.rollback_failure.node_id, 2U);
  EXPECT_EQ(result.rollback_failure.stage, LifecycleStage::kRelease);
  EXPECT_EQ(result.rollback_failure.status, LifecycleStatus::kRetryableFailure);
  EXPECT_FALSE(result.steady_state);
  EXPECT_TRUE(state.completed.empty());
  EXPECT_EQ(state.completed.capacity(), completion_storage);
  EXPECT_EQ(state.rollback_visited.capacity(), rollback_storage);

  std::size_t index = 0U;
  for (const LifecycleStage stage : {LifecycleStage::kConfigure,
                                     LifecycleStage::kDiscover,
                                     LifecycleStage::kAllocate,
                                     LifecycleStage::kWarmUp,
                                     LifecycleStage::kArm}) {
    expect_trace_entry(trace, index++, 1U, stage, LifecycleTransitionPhase::kStartupForward);
    expect_trace_entry(trace, index++, 2U, stage, LifecycleTransitionPhase::kStartupForward);
  }
  expect_trace_entry(
      trace, index++, 1U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward);
  expect_trace_entry(
      trace, index++, 2U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward);

  // Key assertion: rollback order is the reverse of completion order — first completed kStart
  // last so it is rolled back first; second completed kArm last so its kRelease comes first.
  // The first node completed kStart most recently, so it is stopped before the second node,
  // whose latest completed work was kArm. Allocation rollback independently follows the reverse
  // of the allocation completion order.
  expect_trace_entry(
      trace, index++, 1U, LifecycleStage::kStop, LifecycleTransitionPhase::kForwardRollback);
  expect_trace_entry(
      trace, index++, 2U, LifecycleStage::kStop, LifecycleTransitionPhase::kForwardRollback);
  expect_trace_entry(
      trace, index++, 2U, LifecycleStage::kRelease, LifecycleTransitionPhase::kForwardRollback);
  expect_trace_entry(
      trace, index++, 1U, LifecycleStage::kRelease, LifecycleTransitionPhase::kForwardRollback);
  EXPECT_EQ(trace.size, index);
}

// REQ: LIFECYCLE-006
TEST(LifecycleExecutor, RestartSuffixesMatchExactRequestedStageTable) {
  const auto input = one_node_execution_input();
  const auto graph = detail::build_lifecycle_stage_graph(input, capacity_for(input));
  ASSERT_TRUE(graph.has_value());

  struct RestartCase {
    LifecycleStage requested;
    std::vector<LifecycleStage> expected;
  };
  const std::array cases{
      RestartCase{.requested = LifecycleStage::kStart,
                  .expected = {LifecycleStage::kStop, LifecycleStage::kStart}},
      RestartCase{
          .requested = LifecycleStage::kArm,
          .expected = {LifecycleStage::kStop, LifecycleStage::kArm, LifecycleStage::kStart}},
      RestartCase{.requested = LifecycleStage::kWarmUp,
                  .expected = {LifecycleStage::kStop,
                               LifecycleStage::kWarmUp,
                               LifecycleStage::kArm,
                               LifecycleStage::kStart}},
      RestartCase{.requested = LifecycleStage::kAllocate,
                  .expected = {LifecycleStage::kStop,
                               LifecycleStage::kRelease,
                               LifecycleStage::kAllocate,
                               LifecycleStage::kWarmUp,
                               LifecycleStage::kArm,
                               LifecycleStage::kStart}},
      RestartCase{.requested = LifecycleStage::kDiscover,
                  .expected = {LifecycleStage::kStop,
                               LifecycleStage::kRelease,
                               LifecycleStage::kDiscover,
                               LifecycleStage::kAllocate,
                               LifecycleStage::kWarmUp,
                               LifecycleStage::kArm,
                               LifecycleStage::kStart}},
      RestartCase{.requested = LifecycleStage::kConfigure,
                  .expected = {LifecycleStage::kStop,
                               LifecycleStage::kRelease,
                               LifecycleStage::kConfigure,
                               LifecycleStage::kDiscover,
                               LifecycleStage::kAllocate,
                               LifecycleStage::kWarmUp,
                               LifecycleStage::kArm,
                               LifecycleStage::kStart}},
  };

  for (const auto& test_case : cases) {
    CallbackTrace trace;
    CallbackProbe probe{.node_id = 1U, .trace = &trace};
    const auto bindings = one_node_bindings(probe);
    auto state = detail::make_lifecycle_kernel_state(*graph);
    ASSERT_EQ(detail::execute_lifecycle_startup(*graph, bindings, state).disposition,
              LifecycleExecutionDisposition::kCompleted);
    trace.clear();

    const auto result =
        detail::execute_lifecycle_restart(*graph, bindings, test_case.requested, state);
    ASSERT_EQ(result.disposition, LifecycleExecutionDisposition::kCompleted);
    ASSERT_TRUE(result.steady_state);
    if (test_case.requested == LifecycleStage::kStart) {
      const auto arm_count =
          std::count_if(state.completed.begin(), state.completed.end(), [](const auto& checkpoint) {
            return checkpoint.stage == LifecycleStage::kArm;
          });
      EXPECT_EQ(arm_count, 1);
    }
    ASSERT_EQ(trace.size, test_case.expected.size());
    for (std::size_t index = 0U; index < test_case.expected.size(); ++index) {
      const LifecycleStage stage = test_case.expected[index];
      const auto phase = stage == LifecycleStage::kStop || stage == LifecycleStage::kRelease
                             ? LifecycleTransitionPhase::kPreRestartTeardown
                             : LifecycleTransitionPhase::kRestartForward;
      expect_trace_entry(trace, index, 1U, stage, phase);
    }
  }

  for (const LifecycleStage invalid : {LifecycleStage::kStop,
                                       LifecycleStage::kRelease,
                                       std::bit_cast<LifecycleStage>(std::uint8_t{255U})}) {
    CallbackTrace trace;
    CallbackProbe probe{.node_id = 1U, .trace = &trace};
    const auto bindings = one_node_bindings(probe);
    auto state = detail::make_lifecycle_kernel_state(*graph);
    ASSERT_EQ(detail::execute_lifecycle_startup(*graph, bindings, state).disposition,
              LifecycleExecutionDisposition::kCompleted);
    trace.clear();
    const auto result = detail::execute_lifecycle_restart(*graph, bindings, invalid, state);
    EXPECT_EQ(result.disposition, LifecycleExecutionDisposition::kInvalidRestartStage);
    EXPECT_TRUE(result.steady_state);
    EXPECT_EQ(trace.size, 0U);
  }
}

// REQ: LIFECYCLE-004
// REQ: LIFECYCLE-005
TEST(LifecycleExecutor, FailedRestartRestoresOldSuffixAndPreservesPrimaryFailure) {
  const auto input = one_node_execution_input();
  const auto graph = detail::build_lifecycle_stage_graph(input, capacity_for(input));
  ASSERT_TRUE(graph.has_value());

  CallbackTrace trace;
  CallbackProbe probe{.node_id = 1U, .trace = &trace};
  const auto bindings = one_node_bindings(probe);
  auto state = detail::make_lifecycle_kernel_state(*graph);
  ASSERT_EQ(detail::execute_lifecycle_startup(*graph, bindings, state).disposition,
            LifecycleExecutionDisposition::kCompleted);
  trace.clear();
  probe.failures[0] = FailureRule{.enabled = true,
                                  .stage = LifecycleStage::kWarmUp,
                                  .phase = LifecycleTransitionPhase::kRestartForward,
                                  .status = LifecycleStatus::kFatalFailure};

  const auto result =
      detail::execute_lifecycle_restart(*graph, bindings, LifecycleStage::kWarmUp, state);
  EXPECT_EQ(result.disposition, LifecycleExecutionDisposition::kRolledBack);
  ASSERT_TRUE(result.primary_failure.present);
  EXPECT_EQ(result.primary_failure.stage, LifecycleStage::kWarmUp);
  EXPECT_EQ(result.primary_failure.phase, LifecycleTransitionPhase::kRestartForward);
  EXPECT_FALSE(result.rollback_failure.present);
  EXPECT_FALSE(result.restoration_failure.present);
  EXPECT_TRUE(result.steady_state);
  EXPECT_TRUE(state.steady_state);

  ASSERT_EQ(trace.size, 5U);
  expect_trace_entry(
      trace, 0U, 1U, LifecycleStage::kStop, LifecycleTransitionPhase::kPreRestartTeardown);
  expect_trace_entry(
      trace, 1U, 1U, LifecycleStage::kWarmUp, LifecycleTransitionPhase::kRestartForward);
  expect_trace_entry(trace, 2U, 1U, LifecycleStage::kWarmUp, LifecycleTransitionPhase::kRestoreOld);
  expect_trace_entry(trace, 3U, 1U, LifecycleStage::kArm, LifecycleTransitionPhase::kRestoreOld);
  expect_trace_entry(trace, 4U, 1U, LifecycleStage::kStart, LifecycleTransitionPhase::kRestoreOld);
}

// REQ: LIFECYCLE-004
// REQ: LIFECYCLE-005
TEST(LifecycleExecutor, FailedOldStateRestorationLeavesKernelStopped) {
  const auto input = one_node_execution_input();
  const auto graph = detail::build_lifecycle_stage_graph(input, capacity_for(input));
  ASSERT_TRUE(graph.has_value());

  CallbackTrace trace;
  CallbackProbe probe{.node_id = 1U, .trace = &trace};
  const auto bindings = one_node_bindings(probe);
  auto state = detail::make_lifecycle_kernel_state(*graph);
  ASSERT_EQ(detail::execute_lifecycle_startup(*graph, bindings, state).disposition,
            LifecycleExecutionDisposition::kCompleted);
  probe.failures = {
      FailureRule{.enabled = true,
                  .stage = LifecycleStage::kWarmUp,
                  .phase = LifecycleTransitionPhase::kRestartForward,
                  .status = LifecycleStatus::kFatalFailure},
      FailureRule{.enabled = true,
                  .stage = LifecycleStage::kWarmUp,
                  .phase = LifecycleTransitionPhase::kRestoreOld,
                  .status = LifecycleStatus::kRetryableFailure},
  };

  const auto result =
      detail::execute_lifecycle_restart(*graph, bindings, LifecycleStage::kWarmUp, state);
  EXPECT_EQ(result.disposition, LifecycleExecutionDisposition::kRollbackFailed);
  ASSERT_TRUE(result.primary_failure.present);
  EXPECT_EQ(result.primary_failure.phase, LifecycleTransitionPhase::kRestartForward);
  ASSERT_TRUE(result.restoration_failure.present);
  EXPECT_EQ(result.restoration_failure.phase, LifecycleTransitionPhase::kRestoreOld);
  EXPECT_EQ(result.restoration_failure.status, LifecycleStatus::kRetryableFailure);
  EXPECT_FALSE(result.restoration_rollback_failure.present);
  EXPECT_FALSE(result.steady_state);
  EXPECT_FALSE(state.steady_state);
}

// REQ: LIFECYCLE-004
// REQ: LIFECYCLE-005
TEST(LifecycleExecutor, FailedRestorationRollbackPreservesAllFailureEvidence) {
  const auto input = one_node_execution_input();
  const auto graph = detail::build_lifecycle_stage_graph(input, capacity_for(input));
  ASSERT_TRUE(graph.has_value());

  CallbackTrace trace;
  CallbackProbe probe{.node_id = 1U, .trace = &trace};
  const auto bindings = one_node_bindings(probe);
  auto state = detail::make_lifecycle_kernel_state(*graph);
  ASSERT_EQ(detail::execute_lifecycle_startup(*graph, bindings, state).disposition,
            LifecycleExecutionDisposition::kCompleted);
  probe.failures = {
      FailureRule{.enabled = true,
                  .stage = LifecycleStage::kWarmUp,
                  .phase = LifecycleTransitionPhase::kRestartForward,
                  .status = LifecycleStatus::kFatalFailure},
      FailureRule{.enabled = true,
                  .stage = LifecycleStage::kWarmUp,
                  .phase = LifecycleTransitionPhase::kRestoreOld,
                  .status = LifecycleStatus::kRetryableFailure},
      FailureRule{.enabled = true,
                  .stage = LifecycleStage::kRelease,
                  .phase = LifecycleTransitionPhase::kForwardRollback,
                  .status = LifecycleStatus::kFatalFailure,
                  .matching_calls_to_skip = 1U},
  };

  const auto result =
      detail::execute_lifecycle_restart(*graph, bindings, LifecycleStage::kAllocate, state);
  EXPECT_EQ(result.disposition, LifecycleExecutionDisposition::kRollbackFailed);
  ASSERT_TRUE(result.primary_failure.present);
  EXPECT_EQ(result.primary_failure.stage, LifecycleStage::kWarmUp);
  EXPECT_EQ(result.primary_failure.phase, LifecycleTransitionPhase::kRestartForward);
  EXPECT_EQ(result.primary_failure.status, LifecycleStatus::kFatalFailure);
  EXPECT_FALSE(result.rollback_failure.present);
  ASSERT_TRUE(result.restoration_failure.present);
  EXPECT_EQ(result.restoration_failure.stage, LifecycleStage::kWarmUp);
  EXPECT_EQ(result.restoration_failure.phase, LifecycleTransitionPhase::kRestoreOld);
  EXPECT_EQ(result.restoration_failure.status, LifecycleStatus::kRetryableFailure);
  ASSERT_TRUE(result.restoration_rollback_failure.present);
  EXPECT_EQ(result.restoration_rollback_failure.stage, LifecycleStage::kRelease);
  EXPECT_EQ(result.restoration_rollback_failure.phase, LifecycleTransitionPhase::kForwardRollback);
  EXPECT_EQ(result.restoration_rollback_failure.status, LifecycleStatus::kFatalFailure);
  EXPECT_FALSE(result.steady_state);
  EXPECT_FALSE(state.steady_state);
}

// REQ: LIFECYCLE-001
// REQ: LIFECYCLE-004
TEST(LifecycleExecutor, ShutdownUsesAuthoredTeardownDependenciesAndIsIdempotent) {
  // Two-node graph with authored teardown dependency edges:
  //   node 1 "first"  ──kStop──►  node 2 "second"
  //   node 1 "first"  ──kRelease──►  node 2 "second"
  // Expected shutdown sequence: 1.kStop, 2.kStop, 1.kRelease, 2.kRelease.
  // Key setup: direct_edges constrain both teardown stages so second always follows first.
  auto input = two_node_execution_input();
  input.direct_edges = {
      {.declaration_scope_id = 9U,
       .before_node_id = 1U,
       .before_stage = LifecycleStage::kStop,
       .after_node_id = 2U,
       .after_stage = LifecycleStage::kStop},
      {.declaration_scope_id = 9U,
       .before_node_id = 1U,
       .before_stage = LifecycleStage::kRelease,
       .after_node_id = 2U,
       .after_stage = LifecycleStage::kRelease},
  };
  const auto graph = detail::build_lifecycle_stage_graph(input, capacity_for(input));
  ASSERT_TRUE(graph.has_value());
  CallbackTrace trace;
  CallbackProbe first{.node_id = 1U, .trace = &trace};
  CallbackProbe second{.node_id = 2U, .trace = &trace};
  const auto bindings = two_node_bindings(first, second);
  auto state = detail::make_lifecycle_kernel_state(*graph);
  ASSERT_EQ(detail::execute_lifecycle_startup(*graph, bindings, state).disposition,
            LifecycleExecutionDisposition::kCompleted);
  trace.clear();

  // Key assertion: second shutdown call completes with no additional trace entries —
  // shutdown is idempotent once the kernel has already reached the teardown state.
  EXPECT_EQ(detail::execute_lifecycle_shutdown(*graph, bindings, state).disposition,
            LifecycleExecutionDisposition::kCompleted);
  EXPECT_EQ(detail::execute_lifecycle_shutdown(*graph, bindings, state).disposition,
            LifecycleExecutionDisposition::kCompleted);
  ASSERT_EQ(trace.size, 4U);
  expect_trace_entry(
      trace, 0U, 1U, LifecycleStage::kStop, LifecycleTransitionPhase::kShutdownTeardown);
  expect_trace_entry(
      trace, 1U, 2U, LifecycleStage::kStop, LifecycleTransitionPhase::kShutdownTeardown);
  expect_trace_entry(
      trace, 2U, 1U, LifecycleStage::kRelease, LifecycleTransitionPhase::kShutdownTeardown);
  expect_trace_entry(
      trace, 3U, 2U, LifecycleStage::kRelease, LifecycleTransitionPhase::kShutdownTeardown);
}

// REQ: LIFECYCLE-001
// REQ: LIFECYCLE-002
TEST(LifecycleExecutor, MissingOptionalBindingSkipsWhileMissingRequiredBindingIsInvalid) {
  auto input = one_node_execution_input();
  for (auto& callback : input.callbacks) {
    callback.required = callback.stage == LifecycleStage::kStart;
  }
  const auto graph = detail::build_lifecycle_stage_graph(input, capacity_for(input));
  ASSERT_TRUE(graph.has_value());

  // Key setup: graph marks only kStart as required; supply only a kStart binding so
  // every other stage is hit as an optional skip, then supply no binding at all to
  // trigger an invalid-binding rejection for the required kStart stage.
  CallbackTrace trace;
  CallbackProbe probe{.node_id = 1U, .trace = &trace};
  const std::vector<LifecycleCallbackBinding> start_only_binding = {{
      .node_id = 1U,
      .stage = LifecycleStage::kStart,
      .self = &probe,
      .callback = &record_callback,
  }};
  LifecycleLogTrace logs;
  auto state = detail::make_lifecycle_kernel_state(*graph);
  state.log_sink = &record_lifecycle_log;
  state.log_state = &logs;

  const auto completed = detail::execute_lifecycle_startup(*graph, start_only_binding, state);
  EXPECT_EQ(completed.disposition, LifecycleExecutionDisposition::kCompleted);
  EXPECT_TRUE(completed.steady_state);
  ASSERT_EQ(trace.size, 1U);
  expect_trace_entry(
      trace, 0U, 1U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward);
  EXPECT_TRUE(std::any_of(logs.records.begin(),
                          logs.records.begin() + static_cast<std::ptrdiff_t>(logs.size),
                          [](const auto& record) {
                            return record.kind == detail::LifecycleLogEventKind::kCallbackEnd &&
                                   record.stage == LifecycleStage::kConfigure &&
                                   record.status == LifecycleStatus::kSkipped;
                          }));

  trace.clear();
  auto missing_required_state = detail::make_lifecycle_kernel_state(*graph);
  const auto rejected = detail::execute_lifecycle_startup(*graph, {}, missing_required_state);
  EXPECT_EQ(rejected.disposition, LifecycleExecutionDisposition::kInvalidBinding);
  EXPECT_FALSE(rejected.steady_state);
  EXPECT_TRUE(missing_required_state.completed.empty());
  EXPECT_EQ(trace.size, 0U);
}

// REQ: LIFECYCLE-007
TEST(LifecycleExecutor, CompletionBoundFailsBeforeStorageGrowth) {
  const auto input = one_node_execution_input();
  const auto graph = detail::build_lifecycle_stage_graph(input, capacity_for(input));
  ASSERT_TRUE(graph.has_value());
  CallbackTrace trace;
  CallbackProbe probe{.node_id = 1U, .trace = &trace};
  const auto bindings = one_node_bindings(probe);
  auto state = detail::make_lifecycle_kernel_state(*graph);
  const auto storage_capacity = state.completed.capacity();
  state.completion_capacity = 5U;

  const auto result = detail::execute_lifecycle_startup(*graph, bindings, state);
  EXPECT_EQ(result.disposition, LifecycleExecutionDisposition::kCapacityExceeded);
  EXPECT_EQ(state.completed.capacity(), storage_capacity);
  EXPECT_TRUE(state.completed.empty());
  EXPECT_FALSE(state.steady_state);
}

// REQ: LIFECYCLE-007
TEST(LifecycleExecutor, ExhaustedCompletionBoundEntersNoUserCallback) {
  const auto input = one_node_execution_input();
  const auto graph = detail::build_lifecycle_stage_graph(input, capacity_for(input));
  ASSERT_TRUE(graph.has_value());
  CallbackTrace trace;
  CallbackProbe probe{.node_id = 1U, .trace = &trace};
  const auto bindings = one_node_bindings(probe);
  auto state = detail::make_lifecycle_kernel_state(*graph);
  state.completion_capacity = 0U;

  const auto result = detail::execute_lifecycle_startup(*graph, bindings, state);
  EXPECT_EQ(result.disposition, LifecycleExecutionDisposition::kCapacityExceeded);
  EXPECT_EQ(trace.size, 0U);
  EXPECT_TRUE(state.completed.empty());
}

// REQ: LIFECYCLE-009
// REQ: LIFECYCLE-013
// REQ: LIFECYCLE-014
TEST(LifecycleExecutor, LoggingEmitsTraceDebugAndOneFinalErrorWithoutControlFlowAuthority) {
  const auto input = one_node_execution_input();
  const auto graph = detail::build_lifecycle_stage_graph(input, capacity_for(input));
  ASSERT_TRUE(graph.has_value());

  // Key setup: three independent execution paths are exercised — success, failure with a
  // throwing log sink, and shutdown — to confirm log output never changes control flow.
  CallbackTrace callback_trace;
  CallbackProbe success_probe{.node_id = 1U, .trace = &callback_trace};
  const auto success_bindings = one_node_bindings(success_probe);
  LifecycleLogTrace success_logs;
  auto success_state = detail::make_lifecycle_kernel_state(*graph);
  success_state.log_sink = &record_lifecycle_log;
  success_state.log_state = &success_logs;
  const auto success = detail::execute_lifecycle_startup(*graph, success_bindings, success_state);
  ASSERT_EQ(success.disposition, LifecycleExecutionDisposition::kCompleted);
  ASSERT_GE(success_logs.size, 2U);
  EXPECT_EQ(success_logs.records.front().kind, detail::LifecycleLogEventKind::kTransitionBegin);
  EXPECT_EQ(detail::lifecycle_log_severity(success_logs.records.front()),
            detail::LifecycleLogSeverity::kDebug);
  EXPECT_EQ(success_logs.records[success_logs.size - 1U].kind,
            detail::LifecycleLogEventKind::kTransitionEnd);
  EXPECT_EQ(detail::lifecycle_log_severity(success_logs.records[success_logs.size - 1U]),
            detail::LifecycleLogSeverity::kDebug);
  for (std::size_t index = 1U; index + 1U < success_logs.size; ++index) {
    EXPECT_EQ(detail::lifecycle_log_severity(success_logs.records[index]),
              detail::LifecycleLogSeverity::kTrace);
  }
  expect_callback_log_operation(success_logs, detail::LifecycleLogOperation::kStartup);

  callback_trace.clear();
  success_logs.clear();
  success_probe.failures[0] = {
      .enabled = true,
      .stage = LifecycleStage::kStart,
      .phase = LifecycleTransitionPhase::kRestartForward,
      .status = LifecycleStatus::kFatalFailure,
  };
  const auto restarted = detail::execute_lifecycle_restart(
      *graph, success_bindings, LifecycleStage::kAllocate, success_state);
  ASSERT_EQ(restarted.disposition, LifecycleExecutionDisposition::kRolledBack);
  EXPECT_TRUE(restarted.steady_state);
  expect_callback_log_operation(success_logs, detail::LifecycleLogOperation::kRestart);
  EXPECT_TRUE(
      callback_log_contains_phase(success_logs, LifecycleTransitionPhase::kForwardRollback));
  EXPECT_TRUE(callback_log_contains_phase(success_logs, LifecycleTransitionPhase::kRestoreOld));

  callback_trace.clear();
  success_logs.clear();
  success_probe.failures[0] = {};
  const auto stopped = detail::execute_lifecycle_shutdown(*graph, success_bindings, success_state);
  ASSERT_EQ(stopped.disposition, LifecycleExecutionDisposition::kCompleted);
  expect_callback_log_operation(success_logs, detail::LifecycleLogOperation::kShutdown);

  CallbackTrace failure_trace;
  CallbackProbe failure_probe{.node_id = 1U, .trace = &failure_trace};
  failure_probe.failures[0] = {
      .enabled = true,
      .stage = LifecycleStage::kStart,
      .phase = LifecycleTransitionPhase::kStartupForward,
      .status = LifecycleStatus::kFatalFailure,
  };
  const auto failure_bindings = one_node_bindings(failure_probe);
  LifecycleLogTrace failure_logs;
  auto failure_state = detail::make_lifecycle_kernel_state(*graph);
  failure_state.log_sink = &record_lifecycle_log;
  failure_state.log_state = &failure_logs;
  const auto failure = detail::execute_lifecycle_startup(*graph, failure_bindings, failure_state);
  EXPECT_EQ(failure.disposition, LifecycleExecutionDisposition::kStageFailed);
  // Key assertion: exactly one error-severity log record is emitted, at the transition end,
  // regardless of how many trace/debug records precede it.
  const auto error_count = std::count_if(
      failure_logs.records.begin(),
      failure_logs.records.begin() + static_cast<std::ptrdiff_t>(failure_logs.size),
      [](const auto& record) {
        return detail::lifecycle_log_severity(record) == detail::LifecycleLogSeverity::kError;
      });
  EXPECT_EQ(error_count, 1);
  ASSERT_NE(failure_logs.size, 0U);
  EXPECT_EQ(failure_logs.records[failure_logs.size - 1U].kind,
            detail::LifecycleLogEventKind::kTransitionEnd);
  expect_callback_log_operation(failure_logs, detail::LifecycleLogOperation::kStartup);

  CallbackTrace throwing_failure_trace;
  CallbackProbe throwing_probe{.node_id = 1U, .trace = &throwing_failure_trace};
  throwing_probe.failures[0] = failure_probe.failures[0];
  const auto throwing_bindings = one_node_bindings(throwing_probe);
  auto throwing_state = detail::make_lifecycle_kernel_state(*graph);
  throwing_state.log_sink = &throw_from_lifecycle_log;
  // Key assertion: a throwing log sink must not alter the execution result, kernel state,
  // or callback trace relative to the recording sink path.
  const auto throwing =
      detail::execute_lifecycle_startup(*graph, throwing_bindings, throwing_state);
  expect_same_result(failure, throwing);
  expect_same_kernel_state(failure_state, throwing_state);
  expect_same_trace(failure_trace, throwing_failure_trace);
  EXPECT_TRUE(throwing_state.completed.empty());
  EXPECT_FALSE(throwing_state.steady_state);
  EXPECT_TRUE(std::any_of(
      throwing_failure_trace.entries.begin(),
      throwing_failure_trace.entries.begin() +
          static_cast<std::ptrdiff_t>(throwing_failure_trace.size),
      [](const auto& entry) { return entry.phase == LifecycleTransitionPhase::kForwardRollback; }));

  CallbackTrace shutdown_trace;
  CallbackProbe shutdown_probe{.node_id = 1U, .trace = &shutdown_trace};
  const auto shutdown_bindings = one_node_bindings(shutdown_probe);
  auto shutdown_state = detail::make_lifecycle_kernel_state(*graph);
  ASSERT_EQ(
      detail::execute_lifecycle_startup(*graph, shutdown_bindings, shutdown_state).disposition,
      LifecycleExecutionDisposition::kCompleted);
  shutdown_trace.clear();
  LifecycleLogTrace shutdown_logs;
  shutdown_state.log_sink = &record_lifecycle_log;
  shutdown_state.log_state = &shutdown_logs;

  CallbackTrace throwing_shutdown_trace;
  CallbackProbe throwing_shutdown_probe{.node_id = 1U, .trace = &throwing_shutdown_trace};
  const auto throwing_shutdown_bindings = one_node_bindings(throwing_shutdown_probe);
  auto throwing_shutdown_state = detail::make_lifecycle_kernel_state(*graph);
  ASSERT_EQ(
      detail::execute_lifecycle_startup(*graph, throwing_shutdown_bindings, throwing_shutdown_state)
          .disposition,
      LifecycleExecutionDisposition::kCompleted);
  throwing_shutdown_trace.clear();
  throwing_shutdown_state.log_sink = &throw_from_lifecycle_log;

  const auto shutdown =
      detail::execute_lifecycle_shutdown(*graph, shutdown_bindings, shutdown_state);
  const auto throwing_shutdown = detail::execute_lifecycle_shutdown(
      *graph, throwing_shutdown_bindings, throwing_shutdown_state);
  expect_same_result(shutdown, throwing_shutdown);
  expect_same_kernel_state(shutdown_state, throwing_shutdown_state);
  expect_same_trace(shutdown_trace, throwing_shutdown_trace);
  EXPECT_TRUE(throwing_shutdown_state.completed.empty());
  EXPECT_FALSE(throwing_shutdown_state.steady_state);
  expect_callback_log_operation(shutdown_logs, detail::LifecycleLogOperation::kShutdown);
}

}  // namespace
}  // namespace holoscan::test
