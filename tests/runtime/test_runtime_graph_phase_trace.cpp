/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_graph_phase_trace.hpp"

namespace holoscan::benchmarking {
namespace {

using Point = detail::RuntimePhaseTracePoint;
using Counter = detail::RuntimePhaseCounter;

// REQ: RUNTIME-046
TEST(RuntimePhaseCounters, FilteredTracePreservesEosAcrossUnrelatedCheckpoints) {
  std::array<detail::RuntimePhaseTraceRecord, 1U> storage{};
  detail::RuntimePhaseCounters counters{storage, Point::kChannelEndOfStreamRetained};
  const auto record_idle_waits = [&] {
    for (std::size_t index = 0U; index < 512U; ++index) {
      counters.trace(Point::kControllerParkStarted, 0U, SourceId{}, 0U);
      counters.trace(Point::kControllerTimeoutReturned, 0U, SourceId{}, 0U);
      counters.increment(Counter::kTimeoutReturns);
    }
  };

  record_idle_waits();
  counters.trace(Point::kChannelEndOfStreamRetained, 3U, SourceId{}, 2U, 7U);
  counters.increment(Counter::kChannelEndOfStreamRetained);
  record_idle_waits();

  ASSERT_EQ(counters.trace_records().size(), 1U);
  const auto& record = counters.trace_records().front();
  EXPECT_EQ(record.point, Point::kChannelEndOfStreamRetained);
  EXPECT_EQ(record.receiver_index, 3U);
  EXPECT_EQ(record.source_sequence, 2U);
  EXPECT_EQ(record.output_port_id, 7U);
  EXPECT_EQ(counters.trace_attempted(), 1U);
  EXPECT_EQ(counters.trace_dropped(), 0U);
  EXPECT_FALSE(counters.trace_capacity_exhausted());
  const auto snapshot = counters.snapshot();
  EXPECT_EQ(snapshot[detail::RuntimePhaseCounters::index(Counter::kTimeoutReturns)], 1024U);
  EXPECT_EQ(snapshot[detail::RuntimePhaseCounters::index(Counter::kChannelEndOfStreamRetained)],
            1U);
}

// REQ: RUNTIME-046
TEST(RuntimePhaseCounters, FilteredTraceStillReportsSelectedOverflow) {
  std::array<detail::RuntimePhaseTraceRecord, 1U> storage{};
  detail::RuntimePhaseCounters counters{storage, Point::kChannelEndOfStreamRetained};
  counters.trace(Point::kChannelEndOfStreamRetained, 0U, SourceId{}, 2U);
  counters.trace(Point::kControllerParkStarted, 0U, SourceId{}, 0U);
  counters.trace(Point::kChannelEndOfStreamRetained, 0U, SourceId{}, 3U);

  ASSERT_EQ(counters.trace_records().size(), 1U);
  EXPECT_EQ(counters.trace_records().front().source_sequence, 2U);
  EXPECT_EQ(counters.trace_attempted(), 2U);
  EXPECT_EQ(counters.trace_dropped(), 1U);
  EXPECT_TRUE(counters.trace_capacity_exhausted());
}

// REQ: RUNTIME-046
TEST(RuntimePhaseCounters, UnfilteredTraceRetainsAllPointsAndReportsOverflow) {
  std::array<detail::RuntimePhaseTraceRecord, 2U> storage{};
  detail::RuntimePhaseCounters counters{storage};
  counters.trace(Point::kControllerParkStarted, 0U, SourceId{}, 0U);
  counters.trace(Point::kChannelEndOfStreamRetained, 0U, SourceId{}, 2U);
  counters.trace(Point::kControllerTimeoutReturned, 0U, SourceId{}, 0U);

  ASSERT_EQ(counters.trace_records().size(), 2U);
  EXPECT_EQ(counters.trace_records()[0].point, Point::kControllerParkStarted);
  EXPECT_EQ(counters.trace_records()[1].point, Point::kChannelEndOfStreamRetained);
  EXPECT_EQ(counters.trace_attempted(), 3U);
  EXPECT_EQ(counters.trace_dropped(), 1U);
  EXPECT_TRUE(counters.trace_capacity_exhausted());
}

struct TraceFixture {
  std::array<detail::RuntimePhaseTraceRecord, 64U> storage{};
  detail::RuntimePhaseCounters counters{std::span{storage}};

  void record(Point point, std::uint64_t sequence = 1U) {
    counters.trace(point, 0U, SourceId{}, sequence);
  }

  void prefix(bool prepared = true) {
    record(Point::kCandidateFrozen);
    if (prepared) {
      record(Point::kCandidateSelected);
      record(Point::kInputSelectionPrepared);
      record(Point::kActivationPrepared);
    }
    record(Point::kComputeDispatchStarted);
  }

  void completed() {
    prefix();
    record(Point::kComputeCallbackEntered);
    record(Point::kComputeCallbackReturned);
    record(Point::kComputeDispatchReturned);
    record(Point::kActivationRetirementCompleted);
    record(Point::kActivationCutCompleted);
  }

  void stop_evidence(std::uint64_t admissions = 1U, std::uint64_t stops = 1U) {
    counters.add(Counter::kComputeAdmissionsRunClosing, admissions);
    counters.add(Counter::kStopSignals, stops);
  }
};

// REQ: RUNTIME-046
TEST(RuntimeGraphPhaseTrace, TerminalRejectionDoesNotAddCallbackTimingOrCompletion) {
  for (const bool prepared : {false, true}) {
    SCOPED_TRACE(prepared ? "ordinary dispatch" : "closed-island dispatch");
    TraceFixture trace;
    trace.completed();
    trace.prefix(prepared);
    trace.record(Point::kComputeDispatchReturned);
    trace.stop_evidence();

    const auto summary = summarize_phase_trace(&trace.counters, true);
    ASSERT_TRUE(summary.valid);
    EXPECT_EQ(summary.complete_activations, 1U);
    EXPECT_EQ(summary.terminal_frozen_candidates, 0U);
    EXPECT_EQ(summary.terminal_run_closing_dispatches, 1U);
    ASSERT_EQ(summary.receivers.size(), 1U);
    EXPECT_EQ(summary.receivers.front().activations, 1U);
    EXPECT_EQ(summary.receivers.front().compute_dispatch.size(), 1U);
    EXPECT_EQ(summary.receivers.front().callback_entry_to_return.size(), 1U);
  }
}

// REQ: RUNTIME-046
TEST(RuntimeGraphPhaseTrace, RejectionRequiresExactStopEvidence) {
  for (const auto [admissions, stops] : {std::pair{0U, 1U}, std::pair{1U, 0U}, std::pair{2U, 2U}}) {
    SCOPED_TRACE(std::to_string(admissions) + "/" + std::to_string(stops));
    TraceFixture trace;
    trace.prefix();
    trace.record(Point::kComputeDispatchReturned);
    trace.stop_evidence(admissions, stops);
    EXPECT_FALSE(summarize_phase_trace(&trace.counters, true).valid);
  }
  TraceFixture trace;
  trace.completed();
  trace.stop_evidence();
  EXPECT_FALSE(summarize_phase_trace(&trace.counters, true).valid);
}

// REQ: RUNTIME-046
TEST(RuntimeGraphPhaseTrace, RejectsIncompleteOrNonterminalDispatchDespiteStopEvidence) {
  for (const Point extra : {Point::kComputeCallbackEntered,
                            Point::kComputeCallbackReturned,
                            Point::kComputeDispatchReturned,
                            Point::kActivationRetirementCompleted,
                            Point::kActivationCutCompleted,
                            Point::kCandidateFrozen,
                            Point::kCandidateSelected}) {
    SCOPED_TRACE(static_cast<unsigned int>(extra));
    TraceFixture trace;
    trace.prefix();
    trace.record(Point::kComputeDispatchReturned);
    trace.record(extra);
    trace.stop_evidence();
    EXPECT_FALSE(summarize_phase_trace(&trace.counters, true).valid);
  }
  // A started dispatch without its return is not a stop-rejected dispatch.
  TraceFixture trace;
  trace.prefix();
  trace.stop_evidence();
  EXPECT_FALSE(summarize_phase_trace(&trace.counters, true).valid);
}

// REQ: RUNTIME-046
TEST(RuntimeGraphPhaseTrace, RejectsPartialCallbackAndMismatchedIdentityOrTime) {
  TraceFixture callback;
  callback.prefix();
  callback.record(Point::kComputeCallbackEntered);
  callback.record(Point::kComputeDispatchReturned);
  callback.stop_evidence();
  EXPECT_FALSE(summarize_phase_trace(&callback.counters, true).valid);

  TraceFixture identity;
  identity.prefix();
  identity.record(Point::kComputeDispatchReturned, 2U);
  identity.stop_evidence();
  EXPECT_FALSE(summarize_phase_trace(&identity.counters, true).valid);

  TraceFixture timestamp;
  timestamp.prefix();
  timestamp.record(Point::kComputeDispatchReturned);
  timestamp.stop_evidence();
  timestamp.storage[5].timestamp_ns = timestamp.storage[4].timestamp_ns - 1;
  EXPECT_FALSE(summarize_phase_trace(&timestamp.counters, true).valid);
}

// REQ: RUNTIME-046
TEST(RuntimeGraphPhaseTrace, KeepsUnselectedFrozenCandidateSeparateFromRejectedDispatch) {
  TraceFixture trace;
  trace.completed();
  trace.record(Point::kCandidateFrozen);
  const auto summary = summarize_phase_trace(&trace.counters, true);
  EXPECT_TRUE(summary.valid);
  EXPECT_EQ(summary.complete_activations, 1U);
  EXPECT_EQ(summary.terminal_frozen_candidates, 1U);
  EXPECT_EQ(summary.terminal_run_closing_dispatches, 0U);
  trace.stop_evidence();
  EXPECT_FALSE(summarize_phase_trace(&trace.counters, true).valid);
}

// Counts clock callbacks before the admission barrier rejects the next activation.
// Representative test: StopBeforeClaimProducesAValidTerminalTrace.
class TraceStopSource final : public Operator<> {
 public:
  explicit TraceStopSource(std::shared_ptr<std::atomic_uint32_t> callbacks)
      : callbacks_(std::move(callbacks)) {}

  void setup(OperatorSpec&) override {}

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    ++*callbacks_;
    return {};
  }

 private:
  std::shared_ptr<std::atomic_uint32_t> callbacks_;
};

class RuntimeGraphPhaseTraceStopTest : public ::testing::TestWithParam<bool> {};

// REQ: RUNTIME-046
TEST_P(RuntimeGraphPhaseTraceStopTest, StopBeforeClaimProducesAValidTerminalTrace) {
  // Key setup: finish one callback, then hold the next dispatch before its final admission claim.
  auto callbacks = std::make_shared<std::atomic_uint32_t>();
  Graph graph{"trace-stop-before-claim"};
  auto source = graph.op<TraceStopSource>("source", callbacks);
  graph.partition("owner").add(source);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  CompileOptions options;
  if (GetParam()) {
    options.deployment.bind_thread_placement("owner", ThreadPlacement{});
  }
  auto plan = compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  std::array<detail::RuntimePhaseTraceRecord, 256U> storage{};
  auto counters = std::make_shared<detail::RuntimePhaseCounters>(std::span{storage});
  auto control = std::make_shared<detail::RuntimeTerminationAdmissionTestControl>(
      detail::RuntimeTerminationAdmissionTestControl::Point::kBeforeClaim, 1U, 1U);
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  plan = detail::RuntimeTestPlanAccess::with_termination_admission_test_control(plan, control);
  auto session = run_async(plan);
  const bool arrived = control->wait_until_arrived();
  session.request_stop();
  control->release();
  session.wait();

  // Key assertion: the real Runtime trace passes the same summarizer used by the benchmark.
  ASSERT_TRUE(arrived);
  ASSERT_EQ(session.termination(), RunTermination::kStopped);
  ASSERT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(callbacks->load(), 1U);
  const auto snapshot = counters->snapshot();
  EXPECT_EQ(snapshot[detail::RuntimePhaseCounters::index(Counter::kComputeDispatches)], 2U);
  EXPECT_EQ(snapshot[detail::RuntimePhaseCounters::index(Counter::kComputeAdmissionsRunClosing)],
            1U);
  const auto summary = summarize_phase_trace(counters.get(), true);
  EXPECT_TRUE(summary.valid);
  EXPECT_EQ(summary.dropped, 0U);
  EXPECT_EQ(summary.complete_activations, 1U);
  EXPECT_EQ(summary.terminal_run_closing_dispatches, 1U);
  EXPECT_EQ(summary.terminal_frozen_candidates, 0U);
}

INSTANTIATE_TEST_SUITE_P(Owners, RuntimeGraphPhaseTraceStopTest, ::testing::Bool(),
                         [](const auto& info) { return info.param ? "Worker" : "Inline"; });

}  // namespace
}  // namespace holoscan::benchmarking
