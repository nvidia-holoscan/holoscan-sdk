/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_test_access.hpp"
#include "support/direct_port_binding.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

class DeadlineDiagnosticSink final : public detail::RuntimeStructuralDiagnosticSink {
 public:
  void publish(std::span<const Diagnostic> diagnostics) override {
    const std::lock_guard lock(mutex_);
    diagnostics_.assign(diagnostics.begin(), diagnostics.end());
  }

  void publish_runtime_details(std::span<const detail::RuntimeDiagnosticDetail> details) override {
    const std::lock_guard lock(mutex_);
    details_.assign(details.begin(), details.end());
  }

  [[nodiscard]] std::vector<Diagnostic> diagnostics() const {
    const std::lock_guard lock(mutex_);
    return diagnostics_;
  }

  [[nodiscard]] std::vector<detail::RuntimeDiagnosticDetail> details() const {
    const std::lock_guard lock(mutex_);
    return details_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<Diagnostic> diagnostics_;
  std::vector<detail::RuntimeDiagnosticDetail> details_;
};

class DeadlineTerminalSummarySink final : public detail::RuntimeTerminalSummarySink {
 public:
  void publish(const detail::RuntimeTerminalSummary& summary) override {
    const std::lock_guard lock(mutex_);
    ++attempts_;
    diagnostic_count_ = summary.diagnostic_count;
    first_diagnostic_ = summary.first_diagnostic;
    first_token_ = std::string{summary.first_token};
  }

  [[nodiscard]] std::size_t attempts() const {
    const std::lock_guard lock(mutex_);
    return attempts_;
  }

  [[nodiscard]] std::size_t diagnostic_count() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_count_;
  }

  [[nodiscard]] DiagnosticId first_diagnostic() const {
    const std::lock_guard lock(mutex_);
    return first_diagnostic_;
  }

  [[nodiscard]] std::string first_token() const {
    const std::lock_guard lock(mutex_);
    return first_token_;
  }

 private:
  mutable std::mutex mutex_;
  std::size_t attempts_{};
  std::size_t diagnostic_count_{};
  DiagnosticId first_diagnostic_{};
  std::string first_token_;
};

// Shared atomic state for cross-operator observation in deadline runtime tests.
// now_ns is advanced by DeadlineClockAdvanceOperator; observed_reason and observed_policy
// are written by DeadlineStatusOperator and read by the test after the session completes.
struct DeadlineRuntimeState {
  static void reset(std::int64_t advance_to) noexcept {
    now_ns.store(0, std::memory_order_relaxed);
    requested_advance_ns.store(advance_to, std::memory_order_relaxed);
    advance_callbacks.store(0U, std::memory_order_relaxed);
    deadline_callbacks.store(0U, std::memory_order_relaxed);
    observed_reason.store(ActivationReason::kNominal, std::memory_order_relaxed);
    observed_policy.store(OnViolation::kEmitWitness, std::memory_order_relaxed);
  }

  static inline std::atomic<std::int64_t> now_ns{};
  static inline std::atomic<std::int64_t> requested_advance_ns{};
  static inline std::atomic<std::uint32_t> advance_callbacks{};
  static inline std::atomic<std::uint32_t> deadline_callbacks{};
  static inline std::atomic<ActivationReason> observed_reason{ActivationReason::kNominal};
  static inline std::atomic<OnViolation> observed_policy{OnViolation::kEmitWitness};
};

struct DeadlineSelfBacklogState {
  static void reset() noexcept {
    DeadlineRuntimeState::reset(0);
    callback_count.store(0U, std::memory_order_relaxed);
    first_reason.store(ActivationReason::kInvariantViolated, std::memory_order_relaxed);
    second_reason.store(ActivationReason::kInvariantViolated, std::memory_order_relaxed);
  }

  static inline std::atomic<std::uint32_t> callback_count{};
  static inline std::atomic<ActivationReason> first_reason{ActivationReason::kInvariantViolated};
  static inline std::atomic<ActivationReason> second_reason{ActivationReason::kInvariantViolated};
};

struct DeadlineNoFireState {
  static void reset() noexcept {
    evaluation_count.store(0U, std::memory_order_relaxed);
    callback_count.store(0U, std::memory_order_relaxed);
  }

  static void record_evaluation() noexcept {
    evaluation_count.fetch_add(1U, std::memory_order_release);
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_evaluation() {
    std::unique_lock lock(mutex);
    return cv.wait_for(
        lock, 5s, [] { return evaluation_count.load(std::memory_order_acquire) != 0U; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::atomic<std::uint32_t> evaluation_count{};
  static inline std::atomic<std::uint32_t> callback_count{};
};

struct DeadlineSelectionMissState {
  static void reset() noexcept {
    source_tick_count.store(0U, std::memory_order_relaxed);
    sink_callback_count.store(0U, std::memory_order_relaxed);
  }

  static void record_source_tick() noexcept {
    source_tick_count.fetch_add(1U, std::memory_order_release);
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_source_ticks(std::uint32_t count) {
    std::unique_lock lock(mutex);
    return cv.wait_for(
        lock, 5s, [count] { return source_tick_count.load(std::memory_order_acquire) >= count; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::atomic<std::uint32_t> source_tick_count{};
  static inline std::atomic<std::uint32_t> sink_callback_count{};
};

// Manual deterministic clock driven by DeadlineRuntimeState::now_ns.
// sleep_for/sleep_until advance the atomic rather than blocking, enabling instant
// deadline simulation without any real-time dependency.
class DeadlineRuntimeClock final : public Clock {
 public:
  [[nodiscard]] double time() const override { return timestamp_to_time(timestamp()); }
  [[nodiscard]] std::int64_t timestamp() const override {
    return DeadlineRuntimeState::now_ns.load(std::memory_order_relaxed);
  }
  [[nodiscard]] expected<void, Error> sleep_for(std::int64_t duration_ns) override {
    DeadlineRuntimeState::now_ns.fetch_add(duration_ns, std::memory_order_relaxed);
    return {};
  }
  [[nodiscard]] expected<void, Error> sleep_until(std::int64_t target_time_ns) override {
    DeadlineRuntimeState::now_ns.store(target_time_ns, std::memory_order_relaxed);
    return {};
  }
};

// Two direct witnesses share the same frozen source-decision coordinate. The first callback
// advances the manual clock; the second observes the resulting deadline disposition. Keeping
// both callbacks in one Operator avoids relying on cross-root callback order.
class DeadlineAdvanceThenStatusOperator final : public Operator<> {
 public:
  explicit DeadlineAdvanceThenStatusOperator(OnViolation action) : action_(action) {}

  void setup(OperatorSpec& spec) override { spec.spin_source(source_, "activation").capacity(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = source_}).deadline(5ns).on_violation(action_);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (DeadlineRuntimeState::advance_callbacks.load(std::memory_order_relaxed) == 0U) {
      DeadlineRuntimeState::advance_callbacks.store(1U, std::memory_order_relaxed);
      DeadlineRuntimeState::now_ns.store(
          DeadlineRuntimeState::requested_advance_ns.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      return {};
    }
    const ActivationStatus status = context.activation();
    DeadlineRuntimeState::deadline_callbacks.fetch_add(1U, std::memory_order_relaxed);
    DeadlineRuntimeState::observed_reason.store(status.reason, std::memory_order_relaxed);
    DeadlineRuntimeState::observed_policy.store(status.policy, std::memory_order_relaxed);
    return {};
  }

 private:
  SpinSource source_;
  OnViolation action_{OnViolation::kEmitWitness};
};

// Two finite direct witnesses share the same source-decision coordinate. The first callback moves
// the manual clock beyond the 5 ns latest start. With two witnesses, the second callback therefore
// proves same-operator backlog uses its original frozen coordinate. With one witness, the same
// clock movement proves callback duration cannot retroactively create a miss.
class DeadlineSelfBacklogOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.spin_source(source_, "activation").capacity(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = source_}).deadline(5ns).on_violation(OnViolation::kEmitWitness);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::uint32_t index =
        DeadlineSelfBacklogState::callback_count.fetch_add(1U, std::memory_order_relaxed);
    if (index == 0U) {
      DeadlineSelfBacklogState::first_reason.store(context.activation().reason,
                                                   std::memory_order_relaxed);
      DeadlineRuntimeState::now_ns.store(6, std::memory_order_relaxed);
    } else if (index == 1U) {
      DeadlineSelfBacklogState::second_reason.store(context.activation().reason,
                                                    std::memory_order_relaxed);
    }
    return {};
  }

 private:
  SpinSource source_;
};

class DeadlineFalsePredicateOperator final : public Operator<> {
 public:
  void setup(OperatorSpec&) override {}

  [[nodiscard]] Contract contract() const override {
    auto evaluator =
        predicate("deadline-no-fire",
                  [](const PredicateEvaluationContext&) noexcept -> expected<bool, Error> {
                    DeadlineNoFireState::record_evaluation();
                    return false;
                  });
    Contract result;
    result
        .trigger(OnPredicate{
            .evaluate_on = OnClock{.period = 1ms},
            .predicate = std::move(evaluator),
            .firing = PredicateFiring::kOnEveryTrueEvaluation,
        })
        .deadline(1ns)
        .on_violation(OnViolation::kEmitWitness);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    DeadlineNoFireState::callback_count.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }
};

// A live periodic producer deliberately emits no sample. The independently clocked consumer
// therefore reaches a real Channel-backed Exact+Skip selection miss while carrying a deadline.
class DeadlineSelectionMissSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    DeadlineSelectionMissState::record_source_tick();
    return {};
  }

  Output<std::uint32_t> output;
};

class DeadlineSelectionMissSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms})
        .input(input, Exact{})
        .on_missing(input, Skip{})
        .deadline(1ns)
        .on_violation(OnViolation::kEmitWitness);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    DeadlineSelectionMissState::sink_callback_count.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }

  Input<std::uint32_t> input;
};

// Builds one serial two-activation probe using a manual DeadlineRuntimeClock and OnViolation.
[[nodiscard]] ExecutionPlan make_deadline_runtime_plan(OnViolation action) {
  Graph graph{"runtime-deadline-test"};
  const auto probe = graph.op<DeadlineAdvanceThenStatusOperator>("advance-then-observe", action);
  graph.partition("runtime-deadline").add(probe);
  const ClockKey default_clock = graph.add_clock<DeadlineRuntimeClock>("runtime-deadline-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-deadline",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return test::enable_direct_runtime(compile(graph, std::move(options)), 2U);
}

[[nodiscard]] ExecutionPlan make_self_backlog_plan(std::uint32_t activation_count) {
  Graph graph{"runtime-deadline-self-backlog"};
  const auto backlog = graph.op<DeadlineSelfBacklogOperator>("deadline-self-backlog");
  graph.partition("runtime-deadline-worker").add(backlog);
  const ClockKey default_clock = graph.add_clock<DeadlineRuntimeClock>("runtime-deadline-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-deadline-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return test::enable_direct_runtime(compile(graph, std::move(options)), activation_count);
}

[[nodiscard]] ExecutionPlan make_deadline_no_fire_plan() {
  Graph graph{"runtime-deadline-no-fire"};
  const auto predicate = graph.op<DeadlineFalsePredicateOperator>("deadline-no-fire");
  graph.partition("runtime-deadline-worker").add(predicate);
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-deadline-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-deadline-worker", ThreadPlacement{});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_deadline_selection_miss_plan() {
  Graph graph{"runtime-deadline-selection-miss"};
  const auto source = graph.op<DeadlineSelectionMissSourceOperator>("source");
  const auto sink = graph.op<DeadlineSelectionMissSinkOperator>("sink");
  graph.partition("selection-miss-source").add(source);
  graph.partition("selection-miss-sink").add(sink);
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-deadline-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan with_deadline_sink(
    const ExecutionPlan& plan, const std::shared_ptr<DeadlineDiagnosticSink>& sink) {
  return detail::RuntimeTestPlanAccess::with_diagnostic_sink(plan, sink);
}

void expect_deadline_detail(const detail::RuntimeDiagnosticDetail& detail, OnViolation action,
                            std::uint32_t callback_count) {
  EXPECT_TRUE(detail.present);
  EXPECT_NE(detail.operator_id, 0U);
  EXPECT_TRUE(static_cast<bool>(detail.source));
  EXPECT_TRUE(detail.tag.valid());
  EXPECT_EQ(detail.reference_time, (TimePoint{0, detail.reference_time.clock_id}));
  ASSERT_TRUE(detail.deadline_miss.present);
  EXPECT_EQ(detail.deadline_miss.source_coordinate,
            DeadlineSourceCoordinateKind::kReadinessWitness);
  EXPECT_EQ(detail.deadline_miss.source_decision_time,
            (TimePoint{0, detail.deadline_miss.source_decision_time.clock_id}));
  EXPECT_EQ(detail.deadline_miss.reference_time, detail.reference_time);
  EXPECT_EQ(detail.deadline_miss.budget_ns, 5);
  EXPECT_EQ(detail.deadline_miss.latest_start, (TimePoint{5, detail.reference_time.clock_id}));
  EXPECT_EQ(detail.deadline_miss.dispatch_start, (TimePoint{6, detail.reference_time.clock_id}));
  EXPECT_EQ(detail.deadline_miss.checked_lateness_ns, 1);
  EXPECT_EQ(detail.deadline_miss.action, action);
  EXPECT_EQ(detail.deadline_miss.callback_count, callback_count);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
TEST(DeadlineRuntimeTest, LateSafeDefaultAndEmitWitnessInvokeExactlyOnceWithFrozenStatus) {
  for (const OnViolation action : {OnViolation::kSafeDefault, OnViolation::kEmitWitness}) {
    // Key setup: clock advances to 6 ns which exceeds the 5 ns deadline contract.
    DeadlineRuntimeState::reset(6);
    const auto sink = std::make_shared<DeadlineDiagnosticSink>();
    RunSession session = run_async(with_deadline_sink(make_deadline_runtime_plan(action), sink));
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_EQ(DeadlineRuntimeState::advance_callbacks.load(std::memory_order_relaxed), 1U);
    EXPECT_EQ(DeadlineRuntimeState::deadline_callbacks.load(std::memory_order_relaxed), 1U);
    // Key assertion: observed_reason == kDeadlineExceeded and deadline_callbacks == 1.
    EXPECT_EQ(DeadlineRuntimeState::observed_reason.load(std::memory_order_relaxed),
              ActivationReason::kDeadlineExceeded);
    EXPECT_EQ(DeadlineRuntimeState::observed_policy.load(std::memory_order_relaxed), action);
    ASSERT_EQ(session.diagnostics().size(), 1U);
    EXPECT_EQ(session.diagnostics().front().token, "DEADLINE_MISS");
    EXPECT_EQ(session.diagnostics().front().severity, DiagnosticSeverity::kWarning);
    EXPECT_TRUE(static_cast<bool>(session.diagnostics().front().source.vertex));
    const auto details = sink->details();
    ASSERT_EQ(details.size(), 1U);
    expect_deadline_detail(details.front(), action, 1U);
  }
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
// REQ: RUNTIME-035
// REQ: RUNTIME-054
TEST(DeadlineRuntimeTest, LateFailRunConsumesCandidateWithoutInvokingCompute) {
  // Key setup: kFailRun action; clock advances to 6 ns which exceeds the 5 ns deadline.
  DeadlineRuntimeState::reset(6);
  const auto sink = std::make_shared<DeadlineDiagnosticSink>();
  const auto summary_sink = std::make_shared<DeadlineTerminalSummarySink>();
  const ExecutionPlan with_diagnostics =
      with_deadline_sink(make_deadline_runtime_plan(OnViolation::kFailRun), sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(DeadlineRuntimeState::advance_callbacks.load(std::memory_order_relaxed), 1U);
  // Key assertion: deadline_callbacks == 0 confirms compute was not called on the late candidate.
  EXPECT_EQ(DeadlineRuntimeState::deadline_callbacks.load(std::memory_order_relaxed), 0U);
  ASSERT_EQ(session.diagnostics().size(), 2U);
  EXPECT_EQ(session.diagnostics().front().token, "DEADLINE_MISS");
  EXPECT_EQ(session.diagnostics().front().severity, DiagnosticSeverity::kWarning);
  EXPECT_EQ(session.diagnostics()[1].token, "RUNTIME_DEADLINE_FAIL_RUN");
  EXPECT_EQ(session.diagnostics()[1].severity, DiagnosticSeverity::kError);
  const auto details = sink->details();
  ASSERT_EQ(details.size(), 2U);
  expect_deadline_detail(details.front(), OnViolation::kFailRun, 0U);
  EXPECT_FALSE(details[1].present);
  EXPECT_EQ(summary_sink->attempts(), 1U);
  EXPECT_EQ(summary_sink->diagnostic_count(), session.diagnostics().size());
  EXPECT_EQ(summary_sink->first_diagnostic(), session.diagnostics().front().id);
  EXPECT_EQ(summary_sink->first_token(), "DEADLINE_MISS");
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
TEST(DeadlineRuntimeTest, ExactLatestStartBoundaryRemainsNominal) {
  // Key setup: clock advances to exactly 5 ns, the inclusive latest-start boundary.
  DeadlineRuntimeState::reset(5);
  RunSession session = run_async(make_deadline_runtime_plan(OnViolation::kEmitWitness));
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_EQ(DeadlineRuntimeState::deadline_callbacks.load(std::memory_order_relaxed), 1U);
  // Key assertion: observed_reason == kNominal confirms the boundary value is not a violation.
  EXPECT_EQ(DeadlineRuntimeState::observed_reason.load(std::memory_order_relaxed),
            ActivationReason::kNominal);
  EXPECT_EQ(DeadlineRuntimeState::observed_policy.load(std::memory_order_relaxed),
            OnViolation::kEmitWitness);
  EXPECT_TRUE(session.diagnostics().empty());
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
TEST(DeadlineRuntimeTest, SameOperatorBacklogUsesOriginalFrozenSourceDecision) {
  DeadlineSelfBacklogState::reset();
  const auto sink = std::make_shared<DeadlineDiagnosticSink>();
  RunSession session = run_async(with_deadline_sink(make_self_backlog_plan(2U), sink));
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_EQ(DeadlineSelfBacklogState::callback_count.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(DeadlineSelfBacklogState::first_reason.load(std::memory_order_relaxed),
            ActivationReason::kNominal);
  EXPECT_EQ(DeadlineSelfBacklogState::second_reason.load(std::memory_order_relaxed),
            ActivationReason::kDeadlineExceeded);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().severity, DiagnosticSeverity::kWarning);
  const auto details = sink->details();
  ASSERT_EQ(details.size(), 1U);
  expect_deadline_detail(details.front(), OnViolation::kEmitWitness, 1U);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
TEST(DeadlineRuntimeTest, CallbackDurationDoesNotRetroactivelyCreateDeadlineMiss) {
  DeadlineSelfBacklogState::reset();
  RunSession session = run_async(make_self_backlog_plan(1U));
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_EQ(DeadlineSelfBacklogState::callback_count.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(DeadlineSelfBacklogState::first_reason.load(std::memory_order_relaxed),
            ActivationReason::kNominal);
  EXPECT_TRUE(session.diagnostics().empty());
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
TEST(DeadlineRuntimeTest, EvaluationOverflowAfterSelectionHasDistinctBoundedCause) {
  DeadlineSelfBacklogState::reset();
  DeadlineRuntimeState::now_ns.store(std::numeric_limits<std::int64_t>::max(),
                                     std::memory_order_relaxed);
  const auto sink = std::make_shared<DeadlineDiagnosticSink>();
  RunSession session = run_async(with_deadline_sink(make_self_backlog_plan(1U), sink));
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(DeadlineSelfBacklogState::callback_count.load(std::memory_order_relaxed), 0U);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const Diagnostic& diagnostic = session.diagnostics().front();
  EXPECT_EQ(diagnostic.token, "RUNTIME_DEADLINE_EVALUATION_FAILED");
  EXPECT_EQ(diagnostic.severity, DiagnosticSeverity::kError);
  EXPECT_EQ(diagnostic.service, "latest_start");
  EXPECT_TRUE(static_cast<bool>(diagnostic.source.vertex));

  const auto details = sink->details();
  ASSERT_EQ(details.size(), 1U);
  const detail::RuntimeDiagnosticDetail& detail = details.front();
  EXPECT_TRUE(detail.present);
  EXPECT_EQ(detail.error.code, ErrorCode::kResourceExhausted);
  EXPECT_NE(detail.operator_id, 0U);
  EXPECT_TRUE(static_cast<bool>(detail.source));
  EXPECT_NE(detail.source_sequence, 0U);
  EXPECT_TRUE(detail.tag.valid());
  EXPECT_TRUE(detail.reference_time.valid());
  EXPECT_EQ(detail.reference_time.timestamp_ns, std::numeric_limits<std::int64_t>::max());
  EXPECT_EQ(detail.state_generation, detail.tag.generation);
  EXPECT_FALSE(detail.deadline_miss.present);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
TEST(DeadlineRuntimeTest, FalsePredicateNoFireDoesNotCreateDeadlineMiss) {
  DeadlineNoFireState::reset();
  RunSession session = run_async(make_deadline_no_fire_plan());
  ASSERT_TRUE(DeadlineNoFireState::wait_for_evaluation());
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_GE(DeadlineNoFireState::evaluation_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(DeadlineNoFireState::callback_count.load(std::memory_order_relaxed), 0U);
  EXPECT_TRUE(session.diagnostics().empty());
}

// REQ: RUNTIME-018
// REQ: RUNTIME-028
TEST(DeadlineRuntimeTest, SelectionMissSkipDoesNotCreateDeadlineMiss) {
  DeadlineSelectionMissState::reset();
  const ExecutionPlan plan = make_deadline_selection_miss_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.deadlines().size(), 1U);
  ASSERT_EQ(plan.input_selections().size(), 1U);
  EXPECT_EQ(plan.input_selections().front().missing, TemporalRecordMissingKind::kSkip);

  RunSession session = run_async(plan);
  ASSERT_TRUE(DeadlineSelectionMissState::wait_for_source_ticks(3U));
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_EQ(DeadlineSelectionMissState::sink_callback_count.load(std::memory_order_relaxed), 0U);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

}  // namespace
}  // namespace holoscan
