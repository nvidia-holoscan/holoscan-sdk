/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <atomic>
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

class MaxStartDelayDiagnosticSink final : public detail::RuntimeStructuralDiagnosticSink {
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

class MaxStartDelayTerminalSummarySink final : public detail::RuntimeTerminalSummarySink {
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

// Shared atomic state for cross-operator observation in max_start_delay runtime tests.
// now_ns is advanced by MaxStartDelayClockAdvanceOperator; observed_reason and observed_policy
// are written by MaxStartDelayStatusOperator and read by the test after the session completes.
struct MaxStartDelayRuntimeState {
  static void reset(std::int64_t advance_to) noexcept {
    now_ns.store(0, std::memory_order_relaxed);
    requested_advance_ns.store(advance_to, std::memory_order_relaxed);
    advance_callbacks.store(0U, std::memory_order_relaxed);
    max_start_delay_callbacks.store(0U, std::memory_order_relaxed);
    observed_reason.store(ActivationReason::kNominal, std::memory_order_relaxed);
    observed_policy.store(OnViolation::kEmitWitness, std::memory_order_relaxed);
  }

  static inline std::atomic<std::int64_t> now_ns{};
  static inline std::atomic<std::int64_t> requested_advance_ns{};
  static inline std::atomic<std::uint32_t> advance_callbacks{};
  static inline std::atomic<std::uint32_t> max_start_delay_callbacks{};
  static inline std::atomic<ActivationReason> observed_reason{ActivationReason::kNominal};
  static inline std::atomic<OnViolation> observed_policy{OnViolation::kEmitWitness};
};

struct MaxStartDelaySelfBacklogState {
  static void reset() noexcept {
    MaxStartDelayRuntimeState::reset(0);
    callback_count.store(0U, std::memory_order_relaxed);
    first_reason.store(ActivationReason::kInvariantViolated, std::memory_order_relaxed);
    second_reason.store(ActivationReason::kInvariantViolated, std::memory_order_relaxed);
  }

  static inline std::atomic<std::uint32_t> callback_count{};
  static inline std::atomic<ActivationReason> first_reason{ActivationReason::kInvariantViolated};
  static inline std::atomic<ActivationReason> second_reason{ActivationReason::kInvariantViolated};
};

struct MaxStartDelayNoFireState {
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

struct MaxStartDelaySelectionMissState {
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

// Manual deterministic clock driven by MaxStartDelayRuntimeState::now_ns.
// sleep_for/sleep_until advance the atomic rather than blocking, enabling instant
// max_start_delay simulation without any real-time dependency.
class MaxStartDelayRuntimeClock final : public Clock {
 public:
  [[nodiscard]] double time() const override { return timestamp_to_time(timestamp()); }
  [[nodiscard]] std::int64_t timestamp() const override {
    return MaxStartDelayRuntimeState::now_ns.load(std::memory_order_relaxed);
  }
  [[nodiscard]] expected<void, Error> sleep_for(std::int64_t duration_ns) override {
    MaxStartDelayRuntimeState::now_ns.fetch_add(duration_ns, std::memory_order_relaxed);
    return {};
  }
  [[nodiscard]] expected<void, Error> sleep_until(std::int64_t target_time_ns) override {
    MaxStartDelayRuntimeState::now_ns.store(target_time_ns, std::memory_order_relaxed);
    return {};
  }
};

// Two direct witnesses share the same frozen source-decision coordinate. The first callback
// advances the manual clock; the second observes the resulting max_start_delay disposition. Keeping
// both callbacks in one Operator avoids relying on cross-root callback order.
class MaxStartDelayAdvanceThenStatusOperator final : public Operator<> {
 public:
  explicit MaxStartDelayAdvanceThenStatusOperator(OnViolation action) : action_(action) {}

  void setup(OperatorSpec& spec) override { spec.spin_source(source_, "activation").capacity(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = source_}).max_start_delay(5ns).on_violation(action_);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (MaxStartDelayRuntimeState::advance_callbacks.load(std::memory_order_relaxed) == 0U) {
      MaxStartDelayRuntimeState::advance_callbacks.store(1U, std::memory_order_relaxed);
      MaxStartDelayRuntimeState::now_ns.store(
          MaxStartDelayRuntimeState::requested_advance_ns.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      return {};
    }
    const ActivationStatus status = context.activation();
    MaxStartDelayRuntimeState::max_start_delay_callbacks.fetch_add(1U, std::memory_order_relaxed);
    MaxStartDelayRuntimeState::observed_reason.store(status.reason, std::memory_order_relaxed);
    MaxStartDelayRuntimeState::observed_policy.store(status.policy, std::memory_order_relaxed);
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
class MaxStartDelaySelfBacklogOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.spin_source(source_, "activation").capacity(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = source_})
        .max_start_delay(5ns)
        .on_violation(OnViolation::kEmitWitness);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::uint32_t index =
        MaxStartDelaySelfBacklogState::callback_count.fetch_add(1U, std::memory_order_relaxed);
    if (index == 0U) {
      MaxStartDelaySelfBacklogState::first_reason.store(context.activation().reason,
                                                        std::memory_order_relaxed);
      MaxStartDelayRuntimeState::now_ns.store(6, std::memory_order_relaxed);
    } else if (index == 1U) {
      MaxStartDelaySelfBacklogState::second_reason.store(context.activation().reason,
                                                         std::memory_order_relaxed);
    }
    return {};
  }

 private:
  SpinSource source_;
};

class MaxStartDelayFalsePredicateOperator final : public Operator<> {
 public:
  void setup(OperatorSpec&) override {}

  [[nodiscard]] Contract contract() const override {
    auto evaluator =
        predicate("max_start_delay-no-fire",
                  [](const PredicateEvaluationContext&) noexcept -> expected<bool, Error> {
                    MaxStartDelayNoFireState::record_evaluation();
                    return false;
                  });
    Contract result;
    result
        .trigger(OnPredicate{
            .evaluate_on = OnClock{.period = 1ms},
            .predicate = std::move(evaluator),
            .firing = PredicateFiring::kOnEveryTrueEvaluation,
        })
        .max_start_delay(1ns)
        .on_violation(OnViolation::kEmitWitness);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    MaxStartDelayNoFireState::callback_count.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }
};

// A live periodic producer deliberately emits no sample. The independently clocked consumer
// therefore reaches a real Channel-backed Exact+Skip selection miss while carrying a
// max_start_delay.
class MaxStartDelaySelectionMissSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    MaxStartDelaySelectionMissState::record_source_tick();
    return {};
  }

  Output<std::uint32_t> output;
};

class MaxStartDelaySelectionMissSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms})
        .input(input, Exact{})
        .on_missing(input, Skip{})
        .max_start_delay(1ns)
        .on_violation(OnViolation::kEmitWitness);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    MaxStartDelaySelectionMissState::sink_callback_count.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }

  Input<std::uint32_t> input;
};

// Builds one serial two-activation probe using a manual MaxStartDelayRuntimeClock and OnViolation.
[[nodiscard]] ExecutionPlan make_max_start_delay_runtime_plan(OnViolation action) {
  Graph graph{"runtime-max_start_delay-test"};
  const auto probe =
      graph.op<MaxStartDelayAdvanceThenStatusOperator>("advance-then-observe", action);
  graph.partition("runtime-max_start_delay").add(probe);
  const ClockKey default_clock =
      graph.add_clock<MaxStartDelayRuntimeClock>("runtime-max_start_delay-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-max_start_delay",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return test::enable_direct_runtime(compile(graph, std::move(options)), 2U);
}

[[nodiscard]] ExecutionPlan make_self_backlog_plan(std::uint32_t activation_count) {
  Graph graph{"runtime-max_start_delay-self-backlog"};
  const auto backlog = graph.op<MaxStartDelaySelfBacklogOperator>("max_start_delay-self-backlog");
  graph.partition("runtime-max_start_delay-worker").add(backlog);
  const ClockKey default_clock =
      graph.add_clock<MaxStartDelayRuntimeClock>("runtime-max_start_delay-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-max_start_delay-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return test::enable_direct_runtime(compile(graph, std::move(options)), activation_count);
}

[[nodiscard]] ExecutionPlan make_max_start_delay_no_fire_plan() {
  Graph graph{"runtime-max_start_delay-no-fire"};
  const auto predicate = graph.op<MaxStartDelayFalsePredicateOperator>("max_start_delay-no-fire");
  graph.partition("runtime-max_start_delay-worker").add(predicate);
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-max_start_delay-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-max_start_delay-worker", ThreadPlacement{});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_max_start_delay_selection_miss_plan() {
  Graph graph{"runtime-max_start_delay-selection-miss"};
  const auto source = graph.op<MaxStartDelaySelectionMissSourceOperator>("source");
  const auto sink = graph.op<MaxStartDelaySelectionMissSinkOperator>("sink");
  graph.partition("selection-miss-source").add(source);
  graph.partition("selection-miss-sink").add(sink);
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-max_start_delay-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan with_max_start_delay_sink(
    const ExecutionPlan& plan, const std::shared_ptr<MaxStartDelayDiagnosticSink>& sink) {
  return detail::RuntimeTestPlanAccess::with_diagnostic_sink(plan, sink);
}

void expect_max_start_delay_detail(const detail::RuntimeDiagnosticDetail& detail,
                                   OnViolation action, std::uint32_t callback_count) {
  EXPECT_TRUE(detail.present);
  EXPECT_NE(detail.operator_id, 0U);
  EXPECT_TRUE(static_cast<bool>(detail.source));
  EXPECT_TRUE(detail.tag.valid());
  EXPECT_EQ(detail.reference_time, (TimePoint{0, detail.reference_time.clock_id}));
  ASSERT_TRUE(detail.max_start_delay_miss.present);
  EXPECT_EQ(detail.max_start_delay_miss.source_coordinate,
            MaxStartDelaySourceCoordinateKind::kReadinessWitness);
  EXPECT_EQ(detail.max_start_delay_miss.source_decision_time,
            (TimePoint{0, detail.max_start_delay_miss.source_decision_time.clock_id}));
  EXPECT_EQ(detail.max_start_delay_miss.reference_time, detail.reference_time);
  EXPECT_EQ(detail.max_start_delay_miss.budget_ns, 5);
  EXPECT_EQ(detail.max_start_delay_miss.latest_start,
            (TimePoint{5, detail.reference_time.clock_id}));
  EXPECT_EQ(detail.max_start_delay_miss.dispatch_start,
            (TimePoint{6, detail.reference_time.clock_id}));
  EXPECT_EQ(detail.max_start_delay_miss.checked_lateness_ns, 1);
  EXPECT_EQ(detail.max_start_delay_miss.action, action);
  EXPECT_EQ(detail.max_start_delay_miss.callback_count, callback_count);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
// REQ: RUNTIME-081
TEST(MaxStartDelayRuntimeTest, LateSafeDefaultAndEmitWitnessInvokeExactlyOnceWithFrozenStatus) {
  for (const OnViolation action : {OnViolation::kSafeDefault, OnViolation::kEmitWitness}) {
    // Key setup: clock advances to 6 ns which exceeds the 5 ns max_start_delay contract.
    MaxStartDelayRuntimeState::reset(6);
    const auto sink = std::make_shared<MaxStartDelayDiagnosticSink>();
    RunSession session =
        run_async(with_max_start_delay_sink(make_max_start_delay_runtime_plan(action), sink));
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_EQ(MaxStartDelayRuntimeState::advance_callbacks.load(std::memory_order_relaxed), 1U);
    EXPECT_EQ(MaxStartDelayRuntimeState::max_start_delay_callbacks.load(std::memory_order_relaxed),
              1U);
    // Key assertion: observed_reason == kMaxStartDelayExceeded and max_start_delay_callbacks == 1.
    EXPECT_EQ(MaxStartDelayRuntimeState::observed_reason.load(std::memory_order_relaxed),
              ActivationReason::kMaxStartDelayExceeded);
    EXPECT_EQ(MaxStartDelayRuntimeState::observed_policy.load(std::memory_order_relaxed), action);
    ASSERT_EQ(session.diagnostics().size(), 1U);
    EXPECT_EQ(session.diagnostics().front().token, "MAX_START_DELAY_MISS");
    EXPECT_EQ(session.diagnostics().front().severity, DiagnosticSeverity::kWarning);
    EXPECT_TRUE(static_cast<bool>(session.diagnostics().front().source.vertex));
    const auto details = sink->details();
    ASSERT_EQ(details.size(), 1U);
    expect_max_start_delay_detail(details.front(), action, 1U);
  }
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
// REQ: RUNTIME-081
// REQ: RUNTIME-035
// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(MaxStartDelayRuntimeTest, LateFailRunConsumesCandidateWithoutInvokingCompute) {
  // Key setup: kFailRun action; clock advances to 6 ns which exceeds the 5 ns max_start_delay.
  MaxStartDelayRuntimeState::reset(6);
  const auto sink = std::make_shared<MaxStartDelayDiagnosticSink>();
  const auto summary_sink = std::make_shared<MaxStartDelayTerminalSummarySink>();
  const ExecutionPlan with_diagnostics =
      with_max_start_delay_sink(make_max_start_delay_runtime_plan(OnViolation::kFailRun), sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(MaxStartDelayRuntimeState::advance_callbacks.load(std::memory_order_relaxed), 1U);
  // Key assertion: max_start_delay_callbacks == 0 confirms compute was not called on the late
  // candidate.
  EXPECT_EQ(MaxStartDelayRuntimeState::max_start_delay_callbacks.load(std::memory_order_relaxed),
            0U);
  ASSERT_EQ(session.diagnostics().size(), 2U);
  EXPECT_EQ(session.diagnostics().front().token, "MAX_START_DELAY_MISS");
  EXPECT_EQ(session.diagnostics().front().severity, DiagnosticSeverity::kWarning);
  EXPECT_EQ(session.diagnostics()[1].token, "RUNTIME_MAX_START_DELAY_FAIL_RUN");
  EXPECT_EQ(session.diagnostics()[1].severity, DiagnosticSeverity::kError);
  const auto details = sink->details();
  ASSERT_EQ(details.size(), 2U);
  expect_max_start_delay_detail(details.front(), OnViolation::kFailRun, 0U);
  EXPECT_FALSE(details[1].present);
  EXPECT_EQ(summary_sink->attempts(), 1U);
  EXPECT_EQ(summary_sink->diagnostic_count(), session.diagnostics().size());
  EXPECT_EQ(summary_sink->first_diagnostic(), session.diagnostics().front().id);
  EXPECT_EQ(summary_sink->first_token(), "MAX_START_DELAY_MISS");
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
// REQ: RUNTIME-081
TEST(MaxStartDelayRuntimeTest, ExactLatestStartBoundaryRemainsNominal) {
  // Key setup: clock advances to exactly 5 ns, the inclusive latest-start boundary.
  MaxStartDelayRuntimeState::reset(5);
  RunSession session = run_async(make_max_start_delay_runtime_plan(OnViolation::kEmitWitness));
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_EQ(MaxStartDelayRuntimeState::max_start_delay_callbacks.load(std::memory_order_relaxed),
            1U);
  // Key assertion: observed_reason == kNominal confirms the boundary value is not a violation.
  EXPECT_EQ(MaxStartDelayRuntimeState::observed_reason.load(std::memory_order_relaxed),
            ActivationReason::kNominal);
  EXPECT_EQ(MaxStartDelayRuntimeState::observed_policy.load(std::memory_order_relaxed),
            OnViolation::kEmitWitness);
  EXPECT_TRUE(session.diagnostics().empty());
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
// REQ: RUNTIME-081
TEST(MaxStartDelayRuntimeTest, SameOperatorBacklogUsesOriginalFrozenSourceDecision) {
  MaxStartDelaySelfBacklogState::reset();
  const auto sink = std::make_shared<MaxStartDelayDiagnosticSink>();
  RunSession session = run_async(with_max_start_delay_sink(make_self_backlog_plan(2U), sink));
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_EQ(MaxStartDelaySelfBacklogState::callback_count.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(MaxStartDelaySelfBacklogState::first_reason.load(std::memory_order_relaxed),
            ActivationReason::kNominal);
  EXPECT_EQ(MaxStartDelaySelfBacklogState::second_reason.load(std::memory_order_relaxed),
            ActivationReason::kMaxStartDelayExceeded);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().severity, DiagnosticSeverity::kWarning);
  const auto details = sink->details();
  ASSERT_EQ(details.size(), 1U);
  expect_max_start_delay_detail(details.front(), OnViolation::kEmitWitness, 1U);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
// REQ: RUNTIME-081
TEST(MaxStartDelayRuntimeTest, CallbackDurationDoesNotRetroactivelyCreateMaxStartDelayMiss) {
  MaxStartDelaySelfBacklogState::reset();
  RunSession session = run_async(make_self_backlog_plan(1U));
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_EQ(MaxStartDelaySelfBacklogState::callback_count.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(MaxStartDelaySelfBacklogState::first_reason.load(std::memory_order_relaxed),
            ActivationReason::kNominal);
  EXPECT_TRUE(session.diagnostics().empty());
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
// REQ: RUNTIME-081
TEST(MaxStartDelayRuntimeTest, EvaluationOverflowAfterSelectionHasDistinctBoundedCause) {
  MaxStartDelaySelfBacklogState::reset();
  MaxStartDelayRuntimeState::now_ns.store(std::numeric_limits<std::int64_t>::max(),
                                          std::memory_order_relaxed);
  const auto sink = std::make_shared<MaxStartDelayDiagnosticSink>();
  RunSession session = run_async(with_max_start_delay_sink(make_self_backlog_plan(1U), sink));
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(MaxStartDelaySelfBacklogState::callback_count.load(std::memory_order_relaxed), 0U);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const Diagnostic& diagnostic = session.diagnostics().front();
  EXPECT_EQ(diagnostic.token, "RUNTIME_MAX_START_DELAY_EVALUATION_FAILED");
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
  EXPECT_FALSE(detail.max_start_delay_miss.present);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-028
// REQ: RUNTIME-081
TEST(MaxStartDelayRuntimeTest, FalsePredicateNoFireDoesNotCreateMaxStartDelayMiss) {
  MaxStartDelayNoFireState::reset();
  RunSession session = run_async(make_max_start_delay_no_fire_plan());
  ASSERT_TRUE(MaxStartDelayNoFireState::wait_for_evaluation());
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_GE(MaxStartDelayNoFireState::evaluation_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(MaxStartDelayNoFireState::callback_count.load(std::memory_order_relaxed), 0U);
  EXPECT_TRUE(session.diagnostics().empty());
}

// REQ: RUNTIME-018
// REQ: RUNTIME-028
// REQ: RUNTIME-081
TEST(MaxStartDelayRuntimeTest, SelectionMissSkipDoesNotCreateMaxStartDelayMiss) {
  MaxStartDelaySelectionMissState::reset();
  const ExecutionPlan plan = make_max_start_delay_selection_miss_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.max_start_delays().size(), 1U);
  ASSERT_EQ(plan.input_selections().size(), 1U);
  EXPECT_EQ(plan.input_selections().front().missing, TemporalRecordMissingKind::kSkip);

  RunSession session = run_async(plan);
  ASSERT_TRUE(MaxStartDelaySelectionMissState::wait_for_source_ticks(3U));
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_EQ(MaxStartDelaySelectionMissState::sink_callback_count.load(std::memory_order_relaxed),
            0U);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

}  // namespace
}  // namespace holoscan
