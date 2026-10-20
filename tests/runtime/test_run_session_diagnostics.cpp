/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/synthetic_clock.hpp>

#include "common/allocation_counter.hpp"
#include "common/diagnostic_test_environment.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_local_channel_plan_validation.hpp"
#include "run_session_test_support.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/parameter_update_coordinator.hpp"
#include "runtime/physical_readiness_binding.hpp"
#include "runtime/runtime_callback_context.hpp"
#include "runtime/runtime_partition_test_plan_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/source_wait.hpp"
#include "runtime_channel_binding_test_support.hpp"
#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"

namespace holoscan {

namespace {

using namespace std::chrono_literals;

using namespace test::run_session;

enum class DiagnosticReplayMode { kNone, kConstruct, kDeclaration, kContract };

class DiagnosticReplayProbe final : public Operator<> {
 public:
  explicit DiagnosticReplayProbe(bool direct) : direct_(direct) {
    if (mode.load() == DiagnosticReplayMode::kConstruct) {
      try {
        throw RuntimeError{ErrorCode::kInvalidArgument, "nested factory cause"};
      } catch (...) {
        std::throw_with_nested(RuntimeError{ErrorCode::kNotSupported, "factory cause"});
      }
    }
  }
  void setup(OperatorSpec& spec) override {
    spec.output(output_, "output");
    if (mode.load() == DiagnosticReplayMode::kDeclaration) {
      spec.output(output_, "output");
    }
    if (direct_) {
      spec.spin_source(spin_, "activation").capacity(1U);
    }
  }
  Contract contract() const override {
    Contract result;
    if (direct_) {
      result.trigger(OnSpin{.event = spin_});
    } else {
      result.trigger(OnClock{.period = 1ms});
    }
    if (mode.load() == DiagnosticReplayMode::kContract) {
      result.max_start_delay(0ns).on_violation(OnViolation::kFailRun);
    }
    return result;
  }
  void start() override { callbacks.fetch_add(1U); }
  expected<void, Error> compute(ExecutionContext&) override {
    callbacks.fetch_add(1U);
    return {};
  }
  static inline std::atomic<DiagnosticReplayMode> mode{DiagnosticReplayMode::kNone};
  static inline std::atomic<std::size_t> callbacks{};

 private:
  bool direct_{};
  SpinSource spin_;
  Output<std::uint32_t> output_;
};

TEST(RunSessionTest, FactoryAndReturnedReplayErrorsRetainStructuredEvidence) {
  struct Reset {
    ~Reset() { DiagnosticReplayProbe::mode.store(DiagnosticReplayMode::kNone); }
  };
  const Reset reset;
  for (const bool direct : {false, true}) {
    DiagnosticReplayProbe::mode.store(DiagnosticReplayMode::kNone);
    Graph graph{"diagnostic-replay"};
    const auto probe = graph.op<DiagnosticReplayProbe>("probe", direct);
    graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
    CompileOptions options;
    if (direct) {
      graph.partition("worker").add(probe);
      options.deployment.bind_thread_placement("worker",
                                               ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
    }
    const auto compiled = compile(graph, options);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto plan = direct ? test::enable_direct_runtime(compiled) : compiled;
    for (const auto mode : {DiagnosticReplayMode::kDeclaration,
                            DiagnosticReplayMode::kContract,
                            DiagnosticReplayMode::kConstruct}) {
      SCOPED_TRACE(static_cast<int>(mode));
      SCOPED_TRACE(direct);
      DiagnosticReplayProbe::mode.store(mode);
      DiagnosticReplayProbe::callbacks.store(0U);
      auto session = run_async(plan);
      session.wait();
      ASSERT_EQ(session.termination(), RunTermination::kFailed);
      EXPECT_EQ(DiagnosticReplayProbe::callbacks.load(), 0U);
      ASSERT_EQ(session.diagnostics().size(), 1U);
      const auto& diagnostic = session.diagnostics().front();
      const auto& retained_detail = diagnostic.detail;
      if (!retained_detail.has_value()) {
        FAIL() << "Expected diagnostic detail";
      }
      const auto& evidence = *retained_detail;
      EXPECT_EQ(evidence.operator_path, "probe");
      if (mode == DiagnosticReplayMode::kConstruct) {
        EXPECT_EQ(diagnostic.token, "RUNTIME_OPERATOR_RECONSTRUCTION_FAILED");
        EXPECT_EQ(evidence.operation, DiagnosticOperation::kConstruct);
        EXPECT_EQ(evidence.error.code, ErrorCode::kNotSupported);
        EXPECT_EQ(evidence.error.message, "factory cause");
        ASSERT_EQ(evidence.causes.size(), 1U);
        EXPECT_EQ(evidence.causes.front().code, ErrorCode::kInvalidArgument);
        EXPECT_EQ(evidence.causes.front().message, "nested factory cause");
      } else {
        EXPECT_EQ(diagnostic.token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
        EXPECT_TRUE(diagnostic.required.starts_with("sha256:"));
        EXPECT_TRUE(diagnostic.provided.starts_with("sha256:"));
        EXPECT_NE(diagnostic.required, diagnostic.provided);
        EXPECT_EQ(evidence.operation,
                  mode == DiagnosticReplayMode::kDeclaration ? DiagnosticOperation::kSetup
                                                             : DiagnosticOperation::kContract);
        EXPECT_EQ(evidence.error.code,
                  mode == DiagnosticReplayMode::kDeclaration ? ErrorCode::kDuplicateName
                                                             : ErrorCode::kInvalidArgument);
        EXPECT_NE(evidence.error.message.find(mode == DiagnosticReplayMode::kDeclaration
                                                  ? "duplicated"
                                                  : "max_start_delay budget must be positive"),
                  std::string::npos);
      }
    }
  }
}

class FactoryCaptureOperator final : public Operator<> {
 public:
  explicit FactoryCaptureOperator(std::exception_ptr failure) {
    std::rethrow_exception(std::move(failure));
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

class FactoryCaptureClock final : public Clock {
 public:
  explicit FactoryCaptureClock(std::exception_ptr failure) {
    std::rethrow_exception(std::move(failure));
  }
  double time() const override { return 0; }
  std::int64_t timestamp() const override { return 0; }
  expected<void, Error> sleep_for(std::int64_t) override { return {}; }
  expected<void, Error> sleep_until(std::int64_t) override { return {}; }
};

TEST(RunSessionTest, OperatorAndClockFactoryCaptureSurvivesAllocationFailure) {
  std::exception_ptr failure;
  try {
    throw RuntimeError{ErrorCode::kInvalidArgument,
                       "original factory exception survives allocation failure"};
  } catch (...) {
    failure = std::current_exception();
  }
  const detail::OperatorFactoryModel<FactoryCaptureOperator, std::exception_ptr> op_factory{
      failure};
  const detail::ClockFactoryModel<FactoryCaptureClock, std::exception_ptr> clock_factory{failure};
  const auto verify = [](const auto& factory) {
    detail::DiagnosticCapture capture;
    const auto created = [&] {
      const test::ScopedAllocationFailure fail_after_object_allocation{1U};
      return factory.create(&capture);
    }();
    ASSERT_FALSE(created);
    EXPECT_EQ(created.error().code, ErrorCode::kInvalidArgument);
    ASSERT_EQ(capture.count, 1U);
    const auto evidence = capture.detail(DiagnosticOperation::kConstruct);
    EXPECT_EQ(evidence.error.message, "original factory exception survives allocation failure");
    EXPECT_FALSE(evidence.truncated);
  };
  verify(op_factory);
  verify(clock_factory);
}

TEST(RunSessionTest, ConcurrentFactoryCapturesRetainIndependentBoundedCauses) {
  std::exception_ptr failure;
  try {
    throw RuntimeError{ErrorCode::kInvalidArgument, std::string(8192U, 'x')};
  } catch (...) {
    failure = std::current_exception();
  }
  for (std::size_t depth = 0U; depth < 5U; ++depth) {
    try {
      try {
        std::rethrow_exception(failure);
      } catch (...) {
        std::throw_with_nested(RuntimeError{ErrorCode::kFailure, "outer cause"});
      }
    } catch (...) {
      failure = std::current_exception();
    }
  }
  const detail::OperatorFactoryModel<FactoryCaptureOperator, std::exception_ptr> factory{failure};
  std::array<detail::DiagnosticCapture, 2U> captures;
  std::thread first([&] { static_cast<void>(factory.create(&captures[0])); });
  std::thread second([&] { static_cast<void>(factory.create(&captures[1])); });
  first.join();
  second.join();
  for (const auto& capture : captures) {
    EXPECT_EQ(capture.count, detail::DiagnosticCapture::kCauseCapacity);
    EXPECT_TRUE(capture.truncated);
    const auto evidence = capture.detail(DiagnosticOperation::kConstruct);
    EXPECT_EQ(evidence.error.message, "outer cause");
    EXPECT_EQ(evidence.causes.size(), 3U);
  }
}

TEST(RunSessionTest, ClockReconstructionRetainsNestedFactoryEvidence) {
  DiagnosticReplayProbe::mode.store(DiagnosticReplayMode::kNone);
  Graph graph{"clock-failure"};
  graph.op<DiagnosticReplayProbe>("probe", false);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const auto plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  std::exception_ptr failure;
  try {
    try {
      throw RuntimeError{ErrorCode::kNotFound, "clock device missing"};
    } catch (...) {
      std::throw_with_nested(RuntimeError{ErrorCode::kNotSupported, "clock construction failed"});
    }
  } catch (...) {
    failure = std::current_exception();
  }
  auto state =
      std::make_shared<detail::ExecutionPlanState>(*detail::ExecutionPlanAccess::retain(plan));
  ASSERT_EQ(state->executable_factories.clocks.size(), 1U);
  state->executable_factories.clocks.front().factory =
      std::make_shared<detail::ClockFactoryModel<FactoryCaptureClock, std::exception_ptr>>(failure);
  detail::RuntimeDiagnosticDetail evidence;
  const auto reconstructed = detail::GraphExec::reconstruct(state, 1U, {}, nullptr, &evidence);
  ASSERT_FALSE(reconstructed);
  ASSERT_TRUE(evidence.capture);
  EXPECT_EQ(evidence.operation, DiagnosticOperation::kConstruct);
  EXPECT_EQ(evidence.component_path, "clock/clock");
  EXPECT_FALSE(evidence.vertex);
  const auto captured = evidence.capture->detail(evidence.operation);
  EXPECT_EQ(captured.error.message, "clock construction failed");
  ASSERT_EQ(captured.causes.size(), 1U);
  EXPECT_EQ(captured.causes.front().code, ErrorCode::kNotFound);
  EXPECT_EQ(captured.causes.front().message, "clock device missing");
}

class CleanupOverflowProbe final : public Operator<> {
 public:
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    context.request_completion();
    return {};
  }
  void stop() override { throw RuntimeError{ErrorCode::kFailure, "cleanup overflow cause"}; }
};

TEST(RunSessionTest, LifecycleFailureOverflowIsExplicitAndKeepsThePrimary) {
  Graph graph{"lifecycle-failure-overflow"};
  for (std::size_t index = 0U; index < 6U; ++index) {
    graph.op<CleanupOverflowProbe>("probe" + std::to_string(index));
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const auto plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = run_async(plan);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 4U);
  for (std::size_t index = 0U; index < session.diagnostics().size(); ++index) {
    const auto& diagnostic = session.diagnostics()[index];
    const auto& retained_detail = diagnostic.detail;
    if (!retained_detail.has_value()) {
      FAIL() << "Expected diagnostic detail";
    }
    EXPECT_EQ(retained_detail->role, DiagnosticFailureRole::kTeardown);
    EXPECT_EQ(retained_detail->error.message, "cleanup overflow cause");
    EXPECT_EQ(retained_detail->truncated, index == 3U);
    EXPECT_EQ(retained_detail->primary_diagnostic_index,
              index == 0U ? std::nullopt : std::optional<std::uint32_t>{0U});
  }
}

// REQ: RUNTIME-001
// REQ: RUNTIME-004
// REQ: RUNTIME-035
// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(RunSessionTest, BlockingAndAsyncRunsPublishTheExactSameTerminalDiagnostics) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  const auto sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(sink);
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);
  ASSERT_EQ(plan.json(), baseline.json());
  ASSERT_EQ(plan.digest(), baseline.digest());

  test::RuntimeProbeState::reset();
  RunSession session = run_async(plan);
  session.wait();
  const std::vector<Diagnostic> async_diagnostics{session.diagnostics().begin(),
                                                  session.diagnostics().end()};
  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(async_diagnostics.empty());
  EXPECT_EQ(async_diagnostics.front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  EXPECT_TRUE(async_diagnostics.front().id);

  test::RuntimeProbeState::reset();
  EXPECT_NE(run(plan, RunSignalPolicy::kCallerManaged), 0);
  const auto published = sink->snapshot();
  ASSERT_EQ(published.size(), 2U);
  expect_diagnostic_sequences_equal(async_diagnostics, published[0]);
  expect_diagnostic_sequences_equal(async_diagnostics, published[1]);

  const auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 2U);
  for (std::size_t index = 0U; index < summaries.size(); ++index) {
    const auto& summary = summaries[index];
    EXPECT_EQ(summary.category, detail::RuntimeTerminalCategory::kFailed);
    EXPECT_EQ(summary.diagnostic_count, async_diagnostics.size());
    EXPECT_EQ(summary.first_diagnostic, async_diagnostics.front().id);
    EXPECT_EQ(summary.first_token, async_diagnostics.front().token);
    EXPECT_EQ(summary.structured_publication_count, index + 1U);
  }
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: RUNTIME-004
TEST(RunSessionTest, StructuralDiagnosticSinkFailureCannotChangeRunOutcome) {
  test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<RecordingDiagnosticSink>(true);
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, sink);

  test::RuntimeProbeState::reset();
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  test::RuntimeProbeState::reset();
  EXPECT_NE(run(plan, RunSignalPolicy::kCallerManaged), 0);
  EXPECT_TRUE(sink->snapshot().empty());
}

// The test sink's failure injection must cover every callback so a future Runtime publication
// surface cannot silently bypass the containment path exercised above.
// REQ: RUNTIME-004
TEST(RunSessionTest, StructuralDiagnosticSinkFailureInjectionCoversEveryCallback) {
  RecordingDiagnosticSink sink{true};
  EXPECT_THROW(sink.publish(std::span<const Diagnostic>{}), std::runtime_error);
  EXPECT_THROW(sink.publish_runtime_details(std::span<const detail::RuntimeDiagnosticDetail>{}),
               std::runtime_error);
  EXPECT_THROW(
      sink.publish_lifecycle_receipts(std::span<const detail::RuntimeLifecycleReceiptDiagnostic>{}),
      std::runtime_error);
  EXPECT_THROW(
      sink.publish_lane_startup_evidence(std::span<const detail::ExecutionLaneStartupEvidence>{}),
      std::runtime_error);
  EXPECT_THROW(sink.publish_operator_completion_progress(
                   std::span<const OperatorCompletionProgressRecord>{}),
               std::runtime_error);
}
#endif

class AutomaticDiagnosticPreferenceGuard {
 public:
  AutomaticDiagnosticPreferenceGuard() : previous_(automatic_diagnostic_logging_enabled()) {}
  ~AutomaticDiagnosticPreferenceGuard() { set_automatic_diagnostic_logging_enabled(previous_); }

 private:
  bool previous_;
};

// REQ: RUNTIME-060
// REQ: EXPECTED-032
TEST(RunSessionTest, AutomaticOptOutPreservesStructuredSummaryAndDiagnosticSinks) {
  const AutomaticDiagnosticPreferenceGuard guard;
  for (const bool enabled : {false, true}) {
    set_automatic_diagnostic_logging_enabled(enabled);
    test::RuntimeProbeState::reset();
    const auto diagnostics = std::make_shared<RecordingDiagnosticSink>();
    const auto summaries = std::make_shared<RecordingTerminalSummarySink>(diagnostics);
    const auto baseline = test::make_runtime_test_plan();
    const auto with_diagnostics =
        detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostics);
    const auto plan =
        detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summaries);
    auto session = run_async(plan);
    session.wait();
    ASSERT_EQ(session.termination(), RunTermination::kFailed);
    ASSERT_EQ(session.diagnostics().size(), 1U);
    EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
    ASSERT_EQ(diagnostics->snapshot().size(), 1U);
    ASSERT_EQ(summaries->snapshot().size(), 1U);
    EXPECT_EQ(summaries->snapshot().front().structured_publication_count, 1U);
    EXPECT_EQ(summaries->snapshot().front().diagnostic_count, session.diagnostics().size());
    EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
    EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
    session.wait();
    EXPECT_EQ(diagnostics->snapshot().size(), 1U);
    EXPECT_EQ(summaries->snapshot().size(), 1U);
  }
}

// REQ: RUNTIME-060
// REQ: EXPECTED-032
TEST(RunSessionTest, PendingTerminalReportSamplesCurrentAutomaticPolicy) {
  const test::ScopedDiagnosticLogger logger{LogLevel::ERROR};
  const AutomaticDiagnosticPreferenceGuard guard;
  for (const bool initial : {true, false}) {
    test::RuntimeProbeState::reset();
    const auto plan = test::enable_direct_runtime(test::make_runtime_test_plan());
    test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStopThenFail);
    set_automatic_diagnostic_logging_enabled(initial);
    auto session = run_async(plan);
    ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());
    set_automatic_diagnostic_logging_enabled(!initial);
    testing::internal::CaptureStderr();
    session.request_stop();
    session.wait();
    const auto output = testing::internal::GetCapturedStderr();
    EXPECT_EQ(output.find("RunSession failed:") != std::string::npos, !initial);
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_FALSE(session.diagnostics().empty());
    EXPECT_EQ(test::RuntimeProbeState::active.load(std::memory_order_acquire), 0U);
    EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
  }
}

// REQ: RUNTIME-060
// REQ: EXPECTED-033
TEST(RunSessionTest, ExplicitTerminalReportRemainsAvailableAfterAutomaticOptOut) {
  const test::ScopedDiagnosticLogger logger{LogLevel::ERROR};
  const AutomaticDiagnosticPreferenceGuard guard;
  set_automatic_diagnostic_logging_enabled(false);
  const auto plan = test::make_runtime_test_plan();
  testing::internal::CaptureStderr();
  auto session = run_async(plan);
  session.wait();
  const auto automatic_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(automatic_output.empty());
  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  const auto before = format_diagnostics(session.diagnostics(), plan.graph());
  testing::internal::CaptureStderr();
  log_diagnostics(session.diagnostics(), plan.graph());
  const auto explicit_output = testing::internal::GetCapturedStderr();
  EXPECT_NE(explicit_output.find("RUNTIME_READINESS_BINDING_UNAVAILABLE"), std::string::npos);
  EXPECT_EQ(explicit_output.find("RunSession failed:"), std::string::npos);
  EXPECT_EQ(format_diagnostics(session.diagnostics(), plan.graph()), before);
  testing::internal::CaptureStderr();
  log_diagnostics(session.diagnostics(), plan.graph());
  EXPECT_EQ(testing::internal::GetCapturedStderr(), explicit_output);
  testing::internal::CaptureStderr();
  session.wait();
  static_cast<void>(session.diagnostics());
  static_cast<void>(session.termination());
  EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
}

// REQ: EXPECTED-033
// REQ: RUNTIME-060
TEST(RunSessionTest, ExplicitHelperReportsPublishedWarningPrefixWithoutWaiting) {
  const test::ScopedDiagnosticLogger logger{LogLevel::WARN};
  const AutomaticDiagnosticPreferenceGuard guard;
  set_automatic_diagnostic_logging_enabled(false);
  test::RuntimeComputeCallChannelProbeState::reset();
  const auto plan = test::make_runtime_compute_call_channel_plan(
      1000000U, test::RuntimeComputeCallAction::kRestartRollbackAfterFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (session.diagnostics().empty() && session.is_running() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  const auto prefix = session.diagnostics();
  ASSERT_FALSE(prefix.empty());
  ASSERT_EQ(prefix.front().severity, DiagnosticSeverity::kWarning);
  const auto token = std::string{prefix.front().token};
  ASSERT_TRUE(session.is_running());
  testing::internal::CaptureStderr();
  log_diagnostics(prefix, plan.graph());
  const auto early_output = testing::internal::GetCapturedStderr();
  EXPECT_NE(early_output.find(token), std::string::npos);
  EXPECT_TRUE(session.is_running());
  EXPECT_EQ(prefix.front().severity, DiagnosticSeverity::kWarning);

  testing::internal::CaptureStderr();
  session.request_stop();
  session.wait();
  EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_EQ(prefix.front().token, token);
  testing::internal::CaptureStderr();
  log_diagnostics(session.diagnostics(), plan.graph());
  const auto terminal_output = testing::internal::GetCapturedStderr();
  EXPECT_NE(terminal_output.find(token), std::string::npos);
}

// REQ: RUNTIME-035
// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(RunSessionTest, TerminalFailureSummaryFollowsStructuredPublicationExactlyOnce) {
  test::RuntimeProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(diagnostic_sink);
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);

  RunSession session = run_async(plan);
  session.wait();

  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 1U);
  const TerminalSummarySnapshot& summary = summaries.front();
  EXPECT_NE(summary.run_generation, 0U);
  EXPECT_EQ(summary.category, detail::RuntimeTerminalCategory::kFailed);
  EXPECT_EQ(summary.diagnostic_count, session.diagnostics().size());
  EXPECT_EQ(summary.first_diagnostic, session.diagnostics().front().id);
  EXPECT_EQ(summary.first_source.graph_generation,
            session.diagnostics().front().source.graph_generation);
  EXPECT_EQ(summary.first_source.graph, session.diagnostics().front().source.graph);
  EXPECT_EQ(summary.first_source.vertex, session.diagnostics().front().source.vertex);
  EXPECT_EQ(summary.first_source.port, session.diagnostics().front().source.port);
  EXPECT_EQ(summary.first_token, session.diagnostics().front().token);
  EXPECT_EQ(summary.structured_publication_count, 1U);
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: RUNTIME-035
// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(RunSessionTest, TerminalSummarySinkFailureCannotChangeCleanupOrDiagnostics) {
  test::RuntimeProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(diagnostic_sink, true);
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  const auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 1U);
  EXPECT_EQ(summaries.front().structured_publication_count, 1U);
  EXPECT_EQ(test::RuntimeProbeState::active.load(std::memory_order_acquire), 0U);
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
}
#endif

// REQ: RUNTIME-035
// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(RunSessionTest, CompletedRecurringActivationsAttemptNoTerminalFailureSummary) {
  constexpr std::uint32_t kActivationCount = 3U;
  test::RuntimeProbeState::reset();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>();
  const ExecutionPlan direct =
      test::enable_direct_runtime(test::make_runtime_test_plan(), kActivationCount);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(direct, summary_sink);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_TRUE(summary_sink->snapshot().empty());
  EXPECT_EQ(std::ranges::count(test::RuntimeProbeState::snapshot(), 12), kActivationCount);
}

// REQ: RUNTIME-035
// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(RunSessionTest, CooperativelyStoppedSessionAttemptsNoTerminalFailureSummary) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>();
  const ExecutionPlan direct = test::enable_direct_runtime(test::make_runtime_test_plan());
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(direct, summary_sink);

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_TRUE(summary_sink->snapshot().empty());
}

// REQ: RUNTIME-035
// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(RunSessionTest, RepeatedFailedRunsEachAttemptOneTerminalSummary) {
  constexpr std::size_t kRunCount = 4U;
  test::RuntimeProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(diagnostic_sink);
  const ExecutionPlan with_diagnostics = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      test::make_runtime_test_plan(), diagnostic_sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);

  for (std::size_t run = 0U; run < kRunCount; ++run) {
    RunSession session = run_async(plan);
    session.wait();
    ASSERT_EQ(session.termination(), RunTermination::kFailed);
    ASSERT_EQ(session.diagnostics().size(), 1U);
  }

  const auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), kRunCount);
  for (std::size_t run = 0U; run < summaries.size(); ++run) {
    EXPECT_EQ(summaries[run].category, detail::RuntimeTerminalCategory::kFailed);
    EXPECT_EQ(summaries[run].diagnostic_count, 1U);
    EXPECT_EQ(summaries[run].structured_publication_count, run + 1U);
  }
}

// A source-tree-only seed places the run-owned allocator at its final valid sequence. The first
// witness executes with that identity; the next accepted witness must stop the session with the
// exact bounded diagnostic rather than wrap, reuse a Tag, or collapse the cause into generic
// readiness failure.
// REQ: RUNTIME-007
TEST(RunSessionTest, TrueTagSequenceExhaustionPublishesExactTerminalDiagnostic) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan direct = test::enable_direct_runtime(test::make_runtime_test_plan(), 2U);
  const ExecutionPlan seeded = detail::RuntimeTestPlanAccess::with_initial_tag_sequence_for_test(
      direct, std::numeric_limits<std::uint64_t>::max());
  const auto sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(seeded, sink);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_TAG_CAPACITY_EXHAUSTED");
  EXPECT_EQ(session.diagnostics().front().service, "candidate_freeze");
  const Diagnostic& diagnostic = session.diagnostics().front();
  EXPECT_FALSE(diagnostic.source.vertex);
  if (!diagnostic.detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  const auto& detail = *diagnostic.detail;
  EXPECT_EQ(detail.error.code, ErrorCode::kResourceExhausted);
  EXPECT_TRUE(detail.error.message.empty());
  EXPECT_TRUE(detail.causes.empty());
  EXPECT_FALSE(detail.truncated);
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));

  const auto published = sink->snapshot_details();
  ASSERT_EQ(published.size(), 1U);
  ASSERT_EQ(published.front().size(), 1U);
  EXPECT_TRUE(published.front().front().present);
  EXPECT_EQ(published.front().front().error.code, ErrorCode::kResourceExhausted);
}

}  // namespace
}  // namespace holoscan
