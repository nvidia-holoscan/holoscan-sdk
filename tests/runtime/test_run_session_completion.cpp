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
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/time/synthetic_clock.hpp>

#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_placement_internal.hpp"
#include "run_session_test_support.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/parameter_update_coordinator.hpp"
#include "runtime/physical_readiness_binding.hpp"
#include "runtime/runtime_callback_context.hpp"
#include "runtime/runtime_partition_test_plan_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_worker.hpp"
#include "runtime/source_wait.hpp"
#include "runtime_channel_binding_test_support.hpp"
#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"

namespace holoscan {

namespace {

using namespace std::chrono_literals;

using namespace test::run_session;

class ThrowingTerminalClosePublisher final : public channel_internal::InlinePayloadPublisher {
 public:
  explicit ThrowingTerminalClosePublisher(
      std::shared_ptr<std::atomic<std::size_t>> close_count) noexcept
      : close_count_(std::move(close_count)) {}

  [[nodiscard]] channel_internal::InlinePublicationResult emit_phase(
      std::span<const std::byte>, const SampleMetadata&) override {
    return channel_internal::InlinePublicationResult{
        .phase = channel_internal::InlinePublicationPhase::kAccepted,
        .recipient_count = 1U,
        .error = Error{ErrorCode::kSuccess},
    };
  }

  [[nodiscard]] expected<void, Error> close_stream() override {
    close_count_->fetch_add(1U, std::memory_order_relaxed);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    throw std::runtime_error{"injected terminal close failure"};
#else
    return make_unexpected(Error{ErrorCode::kNotSupported});
#endif
  }

  [[nodiscard]] expected<void, Error> update_connections() noexcept override { return {}; }
  [[nodiscard]] bool has_subscribers() const noexcept override { return true; }

 private:
  std::shared_ptr<std::atomic<std::size_t>> close_count_;
};

class ThrowingTerminalCloseChannelBindControl final
    : public detail::RuntimeChannelSessionBindTestControl {
 public:
  explicit ThrowingTerminalCloseChannelBindControl(const PortId output) noexcept
      : output_(output), close_count_(std::make_shared<std::atomic<std::size_t>>()) {}

  [[nodiscard]] expected<void, Error> bind(
      detail::RuntimeChannelSession& session) noexcept override {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
      auto first = std::make_unique<ThrowingTerminalClosePublisher>(close_count_);
      std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
      publishers.push_back(std::move(first));
      return detail::RuntimeChannelSessionTestAccess::replace_output_publishers(
          session, output_, std::move(publishers));
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
#else
    static_cast<void>(session);
    return make_unexpected(Error{ErrorCode::kNotSupported});
#endif
  }

  [[nodiscard]] std::size_t first_close_count() const noexcept {
    return close_count_->load(std::memory_order_relaxed);
  }

 private:
  PortId output_{};
  std::shared_ptr<std::atomic<std::size_t>> close_count_;
};

// REQ: RUNTIME-050
class RuntimeCompletionFenceSource final : public Operator<> {
 public:
  explicit RuntimeCompletionFenceSource(bool with_deadline = false) noexcept
      : with_deadline_(with_deadline) {}

  void setup(OperatorSpec& spec) override {
    spec.spin_source(spin_, "activation").capacity(1U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = spin_});
    if (with_deadline_) {
      result.max_start_delay(1h).on_violation(OnViolation::kEmitWitness);
    }
    result.complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    compute_count.fetch_add(1U, std::memory_order_relaxed);
    return output.emit(RuntimeCompletionFencePayload{.value = 41U});
  }

  static inline std::atomic<std::size_t> compute_count{};
  Output<RuntimeCompletionFencePayload> output;

 private:
  SpinSource spin_;
  bool with_deadline_{};
};

class RuntimeCompletionFenceSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    observed_value.store(sample->data.value, std::memory_order_relaxed);
    compute_count.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }

  static inline std::atomic<std::size_t> compute_count{};
  static inline std::atomic<std::uint32_t> observed_value{};
  Input<RuntimeCompletionFencePayload> input;
};

[[nodiscard]] ExecutionPlan make_runtime_completion_fence_plan(bool with_deadline = false) {
  Graph graph{"runtime-completion-final-entry-fence"};
  const auto source = graph.op<RuntimeCompletionFenceSource>("source", with_deadline);
  const auto sink = graph.op<RuntimeCompletionFenceSink>("sink");
  graph.add_flow(source->output,
                 sink->input,
                 ConnectionOptions{
                     .queue_policy = QueuePolicy::kBlocking,
                     .queue_depth = 1U,
                 });
  graph.partition("runtime-completion-final-entry-fence").add(source).add(sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-completion-fence-clock"));
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-completion-final-entry-fence",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

void reset_runtime_completion_fence_probe() noexcept {
  RuntimeCompletionFenceSource::compute_count.store(0U, std::memory_order_relaxed);
  RuntimeCompletionFenceSink::compute_count.store(0U, std::memory_order_relaxed);
  RuntimeCompletionFenceSink::observed_value.store(0U, std::memory_order_relaxed);
}

[[nodiscard]] ExecutionPlan make_process_group_completion_progress_test_plan() {
  Graph graph{"runtime-process-group-completion-progress"};
  const auto first_source = graph.op<test::RuntimeComputeCallSourceOperator>(
      "a-source", 1U, test::RuntimeComputeCallAction::kNone);
  const auto first_sink = graph.op<test::RuntimeComputeCallSinkOperator>("a-sink");
  const auto second_source = graph.op<test::RuntimeComputeCallSourceOperator>(
      "z-source", 2U, test::RuntimeComputeCallAction::kNone);
  const auto second_sink = graph.op<test::RuntimeComputeCallSinkOperator>("z-sink");
  graph.partition("runtime-completion-first-source").add(first_source);
  graph.partition("runtime-completion-first-sink").add(first_sink);
  graph.partition("runtime-completion-second-source").add(second_source);
  graph.partition("runtime-completion-second-sink").add(second_sink);
  graph.add_flow(first_source->output, first_sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(second_source->output, second_sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock =
      graph.add_clock<RealtimeClock>("runtime-process-group-completion-progress-clock");
  graph.set_default_clock(default_clock);

  constexpr std::array first_selector{
      detail::GraphObjectSelector{.graph_path = "a-source"},
      detail::GraphObjectSelector{.graph_path = "a-sink"},
  };
  constexpr std::array second_selector{
      detail::GraphObjectSelector{.graph_path = "z-source"},
      detail::GraphObjectSelector{.graph_path = "z-sink"},
  };
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("first",
                         detail::ProcessPlacement{
                             .members = first_selector,
                             .launch_group = "first-process",
                         })
      .bind_process_role("second",
                         detail::ProcessPlacement{
                             .members = second_selector,
                             .launch_group = "second-process",
                         });
  return detail::compile_with_process_placement(graph, CompileOptions{}, std::move(placement));
}

[[nodiscard]] ExecutionPlan make_unequal_compute_call_sources_test_plan() {
  Graph graph{"runtime-unequal-compute-call-sources"};
  const auto first_source = graph.op<test::RuntimeComputeCallSourceOperator>(
      "a-source", 1U, test::RuntimeComputeCallAction::kNone);
  const auto first_sink = graph.op<test::RuntimeComputeCallSinkOperator>("a-sink");
  const auto second_source = graph.op<test::RuntimeComputeCallSourceOperator>(
      "z-source", 3U, test::RuntimeComputeCallAction::kNone);
  const auto second_sink = graph.op<test::RuntimeComputeCallSinkOperator>("z-sink");
  graph.partition("runtime-unequal-first-source").add(first_source);
  graph.partition("runtime-unequal-first-sink").add(first_sink);
  graph.partition("runtime-unequal-second-source").add(second_source);
  graph.partition("runtime-unequal-second-sink").add(second_sink);
  graph.add_flow(first_source->output, first_sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(second_source->output, second_sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock =
      graph.add_clock<RealtimeClock>("runtime-unequal-compute-call-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_finite_and_unbounded_source_test_plan() {
  Graph graph{"runtime-finite-and-unbounded-sources"};
  const auto finite = graph.op<test::RuntimeComputeCallSourceOperator>(
      "finite", 1U, test::RuntimeComputeCallAction::kNone);
  const auto sink = graph.op<test::RuntimeComputeCallSinkOperator>("finite-sink");
  const auto unbounded = graph.op<test::RuntimeClockProbeOperator>("unbounded");
  graph.partition("runtime-finite-source").add(finite);
  graph.partition("runtime-finite-sink").add(sink);
  graph.partition("runtime-unbounded-source").add(unbounded);
  graph.add_flow(finite->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock =
      graph.add_clock<RealtimeClock>("runtime-finite-and-unbounded-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// A cold allocation failure cannot retain session-local progress storage, so the shared startup
// terminal explicitly reports the empty sequence even when the selected plan has a completion rule.
// REQ: RUNTIME-050
TEST(RunSessionTest, CompletionPlanPreStorageAllocationFailurePublishesEmptyProgress) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(1U);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);

  const RunSession session = [&] {
    const test::ScopedAllocationFailure fail_next_allocation{0U};
    return run_async(plan);
  }();

  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_STARTUP_FAILED");
  EXPECT_TRUE(session.operator_completion_progress().empty());
  const auto published = diagnostic_sink->snapshot_operator_completion_progress();
  ASSERT_EQ(published.size(), 1U);
  EXPECT_TRUE(published.front().empty());
}

// A failure after progress materialization must retain the selected plan's terminal zero record.
// REQ: RUNTIME-050
TEST(RunSessionTest, CompletionPlanPostProgressStartupFailurePublishesZeroProgress) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(1U);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::
      fail_after_operator_completion_progress_materialization_for_test(with_diagnostics);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_STARTUP_FAILED");
  ASSERT_EQ(session.operator_completion_progress().size(), 1U);
  const auto& rule = plan.operator_completion_rules().front();
  const auto& progress = session.operator_completion_progress().front();
  EXPECT_EQ(progress.owner.id, rule.owner.id);
  EXPECT_EQ(progress.owner.plan_generation, rule.owner.plan_generation);
  EXPECT_TRUE(progress.owner.run_bound());
  EXPECT_EQ(progress.kind, OperatorCompletionRuleKind::kComputeCalls);
  EXPECT_EQ(progress.configured_limit, 1U);
  EXPECT_EQ(progress.successful_compute_calls, 0U);
  EXPECT_FALSE(progress.limit_reached);
  const auto published = diagnostic_sink->snapshot_operator_completion_progress();
  ASSERT_EQ(published.size(), 1U);
  const std::vector<OperatorCompletionProgressRecord> public_progress{
      session.operator_completion_progress().begin(), session.operator_completion_progress().end()};
  EXPECT_EQ(published.front(), public_progress);
}
#endif

// A process-group RunSession owns progress only for Operators reconstructed in that worker. A
// remote rule is absent, not a local zero, and the private terminal sink must publish the same
// canonical local subsequence as the public session accessor.
// REQ: RUNTIME-005
// REQ: RUNTIME-041
// REQ: RUNTIME-050
TEST(RunSessionTest, ProcessGroupWorkersPublishOnlyTheirLocalCompletionProgress) {
  const ExecutionPlan baseline = make_process_group_completion_progress_test_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  ASSERT_EQ(baseline.operator_completion_rules().size(), 2U);
  const auto accepted = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_NE(accepted, nullptr);

  std::array<ProcessGroupId, 2U> groups;
  for (std::size_t index = 0U; index < baseline.operator_completion_rules().size(); ++index) {
    const auto& rule = baseline.operator_completion_rules()[index];
    const auto factory =
        std::find_if(accepted->executable_factories.operators.begin(),
                     accepted->executable_factories.operators.end(),
                     [&](const auto& record) { return record.operator_id == rule.owner.id; });
    ASSERT_NE(factory, accepted->executable_factories.operators.end());
    groups[index] = detail::process_group_for(*accepted, factory->vertex);
    ASSERT_TRUE(groups[index]);
  }
  ASSERT_NE(groups[0], groups[1]);

  for (std::size_t selected_index = 0U; selected_index < groups.size(); ++selected_index) {
    SCOPED_TRACE(selected_index);
    const std::size_t remote_index = 1U - selected_index;
    const auto& local_rule = baseline.operator_completion_rules()[selected_index];
    const auto& remote_rule = baseline.operator_completion_rules()[remote_index];
    const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);

    test::RuntimeComputeCallChannelProbeState::reset(true);
    RunSession session = detail::run_async_worker(plan, groups[selected_index]);
    const bool compute_entered =
        test::RuntimeComputeCallChannelProbeState::wait_until_source_compute();
    EXPECT_TRUE(compute_entered);
    EXPECT_TRUE(session.operator_completion_progress().empty());
    const auto local_ref = session.operator_ref(local_rule.owner);
    EXPECT_TRUE(local_ref);
    const auto remote_ref = session.operator_ref(remote_rule.owner);
    EXPECT_FALSE(remote_ref);
    if (!remote_ref) {
      EXPECT_EQ(remote_ref.error().code, ErrorCode::kStaleOperatorRef);
    }
    test::RuntimeComputeCallChannelProbeState::release_source_compute();
    session.wait();

    ASSERT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty());
    ASSERT_EQ(session.operator_completion_progress().size(), 1U);
    const auto& progress = session.operator_completion_progress().front();
    EXPECT_EQ(progress.owner.id, local_rule.owner.id);
    EXPECT_EQ(progress.owner.plan_generation, local_rule.owner.plan_generation);
    if (local_ref) {
      EXPECT_EQ(progress.owner, *local_ref);
    }
    EXPECT_EQ(progress.kind, local_rule.kind);
    EXPECT_EQ(progress.configured_limit, local_rule.limit);
    EXPECT_EQ(progress.successful_compute_calls, local_rule.limit);
    EXPECT_TRUE(progress.limit_reached);
    EXPECT_EQ(
        test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
        local_rule.limit);
    EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
              local_rule.limit);

    const auto published = diagnostic_sink->snapshot_operator_completion_progress();
    ASSERT_EQ(published.size(), 1U);
    const std::vector<OperatorCompletionProgressRecord> public_progress{
        session.operator_completion_progress().begin(),
        session.operator_completion_progress().end()};
    EXPECT_EQ(published.front(), public_progress);
  }
}

// A failed multi-process launch without a selected group reconstructs no Operator and therefore
// owns no completion-progress cell, while still invoking the terminal sink exactly once.
// REQ: RUNTIME-036
// REQ: RUNTIME-050
TEST(RunSessionTest, UnselectedMultiProcessPlanPublishesEmptyCompletionProgress) {
  const ExecutionPlan baseline = make_process_group_completion_progress_test_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  ASSERT_EQ(baseline.operator_completion_rules().size(), 2U);
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();

  test::RuntimeComputeCallChannelProbeState::reset();
  RunSession session = run_async(plan);
  session.wait();

  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_PROCESS_GROUP_SELECTION_REQUIRED");
  EXPECT_TRUE(session.operator_completion_progress().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            0U);
  const auto published = diagnostic_sink->snapshot_operator_completion_progress();
  ASSERT_EQ(published.size(), 1U);
  EXPECT_TRUE(published.front().empty());
}

void expect_operator_completion_progress_sequences_equal(
    std::span<const OperatorCompletionProgressRecord> expected,
    std::span<const OperatorCompletionProgressRecord> actual) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(actual[index].owner, expected[index].owner);
    EXPECT_EQ(actual[index].kind, expected[index].kind);
    EXPECT_EQ(actual[index].configured_limit, expected[index].configured_limit);
    EXPECT_EQ(actual[index].successful_compute_calls, expected[index].successful_compute_calls);
    EXPECT_EQ(actual[index].limit_reached, expected[index].limit_reached);
  }
}

// Three physical Spin witnesses are available, but the source owns a one-call completion rule.
// The test control retains the finite receiver after the limit-reaching retirement so the next
// two stale selections reach the ordinary lane-side final-entry fence without entering compute.
// REQ: RUNTIME-050
TEST(RunSessionTest, OrdinaryDispatchRejectsQueuedWorkBeyondTheFinalComputeEntry) {
  reset_runtime_completion_fence_probe();
  const ExecutionPlan compiled = make_runtime_completion_fence_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto compiled_state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_NE(compiled_state, nullptr);
  ASSERT_EQ(compiled_state->closed_island_activations.size(), 1U);
  const ExecutionPlan finite = detail::RuntimeTestPlanAccess::enable_direct_count(compiled, 3U);
  const ExecutionPlan ordinary =
      detail::RuntimeTestPlanAccess::disable_direct_continuation_elision_for_test(finite);
  const auto final_entry_control =
      std::make_shared<detail::RuntimeOperatorCompletionFinalEntryTestControl>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_operator_completion_final_entry_test_control(
          ordinary, final_entry_control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(RuntimeCompletionFenceSource::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::observed_value.load(std::memory_order_relaxed), 41U);
  // Only the ruled source's three witnesses reach the final-entry control. Its unruled sibling
  // shares the partition wrapper but falls through after the owner lookup without being counted.
  EXPECT_EQ(final_entry_control->checked_count(), 3U);
  EXPECT_EQ(final_entry_control->rejected_count(), 2U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// A stale retained witness reaches the same lane entry as ordinary admitted work, but the
// Operator-local completion fence precedes deadline sampling. Only the limit-reaching call may
// start deadline evaluation; the two post-limit selections are retired without that side effect.
// REQ: RUNTIME-050
TEST(RunSessionTest, FinalEntryFenceRejectsStaleWorkBeforeMaxStartDelayEvaluation) {
  reset_runtime_completion_fence_probe();
  const ExecutionPlan compiled = make_runtime_completion_fence_plan(true);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.max_start_delays().size(), 1U);
  const auto compiled_state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_NE(compiled_state, nullptr);
  ASSERT_TRUE(compiled_state->closed_island_activations.empty());
  const ExecutionPlan finite = detail::RuntimeTestPlanAccess::enable_direct_count(compiled, 3U);
  const ExecutionPlan ordinary =
      detail::RuntimeTestPlanAccess::disable_direct_continuation_elision_for_test(finite);
  const auto final_entry_control =
      std::make_shared<detail::RuntimeOperatorCompletionFinalEntryTestControl>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_operator_completion_final_entry_test_control(
          ordinary, final_entry_control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(RuntimeCompletionFenceSource::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(final_entry_control->rejected_count(), 2U);
  EXPECT_EQ(final_entry_control->max_start_delay_evaluation_count(), 1U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// Disabling the two-callback composite retains the direct-continuation receipt path. Hold the
// receipt after the source gate has closed; releasing it may drain the final successor, but the
// final-entry fence must reject both retained source witnesses without entering compute.
// REQ: RUNTIME-050
TEST(RunSessionTest, DirectContinuationDrainsTheFinalSuccessorWithoutReenteringTheSource) {
  reset_runtime_completion_fence_probe();
  const ExecutionPlan compiled = make_runtime_completion_fence_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto compiled_state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_NE(compiled_state, nullptr);
  ASSERT_EQ(compiled_state->closed_island_activations.size(), 1U);
  const ExecutionPlan finite = detail::RuntimeTestPlanAccess::enable_direct_count(compiled, 3U);
  const ExecutionPlan ordinary =
      detail::RuntimeTestPlanAccess::disable_closed_island_composite_for_test(finite);
  const auto final_entry_control =
      std::make_shared<detail::RuntimeOperatorCompletionFinalEntryTestControl>();
  const ExecutionPlan with_final_entry =
      detail::RuntimeTestPlanAccess::with_operator_completion_final_entry_test_control(
          ordinary, final_entry_control);
  const auto control = std::make_shared<detail::RuntimeDirectContinuationTestControl>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_direct_continuation_test_control(
      with_final_entry, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool receipt_staged = control->wait_until_receipt_staged();
  if (!receipt_staged) {
    session.request_stop();
    control->release();
    session.wait();
  }
  ASSERT_TRUE(receipt_staged);
  EXPECT_TRUE(session.is_running());
  EXPECT_EQ(RuntimeCompletionFenceSource::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::compute_count.load(std::memory_order_acquire), 0U);
  control->release();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(RuntimeCompletionFenceSource::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::observed_value.load(std::memory_order_relaxed), 41U);
  EXPECT_EQ(final_entry_control->checked_count(), 3U);
  EXPECT_EQ(final_entry_control->rejected_count(), 2U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// The fused closed-island path uses the same final-entry fence. Pause between the source and
// successor transactions after the source's retirement closes its completion gate, then verify
// that the successor alone drains the accepted final publication while two stale source
// selections are rejected at final entry.
// REQ: RUNTIME-050
TEST(RunSessionTest, ClosedIslandDrainsTheFinalSuccessorWithoutReenteringTheSource) {
  reset_runtime_completion_fence_probe();
  const ExecutionPlan compiled = make_runtime_completion_fence_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto compiled_state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_NE(compiled_state, nullptr);
  ASSERT_EQ(compiled_state->closed_island_activations.size(), 1U);
  const ExecutionPlan finite = detail::RuntimeTestPlanAccess::enable_direct_count(compiled, 3U);
  const auto final_entry_control =
      std::make_shared<detail::RuntimeOperatorCompletionFinalEntryTestControl>();
  const ExecutionPlan with_final_entry =
      detail::RuntimeTestPlanAccess::with_operator_completion_final_entry_test_control(
          finite, final_entry_control);
  const auto control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
  control->set_pause_before_successor(true);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_closed_island_test_control(with_final_entry, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool source_retired = control->wait_until_before_successor();
  if (!source_retired) {
    session.request_stop();
    control->release();
    session.wait();
  }
  ASSERT_TRUE(source_retired);
  EXPECT_TRUE(session.is_running());
  EXPECT_EQ(RuntimeCompletionFenceSource::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::compute_count.load(std::memory_order_acquire), 0U);
  control->release();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(RuntimeCompletionFenceSource::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::observed_value.load(std::memory_order_relaxed), 41U);
  EXPECT_EQ(final_entry_control->checked_count(), 3U);
  EXPECT_EQ(final_entry_control->rejected_count(), 2U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// REQ: RUNTIME-050
TEST(RunSessionTest, ComputeCallCompletionStopsTheSourceAtItsLimitWithoutAnExtraCallback) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(kLimit);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            kLimit);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            2U * kLimit);
  expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
}

// REQ: RUNTIME-050
TEST(RunSessionTest, OperatorCompletionProgressRemainsEmptyUntilTerminalOutcomeFreeze) {
  test::RuntimeComputeCallChannelProbeState::reset(true);
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
  EXPECT_TRUE(session.operator_completion_progress().empty());
  test::RuntimeComputeCallChannelProbeState::release_source_compute();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// REQ: RUNTIME-050
TEST(RunSessionTest, FreshRunSessionStartsComputeCallProgressAtZero) {
  constexpr std::uint64_t kLimit = 2U;
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(kLimit);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunGenerationId first_run_generation{};

  for (std::size_t run = 0U; run < 2U; ++run) {
    test::RuntimeComputeCallChannelProbeState::reset();
    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_EQ(
        test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
        kLimit);
    expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
    ASSERT_EQ(session.operator_completion_progress().size(), 1U);
    const RunGenerationId run_generation =
        session.operator_completion_progress().front().owner.run_generation;
    if (run == 0U) {
      first_run_generation = run_generation;
    } else {
      EXPECT_NE(run_generation, first_run_generation);
    }
  }
}

// Completion records are immutable replay identity. Removing or changing the compiled rule fails
// before lifecycle callbacks or compute entry rather than silently changing the session budget.
// REQ: CONTRACT-020
// REQ: CONTRACT-024
// REQ: RUNTIME-050
TEST(RunSessionTest, CompletionRuleReplayRejectsMissingOrChangedRecordBeforeCompute) {
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(2U);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  ASSERT_EQ(baseline.operator_completion_rules().size(), 1U);
  const OperatorRef owner = baseline.operator_completion_rules().front().owner;
  const std::array cases{
      std::pair{"missing",
                detail::RuntimeTestPlanAccess::without_operator_completion_rules(baseline)},
      std::pair{"changed",
                detail::RuntimeTestPlanAccess::with_operator_completion_rule(baseline, owner, 3U)},
  };

  for (const auto& [name, plan] : cases) {
    SCOPED_TRACE(name);
    test::RuntimeComputeCallChannelProbeState::reset();
    ASSERT_TRUE(plan.ok()) << plan.json();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    ASSERT_FALSE(session.diagnostics().empty());
    EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
    EXPECT_EQ(
        test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
        0U);
  }
}

// REQ: RUNTIME-050
TEST(RunSessionTest, AcceptedRestartPreservesLogicalOperatorComputeCallProgress) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(
      kLimit, test::RuntimeComputeCallAction::kRestartAfterFirst);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_TRUE(
      test::RuntimeComputeCallChannelProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(
      test::RuntimeComputeCallChannelProbeState::restart_error.load(std::memory_order_relaxed),
      ErrorCode::kSuccess);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            kLimit);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            2U);
  EXPECT_EQ(
      test::RuntimeComputeCallChannelProbeState::sink_compute_count.load(std::memory_order_acquire),
      2U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::first_value.load(std::memory_order_relaxed),
            78U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::second_value.load(std::memory_order_relaxed),
            78U);
  // Task 036 discards queued old-generation ingress when a restart publishes. Task 039 preserves
  // the logical compute-call count and protects the final limit-reaching call's publications; it
  // does not carry the first call's queued value across the generation boundary.
  const auto lifecycle_receipts = diagnostic_sink->snapshot_lifecycle_receipts();
  ASSERT_EQ(lifecycle_receipts.size(), 1U);
  ASSERT_EQ(lifecycle_receipts.front().size(), 1U);
  const auto& receipt = lifecycle_receipts.front().front();
  EXPECT_EQ(receipt.origin, detail::RuntimeLifecycleRequestOrigin::kComputeSelf);
  EXPECT_EQ(receipt.state, LifecycleTransactionState::kPublished);
  EXPECT_EQ(receipt.recovery, detail::RuntimeLifecycleRecoveryDisposition::kPublished);
  EXPECT_EQ(receipt.previous_epoch, 1U);
  EXPECT_EQ(receipt.candidate_epoch, 2U);
  EXPECT_EQ(receipt.effective_epoch, 2U);
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);
  EXPECT_EQ(receipt.receipt.target.id, plan.operator_completion_rules().front().owner.id);
  EXPECT_EQ(receipt.receipt.target.plan_generation,
            plan.operator_completion_rules().front().owner.plan_generation);
  expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
}

// An accepted self-restart preserves the logical Operator's cumulative progress while Task 036
// retires only the restart-boundary call's queued old-generation publications.
// REQ: RUNTIME-050
TEST(RunSessionTest, AcceptedRestartPreservesThreeHundredComputeCallsAcrossTwoGenerations) {
  constexpr std::uint64_t kSuccessfulComputeCallsBeforeRestart = 120U;
  constexpr std::uint64_t kSuccessfulComputeCallsAfterRestart = 180U;
  constexpr std::uint64_t kLimit =
      kSuccessfulComputeCallsBeforeRestart + kSuccessfulComputeCallsAfterRestart;
  test::RuntimeComputeCallChannelProbeState::reset();
  // The 1 ms clock source cannot wait for a delayed sink, so give each edge room for the whole
  // burst; a depth-one edge fails the run once a sink lags past the graph blocking budget. The
  // source drains both sinks before the restart-boundary call, which keeps the bound below exact.
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(
      kLimit,
      test::RuntimeComputeCallAction::kRestartAfterComputeCalls,
      kSuccessfulComputeCallsBeforeRestart,
      true,
      false,
      detail::ProcessLocalFanoutExperiment::kDisabled,
      false,
      QueuePolicy::kBlocking,
      false,
      false,
      static_cast<std::uint32_t>(kLimit));
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_TRUE(
      test::RuntimeComputeCallChannelProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(
      test::RuntimeComputeCallChannelProbeState::restart_error.load(std::memory_order_relaxed),
      ErrorCode::kSuccess);
  const std::uint64_t successful_compute_calls =
      test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire);
  const std::uint64_t successful_compute_calls_before_restart =
      test::RuntimeComputeCallChannelProbeState::successful_compute_calls_before_restart.load(
          std::memory_order_acquire);
  EXPECT_EQ(successful_compute_calls_before_restart, kSuccessfulComputeCallsBeforeRestart);
  EXPECT_EQ(successful_compute_calls - successful_compute_calls_before_restart,
            kSuccessfulComputeCallsAfterRestart);
  EXPECT_EQ(successful_compute_calls, kLimit);
  // Task 036 discards only restart-boundary publications that are still queued when the restart
  // publishes. Either fanout arm may already have been consumed, so zero, one, or both boundary
  // samples may survive; every other sample, including both final-call samples, must drain.
  const std::size_t sink_count =
      test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire);
  EXPECT_GE(sink_count, 2U * (kLimit - 1U));
  EXPECT_LE(sink_count, 2U * kLimit);
  const auto lifecycle_receipts = diagnostic_sink->snapshot_lifecycle_receipts();
  ASSERT_EQ(lifecycle_receipts.size(), 1U);
  ASSERT_EQ(lifecycle_receipts.front().size(), 1U);
  const auto& receipt = lifecycle_receipts.front().front();
  EXPECT_EQ(receipt.origin, detail::RuntimeLifecycleRequestOrigin::kComputeSelf);
  EXPECT_EQ(receipt.state, LifecycleTransactionState::kPublished);
  EXPECT_EQ(receipt.recovery, detail::RuntimeLifecycleRecoveryDisposition::kPublished);
  EXPECT_EQ(receipt.previous_epoch, 1U);
  EXPECT_EQ(receipt.candidate_epoch, 2U);
  EXPECT_EQ(receipt.effective_epoch, 2U);
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);
  EXPECT_EQ(receipt.receipt.target.id, plan.operator_completion_rules().front().owner.id);
  EXPECT_EQ(receipt.receipt.target.plan_generation,
            plan.operator_completion_rules().front().owner.plan_generation);
  expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
}

// A failed forward restart that restores the old generation does not create a fresh completion
// budget. The first accepted publication survives Task 036 rollback, then the second successful
// call reaches the same logical Operator's original limit.
// REQ: RUNTIME-050
TEST(RunSessionTest, RolledBackRestartPreservesLogicalOperatorComputeCallProgress) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(
      kLimit, test::RuntimeComputeCallAction::kRestartRollbackAfterFirst);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().severity, DiagnosticSeverity::kWarning);
  const auto& failure_detail = session.diagnostics().front().detail;
  if (!failure_detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  EXPECT_EQ(failure_detail->stage, LifecycleStage::kStart);
  EXPECT_NE(failure_detail->transaction_id, 0U);
  EXPECT_TRUE(
      test::RuntimeComputeCallChannelProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::restart_forward_start_count.load(
                std::memory_order_acquire),
            1U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::restore_old_start_count.load(
                std::memory_order_acquire),
            1U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            kLimit);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            2U * kLimit);
  const auto lifecycle_receipts = diagnostic_sink->snapshot_lifecycle_receipts();
  ASSERT_EQ(lifecycle_receipts.size(), 1U);
  ASSERT_EQ(lifecycle_receipts.front().size(), 1U);
  const auto& receipt = lifecycle_receipts.front().front();
  EXPECT_EQ(receipt.state, LifecycleTransactionState::kRolledBack);
  EXPECT_EQ(receipt.recovery, detail::RuntimeLifecycleRecoveryDisposition::kRestoredOldEpoch);
  EXPECT_EQ(receipt.previous_epoch, 1U);
  EXPECT_EQ(receipt.candidate_epoch, 2U);
  EXPECT_EQ(receipt.effective_epoch, 1U);
  expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
}

// REQ: RUNTIME-050
TEST(RunSessionTest, ComputeCallCompletionDrainsFinalOutputAndEosBeforeNaturalCompletion) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            2U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::first_value.load(std::memory_order_relaxed),
            77U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::second_value.load(std::memory_order_relaxed),
            77U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// An ordinary inputful transform must propagate natural closure only after forwarding the source's
// final accepted data. EOS is control, not an extra compute-call witness at either downstream hop.
// REQ: RUNTIME-050
// REQ: CHAN-136
TEST(RunSessionTest, ComputeCallCompletionPropagatesEosThroughOrdinaryThreeHopChain) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_chain_plan(kLimit);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            kLimit);
  EXPECT_EQ(
      test::RuntimeComputeCallChannelProbeState::transform_count.load(std::memory_order_acquire),
      kLimit);
  EXPECT_EQ(
      test::RuntimeComputeCallChannelProbeState::sink_compute_count.load(std::memory_order_acquire),
      kLimit);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            kLimit);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::first_value.load(std::memory_order_relaxed),
            77U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::second_value.load(std::memory_order_relaxed),
            78U);
  expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
}

// Each finite producer closes only its own gate. Natural completion waits for both unequal limits
// and their independent downstream EOS paths to drain.
// REQ: RUNTIME-050
// REQ: CHAN-136
TEST(RunSessionTest, UnequalFiniteSourcesCompleteOnlyAfterEveryOperatorLocalLimitDrains) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = make_unequal_compute_call_sources_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.operator_completion_rules().size(), 2U);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            4U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            4U);
  ASSERT_EQ(session.operator_completion_progress().size(), 2U);
  EXPECT_EQ(session.operator_completion_progress()[0].configured_limit, 1U);
  EXPECT_EQ(session.operator_completion_progress()[0].successful_compute_calls, 1U);
  EXPECT_TRUE(session.operator_completion_progress()[0].limit_reached);
  EXPECT_EQ(session.operator_completion_progress()[1].configured_limit, 3U);
  EXPECT_EQ(session.operator_completion_progress()[1].successful_compute_calls, 3U);
  EXPECT_TRUE(session.operator_completion_progress()[1].limit_reached);
}

// A finite producer reaching its limit does not make an unrelated ordinary source finite. The
// session remains live until an explicit Task 036 stop wins terminal arbitration.
// REQ: RUNTIME-050
// REQ: CHAN-136
TEST(RunSessionTest, FiniteProducerDrainDoesNotCompleteAnUnboundedSource) {
  test::RuntimeComputeCallChannelProbeState::reset();
  test::RuntimeClockProbeState::reset();
  const ExecutionPlan plan = make_finite_and_unbounded_source_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);

  RunSession session = run_async(plan);
  const auto finite_deadline = std::chrono::steady_clock::now() + 5s;
  while (test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire) ==
             0U &&
         std::chrono::steady_clock::now() < finite_deadline) {
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            1U);
  EXPECT_TRUE(test::RuntimeClockProbeState::wait_until_compute_count(1U));
  EXPECT_TRUE(session.is_running());
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// REQ: RUNTIME-050
TEST(RunSessionTest, ZeroComputeCallLimitClosesOutputsWithoutCallingTheSource) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(0U);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            0U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            0U);
  expect_terminal_compute_call_progress(session, plan, 0U, 0U, true);
}

// A terminal close that throws at the backend boundary must fail the RunSession without escaping
// the noexcept close path. The successful compute-call budget stays recorded, while terminal
// closure stops at the throwing arm. The failed closure wakes every other owner, so the run ends
// promptly even when those owners park until their next deadline.
// REQ: RUNTIME-050
// REQ: CHAN-136
// REQ: RUNTIME-087
TEST(RunSessionTest, ThrowingTerminalCloseFailsTheRunSessionAfterCompletionProgressCommits) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  test::RuntimeComputeCallChannelProbeState::reset();
  // The scripted publishers below cannot prove the recipients that delivery inside emit() needs,
  // so the pump delivers.
  const ExecutionPlan baseline =
      test::make_runtime_compute_call_channel_plan(1U,
                                                   test::RuntimeComputeCallAction::kNone,
                                                   1U,
                                                   true,
                                                   false,
                                                   detail::ProcessLocalFanoutExperiment::kDisabled,
                                                   false,
                                                   QueuePolicy::kBlocking,
                                                   true);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto accepted = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_TRUE(accepted);
  ASSERT_FALSE(accepted->process_local_channel_publishers.empty());
  const PortId output = accepted->process_local_channel_publishers.front().source;
  EXPECT_TRUE(
      std::ranges::all_of(accepted->process_local_channel_publishers,
                          [output](const auto& publisher) { return publisher.source == output; }));
  const auto control = std::make_shared<ThrowingTerminalCloseChannelBindControl>(output);
  // A 10 s park cap makes a failure that does not wake the other owners visible as a slow run.
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_owner_park_cap(
      detail::RuntimeTestPlanAccess::with_channel_session_bind_test_control(baseline, control),
      10s);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto started = std::chrono::steady_clock::now();
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_LT(std::chrono::steady_clock::now() - started, 5s);
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_READINESS_BINDING_UNAVAILABLE"));
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  // The test hook intentionally replaces the real transport publishers with scripted endpoints,
  // so this seam verifies completion and failure containment rather than downstream delivery.
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            0U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
  EXPECT_EQ(control->first_close_count(), 1U);
#else
  GTEST_SKIP() << "exception containment requires language exception support";
#endif
}

// REQ: RUNTIME-050
TEST(RunSessionTest, FailedComputeCallRetainsOnlyPreviouslySuccessfulProgressAndPublication) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      3U, test::RuntimeComputeCallAction::kFailOnSecond);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            2U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            2U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::first_value.load(std::memory_order_relaxed),
            77U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::second_value.load(std::memory_order_relaxed),
            77U);
  expect_terminal_compute_call_progress(session, plan, 3U, 1U, false);
}

// An accepted first fanout does not consume completion authority when a later publication in the
// same callback fails and makes the callback return an Error.
// REQ: RUNTIME-050
TEST(RunSessionTest, FailedPublicationConsumesNoComputeCallCompletionUnit) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      kLimit, test::RuntimeComputeCallAction::kFailPublicationOnFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  expect_terminal_compute_call_progress(session, plan, kLimit, 0U, false);
}

// The callback returns success and its first publication is accepted, but Channel activation
// retirement fails before the activation cut and completion-cell commit. This is neither a user
// callback failure nor a publication failure, and therefore consumes no completion unit.
// REQ: RUNTIME-050
// REQ: CHAN-136
TEST(RunSessionTest, FailedPostCallbackChannelRetirementConsumesNoCompletionUnit) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  // Keep the sinks on the source owner so neither can consume the one-shot retirement fault
  // between successful source publication and source retirement.
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(
      kLimit, test::RuntimeComputeCallAction::kNone, 1U, false);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::fail_next_channel_activation_retirement_for_test(baseline);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_LIFECYCLE_CALLBACK_FAILED");
  EXPECT_EQ(
      test::RuntimeComputeCallChannelProbeState::sink_compute_count.load(std::memory_order_acquire),
      0U);
  EXPECT_TRUE(std::ranges::none_of(session.diagnostics(), [](const Diagnostic& diagnostic) {
    return diagnostic.token == "RUNTIME_OPERATOR_CALLBACK_FAILED";
  }));
  expect_terminal_compute_call_progress(session, plan, kLimit, 0U, false);
}

// The structural sink receives one terminal completion-progress sequence for every terminal
// outcome, including successful outcomes whose ordinary Diagnostic sequence is empty. Each
// sequence must be nonempty and byte-for-field identical to the public frozen accessor.
// REQ: RUNTIME-050
TEST(RunSessionTest, TerminalSinkPublishesOneNonemptyCompletionProgressSequenceForEveryOutcome) {
  struct TestCase {
    std::string_view name;
    std::uint64_t limit{};
    test::RuntimeComputeCallAction action{test::RuntimeComputeCallAction::kNone};
    RunTermination expected_termination{RunTermination::kCompleted};
    std::uint64_t expected_progress{};
    bool expected_limit_reached{};
  };
  constexpr std::array test_cases{
      TestCase{
          .name = "natural",
          .limit = 1U,
          .action = test::RuntimeComputeCallAction::kNone,
          .expected_termination = RunTermination::kCompleted,
          .expected_progress = 1U,
          .expected_limit_reached = true,
      },
      TestCase{
          .name = "stop",
          .limit = 3U,
          .action = test::RuntimeComputeCallAction::kRequestStopAfterFirst,
          .expected_termination = RunTermination::kStopped,
          .expected_progress = 1U,
          .expected_limit_reached = false,
      },
      TestCase{
          .name = "requested-completion",
          .limit = 3U,
          .action = test::RuntimeComputeCallAction::kRequestCompletionAfterFirst,
          .expected_termination = RunTermination::kRequestedCompleted,
          .expected_progress = 1U,
          .expected_limit_reached = false,
      },
      TestCase{
          .name = "failure",
          .limit = 3U,
          .action = test::RuntimeComputeCallAction::kFailOnSecond,
          .expected_termination = RunTermination::kFailed,
          .expected_progress = 1U,
          .expected_limit_reached = false,
      },
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    test::RuntimeComputeCallChannelProbeState::reset();
    const ExecutionPlan baseline =
        test::make_runtime_compute_call_channel_plan(test_case.limit, test_case.action);
    ASSERT_TRUE(baseline.ok()) << baseline.json();
    const auto sink = std::make_shared<RecordingDiagnosticSink>();
    const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, sink);
    ASSERT_TRUE(plan.ok()) << plan.json();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), test_case.expected_termination);
    expect_terminal_compute_call_progress(session,
                                          plan,
                                          test_case.limit,
                                          test_case.expected_progress,
                                          test_case.expected_limit_reached);
    const auto published = sink->snapshot_operator_completion_progress();
    ASSERT_EQ(published.size(), 1U);
    ASSERT_FALSE(published.front().empty());
    expect_operator_completion_progress_sequences_equal(session.operator_completion_progress(),
                                                        published.front());
  }
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// A thrown callback after an accepted publication is still a failed compute call and consumes no
// completion authority.
// REQ: RUNTIME-050
TEST(RunSessionTest, ThrownComputeCallConsumesNoCompletionUnit) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      kLimit, test::RuntimeComputeCallAction::kThrowOnFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  expect_terminal_compute_call_progress(session, plan, kLimit, 0U, false);
}
#endif

// REQ: RUNTIME-050
TEST(RunSessionTest, StopPublishesPartialComputeCallProgressWithoutClaimingTheLimit) {
  test::RuntimeComputeCallChannelProbeState::reset(true);
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      3U, test::RuntimeComputeCallAction::kRequestStopAfterFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
  EXPECT_TRUE(session.operator_completion_progress().empty());
  test::RuntimeComputeCallChannelProbeState::release_source_compute();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, 3U, 1U, false);
}

// UINT64_MAX is an exact limit rather than an absent/unbounded sentinel. A stopped session reports
// that unchanged configured value and its bounded partial progress.
// REQ: RUNTIME-050
TEST(RunSessionTest, MaximumComputeCallLimitIsAnOrdinaryExactValue) {
  constexpr std::uint64_t kLimit = std::numeric_limits<std::uint64_t>::max();
  test::RuntimeComputeCallChannelProbeState::reset(true);
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      kLimit, test::RuntimeComputeCallAction::kRequestStopAfterFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
  test::RuntimeComputeCallChannelProbeState::release_source_compute();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, kLimit, 1U, false);
}

// Seed the executable progress immediately below its exact maximum, then require one and only one
// successful compute call to reach natural completion without increment wrap.
// REQ: RUNTIME-050
TEST(RunSessionTest, MaximumComputeCallBoundaryClosesWithoutWrapOrAnExtraCallback) {
  constexpr std::uint64_t kLimit = std::numeric_limits<std::uint64_t>::max();
  constexpr std::uint64_t kSeed = kLimit - 1U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan source = test::make_runtime_compute_call_channel_plan(kLimit);
  ASSERT_TRUE(source.ok()) << source.json();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_operator_compute_call_progress_seed_for_test(source,
                                                                                       kSeed);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            2U);
  expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
}

// A valid readiness source that publishes no event performs no callback entry and consumes no
// completion unit. Explicit stop remains distinct from natural Operator completion.
// REQ: RUNTIME-050
TEST(RunSessionTest, UnfiredNotificationConsumesNoComputeCallCompletionUnit) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_dormant_compute_call_plan(kLimit);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);

  RunSession session = run_async(plan);
  std::this_thread::sleep_for(25ms);
  EXPECT_TRUE(session.is_running());
  EXPECT_TRUE(session.operator_completion_progress().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            0U);
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, kLimit, 0U, false);
}

// Task 036 stop precedence wins even when the same successful return reaches the natural limit.
// REQ: RUNTIME-046
// REQ: RUNTIME-050
TEST(RunSessionTest, StopWinsTheFinalSuccessfulComputeCallRace) {
  test::RuntimeComputeCallChannelProbeState::reset(true);
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      1U, test::RuntimeComputeCallAction::kRequestStopAfterFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
  test::RuntimeComputeCallChannelProbeState::release_source_compute();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// Requested completion similarly outranks natural completion without weakening the final
// successful-call accounting boundary.
// REQ: RUNTIME-046
// REQ: RUNTIME-050
TEST(RunSessionTest, RequestedCompletionWinsTheFinalSuccessfulComputeCallRace) {
  test::RuntimeComputeCallChannelProbeState::reset(true);
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      1U, test::RuntimeComputeCallAction::kRequestCompletionAfterFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
  test::RuntimeComputeCallChannelProbeState::release_source_compute();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// A self-restart queued by the limit-reaching call cannot publish a new producer generation after
// the Operator-local completion gate closes. The lifecycle fence fails closed and Runtime failure
// outranks the otherwise-natural completion.
// REQ: RUNTIME-046
// REQ: RUNTIME-050
TEST(RunSessionTest, RestartQueuedAtTheLimitCannotReopenTerminalCompletion) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      1U, test::RuntimeComputeCallAction::kRestartAfterFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(
      test::RuntimeComputeCallChannelProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// REQ: RUNTIME-050
TEST(RunSessionTest, RequestedCompletionPublishesPartialProgressWithoutClaimingNaturalDrain) {
  test::RuntimeComputeCallChannelProbeState::reset(true);
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      3U, test::RuntimeComputeCallAction::kRequestCompletionAfterFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
  EXPECT_TRUE(session.operator_completion_progress().empty());
  test::RuntimeComputeCallChannelProbeState::release_source_compute();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, 3U, 1U, false);
}

}  // namespace
}  // namespace holoscan
