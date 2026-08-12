/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_placement_internal.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/parameter_update_coordinator.hpp"
#include "runtime/runtime_callback_context.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_worker.hpp"
#include "runtime/source_wait.hpp"
#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"

namespace holoscan {

namespace {

using namespace std::chrono_literals;

class OptionalLifecycleReplayProbe final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    if (!omit_callbacks.load(std::memory_order_relaxed)) {
      spec.lifecycle()
          .optional(LifecycleStage::kStart, &OptionalLifecycleReplayProbe::start_stage)
          .optional(LifecycleStage::kStop, &OptionalLifecycleReplayProbe::stop_stage);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }
  LifecycleStatus start_stage(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }
  LifecycleStatus stop_stage(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }

  static inline std::atomic_bool omit_callbacks{};
};

// Captures one round-trip observation: the data value and SampleMetadata seen
// at either the source (expected) or the sink (observed) side.
// REQ: RUNTIME-012
TEST(RunSessionTime, BoundsFullRangeSourceWaitWithoutSignedOverflow) {
  constexpr std::chrono::nanoseconds kMaximum{1ms};
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();

  EXPECT_EQ(detail::bounded_source_wait_duration(kMin, kMax, kMaximum), kMaximum);
  EXPECT_EQ(detail::bounded_source_wait_duration(kMin, kMin + 999, kMaximum), 999ns);
  EXPECT_EQ(detail::bounded_source_wait_duration(kMax - 999, kMax, kMaximum), 999ns);
  EXPECT_EQ(detail::bounded_source_wait_duration(kMax - kMaximum.count(), kMax, kMaximum),
            kMaximum);
}

// The producer can signal after the controller's last drain but before it begins waiting. The
// durable token must make the subsequent timed wait return without relying on a fallback poll.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, RetainsSignalAcrossTheReadinessScanToWaitBoundary) {
  detail::RuntimeWakeState wake;

  // Key setup: signal after the controller-equivalent drain and before its nonblocking wait.
  EXPECT_EQ(wake.drain(), 0U);
  wake.signal();
  // Key assertion: exactly one durable token crosses the scan-to-wait boundary.
  EXPECT_TRUE(wake.wait_for(0ns));
  EXPECT_FALSE(wake.wait_for(0ns));
}

// A controller discards old wake hints before scanning because exact witness cardinality lives in
// the bounded physical source, not in this coalescing mechanism.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, DrainsEveryCoalescedSignalBeforeTheNextReadinessScan) {
  detail::RuntimeWakeState wake;

  // Key setup: coalesce three producer hints before the next controller scan.
  wake.signal();
  wake.signal();
  wake.signal();
  // Key assertion: drain accounts for every hint and leaves no stale wake token.
  EXPECT_EQ(wake.drain(), 3U);
  EXPECT_FALSE(wake.wait_for(0ns));
}

// Direct-retention progress is a rescan hint rather than a counted witness. One parked partition
// therefore retains at most one durable token while producers continue to advance the bounded
// typed store.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, BoundsCoalescedRescanHintsToOneDurableToken) {
  detail::RuntimeWakeState wake;

  EXPECT_TRUE(wake.signal_coalesced());
  for (std::size_t duplicate = 0U; duplicate < 1024U; ++duplicate) {
    EXPECT_FALSE(wake.signal_coalesced());
  }
  EXPECT_EQ(wake.drain(), 1U);
  EXPECT_FALSE(wake.wait_for(0ns));

  // A new hint after consumption must arm the next scan-to-wait boundary.
  EXPECT_TRUE(wake.signal_coalesced());
  EXPECT_TRUE(wake.wait_for(0ns));
  EXPECT_FALSE(wake.wait_for(0ns));
}

// A producer can be preempted while publishing the semaphore permit as the controller drains its
// previous scan. The durable pending count must keep that race bounded without losing a
// successfully accepted rescan hint.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, ConcurrentCoalescedSignalsAndDrainsRemainBoundedWithoutLosingWakeTokens) {
  detail::RuntimeWakeState wake;
  constexpr std::size_t kProducerCount = 4U;
  constexpr std::size_t kSignalsPerProducer = 4096U;
  std::atomic_bool start{};
  std::atomic_size_t active_producers{kProducerCount};
  std::atomic_size_t accepted{};
  std::array<std::thread, kProducerCount> producers;

  for (std::thread& producer : producers) {
    producer = std::thread([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (std::size_t signal = 0U; signal < kSignalsPerProducer; ++signal) {
        if (wake.signal_coalesced()) {
          accepted.fetch_add(1U, std::memory_order_relaxed);
        }
        std::this_thread::yield();
      }
      active_producers.fetch_sub(1U, std::memory_order_release);
    });
  }

  start.store(true, std::memory_order_release);
  std::size_t consumed = 0U;
  while (active_producers.load(std::memory_order_acquire) != 0U) {
    const std::size_t drained = wake.drain();
    EXPECT_LE(drained, 1U);
    consumed += drained;
    if (wake.wait_for(0ns)) {
      ++consumed;
    }
    std::this_thread::yield();
  }
  for (std::thread& producer : producers) {
    producer.join();
  }

  while (true) {
    const std::size_t drained = wake.drain();
    EXPECT_LE(drained, 1U);
    consumed += drained;
    if (drained == 0U) {
      break;
    }
  }
  EXPECT_FALSE(wake.wait_for(0ns));
  EXPECT_EQ(consumed, accepted.load(std::memory_order_relaxed));
}

// Counted readiness evidence and a coalesced direct-retention rescan can arrive concurrently.
// Every accepted counted token remains distinct while the rescan hint stays bounded to one.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, MixedCountedAndCoalescedSignalsRemainDurableAcrossConcurrentConsumers) {
  detail::RuntimeWakeState wake;
  constexpr std::size_t kCountedSignals = 4096U;
  std::atomic_bool start{};
  std::atomic_size_t accepted_coalesced{};
  std::atomic_size_t consumed{};

  std::thread counted_producer([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (std::size_t signal = 0U; signal < kCountedSignals; ++signal) {
      wake.signal();
      std::this_thread::yield();
    }
  });
  std::thread coalesced_producer([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (std::size_t signal = 0U; signal < kCountedSignals; ++signal) {
      if (wake.signal_coalesced()) {
        accepted_coalesced.fetch_add(1U, std::memory_order_relaxed);
      }
      std::this_thread::yield();
    }
  });
  std::thread waiting_consumer([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    while (consumed.load(std::memory_order_relaxed) <
           kCountedSignals + accepted_coalesced.load(std::memory_order_relaxed)) {
      if (wake.wait_for(0ns)) {
        consumed.fetch_add(1U, std::memory_order_relaxed);
      } else {
        std::this_thread::yield();
      }
    }
  });

  start.store(true, std::memory_order_release);
  counted_producer.join();
  coalesced_producer.join();
  const std::size_t expected = kCountedSignals + accepted_coalesced.load(std::memory_order_relaxed);
  while (consumed.load(std::memory_order_relaxed) < expected) {
    consumed.fetch_add(wake.drain(), std::memory_order_relaxed);
    std::this_thread::yield();
  }
  waiting_consumer.join();

  EXPECT_EQ(consumed.load(std::memory_order_relaxed), expected);
  EXPECT_FALSE(wake.wait_for(0ns));
}

// Cold partition routing maps one compiler-frozen owner to exactly one wake state. Together with
// the Channel callback test that verifies destination/source owner identities, this proves direct
// progress cannot broadcast to an unrelated partition.
// REQ: RUNTIME-041
TEST(RuntimePartitionWakeRouter, TargetsExactlyOneCompilerFrozenOwnerWithoutBroadcast) {
  detail::RuntimeWakeState source;
  detail::RuntimeWakeState sink;
  detail::RuntimeWakeState parked;
  detail::RuntimePartitionWakeRouter router;
  router.reset(4U);

  ASSERT_TRUE(router.bind(VertexId{1U}, &source));
  ASSERT_TRUE(router.bind(VertexId{2U}, &sink));
  ASSERT_TRUE(router.bind(VertexId{3U}, &parked));
  EXPECT_FALSE(router.bind(VertexId{2U}, &parked));
  EXPECT_EQ(router.find(VertexId{}), nullptr);
  EXPECT_EQ(router.find(VertexId{4U}), nullptr);

  ASSERT_NE(router.find(VertexId{2U}), nullptr);
  EXPECT_TRUE(router.find(VertexId{2U})->signal_coalesced());
  EXPECT_FALSE(source.wait_for(0ns));
  EXPECT_TRUE(sink.wait_for(0ns));
  EXPECT_FALSE(parked.wait_for(0ns));

  ASSERT_NE(router.find(VertexId{1U}), nullptr);
  EXPECT_TRUE(router.find(VertexId{1U})->signal_coalesced());
  EXPECT_TRUE(source.wait_for(0ns));
  EXPECT_FALSE(sink.wait_for(0ns));
  EXPECT_FALSE(parked.wait_for(0ns));
}

// The controller drains before every readiness scan, including the common case where no producer
// posted a wake. Empty drains must stay on the atomic fast path rather than probing the semaphore.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, RepeatedEmptyDrainsRemainEmptyWithoutInventingWakeTokens) {
  detail::RuntimeWakeState wake;

  // Key setup: model consecutive controller scans with no intervening physical-source signal.
  for (std::size_t scan = 0U; scan < 1024U; ++scan) {
    // Key assertion: every drain observes the empty pending count and leaves the wait unarmed.
    EXPECT_EQ(wake.drain(), 0U);
  }
  EXPECT_FALSE(wake.wait_for(0ns));
}

// Signaling is valid whether the controller has already entered the kernel wait or is still
// between announcing that wait and blocking; both interleavings consume the same durable token.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, ReleasesAWaitingController) {
  detail::RuntimeWakeState wake;
  std::atomic_bool waiter_started{};
  std::atomic_bool signaled{};
  // Key setup: place one controller-equivalent thread in a bounded kernel wait.
  std::thread waiter([&] {
    waiter_started.store(true, std::memory_order_release);
    signaled.store(wake.wait_for(1s), std::memory_order_release);
  });
  while (!waiter_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  wake.signal();
  waiter.join();

  // Key assertion: a foreign-thread signal releases the waiting controller.
  EXPECT_TRUE(signaled.load(std::memory_order_acquire));
}

// Clock-driven sources still require bounded timed waits when no physical source signals.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, TimesOutWithoutAReadinessSignal) {
  detail::RuntimeWakeState wake;

  // Key assertion: no physical signal produces no synthetic readiness wake.
  EXPECT_FALSE(wake.wait_for(0ns));
}

// REQ: RUNTIME-003
static_assert(!std::is_default_constructible_v<RunSession>);
static_assert(std::is_nothrow_move_constructible_v<RunSession>);
static_assert(!std::is_copy_constructible_v<RunSession>);
static_assert(!std::is_copy_assignable_v<RunSession>);
static_assert(!std::is_move_assignable_v<RunSession>);
static_assert(std::is_nothrow_destructible_v<RunSession>);

// The analyzer intentionally follows the moved-from source into this helper;
// the public contract guarantees that source is inert and remains callable.
// NOLINTBEGIN(clang-analyzer-cplusplus.Move)
void expect_inert_moved_from_session(RunSession& session, OperatorRef plan_ref) {
  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  session.request_stop();
  session.wait();
  EXPECT_EQ(session.operator_ref(plan_ref).error().code, ErrorCode::kStaleOperatorRef);
}
// NOLINTEND(clang-analyzer-cplusplus.Move)

void expect_diagnostic_sequences_equal(std::span<const Diagnostic> expected,
                                       std::span<const Diagnostic> actual) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(actual[index].id, expected[index].id);
    EXPECT_EQ(actual[index].severity, expected[index].severity);
    EXPECT_EQ(actual[index].token, expected[index].token);
    EXPECT_EQ(actual[index].source.graph_generation, expected[index].source.graph_generation);
    EXPECT_EQ(actual[index].source.graph, expected[index].source.graph);
    EXPECT_EQ(actual[index].source.vertex, expected[index].source.vertex);
    EXPECT_EQ(actual[index].source.port, expected[index].source.port);
    EXPECT_EQ(actual[index].service, expected[index].service);
    EXPECT_EQ(actual[index].endpoint, expected[index].endpoint);
    EXPECT_EQ(actual[index].json_pointer, expected[index].json_pointer);
    EXPECT_EQ(actual[index].required, expected[index].required);
    EXPECT_EQ(actual[index].provided, expected[index].provided);
    EXPECT_EQ(actual[index].cause_id, expected[index].cause_id);
  }
}

// Graph topology:
//   a-first --[lifecycle: before(kStart)]--> z-second
// Independent operators with an authored start-dependency used to probe
// parameter-update startup failure with selective stop rollback.
[[nodiscard]] ExecutionPlan make_two_legacy_parameter_startup_plan() {
  Graph graph{"runtime-startup-pre-callback-failure"};
  const auto first = graph.op<test::RuntimeParamProbeOperator>("a-first");
  const auto second = graph.op<test::RuntimeParamProbeOperator>("z-second");
  graph.partition("runtime-first-spin").add(first);
  graph.partition("runtime-second-spin").add(second);
  graph.lifecycle().before(first, second, LifecycleStage::kStart);
  const ClockKey clock = graph.add_clock<test::RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment
      .bind_thread_placement("runtime-first-spin",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("runtime-second-spin",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

// Graph topology:
//   z-second --[lifecycle: before(kConfigure)]--> a-first
// Reversed configure dependency verifies the compiler produces canonical topological
// checkpoint order regardless of authored sequence.
[[nodiscard]] ExecutionPlan make_reverse_topological_lifecycle_plan() {
  Graph graph{"runtime-reverse-topological-lifecycle"};
  const auto first = graph.op<test::RuntimeParamProbeOperator>("a-first");
  const auto second = graph.op<test::RuntimeParamProbeOperator>("z-second");
  graph.partition("runtime-first-spin").add(first);
  graph.partition("runtime-second-spin").add(second);
  graph.lifecycle().before(second, first, LifecycleStage::kConfigure);
  const ClockKey clock = graph.add_clock<test::RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment
      .bind_thread_placement("runtime-first-spin",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("runtime-second-spin",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

// Captures every published Diagnostic sequence, runtime detail sequence, and lane
// startup evidence in memory for post-run assertion. When fail_=true it throws on
// publish to verify that sink exceptions cannot alter run outcome.
// Used by: BlockingAndAsyncRunsPublishTheExactSameTerminalDiagnostics, and others.
class RecordingDiagnosticSink final : public detail::RuntimeStructuralDiagnosticSink {
 public:
  explicit RecordingDiagnosticSink(bool fail = false) : fail_(fail) {}

  void publish(std::span<const Diagnostic> diagnostics) override {
    if (fail_) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      throw std::runtime_error{"injected diagnostic sink failure"};
#else
      return;
#endif
    }
    const std::lock_guard lock(mutex_);
    sequences_.emplace_back(diagnostics.begin(), diagnostics.end());
  }

  void publish_runtime_details(std::span<const detail::RuntimeDiagnosticDetail> details) override {
    const std::lock_guard lock(mutex_);
    detail_sequences_.emplace_back(details.begin(), details.end());
  }

  void publish_lane_startup_evidence(
      std::span<const detail::ExecutionLaneStartupEvidence> evidence) override {
    const std::lock_guard lock(mutex_);
    lane_sequences_.emplace_back(evidence.begin(), evidence.end());
  }

  [[nodiscard]] std::vector<std::vector<Diagnostic>> snapshot() const {
    const std::lock_guard lock(mutex_);
    return sequences_;
  }

  [[nodiscard]] std::vector<std::vector<detail::RuntimeDiagnosticDetail>> snapshot_details() const {
    const std::lock_guard lock(mutex_);
    return detail_sequences_;
  }

  [[nodiscard]] std::vector<std::vector<detail::ExecutionLaneStartupEvidence>>
  snapshot_lane_evidence() const {
    const std::lock_guard lock(mutex_);
    return lane_sequences_;
  }

 private:
  bool fail_{};
  mutable std::mutex mutex_;
  std::vector<std::vector<Diagnostic>> sequences_;
  std::vector<std::vector<detail::RuntimeDiagnosticDetail>> detail_sequences_;
  std::vector<std::vector<detail::ExecutionLaneStartupEvidence>> lane_sequences_;
};

struct TerminalSummarySnapshot {
  RunGenerationId run_generation{};
  detail::RuntimeTerminalCategory category{detail::RuntimeTerminalCategory::kFailed};
  std::size_t diagnostic_count{};
  DiagnosticId first_diagnostic{};
  DiagnosticSource first_source{};
  std::string first_token;
  std::size_t structured_publication_count{};
};

class RecordingTerminalSummarySink final : public detail::RuntimeTerminalSummarySink {
 public:
  explicit RecordingTerminalSummarySink(
      std::shared_ptr<RecordingDiagnosticSink> structured_sink = {}, bool fail = false)
      : structured_sink_(std::move(structured_sink)), fail_(fail) {}

  void publish(const detail::RuntimeTerminalSummary& summary) override {
    const std::size_t structured_publication_count =
        structured_sink_ != nullptr ? structured_sink_->snapshot().size() : 0U;
    {
      const std::lock_guard lock(mutex_);
      snapshots_.push_back(TerminalSummarySnapshot{
          .run_generation = summary.run_generation,
          .category = summary.category,
          .diagnostic_count = summary.diagnostic_count,
          .first_diagnostic = summary.first_diagnostic,
          .first_source = summary.first_source,
          .first_token = std::string{summary.first_token},
          .structured_publication_count = structured_publication_count,
      });
    }
    if (fail_) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      throw std::runtime_error{"injected terminal summary sink failure"};
#endif
    }
  }

  [[nodiscard]] std::vector<TerminalSummarySnapshot> snapshot() const {
    const std::lock_guard lock(mutex_);
    return snapshots_;
  }

 private:
  std::shared_ptr<RecordingDiagnosticSink> structured_sink_;
  bool fail_{};
  mutable std::mutex mutex_;
  std::vector<TerminalSummarySnapshot> snapshots_;
};

[[nodiscard]] ExecutionPlan make_unselected_process_group_test_plan() {
  Graph graph{"runtime-unselected-process-group"};
  const auto producer = graph.op<test::RuntimeProbeOperator>("producer", 1);
  const auto consumer = graph.op<test::RuntimeProbeOperator>("consumer", 2);
  graph.partition("runtime-spin-producer").add(producer);
  graph.partition("runtime-spin-consumer").add(consumer);
  const ClockKey default_clock =
      graph.add_clock<test::RuntimeProbeClock>("runtime-unselected-process-group-clock");
  graph.set_default_clock(default_clock);

  constexpr std::array producer_selector{detail::GraphObjectSelector{.graph_path = "producer"}};
  constexpr std::array consumer_selector{detail::GraphObjectSelector{.graph_path = "consumer"}};
  CompileOptions options;
  options.deployment
      .bind_thread_placement("runtime-spin-producer",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("runtime-spin-consumer",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("producer",
                         detail::ProcessPlacement{
                             .members = producer_selector,
                             .launch_group = "producer-process",
                         })
      .bind_process_role("consumer",
                         detail::ProcessPlacement{
                             .members = consumer_selector,
                             .launch_group = "consumer-process",
                         });
  return detail::compile_with_process_placement(graph, std::move(options), std::move(placement));
}

#if GTEST_HAS_DEATH_TEST
// Triggers a self-join death scenario: publish() spins until the bound session
// pointer is set, then calls session->wait() from the runtime's own thread,
// which must terminate under external supervision.
// Used by: RunOwnedThreadSelfJoinTerminatesUnderExternalSupervision
class SelfJoinDiagnosticSink final : public detail::RuntimeStructuralDiagnosticSink {
 public:
  void publish(std::span<const Diagnostic>) override {
    RunSession* session = nullptr;
    while ((session = session_.load(std::memory_order_acquire)) == nullptr) {
      std::this_thread::yield();
    }
    session->wait();
  }

  void bind(RunSession& session) noexcept { session_.store(&session, std::memory_order_release); }

 private:
  std::atomic<RunSession*> session_{};
};
#endif

// REQ: RUNTIME-001
TEST(RunSessionTest, RejectsEmptyPlanBeforeCreatingSession) {
  const ExecutionPlan plan;
  EXPECT_THROW((void)run_async(plan), CompileError);
  EXPECT_THROW((void)run(plan), CompileError);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-013
TEST(RunSessionTest, RejectsStrayDeadlineInventoryInOperatorFreePlan) {
  Graph graph;
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const ExecutionPlan malformed = detail::RuntimeTestPlanAccess::with_stray_deadline(baseline);
  ASSERT_TRUE(malformed.ok()) << malformed.json();

  RunSession session = run_async(malformed);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
  EXPECT_TRUE(session.diagnostics().front().required.empty());
  EXPECT_TRUE(session.diagnostics().front().provided.empty());
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: RUNTIME-001
// REQ: RUNTIME-004
TEST(RunSessionTest, AcceptedPlanAllocationFailureReturnsTerminalSessionInsteadOfThrowing) {
  test::RuntimeProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(diagnostic_sink);
  const ExecutionPlan baseline = test::enable_direct_runtime(test::make_runtime_test_plan());
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);

  const RunSession session = [&] {
    const test::ScopedAllocationFailure fail_next_allocation{0U};
    return run_async(plan);
  }();
  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_STARTUP_FAILED");
  ASSERT_EQ(diagnostic_sink->snapshot().size(), 1U);
  auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 1U);
  EXPECT_NE(summaries.front().run_generation, 0U);
  EXPECT_EQ(summaries.front().first_source.graph_generation, plan.graph().generation());
  EXPECT_EQ(summaries.front().first_token, "RUNTIME_STARTUP_FAILED");
  EXPECT_EQ(summaries.front().structured_publication_count, 1U);

  int blocking_result{};
  {
    const test::ScopedAllocationFailure fail_next_allocation{0U};
    blocking_result = run(plan);
  }
  EXPECT_NE(blocking_result, 0);
  ASSERT_EQ(diagnostic_sink->snapshot().size(), 2U);
  summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 2U);
  EXPECT_NE(summaries.back().run_generation, 0U);
  EXPECT_EQ(summaries.back().first_source.graph_generation, plan.graph().generation());
  EXPECT_EQ(summaries.back().first_token, "RUNTIME_STARTUP_FAILED");
  EXPECT_EQ(summaries.back().structured_publication_count, 2U);
}
#endif

// REQ: RUNTIME-001
// REQ: RUNTIME-004
// REQ: RUNTIME-035
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
  EXPECT_NE(run(plan), 0);
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

// REQ: RUNTIME-036
TEST(RunSessionTest, UnselectedMultiProcessPlanFailsBeforeRuntimeMaterialization) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan baseline = make_unselected_process_group_test_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto roles = detail::ProcessPlacementPlanAccess::process_roles(baseline);
  const auto process_placements =
      detail::ProcessPlacementPlanAccess::vertex_process_placements(baseline);
  ASSERT_EQ(roles.size(), 2U);
  ASSERT_EQ(process_placements.size(), 2U);
  ASSERT_NE(process_placements[0].process_group, process_placements[1].process_group);

  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(diagnostic_sink);
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);

  // Compilation constructs the authored instances. Reset after admission so any runtime replay,
  // lifecycle callback, or compute callback becomes observable.
  test::RuntimeProbeState::reset();
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_TRUE(session.diagnostics().front().id);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_PROCESS_GROUP_SELECTION_REQUIRED");
  EXPECT_EQ(test::RuntimeProbeState::constructed.load(std::memory_order_acquire), 0U);
  EXPECT_FALSE(test::RuntimeProbeState::setup_entered.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeProbeState::active.load(std::memory_order_acquire), 0U);
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());

  const auto published = diagnostic_sink->snapshot();
  ASSERT_EQ(published.size(), 1U);
  expect_diagnostic_sequences_equal(session.diagnostics(), published.front());

  const auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 1U);
  EXPECT_NE(summaries.front().run_generation, 0U);
  EXPECT_EQ(summaries.front().diagnostic_count, 1U);
  EXPECT_EQ(summaries.front().first_diagnostic, session.diagnostics().front().id);
  EXPECT_EQ(summaries.front().first_token, "RUNTIME_PROCESS_GROUP_SELECTION_REQUIRED");
  EXPECT_EQ(summaries.front().structured_publication_count, 1U);
}

// REQ: RUNTIME-036
TEST(RunSessionTest, InvalidExplicitProcessGroupUsesGenericStartupBoundary) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan baseline = make_unselected_process_group_test_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(diagnostic_sink);
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);
  constexpr ProcessGroupId kInvalidProcessGroup{std::numeric_limits<std::uint32_t>::max()};

  test::RuntimeProbeState::reset();
  RunSession session = detail::run_async_worker(plan, kInvalidProcessGroup);
  session.wait();

  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_STARTUP_FAILED");
  EXPECT_EQ(test::RuntimeProbeState::constructed.load(std::memory_order_acquire), 0U);
  EXPECT_FALSE(test::RuntimeProbeState::setup_entered.load(std::memory_order_acquire));
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  ASSERT_EQ(diagnostic_sink->snapshot().size(), 1U);
  const auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 1U);
  EXPECT_EQ(summaries.front().run_generation, 0U);
  EXPECT_EQ(summaries.front().first_source.graph_generation, plan.graph().generation());
  EXPECT_EQ(summaries.front().first_token, "RUNTIME_STARTUP_FAILED");
  EXPECT_EQ(summaries.front().structured_publication_count, 1U);
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
  EXPECT_NE(run(plan), 0);
  EXPECT_TRUE(sink->snapshot().empty());
}
#endif

// REQ: RUNTIME-035
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

#if GTEST_HAS_DEATH_TEST
// REQ: RUNTIME-003
TEST(RunSessionDeathTest, RunOwnedThreadSelfJoinTerminatesUnderExternalSupervision) {
  EXPECT_DEATH(
      {
        test::RuntimeProbeState::reset();
        const auto sink = std::make_shared<SelfJoinDiagnosticSink>();
        const ExecutionPlan direct = test::enable_direct_runtime(test::make_runtime_test_plan());
        const ExecutionPlan plan =
            detail::RuntimeTestPlanAccess::with_diagnostic_sink(direct, sink);
        RunSession session = run_async(plan);
        sink->bind(session);
        session.wait();
      },
      "RUNTIME_SELF_JOIN_FORBIDDEN");
}

void expect_runtime_callback_session_action_terminates(test::RuntimeProbeMode mode,
                                                       bool destroy_session) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(mode);
  if (destroy_session) {
    auto* session = new RunSession{run_async(plan)};
    test::RuntimeProbeState::bind_callback_session(session);
    while (!test::RuntimeProbeState::callback_action_returned()) {
      std::this_thread::yield();
    }
  } else {
    RunSession session = run_async(plan);
    test::RuntimeProbeState::bind_callback_session(&session);
    session.wait();
  }
}

// REQ: RUNTIME-003
TEST(RunSessionDeathTest, WaitFromComputeCallbackTerminatesUnderExternalSupervision) {
  EXPECT_DEATH(
      {
        expect_runtime_callback_session_action_terminates(test::RuntimeProbeMode::kWaitFromCompute,
                                                          false);
      },
      "RUNTIME_SELF_JOIN_FORBIDDEN");
}

// REQ: RUNTIME-003
TEST(RunSessionDeathTest, WaitFromDispatcherContextTerminatesUnderExternalSupervision) {
  EXPECT_DEATH(
      {
        test::RuntimeProbeState::reset();
        const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
        RunSession session = run_async(plan);
        const detail::RuntimeCallbackContextScope dispatcher_context{
            detail::RuntimeCallbackContext::kDispatcher};
        session.wait();
      },
      "RUNTIME_SELF_JOIN_FORBIDDEN");
}

// REQ: RUNTIME-003
TEST(RunSessionDeathTest, DestructorFromComputeCallbackTerminatesUnderExternalSupervision) {
  EXPECT_DEATH(
      {
        expect_runtime_callback_session_action_terminates(
            test::RuntimeProbeMode::kDestroyFromCompute, true);
      },
      ".*");
}

// REQ: RUNTIME-003
TEST(RunSessionDeathTest, WaitFromLifecycleCallbackTerminatesUnderExternalSupervision) {
  EXPECT_DEATH(
      {
        expect_runtime_callback_session_action_terminates(
            test::RuntimeProbeMode::kWaitFromLifecycle, false);
      },
      ".*");
}

// REQ: RUNTIME-003
TEST(RunSessionDeathTest, DestructorFromLifecycleCallbackTerminatesUnderExternalSupervision) {
  EXPECT_DEATH(
      {
        expect_runtime_callback_session_action_terminates(
            test::RuntimeProbeMode::kDestroyFromLifecycle, true);
      },
      ".*");
}
#endif

// REQ: RUNTIME-001
// REQ: RUNTIME-011
TEST(RunSessionTest, RunsCompilerAcceptedEmptyGraphAsNoOp) {
  Graph graph{"empty-runtime-plan"};
  const ExecutionPlan compiled = compile(graph);
  const auto sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(compiled, sink);
  ASSERT_TRUE(plan.ok());
  EXPECT_TRUE(plan.graph().vertices().empty());
  EXPECT_TRUE(plan.execution_partitions().empty());
  EXPECT_TRUE(plan.partition_decisions().empty());
  EXPECT_TRUE(plan.execution_lanes().empty());
  EXPECT_TRUE(plan.partition_placements().empty());
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 0U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 0U);
  EXPECT_TRUE(plan.profile_manifest().materialization_obligations.empty());

  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);
  auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  EXPECT_TRUE((*reconstructed)->parameter_updates().catalog().operators().empty());
  EXPECT_EQ((*reconstructed)->parameter_updates().max_event_cursors(), 0U);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  const auto lane_evidence = sink->snapshot_lane_evidence();
  ASSERT_EQ(lane_evidence.size(), 1U);
  EXPECT_TRUE(lane_evidence.front().empty());
  EXPECT_EQ(run(plan), 0);
}

// REQ: PARAM-017
TEST(RunSessionTest, RejectsNoncontiguousParameterOwnerInventoriesBeforeReconstruction) {
  using Mutation = detail::RuntimeTestPlanAccess::ParameterInventoryMutation;
  const std::array cases{
      std::pair{test::make_runtime_two_param_test_plan(), Mutation::kInterleavedFields},
      std::pair{test::make_runtime_two_deadline_param_test_plan(),
                Mutation::kInterleavedDerivedTerms},
  };
  for (const auto& [baseline, mutation] : cases) {
    ASSERT_TRUE(baseline.ok()) << baseline.json();
    const ExecutionPlan malformed =
        detail::RuntimeTestPlanAccess::interleave_parameter_inventory(baseline, mutation);
    ASSERT_TRUE(malformed.ok()) << malformed.json();
    const auto accepted = detail::ExecutionPlanAccess::retain(malformed);
    ASSERT_NE(accepted, nullptr);
    const auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});
    ASSERT_FALSE(reconstructed);
    EXPECT_EQ(reconstructed.error().code, ErrorCode::kFailure);
    EXPECT_EQ(reconstructed.error().message, "RUNTIME_PARAMETER_INVENTORY_MISMATCH");
  }
}

// REQ: RUNTIME-010
// REQ: RUNTIME-011
TEST(RunSessionTest, EmptyGraphTreatsAuthoredClockCatalogAsInertAuthoringEvidence) {
  Graph graph{"empty-runtime-plan-with-clocks"};
  const ClockKey default_clock = graph.add_clock<test::RuntimeProbeClock>("default-clock");
  const ClockKey comparison_clock = graph.add_clock<test::RuntimeProbeClock>("comparison-clock");
  graph.set_default_clock(default_clock);
  graph.add_clock_conversion(default_clock, comparison_clock, AffineClockConversion{});
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.clock_catalog().clocks.size(), 2U);
  ASSERT_EQ(plan.clock_conversions().size(), 1U);
  EXPECT_TRUE(plan.profile_manifest().materialization_obligations.empty());

  test::RuntimeProbeState::reset();
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeProbeState::clocks_constructed.load(std::memory_order_relaxed), 0U);
}

// REQ: RUNTIME-010
TEST(RunSessionTest, EmptyGraphRejectsTamperedInertClockInventoryWithoutConstruction) {
  using InertClockMutation = detail::RuntimeTestPlanAccess::InertClockMutation;
  constexpr std::array mutations{
      InertClockMutation::kPlanClockKey,
      InertClockMutation::kMissingFactory,
      InertClockMutation::kFactoryMode,
      InertClockMutation::kRootDefault,
      InertClockMutation::kConversionGeneration,
      InertClockMutation::kConversionEndpoint,
  };
  Graph graph{"empty-runtime-plan-with-corrupted-clocks"};
  const ClockKey default_clock = graph.add_clock<test::RuntimeProbeClock>("default-clock");
  const ClockKey comparison_clock = graph.add_clock<test::RuntimeProbeClock>("comparison-clock");
  graph.set_default_clock(default_clock);
  graph.add_clock_conversion(default_clock, comparison_clock, AffineClockConversion{});
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok()) << baseline.json();

  for (const InertClockMutation mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    test::RuntimeProbeState::reset();
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::mutate_inert_clock(baseline, mutation);
    ASSERT_TRUE(plan.ok());

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    EXPECT_EQ(test::RuntimeProbeState::clocks_constructed.load(std::memory_order_relaxed), 0U);
  }
}

// REQ: RUNTIME-011
TEST(RunSessionTest, RejectsMalformedZeroWorkSchedulerAndPartitionFacts) {
  using ZeroWorkMutation = detail::RuntimeTestPlanAccess::ZeroWorkMutation;
  constexpr std::array mutations{
      ZeroWorkMutation::kWorkerCount,
      ZeroWorkMutation::kExecutionLaneCount,
      ZeroWorkMutation::kMaxConcurrentActivations,
      ZeroWorkMutation::kIntraPathParallelism,
      ZeroWorkMutation::kExecutionLane,
      ZeroWorkMutation::kOrphanPartition,
      ZeroWorkMutation::kPartitionDecision,
      ZeroWorkMutation::kScheduleEdge,
      ZeroWorkMutation::kPlacementProbe,
      ZeroWorkMutation::kPartitionPlacement,
  };
  Graph graph{"malformed-empty-runtime-plan"};
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok());

  for (const ZeroWorkMutation mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    const ExecutionPlan plan = detail::RuntimeTestPlanAccess::mutate_zero_work(baseline, mutation);
    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_PLACEMENT_BINDING_UNAVAILABLE"));
  }
}

// REQ: RUNTIME-010
TEST(RunSessionTest, RejectsMalformedZeroWorkRuntimeOwnedInventoriesBeforeReconstruction) {
  using ZeroWorkMutation = detail::RuntimeTestPlanAccess::ZeroWorkMutation;
  struct Case {
    ZeroWorkMutation mutation;
    std::string_view token;
  };
  constexpr std::array cases{
      Case{ZeroWorkMutation::kOperatorFactory, "RUNTIME_OPERATOR_INVENTORY_MISMATCH"},
      Case{ZeroWorkMutation::kMaterializationObligation, "PROFILE_MATERIALIZATION_FAILED"},
  };
  Graph graph{"malformed-empty-runtime-inventory"};
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok());

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.token);
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::mutate_zero_work(baseline, test_case.mutation);
    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, test_case.token));
  }
}

// REQ: RUNTIME-010
TEST(RunSessionTest, EmptyGraphStillRejectsEveryPackageMismatch) {
  using ProfileMutation = detail::RuntimeTestPlanAccess::ProfileMutation;
  struct Case {
    ProfileMutation mutation;
    std::string_view token;
  };
  constexpr std::array cases{
      Case{ProfileMutation::kPlanSchema, "PROFILE_PLAN_SCHEMA_UNSUPPORTED"},
      Case{ProfileMutation::kProfilePolicy, "PROFILE_POLICY_UNSUPPORTED"},
      Case{ProfileMutation::kRuntimeAbi, "PROFILE_RUNTIME_ABI_INCOMPATIBLE"},
      Case{ProfileMutation::kTarget, "PROFILE_TARGET_INCOMPATIBLE"},
      Case{ProfileMutation::kCapability, "PROFILE_CAPABILITY_UNSATISFIED"},
      Case{ProfileMutation::kEvidence, "PROFILE_EVIDENCE_UNSATISFIED"},
  };
  Graph graph{"empty-runtime-profile-plan"};
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok());

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.token);
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::mutate_profile(baseline, test_case.mutation);
    ASSERT_TRUE(plan.ok());

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, test_case.token));
  }
}

// REQ: RUNTIME-001
// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-041
TEST(RunSessionTest, RunsAfterSourceGraphIsDestroyedAndPreservesPartitionLocalOrder) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan(true));
  ASSERT_TRUE(plan.ok()) << plan.json();
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  const auto events = test::RuntimeProbeState::snapshot();
  auto sorted_events = events;
  std::sort(sorted_events.begin(), sorted_events.end());
  EXPECT_EQ(sorted_events, (std::vector<std::int32_t>{11, 12, 13, 21, 22, 23}));
  for (const std::int32_t identity : {1, 2}) {
    const auto start = std::find(events.begin(), events.end(), identity * 10 + 1);
    const auto compute = std::find(events.begin(), events.end(), identity * 10 + 2);
    const auto stop = std::find(events.begin(), events.end(), identity * 10 + 3);
    ASSERT_NE(start, events.end());
    ASSERT_NE(compute, events.end());
    ASSERT_NE(stop, events.end());
    EXPECT_LT(start, compute);
    EXPECT_LT(compute, stop);
  }
  const auto max_active = test::RuntimeProbeState::max_active.load(std::memory_order_relaxed);
  EXPECT_GE(max_active, 1U);
  EXPECT_LE(max_active, 2U);
}

// REQ: RUNTIME-004
// REQ: LIFECYCLE-004
TEST(RunSessionTest, PreservesAuthoredStopDependenciesWhileReversingIndependentOperators) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan =
      test::enable_direct_runtime(test::make_runtime_stop_dependency_test_plan());
  ASSERT_TRUE(plan.ok()) << plan.json();
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  const auto events = test::RuntimeProbeState::snapshot();
  auto sorted_events = events;
  std::sort(sorted_events.begin(), sorted_events.end());
  EXPECT_EQ(sorted_events, (std::vector<std::int32_t>{11, 12, 13, 21, 22, 23}));
  for (const std::int32_t identity : {1, 2}) {
    const auto start = std::find(events.begin(), events.end(), identity * 10 + 1);
    const auto compute = std::find(events.begin(), events.end(), identity * 10 + 2);
    const auto stop = std::find(events.begin(), events.end(), identity * 10 + 3);
    ASSERT_NE(start, events.end());
    ASSERT_NE(compute, events.end());
    ASSERT_NE(stop, events.end());
    EXPECT_LT(start, compute);
    EXPECT_LT(compute, stop);
  }
  const auto first_stop = std::find(events.begin(), events.end(), 13);
  const auto dependent_stop = std::find(events.begin(), events.end(), 23);
  ASSERT_NE(first_stop, events.end());
  ASSERT_NE(dependent_stop, events.end());
  EXPECT_LT(first_stop, dependent_stop);
}

// REQ: RUNTIME-001
// REQ: RUNTIME-002
TEST(RunSessionTest, BlockingRunCompletesThroughExplicitBackendWithBoundClockContext) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset();

  EXPECT_EQ(run(plan), 0);
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
  EXPECT_EQ(test::RuntimeProbeState::constructed.load(std::memory_order_relaxed), 1U);
  EXPECT_TRUE(
      test::RuntimeProbeState::observed_clock_read_succeeded.load(std::memory_order_relaxed));
  const auto clock_id = test::RuntimeProbeState::observed_clock_id.load(std::memory_order_relaxed);
  EXPECT_NE(clock_id, 0U);
  EXPECT_EQ(test::RuntimeProbeState::observed_activation_clock_id.load(std::memory_order_relaxed),
            clock_id);
  EXPECT_EQ(test::RuntimeProbeState::observed_activation_time_ns.load(std::memory_order_relaxed),
            0);
  EXPECT_EQ(test::RuntimeProbeState::observed_clock_time_ns.load(std::memory_order_relaxed), 0);
}

// Source-tree performance counters are absent from ordinary compiled plans and, when explicitly
// attached, count the same accepted candidate through freeze, selection, compute, and retirement.
// REQ: RUNTIME-011
TEST(RunSessionTest, PrivatePhaseCountersTrackOneCompleteActivationPerDirectWitness) {
  constexpr std::uint32_t kActivationCount = 3U;
  constexpr std::array expected_points{
      detail::RuntimePhaseTracePoint::kCandidateFrozen,
      detail::RuntimePhaseTracePoint::kComputeDispatchStarted,
      detail::RuntimePhaseTracePoint::kComputeCallbackEntered,
      detail::RuntimePhaseTracePoint::kComputeCallbackReturned,
      detail::RuntimePhaseTracePoint::kComputeDispatchReturned,
      detail::RuntimePhaseTracePoint::kActivationRetirementCompleted,
      detail::RuntimePhaseTracePoint::kActivationCutCompleted,
  };
  test::RuntimeProbeState::reset();
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  const auto baseline_state = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_TRUE(baseline_state);
  ASSERT_EQ(baseline_state->executable_factories.phase_counters, nullptr);

  std::array<detail::RuntimePhaseTraceRecord, kActivationCount * expected_points.size()>
      trace_storage;
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>(std::span{trace_storage});
  const ExecutionPlan direct = test::enable_direct_runtime(baseline, kActivationCount);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(direct, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  // Key setup: run three finite direct witnesses through the production coordinator and lane.
  RunSession session = run_async(plan);
  session.wait();
  ASSERT_EQ(session.termination(), RunTermination::kCompleted);
  const auto snapshot = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };

  // Key assertion: every accepted witness has exactly one frozen candidate, selected compute,
  // and completed activation; the direct backend needs neither Channel polling nor a parked wait.
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kCandidatesFrozen), kActivationCount);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kCandidatesSelected), kActivationCount);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kComputeDispatches), kActivationCount);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kActivationsCompleted), kActivationCount);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kReadinessPollDispatches), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kWaitCalls), 0U);

  // Key assertion: opt-in trace storage preserves exact controller checkpoint order without
  // inventing a readiness poll for this direct source.
  const auto trace = counters->trace_records();
  ASSERT_EQ(trace.size(), trace_storage.size());
  EXPECT_EQ(counters->trace_dropped(), 0U);
  for (std::size_t activation = 0U; activation < kActivationCount; ++activation) {
    const auto group = trace.subspan(activation * expected_points.size(), expected_points.size());
    const SourceId source = group.front().source;
    const std::uint64_t source_sequence = group.front().source_sequence;
    EXPECT_TRUE(source);
    EXPECT_NE(source_sequence, 0U);
    for (std::size_t point = 0U; point < group.size(); ++point) {
      EXPECT_EQ(group[point].point, expected_points[point]);
      EXPECT_EQ(group[point].receiver_index, 0U);
      EXPECT_EQ(group[point].source, source);
      EXPECT_EQ(group[point].source_sequence, source_sequence);
      if (point != 0U) {
        EXPECT_LE(group[point - 1U].timestamp_ns, group[point].timestamp_ns);
      }
    }
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
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));

  const auto published = sink->snapshot_details();
  ASSERT_EQ(published.size(), 1U);
  ASSERT_EQ(published.front().size(), 1U);
  EXPECT_TRUE(published.front().front().present);
  EXPECT_EQ(published.front().front().error.code, ErrorCode::kResourceExhausted);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-011
TEST(RunSessionTest, RoutesLifecycleAndComputeThroughAcceptedPartitionLane) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan compiled = test::make_runtime_role_test_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.execution_lanes().size(), 1U);
  ASSERT_EQ(compiled.scheduler_plan().execution_lane_count, 1U);
  const ExecutionPlan plan = test::enable_direct_runtime(compiled);
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted)
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
  const auto callback_threads = test::RuntimeProbeState::snapshot_threads();
  ASSERT_EQ(callback_threads.size(), 3U);
  EXPECT_NE(callback_threads.front(), std::thread::id{});
  EXPECT_TRUE(std::all_of(callback_threads.begin(), callback_threads.end(), [&](const auto& id) {
    return id == callback_threads.front();
  }));
}

// REQ: RUNTIME-002
TEST(RunSessionTest, ReconstructsEveryPlannedClockCapability) {
  const auto expect_capabilities = [](const ExecutionPlan& source) {
    ASSERT_EQ(source.clock_catalog().clocks.size(), 1U);
    const ClockPlanRecord planned = source.clock_catalog().clocks.front();
    const ExecutionPlan plan = test::enable_direct_runtime(source);
    test::RuntimeProbeState::reset();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_EQ(test::RuntimeProbeState::observed_clock_can_read_now.load(std::memory_order_relaxed),
              planned.can_read_now);
    EXPECT_EQ(
        test::RuntimeProbeState::observed_clock_can_wait_until.load(std::memory_order_relaxed),
        planned.can_wait_until);
    EXPECT_EQ(test::RuntimeProbeState::observed_clock_steady.load(std::memory_order_relaxed),
              planned.steady);
  };

  expect_capabilities(test::make_runtime_test_plan_with_clock<RealtimeClock>(false));
  expect_capabilities(test::make_runtime_test_plan_with_clock<ManualClock>(false, std::int64_t{0}));
  expect_capabilities(
      test::make_runtime_test_plan_with_clock<SyntheticClock>(false, std::int64_t{0}));
  expect_capabilities(test::make_runtime_test_plan());
  EXPECT_FALSE(test::RuntimeProbeState::observed_clock_steady.load(std::memory_order_relaxed));
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-027
TEST(RunSessionTest, RejectsReplayedContractMismatchBeforeLifecycleCallbacks) {
  struct CallbackProbeReset {
    CallbackProbeReset() {
      test::RuntimeParamProbeState::reset();
      test::RuntimeDeadlineParamProbeState::reset();
    }
    ~CallbackProbeReset() {
      test::RuntimeParamProbeState::reset();
      test::RuntimeDeadlineParamProbeState::reset();
    }
  };
  const CallbackProbeReset callback_probe_reset{};
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kMismatchedContract);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
  {
    const std::lock_guard lock(test::RuntimeParamProbeState::mutex);
    EXPECT_FALSE(test::RuntimeParamProbeState::start_entered);
    EXPECT_FALSE(test::RuntimeParamProbeState::stop_entered);
  }
  EXPECT_EQ(test::RuntimeDeadlineParamProbeState::compute_count.load(std::memory_order_acquire),
            0U);
}

// REQ: RUNTIME-004
// REQ: LIFECYCLE-002
TEST(RunSessionTest, RejectsMissingAdvancedOptionalLifecycleCallbacksDuringReplay) {
  struct ReplayReset {
    ~ReplayReset() {
      OptionalLifecycleReplayProbe::omit_callbacks.store(false, std::memory_order_relaxed);
    }
  };
  const ReplayReset reset{};

  Graph graph{"optional-lifecycle-replay"};
  static_cast<void>(graph.op<OptionalLifecycleReplayProbe>("probe"));
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  OptionalLifecycleReplayProbe::omit_callbacks.store(true, std::memory_order_relaxed);
  auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                      RunGenerationId{29U});

  ASSERT_FALSE(reconstructed);
  EXPECT_EQ(reconstructed.error().message, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
}

// A runtime-mutable deadline retains its independently frozen violation action. Replaying the
// same Parameter dependency must not allow a different action to bypass cold reconstruction.
// REQ: RUNTIME-004
// REQ: RUNTIME-027
// REQ: PARAM-017
TEST(RunSessionTest, RejectsParameterDerivedDeadlineViolationReplayMismatchBeforeCallbacks) {
  struct ReplayViolationReset {
    ~ReplayViolationReset() { test::RuntimeDeadlineParamProbeState::reset(); }
  };
  const ReplayViolationReset reset{};

  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  test::RuntimeDeadlineParamProbeState::reset();
  const ExecutionPlan compiled = test::make_runtime_deadline_param_test_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(compiled);

  test::RuntimeDeadlineParamProbeState::replay_violation.store(OnViolation::kSafeDefault,
                                                               std::memory_order_relaxed);
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
}

// REQ: RUNTIME-004
// REQ: RUNTIME-016
TEST(RunSessionTest, ReconstructsPhaseSeparatedInputPredicateWithoutFakeReadinessSource) {
  const ExecutionPlan plan = test::make_runtime_input_predicate_replay_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  ASSERT_EQ(plan.temporal_triggers().size(), 2U);
  const auto predicate_trigger = std::find_if(
      plan.temporal_triggers().begin(), plan.temporal_triggers().end(), [](const auto& trigger) {
        return trigger.kind == TemporalRecordTriggerKind::kPredicate;
      });
  ASSERT_NE(predicate_trigger, plan.temporal_triggers().end());
  ASSERT_TRUE(static_cast<bool>(predicate_trigger->readiness_source));
  EXPECT_TRUE(std::none_of(plan.readiness().begin(), plan.readiness().end(), [&](const auto& row) {
    return row.source_id == predicate_trigger->readiness_source;
  }));

  auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                      RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  EXPECT_EQ((*reconstructed)->operators().size(), 2U);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-015
TEST(RunSessionTest, ReconstructsEveryInputBackedTriggerWithoutFakeReadinessRecords) {
  constexpr std::array trigger_kinds{
      test::RuntimeInputTriggerReplayKind::kAny,
      test::RuntimeInputTriggerReplayKind::kEach,
      test::RuntimeInputTriggerReplayKind::kAligned,
      test::RuntimeInputTriggerReplayKind::kWindow,
  };
  for (const auto kind : trigger_kinds) {
    SCOPED_TRACE(static_cast<unsigned>(kind));
    const ExecutionPlan plan = test::make_runtime_input_trigger_replay_plan(kind);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.temporal_triggers().size(), 2U);
    const auto trigger = std::find_if(
        plan.temporal_triggers().begin(), plan.temporal_triggers().end(), [](const auto& row) {
          return row.kind != TemporalRecordTriggerKind::kClock;
        });
    ASSERT_NE(trigger, plan.temporal_triggers().end());
    ASSERT_TRUE(static_cast<bool>(trigger->readiness_source));
    EXPECT_TRUE(
        std::none_of(plan.readiness().begin(), plan.readiness().end(), [&](const auto& row) {
          return row.source_id == trigger->readiness_source;
        }));

    auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                        RunGenerationId{18U});
    ASSERT_TRUE(reconstructed) << reconstructed.error().message;
    EXPECT_EQ((*reconstructed)->operators().size(), 2U);
  }
}

// Clock scheduling is independent of input arrival, but its callback still consumes the complete
// frozen selector, missing-policy, and deadline plan. Replay must preserve both dimensions.
// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-016
TEST(RunSessionTest, ReconstructsClockDrivenOperatorsWithConnectedInputs) {
  constexpr std::array kinds{
      test::RuntimeClockInputReplayKind::kClock,
      test::RuntimeClockInputReplayKind::kPredicate,
  };
  constexpr std::array profiles{Profile::kStandard, Profile::kResearch};
  for (const auto profile : profiles) {
    for (const auto kind : kinds) {
      SCOPED_TRACE(static_cast<unsigned>(profile));
      SCOPED_TRACE(static_cast<unsigned>(kind));
      const ExecutionPlan plan = test::make_runtime_clock_input_replay_plan(kind, profile);
      ASSERT_TRUE(plan.ok()) << plan.json();
      ASSERT_EQ(plan.input_selections().size(), 2U);
      ASSERT_EQ(plan.deadlines().size(), 1U);
      ASSERT_EQ(plan.clock_triggers().size(), 2U);

      auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                          RunGenerationId{19U});
      ASSERT_TRUE(reconstructed) << reconstructed.error().message;
      EXPECT_EQ((*reconstructed)->operators().size(), 2U);
    }
  }
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-016
// REQ: RUNTIME-027
TEST(RunSessionTest, ClockReplayMismatchCarriesStableOperatorAndFingerprintContext) {
  struct ReplayModeReset {
    ~ReplayModeReset() {
      test::RuntimeClockInputReplayOperator::mismatch_ordinal.store(0U, std::memory_order_relaxed);
    }
  };
  const ReplayModeReset reset{};

  const ExecutionPlan baseline = test::make_runtime_dual_clock_input_replay_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();

  const auto reconstruct_mismatch = [&](std::uint32_t ordinal) {
    test::RuntimeClockInputReplayOperator::mismatch_ordinal.store(ordinal,
                                                                  std::memory_order_relaxed);
    detail::RuntimeReplayMismatchDetail mismatch;
    auto reconstructed = detail::GraphExec::reconstruct(
        detail::ExecutionPlanAccess::retain(baseline), RunGenerationId{19U}, {}, &mismatch);
    EXPECT_FALSE(reconstructed);
    if (!reconstructed) {
      EXPECT_EQ(reconstructed.error().message, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
    }
    return mismatch;
  };

  const detail::RuntimeReplayMismatchDetail ordinary = reconstruct_mismatch(1U);
  const detail::RuntimeReplayMismatchDetail predicate = reconstruct_mismatch(2U);
  ASSERT_TRUE(ordinary.present);
  ASSERT_TRUE(predicate.present);
  EXPECT_EQ(ordinary.category, detail::RuntimeReplayMismatchCategory::kTemporalContract);
  EXPECT_EQ(predicate.category, detail::RuntimeReplayMismatchCategory::kTemporalContract);
  EXPECT_EQ(ordinary.factory_slot, 1U);
  EXPECT_EQ(predicate.factory_slot, 2U);
  EXPECT_EQ(ordinary.operator_id, 1U);
  EXPECT_EQ(predicate.operator_id, 2U);
  EXPECT_NE(ordinary.vertex, predicate.vertex);
  EXPECT_NE(ordinary.expected, predicate.expected);
  EXPECT_NE(ordinary.observed, predicate.observed);

  const auto sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, sink);
  test::RuntimeClockInputReplayOperator::mismatch_ordinal.store(1U, std::memory_order_relaxed);
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const Diagnostic& diagnostic = session.diagnostics().front();
  EXPECT_EQ(diagnostic.token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
  EXPECT_EQ(diagnostic.source.vertex, ordinary.vertex);
  EXPECT_EQ(diagnostic.required.size(), 71U);
  EXPECT_EQ(diagnostic.provided.size(), 71U);
  EXPECT_TRUE(diagnostic.required.starts_with("sha256:"));
  EXPECT_TRUE(diagnostic.provided.starts_with("sha256:"));
  EXPECT_NE(diagnostic.required, diagnostic.provided);

  const auto published = sink->snapshot_details();
  ASSERT_EQ(published.size(), 1U);
  ASSERT_EQ(published.front().size(), 1U);
  const detail::RuntimeDiagnosticDetail& retained = published.front().front();
  EXPECT_TRUE(retained.present);
  EXPECT_EQ(retained.error.code, ErrorCode::kFailure);
  EXPECT_EQ(retained.operator_id, ordinary.operator_id);
  EXPECT_EQ(retained.replay_mismatch.vertex, ordinary.vertex);
  EXPECT_EQ(retained.replay_mismatch.operator_id, ordinary.operator_id);
  EXPECT_EQ(retained.replay_mismatch.factory_slot, ordinary.factory_slot);
  EXPECT_EQ(retained.replay_mismatch.category, ordinary.category);
  EXPECT_EQ(retained.replay_mismatch.expected, ordinary.expected);
  EXPECT_EQ(retained.replay_mismatch.observed, ordinary.observed);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: PARAM-017
TEST(RunSessionTest, BindsTypedParameterGenerationZeroBeforeFirstCallback) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_param_test_plan());
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  const auto observations = test::RuntimeParamProbeState::snapshot();
  ASSERT_EQ(observations.size(), 1U);
  EXPECT_EQ(observations[0].before_value, 7);
  EXPECT_EQ(observations[0].after_value, 7);
  EXPECT_EQ(observations[0].before_generation, 0U);
  EXPECT_EQ(observations[0].after_generation, 0U);
  EXPECT_EQ(observations[0].before_address, observations[0].after_address);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-007
TEST(RunSessionTest, FreezesActivationTimeAfterEveryStartCallbackCompletes) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kAdvanceClockInStart);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_EQ(test::RuntimeProbeState::observed_activation_time_ns.load(std::memory_order_relaxed),
            37);
  EXPECT_EQ(test::RuntimeProbeState::observed_clock_time_ns.load(std::memory_order_relaxed), 37);
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
}

// REQ: RUNTIME-004
// REQ: RUNTIME-012
TEST(RunSessionTest, OnClockRepeatsDeterministicallyUntilCooperativeStop) {
  constexpr std::size_t kObservedActivations = 3U;
  constexpr std::size_t kRuns = 2U;
  std::array<std::array<Tag, kObservedActivations>, kRuns> tags{};
  std::array<std::array<TagTime, kObservedActivations>, kRuns> tag_times{};
  std::array<std::array<TimePoint, kObservedActivations>, kRuns> activation_times{};

  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_clock_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  for (std::size_t run_index = 0U; run_index < kRuns; ++run_index) {
    test::RuntimeProbeState::reset();
    test::RuntimeClockProbeState::reset();
    RunSession session = run_async(plan);
    ASSERT_TRUE(test::RuntimeClockProbeState::wait_until_compute_count(kObservedActivations));

    session.request_stop();
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty());
    ASSERT_GE(test::RuntimeClockProbeState::compute_count.load(std::memory_order_acquire),
              kObservedActivations);
    std::copy_n(
        test::RuntimeClockProbeState::tags.begin(), kObservedActivations, tags[run_index].begin());
    std::copy_n(test::RuntimeClockProbeState::tag_times.begin(),
                kObservedActivations,
                tag_times[run_index].begin());
    std::copy_n(test::RuntimeClockProbeState::activation_times.begin(),
                kObservedActivations,
                activation_times[run_index].begin());
  }

  EXPECT_EQ(tags[0], tags[1]);
  EXPECT_EQ(tag_times[0], tag_times[1]);
  EXPECT_EQ(activation_times[0], activation_times[1]);
  for (std::size_t index = 0U; index < kObservedActivations; ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(tags[0][index], (Tag{index + 1U, 1U, 0U}));
    EXPECT_EQ(
        tag_times[0][index],
        (TagTime{
            static_cast<std::int64_t>(index) * test::RuntimeClockProbeState::kPeriodNs, 0U, 0U}));
    EXPECT_EQ(activation_times[0][index].timestamp_ns, tag_times[0][index].logical_ns);
    EXPECT_EQ(activation_times[0][index].clock_id, activation_times[0].front().clock_id);
  }
  EXPECT_TRUE(activation_times[0].front().valid());
}

// The ordinary run path, without an injected binding factory, must expose the exact bounded
// sender only during Arm and preserve capacity through complete callback retirement.
// REQ: LIFECYCLE-005
// REQ: RUNTIME-025
TEST(RunSessionTest, NotificationSenderIsLifecycleBoundCapacityLimitedAndClosedAtStop) {
  // Key setup: compile and run an OnNotified graph without an injected PortBinding factory.
  test::RuntimeNotificationProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_notification_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeNotificationProbeState::wait_until_sender());
  // Key assertion: Arm can acquire only the admitted sender, but cannot post before global Start.
  EXPECT_EQ(test::RuntimeNotificationProbeState::arm_post_error, ErrorCode::kNotReady);
  EXPECT_EQ(test::RuntimeNotificationProbeState::second_acquire_error,
            ErrorCode::kResourceExhausted);

  const NotificationSender sender = test::RuntimeNotificationProbeState::sender_copy();
  auto first_post = sender.post();
  const auto start_deadline = std::chrono::steady_clock::now() + 5s;
  while (!first_post && first_post.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < start_deadline) {
    std::this_thread::yield();
    first_post = sender.post();
  }
  ASSERT_TRUE(first_post);
  ASSERT_TRUE(test::RuntimeNotificationProbeState::wait_until_compute_enters());

  // Key assertion: capacity=1 remains charged while the first callback is still active.
  const auto backpressured = sender.post();
  ASSERT_FALSE(backpressured);
  EXPECT_EQ(backpressured.error().code, ErrorCode::kBackpressured);

  test::RuntimeNotificationProbeState::release_blocked_compute();
  ASSERT_TRUE(test::RuntimeNotificationProbeState::wait_until_compute_count(1U));

  auto second_post = sender.post();
  const auto retirement_deadline = std::chrono::steady_clock::now() + 5s;
  while (!second_post && second_post.error().code == ErrorCode::kBackpressured &&
         std::chrono::steady_clock::now() < retirement_deadline) {
    std::this_thread::yield();
    second_post = sender.post();
  }
  ASSERT_TRUE(second_post);
  ASSERT_TRUE(test::RuntimeNotificationProbeState::wait_until_compute_count(2U));

  session.request_stop();
  session.wait();

  // Key assertion: the escaped sender retains only a closed tombstone after session teardown.
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const auto after_stop = sender.post();
  ASSERT_FALSE(after_stop);
  EXPECT_EQ(after_stop.error().code, ErrorCode::kClosed);
}

// REQ: RUNTIME-001, RUNTIME-007, RUNTIME-012
TEST(RunSessionTest, MaterializesMixedPhysicalClockAndClockPredicateSourcesTogether) {
  test::RuntimeProbeState::reset();
  test::RuntimeClockProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();
  const ExecutionPlan plan =
      test::enable_direct_runtime(test::make_runtime_mixed_source_test_plan());
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.readiness().size(), 1U);
  ASSERT_EQ(plan.temporal_triggers().size(), 3U);

  test::RuntimeProbeState::reset();
  test::RuntimeClockProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();
  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());
  ASSERT_TRUE(test::RuntimeClockProbeState::wait_until_compute_count(1U));
  ASSERT_TRUE(test::RuntimeClockPredicateProbeState::wait_until_compute_count(1U));

  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}

void expect_clock_predicate_snapshot(
    const test::RuntimeClockPredicateEvaluationEvidence& evaluation,
    const test::RuntimeClockPredicateComputeEvidence& compute) {
  EXPECT_EQ(evaluation.selection_count, 0U);
  EXPECT_EQ(compute.tag, evaluation.tag);
  EXPECT_EQ(compute.tag_time, evaluation.tag_time);
  EXPECT_EQ(compute.activation_time, evaluation.reference_time);
  EXPECT_EQ(evaluation.reference_time.timestamp_ns, evaluation.tag_time.logical_ns);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-012
TEST(RunSessionTest, ClockPredicateCommitsFalseAndRetiresNoFireBeforeNextTrue) {
  test::RuntimeProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_clock_predicate_plan(
      test::RuntimeClockPredicatePattern::kTrueFalseTrue, PredicateFiring::kOnFalseToTrue);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  EXPECT_EQ(plan.predicate_triggers().front().firing, PredicateFiring::kOnFalseToTrue);
  EXPECT_TRUE(plan.predicate_inputs().empty());
  EXPECT_TRUE(plan.input_selections().empty());
  test::RuntimeProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeClockPredicateProbeState::wait_until_compute_count(2U));
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  ASSERT_EQ(test::RuntimeClockPredicateProbeState::evaluation_count.load(std::memory_order_acquire),
            3U);
  ASSERT_EQ(test::RuntimeClockPredicateProbeState::compute_count.load(std::memory_order_acquire),
            2U);
  const auto& evaluations = test::RuntimeClockPredicateProbeState::evaluations;
  EXPECT_EQ(evaluations[0].state_count, 1U);
  EXPECT_TRUE(evaluations[0].result);
  EXPECT_EQ(evaluations[1].state_count, 2U);
  EXPECT_FALSE(evaluations[1].result);
  EXPECT_EQ(evaluations[2].state_count, 3U);
  EXPECT_TRUE(evaluations[2].result);
  EXPECT_EQ(evaluations[0].tag.sequence + 1U, evaluations[1].tag.sequence);
  EXPECT_EQ(evaluations[1].tag.sequence + 1U, evaluations[2].tag.sequence);
  EXPECT_EQ(evaluations[0].tag.generation, evaluations[2].tag.generation);
  expect_clock_predicate_snapshot(evaluations[0],
                                  test::RuntimeClockPredicateProbeState::computes[0]);
  expect_clock_predicate_snapshot(evaluations[2],
                                  test::RuntimeClockPredicateProbeState::computes[1]);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-012
TEST(RunSessionTest, ClockPredicateEveryTrueEvaluationFiresWithExactSnapshot) {
  test::RuntimeProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_clock_predicate_plan(
      test::RuntimeClockPredicatePattern::kAlwaysTrue, PredicateFiring::kOnEveryTrueEvaluation);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  EXPECT_EQ(plan.predicate_triggers().front().firing, PredicateFiring::kOnEveryTrueEvaluation);
  EXPECT_TRUE(plan.predicate_inputs().empty());
  EXPECT_TRUE(plan.input_selections().empty());
  test::RuntimeProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeClockPredicateProbeState::wait_until_compute_count(2U));
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  ASSERT_EQ(test::RuntimeClockPredicateProbeState::evaluation_count.load(std::memory_order_acquire),
            2U);
  ASSERT_EQ(test::RuntimeClockPredicateProbeState::compute_count.load(std::memory_order_acquire),
            2U);
  const auto& evaluations = test::RuntimeClockPredicateProbeState::evaluations;
  EXPECT_EQ(evaluations[0].state_count, 1U);
  EXPECT_EQ(evaluations[1].state_count, 2U);
  EXPECT_TRUE(evaluations[0].result);
  EXPECT_TRUE(evaluations[1].result);
  expect_clock_predicate_snapshot(evaluations[0],
                                  test::RuntimeClockPredicateProbeState::computes[0]);
  expect_clock_predicate_snapshot(evaluations[1],
                                  test::RuntimeClockPredicateProbeState::computes[1]);
}

// REQ: RUNTIME-008
// REQ: RUNTIME-012
TEST(RunSessionTest, ClockPredicateTruthAndStateResetAfterPublishedSelfRestart) {
  test::RuntimeProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_clock_predicate_plan(
      test::RuntimeClockPredicatePattern::kAlwaysTrue, PredicateFiring::kOnFalseToTrue, true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  ASSERT_EQ(plan.lifecycle_restart_targets().size(), 1U);
  EXPECT_TRUE(plan.predicate_inputs().empty());
  EXPECT_TRUE(plan.input_selections().empty());
  test::RuntimeProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeClockPredicateProbeState::wait_until_compute_count(2U));
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  ASSERT_EQ(test::RuntimeClockPredicateProbeState::evaluation_count.load(std::memory_order_acquire),
            2U);
  ASSERT_EQ(test::RuntimeClockPredicateProbeState::compute_count.load(std::memory_order_acquire),
            2U);
  EXPECT_TRUE(
      test::RuntimeClockPredicateProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeClockPredicateProbeState::restart_error.load(std::memory_order_relaxed),
            ErrorCode::kSuccess);
  EXPECT_NE(
      test::RuntimeClockPredicateProbeState::restart_receipt_id.load(std::memory_order_relaxed),
      0U);
  EXPECT_EQ(test::RuntimeClockPredicateProbeState::restart_requested_stage.load(
                std::memory_order_relaxed),
            LifecycleStage::kConfigure);
  const auto& evaluations = test::RuntimeClockPredicateProbeState::evaluations;
  EXPECT_EQ(evaluations[0].state_count, 1U);
  EXPECT_EQ(evaluations[1].state_count, 1U);
  EXPECT_TRUE(evaluations[0].result);
  EXPECT_TRUE(evaluations[1].result);
  EXPECT_LT(evaluations[0].tag.generation, evaluations[1].tag.generation);
  expect_clock_predicate_snapshot(evaluations[0],
                                  test::RuntimeClockPredicateProbeState::computes[0]);
  expect_clock_predicate_snapshot(evaluations[1],
                                  test::RuntimeClockPredicateProbeState::computes[1]);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-012
// REQ: RUNTIME-028
TEST(RunSessionTest, PredicateFailureRetainsExactPrivateCandidateContext) {
  test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      test::make_runtime_clock_predicate_error_plan(), sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  ASSERT_EQ(plan.deadlines().size(), 1U);
  ASSERT_TRUE(plan.input_selections().empty()) << plan.json();
  ASSERT_EQ(plan.clock_triggers().size(), 1U) << plan.json();
  ASSERT_EQ(plan.clock_triggers().front().phase_origin,
            ClockPhaseOrigin::kCapturedAtSuccessfulStart)
      << plan.json();
  ASSERT_FALSE(plan.clock_triggers().front().explicit_phase) << plan.json();
  const PredicateTriggerPlanRecord trigger = plan.predicate_triggers().front();
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
  EXPECT_NE(session.diagnostics().front().token, "DEADLINE_MISS");
  EXPECT_EQ(session.diagnostics().front().service, "predicate_evaluator");
  EXPECT_TRUE(static_cast<bool>(session.diagnostics().front().source.vertex));
  EXPECT_TRUE(session.diagnostics().front().endpoint.empty());
  EXPECT_TRUE(session.diagnostics().front().json_pointer.empty());

  const auto published = sink->snapshot_details();
  ASSERT_EQ(published.size(), 1U);
  ASSERT_EQ(published.front().size(), 1U);
  const detail::RuntimeDiagnosticDetail& detail = published.front().front();
  EXPECT_TRUE(detail.present);
  EXPECT_EQ(detail.error.code, ErrorCode::kProtocolError);
  EXPECT_EQ(detail.error.message, "runtime predicate failure detail");
  EXPECT_EQ(detail.predicate_id, trigger.id);
  EXPECT_EQ(detail.operator_id, trigger.owner.id);
  EXPECT_TRUE(static_cast<bool>(detail.source));
  EXPECT_EQ(detail.source_sequence, 1U);
  EXPECT_EQ(detail.tag, (Tag{1U, 1U, 0U}));
  EXPECT_EQ(detail.tag_time, (TagTime{0, 0U, 0U}));
  EXPECT_TRUE(detail.reference_time.valid());
  EXPECT_EQ(detail.reference_time.timestamp_ns, detail.tag_time.logical_ns);
  EXPECT_EQ(detail.state_generation, detail.tag.generation);
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: RUNTIME-002
// REQ: RUNTIME-004
TEST(RunSessionTest, ClockReadFailureAfterStartStillRunsReverseCleanup) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kThrowClockRead);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_CLOCK_READ_FAILED"));
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 13}));
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
}
#endif

// REQ: RUNTIME-002
// REQ: RUNTIME-004
TEST(RunSessionTest, RejectsUninjectedReadinessBeforeLifecycleCallbacks) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_test_plan();
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_READINESS_BINDING_UNAVAILABLE"));
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeProbeState::active.load(std::memory_order_acquire), 0U);
}

// REQ: RUNTIME-010
TEST(RunSessionTest, ReportsMaterializationFailureBeforeDeadlineCallback) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan baseline = test::enable_direct_runtime(test::make_runtime_test_plan());
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::mutate_materialization_strategy(
      baseline, "holoscan.runtime.unknown/v1");
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "PROFILE_MATERIALIZATION_FAILED"));
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
}

// REQ: RUNTIME-010
TEST(RunSessionTest, RejectsEveryPackageMismatchBeforeOperatorReconstruction) {
  using ProfileMutation = detail::RuntimeTestPlanAccess::ProfileMutation;
  struct Case {
    ProfileMutation mutation;
    std::string_view token;
  };
  constexpr std::array cases{
      Case{ProfileMutation::kPlanSchema, "PROFILE_PLAN_SCHEMA_UNSUPPORTED"},
      Case{ProfileMutation::kProfilePolicy, "PROFILE_POLICY_UNSUPPORTED"},
      Case{ProfileMutation::kRuntimeAbi, "PROFILE_RUNTIME_ABI_INCOMPATIBLE"},
      Case{ProfileMutation::kTarget, "PROFILE_TARGET_INCOMPATIBLE"},
      Case{ProfileMutation::kCapability, "PROFILE_CAPABILITY_UNSATISFIED"},
      Case{ProfileMutation::kEvidence, "PROFILE_EVIDENCE_UNSATISFIED"},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.token);
    test::RuntimeProbeState::reset();
    const ExecutionPlan baseline = test::enable_direct_runtime(test::make_runtime_test_plan());
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::mutate_profile(baseline, test_case.mutation);
    ASSERT_TRUE(plan.ok());
    test::RuntimeProbeState::reset();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, test_case.token));
    EXPECT_EQ(test::RuntimeProbeState::constructed.load(std::memory_order_acquire), 0U);
    EXPECT_FALSE(test::RuntimeProbeState::setup_entered.load(std::memory_order_acquire));
    EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  }
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-008
TEST(RunSessionTest, RejectsMalformedLifecyclePlanBeforeCallbacks) {
  using LifecycleMutation = detail::RuntimeTestPlanAccess::LifecycleMutation;
  constexpr std::array mutations{
      LifecycleMutation::kCheckpointMissing,
      LifecycleMutation::kCheckpointRequired,
      LifecycleMutation::kCheckpointLevel,
      LifecycleMutation::kCheckpointOrder,
      LifecycleMutation::kDependency,
      LifecycleMutation::kRestartPermission,
      LifecycleMutation::kMutableField,
      LifecycleMutation::kTransactionCapacity,
      LifecycleMutation::kRestartTargetMissing,
      LifecycleMutation::kRestartTargetIdentity,
      LifecycleMutation::kRestartTargetAuthority,
      LifecycleMutation::kRestartClosureMissing,
  };

  for (const LifecycleMutation mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    test::RuntimeProbeState::reset();
    const ExecutionPlan baseline = test::enable_direct_runtime(test::make_runtime_test_plan());
    const ExecutionPlan plan = detail::RuntimeTestPlanAccess::mutate_lifecycle(baseline, mutation);
    test::RuntimeProbeState::reset();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_LIFECYCLE_PLAN_INVALID"));
    EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
    EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
  }
}

// REQ: RUNTIME-001
// REQ: RUNTIME-002
// REQ: RUNTIME-003
TEST(RunSessionTest, RetainsPlanStateAfterPlanWrapperIsDestroyed) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  RunSession session = [] {
    const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
    test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
    return run_async(plan);
  }();
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());

  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
}

// REQ: RUNTIME-003
TEST(RunSessionTest, MoveLeavesSourceInDefinedInertState) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);

  RunSession source = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());
  const OperatorRef plan_ref = plan.lifecycle_checkpoints().front().operator_instance;
  RunSession destination{std::move(source)};

  // Exercising the documented inert moved-from state is the purpose of this test.
  expect_inert_moved_from_session(
      source, plan_ref);  // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)

  destination.request_stop();
  destination.wait();
  EXPECT_EQ(destination.termination(), RunTermination::kStopped);
}

// REQ: RUNTIME-003
TEST(RunSessionTest, ConcurrentStopRequestsAndWaitersShareOneCompletion) {
  // Key setup: 8 threads call session.wait() and 8 threads call request_stop()
  // simultaneously; the wait interleaving control ensures peer waiters are
  // registered before the join owner is released.
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  auto wait_control = std::make_shared<detail::RuntimeWaitTestControl>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_wait_interleaving(
      test::enable_direct_runtime(test::make_runtime_test_plan()), wait_control);
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());

  std::vector<std::thread> waiters;
  waiters.reserve(8U);
  waiters.emplace_back([&] { session.wait(); });
  EXPECT_TRUE(wait_control->wait_for_join_owner());
  for (int index = 1; index < 8; ++index) {
    waiters.emplace_back([&] { session.wait(); });
  }
  EXPECT_TRUE(wait_control->wait_for_peer_waiters(7U));

  std::vector<std::thread> stoppers;
  stoppers.reserve(8U);
  for (int index = 0; index < 8; ++index) {
    stoppers.emplace_back([&] { session.request_stop(); });
  }
  for (auto& stopper : stoppers) {
    stopper.join();
  }
  wait_control->release_join_owner();
  for (auto& waiter : waiters) {
    waiter.join();
  }

  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_EQ(test::RuntimeProbeState::max_active.load(std::memory_order_relaxed), 1U);
  // Key assertion: join_owner_count==1 proves exactly one thread performed the
  // underlying thread join regardless of how many concurrent waiters arrived.
  EXPECT_EQ(wait_control->join_owner_count(), 1U);
}

// REQ: RUNTIME-003
// REQ: RUNTIME-004
TEST(RunSessionTest, DestructorRequestsStopAndJoinsWorker) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);

  {
    const RunSession session = run_async(plan);
    ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());
  }

  EXPECT_EQ(test::RuntimeProbeState::active.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
}

// REQ: RUNTIME-004
TEST(RunSessionTest, ComputeFailureIsTerminalAndNotRetried) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kFail);
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kFail);
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_OPERATOR_CALLBACK_FAILED"));
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
  EXPECT_NE(run(plan), 0);
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: RUNTIME-004
TEST(RunSessionTest, SetupExceptionFailsStartupBeforeLifecycleCallbacks) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kThrowSetup);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_OPERATOR_SETUP_FAILED"));
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
}

// REQ: RUNTIME-003
// REQ: RUNTIME-004
TEST(RunSessionTest, StartExceptionStopsOnlyPreviouslyStartedOperators) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan(true));
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kThrowStart);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_LIFECYCLE_CALLBACK_FAILED"));
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 21, 13}));
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
}

// REQ: RUNTIME-004
TEST(RunSessionTest, ComputeExceptionBecomesTerminalCallbackFailure) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kThrowCompute);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_OPERATOR_CALLBACK_FAILED"));
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
  EXPECT_EQ(test::RuntimeProbeState::active.load(std::memory_order_acquire), 0U);
}

// REQ: RUNTIME-003
// REQ: RUNTIME-004
TEST(RunSessionTest, StopExceptionCompletesCleanupAndMarksRunFailed) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kThrowStop);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_LIFECYCLE_CALLBACK_FAILED"));
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
}
#endif

// REQ: RUNTIME-004
TEST(RuntimeLifecycleExecutorTest, StartupPreparationFailureStopsOnlyPreviouslyStartedOperators) {
  // Key setup: pin count for z-second is set to near-overflow before lifecycle
  // startup, forcing a parameter-preparation failure after a-first has started.
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  const ExecutionPlan compiled = make_two_legacy_parameter_startup_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(compiled);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);

  constexpr RunGenerationId kRunGeneration = 17U;
  auto reconstructed = detail::GraphExec::reconstruct(accepted, kRunGeneration);
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  auto exec = std::move(reconstructed).value();
  ASSERT_EQ(exec->operators().size(), 2U);
  ASSERT_EQ(exec->operators()[0].operator_id, 1U);
  ASSERT_EQ(exec->operators()[1].operator_id, 2U);

  auto created_lanes =
      detail::ExecutionLaneSet::create(accepted->scheduler_plan, accepted->execution_lanes);
  ASSERT_TRUE(created_lanes) << created_lanes.error().message;
  auto lanes = std::move(created_lanes).value();
  ASSERT_TRUE(lanes->start());
  const std::atomic_bool cancellation{};
  ASSERT_TRUE(exec->initialize_lifecycle(kRunGeneration, &cancellation, lanes.get()));

  const OperatorRef failing_owner{
      .id = 2U,
      .plan_generation = accepted->plan_generation,
      .run_generation = kRunGeneration,
  };
  detail::ParameterUpdateCoordinatorTestAccess::set_published_pin_count(
      exec->parameter_updates(), failing_owner, std::numeric_limits<std::uint64_t>::max() >> 1U);
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();

  // Key assertion: lifecycle_startup() fails and only the stop callbacks for
  // a-first (901, 903) appear in the snapshot — z-second was never started.
  EXPECT_FALSE(exec->lifecycle_startup());
  EXPECT_EQ(exec->lifecycle_failure_token(), "RUNTIME_LIFECYCLE_CALLBACK_FAILED");
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{901, 903}));
  EXPECT_TRUE(test::RuntimeParamProbeState::stop_observed_params.load(std::memory_order_acquire));

  detail::ParameterUpdateCoordinatorTestAccess::set_published_pin_count(
      exec->parameter_updates(), failing_owner, 0U);

  lanes->request_stop();
  EXPECT_TRUE(lanes->join());
}

// REQ: RUNTIME-004
// REQ: RUNTIME-008
TEST(RuntimeLifecycleExecutorTest, AcceptsCanonicalTopologicalCheckpointOrder) {
  const ExecutionPlan compiled = make_reverse_topological_lifecycle_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  std::vector<OperatorInstanceId> configure_order;
  for (const auto& checkpoint : compiled.lifecycle_checkpoints()) {
    if (checkpoint.stage == LifecycleStage::kConfigure) {
      configure_order.push_back(checkpoint.operator_instance.id);
    }
  }
  ASSERT_EQ(configure_order, (std::vector<OperatorInstanceId>{2U, 1U}));

  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(compiled);
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);
  auto reconstructed = detail::GraphExec::reconstruct(accepted, 17U);
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
}

// REQ: RUNTIME-005
TEST(RunSessionTest, OperatorRefBindsOnlyMatchingPlanIdentity) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  const OperatorRef plan_ref = plan.lifecycle_checkpoints().front().operator_instance;
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());

  const auto bound = session.operator_ref(plan_ref);
  ASSERT_TRUE(bound);
  EXPECT_TRUE(bound->run_bound());
  EXPECT_EQ(bound->id, plan_ref.id);
  EXPECT_EQ(bound->plan_generation, plan_ref.plan_generation);
  EXPECT_NE(bound->run_generation, 0U);

  OperatorRef already_bound = plan_ref;
  already_bound.run_generation = 1U;
  EXPECT_EQ(session.operator_ref(already_bound).error().code, ErrorCode::kStaleOperatorRef);
  OperatorRef colliding_foreign = plan_ref;
  ++colliding_foreign.plan_generation;
  ASSERT_EQ(colliding_foreign.id, plan_ref.id);
  EXPECT_EQ(session.operator_ref(colliding_foreign).error().code, ErrorCode::kStaleOperatorRef);
  EXPECT_EQ(session.operator_ref(OperatorRef{}).error().code, ErrorCode::kStaleOperatorRef);
  OperatorRef unresolved = plan_ref;
  unresolved.id = std::numeric_limits<OperatorInstanceId>::max();
  EXPECT_EQ(session.operator_ref(unresolved).error().code, ErrorCode::kStaleOperatorRef);

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.operator_ref(plan_ref).error().code, ErrorCode::kStaleOperatorRef);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-005
TEST(RunSessionTest, ConcurrentSessionsReceiveUniqueRunGenerations) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  const OperatorRef plan_ref = plan.lifecycle_checkpoints().front().operator_instance;
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);

  RunSession first = run_async(plan);
  RunSession second = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_active(2U));

  const auto first_ref = first.operator_ref(plan_ref);
  const auto second_ref = second.operator_ref(plan_ref);
  ASSERT_TRUE(first_ref);
  ASSERT_TRUE(second_ref);
  EXPECT_NE(first_ref->run_generation, 0U);
  EXPECT_NE(second_ref->run_generation, 0U);
  EXPECT_NE(first_ref->run_generation, second_ref->run_generation);

  first.request_stop();
  second.request_stop();
  first.wait();
  second.wait();
}

}  // namespace
}  // namespace holoscan
