/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/time/synthetic_clock.hpp>

#include "compiler/execution_plan_state.hpp"
#include "compiler/process_local_channel_plan_validation.hpp"
#include "compiler/process_placement_internal.hpp"
#include "run_session_test_support.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/parameter_update_coordinator.hpp"
#include "runtime/physical_readiness_binding.hpp"
#include "runtime/runtime_callback_context.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/source_wait.hpp"
#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"
#include "support/optional_assertions.hpp"

namespace holoscan {

namespace {

using namespace std::chrono_literals;

using namespace test::run_session;

// EOS must make progress even when no later data publication triggers a cut. A watchdog is only
// cleanup: stopped termination fails the test. Compare the eventless family with an eventful
// control at identical partitioning, including final data already consumed before the last call.
// REQ: RUNTIME-050
// REQ: CHAN-136
TEST(RunSessionTest, SamePartitionFanoutCompletesWithoutFinalPublication) {
  for (const bool force_event_wake : {false, true}) {
    for (const std::uint64_t limit : {0U, 1U, 2U}) {
      SCOPED_TRACE(::testing::Message() << "event_wake=" << force_event_wake << " limit=" << limit);
      test::RuntimeComputeCallChannelProbeState::reset();
      // The fanout control keeps this blocking family on HoloMQ, whose cut-driven terminal
      // progress is under test; compile() would deliver it directly (CHAN-167).
      const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
          limit,
          test::RuntimeComputeCallAction::kSkipFinalPublication,
          1U,
          false,
          force_event_wake,
          detail::ProcessLocalFanoutExperiment::kDisabled,
          false,
          QueuePolicy::kBlocking,
          false,
          true,
          1U);
      ASSERT_TRUE(plan.ok()) << plan.json();
      const auto accepted = detail::ExecutionPlanAccess::retain(plan);
      ASSERT_TRUE(accepted);
      ASSERT_EQ(accepted->process_local_channel_subscribers.size(), 2U);
      ASSERT_EQ(accepted->process_local_channel_services.size(), 1U);
      EXPECT_EQ(accepted->process_local_channel_services.front().delivery_kind,
                detail::ProcessLocalChannelDeliveryKind::kHoloMQ);
      for (const auto& subscriber : accepted->process_local_channel_subscribers) {
        EXPECT_EQ(subscriber.event_wake, force_event_wake);
      }

      RunSession session = run_async(plan);
      const auto deadline = std::chrono::steady_clock::now() + 2s;
      while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
      }
      if (session.is_running()) {
        session.request_stop();
      }
      session.wait();

      EXPECT_EQ(session.termination(), RunTermination::kCompleted);
      EXPECT_TRUE(session.diagnostics().empty());
      EXPECT_EQ(
          test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
          limit);
      EXPECT_EQ(
          test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
          limit == 0U ? 0U : 2U * (limit - 1U));
      expect_terminal_compute_call_progress(session, plan, limit, limit, true);
    }
  }
}

// A drop-oldest fanout family with a recipient in another partition completes naturally and
// restarts by direct retention as it does on HoloMQ: each recipient receives the final
// publication and its end of stream. A recipient may consume the first generation's sample before
// the restart discards it, under either delivery, so the restart arm bounds the deliveries.
// REQ: CHAN-160
// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(RunSessionTest, DirectDropOldestFanoutCompletesAndRestartsLikeHoloMQ) {
  for (const bool force_holomq_fanout : {false, true}) {
    for (const auto action : {test::RuntimeComputeCallAction::kNone,
                              test::RuntimeComputeCallAction::kRestartAfterFirst}) {
      const bool restart = action == test::RuntimeComputeCallAction::kRestartAfterFirst;
      SCOPED_TRACE(::testing::Message()
                   << "holomq=" << force_holomq_fanout << " restart=" << restart);
      test::RuntimeComputeCallChannelProbeState::reset();
      const std::uint64_t limit = restart ? 2U : 1U;
      const ExecutionPlan plan =
          test::make_runtime_drop_oldest_fanout_plan(limit, action, force_holomq_fanout);
      ASSERT_TRUE(plan.ok()) << plan.json();
      const auto accepted = detail::ExecutionPlanAccess::retain(plan);
      ASSERT_TRUE(accepted);
      ASSERT_EQ(accepted->process_local_channel_services.size(), 1U);
      EXPECT_EQ(accepted->process_local_channel_services.front().delivery_kind,
                force_holomq_fanout ? detail::ProcessLocalChannelDeliveryKind::kHoloMQ
                                    : detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
      EXPECT_EQ(accepted->process_local_channel_services.front().max_subscribers, 2U);

      RunSession session = run_async(plan);
      const auto deadline = std::chrono::steady_clock::now() + 5s;
      while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
      }
      if (session.is_running()) {
        session.request_stop();
      }
      session.wait();

      // Key assertion: the session completes without a stop, so each recipient received the
      // final publication and end of stream. Each recipient takes at most one sample per
      // generation, and without a restart both take the single publication.
      EXPECT_EQ(session.termination(), RunTermination::kCompleted);
      EXPECT_TRUE(session.diagnostics().empty());
      EXPECT_EQ(
          test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
          limit);
      const std::size_t deliveries =
          test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire);
      EXPECT_GE(deliveries, 2U);
      EXPECT_LE(deliveries, 2U * limit);
      if (!restart) {
        EXPECT_EQ(
            test::RuntimeComputeCallChannelProbeState::first_value.load(std::memory_order_relaxed),
            77U);
        EXPECT_EQ(
            test::RuntimeComputeCallChannelProbeState::second_value.load(std::memory_order_relaxed),
            77U);
      }
      EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::restart_accepted.load(
                    std::memory_order_acquire),
                restart);
    }
  }
}

// A blocking or mixed-policy fanout family with recipients in other partitions is delivered by
// direct retention and, like the HoloMQ control, hands every blocking recipient every sample before
// completing naturally. A drop-oldest recipient may skip samples under either delivery.
// REQ: CHAN-167
TEST(RunSessionTest, DirectBlockingFanoutDeliversEverySampleLikeHoloMQ) {
  constexpr std::uint64_t kLimit = 32U;
  for (const QueuePolicy second_policy : {QueuePolicy::kBlocking, QueuePolicy::kDropOldest}) {
    for (const bool force_holomq_fanout : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "mixed=" << (second_policy == QueuePolicy::kDropOldest)
                                        << " holomq=" << force_holomq_fanout);
      test::RuntimeComputeCallChannelProbeState::reset();
      const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
          kLimit,
          test::RuntimeComputeCallAction::kNone,
          1U,
          true,
          false,
          detail::ProcessLocalFanoutExperiment::kDisabled,
          false,
          second_policy,
          false,
          force_holomq_fanout,
          1U);
      ASSERT_TRUE(plan.ok()) << plan.json();
      const auto accepted = detail::ExecutionPlanAccess::retain(plan);
      ASSERT_TRUE(accepted);
      const auto expected_kind = force_holomq_fanout
                                     ? detail::ProcessLocalChannelDeliveryKind::kHoloMQ
                                     : detail::ProcessLocalChannelDeliveryKind::kDirectRetained;
      EXPECT_TRUE(std::ranges::all_of(
          accepted->process_local_channel_services,
          [&](const auto& service) { return service.delivery_kind == expected_kind; }));

      RunSession session = run_async(plan);
      const auto deadline = std::chrono::steady_clock::now() + 10s;
      while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
      }
      if (session.is_running()) {
        session.request_stop();
      }
      session.wait();

      // Key assertion: the run completes, and the blocking recipient received every sample.
      EXPECT_EQ(session.termination(), RunTermination::kCompleted);
      EXPECT_TRUE(session.diagnostics().empty())
          << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
      EXPECT_EQ(
          test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
          kLimit);
      const std::size_t deliveries =
          test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire);
      const std::size_t first_deliveries =
          test::RuntimeComputeCallChannelProbeState::first_sink_count.load(
              std::memory_order_acquire);
      // Key assertion: the first recipient, blocking in both arms, received every sample.
      EXPECT_EQ(first_deliveries, kLimit);
      if (second_policy == QueuePolicy::kBlocking) {
        EXPECT_EQ(deliveries - first_deliveries, kLimit);
        EXPECT_EQ(deliveries, 2U * kLimit);
      } else {
        // The drop-oldest recipient may skip samples.
        EXPECT_GE(deliveries - first_deliveries, 1U);
        EXPECT_LE(deliveries - first_deliveries, kLimit);
      }
    }
  }
}

// A recipient stalled well beyond the graph blocking budget turns the producer's emit into the
// ordinary pre-submit kBackpressured: the capacity check before any store rejects the publication
// for every recipient, so the run never sees a partial fanout.
// REQ: CHAN-167
TEST(RunSessionTest, DirectBlockingFanoutStallRejectsBeforeAnyStore) {
  test::RuntimeComputeCallChannelProbeState::reset();
  test::RuntimeComputeCallChannelProbeState::sink_stall_us.store(30'000U);
  ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(4U);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);
  ASSERT_EQ(accepted->process_local_channel_services.size(), 1U);
  EXPECT_EQ(accepted->process_local_channel_services.front().delivery_kind,
            detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);

  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  if (session.is_running()) {
    session.request_stop();
  }
  session.wait();
  test::RuntimeComputeCallChannelProbeState::sink_stall_us.store(0U);

  // Key assertion: the failure is pre-submit backpressure, not a partial publication.
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  const Diagnostic& diagnostic = session.diagnostics().front();
  EXPECT_EQ(diagnostic.token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
  const auto& detail = diagnostic.detail;
  HOLOSCAN_ASSERT_HAS_VALUE(detail);
  EXPECT_EQ(detail->error.code, ErrorCode::kBackpressured);
  EXPECT_GT(counters->snapshot()[detail::RuntimePhaseCounters::index(
                detail::RuntimePhaseCounter::kChannelDirectPublicationCapacityWaitsExpired)],
            0U);
}

struct DirectChainRun {
  bool timed_out{};
  RunTermination termination{};
  std::vector<Diagnostic> diagnostics;
  detail::RuntimePhaseCounters::Snapshot counters{};
};

// Run the source -> relay -> sink direct chain, each stage in its own partition, with a sink
// that sleeps for sink_stall_us after each sample while the source ticks every 1 ms, or that holds
// each sample until the relay has begun its next publication.
DirectChainRun run_slow_direct_chain(const std::uint64_t limit, const std::uint32_t sink_stall_us,
                                     const bool hold_until_relay_progress = false) {
  test::RuntimeComputeCallChannelProbeState::reset();
  test::RuntimeComputeCallChannelProbeState::sink_stall_us.store(sink_stall_us);
  test::RuntimeComputeCallChannelProbeState::sink_holds_until_relay_progress.store(
      hold_until_relay_progress);
  ExecutionPlan plan = test::make_runtime_compute_call_chain_plan(limit);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  DirectChainRun result;
  result.timed_out = session.is_running();
  if (result.timed_out) {
    session.request_stop();
  }
  session.wait();
  test::RuntimeComputeCallChannelProbeState::sink_stall_us.store(0U);
  test::RuntimeComputeCallChannelProbeState::sink_holds_until_relay_progress.store(false);
  result.termination = session.termination();
  result.diagnostics.assign(session.diagnostics().begin(), session.diagnostics().end());
  result.counters = counters->snapshot();
  return result;
}

[[nodiscard]] std::uint64_t counter(const DirectChainRun& run,
                                    const detail::RuntimePhaseCounter counter) {
  return run.counters[detail::RuntimePhaseCounters::index(counter)];
}

// A blocking direct-retained edge has no transport queue. When its consumer runs on another lane
// and falls briefly behind, the producer waits for a typed slot within the same per-recipient
// budget as a transported blocking send, instead of failing its emit at once. Here the sink holds
// each sample until the relay has begun its next publication, so the relay, and through it the
// 1 ms source, repeatedly wait for a slot. A fixed 2 ms sink stall kept the sink behind for the
// whole run, and a few milliseconds of CI scheduling delay then expired the budget.
// REQ: CHAN-128
// REQ: CHAN-136
TEST(RunSessionTest, DirectPublicationWaitsForSlowCrossLaneConsumer) {
  const ExecutionPlan plan = test::make_runtime_compute_call_chain_plan(16U);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);
  ASSERT_EQ(accepted->execution_partitions.size(), 3U);
  EXPECT_EQ(std::ranges::count_if(accepted->process_local_channel_services,
                                  [](const auto& service) {
                                    return service.delivery_kind ==
                                           detail::ProcessLocalChannelDeliveryKind::kDirectRetained;
                                  }),
            2);
  const DirectChainRun run = run_slow_direct_chain(16U, 0U, true);

  // Key assertion: every sample reaches the slow sink and the run completes naturally.
  EXPECT_FALSE(run.timed_out);
  EXPECT_EQ(run.termination, RunTermination::kCompleted);
  EXPECT_TRUE(run.diagnostics.empty())
      << (run.diagnostics.empty() ? "" : run.diagnostics.front().token);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(), 16U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::transform_count.load(), 16U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(), 16U);
  // Key assertion: producers really met a full slot and waited within the budget.
  EXPECT_GT(counter(run, detail::RuntimePhaseCounter::kChannelDirectPublicationCapacityWaits), 0U);
  EXPECT_EQ(
      counter(run, detail::RuntimePhaseCounter::kChannelDirectPublicationCapacityWaitsExpired), 0U);
}

// The wait is bounded: a consumer stalled well beyond the graph blocking budget still turns the
// producer's emit into the ordinary pre-submit kBackpressured, which this relay propagates.
// REQ: CHAN-128
TEST(RunSessionTest, DirectPublicationWaitExpiresAsBackpressure) {
  const DirectChainRun run = run_slow_direct_chain(4U, 30'000U);
  EXPECT_FALSE(run.timed_out);
  EXPECT_EQ(run.termination, RunTermination::kFailed);
  ASSERT_FALSE(run.diagnostics.empty());
  const Diagnostic& diagnostic = run.diagnostics.front();
  EXPECT_EQ(diagnostic.token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
  const auto& detail = diagnostic.detail;
  HOLOSCAN_ASSERT_HAS_VALUE(detail);
  EXPECT_EQ(detail->error.code, ErrorCode::kBackpressured);
  EXPECT_GT(
      counter(run, detail::RuntimePhaseCounter::kChannelDirectPublicationCapacityWaitsExpired), 0U);
}

// REQ: RUNTIME-077
// REQ: CHAN-136
TEST(RunSessionFanout, ExperimentalFanoutRetainsNaturalCompletionWithoutFinalPublication) {
  for (const auto experiment : {detail::ProcessLocalFanoutExperiment::kPostComputeCut,
                                detail::ProcessLocalFanoutExperiment::kEmitRecipients}) {
    for (const std::uint32_t sink_stall_us : {0U, 5000U}) {
      for (const std::uint64_t limit : {0U, 1U, 2U, 64U}) {
        SCOPED_TRACE(::testing::Message()
                     << "mode=" << static_cast<unsigned>(experiment) << " limit=" << limit
                     << " sink_stall_us=" << sink_stall_us);
        test::RuntimeComputeCallChannelProbeState::reset();
        test::RuntimeComputeCallChannelProbeState::sink_stall_us.store(sink_stall_us);
        // Completion coverage retains every publication even when a recipient stalls.
        const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
            limit,
            test::RuntimeComputeCallAction::kSkipFinalPublication,
            1U,
            true,
            false,
            experiment,
            false,
            QueuePolicy::kBlocking,
            false,
            false,
            static_cast<std::uint32_t>(limit + 1U));
        ASSERT_TRUE(plan.ok()) << plan.json();
        const auto accepted = detail::ExecutionPlanAccess::retain(plan);
        ASSERT_TRUE(detail::valid_process_local_channel_plan(*accepted));
        ASSERT_EQ(accepted->execution_partitions.size(), 3U);
        ASSERT_EQ(accepted->process_local_channel_subscribers.size(), 2U);
        for (const auto& subscriber : accepted->process_local_channel_subscribers) {
          EXPECT_FALSE(subscriber.event_wake);
        }
        RunSession session = run_async(plan);
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
          std::this_thread::sleep_for(1ms);
        }
        if (session.is_running()) {
          session.request_stop();
        }
        session.wait();
        EXPECT_EQ(session.termination(), RunTermination::kCompleted);
        EXPECT_TRUE(session.diagnostics().empty());
        EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(), limit);
        EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_compute_count.load(),
                  limit == 0U ? 0U : 2U * (limit - 1U));
        EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(),
                  limit == 0U ? 0U : 2U * (limit - 1U));
      }
    }
  }
}

// Emit-time and post-compute-cut delivery also serve a mixed-policy family: a blocking and a
// drop-oldest consumer, each in its own partition, take one source output through two eventless
// services, with no pump. Natural completion shows end of stream reached both after their data.
// REQ: CHAN-128
// REQ: RUNTIME-050
// REQ: RUNTIME-083
TEST(RunSessionTest, ExperimentalFanoutDeliversAMixedPolicyFamilyWithoutAPump) {
  constexpr std::uint64_t kLimit = 16U;
  for (const auto experiment : {detail::ProcessLocalFanoutExperiment::kPostComputeCut,
                                detail::ProcessLocalFanoutExperiment::kEmitRecipients}) {
    SCOPED_TRACE(::testing::Message() << "mode=" << static_cast<unsigned>(experiment));
    test::RuntimeComputeCallChannelProbeState::reset();
    ExecutionPlan plan =
        test::make_runtime_compute_call_channel_plan(kLimit,
                                                     test::RuntimeComputeCallAction::kNone,
                                                     1U,
                                                     true,
                                                     false,
                                                     experiment,
                                                     false,
                                                     QueuePolicy::kDropOldest);
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto accepted = detail::ExecutionPlanAccess::retain(plan);
    ASSERT_TRUE(accepted);
    ASSERT_TRUE(detail::valid_process_local_channel_plan(*accepted));
    ASSERT_EQ(accepted->process_local_channel_services.size(), 2U);
    bool saw_blocking = false;
    bool saw_drop_oldest = false;
    for (const auto& service : accepted->process_local_channel_services) {
      EXPECT_FALSE(service.event_wake);
      saw_blocking |= service.queue_policy == QueuePolicy::kBlocking;
      saw_drop_oldest |= service.queue_policy == QueuePolicy::kDropOldest;
    }
    EXPECT_TRUE(saw_blocking && saw_drop_oldest);
    RunSession session = run_async(plan);
    if (!session.wait_for(10s)) {
      ADD_FAILURE() << "session did not complete";
      session.request_stop();
    }
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty());
    // Key assertion: the blocking consumer sees every sample, the drop-oldest one at least the
    // newest, and no pump ever waited.
    const std::uint64_t delivered = test::RuntimeComputeCallChannelProbeState::sink_count.load();
    EXPECT_GE(delivered, kLimit + 1U);
    EXPECT_LE(delivered, 2U * kLimit);
    EXPECT_EQ(counters->snapshot()[detail::RuntimePhaseCounters::index(
                  detail::RuntimePhaseCounter::kChannelPumpWaits)],
              0U);
  }
}

// A split family can feed both direct retention and a second eventless family in one partition.
// Zero/final-no-data completion must traverse all three service kinds without a pump.
// REQ: CHAN-136
// REQ: RUNTIME-077
TEST(RunSessionFanout, MixedEventlessFamiliesRetainDataAndTerminalProgress) {
  for (const auto experiment : {detail::ProcessLocalFanoutExperiment::kPostComputeCut,
                                detail::ProcessLocalFanoutExperiment::kEmitRecipients}) {
    for (const std::uint32_t sink_stall_us : {0U, 5000U}) {
      for (const std::uint64_t limit : {0U, 1U, 2U, 64U}) {
        SCOPED_TRACE(::testing::Message()
                     << "mode=" << static_cast<unsigned>(experiment) << " limit=" << limit
                     << " sink_stall_us=" << sink_stall_us);
        test::RuntimeComputeCallChannelProbeState::reset();
        test::RuntimeComputeCallChannelProbeState::sink_stall_us.store(sink_stall_us);
        const ExecutionPlan plan = test::make_runtime_mixed_eventless_fanout_plan(
            limit,
            test::RuntimeComputeCallAction::kSkipFinalPublication,
            experiment,
            static_cast<std::uint32_t>(limit + 1U));
        ASSERT_TRUE(plan.ok()) << plan.json();
        const auto accepted = detail::ExecutionPlanAccess::retain(plan);
        ASSERT_TRUE(accepted);
        ASSERT_TRUE(detail::valid_process_local_channel_plan(*accepted));
        ASSERT_EQ(accepted->process_local_channel_services.size(), 3U);
        EXPECT_EQ(
            std::ranges::count_if(accepted->process_local_channel_services,
                                  [](const auto& service) {
                                    return service.delivery_kind ==
                                           detail::ProcessLocalChannelDeliveryKind::kDirectRetained;
                                  }),
            1);
        EXPECT_TRUE(
            std::ranges::none_of(accepted->process_local_channel_subscribers,
                                 [](const auto& subscriber) { return subscriber.event_wake; }));
        RunSession session = run_async(plan);
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
          std::this_thread::sleep_for(1ms);
        }
        if (session.is_running()) {
          session.request_stop();
        }
        session.wait();
        EXPECT_EQ(session.termination(), RunTermination::kCompleted);
        EXPECT_TRUE(session.diagnostics().empty());
        const std::uint64_t publications = limit == 0U ? 0U : limit - 1U;
        EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(), limit);
        EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::transform_count.load(),
                  2U * publications);
        EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_compute_count.load(),
                  3U * publications);
        EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(), 3U * publications);
      }
    }
  }
}

// The source waits for both consumers while still inside compute. Only an independently
// progressing receive path can satisfy it; no later post-compute cut can rescue this test.
// REQ: CHAN-136
TEST(RunSessionFanout, ExperimentalFanoutAdmitsWhilePublisherCallbackIsStillRunning) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      8U,
      test::RuntimeComputeCallAction::kObserveCallbackProgress,
      1U,
      true,
      false,
      detail::ProcessLocalFanoutExperiment::kEmitRecipients);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  if (session.is_running()) {
    session.request_stop();
  }
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::callback_progress_observed.load());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(), 16U);
}

// Concurrent admission admits emit-time recipients without the channel scan lock, retries
// blocked heads on capacity release, and skips known-blocked heads. Every ordinary completion and
// callback-time progress contract still holds, including mixed eventless families and completion
// without a final publication.
// REQ: CHAN-136
// REQ: RUNTIME-077
TEST(RunSessionFanout, ConcurrentAdmissionRetainsProgressAndCompletion) {
  const auto with_concurrent_admission = [](const ExecutionPlan& plan) {
    auto control = std::make_shared<detail::RuntimeFanoutTestControl>();
    control->concurrent_admission = true;
    return detail::RuntimeTestPlanAccess::with_fanout_test_control(plan, std::move(control));
  };
  const auto run_to_completion = [](const ExecutionPlan& plan) {
    RunSession session = run_async(plan);
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
    }
    const bool timed_out = session.is_running();
    if (timed_out) {
      session.request_stop();
    }
    session.wait();
    EXPECT_FALSE(timed_out);
    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty());
  };
  for (const std::uint32_t sink_stall_us : {0U, 5000U}) {
    for (const std::uint64_t limit : {0U, 1U, 2U, 64U}) {
      SCOPED_TRACE(::testing::Message() << "limit=" << limit << " sink_stall_us=" << sink_stall_us);
      test::RuntimeComputeCallChannelProbeState::reset();
      test::RuntimeComputeCallChannelProbeState::sink_stall_us.store(sink_stall_us);
      const ExecutionPlan plan =
          with_concurrent_admission(test::make_runtime_compute_call_channel_plan(
              limit,
              test::RuntimeComputeCallAction::kSkipFinalPublication,
              1U,
              true,
              false,
              detail::ProcessLocalFanoutExperiment::kEmitRecipients,
              false,
              QueuePolicy::kBlocking,
              false,
              false,
              static_cast<std::uint32_t>(limit + 1U)));
      ASSERT_TRUE(plan.ok()) << plan.json();
      run_to_completion(plan);
      EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(), limit);
      EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_compute_count.load(),
                limit == 0U ? 0U : 2U * (limit - 1U));
      EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(),
                limit == 0U ? 0U : 2U * (limit - 1U));

      test::RuntimeComputeCallChannelProbeState::reset();
      test::RuntimeComputeCallChannelProbeState::sink_stall_us.store(sink_stall_us);
      const ExecutionPlan mixed =
          with_concurrent_admission(test::make_runtime_mixed_eventless_fanout_plan(
              limit,
              test::RuntimeComputeCallAction::kSkipFinalPublication,
              detail::ProcessLocalFanoutExperiment::kEmitRecipients,
              static_cast<std::uint32_t>(limit + 1U)));
      ASSERT_TRUE(mixed.ok()) << mixed.json();
      run_to_completion(mixed);
      const std::uint64_t publications = limit == 0U ? 0U : limit - 1U;
      EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(), limit);
      EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::transform_count.load(),
                2U * publications);
      EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_compute_count.load(),
                3U * publications);
      EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(), 3U * publications);
    }
  }
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan progress =
      with_concurrent_admission(test::make_runtime_compute_call_channel_plan(
          8U,
          test::RuntimeComputeCallAction::kObserveCallbackProgress,
          1U,
          true,
          false,
          detail::ProcessLocalFanoutExperiment::kEmitRecipients));
  ASSERT_TRUE(progress.ok()) << progress.json();
  run_to_completion(progress);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::callback_progress_observed.load());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(), 16U);
}

// Stop shares the aggregate wake mechanism with data hints and must win over natural completion,
// with admission under the channel scan lock or concurrent.
// REQ: RUNTIME-077
TEST(RunSessionFanout, CooperativeStopPreservesPartialCompletion) {
  for (const bool concurrent : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "concurrent=" << concurrent);
    test::RuntimeComputeCallChannelProbeState::reset(true);
    ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
        3U,
        test::RuntimeComputeCallAction::kRequestStopAfterFirst,
        1U,
        true,
        false,
        detail::ProcessLocalFanoutExperiment::kEmitRecipients);
    if (concurrent) {
      auto control = std::make_shared<detail::RuntimeFanoutTestControl>();
      control->concurrent_admission = true;
      plan = detail::RuntimeTestPlanAccess::with_fanout_test_control(plan, std::move(control));
    }
    ASSERT_TRUE(plan.ok()) << plan.json();
    RunSession session = run_async(plan);
    EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
    test::RuntimeComputeCallChannelProbeState::release_source_compute();
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty());
    EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(), 1U);
  }
}

// Channel-driven owners honor explicitly admitted busy-poll placement, including an idle
// recipient waiting for the producer's next callback. No owner may enter the sleeping wait path.
// REQ: RUNTIME-041
TEST(RunSessionFanout, BusyPollChannelOwnersNeverParkAndStillCompleteOrStop) {
  for (const auto action : {test::RuntimeComputeCallAction::kSkipFinalPublication,
                            test::RuntimeComputeCallAction::kRequestStopAfterFirst}) {
    test::RuntimeComputeCallChannelProbeState::reset(true);
    auto plan = test::make_runtime_compute_call_channel_plan(
        8U, action, 1U, true, false, detail::ProcessLocalFanoutExperiment::kEmitRecipients, true);
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
    ASSERT_TRUE(plan.ok()) << plan.json();
    RunSession session = run_async(plan);
    EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
    // Hold the producer inside compute so its empty recipients must actually choose a wait policy.
    std::this_thread::sleep_for(2ms);
    test::RuntimeComputeCallChannelProbeState::release_source_compute();
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
    }
    const bool timed_out = session.is_running();
    if (timed_out)
      session.request_stop();
    session.wait();
    EXPECT_FALSE(timed_out);
    EXPECT_EQ(session.termination(),
              action == test::RuntimeComputeCallAction::kRequestStopAfterFirst
                  ? RunTermination::kStopped
                  : RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty());
    if (action == test::RuntimeComputeCallAction::kSkipFinalPublication) {
      EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(), 14U);
    }
    const auto snapshot = counters->snapshot();
    EXPECT_EQ(
        snapshot[detail::RuntimePhaseCounters::index(detail::RuntimePhaseCounter::kWaitCalls)], 0U);
  }
}

// REQ: CHAN-136
TEST(RunSessionTest, ExperimentalFanoutRejectsUnsupportedAndInconsistentPlans) {
  using Experiment = detail::ProcessLocalFanoutExperiment;
  const auto plan = test::make_runtime_compute_call_channel_plan(
      1U, test::RuntimeComputeCallAction::kNone, 1U, true, false, Experiment::kPostComputeCut);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);
  // Each rejection below must come from its mutation, not from an already invalid plan.
  ASSERT_TRUE(detail::valid_process_local_channel_plan(*accepted));
  auto changed = *accepted;
  changed.fanout_experiment = Experiment::kDisabled;
  EXPECT_FALSE(detail::valid_process_local_channel_plan(changed));
  changed = *accepted;
  changed.executable_factories.test_force_process_local_channel_pump = true;
  EXPECT_FALSE(detail::valid_process_local_channel_plan(changed));
  changed = *accepted;
  changed.process_local_channel_services.front().event_wake = true;
  EXPECT_FALSE(detail::valid_process_local_channel_plan(changed));
  changed = *accepted;
  changed.process_local_channel_subscribers.front().event_wake = true;
  EXPECT_FALSE(detail::valid_process_local_channel_plan(changed));
  changed = *accepted;
  changed.process_local_channel_subscribers.pop_back();
  EXPECT_FALSE(detail::valid_process_local_channel_plan(changed));
  changed = *accepted;
  changed.execution_partitions.back().members.clear();
  EXPECT_FALSE(detail::valid_process_local_channel_plan(changed));
  changed = *accepted;
  changed.vertex_process_placements.back().process_group = ProcessGroupId{2U};
  EXPECT_FALSE(detail::valid_process_local_channel_plan(changed));
  changed = *accepted;
  changed.payload_edges.front().queue_policy = QueuePolicy::kDropOldest;
  EXPECT_FALSE(detail::valid_process_local_channel_plan(changed));
  changed = *accepted;
  // Unknown fanout policies must fail plan validation.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  changed.fanout_experiment = static_cast<Experiment>(255U);
  EXPECT_FALSE(detail::valid_process_local_channel_plan(changed));

  EXPECT_FALSE(
      test::make_runtime_compute_call_channel_plan(
          1U, test::RuntimeComputeCallAction::kNone, 1U, false, false, Experiment::kPostComputeCut)
          .ok());
  EXPECT_FALSE(
      test::make_runtime_compute_call_channel_plan(
          1U, test::RuntimeComputeCallAction::kNone, 1U, true, true, Experiment::kPostComputeCut)
          .ok());
  EXPECT_FALSE(test::make_runtime_compute_call_channel_plan(
                   1U,
                   test::RuntimeComputeCallAction::kRestartAfterFirst,
                   1U,
                   true,
                   false,
                   Experiment::kPostComputeCut)
                   .ok());
}

}  // namespace
}  // namespace holoscan
