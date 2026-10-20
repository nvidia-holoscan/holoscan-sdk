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
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/time/synthetic_clock.hpp>

#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
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
#include "support/host_unique_test_name.hpp"

namespace holoscan {

namespace {

using namespace std::chrono_literals;

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

// A signal or hint to an owner that is not parked never enters the kernel.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, SignalsWithoutAParkedOwnerNeedNoKernelWake) {
  detail::RuntimeWakeState wake;
  EXPECT_FALSE(wake.signal_observed());
  bool kernel_wake = true;
  EXPECT_FALSE(wake.signal_coalesced(&kernel_wake));  // a token is already pending
  EXPECT_TRUE(wake.pending());
  EXPECT_EQ(wake.drain(), 1U);
  EXPECT_FALSE(wake.pending());
  EXPECT_TRUE(wake.signal_coalesced(&kernel_wake));
  EXPECT_FALSE(kernel_wake);
  EXPECT_TRUE(wake.wait_for(0ns));
  EXPECT_FALSE(wake.failed());
}

// A parked owner is woken by one signal, which reports that it had to wake it; the owner takes
// exactly that token.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, ParkedOwnerWakesOnOneSignalThatReportsTheKernelWake) {
  detail::RuntimeWakeState wake;
  std::atomic<bool> woke{false};
  std::atomic<bool> waiting{false};
  std::thread owner([&] {
    waiting.store(true, std::memory_order_release);
    woke.store(wake.wait_until(std::chrono::steady_clock::now() + 10s), std::memory_order_release);
  });
  // Signal only once the owner is waiting. On Linux that means it has published its parked state,
  // so the signal must wake it. The portable fallback cannot report that state; its token is
  // delivered whether or not the owner has blocked yet.
  const auto give_up = std::chrono::steady_clock::now() + 5s;
#if defined(__linux__)
  const auto ready = [&] { return wake.parked(); };
#else
  const auto ready = [&] { return waiting.load(std::memory_order_acquire); };
#endif
  while (!ready() && std::chrono::steady_clock::now() < give_up) {
    std::this_thread::yield();
  }
  EXPECT_TRUE(ready());
  const auto signaled = std::chrono::steady_clock::now();
  const bool kernel_wake = wake.signal_observed();
  owner.join();
#if defined(__linux__)
  // Key assertion (Linux): the signal reports that it woke the parked owner.
  EXPECT_TRUE(kernel_wake);
#else
  static_cast<void>(kernel_wake);
#endif
  EXPECT_TRUE(woke.load(std::memory_order_acquire));
  EXPECT_LT(std::chrono::steady_clock::now() - signaled, 5s);
  EXPECT_EQ(wake.drain(), 0U);
  EXPECT_FALSE(wake.failed());
}

// An absolute deadline is not moved by retries: the wait returns at, never before, the deadline.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, WaitUntilReturnsAtItsAbsoluteDeadline) {
  detail::RuntimeWakeState wake;
  const auto deadline = std::chrono::steady_clock::now() + 3ms;
  EXPECT_FALSE(wake.wait_until(deadline));
  EXPECT_GE(std::chrono::steady_clock::now(), deadline);
  // A deadline already in the past returns at once without consuming anything.
  EXPECT_FALSE(wake.wait_until(std::chrono::steady_clock::now() - 1ms));
  wake.signal();
  EXPECT_TRUE(wake.wait_until(std::chrono::steady_clock::now() - 1ms));
  EXPECT_FALSE(wake.failed());
}

// Per-partition accounting attributes each owner park to exactly one ending, and each progress
// signal to the partition that receives it: a fanout source wakes both separately owned sinks.
// REQ: RUNTIME-041
TEST(RunSessionTest, PartitionAccountingAttributesParksAndSignalsToEachOwner) {
  using Counter = detail::RuntimePartitionCounter;
  constexpr std::uint64_t kLimit = 8U;
  test::RuntimeComputeCallChannelProbeState::reset(true);
  auto plan = test::make_runtime_compute_call_channel_plan(kLimit);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Accounting indices follow the runners' plan ordinals; the sinks' runners have incoming edges.
  std::vector<std::size_t> sink_partitions;
  {
    const auto state = detail::ExecutionPlanAccess::retain(plan);
    ASSERT_NE(state, nullptr);
    std::vector<const detail::PartitionRunnerPlanRecord*> runners;
    for (const auto& runner : state->partition_runners) {
      runners.push_back(&runner);
    }
    std::ranges::sort(runners, {}, &detail::PartitionRunnerPlanRecord::ordinal);
    for (std::size_t index = 0U; index < runners.size(); ++index) {
      if (!runners[index]->incoming_boundary_edges.empty()) {
        sink_partitions.push_back(index);
      }
    }
  }
  ASSERT_EQ(sink_partitions.size(), 2U);
  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
  // Hold the producer until both sink owners have entered a park, so the first delivery reaches
  // parked owners.
  const auto sinks_parked = [&] {
    const auto snapshot = counters->partition_snapshot();
    return std::ranges::all_of(sink_partitions, [&](const std::size_t partition) {
      return snapshot[partition][detail::RuntimePhaseCounters::index(Counter::kWaits)] != 0U;
    });
  };
  const auto give_up = std::chrono::steady_clock::now() + 5s;
  while (!sinks_parked() && std::chrono::steady_clock::now() < give_up) {
    std::this_thread::yield();
  }
  EXPECT_TRUE(sinks_parked());
  test::RuntimeComputeCallChannelProbeState::release_source_compute();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());

  const auto owners = counters->partition_owner_threads();
  const auto snapshot = counters->partition_snapshot();
  const auto value = [&](const std::size_t partition, const Counter counter) {
    return snapshot[partition][detail::RuntimePhaseCounters::index(counter)];
  };
  std::vector<std::uint64_t> threads;
  std::size_t signaled_partitions = 0U;
  for (std::size_t partition = 0U; partition < owners.size(); ++partition) {
    if (owners[partition] == 0U) {
      EXPECT_EQ(value(partition, Counter::kWaits), 0U);
      continue;
    }
    threads.push_back(owners[partition]);
    // Key assertion: every park ended for exactly one recorded reason.
    EXPECT_EQ(value(partition, Counter::kTokenReturns) +
                  value(partition, Counter::kDeadlineReturns) +
                  value(partition, Counter::kCapReturns),
              value(partition, Counter::kWaits));
    EXPECT_LE(value(partition, Counter::kSelfSignals) + value(partition, Counter::kForeignSignals),
              value(partition, Counter::kSignals));
    EXPECT_LE(value(partition, Counter::kSignalsPublished), value(partition, Counter::kSignals));
    EXPECT_GE(value(partition, Counter::kReadinessScans),
              value(partition, Counter::kNoProgressScans));
    signaled_partitions += value(partition, Counter::kSignals) != 0U ? 1U : 0U;
  }
  // Key assertion: three distinct owner threads, and both sink partitions received signals.
  EXPECT_EQ(threads.size(), 3U);
  std::ranges::sort(threads);
  EXPECT_EQ(std::ranges::adjacent_find(threads), threads.end());
  EXPECT_GE(signaled_partitions, 2U);
}

struct ChainAccounting {
  RunTermination termination{};
  bool diagnostics_empty{};
  detail::RuntimePhaseCounters::PartitionSnapshot partitions{};
  std::array<std::uint64_t, detail::kRuntimePartitionAccountingLimit> owners{};
  std::size_t source_partition{detail::kRuntimePartitionAccountingLimit};
};

// Accounting indices follow the runners' plan ordinals, not graph order. The source's runner is
// the one without incoming boundary edges.
[[nodiscard]] std::size_t source_partition_index(const ExecutionPlan& plan) {
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  if (state == nullptr) {
    return detail::kRuntimePartitionAccountingLimit;
  }
  std::vector<const detail::PartitionRunnerPlanRecord*> runners;
  for (const auto& runner : state->partition_runners) {
    runners.push_back(&runner);
  }
  std::ranges::sort(runners, {}, &detail::PartitionRunnerPlanRecord::ordinal);
  const auto source = std::ranges::find_if(
      runners, [](const auto* runner) { return runner->incoming_boundary_edges.empty(); });
  return static_cast<std::size_t>(source - runners.begin());
}

ChainAccounting run_accounted_chain(const std::uint64_t limit, const bool colocate,
                                    const QueuePolicy policy,
                                    const std::chrono::nanoseconds park_cap = {}) {
  test::RuntimeComputeCallChannelProbeState::reset();
  auto plan = test::make_runtime_compute_call_chain_plan(limit, colocate, policy);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  if (park_cap.count() > 0) {
    plan = detail::RuntimeTestPlanAccess::with_owner_park_cap(plan, park_cap);
  }
  EXPECT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  const bool terminal = session.wait_for(10s);
  if (!terminal) {
    session.request_stop();
  }
  session.wait();
  EXPECT_TRUE(terminal);
  return ChainAccounting{.termination = session.termination(),
                         .diagnostics_empty = session.diagnostics().empty(),
                         .partitions = counters->partition_snapshot(),
                         .owners = counters->partition_owner_threads(),
                         .source_partition = source_partition_index(plan)};
}

[[nodiscard]] std::uint64_t partition_value(const ChainAccounting& run, const std::size_t partition,
                                            const detail::RuntimePartitionCounter counter) {
  return run.partitions[partition][detail::RuntimePhaseCounters::index(counter)];
}

// A partition owner that delivers to its own partition sets a rescan flag instead of posting a
// wake: it rescans before it parks, so the delivery needs no token and no system call.
// REQ: RUNTIME-025
TEST(RunSessionTest, DeliveriesWithinTheOwnersPartitionPostNoWake) {
  using Counter = detail::RuntimePartitionCounter;
  constexpr std::uint64_t kLimit = 32U;
  const ChainAccounting run = run_accounted_chain(kLimit, true, QueuePolicy::kBlocking);
  EXPECT_EQ(run.termination, RunTermination::kCompleted);
  EXPECT_TRUE(run.diagnostics_empty);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(), kLimit);
  std::uint64_t self_signals = 0U;
  for (std::size_t partition = 0U; partition < run.owners.size(); ++partition) {
    if (run.owners[partition] == 0U) {
      continue;
    }
    // Key assertion: only signals from other threads publish a wake permit.
    const auto signals = partition_value(run, partition, Counter::kSignals);
    const auto owner_signals = partition_value(run, partition, Counter::kSelfSignals);
    self_signals += owner_signals;
    ASSERT_GE(signals, owner_signals);
    EXPECT_LE(partition_value(run, partition, Counter::kSignalsPublished), signals - owner_signals);
  }
  EXPECT_GT(self_signals, 0U);
}

// Retiring an input wakes the producer's partition only when the producer can be waiting for
// that capacity: always on a blocking edge, and on a drop-oldest edge only once the producer is
// closing its outputs. A source partition, which has no inputs, is then never woken mid-stream.
// REQ: RUNTIME-025
TEST(RunSessionTest, RetirementWakesProducersOnlyWhenTheyCanWaitForCapacity) {
  using Counter = detail::RuntimePartitionCounter;
  constexpr std::uint64_t kLimit = 32U;
  for (const QueuePolicy policy : {QueuePolicy::kBlocking, QueuePolicy::kDropOldest}) {
    SCOPED_TRACE(static_cast<unsigned>(policy));
    const ChainAccounting run = run_accounted_chain(kLimit, false, policy);
    EXPECT_EQ(run.termination, RunTermination::kCompleted);
    EXPECT_TRUE(run.diagnostics_empty);
    // The source has no inputs, so every signal its partition receives is a capacity wake from
    // the transform's retirements.
    ASSERT_LT(run.source_partition, run.owners.size());
    ASSERT_NE(run.owners[run.source_partition], 0U);
    const std::uint64_t source_signals =
        partition_value(run, run.source_partition, Counter::kSignals);
    if (policy == QueuePolicy::kBlocking) {
      EXPECT_GT(source_signals, 0U);
    } else {
      // Key assertion: a drop-oldest producer never waits in steady state, so it is not woken.
      EXPECT_LT(source_signals, kLimit / 4U);
    }
  }
}

[[nodiscard]] std::uint64_t phase_count(const detail::RuntimePhaseCounters& counters,
                                        const detail::RuntimePhaseCounter counter) {
  return counters.snapshot()[detail::RuntimePhaseCounters::index(counter)];
}

// Wait, for a few seconds at most, until `counters` has counted `target` events. Returns whether
// it did.
[[nodiscard]] bool wait_for_phase_count(const detail::RuntimePhaseCounters& counters,
                                        const detail::RuntimePhaseCounter counter,
                                        const std::uint64_t target) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(counters, counter) < target && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(100us);
  }
  return phase_count(counters, counter) >= target;
}

// Empty scans the idle recipient must complete before its wait decision counts as observed.
constexpr std::uint64_t kIdleRecipientScans = 16U;

// A Channel recipient whose partition owner runs on an explicitly admitted busy-poll lane keeps
// polling while idle. The source is held inside compute(), so the only owner scanning is the
// recipient's: it must scan without progress and never enter the sleeping wait, and the run still
// completes or stops normally. The same graph on sleeping lanes parks, so the witness
// discriminates.
// REQ: RUNTIME-041
TEST(RunSessionTest, BusyPollChannelOwnersNeverParkAndStillCompleteOrStop) {
  constexpr std::uint64_t kLimit = 8U;
  struct Arm {
    WakeMode wake_mode;
    test::RuntimeComputeCallAction action;
  };
  for (const Arm arm :
       {Arm{WakeMode::kBusyPoll, test::RuntimeComputeCallAction::kNone},
        Arm{WakeMode::kBusyPoll, test::RuntimeComputeCallAction::kRequestStopAfterFirst},
        Arm{WakeMode::kSleepingWaitset, test::RuntimeComputeCallAction::kNone}}) {
    const bool busy_poll = arm.wake_mode == WakeMode::kBusyPoll;
    SCOPED_TRACE(::testing::Message()
                 << "busy_poll=" << busy_poll << " action=" << static_cast<unsigned>(arm.action));
    test::RuntimeComputeCallChannelProbeState::reset(true);
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(
        test::make_runtime_idle_recipient_plan(kLimit, arm.action, arm.wake_mode, false), counters);
    if (!busy_poll) {
      // The sleeping witness counts empty scans, so keep the 1 ms fallback, not a long park.
      plan = detail::RuntimeTestPlanAccess::with_owner_park_cap(plan, 1ms);
    }
    ASSERT_TRUE(plan.ok()) << plan.json();
    RunSession session = run_async(plan);
    const bool held = test::RuntimeComputeCallChannelProbeState::wait_until_source_compute();
    // From here on the source's owner is inside compute(), so new empty scans are the recipient's.
    const std::uint64_t scans_before =
        phase_count(*counters, detail::RuntimePhaseCounter::kNoProgressScans);
    const std::uint64_t waits_before =
        phase_count(*counters, detail::RuntimePhaseCounter::kWaitCalls);
    const bool idle = held && wait_for_phase_count(*counters,
                                                   detail::RuntimePhaseCounter::kNoProgressScans,
                                                   scans_before + kIdleRecipientScans);
    const std::uint64_t idle_waits =
        phase_count(*counters, detail::RuntimePhaseCounter::kWaitCalls) - waits_before;
    test::RuntimeComputeCallChannelProbeState::release_source_compute();
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
    }
    const bool timed_out = session.is_running();
    if (timed_out) {
      session.request_stop();
    }
    session.wait();
    EXPECT_TRUE(held);
    EXPECT_TRUE(idle);
    EXPECT_FALSE(timed_out);
    EXPECT_EQ(session.termination(),
              arm.action == test::RuntimeComputeCallAction::kRequestStopAfterFirst
                  ? RunTermination::kStopped
                  : RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty());
    if (arm.action == test::RuntimeComputeCallAction::kNone) {
      EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(), kLimit);
    }
    if (busy_poll) {
      // Key assertion: the idle recipient, and every other owner, never parked.
      EXPECT_EQ(idle_waits, 0U);
      EXPECT_EQ(phase_count(*counters, detail::RuntimePhaseCounter::kWaitCalls), 0U);
    } else {
      EXPECT_GT(idle_waits, 0U);
    }
  }
}

// A worker that owns only one local partition applies its lane's busy-poll placement too. The
// consumer process group's sole owner waits on a cross-process import whose producer is withheld:
// it must scan without progress and never enter the sleeping wait, then receive once the producer
// starts, and stop cooperatively. The same consumer on a sleeping lane parks.
// REQ: RUNTIME-041
TEST(RunSessionTest, BusyPollSoleChannelOwnerNeverParksOnAWithheldImport) {
  // Long enough that both workers are still running when the test stops them.
  constexpr std::uint64_t kLimit = 1'000'000U;
  for (const WakeMode wake_mode : {WakeMode::kBusyPoll, WakeMode::kSleepingWaitset}) {
    const bool busy_poll = wake_mode == WakeMode::kBusyPoll;
    SCOPED_TRACE(::testing::Message() << "busy_poll=" << busy_poll);
    test::RuntimeComputeCallChannelProbeState::reset();
    const ExecutionPlan plan = test::make_runtime_idle_recipient_plan(
        kLimit, test::RuntimeComputeCallAction::kNone, wake_mode, true);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto* state = detail::ExecutionPlanAccess::state(plan);
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
    const auto& service = state->cross_process_channel_services.front();
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    ExecutionPlan consumer_plan =
        detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
    if (!busy_poll) {
      // The sleeping witness counts empty scans, so keep the 1 ms fallback, not a long park.
      consumer_plan = detail::RuntimeTestPlanAccess::with_owner_park_cap(consumer_plan, 1ms);
    }
    ASSERT_TRUE(consumer_plan.ok()) << consumer_plan.json();

    RunSession consumer = detail::run_async_worker(consumer_plan, service.target_process_group);
    const bool idle = wait_for_phase_count(
        *counters, detail::RuntimePhaseCounter::kNoProgressScans, kIdleRecipientScans);
    const std::uint64_t idle_waits =
        phase_count(*counters, detail::RuntimePhaseCounter::kWaitCalls);
    bool received = false;
    if (busy_poll) {
      RunSession producer = detail::run_async_worker(plan, service.source_process_group);
      received = test::RuntimeComputeCallChannelProbeState::wait_until_sink_count(2U);
      producer.request_stop();
      producer.wait();
      EXPECT_EQ(producer.termination(), RunTermination::kStopped);
      EXPECT_TRUE(producer.diagnostics().empty())
          << (producer.diagnostics().empty() ? "" : producer.diagnostics().front().token);
    }
    consumer.request_stop();
    consumer.wait();
    EXPECT_TRUE(idle);
    EXPECT_EQ(consumer.termination(), RunTermination::kStopped);
    EXPECT_TRUE(consumer.diagnostics().empty())
        << (consumer.diagnostics().empty() ? "" : consumer.diagnostics().front().token);
    if (busy_poll) {
      EXPECT_TRUE(received);
      // Key assertion: the sole owner never parked, idle or receiving.
      EXPECT_EQ(idle_waits, 0U);
      EXPECT_EQ(phase_count(*counters, detail::RuntimePhaseCounter::kWaitCalls), 0U);
    } else {
      EXPECT_GT(idle_waits, 0U);
    }
  }
}

// An owner whose sources all announce their progress parks until its next deadline. With the long
// cap raised to 10 s, any delivery, capacity retirement, or terminal record that failed to wake its
// owner would stall the chain for 10 s. The chain instead completes promptly with no long park
// ending in a discovery, on blocking and drop-oldest edges.
// REQ: RUNTIME-087
TEST(RunSessionTest, LongParkedOwnersWakeForEveryChainProgress) {
  using Counter = detail::RuntimePartitionCounter;
  constexpr std::uint64_t kLimit = 32U;
  for (const QueuePolicy policy : {QueuePolicy::kBlocking, QueuePolicy::kDropOldest}) {
    SCOPED_TRACE(static_cast<unsigned>(policy));
    const auto started = std::chrono::steady_clock::now();
    const ChainAccounting run = run_accounted_chain(kLimit, false, policy, 10s);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    EXPECT_EQ(run.termination, RunTermination::kCompleted);
    EXPECT_TRUE(run.diagnostics_empty);
    EXPECT_LT(elapsed, 5s);
    if (policy == QueuePolicy::kBlocking) {
      EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(), kLimit);
    }
    std::uint64_t long_parks = 0U;
    std::uint64_t discoveries = 0U;
    for (std::size_t partition = 0U; partition < run.owners.size(); ++partition) {
      long_parks += partition_value(run, partition, Counter::kLongParks);
      discoveries += partition_value(run, partition, Counter::kLongParkDiscoveries);
    }
    // Key assertion: owners parked long, and every long park ended on a wake or a deadline.
    EXPECT_GT(long_parks, 0U);
    EXPECT_EQ(discoveries, 0U);
  }
}

// Ticks on a 20 ms clock in its own partition, so its owner idles between ticks.
class LongParkClockProbe final : public Operator<> {
 public:
  void setup(OperatorSpec&) override {}

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 20ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    compute_count.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }

  static inline std::atomic<std::size_t> compute_count{};
};

// A clock-only owner parks until its next tick instead of returning every 1 ms. A steady-clock
// wait matches a deadline only on a clock that advances at the steady rate, so a scaled clock
// keeps the 1 ms bound.
// REQ: RUNTIME-087
TEST(RunSessionTest, ClockedOwnerParksUntilItsNextTickOnlyOnASteadyRateClock) {
  using Counter = detail::RuntimePartitionCounter;
  constexpr std::size_t kTicks = 10U;
  for (const double scale : {1.0, 2.0}) {
    SCOPED_TRACE(testing::Message() << "time_scale=" << scale);
    LongParkClockProbe::compute_count.store(0U);
    Graph graph{"long-park-clock"};
    graph.partition("first").add(graph.op<LongParkClockProbe>("first"));
    graph.partition("second").add(graph.op<LongParkClockProbe>("second"));
    graph.set_default_clock(graph.add_clock<RealtimeClock>(
        "clock", RealtimeClock::Config{.initial_time_scale = scale}));
    const ExecutionPlan compiled = compile(graph);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
    ASSERT_TRUE(plan.ok()) << plan.json();

    RunSession session = run_async(plan);
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (LongParkClockProbe::compute_count.load() < 2U * kTicks &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
    }
    session.request_stop();
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty());
    EXPECT_GE(LongParkClockProbe::compute_count.load(), 2U * kTicks);

    const auto snapshot = counters->partition_snapshot();
    std::uint64_t long_parks = 0U;
    std::uint64_t cap_returns = 0U;
    for (const auto& partition : snapshot) {
      long_parks += partition[detail::RuntimePhaseCounters::index(Counter::kLongParks)];
      cap_returns += partition[detail::RuntimePhaseCounters::index(Counter::kCapReturns)];
    }
    if (scale == 1.0) {
      // Key assertion: the owners waited for their ticks, not for the 1 ms fallback.
      EXPECT_GT(long_parks, 0U);
      EXPECT_LE(cap_returns, 2U * kTicks);
    } else {
      // Key assertion: a scaled clock keeps the fallback bound.
      EXPECT_EQ(long_parks, 0U);
      EXPECT_GT(cap_returns, 2U * kTicks);
    }
  }
}

// Runs once at the fixed clock phase; the following round has nothing to freeze.
class IdleRoundClockProbe final : public Operator<> {
 public:
  void setup(OperatorSpec&) override {}

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    compute_count.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }

  static inline std::atomic<std::size_t> compute_count{};
};

// REQ: RUNTIME-012
TEST(RunSessionTest, OnClockSkipsTheIdleLateReadinessDrain) {
  // Key setup: an OnClock partition has no freshness reconciliation, window timeouts, or CUDA
  // graph egress, so a round that finds nothing between ticks may skip the late readiness drain.
  // Hold the clock at its initial phase: after one tick, the next tick cannot become due.
  // This forces an idle round even when host scheduling is slower than the authored period.
  IdleRoundClockProbe::compute_count.store(0U);
  Graph graph{"idle-late-drain"};
  const auto probe = graph.op<IdleRoundClockProbe>("probe");
  graph.partition("worker").add(probe);
  const auto clock = std::make_shared<SyntheticClock>(0);
  graph.set_default_clock(graph.bind_shared_clock("clock", clock));
  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto late_drain = std::make_shared<detail::RuntimeLateReadinessDrainTestControl>();
  late_drain->arm();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_late_readiness_drain_test_control(
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters), late_drain);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto elided = [&] {
    return counters->snapshot()[detail::RuntimePhaseCounters::index(
        detail::RuntimePhaseCounter::kLateReadinessDrainsElided)];
  };

  RunSession session = run_async(plan);
  const bool paused = late_drain->wait_until_paused();
  late_drain->release();
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (elided() == 0U && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  session.request_stop();
  clock->stop();
  session.wait();

  // The fixed phase permits exactly one tick, followed by the forced idle-drain elision.
  ASSERT_TRUE(paused);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(IdleRoundClockProbe::compute_count.load(), 1U);
  EXPECT_GT(elided(), 0U);
}

// A clock-driven source that publishes one sample per explicit request. Between requests it
// keeps ticking without emitting, so a parked downstream owner receives no further wake.
class OwnerPollContentionSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (!emit_requested.exchange(false, std::memory_order_acq_rel)) {
      return {};
    }
    const std::uint32_t value = published.fetch_add(1U, std::memory_order_acq_rel) + 1U;
    published_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                       std::memory_order_release);
    return output.emit(value);
  }

  static void reset() noexcept {
    emit_requested.store(false, std::memory_order_relaxed);
    published.store(0U, std::memory_order_relaxed);
    published_ns.store(0, std::memory_order_relaxed);
  }

  static inline std::atomic_bool emit_requested{};
  static inline std::atomic<std::uint32_t> published{};
  static inline std::atomic<std::int64_t> published_ns{};
  Output<std::uint32_t> output;
};

// One of two fanout sinks, each on its own partition so both are pump-fed HoloMQ owners.
class OwnerPollContentionSink final : public Operator<> {
 public:
  explicit OwnerPollContentionSink(const std::size_t index) noexcept : index_(index) {}

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
    received_ns[index_].store(std::chrono::steady_clock::now().time_since_epoch().count(),
                              std::memory_order_release);
    received[index_].fetch_add(1U, std::memory_order_acq_rel);
    return {};
  }

  static void reset() noexcept {
    for (auto& count : received) {
      count.store(0U, std::memory_order_relaxed);
    }
    for (auto& stamp : received_ns) {
      stamp.store(0, std::memory_order_relaxed);
    }
  }

  static inline std::array<std::atomic<std::uint32_t>, 2> received{};
  static inline std::array<std::atomic<std::int64_t>, 2> received_ns{};
  Input<std::uint32_t> input;

 private:
  std::size_t index_{};
};

// Request one publication and wait until both sinks have consumed it.
[[nodiscard]] bool publish_owner_poll_contention_sample(const std::uint32_t expected_deliveries) {
  OwnerPollContentionSource::emit_requested.store(true, std::memory_order_release);
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (OwnerPollContentionSink::received[0].load(std::memory_order_acquire) ==
            expected_deliveries &&
        OwnerPollContentionSink::received[1].load(std::memory_order_acquire) ==
            expected_deliveries) {
      return true;
    }
    std::this_thread::sleep_for(20us);
  }
  return false;
}

// A sink owner whose ordinary readiness poll loses its input cache lock has already consumed the
// wake token that announced the sample, and the lock holder never signals again. Without a retry
// the owner parks for the whole one-millisecond fallback with the sample retained; the retry must
// convert the published revision within the holder's bounded critical section instead.
// REQ: RUNTIME-025
// REQ: RUNTIME-026
TEST(RunSessionTest, ContendedReadinessPollRetriesInsteadOfParkingForTheFallbackWait) {
  OwnerPollContentionSource::reset();
  OwnerPollContentionSink::reset();

  Graph graph{"runtime-owner-poll-contention"};
  const auto source = graph.op<OwnerPollContentionSource>("source");
  const auto first_sink = graph.op<OwnerPollContentionSink>("sink-1", std::size_t{0});
  const auto second_sink = graph.op<OwnerPollContentionSink>("sink-2", std::size_t{1});
  graph.add_flow(source->output, first_sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->output, second_sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition("runtime-owner-poll-contention-source").add(source);
  graph.partition("runtime-owner-poll-contention-sink-1").add(first_sink);
  graph.partition("runtime-owner-poll-contention-sink-2").add(second_sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-owner-poll-contention-clock"));

  // Key setup: the control is armed only after a warm-up delivery, so exactly one steady-state
  // ordinary poll reports the contended decline while nothing else can wake that owner.
  const auto control = std::make_shared<detail::RuntimeChannelIncompletePollTestControl>(
      0U, detail::RuntimeChannelIncompletePollKind::kNonblocking);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_channel_incomplete_poll_test_control(
          detail::RuntimeTestPlanAccess::with_phase_counters(compile(graph), counters), control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  ASSERT_TRUE(publish_owner_poll_contention_sample(1U)) << "warm-up sample was not delivered";

  // Three contended publications; the minimum latency is the oracle so one preempted lane on a
  // loaded host cannot fail the run, while an unfixed owner parks >= 1.06 ms on every trial.
  constexpr std::uint32_t kTrials = 3U;
  std::array<std::int64_t, kTrials> latencies_ns{};
  bool delivered = true;
  for (std::uint32_t trial = 0U; trial < kTrials && delivered; ++trial) {
    control->arm(1U);
    delivered = publish_owner_poll_contention_sample(2U + trial);
    const std::int64_t published_ns =
        OwnerPollContentionSource::published_ns.load(std::memory_order_acquire);
    const std::int64_t completed_ns =
        std::max(OwnerPollContentionSink::received_ns[0].load(std::memory_order_acquire),
                 OwnerPollContentionSink::received_ns[1].load(std::memory_order_acquire));
    latencies_ns[trial] = completed_ns - published_ns;
  }
  session.request_stop();
  session.wait();

  ASSERT_TRUE(delivered) << "contended sample was not delivered";
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto snapshot = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  // Key assertion: every forced decline happened exactly once per trial and each was answered by
  // one retry.
  EXPECT_EQ(control->forced(), kTrials);
  EXPECT_GE(count(detail::RuntimePhaseCounter::kReadinessPollContentionRetries), kTrials);
  EXPECT_GE(count(detail::RuntimePhaseCounter::kChannelPollIncomplete), kTrials);
  // Key assertion: the retained sample completed within the holder's critical section, not after
  // the one-millisecond fallback (which measures ~1.06 ms on a 50 us timer-slack host).
  const std::int64_t min_latency_ns = *std::min_element(latencies_ns.begin(), latencies_ns.end());
  EXPECT_GT(min_latency_ns, 0);
  EXPECT_LT(min_latency_ns, 800'000)
      << "contended-poll completion latencies " << latencies_ns[0] << " / " << latencies_ns[1]
      << " / " << latencies_ns[2] << " ns";
}

struct PumpHeldSlotsProbe {
  static constexpr std::uint32_t kPublications = 4U;
  std::atomic<std::uint32_t> requested{};
  std::atomic<std::uint32_t> received{};
  std::atomic_bool release_sinks{};
};

// Publish only the requested prefix, so a held sink never makes emit() exhaust the graph's 5 ms
// blocking budget. Hold the final callback after emit() so EOS publication starts after release.
class PumpHeldSlotsSource final : public Operator<> {
 public:
  explicit PumpHeldSlotsSource(std::shared_ptr<PumpHeldSlotsProbe> probe)
      : probe_(std::move(probe)) {}

  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms})
        .complete_after(ComputeCalls{.limit = PumpHeldSlotsProbe::kPublications});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    while (published_ >= probe_->requested.load(std::memory_order_acquire) &&
           !probe_->release_sinks.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(100us);
    }
    auto emitted = output.emit(++published_);
    if (!emitted) {
      return make_unexpected(std::move(emitted).error());
    }
    while (published_ == PumpHeldSlotsProbe::kPublications &&
           !probe_->release_sinks.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(100us);
    }
    return {};
  }

  Output<std::uint32_t> output;

 private:
  std::shared_ptr<PumpHeldSlotsProbe> probe_;
  std::uint32_t published_{};
};

class PumpHeldSlotsSink final : public Operator<> {
 public:
  explicit PumpHeldSlotsSink(std::shared_ptr<PumpHeldSlotsProbe> probe)
      : probe_(std::move(probe)) {}

  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    if (sample->data != ++received_) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    probe_->received.fetch_add(1U, std::memory_order_release);
    // The test's scope guard releases this hold even when a setup assertion fails.
    while (!probe_->release_sinks.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(100us);
    }
    return {};
  }

  Input<std::uint32_t> input;

 private:
  std::shared_ptr<PumpHeldSlotsProbe> probe_;
  std::uint32_t received_{};
};

// Fill both typed slots, then stage a third sample with a fourth queued behind it. The pump must
// sleep through an explicitly gated consumer hold, then wake and deliver every sample on release.
// Pump delivery and HoloMQ event wake both serve the depth-two edges here with serial admission.
// REQ: RUNTIME-090
TEST(RunSessionTest, PumpReadyWakesStayBoundedWhileSinksHoldTheirSlots) {
  constexpr std::uint64_t kDeliveries = 2U * PumpHeldSlotsProbe::kPublications;
  for (const bool event_wake : {false, true}) {
    for (const auto hold : {20ms, 40ms}) {
      SCOPED_TRACE(::testing::Message()
                   << "event_wake=" << event_wake << " hold_ms=" << hold.count());
      const auto probe = std::make_shared<PumpHeldSlotsProbe>();
      Graph graph{test::make_host_unique_test_name("runtime-pump-held-slots")};
      const auto source = graph.op<PumpHeldSlotsSource>("source", probe);
      const auto first = graph.op<PumpHeldSlotsSink>("first", probe);
      const auto second = graph.op<PumpHeldSlotsSink>("second", probe);
      graph.add_flow(source->output, first->input, ConnectionOptions{.queue_depth = 2U});
      graph.add_flow(source->output, second->input, ConnectionOptions{.queue_depth = 2U});
      graph.partition("pump-held-slots-source").add(source);
      graph.partition("pump-held-slots-first").add(first);
      graph.partition("pump-held-slots-second").add(second);
      graph.set_default_clock(graph.add_clock<RealtimeClock>("pump-held-slots-clock"));
      ExecutionPlan plan = event_wake
                               ? detail::compile_with_process_local_holomq_event_wake(graph, {})
                               : detail::compile_with_process_local_pump_delivery(graph, {});
      ASSERT_TRUE(plan.ok()) << plan.json();
      const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
      plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
      RunSession session = run_async(plan);
      const auto release_sinks = detail::runtime_channel_binding_test::ScopeExit(
          [&] { probe->release_sinks.store(true, std::memory_order_release); });

      // Wait until both recipients have stored the first two samples before requesting the next
      // two. Each burst fits the transport credits even if CI deschedules every consumer.
      probe->requested.store(2U, std::memory_order_release);
      ASSERT_TRUE(wait_for_phase_count(
          *counters, detail::RuntimePhaseCounter::kChannelSamplesReceived, 4U));
      probe->requested.store(PumpHeldSlotsProbe::kPublications, std::memory_order_release);
      ASSERT_TRUE(wait_for_phase_count(
          *counters, detail::RuntimePhaseCounter::kChannelBlockedHeadsMasked, 2U));
      std::this_thread::sleep_for(hold);
      probe->release_sinks.store(true, std::memory_order_release);

      const auto deadline = std::chrono::steady_clock::now() + 5s;
      while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
      }
      if (session.is_running()) {
        session.request_stop();
      }
      session.wait();
      EXPECT_EQ(session.termination(), RunTermination::kCompleted);
      EXPECT_TRUE(session.diagnostics().empty())
          << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
      EXPECT_EQ(probe->received.load(), kDeliveries);
      EXPECT_EQ(phase_count(*counters, detail::RuntimePhaseCounter::kChannelEmitBackpressured), 0U);
      // Key assertion: at most a spin's worth of ready wakes per delivery, where waking for each
      // queued sample cost thousands per held slot.
      EXPECT_LE(phase_count(*counters, detail::RuntimePhaseCounter::kChannelPumpWaitsReady),
                64U * kDeliveries);
    }
  }
}

// A cross-process import has no publication credits, so samples queue behind a staged head for as
// long as the consumer holds its slot. The consumer process's pump must sleep through each hold and
// still deliver every sample once the sink releases its slot.
// REQ: RUNTIME-090
TEST(RunSessionTest, CrossProcessPumpSleepsWhileTheSinkHoldsItsSlot) {
  constexpr std::uint64_t kLimit = 30U;
  test::RuntimeComputeCallChannelProbeState::reset();
  test::RuntimeComputeCallChannelProbeState::sink_stall_us.store(2'000U);
  const ExecutionPlan plan = test::make_runtime_idle_recipient_plan(
      kLimit, test::RuntimeComputeCallAction::kNone, WakeMode::kSleepingWaitset, true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
  const auto& service = state->cross_process_channel_services.front();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  // Scan on a 1 ms fallback, so the idle consumer's empty scans show that it is running.
  ExecutionPlan consumer_plan = detail::RuntimeTestPlanAccess::with_owner_park_cap(
      detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters), 1ms);
  // Isolate masking/release behavior from the production spin's CPU-dependent iteration count.
  // PumpSleepsThroughAConsumerHoldAndWakesOnItsRelease covers the default timed spin.
  auto control = std::make_shared<detail::RuntimeFanoutTestControl>();
  control->blocked_head_spin = 0ns;
  consumer_plan =
      detail::RuntimeTestPlanAccess::with_fanout_test_control(consumer_plan, std::move(control));
  ASSERT_TRUE(consumer_plan.ok()) << consumer_plan.json();

  // Start the producer only once the consumer is running, so its first publication finds the
  // cross-process subscriber.
  RunSession consumer = detail::run_async_worker(consumer_plan, service.target_process_group);
  const bool idle = wait_for_phase_count(
      *counters, detail::RuntimePhaseCounter::kNoProgressScans, kIdleRecipientScans);
  RunSession producer = detail::run_async_worker(plan, service.source_process_group);
  const bool received = test::RuntimeComputeCallChannelProbeState::wait_until_sink_count(kLimit);
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (producer.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  if (producer.is_running()) {
    producer.request_stop();
  }
  producer.wait();
  consumer.request_stop();
  consumer.wait();
  test::RuntimeComputeCallChannelProbeState::sink_stall_us.store(0U);
  EXPECT_TRUE(idle);
  EXPECT_TRUE(received);
  EXPECT_TRUE(producer.diagnostics().empty())
      << (producer.diagnostics().empty() ? "" : producer.diagnostics().front().token);
  EXPECT_TRUE(consumer.diagnostics().empty())
      << (consumer.diagnostics().empty() ? "" : consumer.diagnostics().front().token);
  EXPECT_GT(phase_count(*counters, detail::RuntimePhaseCounter::kChannelBlockedHeadsMasked), 0U);
  // With the timed spin disabled, publications and releases cause only bounded ready wakes;
  // repeatedly reporting samples behind a blocked head still fails this count guard.
  EXPECT_LE(phase_count(*counters, detail::RuntimePhaseCounter::kChannelPumpWaitsReady),
            64U * kLimit);
}

}  // namespace
}  // namespace holoscan
