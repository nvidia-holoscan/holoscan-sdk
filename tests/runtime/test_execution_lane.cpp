/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "core/source_id_internal.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "threading/thread_placement_contracts.hpp"

namespace holoscan::detail {
namespace {

TEST(ThreadPlacementContracts, ComparesEverySemanticFieldExplicitly) {
  const ThreadPlacement baseline{
      .affinity = CPUAffinity{1U, 3U},
      .scheduling =
          ThreadScheduling{
              .policy = SchedulingPolicy::kFIFO,
              .priority = 7,
          },
      .wake_mode = WakeMode::kBusyPoll,
      .irq_isolation = IRQIsolation::kRequireIsolated,
      .probe_failure_policy = ProbeFailurePolicy::kFailAdmission,
  };
  const auto expect_different = [&](auto mutate) {
    ThreadPlacement candidate = baseline;
    mutate(candidate);
    EXPECT_FALSE(thread_placement_equal(baseline, candidate));
  };

  // Key setup: compare one complete placement with copies that change exactly one value field.
  EXPECT_TRUE(thread_placement_equal(baseline, baseline));
  EXPECT_TRUE(thread_placement_equal(ThreadPlacement{}, ThreadPlacement{}));
  expect_different([](auto& value) { value.affinity.reset(); });
  expect_different([](auto& value) { value.affinity = CPUAffinity{1U, 4U}; });
  expect_different([](auto& value) { value.scheduling.policy = SchedulingPolicy::kRoundRobin; });
  expect_different([](auto& value) { value.scheduling.priority.reset(); });
  expect_different([](auto& value) { value.wake_mode = WakeMode::kSleepingWaitset; });
  expect_different([](auto& value) { value.irq_isolation = IRQIsolation::kPreferIsolated; });
  expect_different(
      [](auto& value) { value.probe_failure_policy = ProbeFailurePolicy::kWarnAndRecordFallback; });

  // Key assertion: equality changes for every field and distinguishes absent optional values.
  ThreadPlacement no_affinity = baseline;
  no_affinity.affinity.reset();
  EXPECT_FALSE(thread_placement_equal(no_affinity, baseline));

  ThreadPlacement deadline = baseline;
  deadline.scheduling = ThreadScheduling{
      .policy = SchedulingPolicy::kDeadline,
      .deadline = DeadlineScheduling{.runtime_ns = 10U, .deadline_ns = 20U, .period_ns = 30U},
  };
  const auto expect_deadline_different = [&](auto mutate) {
    ThreadPlacement candidate = deadline;
    mutate(*candidate.scheduling.deadline);
    EXPECT_FALSE(thread_placement_equal(deadline, candidate));
  };
  EXPECT_TRUE(deadline_scheduling_equal(std::nullopt, std::nullopt));
  EXPECT_FALSE(deadline_scheduling_equal(deadline.scheduling.deadline, std::nullopt));
  expect_deadline_different([](auto& value) { ++value.runtime_ns; });
  expect_deadline_different([](auto& value) { ++value.deadline_ns; });
  expect_deadline_different([](auto& value) { ++value.period_ns; });
}

// Fake PlacementAdapter that records every apply/snapshot call by thread and supports
// injectable faults: fail a specific core, misreport affinity, or block until released.
// Used as the placement backend in every ExecutionLaneSet test in this file.
class RecordingPlacementAdapter final : public ExecutionLanePlacementAdapter {
 public:
  [[nodiscard]] expected<void, ExecutionLanePlacementError> apply(
      const ExecutionLaneOsPlacement& placement) override {
    std::unique_lock lock(mutex_);
    applied_.push_back(placement);
    by_thread_[std::this_thread::get_id()] = placement;
    if (block_core_ && placement.affinity &&
        std::find(placement.affinity->cores.begin(),
                  placement.affinity->cores.end(),
                  *block_core_) != placement.affinity->cores.end()) {
      blocked_ = true;
      block_cv_.notify_all();
      block_cv_.wait(lock, [&] { return release_block_; });
    }
    if (fail_core_ && placement.affinity &&
        std::find(placement.affinity->cores.begin(),
                  placement.affinity->cores.end(),
                  *fail_core_) != placement.affinity->cores.end()) {
      return make_unexpected(ExecutionLanePlacementError{
          .kind = ExecutionLanePlacementErrorKind::kUnavailable,
          .error = Error{ErrorCode::kNotReady, "TEST_PLACEMENT_UNAVAILABLE"},
      });
    }
    return {};
  }

  [[nodiscard]] expected<ExecutionLanePlacementSnapshot, ExecutionLanePlacementError> snapshot()
      override {
    const std::lock_guard lock(mutex_);
    ++snapshot_count_;
    const auto found = by_thread_.find(std::this_thread::get_id());
    if (found == by_thread_.end()) {
      return ExecutionLanePlacementSnapshot{
          .affinity = CPUAffinity{19U},
          .policy = SchedulingPolicy::kOther,
          .priority = std::nullopt,
          .deadline = std::nullopt,
          .reset_on_fork = false,
      };
    }
    const ExecutionLaneOsPlacement& placement = found->second;
    ExecutionLanePlacementSnapshot result{
        .affinity = placement.affinity.value_or(CPUAffinity{}),
        .policy = placement.scheduling.policy,
        .priority = placement.scheduling.priority,
        .deadline = placement.scheduling.deadline,
        .reset_on_fork = placement.scheduling.policy != SchedulingPolicy::kOther,
    };
    if (mismatch_core_ && result.affinity.cores == std::vector<std::uint32_t>{*mismatch_core_}) {
      result.affinity.cores.push_back(*mismatch_core_ + 1U);
    }
    return result;
  }

  [[nodiscard]] expected<void, ExecutionLanePlacementError> restore(
      const ExecutionLanePlacementSnapshot& snapshot) override {
    const std::lock_guard lock(mutex_);
    restored_.push_back(snapshot);
    by_thread_[std::this_thread::get_id()] = ExecutionLaneOsPlacement{
        .affinity = snapshot.affinity,
        .scheduling =
            ThreadScheduling{
                .policy = snapshot.policy,
                .priority = snapshot.priority,
                .deadline = snapshot.deadline,
            },
    };
    return {};
  }

  void fail_on_core(std::uint32_t core) { fail_core_ = core; }
  void mismatch_on_core(std::uint32_t core) { mismatch_core_ = core; }
  void block_on_core(std::uint32_t core) { block_core_ = core; }

  void wait_until_blocked() {
    std::unique_lock lock(mutex_);
    block_cv_.wait(lock, [&] { return blocked_; });
  }

  void release_block() {
    const std::lock_guard lock(mutex_);
    release_block_ = true;
    block_cv_.notify_all();
  }

  [[nodiscard]] std::size_t apply_count() const {
    const std::lock_guard lock(mutex_);
    return applied_.size();
  }

  [[nodiscard]] std::size_t snapshot_count() const {
    const std::lock_guard lock(mutex_);
    return snapshot_count_;
  }

  [[nodiscard]] std::vector<ExecutionLanePlacementSnapshot> restored() const {
    const std::lock_guard lock(mutex_);
    return restored_;
  }

  [[nodiscard]] std::vector<ExecutionLaneOsPlacement> applied() const {
    const std::lock_guard lock(mutex_);
    return applied_;
  }

  [[nodiscard]] bool applied_on(std::thread::id thread) const {
    const std::lock_guard lock(mutex_);
    return by_thread_.contains(thread);
  }

 private:
  mutable std::mutex mutex_;
  std::map<std::thread::id, ExecutionLaneOsPlacement> by_thread_;
  std::vector<ExecutionLaneOsPlacement> applied_;
  std::vector<ExecutionLanePlacementSnapshot> restored_;
  std::size_t snapshot_count_{};
  std::optional<std::uint32_t> fail_core_;
  std::optional<std::uint32_t> mismatch_core_;
  std::optional<std::uint32_t> block_core_;
  std::condition_variable block_cv_;
  bool blocked_{};
  bool release_block_{};
};

// Returns a SchedulerPlanRecord for deterministic serial execution across `lane_count`
// lanes (max_concurrent_activations=1). Used as the baseline scheduler in all tests.
[[nodiscard]] SchedulerPlanRecord serial_scheduler(std::uint32_t lane_count) {
  return SchedulerPlanRecord{
      .ready_policy = AlgorithmKey{"holoscan.ready.deterministic_serial", 1U},
      .execution_lane_count = lane_count,
      .max_concurrent_activations = 1U,
      .activation_overlap = false,
      .worker_count = lane_count,
      .intra_path_parallelism = 1U,
  };
}

// Returns the partition-owner scheduler geometry admitted by persistent lane tasks.
[[nodiscard]] SchedulerPlanRecord parallel_owner_scheduler(std::uint32_t lane_count) {
  return SchedulerPlanRecord{
      .ready_policy = AlgorithmKey{"holoscan.ready.deterministic_serial", 1U},
      .execution_lane_count = lane_count,
      .max_concurrent_activations = lane_count,
      .activation_overlap = true,
      .worker_count = lane_count,
      .intra_path_parallelism = 1U,
  };
}

// Returns the first partition lane with no explicit OS placement.
[[nodiscard]] ExecutionLanePlanRecord default_lane() {
  return ExecutionLanePlanRecord{
      .execution_lane = 0U,
      .partition = ExecutionPartitionId{1U},
      .accepted_placement = ThreadPlacement{},
  };
}

// Returns an ExecutionLanePlanRecord with an explicit CPU core affinity and wake mode.
// Used to build lanes with OS placement for placement, mismatch, and portability tests.
[[nodiscard]] ExecutionLanePlanRecord explicit_lane(std::uint32_t lane, std::uint32_t partition,
                                                    std::uint32_t core, WakeMode wake_mode) {
  return ExecutionLanePlanRecord{
      .execution_lane = lane,
      .partition = ExecutionPartitionId{partition + 1U},
      .accepted_placement =
          ThreadPlacement{
              .affinity = CPUAffinity{core},
              .wake_mode = wake_mode,
              .irq_isolation = IRQIsolation::kPreferIsolated,
          },
  };
}

// Shared counters tracking active concurrency, peak concurrency, and total call count.
// Used with record_callback to verify the serial token prevents concurrent dispatch.
struct CallbackState {
  std::atomic<int>* active;
  std::atomic<int>* maximum;
  std::atomic<int>* calls;
};

// Atomically increments an active counter, sleeps 5 ms, then decrements. Tracks the
// peak observed concurrency in CallbackState::maximum across concurrent dispatches.
[[nodiscard]] expected<void, Error> record_callback(void* opaque) noexcept {
  auto& state = *static_cast<CallbackState*>(opaque);
  const int active = state.active->fetch_add(1, std::memory_order_acq_rel) + 1;
  int maximum = state.maximum->load(std::memory_order_relaxed);
  while (maximum < active &&
         !state.maximum->compare_exchange_weak(maximum, active, std::memory_order_relaxed)) {
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  state.calls->fetch_add(1, std::memory_order_relaxed);
  state.active->fetch_sub(1, std::memory_order_acq_rel);
  return {};
}

// Captures the RuntimeCallbackIdentity observed inside a lane callback invocation.
// Used in RetainsExactCallbackIdentityWhenTheCallbackFails to verify identity threading.
struct CallbackIdentityState {
  RuntimeCallbackIdentity observed{};
};

struct InlineOwnerCallbackState {
  std::thread::id thread;
  RuntimeCallbackIdentity identity;
  std::uint32_t calls{};
};

[[nodiscard]] expected<void, Error> record_inline_owner_callback(void* opaque) noexcept {
  auto& state = *static_cast<InlineOwnerCallbackState*>(opaque);
  state.thread = std::this_thread::get_id();
  state.identity = runtime_callback_identity();
  ++state.calls;
  return {};
}

struct DestroyInlineOwnerCallbackState {
  std::unique_ptr<ExecutionLaneSet>* lanes{};
};

[[nodiscard]] expected<void, Error> destroy_inline_owner_from_callback(void* opaque) noexcept {
  auto& state = *static_cast<DestroyInlineOwnerCallbackState*>(opaque);
  state.lanes->reset();
  return {};
}

[[nodiscard]] expected<void, Error> return_success(void*) noexcept {
  return {};
}

struct RecursiveInlineCallbackState {
  ExecutionLaneSet* lanes{};
  std::optional<expected<void, Error>> nested;
};

[[nodiscard]] expected<void, Error> recurse_inline_owner_callback(void* opaque) noexcept {
  auto& state = *static_cast<RecursiveInlineCallbackState*>(opaque);
  state.nested =
      state.lanes->dispatch(0U, ExecutionLaneTask{.context = nullptr, .invoke = &return_success});
  return {};
}

struct InlineJoinCallbackState {
  ExecutionLaneSet* lanes{};
  std::promise<void>* entered{};
  std::shared_future<void> release;
  std::optional<expected<void, Error>> self_join;
};

[[nodiscard]] expected<void, Error> wait_for_inline_join(void* opaque) noexcept {
  auto& state = *static_cast<InlineJoinCallbackState*>(opaque);
  state.entered->set_value();
  state.release.wait();
  return {};
}

[[nodiscard]] expected<void, Error> join_inline_owner_from_callback(void* opaque) noexcept {
  auto& state = *static_cast<InlineJoinCallbackState*>(opaque);
  state.self_join = state.lanes->join();
  return {};
}

// Snapshots the current runtime identity into CallbackIdentityState and returns a
// failure. Verifies that identity is propagated to the callback even when it errors out.
[[nodiscard]] expected<void, Error> record_callback_identity(void* opaque) noexcept {
  auto& state = *static_cast<CallbackIdentityState*>(opaque);
  state.observed = runtime_callback_identity();
  return make_unexpected(Error{ErrorCode::kFailure, "TEST_CALLBACK_FAILURE"});
}

// Throws a C++ exception from inside a lane task to simulate a post-startup failure.
// Used in PostStartupExceptionPreservesArchivedEvidenceAndReleasesDispatch.
[[nodiscard]] expected<void, Error> throw_from_lane(void*) {
  throw std::runtime_error{"TEST_POST_START_LANE_FAILURE"};
}

// Blocks one lane callback until the test releases it, allowing shutdown to race a known-running
// busy-poll task without relying on sleeps.
struct BlockingCallbackState {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered{};
  bool released{};
};

[[nodiscard]] expected<void, Error> block_callback(void* opaque) noexcept {
  auto& state = *static_cast<BlockingCallbackState*>(opaque);
  std::unique_lock lock(state.mutex);
  state.entered = true;
  state.cv.notify_all();
  state.cv.wait(lock, [&] { return state.released; });
  return {};
}

[[nodiscard]] expected<void, Error> fail_callback(void*) noexcept {
  return make_unexpected(Error{ErrorCode::kProtocolError, "TEST_BUSY_TASK_FAILURE"});
}

struct OwnerTaskHarness {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<std::thread::id> threads;
  std::uint32_t entered{};
  std::uint32_t active{};
  std::uint32_t maximum{};
  bool released{};
};

struct OwnerTaskContext {
  OwnerTaskHarness* harness{};
  std::size_t index{};
};

[[nodiscard]] expected<void, Error> run_owner_task(void* opaque) noexcept {
  const auto& context = *static_cast<OwnerTaskContext*>(opaque);
  OwnerTaskHarness& harness = *context.harness;
  std::unique_lock lock(harness.mutex);
  harness.threads[context.index] = std::this_thread::get_id();
  ++harness.entered;
  ++harness.active;
  harness.maximum = std::max(harness.maximum, harness.active);
  harness.cv.notify_all();
  harness.cv.wait(lock, [&] { return harness.released; });
  --harness.active;
  return {};
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, AppliesAndReadsBackEveryAcceptedLaneOnce) {
  const std::vector lanes{
      default_lane(),
      explicit_lane(1U, 1U, 3U, WakeMode::kSleepingWaitset),
      explicit_lane(2U, 2U, 5U, WakeMode::kBusyPoll),
  };
  auto adapter = std::make_shared<RecordingPlacementAdapter>();
  auto created = ExecutionLaneSet::create(serial_scheduler(3U), lanes, adapter);
  ASSERT_TRUE(created) << runtime_error_message(created.error());
  auto lane_set = std::move(created).value();

  auto started = lane_set->start();
  ASSERT_TRUE(started);
  EXPECT_TRUE(lane_set->ready());
  EXPECT_EQ(adapter->apply_count(), 3U);
  EXPECT_EQ(adapter->snapshot_count(), 3U);
  ASSERT_EQ(lane_set->startup_evidence().size(), 3U);
  for (std::size_t index = 0; index < lane_set->startup_evidence().size(); ++index) {
    const auto& evidence = lane_set->startup_evidence()[index];
    EXPECT_EQ(evidence.execution_lane, index);
    EXPECT_EQ(evidence.disposition, ExecutionLaneStartupDisposition::kReady);
    EXPECT_EQ(evidence.phase, ExecutionLaneStartupPhase::kAllLanesBarrier);
    EXPECT_TRUE(evidence.applied.has_value());
    EXPECT_FALSE(evidence.error.has_value());
  }
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-031
TEST(ExecutionLaneSet, RetainsGlobalIdsForAWorkerLocalSparseLaneSet) {
  const std::vector lanes{
      explicit_lane(1U, 1U, 3U, WakeMode::kSleepingWaitset),
      explicit_lane(2U, 2U, 5U, WakeMode::kSleepingWaitset),
  };
  auto adapter = std::make_shared<RecordingPlacementAdapter>();

  // Key setup: the global topology has lanes 0..2, but this worker materializes only lanes 1 and 2.
  auto scheduler = serial_scheduler(3U);
  scheduler.worker_count = 2U;
  auto created = ExecutionLaneSet::create(scheduler, lanes, adapter);
  ASSERT_TRUE(created) << runtime_error_message(created.error());
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  std::atomic<int> active{};
  std::atomic<int> maximum{};
  std::atomic<int> calls{};
  CallbackState state{&active, &maximum, &calls};
  const auto remote = lane_set->dispatch(0U, ExecutionLaneTask{&state, &record_callback});
  EXPECT_FALSE(remote);
  EXPECT_EQ(remote.error().code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(lane_set->dispatch(1U, ExecutionLaneTask{&state, &record_callback}));
  EXPECT_TRUE(lane_set->dispatch(2U, ExecutionLaneTask{&state, &record_callback}));

  // Key assertion: only local lanes are started, while dispatch retains their global lane IDs.
  EXPECT_EQ(adapter->apply_count(), 2U);
  ASSERT_EQ(lane_set->startup_evidence().size(), 2U);
  EXPECT_EQ(lane_set->startup_evidence()[0].execution_lane, 1U);
  EXPECT_EQ(lane_set->startup_evidence()[1].execution_lane, 2U);
  EXPECT_EQ(calls.load(), 2);
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-011
// GoogleTest fatal assertions terminate this test before optional access, but clang-tidy does not
// model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
TEST(ExecutionLaneSet, ArchivesCompleteSuccessfulPlacementEvidenceAfterJoin) {
  auto placed = explicit_lane(1U, 1U, 3U, WakeMode::kBusyPoll);
  placed.accepted_placement.scheduling = ThreadScheduling{
      .policy = SchedulingPolicy::kFIFO,
      .priority = 7,
  };
  placed.accepted_placement.irq_isolation = IRQIsolation::kRequireIsolated;
  const std::vector lanes{default_lane(), placed};
  auto adapter = std::make_shared<RecordingPlacementAdapter>();
  auto created = ExecutionLaneSet::create(serial_scheduler(2U), lanes, adapter);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  const auto live = lane_set->startup_evidence();
  ASSERT_EQ(live.size(), 2U);
  const std::vector<ExecutionLaneStartupEvidence> archived{live.begin(), live.end()};
  EXPECT_TRUE(lane_set->join());

  const auto retained = lane_set->startup_evidence();
  ASSERT_EQ(retained.size(), archived.size());
  ASSERT_TRUE(retained[1].applied.has_value());
  EXPECT_EQ(retained[1].execution_lane, 1U);
  EXPECT_EQ(retained[1].partition, ExecutionPartitionId{2U});
  EXPECT_EQ(retained[1].disposition, ExecutionLaneStartupDisposition::kReady);
  EXPECT_EQ(retained[1].phase, ExecutionLaneStartupPhase::kAllLanesBarrier);
  EXPECT_EQ(retained[1].applied->affinity.cores, (std::vector<std::uint32_t>{3U}));
  EXPECT_EQ(retained[1].applied->policy, SchedulingPolicy::kFIFO);
  EXPECT_EQ(retained[1].applied->priority, 7);
  EXPECT_TRUE(retained[1].applied->reset_on_fork);
  EXPECT_EQ(retained[1].wake_mode, WakeMode::kBusyPoll);
  EXPECT_EQ(retained[1].irq_isolation, IRQIsolation::kRequireIsolated);
  EXPECT_FALSE(retained[1].error.has_value());

  ASSERT_TRUE(archived[1].applied.has_value());
  EXPECT_EQ(archived[1].applied->affinity.cores, retained[1].applied->affinity.cores);
  EXPECT_EQ(archived[1].applied->policy, retained[1].applied->policy);
  EXPECT_EQ(archived[1].wake_mode, retained[1].wake_mode);
  EXPECT_EQ(archived[1].irq_isolation, retained[1].irq_isolation);
}

// REQ: RUNTIME-004
// NOLINTEND(bugprone-unchecked-optional-access)
TEST(ExecutionLaneSet, RetainsExactCallbackIdentityWhenTheCallbackFails) {
  const std::vector lanes{default_lane()};
  auto created = ExecutionLaneSet::create(
      serial_scheduler(1U), lanes, std::make_shared<RecordingPlacementAdapter>());
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  CallbackIdentityState state;
  const RuntimeCallbackIdentity expected{
      .context = RuntimeCallbackContext::kCompute,
      .operator_id = 23U,
      .source = make_source_id(7U),
      .source_sequence = 41U,
      .tag = Tag{9U, 3U, 0U},
      .tag_time = TagTime{77, 2U, 0U},
      .reference_time = TimePoint{71, ClockId{5U}},
      .state_generation = 3U,
  };
  auto result = lane_set->dispatch(0U,
                                   ExecutionLaneTask{
                                       .context = &state,
                                       .invoke = &record_callback_identity,
                                       .callback_identity = expected,
                                   });
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kFailure);
  EXPECT_EQ(result.error().message, "TEST_CALLBACK_FAILURE");
  EXPECT_EQ(state.observed.context, expected.context);
  EXPECT_EQ(state.observed.operator_id, expected.operator_id);
  EXPECT_EQ(state.observed.source, expected.source);
  EXPECT_EQ(state.observed.source_sequence, expected.source_sequence);
  EXPECT_EQ(state.observed.tag, expected.tag);
  EXPECT_EQ(state.observed.tag_time, expected.tag_time);
  EXPECT_EQ(state.observed.reference_time, expected.reference_time);
  EXPECT_EQ(state.observed.state_generation, expected.state_generation);
  EXPECT_EQ(runtime_callback_identity().context, RuntimeCallbackContext::kNone);
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, PostStartupExceptionPreservesArchivedEvidenceAndReleasesDispatch) {
  const std::vector lanes{default_lane()};
  auto counters = std::make_shared<RuntimePhaseCounters>();
  auto created = ExecutionLaneSet::create(
      serial_scheduler(1U), lanes, std::make_shared<RecordingPlacementAdapter>(), counters.get());
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  const std::vector<ExecutionLaneStartupEvidence> archived = lane_set->take_startup_evidence();
  ASSERT_EQ(archived.size(), 1U);
  EXPECT_EQ(archived.front().disposition, ExecutionLaneStartupDisposition::kReady);
  EXPECT_TRUE(lane_set->startup_evidence().empty());

  auto result = lane_set->dispatch(0U, ExecutionLaneTask{nullptr, &throw_from_lane});

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kFailure);
  EXPECT_EQ(result.error().message, "TEST_POST_START_LANE_FAILURE");
  EXPECT_FALSE(lane_set->ready());
  EXPECT_TRUE(lane_set->startup_evidence().empty());
  EXPECT_TRUE(lane_set->join());
  const auto snapshot = counters->snapshot();
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kLaneCallbacksEntered)], 1U);
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kLaneCallbacksExited)], 1U);
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, HoldsOneGlobalSerialTokenAcrossDistinctLanes) {
  const std::vector lanes{
      default_lane(),
      explicit_lane(1U, 1U, 3U, WakeMode::kSleepingWaitset),
      explicit_lane(2U, 2U, 5U, WakeMode::kBusyPoll),
  };
  auto adapter = std::make_shared<RecordingPlacementAdapter>();
  auto created = ExecutionLaneSet::create(serial_scheduler(3U), lanes, adapter);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  std::atomic<int> active{};
  std::atomic<int> maximum{};
  std::atomic<int> calls{};
  std::atomic<int> successful{};
  CallbackState state{&active, &maximum, &calls};
  // Key setup: three threads each dispatch a 5 ms callback to a distinct lane concurrently.
  std::vector<std::thread> callers;
  callers.reserve(3U);
  for (std::uint32_t lane = 0U; lane < 3U; ++lane) {
    callers.emplace_back([&, lane] {
      auto result = lane_set->dispatch(lane, ExecutionLaneTask{&state, &record_callback});
      if (result) {
        successful.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& caller : callers) {
    caller.join();
  }

  // Key assertion: serial token enforces mutual exclusion; peak active count must be 1.
  EXPECT_EQ(successful.load(), 3);
  EXPECT_EQ(calls.load(), 3);
  EXPECT_EQ(maximum.load(), 1);
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, RunsOnePersistentOwnerTaskPerLaneWithoutLegacyHandoffs) {
  const std::vector lanes{
      default_lane(),
      explicit_lane(1U, 1U, 3U, WakeMode::kBusyPoll),
  };
  auto adapter = std::make_shared<RecordingPlacementAdapter>();
  auto counters = std::make_shared<RuntimePhaseCounters>();
  auto created =
      ExecutionLaneSet::create(parallel_owner_scheduler(2U), lanes, adapter, counters.get());
  ASSERT_TRUE(created) << runtime_error_message(created.error());
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  OwnerTaskHarness harness{.threads = std::vector<std::thread::id>(2U)};
  std::vector contexts{
      OwnerTaskContext{.harness = &harness, .index = 0U},
      OwnerTaskContext{.harness = &harness, .index = 1U},
  };
  const std::vector tasks{
      ExecutionLaneOwnerTask{
          .execution_lane = 0U, .context = &contexts[0], .invoke = &run_owner_task},
      ExecutionLaneOwnerTask{
          .execution_lane = 1U, .context = &contexts[1], .invoke = &run_owner_task},
  };
  ASSERT_TRUE(lane_set->launch_owner_tasks(tasks));
  const auto overlapping = lane_set->launch_owner_tasks(tasks);
  ASSERT_FALSE(overlapping);
  EXPECT_EQ(overlapping.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(overlapping.error().message, "RUNTIME_LANE_OWNER_TASKS_ACTIVE");
  {
    std::unique_lock lock(harness.mutex);
    ASSERT_TRUE(harness.cv.wait_for(
        lock, std::chrono::seconds(2), [&] { return harness.entered == tasks.size(); }));
    EXPECT_EQ(harness.maximum, tasks.size());
    EXPECT_NE(harness.threads[0], harness.threads[1]);
    EXPECT_TRUE(adapter->applied_on(harness.threads[0]));
    EXPECT_TRUE(adapter->applied_on(harness.threads[1]));
    harness.released = true;
  }
  harness.cv.notify_all();

  EXPECT_TRUE(lane_set->wait_owner_tasks());

  const auto snapshot = counters->snapshot();
  const auto count = [&](RuntimePhaseCounter counter) {
    return snapshot[RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneDispatches), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneBusyTaskPublications), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneSleepingWakeSignals), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneCompletionsObserved), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneInlineOwnerInvocations), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneCallbacksEntered), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneCallbacksExited), 0U);

  // Key assertion: the completed owner round reopens ordinary dispatch for
  // lifecycle/control work before the same stable workers accept another round.
  EXPECT_TRUE(
      lane_set->dispatch(0U, ExecutionLaneTask{.context = nullptr, .invoke = &return_success}));

  OwnerTaskHarness second_harness{.threads = std::vector<std::thread::id>(2U)};
  std::vector second_contexts{
      OwnerTaskContext{.harness = &second_harness, .index = 0U},
      OwnerTaskContext{.harness = &second_harness, .index = 1U},
  };
  const std::vector second_tasks{
      ExecutionLaneOwnerTask{
          .execution_lane = 0U, .context = &second_contexts[0], .invoke = &run_owner_task},
      ExecutionLaneOwnerTask{
          .execution_lane = 1U, .context = &second_contexts[1], .invoke = &run_owner_task},
  };
  ASSERT_TRUE(lane_set->launch_owner_tasks(second_tasks));
  {
    std::unique_lock lock(second_harness.mutex);
    ASSERT_TRUE(second_harness.cv.wait_for(lock, std::chrono::seconds(2), [&] {
      return second_harness.entered == second_tasks.size();
    }));
    EXPECT_EQ(second_harness.threads, harness.threads);
    second_harness.released = true;
  }
  second_harness.cv.notify_all();
  EXPECT_TRUE(lane_set->wait_owner_tasks());
  EXPECT_TRUE(lane_set->join());
}

// Interface-only partitions retain global topology lanes without manufacturing callback owners.
// The materialized owner set may therefore be sparse while its worker/concurrency facts remain
// exact for the two callback-bearing partitions.
// REQ: RUNTIME-041
TEST(ExecutionLaneSet, PersistentOwnerTasksAdmitSparseCallbackOwnerProjection) {
  const std::vector lanes{
      default_lane(),
      explicit_lane(2U, 2U, 5U, WakeMode::kSleepingWaitset),
  };
  auto scheduler = parallel_owner_scheduler(3U);
  scheduler.max_concurrent_activations = 2U;
  scheduler.worker_count = 2U;
  auto created =
      ExecutionLaneSet::create(scheduler, lanes, std::make_shared<RecordingPlacementAdapter>());
  ASSERT_TRUE(created) << runtime_error_message(created.error());
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  OwnerTaskHarness harness{.threads = std::vector<std::thread::id>(2U)};
  std::vector contexts{
      OwnerTaskContext{.harness = &harness, .index = 0U},
      OwnerTaskContext{.harness = &harness, .index = 1U},
  };
  const std::vector tasks{
      ExecutionLaneOwnerTask{
          .execution_lane = 0U, .context = &contexts[0], .invoke = &run_owner_task},
      ExecutionLaneOwnerTask{
          .execution_lane = 2U, .context = &contexts[1], .invoke = &run_owner_task},
  };
  ASSERT_TRUE(lane_set->launch_owner_tasks(tasks));
  {
    std::unique_lock lock(harness.mutex);
    ASSERT_TRUE(harness.cv.wait_for(
        lock, std::chrono::seconds(2), [&] { return harness.entered == tasks.size(); }));
    EXPECT_EQ(harness.maximum, tasks.size());
    EXPECT_NE(harness.threads[0], harness.threads[1]);
    harness.released = true;
  }
  harness.cv.notify_all();
  EXPECT_TRUE(lane_set->wait_owner_tasks());
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, PersistentOwnerTasksRespectStartupCoverAndCooperativeStop) {
  const std::vector lanes{
      default_lane(),
      explicit_lane(1U, 1U, 3U, WakeMode::kSleepingWaitset),
  };
  auto created = ExecutionLaneSet::create(
      parallel_owner_scheduler(2U), lanes, std::make_shared<RecordingPlacementAdapter>());
  ASSERT_TRUE(created) << runtime_error_message(created.error());
  auto lane_set = std::move(created).value();

  OwnerTaskHarness harness{.threads = std::vector<std::thread::id>(2U)};
  std::vector contexts{
      OwnerTaskContext{.harness = &harness, .index = 0U},
      OwnerTaskContext{.harness = &harness, .index = 1U},
  };
  const std::vector tasks{
      ExecutionLaneOwnerTask{
          .execution_lane = 0U, .context = &contexts[0], .invoke = &run_owner_task},
      ExecutionLaneOwnerTask{
          .execution_lane = 1U, .context = &contexts[1], .invoke = &run_owner_task},
  };

  const auto before_start = lane_set->launch_owner_tasks(tasks);
  ASSERT_FALSE(before_start);
  EXPECT_EQ(before_start.error().code, ErrorCode::kNotReady);
  ASSERT_TRUE(lane_set->start());

  const std::vector duplicate_cover{
      tasks[0],
      ExecutionLaneOwnerTask{
          .execution_lane = 0U, .context = &contexts[1], .invoke = &run_owner_task},
  };
  const auto duplicate = lane_set->launch_owner_tasks(duplicate_cover);
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(duplicate.error().message, "RUNTIME_LANE_OWNER_TASK_COVER_INVALID");

  ASSERT_TRUE(lane_set->launch_owner_tasks(tasks));
  {
    std::unique_lock lock(harness.mutex);
    ASSERT_TRUE(harness.cv.wait_for(
        lock, std::chrono::seconds(2), [&] { return harness.entered == tasks.size(); }));
  }
  lane_set->request_stop();
  {
    const std::lock_guard lock(harness.mutex);
    harness.released = true;
  }
  harness.cv.notify_all();

  EXPECT_TRUE(lane_set->wait_owner_tasks());
  EXPECT_TRUE(lane_set->join());
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-011
// GoogleTest fatal assertions terminate this test before optional access, but clang-tidy does not
// model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
TEST(ExecutionLaneSet, FailsTheBarrierBeforeCallbacksWhenPlacementIsUnavailable) {
  const std::vector lanes{
      default_lane(),
      explicit_lane(1U, 1U, 7U, WakeMode::kSleepingWaitset),
  };
  auto adapter = std::make_shared<RecordingPlacementAdapter>();
  adapter->fail_on_core(7U);
  auto created = ExecutionLaneSet::create(serial_scheduler(2U), lanes, adapter);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();

  auto started = lane_set->start();
  ASSERT_FALSE(started);
  EXPECT_EQ(started.error().execution_lane, 1U);
  EXPECT_EQ(started.error().disposition, ExecutionLaneStartupDisposition::kUnavailable);
  EXPECT_EQ(started.error().phase, ExecutionLaneStartupPhase::kPlacementApply);
  ASSERT_TRUE(started.error().error.has_value());
  EXPECT_EQ(started.error().error->code, ErrorCode::kNotReady);
  EXPECT_FALSE(lane_set->ready());

  std::atomic<int> active{};
  std::atomic<int> maximum{};
  std::atomic<int> calls{};
  CallbackState state{&active, &maximum, &calls};
  auto dispatched = lane_set->dispatch(0U, ExecutionLaneTask{&state, &record_callback});
  EXPECT_FALSE(dispatched);
  EXPECT_EQ(dispatched.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(calls.load(), 0);
}

// REQ: RUNTIME-011
// NOLINTEND(bugprone-unchecked-optional-access)
TEST(ExecutionLaneSet, RejectsWorkerCountThatDoesNotMatchTheFrozenLaneSet) {
  const std::vector lanes{
      default_lane(),
      explicit_lane(1U, 1U, 7U, WakeMode::kSleepingWaitset),
  };
  auto scheduler = serial_scheduler(2U);
  scheduler.worker_count = 1U;

  auto created =
      ExecutionLaneSet::create(scheduler, lanes, std::make_shared<RecordingPlacementAdapter>());

  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(created.error().message, "RUNTIME_EXECUTION_LANE_PLAN_INVALID");
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, RejectsMissingOrDuplicatePartitionIdentity) {
  auto missing_partition = default_lane();
  missing_partition.partition = {};
  const std::vector missing_lanes{missing_partition};

  const auto missing = ExecutionLaneSet::create(
      serial_scheduler(1U), missing_lanes, std::make_shared<RecordingPlacementAdapter>());

  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(missing.error().message, "RUNTIME_EXECUTION_LANE_PLAN_INVALID");

  const std::vector duplicate_lanes{
      default_lane(),
      explicit_lane(1U, 0U, 7U, WakeMode::kSleepingWaitset),
  };

  const auto duplicate = ExecutionLaneSet::create(
      serial_scheduler(2U), duplicate_lanes, std::make_shared<RecordingPlacementAdapter>());

  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(duplicate.error().message, "PARTITION_PLACEMENT_UNKNOWN_OR_DUPLICATE");
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, RejectsDispatchUntilEveryLaneCrossesTheStartupBarrier) {
  const std::vector lanes{
      default_lane(),
      explicit_lane(1U, 1U, 13U, WakeMode::kSleepingWaitset),
  };
  auto adapter = std::make_shared<RecordingPlacementAdapter>();
  adapter->block_on_core(13U);
  auto created = ExecutionLaneSet::create(serial_scheduler(2U), lanes, adapter);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();

  // Key setup: lane 1 is blocked in apply(); start() runs on a separate thread.
  std::atomic<bool> startup_succeeded{};
  std::thread starter([&] { startup_succeeded.store(lane_set->start().has_value()); });
  adapter->wait_until_blocked();

  std::atomic<int> active{};
  std::atomic<int> maximum{};
  std::atomic<int> calls{};
  CallbackState state{&active, &maximum, &calls};
  auto early = lane_set->dispatch(0U, ExecutionLaneTask{&state, &record_callback});
  EXPECT_FALSE(early);
  EXPECT_EQ(early.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(calls.load(), 0);

  adapter->release_block();
  starter.join();
  // Key assertion: after releasing the block, startup completes and dispatch succeeds.
  ASSERT_TRUE(startup_succeeded.load());
  EXPECT_TRUE(lane_set->dispatch(0U, ExecutionLaneTask{&state, &record_callback}));
  EXPECT_EQ(calls.load(), 1);
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-011
// GoogleTest fatal assertions terminate these placement tests before optional access, but
// clang-tidy does not model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
TEST(ExecutionLaneSet, ReportsTypedReadbackMismatchWithoutMutatingThePlan) {
  std::vector lanes{
      default_lane(),
      explicit_lane(1U, 1U, 9U, WakeMode::kSleepingWaitset),
  };
  const std::vector original = lanes[1].accepted_placement.affinity->cores;
  auto adapter = std::make_shared<RecordingPlacementAdapter>();
  adapter->mismatch_on_core(9U);
  auto created = ExecutionLaneSet::create(serial_scheduler(2U), lanes, adapter);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();

  auto started = lane_set->start();
  ASSERT_FALSE(started);
  EXPECT_EQ(started.error().execution_lane, 1U);
  EXPECT_EQ(started.error().disposition, ExecutionLaneStartupDisposition::kFailed);
  EXPECT_EQ(started.error().phase, ExecutionLaneStartupPhase::kPlacementVerify);
  ASSERT_TRUE(started.error().error.has_value());
  EXPECT_EQ(started.error().error->code, ErrorCode::kIncompatible);
  EXPECT_EQ(lanes[1].accepted_placement.affinity->cores, original);
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, KeepsWakeBehaviorSeparateFromOsPlacement) {
  const std::vector lanes{
      default_lane(),
      explicit_lane(1U, 1U, 11U, WakeMode::kSleepingWaitset),
      explicit_lane(2U, 2U, 11U, WakeMode::kBusyPoll),
  };
  auto adapter = std::make_shared<RecordingPlacementAdapter>();
  auto created = ExecutionLaneSet::create(serial_scheduler(3U), lanes, adapter);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  const auto applied = adapter->applied();
  ASSERT_EQ(applied.size(), 3U);
  std::vector<ExecutionLaneOsPlacement> explicit_placements;
  std::copy_if(applied.begin(),
               applied.end(),
               std::back_inserter(explicit_placements),
               [](const auto& placement) { return placement.affinity.has_value(); });
  ASSERT_EQ(explicit_placements.size(), 2U);
  EXPECT_EQ(explicit_placements[0].affinity->cores, explicit_placements[1].affinity->cores);
  EXPECT_EQ(explicit_placements[0].scheduling.policy, explicit_placements[1].scheduling.policy);
  EXPECT_EQ(lane_set->startup_evidence()[1].wake_mode, WakeMode::kSleepingWaitset);
  EXPECT_EQ(lane_set->startup_evidence()[2].wake_mode, WakeMode::kBusyPoll);

  std::atomic<int> active{};
  std::atomic<int> maximum{};
  std::atomic<int> calls{};
  CallbackState state{&active, &maximum, &calls};
  EXPECT_TRUE(lane_set->dispatch(1U, ExecutionLaneTask{&state, &record_callback}));
  EXPECT_TRUE(lane_set->dispatch(2U, ExecutionLaneTask{&state, &record_callback}));
  EXPECT_EQ(calls.load(), 2);
  EXPECT_TRUE(lane_set->join());
}

// Private phase counters distinguish accepted callback contexts and sleeping/busy lane handoffs
// without changing the public scheduler or placement records.
// REQ: RUNTIME-011
TEST(ExecutionLaneSet, PrivatePhaseCountersTrackExactLaneHandoffs) {
  const std::vector lanes{
      default_lane(),
      explicit_lane(1U, 1U, 11U, WakeMode::kSleepingWaitset),
      explicit_lane(2U, 2U, 11U, WakeMode::kBusyPoll),
  };
  auto counters = std::make_shared<RuntimePhaseCounters>();
  auto created = ExecutionLaneSet::create(
      serial_scheduler(3U), lanes, std::make_shared<RecordingPlacementAdapter>(), counters.get());
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  std::atomic<int> active{};
  std::atomic<int> maximum{};
  std::atomic<int> calls{};
  CallbackState state{&active, &maximum, &calls};

  // Key setup: publish dispatcher and lifecycle work to the sleeping lane and compute work to the
  // busy lane so both task handoff implementations contribute to one immutable snapshot.
  ASSERT_TRUE(lane_set->dispatch(1U,
                                 ExecutionLaneTask{
                                     .context = &state,
                                     .invoke = &record_callback,
                                     .callback_context = RuntimeCallbackContext::kDispatcher,
                                 }));
  ASSERT_TRUE(lane_set->dispatch(
      2U,
      ExecutionLaneTask{
          .context = &state,
          .invoke = &record_callback,
          .callback_identity = RuntimeCallbackIdentity{.context = RuntimeCallbackContext::kCompute},
      }));
  ASSERT_TRUE(lane_set->dispatch(1U,
                                 ExecutionLaneTask{
                                     .context = &state,
                                     .invoke = &record_callback,
                                     .callback_context = RuntimeCallbackContext::kLifecycle,
                                 }));
  ASSERT_TRUE(lane_set->join());

  const auto snapshot = counters->snapshot();
  const auto count = [&](RuntimePhaseCounter counter) {
    return snapshot[RuntimePhaseCounters::index(counter)];
  };
  // Key assertion: every accepted task has one entry, normal exit, and observed completion; its
  // bounded callback identity and wake implementation are each classified exactly once.
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneDispatches), 3U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneComputeDispatches), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneDispatcherDispatches), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneLifecycleDispatches), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneUnknownDispatches), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneBusyTaskPublications), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneSleepingWakeSignals), 2U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneCallbacksEntered), 3U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneCallbacksExited), 3U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneCompletionsObserved), 3U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneSerializationContentions), 0U);

  const auto usage = counters->thread_usage_snapshot();
  const auto& lane_usage =
      usage[RuntimePhaseCounters::thread_usage_index(RuntimeThreadUsageRole::kExecutionLane)];
  EXPECT_EQ(lane_usage.supported, runtime_thread_usage_detail::kSupported);
  EXPECT_EQ(lane_usage.samples,
            runtime_thread_usage_detail::kSupported ? lanes.size() : std::size_t{0U});
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, InlineOwnerAppliesPlacementAndInvokesLifecycleWithoutLaneHandoffs) {
  const std::vector lanes{
      explicit_lane(0U, 0U, 3U, WakeMode::kSleepingWaitset),
  };
  auto adapter = std::make_shared<RecordingPlacementAdapter>();
  auto counters = std::make_shared<RuntimePhaseCounters>();
  auto created = ExecutionLaneSet::create(serial_scheduler(1U), lanes, adapter, counters.get(), 0U);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();

  const std::thread::id owner = std::this_thread::get_id();
  ASSERT_TRUE(lane_set->start());
  EXPECT_TRUE(adapter->applied_on(owner));
  EXPECT_EQ(adapter->apply_count(), 1U);
  EXPECT_EQ(adapter->snapshot_count(), 2U);

  InlineOwnerCallbackState callback;
  const RuntimeCallbackIdentity identity{
      .context = RuntimeCallbackContext::kLifecycle,
      .operator_id = OperatorInstanceId{7U},
      .lifecycle_stage = LifecycleStage::kStart,
      .lifecycle_phase = LifecycleTransitionPhase::kStartupForward,
  };
  ASSERT_TRUE(lane_set->dispatch(0U,
                                 ExecutionLaneTask{
                                     .context = &callback,
                                     .invoke = &record_inline_owner_callback,
                                     .callback_identity = identity,
                                 }));

  EXPECT_EQ(callback.calls, 1U);
  EXPECT_EQ(callback.thread, owner);
  EXPECT_EQ(callback.identity.context, RuntimeCallbackContext::kLifecycle);
  EXPECT_EQ(callback.identity.operator_id, OperatorInstanceId{7U});
  EXPECT_EQ(callback.identity.lifecycle_stage, LifecycleStage::kStart);
  EXPECT_EQ(callback.identity.lifecycle_phase, LifecycleTransitionPhase::kStartupForward);

  const auto snapshot = counters->snapshot();
  const auto count = [&](RuntimePhaseCounter counter) {
    return snapshot[RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneInlineOwnerInvocations), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneDispatches), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneBusyTaskPublications), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneSleepingWakeSignals), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneCompletionsObserved), 0U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneCallbacksEntered), 1U);
  EXPECT_EQ(count(RuntimePhaseCounter::kLaneCallbacksExited), 1U);
  EXPECT_TRUE(lane_set->join());
  const auto usage = counters->thread_usage_snapshot();
  const auto& lane_usage =
      usage[RuntimePhaseCounters::thread_usage_index(RuntimeThreadUsageRole::kExecutionLane)];
  EXPECT_EQ(lane_usage.samples, 0U);
  const auto restored = adapter->restored();
  ASSERT_EQ(restored.size(), 1U);
  EXPECT_EQ(restored.front().affinity.cores, std::vector<std::uint32_t>{19U});
  EXPECT_EQ(restored.front().policy, SchedulingPolicy::kOther);
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, InlineOwnerRestoresPlacementWhenStartupApplyFails) {
  const std::vector lanes{
      explicit_lane(0U, 0U, 3U, WakeMode::kSleepingWaitset),
  };
  auto adapter = std::make_shared<RecordingPlacementAdapter>();
  adapter->fail_on_core(3U);
  auto created = ExecutionLaneSet::create(serial_scheduler(1U), lanes, adapter, nullptr, 0U);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();

  const auto started = lane_set->start();
  ASSERT_FALSE(started);
  EXPECT_EQ(started.error().phase, ExecutionLaneStartupPhase::kPlacementApply);
  const auto restored = adapter->restored();
  ASSERT_EQ(restored.size(), 1U);
  EXPECT_EQ(restored.front().affinity.cores, std::vector<std::uint32_t>{19U});
  EXPECT_EQ(restored.front().policy, SchedulingPolicy::kOther);
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, InlineOwnerAdmitsSparseSingletonProjectionOfSerialPlan) {
  const std::vector lanes{
      explicit_lane(1U, 1U, 3U, WakeMode::kSleepingWaitset),
  };
  auto adapter = std::make_shared<RecordingPlacementAdapter>();
  auto counters = std::make_shared<RuntimePhaseCounters>();
  auto scheduler = serial_scheduler(2U);
  scheduler.worker_count = 1U;
  auto created = ExecutionLaneSet::create(scheduler, lanes, adapter, counters.get(), 1U);
  ASSERT_TRUE(created) << runtime_error_message(created.error());
  auto lane_set = std::move(created).value();

  const std::thread::id owner = std::this_thread::get_id();
  ASSERT_TRUE(lane_set->start());
  EXPECT_TRUE(adapter->applied_on(owner));

  InlineOwnerCallbackState callback;
  ASSERT_TRUE(lane_set->dispatch(
      1U, ExecutionLaneTask{.context = &callback, .invoke = &record_inline_owner_callback}));
  EXPECT_EQ(callback.calls, 1U);
  EXPECT_EQ(callback.thread, owner);

  const auto snapshot = counters->snapshot();
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kLaneInlineOwnerInvocations)],
            1U);
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kLaneDispatches)], 0U);
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, InlineOwnerRejectsDispatchFromAnotherThread) {
  const std::vector lanes{default_lane()};
  auto created = ExecutionLaneSet::create(
      serial_scheduler(1U), lanes, std::make_shared<RecordingPlacementAdapter>(), nullptr, 0U);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  InlineOwnerCallbackState callback;
  std::optional<expected<void, Error>> dispatched;
  std::thread caller([&] {
    dispatched = lane_set->dispatch(
        0U, ExecutionLaneTask{.context = &callback, .invoke = &record_inline_owner_callback});
  });
  caller.join();

  ASSERT_TRUE(dispatched.has_value());
  ASSERT_FALSE(*dispatched);
  EXPECT_EQ(dispatched->error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(dispatched->error().message, "RUNTIME_INLINE_LANE_OWNER_THREAD_REQUIRED");
  EXPECT_EQ(callback.calls, 0U);
  lane_set->request_stop();
  lane_set->request_stop();
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, InlineOwnerRejectsRecursiveDispatchWithoutDeadlock) {
  const std::vector lanes{default_lane()};
  auto created = ExecutionLaneSet::create(
      serial_scheduler(1U), lanes, std::make_shared<RecordingPlacementAdapter>(), nullptr, 0U);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  RecursiveInlineCallbackState callback{.lanes = lane_set.get()};
  ASSERT_TRUE(lane_set->dispatch(
      0U, ExecutionLaneTask{.context = &callback, .invoke = &recurse_inline_owner_callback}));

  ASSERT_TRUE(callback.nested.has_value());
  ASSERT_FALSE(*callback.nested);
  EXPECT_EQ(callback.nested->error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(callback.nested->error().message, "RUNTIME_LANE_RECURSIVE_DISPATCH_FORBIDDEN");
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, InlineOwnerJoinWaitsForActiveCallbackToRetire) {
  using namespace std::chrono_literals;
  const std::vector lanes{default_lane()};
  auto adapter = std::make_shared<RecordingPlacementAdapter>();
  auto created = ExecutionLaneSet::create(serial_scheduler(1U), lanes, adapter, nullptr, 0U);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();

  std::promise<void> entered;
  std::promise<void> release;
  InlineJoinCallbackState callback{
      .lanes = lane_set.get(),
      .entered = &entered,
      .release = release.get_future().share(),
  };
  std::optional<expected<void, ExecutionLaneStartupEvidence>> started;
  std::optional<expected<void, Error>> dispatched;
  auto entered_future = entered.get_future();
  std::thread owner([&] {
    started = lane_set->start();
    if (*started) {
      dispatched = lane_set->dispatch(
          0U, ExecutionLaneTask{.context = &callback, .invoke = &wait_for_inline_join});
    }
  });
  if (entered_future.wait_for(5s) != std::future_status::ready) {
    release.set_value();
    owner.join();
    FAIL() << "inline callback did not enter";
  }

  std::promise<void> join_entered;
  auto join_entered_future = join_entered.get_future();
  auto joined = std::async(std::launch::async, [&]() {
    join_entered.set_value();
    return lane_set->join();
  });
  ASSERT_EQ(join_entered_future.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(joined.wait_for(10ms), std::future_status::timeout);

  release.set_value();
  owner.join();
  ASSERT_TRUE(started.has_value());
  EXPECT_TRUE(*started);
  ASSERT_TRUE(dispatched.has_value());
  EXPECT_TRUE(*dispatched);
  EXPECT_TRUE(joined.get());
  EXPECT_EQ(adapter->restored().size(), 1U);
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, InlineOwnerRejectsDispatchAfterStopAdmissionCloses) {
  const std::vector lanes{default_lane()};
  auto created = ExecutionLaneSet::create(
      serial_scheduler(1U), lanes, std::make_shared<RecordingPlacementAdapter>(), nullptr, 0U);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  lane_set->request_stop();
  InlineOwnerCallbackState callback;
  const auto dispatched = lane_set->dispatch(
      0U, ExecutionLaneTask{.context = &callback, .invoke = &record_inline_owner_callback});

  ASSERT_FALSE(dispatched);
  EXPECT_EQ(dispatched.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(dispatched.error().message, "RUNTIME_LANES_NOT_READY");
  EXPECT_EQ(callback.calls, 0U);
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, InlineOwnerRejectsSelfJoinWithoutDeadlock) {
  const std::vector lanes{default_lane()};
  auto created = ExecutionLaneSet::create(
      serial_scheduler(1U), lanes, std::make_shared<RecordingPlacementAdapter>(), nullptr, 0U);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  InlineJoinCallbackState callback{.lanes = lane_set.get()};
  ASSERT_TRUE(lane_set->dispatch(
      0U, ExecutionLaneTask{.context = &callback, .invoke = &join_inline_owner_from_callback}));

  ASSERT_TRUE(callback.self_join.has_value());
  ASSERT_FALSE(*callback.self_join);
  EXPECT_EQ(callback.self_join->error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(callback.self_join->error().message, "RUNTIME_LANE_SELF_JOIN_FORBIDDEN");
  EXPECT_TRUE(lane_set->join());
}

#if GTEST_HAS_DEATH_TEST
// REQ: RUNTIME-011
TEST(ExecutionLaneSetDeathTest, InlineOwnerDestructionTerminatesBeforeReclaim) {
  EXPECT_DEATH(
      {
        const std::vector lanes{default_lane()};
        auto created = ExecutionLaneSet::create(serial_scheduler(1U),
                                                lanes,
                                                std::make_shared<RecordingPlacementAdapter>(),
                                                nullptr,
                                                0U);
        if (!created) {
          std::terminate();
        }
        auto lane_set = std::move(created).value();
        if (!lane_set->start()) {
          std::terminate();
        }
        DestroyInlineOwnerCallbackState state{.lanes = &lane_set};
        static_cast<void>(lane_set->dispatch(
            0U,
            ExecutionLaneTask{.context = &state, .invoke = &destroy_inline_owner_from_callback}));
      },
      "");
}
#endif

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, RejectsInlineOwnerLaneOutsideTheMaterializedSet) {
  const std::vector lanes{default_lane()};
  auto created = ExecutionLaneSet::create(
      serial_scheduler(1U), lanes, std::make_shared<RecordingPlacementAdapter>(), nullptr, 1U);

  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(created.error().message, "RUNTIME_INLINE_EXECUTION_LANE_INVALID");
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, RejectsInlineOwnerModeForMultipleLocalLanes) {
  const std::vector lanes{
      default_lane(),
      explicit_lane(1U, 1U, 3U, WakeMode::kSleepingWaitset),
  };
  auto created = ExecutionLaneSet::create(
      serial_scheduler(2U), lanes, std::make_shared<RecordingPlacementAdapter>(), nullptr, 0U);

  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(created.error().message, "RUNTIME_INLINE_EXECUTION_LANE_INVALID");
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, BusyPollReturnsExactTaskFailure) {
  // Key setup: dispatch one typed failure through a started busy-poll lane.
  const std::vector lanes{default_lane(), explicit_lane(1U, 1U, 11U, WakeMode::kBusyPoll)};
  auto created = ExecutionLaneSet::create(
      serial_scheduler(2U), lanes, std::make_shared<RecordingPlacementAdapter>());
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  auto result = lane_set->dispatch(1U, ExecutionLaneTask{nullptr, &fail_callback});

  // Key assertion: the atomic handoff preserves the task's exact code and message.
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kProtocolError);
  EXPECT_EQ(result.error().message, "TEST_BUSY_TASK_FAILURE");
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, BusyPollExceptionClosesLaneAndReleasesDispatch) {
  // Key setup: throw from a task after the busy-poll worker has started.
  const std::vector lanes{default_lane(), explicit_lane(1U, 1U, 11U, WakeMode::kBusyPoll)};
  auto counters = std::make_shared<RuntimePhaseCounters>();
  auto created = ExecutionLaneSet::create(
      serial_scheduler(2U), lanes, std::make_shared<RecordingPlacementAdapter>(), counters.get());
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  auto result = lane_set->dispatch(1U, ExecutionLaneTask{nullptr, &throw_from_lane});

  // Key assertion: exception closure returns the archived error and makes the lane non-ready.
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kFailure);
  EXPECT_EQ(result.error().message, "TEST_POST_START_LANE_FAILURE");
  EXPECT_FALSE(lane_set->ready());
  EXPECT_TRUE(lane_set->join());
  const auto snapshot = counters->snapshot();
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kLaneCallbacksEntered)], 1U);
  EXPECT_EQ(snapshot[RuntimePhaseCounters::index(RuntimePhaseCounter::kLaneCallbacksExited)], 1U);
}

// REQ: RUNTIME-011
TEST(ExecutionLaneSet, BusyPollStopDuringRunningTaskPreservesCompletion) {
  // Key setup: hold one busy-poll task in its callback while another thread requests stop.
  const std::vector lanes{default_lane(), explicit_lane(1U, 1U, 11U, WakeMode::kBusyPoll)};
  auto created = ExecutionLaneSet::create(
      serial_scheduler(2U), lanes, std::make_shared<RecordingPlacementAdapter>());
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  BlockingCallbackState callback;
  std::optional<expected<void, Error>> dispatch_result;
  std::thread caller([&] {
    dispatch_result =
        lane_set->dispatch(1U, ExecutionLaneTask{.context = &callback, .invoke = &block_callback});
  });
  {
    std::unique_lock lock(callback.mutex);
    callback.cv.wait(lock, [&] { return callback.entered; });
  }

  // Shutdown rejects later work but lets the already-running callback publish its exact result.
  lane_set->request_stop();
  {
    const std::lock_guard lock(callback.mutex);
    callback.released = true;
  }
  callback.cv.notify_all();
  caller.join();

  // Key assertion: stop does not overwrite the successful completion already in flight.
  ASSERT_TRUE(dispatch_result.has_value());
  EXPECT_TRUE(*dispatch_result);
  EXPECT_TRUE(lane_set->join());
}

// REQ: RUNTIME-011
// NOLINTEND(bugprone-unchecked-optional-access)
TEST(ExecutionLaneSet, StopAndJoinAreIdempotentAndRejectLaterDispatch) {
  const std::vector lanes{default_lane()};
  auto adapter = std::make_shared<RecordingPlacementAdapter>();
  auto created = ExecutionLaneSet::create(serial_scheduler(1U), lanes, adapter);
  ASSERT_TRUE(created);
  auto lane_set = std::move(created).value();
  ASSERT_TRUE(lane_set->start());

  lane_set->request_stop();
  lane_set->request_stop();
  EXPECT_TRUE(lane_set->join());
  EXPECT_TRUE(lane_set->join());
  EXPECT_FALSE(lane_set->ready());

  std::atomic<int> active{};
  std::atomic<int> maximum{};
  std::atomic<int> calls{};
  CallbackState state{&active, &maximum, &calls};
  auto dispatched = lane_set->dispatch(0U, ExecutionLaneTask{&state, &record_callback});
  EXPECT_FALSE(dispatched);
  EXPECT_EQ(dispatched.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(calls.load(), 0);
}

// REQ: RUNTIME-011
// GoogleTest fatal assertions terminate this platform test before optional access, but clang-tidy
// does not model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
TEST(ExecutionLaneSet, AppliesOneAllowedCpusetCpuWithoutElevatedSchedulingPrivilege) {
#ifdef __linux__
  auto system_adapter = make_system_execution_lane_placement_adapter();
  const auto current = system_adapter->snapshot();
  ASSERT_TRUE(current) << current.error().error.message;
  if (current->affinity.cores.empty()) {
    GTEST_SKIP() << "Host exposed no allowed cpuset CPU";
  }

  const std::uint32_t allowed_cpu = current->affinity.cores.front();
  auto lane = default_lane();
  lane.accepted_placement.affinity = CPUAffinity{allowed_cpu};
  lane.accepted_placement.wake_mode = WakeMode::kSleepingWaitset;
  const std::vector lanes{lane};
  auto created = ExecutionLaneSet::create(serial_scheduler(1U), lanes, system_adapter);
  ASSERT_TRUE(created) << created.error().message;
  auto lane_set = std::move(created).value();

  const auto started = lane_set->start();
  ASSERT_TRUE(started) << started.error().error->message;
  ASSERT_EQ(lane_set->startup_evidence().size(), 1U);
  const auto& evidence = lane_set->startup_evidence().front();
  ASSERT_TRUE(evidence.applied.has_value());
  EXPECT_EQ(evidence.applied->affinity.cores, std::vector<std::uint32_t>{allowed_cpu});
  EXPECT_EQ(evidence.applied->policy, SchedulingPolicy::kOther);
  EXPECT_TRUE(lane_set->join());
#else
  GTEST_SKIP() << "Linux cpuset validation is not applicable on this host";
#endif
}

// REQ: RUNTIME-011
// NOLINTEND(bugprone-unchecked-optional-access)
TEST(ExecutionLanePortability, ClassifiesUnsupportedOsPlacementWithoutNativeExecution) {
  const ExecutionLaneOsPlacement default_placement{};
  EXPECT_TRUE(validate_execution_lane_platform_placement(false, default_placement));

  ExecutionLaneOsPlacement affinity_placement{};
  affinity_placement.affinity = CPUAffinity{3U};
  auto affinity = validate_execution_lane_platform_placement(false, affinity_placement);
  ASSERT_FALSE(affinity);
  EXPECT_EQ(affinity.error().kind, ExecutionLanePlacementErrorKind::kUnavailable);
  EXPECT_EQ(affinity.error().error.code, ErrorCode::kNotSupported);
  EXPECT_EQ(affinity.error().error.message, "RUNTIME_LANE_OS_PLACEMENT_UNAVAILABLE");

  ExecutionLaneOsPlacement scheduling_placement{};
  scheduling_placement.scheduling.policy = SchedulingPolicy::kFIFO;
  auto scheduling = validate_execution_lane_platform_placement(false, scheduling_placement);
  ASSERT_FALSE(scheduling);
  EXPECT_EQ(scheduling.error().kind, ExecutionLanePlacementErrorKind::kUnavailable);
  EXPECT_EQ(scheduling.error().error.code, ErrorCode::kNotSupported);
  EXPECT_EQ(scheduling.error().error.message, "RUNTIME_LANE_OS_PLACEMENT_UNAVAILABLE");
}

// REQ: RUNTIME-011
TEST(ExecutionLanePortability, ProjectsStructuralCapabilityWhileRetainingLaneCause) {
  const ExecutionLaneStartupEvidence unsupported{
      .execution_lane = 2U,
      .partition = ExecutionPartitionId{7U},
      .disposition = ExecutionLaneStartupDisposition::kUnavailable,
      .phase = ExecutionLaneStartupPhase::kPlacementApply,
      .error = Error{ErrorCode::kNotSupported, "RUNTIME_LANE_OS_PLACEMENT_UNAVAILABLE"},
  };
  EXPECT_EQ(execution_lane_startup_diagnostic_token(unsupported),
            "PLATFORM_CAPABILITY_UNAVAILABLE");
  ASSERT_TRUE(unsupported.error.has_value());
  EXPECT_EQ(unsupported.error->code, ErrorCode::kNotSupported);
  EXPECT_EQ(unsupported.error->message, "RUNTIME_LANE_OS_PLACEMENT_UNAVAILABLE");

  const ExecutionLaneStartupEvidence privilege_unavailable{
      .execution_lane = 2U,
      .partition = ExecutionPartitionId{7U},
      .disposition = ExecutionLaneStartupDisposition::kUnavailable,
      .phase = ExecutionLaneStartupPhase::kPlacementApply,
      .error = Error{ErrorCode::kPermissionDenied, "RUNTIME_LANE_AFFINITY_PERMISSION_DENIED"},
  };
  EXPECT_EQ(execution_lane_startup_diagnostic_token(privilege_unavailable),
            "RUNTIME_LANE_AFFINITY_PERMISSION_DENIED");
}

}  // namespace
}  // namespace holoscan::detail
