/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "../common/allocation_counter.hpp"
#include "core/diagnostic_registry.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/physical_readiness_binding.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan {
namespace {

constexpr ClockId kClock{17U};

// Wraps a nanosecond timestamp into a TimePoint on the shared test clock (kClock = 17).
[[nodiscard]] TimePoint at(std::int64_t timestamp_ns) {
  return TimePoint{timestamp_ns, kClock};
}

// Tracks poll/wake/assertion counts for the poll_asserted and wake callbacks injected into
// a PhysicalReadinessBinding under test. In alternate mode, asserts on odd polls up to
// max_assertions. Representative test: HoldsReadableCreditUntilRetirementAndRearmsLevel.
struct CallbackState {
  std::atomic_bool asserted{};
  std::atomic<std::uint32_t> polls{};
  std::atomic<std::uint32_t> wakes{};
  std::atomic<std::size_t> poll_thread_hash{};
  std::atomic<std::uint32_t> assertions{};
  bool alternate{};
  std::uint32_t max_assertions{};
};

// Suspends a publisher mid-flight at a specific PhysicalReadinessPublicationCheckpoint so
// a concurrent terminal action (close/fail/invalidate) can race the publication path.
// Representative test: TerminalTransitionOwnsPublicationCreditExactlyOnce.
class PublicationBarrier {
 public:
  explicit PublicationBarrier(detail::PhysicalReadinessPublicationCheckpoint checkpoint)
      : checkpoint_(checkpoint) {}

  [[nodiscard]] bool wait_until_entered() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds{5}, [this] { return entered_; });
  }

  void release() {
    const std::lock_guard lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

  void checkpoint(detail::PhysicalReadinessPublicationCheckpoint checkpoint) noexcept {
    if (checkpoint != checkpoint_) {
      return;
    }
    std::unique_lock lock(mutex_);
    entered_ = true;
    cv_.notify_all();
    cv_.wait(lock, [this] { return released_; });
  }

 private:
  detail::PhysicalReadinessPublicationCheckpoint checkpoint_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool entered_{};
  bool released_{};
};

// Free-function shim that forwards checkpoint callbacks to the owning PublicationBarrier.
void publication_checkpoint(void* state,
                            detail::PhysicalReadinessPublicationCheckpoint checkpoint) noexcept {
  static_cast<PublicationBarrier*>(state)->checkpoint(checkpoint);
}

// Busy-poll predicate callback: returns whether the bound source is considered "ready".
// In alternate mode, returns true only on odd-numbered polls, up to max_assertions times.
[[nodiscard]] expected<bool, Error> poll_asserted(void* state) noexcept {
  auto& callbacks = *static_cast<CallbackState*>(state);
  const std::uint32_t poll = callbacks.polls.fetch_add(1U, std::memory_order_relaxed) + 1U;
  callbacks.poll_thread_hash.store(std::hash<std::thread::id>{}(std::this_thread::get_id()),
                                   std::memory_order_release);
  if (callbacks.alternate) {
    if ((poll & 1U) == 0U) {
      return false;
    }
    const std::uint32_t assertion = callbacks.assertions.fetch_add(1U, std::memory_order_relaxed);
    return assertion < callbacks.max_assertions;
  }
  return callbacks.asserted.load(std::memory_order_acquire);
}

[[nodiscard]] expected<bool, Error> poll_clock_binding_failure(void*) noexcept {
  return make_unexpected(
      Error{ErrorCode::kProtocolError,
            "RUNTIME_CLOCK_BINDING_UNAVAILABLE: injected readiness clock failure"});
}

// Wake callback: increments the wake count each time the binding signals the executor.
void wake(void* state) noexcept {
  static_cast<CallbackState*>(state)->wakes.fetch_add(1U, std::memory_order_relaxed);
}

// Factory helper that builds a PhysicalReadinessBindingDescriptor with sensible defaults.
// Wires poll/wake callbacks when provided; sets wake mode based on kSpin vs. other kinds.
[[nodiscard]] detail::PhysicalReadinessBindingDescriptor descriptor(
    std::uint32_t source, detail::RuntimeEventKind kind, std::uint32_t capacity,
    CallbackState* callbacks = nullptr) {
  return detail::PhysicalReadinessBindingDescriptor{
      .request =
          detail::RuntimePortBindingRequest{
              .plan_generation = 1U,
              .run_generation = 1U,
              .kind = kind,
              .source = detail::make_source_id(source),
              .source_clock = kClock,
              .reference_clock = kClock,
              .execution_lane = source - 1U,
              .capacity = capacity,
              .source_generation = 1U,
          },
      .wake_requirement = kind == detail::RuntimeEventKind::kSpin
                              ? ReadinessWakeRequirement::kBusyPollRequired
                              : ReadinessWakeRequirement::kDeploymentSelected,
      .wake_mode = kind == detail::RuntimeEventKind::kSpin ? WakeMode::kBusyPoll
                                                           : WakeMode::kSleepingWaitset,
      .execution_lane = source - 1U,
      .poll_state = callbacks,
      .poll_asserted = callbacks == nullptr ? nullptr : &poll_asserted,
      .wake_state = callbacks,
      .wake = callbacks == nullptr ? nullptr : &wake,
  };
}

// Groups a PhysicalReadinessBindingFactory, one bound PortBinding, and a control handle
// into a single object for concise per-test setup via the bind() helper below.
struct BoundSource {
  std::shared_ptr<detail::PhysicalReadinessBindingFactory> factory;
  std::unique_ptr<detail::PortBinding> binding;
  detail::PhysicalReadinessControl control;
};

// RuntimePortBindingFactory adapter that creates a busy-poll PhysicalReadinessBinding
// backed by CallbackState; exposes the outstanding credit count for post-run assertions.
// Used by: RunSessionPollsOnSelectedLane..., RunSessionRejectsUnaccepted...,
//          RunSessionReturnsCreditOnCallbackFailureAndCooperativeStop.
class PollingPhysicalFactory final : public detail::RuntimePortBindingFactory {
 public:
  PollingPhysicalFactory(CallbackState& callbacks, std::uint32_t execution_lane,
                         expected<bool, Error> (*poll)(void*) noexcept = &poll_asserted)
      : callbacks_(&callbacks), execution_lane_(execution_lane), poll_(poll) {}

  [[nodiscard]] expected<std::unique_ptr<detail::PortBinding>, Error> bind(
      const detail::RuntimePortBindingRequest& request,
      TimePoint start_coordinate) noexcept override {
    detail::PhysicalReadinessBindingDescriptor source{
        .request = request,
        .wake_requirement = ReadinessWakeRequirement::kBusyPollRequired,
        .wake_mode = WakeMode::kBusyPoll,
        .execution_lane = execution_lane_,
        .poll_state = callbacks_,
        .poll_asserted = poll_,
    };
    auto created = detail::PhysicalReadinessBindingFactory::create(
        std::span<const detail::PhysicalReadinessBindingDescriptor>{&source, 1U});
    if (!created) {
      return make_unexpected(std::move(created.error()));
    }
    auto binding = (*created)->bind(request, start_coordinate);
    if (!binding) {
      return make_unexpected(std::move(binding.error()));
    }
    auto control = (*created)->control(
        request.source, request.plan_generation, request.run_generation, request.source_generation);
    if (!control) {
      return make_unexpected(std::move(control.error()));
    }
    physical_ = std::move(*created);
    control_ = std::move(*control);
    return binding;
  }

  [[nodiscard]] std::size_t outstanding() const noexcept { return control_.outstanding(); }

 private:
  CallbackState* callbacks_{};
  std::uint32_t execution_lane_{};
  expected<bool, Error> (*poll_)(void*) noexcept {};
  std::shared_ptr<detail::PhysicalReadinessBindingFactory> physical_;
  detail::PhysicalReadinessControl control_;
};

class FreeFormRejectingFactory final : public detail::RuntimePortBindingFactory {
 public:
  [[nodiscard]] expected<std::unique_ptr<detail::PortBinding>, Error> bind(
      const detail::RuntimePortBindingRequest&, TimePoint) noexcept override {
    return make_unexpected(Error{ErrorCode::kIncompatible, "driver-specific readiness failure"});
  }
};

struct TerminalSummarySnapshot {
  DiagnosticId first_diagnostic{};
  std::string first_token;
};

class RecordingTerminalSummarySink final : public detail::RuntimeTerminalSummarySink {
 public:
  void publish(const detail::RuntimeTerminalSummary& summary) override {
    const std::lock_guard lock(mutex_);
    snapshots_.push_back(TerminalSummarySnapshot{
        .first_diagnostic = summary.first_diagnostic,
        .first_token = std::string{summary.first_token},
    });
  }

  [[nodiscard]] std::vector<TerminalSummarySnapshot> snapshot() const {
    const std::lock_guard lock(mutex_);
    return snapshots_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<TerminalSummarySnapshot> snapshots_;
};

// Test helper: creates a PhysicalReadinessBindingFactory, calls bind(), and acquires
// the control handle; panics via EXPECT macros on any failure.
[[nodiscard]] BoundSource bind(detail::PhysicalReadinessBindingDescriptor source) {
  auto created = detail::PhysicalReadinessBindingFactory::create(
      std::span<const detail::PhysicalReadinessBindingDescriptor>{&source, 1U});
  EXPECT_TRUE(created.has_value());
  auto bound = (*created)->bind(source.request, at(1));
  EXPECT_TRUE(bound.has_value());
  auto control = (*created)->control(source.request.source,
                                     source.request.plan_generation,
                                     source.request.run_generation,
                                     source.request.source_generation);
  EXPECT_TRUE(control.has_value());
  return BoundSource{
      .factory = std::move(*created),
      .binding = std::move(*bound),
      .control = std::move(*control),
  };
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, RejectsDuplicateMissingAndUnmonitoredDescriptorsBeforeBind) {
  // Key setup: same descriptor used twice; also tests monitored_health_required without _bound.
  auto source = descriptor(1U, detail::RuntimeEventKind::kNotification, 2U);
  const std::array duplicates{source, source};
  auto duplicate = detail::PhysicalReadinessBindingFactory::create(duplicates);
  ASSERT_FALSE(duplicate.has_value());
  EXPECT_EQ(duplicate.error().code, ErrorCode::kDuplicateName);

  source.monitored_health_required = true;
  source.monitored_health_disposition = EndpointClockHealthDisposition::kFailRun;
  auto unavailable = detail::PhysicalReadinessBindingFactory::create(
      std::span<const detail::PhysicalReadinessBindingDescriptor>{&source, 1U});
  ASSERT_FALSE(unavailable.has_value());
  EXPECT_EQ(unavailable.error().code, ErrorCode::kNotSupported);

  source.monitored_health_bound = true;
  auto created = detail::PhysicalReadinessBindingFactory::create(
      std::span<const detail::PhysicalReadinessBindingDescriptor>{&source, 1U});
  ASSERT_TRUE(created.has_value());
  auto missing_request = source.request;
  missing_request.source = detail::make_source_id(77U);
  auto missing = (*created)->bind(missing_request, at(1));
  ASSERT_FALSE(missing.has_value());
  EXPECT_EQ(missing.error().code, ErrorCode::kNotFound);

  auto mismatched = source.request;
  mismatched.capacity = 3U;
  auto wrong = (*created)->bind(mismatched, at(1));
  ASSERT_FALSE(wrong.has_value());
  EXPECT_EQ(wrong.error().code, ErrorCode::kIncompatible);

  auto foreign_plan = source.request;
  ++foreign_plan.plan_generation;
  wrong = (*created)->bind(foreign_plan, at(1));
  ASSERT_FALSE(wrong.has_value());
  EXPECT_EQ(wrong.error().code, ErrorCode::kIncompatible);
  auto foreign_control = (*created)->control(source.request.source,
                                             foreign_plan.plan_generation,
                                             source.request.run_generation,
                                             source.request.source_generation);
  // Key assertion: each violation returns a distinct error code without corrupting state.
  ASSERT_FALSE(foreign_control.has_value());
  EXPECT_EQ(foreign_control.error().code, ErrorCode::kStaleEndpoint);

  auto stale_run = source.request;
  ++stale_run.run_generation;
  wrong = (*created)->bind(stale_run, at(1));
  ASSERT_FALSE(wrong.has_value());
  EXPECT_EQ(wrong.error().code, ErrorCode::kIncompatible);
  foreign_control = (*created)->control(source.request.source,
                                        source.request.plan_generation,
                                        stale_run.run_generation,
                                        source.request.source_generation);
  ASSERT_FALSE(foreign_control.has_value());
  EXPECT_EQ(foreign_control.error().code, ErrorCode::kStaleEndpoint);

  auto foreign_clock = source.request;
  foreign_clock.reference_clock = ClockId{18U};
  wrong = (*created)->bind(foreign_clock, TimePoint{1, foreign_clock.reference_clock});
  ASSERT_FALSE(wrong.has_value());
  EXPECT_EQ(wrong.error().code, ErrorCode::kIncompatible);

  auto foreign_lane = source.request;
  ++foreign_lane.execution_lane;
  wrong = (*created)->bind(foreign_lane, at(1));
  ASSERT_FALSE(wrong.has_value());
  EXPECT_EQ(wrong.error().code, ErrorCode::kIncompatible);

  auto mismatched_lane = source;
  ++mismatched_lane.execution_lane;
  auto mismatched_lane_factory = detail::PhysicalReadinessBindingFactory::create(
      std::span<const detail::PhysicalReadinessBindingDescriptor>{&mismatched_lane, 1U});
  ASSERT_FALSE(mismatched_lane_factory.has_value());
  EXPECT_EQ(mismatched_lane_factory.error().code, ErrorCode::kInvalidArgument);

  auto invalid_lane = source;
  invalid_lane.execution_lane = std::numeric_limits<std::uint32_t>::max();
  auto invalid_lane_factory = detail::PhysicalReadinessBindingFactory::create(
      std::span<const detail::PhysicalReadinessBindingDescriptor>{&invalid_lane, 1U});
  ASSERT_FALSE(invalid_lane_factory.has_value());
  EXPECT_EQ(invalid_lane_factory.error().code, ErrorCode::kInvalidArgument);
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, HoldsReadableCreditUntilRetirementAndRearmsLevel) {
  // Key setup: kReadable kind with capacity=1; credit is held from accept through retire.
  CallbackState callbacks;
  auto source = bind(descriptor(1U, detail::RuntimeEventKind::kReadable, 1U, &callbacks));

  ASSERT_TRUE(source.control.observe_readable(true, at(10)).has_value());
  ASSERT_TRUE(source.control.observe_readable(true, at(11)).has_value());
  EXPECT_EQ(source.control.outstanding(), 1U);

  detail::RuntimeEvent event;
  auto drained = source.binding->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained.has_value());
  ASSERT_EQ(*drained, 1U);
  EXPECT_EQ(event.kind, detail::RuntimeEventKind::kReadable);
  EXPECT_EQ(event.source_sequence, 1U);
  EXPECT_EQ(event.reference_time, at(10));

  // Dequeue is not retirement; the accepted-through-retirement credit remains charged.
  EXPECT_EQ(source.control.outstanding(), 1U);
  ASSERT_TRUE(source.control.observe_readable(true, at(12)).has_value());
  drained = source.binding->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained.has_value());
  EXPECT_EQ(*drained, 0U);

  ASSERT_TRUE(source.control.retire(1U).has_value());
  EXPECT_EQ(source.control.outstanding(), 0U);
  ASSERT_TRUE(source.control.observe_readable(true, at(13)).has_value());
  drained = source.binding->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained.has_value());
  // Key assertion: after retire(), the next observe_readable arms a fresh event (seq 2).
  ASSERT_EQ(*drained, 1U);
  EXPECT_EQ(event.source_sequence, 2U);
  EXPECT_EQ(event.reference_time, at(13));
  EXPECT_GE(callbacks.wakes.load(std::memory_order_relaxed), 3U);
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, OrdersConcurrentNotificationsAndDoesNotReturnCreditOnDrain) {
  // Key setup: capacity-4 binding; 8 concurrent producers race to post — only 4 accepted.
  auto source = bind(descriptor(2U, detail::RuntimeEventKind::kNotification, 4U));

  std::atomic<std::uint32_t> accepted{};
  std::vector<std::thread> producers;
  producers.reserve(8U);
  for (std::uint32_t index = 0U; index < 8U; ++index) {
    producers.emplace_back([&, index] {
      auto posted = source.control.post_notification(at(20 + index));
      if (posted) {
        accepted.fetch_add(1U, std::memory_order_relaxed);
      } else {
        EXPECT_EQ(posted.error().code, ErrorCode::kBackpressured);
      }
    });
  }
  for (auto& producer : producers) {
    producer.join();
  }
  EXPECT_EQ(accepted.load(std::memory_order_relaxed), 4U);
  EXPECT_EQ(source.control.outstanding(), 4U);

  std::array<detail::RuntimeEvent, 4U> events;
  auto drained = source.binding->drain(events);
  ASSERT_TRUE(drained.has_value());
  ASSERT_EQ(*drained, events.size());
  for (std::size_t index = 0U; index < events.size(); ++index) {
    EXPECT_EQ(events[index].source_sequence, index + 1U);
  }
  auto still_full = source.control.post_notification(at(40));
  ASSERT_FALSE(still_full.has_value());
  EXPECT_EQ(still_full.error().code, ErrorCode::kBackpressured);

  for (const auto& event : events) {
    ASSERT_TRUE(source.control.retire(event.source_sequence).has_value());
  }
  // Key assertion: capacity restores only after explicit retire; next post gets sequence 5.
  ASSERT_TRUE(source.control.post_notification(at(41)).has_value());
  detail::RuntimeEvent next;
  drained = source.binding->drain(std::span<detail::RuntimeEvent>{&next, 1U});
  ASSERT_TRUE(drained.has_value());
  ASSERT_EQ(*drained, 1U);
  EXPECT_EQ(next.source_sequence, 5U);
}

// REQ: RUNTIME-025
TEST(PhysicalReadinessBinding, CapacityOneConcurrentReusePreservesEveryNotification) {
  // Key setup: a producer continuously reuses one notification cell while a consumer drains and
  // retires it, repeatedly exercising the release-to-publish handoff at the same ring index.
  constexpr std::uint64_t kNotificationCount = 20'000U;
  auto source = bind(descriptor(18U, detail::RuntimeEventKind::kNotification, 1U));
  std::atomic_bool stop{};
  std::atomic<ErrorCode> first_error{ErrorCode::kSuccess};
  std::atomic<std::uint64_t> accepted{};
  std::atomic<std::uint64_t> retired{};

  std::thread producer([&] {
    while (!stop.load(std::memory_order_acquire) &&
           accepted.load(std::memory_order_relaxed) < kNotificationCount) {
      const auto posted = source.control.post_notification(at(100));
      if (posted) {
        accepted.fetch_add(1U, std::memory_order_relaxed);
      } else if (posted.error().code == ErrorCode::kBackpressured) {
        std::this_thread::yield();
      } else {
        first_error.store(posted.error().code, std::memory_order_release);
        stop.store(true, std::memory_order_release);
      }
    }
  });

  std::thread consumer([&] {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    std::uint64_t expected_sequence = 1U;
    while (!stop.load(std::memory_order_acquire) && expected_sequence <= kNotificationCount &&
           std::chrono::steady_clock::now() < deadline) {
      detail::RuntimeEvent event;
      const auto drained = source.binding->drain(std::span<detail::RuntimeEvent>{&event, 1U});
      if (!drained) {
        first_error.store(drained.error().code, std::memory_order_release);
        stop.store(true, std::memory_order_release);
        break;
      }
      if (*drained == 0U) {
        std::this_thread::yield();
        continue;
      }
      if (event.source_sequence != expected_sequence) {
        first_error.store(ErrorCode::kProtocolError, std::memory_order_release);
        stop.store(true, std::memory_order_release);
        break;
      }
      const auto released = source.control.retire(event.source_sequence);
      if (!released) {
        first_error.store(released.error().code, std::memory_order_release);
        stop.store(true, std::memory_order_release);
        break;
      }
      retired.fetch_add(1U, std::memory_order_relaxed);
      ++expected_sequence;
    }
    if (expected_sequence <= kNotificationCount) {
      stop.store(true, std::memory_order_release);
    }
  });

  consumer.join();
  stop.store(true, std::memory_order_release);
  producer.join();

  // Key assertion: no accepted cell disappears across repeated capacity-one retirement/reuse.
  EXPECT_EQ(first_error.load(std::memory_order_acquire), ErrorCode::kSuccess);
  EXPECT_EQ(accepted.load(std::memory_order_relaxed), kNotificationCount);
  EXPECT_EQ(retired.load(std::memory_order_relaxed), kNotificationCount);
  EXPECT_EQ(source.control.outstanding(), 0U);
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, DoorbellCoalescesEqualRejectsLowerAndPreservesTokenOnPressure) {
  // Key setup: kDoorbell with capacity=1; equal witness coalesces, lower is kProtocolError.
  auto source = bind(descriptor(3U, detail::RuntimeEventKind::kDoorbell, 1U));

  auto first = source.control.ring_doorbell(3U, at(30));
  ASSERT_TRUE(first.has_value());
  EXPECT_TRUE(first->witness_accepted);
  EXPECT_EQ(first->missed_count, 2U);

  auto equal = source.control.ring_doorbell(3U, at(31));
  ASSERT_TRUE(equal.has_value());
  EXPECT_TRUE(equal->coalesced);
  auto lower = source.control.ring_doorbell(2U, at(32));
  ASSERT_FALSE(lower.has_value());
  EXPECT_EQ(lower.error().code, ErrorCode::kProtocolError);

  auto full = source.control.ring_doorbell(4U, at(33));
  ASSERT_FALSE(full.has_value());
  EXPECT_EQ(full.error().code, ErrorCode::kBackpressured);

  detail::RuntimeEvent event;
  auto drained = source.binding->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained.has_value());
  ASSERT_EQ(*drained, 1U);
  ASSERT_TRUE(source.control.retire(event.source_sequence).has_value());

  // Key assertion: after retire, higher-witness doorbell succeeds and exposes previous_token.
  auto retried = source.control.ring_doorbell(4U, at(34));
  ASSERT_TRUE(retried.has_value());
  EXPECT_TRUE(retried->witness_accepted);
  EXPECT_EQ(retried->previous_token, 3U);
  EXPECT_EQ(retried->missed_count, 0U);
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, PollsSpinOnlyOnAdmittedBusyLaneAndRequiresFalseRearm) {
  // Key setup: kSpin requires WakeMode::kBusyPoll; sleeping waitset mode is rejected.
  // Rearm requires a false->true transition; continuous true does not rearm after retire.
  CallbackState callbacks;
  auto source_descriptor = descriptor(4U, detail::RuntimeEventKind::kSpin, 1U, &callbacks);

  auto invalid = source_descriptor;
  invalid.wake_mode = WakeMode::kSleepingWaitset;
  auto rejected = detail::PhysicalReadinessBindingFactory::create(
      std::span<const detail::PhysicalReadinessBindingDescriptor>{&invalid, 1U});
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);

  auto source = bind(source_descriptor);
  EXPECT_EQ(source.control.execution_lane(), 3U);
  EXPECT_EQ(source.control.wake_mode(), WakeMode::kBusyPoll);

  callbacks.asserted.store(false, std::memory_order_release);
  ASSERT_TRUE(source.binding->poll(at(50)).has_value());
  callbacks.asserted.store(true, std::memory_order_release);
  ASSERT_TRUE(source.binding->poll(at(51)).has_value());
  ASSERT_TRUE(source.binding->poll(at(52)).has_value());

  detail::RuntimeEvent event;
  auto drained = source.binding->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained.has_value());
  ASSERT_EQ(*drained, 1U);
  EXPECT_EQ(event.reference_time, at(51));
  ASSERT_TRUE(source.control.retire(event.source_sequence).has_value());

  // Continuous true does not rearm even after retirement.
  ASSERT_TRUE(source.binding->poll(at(53)).has_value());
  drained = source.binding->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained.has_value());
  EXPECT_EQ(*drained, 0U);

  callbacks.asserted.store(false, std::memory_order_release);
  ASSERT_TRUE(source.binding->poll(at(54)).has_value());
  callbacks.asserted.store(true, std::memory_order_release);
  ASSERT_TRUE(source.binding->poll(at(55)).has_value());
  drained = source.binding->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained.has_value());
  ASSERT_EQ(*drained, 1U);
  // Key assertion: exactly 6 total polls consumed; false->true transition fires once per cycle.
  EXPECT_EQ(event.source_sequence, 2U);
  EXPECT_EQ(callbacks.polls.load(std::memory_order_relaxed), 6U);
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, WarmedNotificationAndSpinPathsDoNotAllocate) {
  auto notification = bind(descriptor(8U, detail::RuntimeEventKind::kNotification, 1U));
  CallbackState callbacks;
  auto spin = bind(descriptor(9U, detail::RuntimeEventKind::kSpin, 1U, &callbacks));

  detail::RuntimeEvent notification_event;
  detail::RuntimeEvent spin_event;

  // Key setup: hot paths warmed through one full post->drain->retire cycle before measuring;
  // ScopedAllocationCounter then verifies zero heap activity on the measured second cycle.
  // Warm the exact accepted-through-retirement paths before measuring them.
  ASSERT_TRUE(notification.control.post_notification(at(70)).has_value());
  auto drained =
      notification.binding->drain(std::span<detail::RuntimeEvent>{&notification_event, 1U});
  ASSERT_TRUE(drained.has_value());
  ASSERT_EQ(*drained, 1U);
  ASSERT_TRUE(notification.control.retire(notification_event.source_sequence).has_value());
  callbacks.asserted.store(false, std::memory_order_release);
  ASSERT_TRUE(spin.binding->poll(at(71)).has_value());
  callbacks.asserted.store(true, std::memory_order_release);
  ASSERT_TRUE(spin.binding->poll(at(72)).has_value());
  drained = spin.binding->drain(std::span<detail::RuntimeEvent>{&spin_event, 1U});
  ASSERT_TRUE(drained.has_value());
  ASSERT_EQ(*drained, 1U);
  ASSERT_TRUE(spin.control.retire(spin_event.source_sequence).has_value());
  callbacks.asserted.store(false, std::memory_order_release);
  ASSERT_TRUE(spin.binding->poll(at(73)).has_value());

  bool operations_succeeded = false;
  std::size_t notification_count = 0U;
  std::size_t spin_count = 0U;
  std::size_t allocation_count = 0U;
  {
    const test::ScopedAllocationCounter allocations;
    const auto posted = notification.control.post_notification(at(74));
    const auto notification_drained =
        notification.binding->drain(std::span<detail::RuntimeEvent>{&notification_event, 1U});
    const auto notification_retired =
        notification.control.retire(notification_event.source_sequence);

    callbacks.asserted.store(true, std::memory_order_release);
    const auto spin_polled = spin.binding->poll(at(75));
    const auto spin_drained = spin.binding->drain(std::span<detail::RuntimeEvent>{&spin_event, 1U});
    const auto spin_retired = spin.control.retire(spin_event.source_sequence);

    notification_count = notification_drained ? *notification_drained : 0U;
    spin_count = spin_drained ? *spin_drained : 0U;
    operations_succeeded = posted.has_value() && notification_drained.has_value() &&
                           notification_retired.has_value() && spin_polled.has_value() &&
                           spin_drained.has_value() && spin_retired.has_value();
    allocation_count = allocations.count();
  }
  EXPECT_TRUE(operations_succeeded);
  EXPECT_EQ(notification_count, 1U);
  EXPECT_EQ(spin_count, 1U);
  // Key assertion: zero allocations across both notification and spin accept-retire paths.
  EXPECT_EQ(allocation_count, 0U);
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, RunSessionPollsOnSelectedLaneRetiresAndStopsCooperatively) {
  // Key setup: PollingPhysicalFactory injects busy-poll on the plan's execution lane;
  // alternate=true fires compute on every other poll, up to max_assertions=3 times.
  test::RuntimeProbeState::reset();
  CallbackState callbacks;
  callbacks.alternate = true;
  callbacks.max_assertions = 3U;
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  ASSERT_FALSE(baseline.execution_partitions().empty());
  auto factory = std::make_shared<PollingPhysicalFactory>(
      callbacks, baseline.execution_partitions().front().execution_lane);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_port_binding_factory(baseline, factory);

  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  std::vector<std::int32_t> events;
  std::vector<std::thread::id> event_threads;
  do {
    events = test::RuntimeProbeState::snapshot();
    event_threads = test::RuntimeProbeState::snapshot_threads();
    if (std::count(events.begin(), events.end(), 12) >= 3) {
      break;
    }
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < deadline && session.is_running());

  session.request_stop();
  session.wait();
  ASSERT_GE(std::count(events.begin(), events.end(), 12), 3);
  ASSERT_EQ(events.size(), event_threads.size());
  const auto compute = std::find(events.begin(), events.end(), 12);
  ASSERT_NE(compute, events.end());
  const std::size_t compute_index =
      static_cast<std::size_t>(std::distance(events.begin(), compute));
  // Key assertion: poll_thread_hash matches the compute event's thread (busy poll on same lane).
  EXPECT_EQ(callbacks.poll_thread_hash.load(std::memory_order_acquire),
            std::hash<std::thread::id>{}(event_threads[compute_index]));
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_EQ(factory->outstanding(), 0U);
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, RunSessionRejectsUnacceptedPollingLaneBeforeActivation) {
  // Key setup: factory substitutes lane 99 for the frozen request lane. Materialization must
  // reject that mismatch before the source can poll or enter user compute.
  test::RuntimeProbeState::reset();
  CallbackState callbacks;
  callbacks.asserted.store(true, std::memory_order_release);
  auto factory = std::make_shared<PollingPhysicalFactory>(callbacks, 99U);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_port_binding_factory(
      test::make_runtime_test_plan(), factory);

  RunSession session = run_async(plan);
  session.wait();
  // Key assertion: session fails before compute with RUNTIME_READINESS_BINDING_UNAVAILABLE.
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  const auto owner =
      std::ranges::find(plan.graph().vertices(), VertexKind::kOperator, &VertexRecord::kind);
  ASSERT_NE(owner, plan.graph().vertices().end());
  EXPECT_EQ(session.diagnostics().front().source.graph_generation, plan.graph().generation());
  EXPECT_EQ(session.diagnostics().front().source.vertex, owner->id);
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
  EXPECT_EQ(callbacks.polls.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(factory->outstanding(), 0U);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-035
TEST(PhysicalReadinessBinding, PollClockFailurePreservesRegisteredClockDiagnostic) {
  test::RuntimeProbeState::reset();
  CallbackState callbacks;
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  ASSERT_FALSE(baseline.execution_partitions().empty());
  // Key setup: inject a polling readiness binding whose clock lookup fails with the registered
  // clock-specific diagnostic before the operator can enter compute.
  auto factory = std::make_shared<PollingPhysicalFactory>(
      callbacks,
      baseline.execution_partitions().front().execution_lane,
      &poll_clock_binding_failure);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_port_binding_factory(baseline, factory);

  RunSession session = run_async(plan);
  session.wait();

  // Key assertion: Runtime publishes the precise clock-binding diagnostic, performs no compute,
  // and releases the failed physical binding.
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_CLOCK_BINDING_UNAVAILABLE");
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
  EXPECT_EQ(factory->outstanding(), 0U);
}

// REQ: RUNTIME-035
// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(PhysicalReadinessBinding, FreeFormBindingFailurePublishesRegisteredBoundedIdentity) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  auto factory = std::make_shared<FreeFormRejectingFactory>();
  auto summary_sink = std::make_shared<RecordingTerminalSummarySink>();
  const ExecutionPlan with_factory =
      detail::RuntimeTestPlanAccess::with_port_binding_factory(baseline, factory);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_factory, summary_sink);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const DiagnosticId expected_id = detail::diagnostic_id("RUNTIME_READINESS_BINDING_UNAVAILABLE");
  ASSERT_TRUE(expected_id);
  EXPECT_EQ(session.diagnostics().front().id, expected_id);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  const auto owner =
      std::ranges::find(plan.graph().vertices(), VertexKind::kOperator, &VertexRecord::kind);
  ASSERT_NE(owner, plan.graph().vertices().end());
  EXPECT_EQ(session.diagnostics().front().source.vertex, owner->id);

  const auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 1U);
  EXPECT_EQ(summaries.front().first_diagnostic, expected_id);
  EXPECT_EQ(summaries.front().first_token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, RunSessionReturnsCreditOnCallbackFailureAndCooperativeStop) {
  // Key setup: parametrized over kFail (compute returns error) and kWaitForStop (cooperative).
  for (const auto [mode, expected_termination] :
       {std::pair{test::RuntimeProbeMode::kFail, RunTermination::kFailed},
        std::pair{test::RuntimeProbeMode::kWaitForStop, RunTermination::kStopped}}) {
    test::RuntimeProbeState::reset(mode);
    CallbackState callbacks;
    callbacks.alternate = true;
    callbacks.max_assertions = 1U;
    const ExecutionPlan baseline = test::make_runtime_test_plan();
    ASSERT_FALSE(baseline.execution_partitions().empty());
    auto factory = std::make_shared<PollingPhysicalFactory>(
        callbacks, baseline.execution_partitions().front().execution_lane);
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::with_port_binding_factory(baseline, factory);

    RunSession session = run_async(plan);
    ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());
    if (mode == test::RuntimeProbeMode::kWaitForStop) {
      session.request_stop();
    }
    session.wait();
    EXPECT_EQ(session.termination(), expected_termination);
    // Key assertion: outstanding credit returns to 0 in both termination modes.
    EXPECT_EQ(factory->outstanding(), 0U);
  }
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, TerminalCancellationReleasesDrainedCreditExactlyOnce) {
  // Key setup: notification drained but not retired; cancel() is then called twice.
  auto source = bind(descriptor(10U, detail::RuntimeEventKind::kNotification, 1U));
  ASSERT_TRUE(source.control.post_notification(at(80)).has_value());
  detail::RuntimeEvent event;
  auto drained = source.binding->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained.has_value());
  ASSERT_EQ(*drained, 1U);
  ASSERT_EQ(source.control.outstanding(), 1U);

  source.binding->cancel();
  source.binding->cancel();
  // Key assertion: idempotent cancel() drops outstanding to 0; retire() returns kStaleEndpoint.
  EXPECT_EQ(source.control.outstanding(), 0U);
  auto retired_again = source.control.retire(event.source_sequence);
  ASSERT_FALSE(retired_again.has_value());
  EXPECT_EQ(retired_again.error().code, ErrorCode::kStaleEndpoint);
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, TerminalTransitionOwnsPublicationCreditExactlyOnce) {
  // Key setup: three race cases where a terminal action races an in-flight publisher
  // paused at a different PhysicalReadinessPublicationCheckpoint each time.
  enum class TerminalAction : std::uint8_t {
    kClose,
    kFail,
    kInvalidate,
  };
  struct RaceCase {
    detail::PhysicalReadinessPublicationCheckpoint checkpoint;
    TerminalAction action;
    ErrorCode expected_error;
  };
  constexpr std::array cases{
      RaceCase{detail::PhysicalReadinessPublicationCheckpoint::kPositionClaimed,
               TerminalAction::kClose,
               ErrorCode::kClosed},
      RaceCase{detail::PhysicalReadinessPublicationCheckpoint::kCreditReserved,
               TerminalAction::kFail,
               ErrorCode::kProtocolError},
      RaceCase{detail::PhysicalReadinessPublicationCheckpoint::kReadyPublished,
               TerminalAction::kInvalidate,
               ErrorCode::kStaleEndpoint},
  };

  for (const RaceCase& race : cases) {
    SCOPED_TRACE(static_cast<unsigned>(race.checkpoint));
    PublicationBarrier barrier{race.checkpoint};
    auto requested = descriptor(12U, detail::RuntimeEventKind::kNotification, 1U);
    requested.test_publish_state = &barrier;
    requested.test_publish_checkpoint = &publication_checkpoint;
    auto source = bind(requested);
    ErrorCode producer_error{ErrorCode::kSuccess};
    std::thread producer([&] {
      auto published = source.control.post_notification(at(100));
      producer_error = published ? ErrorCode::kSuccess : published.error().code;
    });
    const bool entered = barrier.wait_until_entered();
    if (!entered) {
      barrier.release();
      producer.join();
      FAIL() << "publisher did not reach the requested checkpoint";
    }

    switch (race.action) {
      case TerminalAction::kClose:
        source.control.close();
        break;
      case TerminalAction::kFail:
        source.control.fail(ErrorCode::kProtocolError);
        break;
      case TerminalAction::kInvalidate:
        source.control.invalidate_stale();
        break;
    }
    barrier.release();
    producer.join();

    // Key assertion: producer gets the action-specific error code; credit is 0 afterwards.
    EXPECT_EQ(producer_error, race.expected_error);
    EXPECT_EQ(source.control.outstanding(), 0U);
    EXPECT_TRUE(source.binding->closed());
  }
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, GenerationAdvanceRejectsAnInFlightPublisherWithoutLeakingState) {
  // Key setup: producer paused at kPublisherEntered; advance_source_generation() attempted
  // concurrently while the producer holds an in-flight position.
  PublicationBarrier barrier{detail::PhysicalReadinessPublicationCheckpoint::kPublisherEntered};
  auto requested = descriptor(13U, detail::RuntimeEventKind::kNotification, 1U);
  requested.test_publish_state = &barrier;
  requested.test_publish_checkpoint = &publication_checkpoint;
  auto source = bind(requested);
  ErrorCode producer_error{ErrorCode::kFailure};
  std::thread producer([&] {
    auto published = source.control.post_notification(at(110));
    producer_error = published ? ErrorCode::kSuccess : published.error().code;
  });
  const bool entered = barrier.wait_until_entered();
  if (!entered) {
    barrier.release();
    producer.join();
    FAIL() << "publisher did not reach the requested checkpoint";
  }

  auto blocked = source.binding->advance_source_generation();
  ASSERT_FALSE(blocked.has_value());
  EXPECT_EQ(blocked.error().code, ErrorCode::kNotReady);
  barrier.release();
  producer.join();
  ASSERT_EQ(producer_error, ErrorCode::kSuccess);

  detail::RuntimeEvent event;
  auto drained = source.binding->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained.has_value());
  ASSERT_EQ(*drained, 1U);
  ASSERT_TRUE(source.control.retire(event.source_sequence).has_value());
  EXPECT_EQ(source.control.outstanding(), 0U);

  // Key assertion: advance succeeds after quiescence and returns generation 2; no leaked credit.
  auto advanced = source.binding->advance_source_generation();
  ASSERT_TRUE(advanced.has_value());
  EXPECT_EQ(*advanced, 2U);
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, RestartRequiresQuiescenceAndAdvancesExactSourceGeneration) {
  // Key setup: advance_source_generation() blocked by a drained-but-unretired event;
  // advances only after retire(), at which point old control handle is kStaleEndpoint.
  auto source = bind(descriptor(11U, detail::RuntimeEventKind::kNotification, 1U));
  ASSERT_TRUE(source.control.post_notification(at(90)).has_value());
  detail::RuntimeEvent event;
  auto drained = source.binding->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained.has_value());
  ASSERT_EQ(*drained, 1U);
  ASSERT_EQ(event.source_generation, 1U);

  auto blocked = source.binding->advance_source_generation();
  ASSERT_FALSE(blocked.has_value());
  EXPECT_EQ(blocked.error().code, ErrorCode::kNotReady);
  ASSERT_TRUE(source.control.retire(event.source_sequence).has_value());

  auto advanced = source.binding->advance_source_generation();
  ASSERT_TRUE(advanced.has_value());
  ASSERT_EQ(*advanced, 2U);
  auto old_post = source.control.post_notification(at(91));
  ASSERT_FALSE(old_post.has_value());
  EXPECT_EQ(old_post.error().code, ErrorCode::kStaleEndpoint);
  auto old_lookup = source.factory->control(detail::make_source_id(11U), 1U, 1U, 1U);
  ASSERT_FALSE(old_lookup.has_value());
  EXPECT_EQ(old_lookup.error().code, ErrorCode::kStaleEndpoint);

  auto current = source.factory->control(detail::make_source_id(11U), 1U, 1U, 2U);
  ASSERT_TRUE(current.has_value());
  ASSERT_TRUE(current->post_notification(at(92)).has_value());
  drained = source.binding->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained.has_value());
  ASSERT_EQ(*drained, 1U);
  // Key assertion: new generation drain sees source_generation=2 with sequence restarting at 1.
  EXPECT_EQ(event.source_generation, 2U);
  EXPECT_EQ(event.source_sequence, 1U);
  ASSERT_TRUE(current->retire(event.source_sequence).has_value());
  EXPECT_EQ(current->outstanding(), 0U);
}

// REQ: RUNTIME-007
TEST(PhysicalReadinessBinding, DistinguishesFatalFailureStaleGenerationAndNormalClose) {
  // Key setup: three bindings closed via fail(), invalidate_stale(), and close() respectively;
  // each produces a distinct error code on the next publish or drain attempt.
  auto failed = bind(descriptor(5U, detail::RuntimeEventKind::kNotification, 1U));
  failed.control.fail(ErrorCode::kProtocolError);
  detail::RuntimeEvent event;
  auto drained = failed.binding->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_FALSE(drained.has_value());
  EXPECT_EQ(drained.error().code, ErrorCode::kProtocolError);
  EXPECT_TRUE(failed.binding->closed());

  auto stale = bind(descriptor(6U, detail::RuntimeEventKind::kNotification, 1U));
  stale.control.invalidate_stale();
  auto post = stale.control.post_notification(at(60));
  ASSERT_FALSE(post.has_value());
  EXPECT_EQ(post.error().code, ErrorCode::kStaleEndpoint);

  auto closed = bind(descriptor(7U, detail::RuntimeEventKind::kNotification, 1U));
  const std::uint64_t epoch = closed.control.wake_epoch();
  closed.control.close();
  post = closed.control.post_notification(at(61));
  ASSERT_FALSE(post.has_value());
  // Key assertion: close() returns kClosed (not kStaleEndpoint/kProtocolError) and bumps epoch.
  EXPECT_EQ(post.error().code, ErrorCode::kClosed);
  EXPECT_GT(closed.control.wake_epoch(), epoch);
}

}  // namespace
}  // namespace holoscan
