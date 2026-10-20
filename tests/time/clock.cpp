/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>

#include <holoscan/time/clock.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/time/synthetic_clock.hpp>

#include "../../src/time/clock_internal.hpp"
#include "../../src/time/synthetic_clock_test_access.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

template <typename T>
concept HasGxfComponentLifecycle = requires(T value) {
  value.registerInterface();
  value.initialize();
  value.deinitialize();
};

// REQ: CLOCK-001
// The native clock facade must not depend on the GXF-compatible Component lifecycle.
static_assert(std::is_base_of_v<Clock, RealtimeClock>);
static_assert(std::is_base_of_v<Clock, ManualClock>);
static_assert(std::is_base_of_v<Clock, SyntheticClock>);
static_assert(!HasGxfComponentLifecycle<Clock>);
static_assert(!HasGxfComponentLifecycle<RealtimeClock>);
static_assert(!HasGxfComponentLifecycle<ManualClock>);
static_assert(!HasGxfComponentLifecycle<SyntheticClock>);
static_assert(std::same_as<decltype(std::declval<const Clock&>().time()), double>);
static_assert(std::same_as<decltype(std::declval<const Clock&>().timestamp()), std::int64_t>);
static_assert(time_to_timestamp(0.0) == 0);
static_assert(time_to_timestamp(1.5) == 1'500'000'000);
static_assert(time_to_timestamp(-7.7e-9) == -7);
static_assert(time_to_timestamp(1.0e30) == std::numeric_limits<int64_t>::max());
static_assert(time_to_timestamp(-1.0e30) == std::numeric_limits<int64_t>::min());
static_assert(time_to_timestamp(std::numeric_limits<double>::quiet_NaN()) == 0);
static_assert(timestamp_to_time(2'500'000'000) == 2.5);
static_assert(detail::saturating_add(40, 2) == 42);
static_assert(detail::saturating_add(std::numeric_limits<int64_t>::max() - 1, 2) ==
              std::numeric_limits<int64_t>::max());
static_assert(detail::saturating_add(std::numeric_limits<int64_t>::min() + 1, -2) ==
              std::numeric_limits<int64_t>::min());

// REQ: CLOCK-003
// Second/nanosecond conversions must agree with the clock accessors.
TEST(Clock, TimeTimestampRoundTrip) {
  EXPECT_EQ(time_to_timestamp(1.5), 1'500'000'000);
  EXPECT_DOUBLE_EQ(timestamp_to_time(2'500'000'000), 2.5);
  EXPECT_EQ(time_to_timestamp(timestamp_to_time(1'234'000'000)), 1'234'000'000);
}

// REQ: CLOCK-008
// time_to_timestamp truncates toward zero and clamps out-of-range inputs instead of
// invoking undefined behavior.
TEST(Clock, TimeToTimestampTruncatesAndClamps) {
  // Truncates toward zero (sub-nanosecond rounding is intentionally not performed).
  EXPECT_EQ(time_to_timestamp(7.7e-9), 7);
  EXPECT_EQ(time_to_timestamp(-7.7e-9), -7);

  EXPECT_EQ(time_to_timestamp(1.0e30), std::numeric_limits<int64_t>::max());
  EXPECT_EQ(time_to_timestamp(-1.0e30), std::numeric_limits<int64_t>::min());
}

// REQ: CLOCK-RT-001
// Real time advances from the configured offset and stays monotonic.
TEST(RealtimeClock, AdvancesFromOffsetAndIsMonotonic) {
  const RealtimeClock clock(RealtimeClock::Config{.initial_time_offset = 100.0});

  const double first = clock.time();
  EXPECT_GE(first, 100.0);

  const double second = clock.time();
  EXPECT_GE(second, first);
  EXPECT_GE(clock.timestamp(), time_to_timestamp(100.0));
}

// REQ: CLOCK-RT-004
// Immutable realtime-clock reads remain race-free and monotonic for each concurrent observer.
TEST(RealtimeClock, ConcurrentReadsRemainMonotonic) {
  const RealtimeClock clock;
  std::atomic<bool> first_monotonic{true};
  std::atomic<bool> second_monotonic{true};

  const auto read_repeatedly = [&clock](std::atomic<bool>& monotonic) {
    auto previous = clock.timestamp();
    for (int index = 0; index < 10'000; ++index) {
      const auto current = clock.timestamp();
      if (current < previous) {
        monotonic.store(false, std::memory_order_relaxed);
      }
      previous = current;
    }
  };

  std::thread first(read_repeatedly, std::ref(first_monotonic));
  std::thread second(read_repeatedly, std::ref(second_monotonic));
  first.join();
  second.join();

  EXPECT_TRUE(first_monotonic.load(std::memory_order_relaxed));
  EXPECT_TRUE(second_monotonic.load(std::memory_order_relaxed));
}

// REQ: CLOCK-RT-002
// With use_time_since_epoch the clock is aligned to wall-clock time.
TEST(RealtimeClock, UsesTimeSinceEpoch) {
  const RealtimeClock clock(RealtimeClock::Config{.use_time_since_epoch = true});

  const double clock_time = clock.time();
  const double wall_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
  EXPECT_NEAR(clock_time, wall_time, 1.0);
}

// REQ: CLOCK-RT-006
// At epoch scale, timestamp() is computed in integer nanoseconds (the offset is kept
// out of the floating-point path), so it tracks wall-clock nanoseconds tightly rather than being
// quantized by double's ~microsecond resolution at epoch magnitude.
TEST(RealtimeClock, TimestampReportsEpochNanoseconds) {
  const int64_t wall_before_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count();
  const RealtimeClock clock(RealtimeClock::Config{.use_time_since_epoch = true});
  const int64_t first = clock.timestamp();
  const int64_t second = clock.timestamp();
  const int64_t wall_after_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count();

  // The clock's nanosecond timestamp sits within the wall-clock window around construction (small
  // slack for scheduling), confirming a correct epoch-scale ns value rather than a lossy double.
  constexpr int64_t kSlackNs = 100'000'000;  // 100 ms
  EXPECT_GE(first, wall_before_ns - kSlackNs);
  EXPECT_LE(first, wall_after_ns + kSlackNs);
  EXPECT_GE(second, first);
}

// REQ: CLOCK-RT-003
// REQ: CLOCK-RT-008
// A non-positive initial scale is clamped to 1.0 so the clock still moves forward.
TEST(RealtimeClock, RejectsNonPositiveScale) {
  const RealtimeClock clock(RealtimeClock::Config{.initial_time_scale = -2.0});

  // Clamped to 1.0, so time keeps moving forward rather than backward or stalling.
  const double first = clock.time();
  const double second = clock.time();
  EXPECT_GE(second, first);
}

// REQ: CLOCK-002
// REQ: CLOCK-RT-005
// The real-time clock cannot sleep for a negative duration.
TEST(RealtimeClock, RejectsNegativeSleep) {
  RealtimeClock clock;
  EXPECT_FALSE(clock.sleep_for(-1));
  EXPECT_TRUE(clock.sleep_for(0));
}

// REQ: CLOCK-RT-007
// sleep_until() with a target at or before the current time is a no-op success (a
// missed deadline should let the caller proceed, not error).
TEST(RealtimeClock, SleepUntilPastReturnsImmediately) {
  RealtimeClock clock;
  const auto before = std::chrono::steady_clock::now();
  auto result = clock.sleep_until(clock.timestamp() - 1'000'000'000);  // 1 s in the past
  const auto elapsed = std::chrono::steady_clock::now() - before;

  EXPECT_TRUE(result) << runtime_error_message(result.error());
  EXPECT_LT(elapsed, std::chrono::milliseconds(100));  // returned without sleeping
}

// REQ: CLOCK-002
// REQ: CLOCK-MAN-001
// Sleeping advances the manual clock monotonically and never moves it backwards.
TEST(ManualClock, SleepAdvancesMonotonically) {
  ManualClock clock;
  EXPECT_EQ(clock.timestamp(), 0);

  ASSERT_TRUE(clock.sleep_for(100));
  EXPECT_EQ(clock.timestamp(), 100);

  auto backward = clock.sleep_until(50);
  EXPECT_FALSE(backward);
  EXPECT_EQ(backward.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(clock.timestamp(), 100);

  ASSERT_TRUE(clock.sleep_until(200));
  EXPECT_EQ(clock.timestamp(), 200);
}

// REQ: CLOCK-MAN-002
// advance_to / advance_by give direct, unrestricted manual control.
TEST(ManualClock, AdvanceControlsTimeDirectly) {
  ManualClock clock(2'000);
  EXPECT_EQ(clock.timestamp(), 2'000);

  clock.advance_by(100);
  EXPECT_EQ(clock.timestamp(), 2'100);

  clock.advance_by(100ns);
  EXPECT_EQ(clock.timestamp(), 2'200);

  // advance_to may move the clock backwards, unlike sleep.
  clock.advance_to(500);
  EXPECT_EQ(clock.timestamp(), 500);
  EXPECT_DOUBLE_EQ(clock.time(), timestamp_to_time(500));
}

// REQ: CLOCK-SYN-002
// A synthetic clock can be set directly when no one is waiting (GXF SimpleAdvance).
TEST(SyntheticClock, SimpleAdvance) {
  SyntheticClock clock;
  EXPECT_EQ(clock.timestamp(), 0);
  clock.advance_to(1'000);
  EXPECT_EQ(clock.timestamp(), 1'000);
}

// REQ: CLOCK-008
// Advancing past the representable range saturates instead of overflowing (UB).
TEST(SyntheticClock, AdvanceSaturatesInsteadOfOverflowing) {
  SyntheticClock clock(std::numeric_limits<int64_t>::max() - 100);
  clock.advance_by(1'000);
  EXPECT_EQ(clock.timestamp(), std::numeric_limits<int64_t>::max());
}

// REQ: CLOCK-SYN-001
// REQ: CLOCK-SYN-002
// A sleeper blocks until another thread advances past its target
// (adapted from the GXF ParallelAdvance / CompressesTime tests).
TEST(SyntheticClock, SleepBlocksUntilAdvanced) {
  SyntheticClock clock;
  clock.advance_to(23);

  std::atomic<bool> woke{false};
  bool slept_ok = false;
  int64_t observed = -1;

  std::thread waiter([&] {
    auto result = clock.sleep_until(42);
    observed = clock.timestamp();
    slept_ok = result.has_value();
    woke.store(true);
  });

  // Give the waiter time to block; it must not wake until the clock is advanced.
  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(woke.load());

  clock.advance_to(42);
  waiter.join();

  EXPECT_TRUE(slept_ok);
  EXPECT_GE(observed, 42);
}

// REQ: CLOCK-SYN-001
// Sleeping returns immediately when the clock has already passed the target time
// (adapted from the GXF AlreadyAdvanced / Overshooting tests).
TEST(SyntheticClock, AlreadyAdvancedSleepReturnsImmediately) {
  constexpr int64_t kTargetTime = 123'456'789;
  SyntheticClock clock;
  clock.advance_by(kTargetTime * 5);

  auto result = clock.sleep_until(kTargetTime);
  EXPECT_TRUE(result);
  EXPECT_GE(clock.timestamp(), kTargetTime);
}

// REQ: CLOCK-SYN-003
// stop() releases blocked sleepers that have not reached their target time, so the
// clock can be torn down without deadlocking waiting threads.
TEST(SyntheticClock, StopWakesSleepers) {
  SyntheticClock clock;

  std::atomic<bool> woke{false};
  bool got_error = false;
  ErrorCode code = ErrorCode::kSuccess;

  std::thread waiter([&] {
    auto result = clock.sleep_until(1'000'000);
    if (!result) {
      got_error = true;
      code = result.error().code;
    }
    woke.store(true);
  });

  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(woke.load());

  clock.stop();
  waiter.join();

  EXPECT_TRUE(got_error);
  EXPECT_EQ(code, ErrorCode::kClosed);
}

// REQ: CLOCK-SYN-003
// Destroying the clock while a sleeper is blocked must wake the sleeper and wait for
// it to leave its wait before the synchronization primitives are torn down (no use-after-free, no
// deadlock).
TEST(SyntheticClock, DestructorDrainsBlockedSleeper) {
  auto clock = std::make_unique<SyntheticClock>();

  std::atomic<bool> woke{false};
  bool got_closed = false;

  // Capture the clock itself, not the unique_ptr that reset() below rewrites.
  std::thread waiter([&woke, &got_closed, sleeper = clock.get()] {
    auto result = sleeper->sleep_until(1'000'000);
    got_closed = !result && result.error().code == ErrorCode::kClosed;
    woke.store(true);
  });

  // Key setup: destroy the clock only after the sleeper has registered inside its wait. Starting a
  // sleep concurrently with destruction is outside the clock's contract.
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (detail::SyntheticClockTestAccess::active_sleepers(*clock) == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  if (detail::SyntheticClockTestAccess::active_sleepers(*clock) == 0) {
    clock->stop();
    waiter.join();
    FAIL() << "the sleeper did not register within 5 s";
  }
  EXPECT_FALSE(woke.load());

  // The destructor (via stop()) must wake and drain the in-flight sleeper.
  clock.reset();
  waiter.join();

  EXPECT_TRUE(got_closed);
}

}  // namespace
}  // namespace holoscan
