/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef __linux__
#include <sched.h>
#endif

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/threading/thread_placement.hpp>

namespace holoscan {
namespace {

// THREAD-PLACE-001: placement declarations are plain values, not runtime or GXF components.
static_assert(std::is_copy_constructible_v<ThreadPlacement>);
static_assert(std::is_move_constructible_v<ThreadPlacement>);
static_assert(std::is_copy_assignable_v<ThreadPlacement>);
static_assert(!std::is_polymorphic_v<ThreadPlacement>);
static_assert(!std::is_polymorphic_v<CPUAffinity>);
static_assert(!std::is_polymorphic_v<ThreadScheduling>);
static_assert(!std::is_polymorphic_v<DeadlineScheduling>);

// THREAD-PLACE-002: CPU affinity accepts core lists and inclusive ranges.
TEST(ThreadAffinity, ParsesListAndRanges) {
  const std::array<uint32_t, 2> cores{7, 2};
  const std::array<CPUCoreRange, 2> ranges{CPUCoreRange{4, 6}, CPUCoreRange{1, 2}};

  auto affinity = CPUAffinity::from_cores_and_ranges(cores, ranges);

  ASSERT_TRUE(affinity) << runtime_error_message(affinity.error());
  EXPECT_EQ(affinity->cores, (std::vector<uint32_t>{1, 2, 4, 5, 6, 7}));
}

// THREAD-PLACE-002: CPU affinity canonicalizes downstream state.
TEST(ThreadAffinity, NormalizesSortedUnique) {
  const CPUAffinity affinity{5, 3, 5, 1, 3};

  EXPECT_EQ(affinity.cores, (std::vector<uint32_t>{1, 3, 5}));
}

TEST(ThreadAffinity, RejectsInvalidRange) {
  auto affinity = CPUAffinity::from_ranges({CPUCoreRange{4, 3}});

  EXPECT_FALSE(affinity);
  EXPECT_EQ(affinity.error().code, ErrorCode::kInvalidArgument);
}

TEST(ThreadAffinity, RejectsExcessiveRangeExpansion) {
  auto affinity = CPUAffinity::from_ranges({CPUCoreRange{0, 1'048'576}});

  EXPECT_FALSE(affinity);
  EXPECT_EQ(affinity.error().code, ErrorCode::kInvalidArgument);
}

// THREAD-PLACE-003: present-but-empty affinity is a configuration error.
TEST(ThreadPlacement, RejectsEmptyAffinity) {
  const ThreadPlacement placement{.affinity = CPUAffinity{}};

  auto result = placement.validate();

  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

// THREAD-PLACE-004: FIFO/RR are priority driven.
TEST(ThreadPlacement, FifoRequiresPriority) {
  const ThreadPlacement placement{.scheduling = ThreadScheduling{
                                      .policy = SchedulingPolicy::kFIFO,
                                  }};

  auto result = placement.validate();

  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

TEST(ThreadPlacement, FifoAcceptsPriorityWithinLinuxRange) {
#ifdef __linux__
  const ThreadPlacement placement{.scheduling = ThreadScheduling{
                                      .policy = SchedulingPolicy::kFIFO,
                                      .priority = sched_get_priority_min(SCHED_FIFO),
                                  }};

  auto result = placement.validate();

  EXPECT_TRUE(result) << runtime_error_message(result.error());
#else
  GTEST_SKIP() << "Linux priority bounds are only available on Linux";
#endif
}

// THREAD-PLACE-004: deadline fields do not apply to FIFO/RR.
TEST(ThreadPlacement, FifoRejectsDeadlineFields) {
  const ThreadPlacement placement{
      .scheduling = ThreadScheduling{
          .policy = SchedulingPolicy::kFIFO,
          .priority = 1,
          .deadline = DeadlineScheduling{.runtime_ns = 1, .deadline_ns = 2, .period_ns = 3},
      }};

  auto result = placement.validate();

  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

// THREAD-PLACE-005: deadline tuple fields are mandatory and ordered.
TEST(ThreadPlacement, DeadlineRequiresOrderedNonZeroTuple) {
  // Key setup: three invalid tuples cover zero runtime, runtime > deadline, and
  //            deadline > period; a valid tuple confirms the acceptance boundary on Linux.
  const ThreadPlacement zero_tuple{
      .scheduling = ThreadScheduling{
          .policy = SchedulingPolicy::kDeadline,
          .deadline = DeadlineScheduling{.runtime_ns = 0, .deadline_ns = 2, .period_ns = 3},
      }};
  // Key assertion: each invalid ordering returns kInvalidArgument.
  EXPECT_FALSE(zero_tuple.validate());

  const ThreadPlacement runtime_after_deadline{
      .scheduling = ThreadScheduling{
          .policy = SchedulingPolicy::kDeadline,
          .deadline = DeadlineScheduling{.runtime_ns = 3, .deadline_ns = 2, .period_ns = 4},
      }};
  EXPECT_FALSE(runtime_after_deadline.validate());

  const ThreadPlacement deadline_after_period{
      .scheduling = ThreadScheduling{
          .policy = SchedulingPolicy::kDeadline,
          .deadline = DeadlineScheduling{.runtime_ns = 2, .deadline_ns = 4, .period_ns = 3},
      }};
  EXPECT_FALSE(deadline_after_period.validate());

#ifdef __linux__
  const ThreadPlacement valid_deadline{
      .scheduling = ThreadScheduling{
          .policy = SchedulingPolicy::kDeadline,
          .deadline = DeadlineScheduling{.runtime_ns = 1, .deadline_ns = 2, .period_ns = 3},
      }};
  EXPECT_TRUE(valid_deadline.validate());
#endif
}

// THREAD-PLACE-005: deadline scheduling uses the deadline tuple, not FIFO/RR priority.
TEST(ThreadPlacement, DeadlineRejectsPriority) {
  const ThreadPlacement placement{
      .scheduling = ThreadScheduling{
          .policy = SchedulingPolicy::kDeadline,
          .priority = 1,
          .deadline = DeadlineScheduling{.runtime_ns = 1, .deadline_ns = 2, .period_ns = 3},
      }};

  auto result = placement.validate();

  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

// THREAD-PLACE-006: Linux can reject priority values outside the native FIFO/RR range up front.
TEST(ThreadPlacement, RejectsOutOfRangePriority) {
#ifdef __linux__
  const int invalid_priority = sched_get_priority_max(SCHED_FIFO) + 1;
  const ThreadPlacement placement{.scheduling = ThreadScheduling{
                                      .policy = SchedulingPolicy::kFIFO,
                                      .priority = invalid_priority,
                                  }};

  auto result = placement.validate();

  EXPECT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
#else
  GTEST_SKIP() << "Linux priority bounds are only available on Linux";
#endif
}

// THREAD-PLACE-007: the default placement is an explicit no-op.
TEST(ThreadPlacement, DefaultPlacementIsValid) {
  const ThreadPlacement placement;

  auto result = placement.validate();

  EXPECT_TRUE(result);
}

TEST(ThreadPlacement, RejectsEveryUnknownClosedEnumValue) {
  // Deliberately construct out-of-domain values to exercise closed-enum validation.
  // static_cast is used because no valid code path produces these values; the validator
  // must reject each closed-enum field independently.
  // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto kUnknownScheduling = static_cast<SchedulingPolicy>(0x7F);
  constexpr auto kUnknownWake = static_cast<WakeMode>(0x7F);
  constexpr auto kUnknownIsolation = static_cast<IRQIsolation>(0x7F);
  constexpr auto kUnknownFailurePolicy = static_cast<ProbeFailurePolicy>(0x7F);
  // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange)

  // Key assertion: every closed enum field rejects an unknown value from validate().
  EXPECT_FALSE(
      ThreadPlacement{.scheduling = ThreadScheduling{.policy = kUnknownScheduling}}.validate());
  EXPECT_FALSE(ThreadPlacement{.wake_mode = kUnknownWake}.validate());
  EXPECT_FALSE(ThreadPlacement{.irq_isolation = kUnknownIsolation}.validate());
  EXPECT_FALSE(ThreadPlacement{.probe_failure_policy = kUnknownFailurePolicy}.validate());
}

TEST(ThreadPlacement, OtherRejectsRealtimeFields) {
  const ThreadPlacement priority{
      .scheduling = ThreadScheduling{.policy = SchedulingPolicy::kOther, .priority = 1}};
  EXPECT_FALSE(priority.validate());

  const ThreadPlacement deadline{
      .scheduling = ThreadScheduling{
          .policy = SchedulingPolicy::kOther,
          .deadline = DeadlineScheduling{.runtime_ns = 1, .deadline_ns = 2, .period_ns = 3},
      }};
  EXPECT_FALSE(deadline.validate());
}

TEST(ThreadPlacement, NonLinuxRejectsRealtimeScheduling) {
#ifdef __linux__
  GTEST_SKIP() << "Non-Linux validation behavior is only asserted on non-Linux builds";
#else
  EXPECT_TRUE(ThreadPlacement{}.validate());

  const ThreadPlacement fifo{
      .scheduling = ThreadScheduling{.policy = SchedulingPolicy::kFIFO, .priority = 1}};
  EXPECT_FALSE(fifo.validate());

  const ThreadPlacement deadline{
      .scheduling = ThreadScheduling{
          .policy = SchedulingPolicy::kDeadline,
          .deadline = DeadlineScheduling{.runtime_ns = 1, .deadline_ns = 2, .period_ns = 3},
      }};
  EXPECT_FALSE(deadline.validate());
#endif
}

}  // namespace
}  // namespace holoscan
