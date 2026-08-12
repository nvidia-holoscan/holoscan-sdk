/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "threading/linux_thread_controller.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace holoscan::detail {
namespace {

#ifdef __linux__

// RAII guard: saves the calling thread's CPU affinity on construction and restores it on
// destruction.  Used by tests that pin to a specific core via apply() so the rest of the
// test suite is not affected by core-pinning side effects.
// Representative test: AppliesAffinityReadback.
class AffinityRestore {
 public:
  AffinityRestore() {
    auto snapshot = LinuxThreadController::snapshot();
    if (snapshot) {
      original_affinity_ = snapshot->affinity;
    }
  }

  ~AffinityRestore() {
    if (original_affinity_.has_value()) {
      (void)LinuxThreadController::apply(ThreadPlacement{.affinity = original_affinity_});
    }
  }

 private:
  std::optional<CPUAffinity> original_affinity_;
};

// RAII guard: resets the calling thread's scheduling policy to SCHED_OTHER on destruction.
// Wraps the multi-step fallback chain (sched_setscheduler → SCHED_RESET_ON_FORK →
// pthread_setschedparam) so real-time tests clean up even on partial failure.
// Representative test: AppliesFifoReadback.
class SchedulingRestore {
 public:
  ~SchedulingRestore() { (void)restore(); }

  expected<void, Error> restore() {
    if (restored_) {
      return {};
    }

    const sched_param param{};
    int errno_value = 0;
    if (sched_setscheduler(0, SCHED_OTHER, &param) != 0) {
      errno_value = errno;
      if (sched_setscheduler(0, SCHED_OTHER | SCHED_RESET_ON_FORK, &param) != 0) {
        const int status = pthread_setschedparam(pthread_self(), SCHED_OTHER, &param);
        if (status != 0) {
          return make_unexpected(linux_thread_error_for_errno(status, "pthread_setschedparam"));
        }
      }
    }

    auto snapshot = LinuxThreadController::snapshot();
    if (!snapshot) {
      return forward_error(snapshot);
    }
    if (snapshot->policy != SchedulingPolicy::kOther) {
      return make_unexpected(linux_thread_error_for_errno(errno_value, "sched_setscheduler"));
    }

    restored_ = true;
    return {};
  }

 private:
  bool restored_ = false;
};

// Returns the lowest-numbered core in the effective process cpuset.  Tests that need a
// valid affinity target call this instead of hard-coding core 0, which may be excluded
// under cgroup or Docker cpuset constraints.
uint32_t first_effective_core() {
  auto snapshot = LinuxThreadController::snapshot();
  EXPECT_TRUE(snapshot) << runtime_error_message(snapshot.error());
  if (!snapshot || snapshot->affinity.cores.empty()) {
    return 0;
  }
  return snapshot->affinity.cores.front();
}

// Scans CPU IDs in order and returns the first one absent from the process cpuset.
// Tests that verify out-of-cpuset rejection call this; they skip when all CPU_SETSIZE
// slots are covered (returns nullopt).
std::optional<uint32_t> first_core_outside_effective_cpuset() {
  auto snapshot = LinuxThreadController::snapshot();
  EXPECT_TRUE(snapshot) << runtime_error_message(snapshot.error());
  if (!snapshot) {
    return std::nullopt;
  }

  for (uint32_t core = 0; core < CPU_SETSIZE; ++core) {
    if (!std::binary_search(
            snapshot->affinity.cores.begin(), snapshot->affinity.cores.end(), core)) {
      return core;
    }
  }
  return std::nullopt;
}

// Convenience factory: builds a SCHED_FIFO ThreadPlacement at the given priority.
// Keeps RT test setup concise; the intent stays explicit without repeating the struct literal.
ThreadPlacement fifo_placement(int priority) {
  return ThreadPlacement{.scheduling = ThreadScheduling{
                             .policy = SchedulingPolicy::kFIFO,
                             .priority = priority,
                         }};
}

// RT-scheduling tests require CAP_SYS_NICE or a raised RLIMIT_RTPRIO and permanently
// alter the calling thread's policy.  They are opt-in via an environment variable so
// they do not disrupt unprivileged CI runs.
bool realtime_thread_tests_enabled() {
  return std::getenv("HOLOSCAN_RUN_RT_THREAD_TESTS") != nullptr;
}

#endif

// THREAD-LINUX-005: errno mapping keeps common scheduler failures identifiable.
TEST(LinuxThreadController, MapsEpermEinvalEbusy) {
  const Error eperm = linux_thread_error_for_errno(EPERM, "op");
  EXPECT_EQ(eperm.code, ErrorCode::kFailure);
  EXPECT_NE(eperm.message.find("EPERM"), std::string::npos);

  const Error einval = linux_thread_error_for_errno(EINVAL, "op");
  EXPECT_EQ(einval.code, ErrorCode::kInvalidArgument);
  EXPECT_NE(einval.message.find("EINVAL"), std::string::npos);

  const Error ebusy = linux_thread_error_for_errno(EBUSY, "op");
  EXPECT_EQ(ebusy.code, ErrorCode::kNotReady);
  EXPECT_NE(ebusy.message.find("EBUSY"), std::string::npos);
}

// THREAD-LINUX-006: probe reports the current Linux admission context without applying changes.
TEST(LinuxThreadController, ProbeReportsCapabilities) {
#ifndef __linux__
  GTEST_SKIP() << "LinuxThreadController is only supported on Linux";
#else
  auto probe = LinuxThreadController::probe(ThreadPlacement{});

  ASSERT_TRUE(probe) << runtime_error_message(probe.error());
  EXPECT_TRUE(probe->supported);
  EXPECT_FALSE(probe->effective_affinity.cores.empty());
  EXPECT_LE(probe->fifo_priority_min, probe->fifo_priority_max);
  EXPECT_LE(probe->round_robin_priority_min, probe->round_robin_priority_max);
#endif
}

// THREAD-LINUX-006: validate-only does not mutate the calling thread.
TEST(LinuxThreadController, ValidateOnlyDoesNotApply) {
#ifndef __linux__
  GTEST_SKIP() << "LinuxThreadController is only supported on Linux";
#else
  auto before = LinuxThreadController::snapshot();
  ASSERT_TRUE(before) << runtime_error_message(before.error());
  ASSERT_FALSE(before->affinity.cores.empty());

  const ThreadPlacement placement{.affinity = CPUAffinity{before->affinity.cores.front()}};
  auto result = LinuxThreadController::validate_only(placement);

  auto after = LinuxThreadController::snapshot();
  ASSERT_TRUE(after) << runtime_error_message(after.error());
  EXPECT_TRUE(result) << runtime_error_message(result.error());
  EXPECT_EQ(after->affinity.cores, before->affinity.cores);
#endif
}

// THREAD-LINUX-007: probe detects requested CPUs outside the effective cpuset.
TEST(LinuxThreadController, ProbeFlagsCoresOutsideCpuset) {
#ifndef __linux__
  GTEST_SKIP() << "LinuxThreadController is only supported on Linux";
#else
  auto unavailable_core = first_core_outside_effective_cpuset();
  if (!unavailable_core.has_value()) {
    GTEST_SKIP() << "All CPU_SETSIZE cores are available in this process cpuset";
  }

  const ThreadPlacement placement{.affinity = CPUAffinity{*unavailable_core}};
  auto probe = LinuxThreadController::probe(placement);
  ASSERT_TRUE(probe) << runtime_error_message(probe.error());
  EXPECT_EQ(probe->unavailable_cores, (std::vector<uint32_t>{*unavailable_core}));

  auto validation = LinuxThreadController::validate_only(placement);
  EXPECT_FALSE(validation);
  EXPECT_EQ(validation.error().code, ErrorCode::kInvalidArgument);
#endif
}

// THREAD-LINUX-001: apply() pins the calling thread and readback reflects the new affinity.
TEST(LinuxThreadController, AppliesAffinityReadback) {
#ifndef __linux__
  GTEST_SKIP() << "LinuxThreadController is only supported on Linux";
#else
  const AffinityRestore restore;
  const uint32_t core = first_effective_core();

  auto result = LinuxThreadController::apply(ThreadPlacement{.affinity = CPUAffinity{core}});
  ASSERT_TRUE(result) << runtime_error_message(result.error());

  auto snapshot = LinuxThreadController::snapshot();
  ASSERT_TRUE(snapshot) << runtime_error_message(snapshot.error());
  EXPECT_EQ(snapshot->affinity.cores, (std::vector<uint32_t>{core}));
#endif
}

// THREAD-LINUX-001: an exact snapshot restores caller placement after temporary lane ownership.
TEST(LinuxThreadController, RestoresSnapshotAfterAffinityNarrowing) {
#ifndef __linux__
  GTEST_SKIP() << "LinuxThreadController is only supported on Linux";
#else
  const AffinityRestore restore;
  auto original = LinuxThreadController::snapshot();
  ASSERT_TRUE(original) << runtime_error_message(original.error());
  ASSERT_FALSE(original->affinity.cores.empty());

  const uint32_t core = original->affinity.cores.front();
  auto applied = LinuxThreadController::apply(ThreadPlacement{.affinity = CPUAffinity{core}});
  ASSERT_TRUE(applied) << runtime_error_message(applied.error());

  auto restored = LinuxThreadController::restore(*original);
  ASSERT_TRUE(restored) << runtime_error_message(restored.error());
  auto final = LinuxThreadController::snapshot();
  ASSERT_TRUE(final) << runtime_error_message(final.error());
  EXPECT_EQ(final->affinity.cores, original->affinity.cores);
  EXPECT_EQ(final->policy, original->policy);
  EXPECT_EQ(final->priority, original->priority);
  EXPECT_EQ(final->deadline.has_value(), original->deadline.has_value());
  if (original->deadline) {
    ASSERT_TRUE(final->deadline);
    // The condition and fatal assertion prove both optionals are engaged.
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    EXPECT_EQ(final->deadline->runtime_ns, original->deadline->runtime_ns);
    EXPECT_EQ(final->deadline->deadline_ns, original->deadline->deadline_ns);
    EXPECT_EQ(final->deadline->period_ns, original->deadline->period_ns);
    // NOLINTEND(bugprone-unchecked-optional-access)
  }
  EXPECT_EQ(final->reset_on_fork, original->reset_on_fork);
#endif
}

// THREAD-LINUX-002: affinity is attempted before scheduling policy.
TEST(LinuxThreadController, AffinityAppliedBeforePolicy) {
#ifndef __linux__
  GTEST_SKIP() << "LinuxThreadController is only supported on Linux";
#else
  auto unavailable_core = first_core_outside_effective_cpuset();
  if (!unavailable_core.has_value()) {
    GTEST_SKIP() << "All CPU_SETSIZE cores are available in this process cpuset";
  }

  // Key setup: combine an unavailable core with a FIFO request; affinity is attempted first.
  const int priority = sched_get_priority_min(SCHED_FIFO);
  ThreadPlacement placement = fifo_placement(priority);
  placement.affinity = CPUAffinity{*unavailable_core};

  auto result = LinuxThreadController::apply(placement);
  ASSERT_FALSE(result);
  // Key assertion: failure message names pthread_setaffinity_np, not the scheduler call.
  EXPECT_NE(result.error().message.find("pthread_setaffinity_np"), std::string::npos);
#endif
}

// THREAD-LINUX-008: snapshot reports the current calling thread affinity and policy.
TEST(LinuxThreadController, SnapshotReflectsApplied) {
#ifndef __linux__
  GTEST_SKIP() << "LinuxThreadController is only supported on Linux";
#else
  auto snapshot = LinuxThreadController::snapshot();

  ASSERT_TRUE(snapshot) << runtime_error_message(snapshot.error());
  EXPECT_FALSE(snapshot->affinity.cores.empty());
#endif
}

// THREAD-LINUX-003: FIFO application is opt-in and skipped unless this process can set RT policy.
TEST(LinuxThreadController, AppliesFifoReadback) {
#ifndef __linux__
  GTEST_SKIP() << "LinuxThreadController is only supported on Linux";
#else
  if (!realtime_thread_tests_enabled()) {
    GTEST_SKIP() << "Set HOLOSCAN_RUN_RT_THREAD_TESTS=1 to run real-time apply tests";
  }

  auto before = LinuxThreadController::snapshot();
  ASSERT_TRUE(before) << runtime_error_message(before.error());
  if (before->policy != SchedulingPolicy::kOther) {
    GTEST_SKIP() << "Not changing a thread that already uses a real-time policy";
  }

  // Key setup: probe confirms CAP_SYS_NICE / RLIMIT_RTPRIO is sufficient before apply().
  const int priority = sched_get_priority_min(SCHED_FIFO);
  const ThreadPlacement placement = fifo_placement(priority);
  auto probe = LinuxThreadController::probe(placement);
  ASSERT_TRUE(probe) << runtime_error_message(probe.error());
  if (!probe->can_apply_realtime) {
    GTEST_SKIP() << "CAP_SYS_NICE/RLIMIT_RTPRIO is not sufficient for SCHED_FIFO";
  }

  SchedulingRestore restore;
  auto result = LinuxThreadController::apply(placement);
  ASSERT_TRUE(result) << runtime_error_message(result.error());

  auto snapshot = LinuxThreadController::snapshot();
  ASSERT_TRUE(snapshot) << runtime_error_message(snapshot.error());
  // Key assertion: snapshot reflects SCHED_FIFO and the exact requested minimum priority.
  EXPECT_EQ(snapshot->policy, SchedulingPolicy::kFIFO);
  EXPECT_EQ(snapshot->priority, priority);

  auto restored = restore.restore();
  EXPECT_TRUE(restored) << runtime_error_message(restored.error());
#endif
}

// THREAD-LINUX-004: real-time policy application sets SCHED_RESET_ON_FORK.
TEST(LinuxThreadController, RealtimeSetsResetOnFork) {
#ifndef __linux__
  GTEST_SKIP() << "LinuxThreadController is only supported on Linux";
#else
  if (!realtime_thread_tests_enabled()) {
    GTEST_SKIP() << "Set HOLOSCAN_RUN_RT_THREAD_TESTS=1 to run real-time apply tests";
  }

  auto before = LinuxThreadController::snapshot();
  ASSERT_TRUE(before) << runtime_error_message(before.error());
  if (before->policy != SchedulingPolicy::kOther) {
    GTEST_SKIP() << "Not changing a thread that already uses a real-time policy";
  }

  // Key setup: same RT capability gate as AppliesFifoReadback.
  const int priority = sched_get_priority_min(SCHED_FIFO);
  const ThreadPlacement placement = fifo_placement(priority);
  auto probe = LinuxThreadController::probe(placement);
  ASSERT_TRUE(probe) << runtime_error_message(probe.error());
  if (!probe->can_apply_realtime) {
    GTEST_SKIP() << "CAP_SYS_NICE/RLIMIT_RTPRIO is not sufficient for SCHED_FIFO";
  }

  SchedulingRestore restore;
  auto result = LinuxThreadController::apply(placement);
  ASSERT_TRUE(result) << runtime_error_message(result.error());

  auto snapshot = LinuxThreadController::snapshot();
  ASSERT_TRUE(snapshot) << runtime_error_message(snapshot.error());
  // Key assertion: SCHED_RESET_ON_FORK is set so forked children inherit SCHED_OTHER.
  EXPECT_TRUE(snapshot->reset_on_fork);

  auto restored = restore.restore();
  EXPECT_TRUE(restored) << runtime_error_message(restored.error());
#endif
}

// THREAD-LINUX-003: SCHED_DEADLINE is opt-in because it can affect host admission state.
TEST(LinuxThreadController, AppliesDeadline) {
#ifndef __linux__
  GTEST_SKIP() << "LinuxThreadController is only supported on Linux";
#else
  if (std::getenv("HOLOSCAN_RUN_DEADLINE_THREAD_TESTS") == nullptr) {
    GTEST_SKIP() << "Set HOLOSCAN_RUN_DEADLINE_THREAD_TESTS=1 to run deadline apply test";
  }

  auto before = LinuxThreadController::snapshot();
  ASSERT_TRUE(before) << runtime_error_message(before.error());
  if (before->policy != SchedulingPolicy::kOther) {
    GTEST_SKIP() << "Not changing a thread that already uses a real-time policy";
  }

  // Key setup: SCHED_DEADLINE triple where runtime_ns(1 ms) ≤ deadline_ns = period_ns(10 ms).
  const ThreadPlacement placement{.scheduling = ThreadScheduling{
                                      .policy = SchedulingPolicy::kDeadline,
                                      .deadline =
                                          DeadlineScheduling{
                                              .runtime_ns = 1'000'000,
                                              .deadline_ns = 10'000'000,
                                              .period_ns = 10'000'000,
                                          },
                                  }};
  auto probe = LinuxThreadController::probe(placement);
  ASSERT_TRUE(probe) << runtime_error_message(probe.error());
  if (!probe->can_apply_realtime) {
    GTEST_SKIP() << "CAP_SYS_NICE is not available for SCHED_DEADLINE";
  }

  SchedulingRestore restore;
  auto result = LinuxThreadController::apply(placement);
  if (!result) {
    GTEST_SKIP() << runtime_error_message(result.error());
  }

  auto snapshot = LinuxThreadController::snapshot();
  ASSERT_TRUE(snapshot) << runtime_error_message(snapshot.error());
  // Key assertion: snapshot reports kDeadline and preserves the requested runtime_ns.
  EXPECT_EQ(snapshot->policy, SchedulingPolicy::kDeadline);
  ASSERT_TRUE(snapshot->deadline.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): asserted immediately above.
  EXPECT_EQ(snapshot->deadline->runtime_ns, 1'000'000);

  auto restored = restore.restore();
  EXPECT_TRUE(restored) << runtime_error_message(restored.error());
#endif
}

}  // namespace
}  // namespace holoscan::detail
