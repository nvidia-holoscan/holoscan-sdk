/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <latch>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include <holoscan/core/managed_shutdown_signals.hpp>

#include "managed_signal_benchmark.hpp"

namespace holoscan::benchmark_support {
namespace {

// The injected-census cases are supporting benchmark-preflight evidence, not Runtime admission.
// REQ: RUNTIME-049
TEST(BenchmarkSignalAdmission, SingleThreadNeedsOnlyOneCensus) {
  std::size_t calls{};
  const auto result = wait_for_single_thread([&] {
    ++calls;
    return ThreadCensus{.count = 1U};
  });
  EXPECT_TRUE(result);
  EXPECT_EQ(calls, 1U);
}

// REQ: RUNTIME-049
TEST(BenchmarkSignalAdmission, WaitsForJoinedTaskEntriesToRetire) {
  // Model two joined workers whose procfs entries disappear in successive censuses.
  std::size_t count = 3U;
  const auto result = wait_for_single_thread([&] { return ThreadCensus{.count = count--}; });
  EXPECT_TRUE(result);
  EXPECT_EQ(count, 0U);
}

// REQ: RUNTIME-049
TEST(BenchmarkSignalAdmission, PersistentTasksFailWithCountAtDeadline) {
  const auto result = wait_for_single_thread([] { return ThreadCensus{.count = 2U}; },
                                             std::chrono::nanoseconds::zero());
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kTimeout);
  EXPECT_NE(result.error().message.find("observed_threads=2"), std::string::npos);
}

// REQ: RUNTIME-049
TEST(BenchmarkSignalAdmission, CensusErrorsFailImmediatelyEvenWithOneObservedThread) {
  std::size_t calls{};
  const auto result = wait_for_single_thread([&] {
    ++calls;
    return ThreadCensus{.count = 1U, .error = EACCES};
  });
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kFailure);
  EXPECT_NE(result.error().message.find("errno=" + std::to_string(EACCES)), std::string::npos);
  EXPECT_EQ(calls, 1U);
}

// REQ: RUNTIME-049
TEST(BenchmarkSignalAdmission, EmptyCensusIsNotAcceptedAsQuiescence) {
  const auto result = wait_for_single_thread([] { return ThreadCensus{}; });
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kFailure);
}

// The following cases exercise production managed-signal admission after benchmark preflight.
// REQ: RUNTIME-049
TEST(BenchmarkSignalAdmission, RepeatedJoinThenAdmission) {
  for (int iteration = 0; iteration != 256; ++iteration) {
    SCOPED_TRACE(iteration);
    std::thread worker([] {});
    worker.join();
    ASSERT_TRUE(wait_for_single_thread());
    auto admitted = ManagedShutdownSignals::create();
    ASSERT_TRUE(admitted);
    ASSERT_TRUE(admitted->close());
  }
}

// REQ: RUNTIME-049
TEST(BenchmarkSignalAdmission, LiveWorkerIsRejectedAndAdmissionSucceedsAfterJoin) {
  std::latch started{1};
  std::latch release{1};
  std::thread worker([&] {
    started.count_down();
    release.wait();
  });
  started.wait();
  const auto blocked =
      wait_for_single_thread(read_process_thread_census, std::chrono::nanoseconds::zero());
  auto rejected = ManagedShutdownSignals::create();
  release.count_down();
  worker.join();

  ASSERT_FALSE(blocked);
  EXPECT_EQ(blocked.error().code, ErrorCode::kTimeout);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
  ASSERT_TRUE(wait_for_single_thread());
  auto admitted = ManagedShutdownSignals::create();
  ASSERT_TRUE(admitted);
  EXPECT_TRUE(admitted->close());
}

}  // namespace
}  // namespace holoscan::benchmark_support
