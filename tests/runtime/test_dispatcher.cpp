/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>

#include <gtest/gtest.h>

#include <holoscan/core/run.hpp>

#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"

namespace holoscan {
namespace {

// REQ: RUNTIME-002
TEST(DispatcherTest, ConcurrentSessionsOwnIndependentOperators) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset();

  // Key setup: two concurrent RunSessions created from the same immutable ExecutionPlan.
  RunSession first = run_async(plan);
  RunSession second = run_async(plan);
  first.wait();
  second.wait();

  EXPECT_EQ(first.termination(), RunTermination::kCompleted);
  EXPECT_EQ(second.termination(), RunTermination::kCompleted);
  // Key assertion: each session creates its own operator instances (constructed count = 2).
  EXPECT_EQ(test::RuntimeProbeState::constructed.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(test::RuntimeProbeState::clocks_constructed.load(std::memory_order_relaxed), 2U);
}

}  // namespace
}  // namespace holoscan
