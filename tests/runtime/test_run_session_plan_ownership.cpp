/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <atomic>

#include <holoscan/core/run.hpp>

#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"

namespace holoscan {
namespace {

// REQ: RUNTIME-001
// REQ: RUNTIME-002
// REQ: RUNTIME-005
TEST(RunSessionPlanOwnershipTest, EquivalentPlansShareProvenanceButSessionsOwnRunIdentity) {
  // Key setup: first_plan and second_plan compile the same graph (equal plan_generation);
  //            different_plan compiles a modified graph and receives a different generation.
  //            Two concurrent sessions are then started on the equivalent plans.
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  const ExecutionPlan first_plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  const ExecutionPlan second_plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  const ExecutionPlan different_plan =
      test::enable_direct_runtime(test::make_runtime_test_plan(true));
  ASSERT_TRUE(first_plan.ok()) << first_plan.json();
  ASSERT_TRUE(second_plan.ok()) << second_plan.json();
  ASSERT_TRUE(different_plan.ok()) << different_plan.json();

  const OperatorRef first_plan_ref = first_plan.lifecycle_checkpoints().front().operator_instance;
  const OperatorRef second_plan_ref = second_plan.lifecycle_checkpoints().front().operator_instance;
  const OperatorRef different_plan_ref =
      different_plan.lifecycle_checkpoints().front().operator_instance;
  // Plan generation identifies semantic plan/catalog provenance. Recompiling
  // identical content therefore preserves the generation, while a materially
  // different graph receives a different provenance identity.
  ASSERT_EQ(first_plan_ref.plan_generation, second_plan_ref.plan_generation);
  ASSERT_NE(first_plan_ref.plan_generation, different_plan_ref.plan_generation);

  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  RunSession first = run_async(first_plan);
  RunSession second = run_async(second_plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_active(2U));

  const auto first_bound = first.operator_ref(first_plan_ref);
  const auto second_bound = second.operator_ref(second_plan_ref);
  ASSERT_TRUE(first_bound);
  ASSERT_TRUE(second_bound);
  // Key assertion: concurrent sessions yield different run_generation values, but cross-querying
  //               by the other plan's OperatorRef still resolves to the same bound OperatorRef.
  EXPECT_NE(first_bound->run_generation, second_bound->run_generation);
  const auto first_equivalent = first.operator_ref(second_plan_ref);
  const auto second_equivalent = second.operator_ref(first_plan_ref);
  ASSERT_TRUE(first_equivalent);
  ASSERT_TRUE(second_equivalent);
  EXPECT_EQ(*first_equivalent, *first_bound);
  EXPECT_EQ(*second_equivalent, *second_bound);
  EXPECT_EQ(first.operator_ref(different_plan_ref).error().code, ErrorCode::kStaleOperatorRef);
  EXPECT_EQ(second.operator_ref(different_plan_ref).error().code, ErrorCode::kStaleOperatorRef);

  first.request_stop();
  second.request_stop();
  first.wait();
  second.wait();
  EXPECT_EQ(first.termination(), RunTermination::kStopped);
  EXPECT_EQ(second.termination(), RunTermination::kStopped);

  test::RuntimeProbeState::reset();
  RunSession first_sequential = run_async(first_plan);
  first_sequential.wait();
  test::RuntimeProbeState::reset();
  RunSession second_sequential = run_async(second_plan);
  second_sequential.wait();
  EXPECT_EQ(first_sequential.termination(), RunTermination::kCompleted);
  EXPECT_EQ(second_sequential.termination(), RunTermination::kCompleted);
  EXPECT_EQ(test::RuntimeProbeState::max_active.load(std::memory_order_relaxed), 1U);
}

}  // namespace
}  // namespace holoscan
