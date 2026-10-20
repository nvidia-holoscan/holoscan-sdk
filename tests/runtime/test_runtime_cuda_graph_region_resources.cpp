/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>
#include <cstdint>
#include <utility>
#include <vector>

#include "compiler/execution_plan_state.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_cuda_graph_region_resources.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] cudaStream_t stream_handle(std::uintptr_t value) noexcept {
  return reinterpret_cast<cudaStream_t>(value);
}

[[nodiscard]] ExecutionPlanState two_member_region_plan() {
  ExecutionPlanState plan;
  plan.cuda_graph_capture_plans.push_back(CUDAGraphCapturePlanRecord{
      .graph = GraphId{1U},
      .region = RegionId{1U},
      .members = {{.operator_ref = OperatorRef{.id = 1U, .plan_generation = 1U},
                   .vertex = VertexId{1U}},
                  {.operator_ref = OperatorRef{.id = 2U, .plan_generation = 1U},
                   .vertex = VertexId{2U}}},
  });
  plan.cuda_graph_execution_regions.push_back(CUDAGraphExecutionRegionPlanRecord{
      .graph = GraphId{1U},
      .region = RegionId{1U},
      .fused_operator = VertexId{1U},
      .partition = ExecutionPartitionId{1U},
      .execution_lane = 0U,
      .process_group = ProcessGroupId{1U},
      .device = DeviceId{0},
      .stream_id = "partition-1-device-0",
  });
  plan.cuda_compute_streams.push_back(CudaComputeStreamPlanRecord{
      .process_group = ProcessGroupId{1U},
      .device = DeviceId{0},
      .partition = ExecutionPartitionId{1U},
      .stream_id = "partition-1-device-0",
  });
  return plan;
}

[[nodiscard]] std::vector<RuntimeOperatorInstance> two_member_region_operators() {
  std::vector<RuntimeOperatorInstance> operators(2U);
  for (std::size_t index = 0U; index < operators.size(); ++index) {
    operators[index].vertex = VertexId{index + 1U};
    operators[index].partition = ExecutionPartitionId{1U};
    operators[index].execution_lane = 0U;
    operators[index].compute_stream = stream_handle(1U);
    operators[index].compute_device = DeviceId{0};
  }
  return operators;
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, JoinsCaptureAndExecutionPlansToRuntimeOperators) {
  auto plan = two_member_region_plan();
  auto operators = two_member_region_operators();

  auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{1U});

  ASSERT_TRUE(resources) << resources.error().message;
  EXPECT_EQ((*resources)->size(), 1U);
  EXPECT_NE(operators.front().fused_cuda_graph_operator, nullptr);
  EXPECT_EQ(operators.back().fused_cuda_graph_operator, nullptr);
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, RemovesFusedOperatorBindingBeforeResourceDestruction) {
  auto plan = two_member_region_plan();
  auto operators = two_member_region_operators();
  {
    auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{1U});
    ASSERT_TRUE(resources) << resources.error().message;
    ASSERT_NE(operators.front().fused_cuda_graph_operator, nullptr);
  }

  EXPECT_EQ(operators.front().fused_cuda_graph_operator, nullptr);
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, DetachRemovesOperatorBackReferencesBeforeStorageRelease) {
  auto plan = two_member_region_plan();
  auto operators = two_member_region_operators();
  auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{1U});
  ASSERT_TRUE(resources) << resources.error().message;
  ASSERT_NE(operators.front().fused_cuda_graph_operator, nullptr);

  (*resources)->detach_session_state();

  EXPECT_EQ(operators.front().fused_cuda_graph_operator, nullptr);
  operators.clear();
  operators.shrink_to_fit();
  resources->reset();
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, IgnoresRegionsOwnedByAnotherProcessGroup) {
  auto plan = two_member_region_plan();
  auto operators = two_member_region_operators();

  auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{2U});

  ASSERT_TRUE(resources) << resources.error().message;
  EXPECT_EQ((*resources)->size(), 0U);
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, RejectsExecutionRegionWithoutOneCapturePlan) {
  auto plan = two_member_region_plan();
  plan.cuda_graph_capture_plans.clear();
  auto operators = two_member_region_operators();

  auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{1U});

  ASSERT_FALSE(resources);
  EXPECT_EQ(resources.error().message, "CUDA_GRAPH_RUNTIME_PLAN_INVALID");
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, RejectsSourcePlacementThatDiffersFromExecutionPlan) {
  auto plan = two_member_region_plan();
  auto operators = two_member_region_operators();
  operators.front().partition = ExecutionPartitionId{2U};

  auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{1U});

  ASSERT_FALSE(resources);
  EXPECT_EQ(resources.error().message, "CUDA_GRAPH_RUNTIME_STREAM_MISMATCH");
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, IgnoresOriginalMemberStreamDuringPrivateStreamCapture) {
  auto plan = two_member_region_plan();
  auto operators = two_member_region_operators();
  operators.back().compute_stream = stream_handle(2U);
  operators.back().compute_device = {};

  auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{1U});

  ASSERT_TRUE(resources) << resources.error().message;
  EXPECT_EQ((*resources)->size(), 1U);
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, RejectsMissingSelectedStreamPlan) {
  auto plan = two_member_region_plan();
  plan.cuda_compute_streams.clear();
  auto operators = two_member_region_operators();

  auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{1U});

  ASSERT_FALSE(resources);
  EXPECT_EQ(resources.error().message, "CUDA_GRAPH_RUNTIME_STREAM_MISMATCH");
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, RejectsGreenContextThatDiffersFromSelectedStream) {
  auto plan = two_member_region_plan();
  plan.cuda_graph_execution_regions.front().green_context_id = "green-1";
  auto operators = two_member_region_operators();

  auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{1U});

  ASSERT_FALSE(resources);
  EXPECT_EQ(resources.error().message, "CUDA_GRAPH_RUNTIME_STREAM_MISMATCH");
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, RejectsMutableCaptureMemberStateAtRuntime) {
  auto plan = two_member_region_plan();
  auto operators = two_member_region_operators();
  operators.back().runtime_parameter_updates_enabled = true;

  auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{1U});

  ASSERT_FALSE(resources);
  EXPECT_EQ(resources.error().message, "CUDA_GRAPH_CAPTURE_STATE_MUTABLE");
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, RejectsLifecycleParameterUpdatesAtRuntime) {
  auto plan = two_member_region_plan();
  auto operators = two_member_region_operators();
  operators.back().lifecycle_parameter_updates_enabled = true;

  auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{1U});

  ASSERT_FALSE(resources);
  EXPECT_EQ(resources.error().message, "CUDA_GRAPH_CAPTURE_STATE_MUTABLE");
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, RejectsCapturedMemberInLifecycleRestartClosure) {
  auto plan = two_member_region_plan();
  const OperatorRef target{.id = 3U, .plan_generation = 1U};
  plan.lifecycle_restart_targets.push_back(LifecycleRestartTargetPlanRecord{
      .target = target,
      .disposition = LifecycleRestartTargetDisposition::kAdmitted,
  });
  plan.lifecycle_restart_closures.push_back(LifecycleRestartClosurePlanRecord{
      .target = target,
      .closure = {target, plan.cuda_graph_capture_plans.front().members.back().operator_ref},
  });
  auto operators = two_member_region_operators();

  auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{1U});

  ASSERT_FALSE(resources);
  EXPECT_EQ(resources.error().message, "CUDA_GRAPH_CAPTURE_STATE_MUTABLE");
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, RejectsTwoExecutionRegionsWithTheSameFusedOperator) {
  auto plan = two_member_region_plan();
  plan.cuda_graph_capture_plans.push_back(CUDAGraphCapturePlanRecord{
      .graph = GraphId{1U},
      .region = RegionId{2U},
      .members = {{.vertex = VertexId{1U}}},
  });
  auto duplicate = plan.cuda_graph_execution_regions.front();
  duplicate.region = RegionId{2U};
  plan.cuda_graph_execution_regions.push_back(std::move(duplicate));
  auto operators = two_member_region_operators();

  auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{1U});

  ASSERT_FALSE(resources);
  EXPECT_EQ(resources.error().message, "CUDA_GRAPH_RUNTIME_PLAN_INVALID");
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphRegionResources, RejectsExecutionEgressWithoutTensorBuffer) {
  constexpr SourceId kRegionSource = make_source_id(11U);
  auto plan = two_member_region_plan();
  plan.cuda_graph_execution_regions.front().egress = CUDAGraphCaptureEdgeRecord{};
  auto operators = two_member_region_operators();
  const TemporalTriggerPlanRecord trigger{.readiness_source = kRegionSource};
  operators.front().temporal_trigger = &trigger;
  auto resources = RuntimeCUDAGraphRegionResources::create(plan, operators, ProcessGroupId{1U});

  ASSERT_FALSE(resources);
  EXPECT_EQ(resources.error().message, "CUDA_GRAPH_RUNTIME_PLAN_INVALID");
}

}  // namespace
}  // namespace holoscan::detail
