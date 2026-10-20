/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>

namespace holoscan {
namespace {

static_assert(std::is_nothrow_copy_constructible_v<CudaGreenContextPlacement>);
static_assert(noexcept(CudaGreenContextPlacement{} == CudaGreenContextPlacement{}));
static_assert(noexcept(CudaGreenContextCapability{} == CudaGreenContextCapability{}));

// REQ: MEM-GREENCTX-014
// REQ: MEM-GREENCTX-MIG-001
TEST(CudaGreenContextAuthoring, PlacementAndCapabilityHaveValueSemantics) {
  // Key setup: construct equal and unequal placement values plus a fully populated capability.
  constexpr CudaGreenContextPlacement first{.device = DeviceId{1}, .sm_count = 24U};
  constexpr CudaGreenContextPlacement same{.device = DeviceId{1}, .sm_count = 24U};
  constexpr CudaGreenContextPlacement other_device{.device = DeviceId{2}, .sm_count = 24U};
  constexpr CudaGreenContextPlacement other_count{.device = DeviceId{1}, .sm_count = 32U};
  static_assert(first == same);
  static_assert(first != other_device);
  static_assert(first != other_count);

  const CudaGreenContextCapability capability{
      .device = DeviceId{1},
      .deployment_device_id = "gpu.test.uuid.1",
      .total_sm_count = 80U,
      .min_sm_partition_size = 4U,
      .sm_coscheduled_alignment = 8U,
      .structured_sm_quantum = 8U,
      .max_structured_context_count = 10U,
  };
  // Exercise value-copy semantics rather than borrowing the original capability.
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const auto copy = capability;
  // Key assertion: placement comparisons are constexpr and capability copies preserve all fields.
  EXPECT_EQ(copy, capability);
}

// REQ: MEM-GREENCTX-014
TEST(CudaGreenContextAuthoring, BindingDeepCopiesAndPreservesDuplicateInsertionOrder) {
  // Key setup: bind the same mutable partition name twice, then mutate the source string.
  std::string partition_name{"inference"};
  DeploymentBinding binding;
  DeploymentBinding* chained =
      &binding.bind_cuda_green_context(partition_name, {.device = DeviceId{0}, .sm_count = 48U})
           .bind_cuda_green_context(partition_name, {.device = DeviceId{0}, .sm_count = 56U});
  partition_name.assign("mutated");

  // Key assertion: the binding owns both original names in insertion order and remains copyable.
  EXPECT_EQ(chained, &binding);
  ASSERT_EQ(binding.cuda_green_contexts().size(), 2U);
  EXPECT_EQ(binding.cuda_green_contexts()[0].partition_name, "inference");
  EXPECT_EQ(binding.cuda_green_contexts()[0].placement.sm_count, 48U);
  EXPECT_EQ(binding.cuda_green_contexts()[1].partition_name, "inference");
  EXPECT_EQ(binding.cuda_green_contexts()[1].placement.sm_count, 56U);

  DeploymentBinding copied = binding;
  const DeploymentBinding moved = std::move(copied);
  ASSERT_EQ(moved.cuda_green_contexts().size(), 2U);
  EXPECT_EQ(moved.cuda_green_contexts()[0].partition_name, "inference");
}

// REQ: MEM-GREENCTX-014
TEST(CudaGreenContextAuthoring, CapabilityQueryRejectsUnresolvedDeviceWithoutCudaWork) {
  // Key setup: query capability with an unresolved DeviceId and no need to initialize CUDA.
  auto capability = query_cuda_green_context_capability(DeviceId{});
  // Key assertion: validation rejects the selector as an invalid argument before CUDA work.
  ASSERT_FALSE(capability);
  EXPECT_EQ(capability.error().code, ErrorCode::kInvalidArgument);
}

}  // namespace
}  // namespace holoscan
