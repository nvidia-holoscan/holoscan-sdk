/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include <holoscan/core/tensor_output_loan.hpp>

#include "../support/optional_assertions.hpp"
#include "compiler/execution_plan_state.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_cuda_graph_tensor_resources.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] TensorLoanRequest contiguous_request(TensorShapeView shape,
                                                   DLDataType dtype = DLDataType{
                                                       .code = kDLUInt, .bits = 8U, .lanes = 1U}) {
  return TensorLoanRequest{
      .shape = shape,
      .dtype = dtype,
  };
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphTensorResources, ComputesZeroBytesForAnyZeroExtentPosition) {
  const std::array<std::array<std::int64_t, 3U>, 3U> shapes{{
      {0, 4, 8},
      {4, 0, 8},
      {4, 8, 0},
  }};

  for (const auto& shape : shapes) {
    const auto bytes = cuda_graph_contiguous_byte_span(contiguous_request(shape));

    HOLOSCAN_ASSERT_HAS_VALUE(bytes) << "Expected contiguous byte span";
    EXPECT_EQ(*bytes, 0U);
  }
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphTensorResources, StillRejectsNegativeExtentAfterZeroExtent) {
  const std::array<std::int64_t, 3U> shape{0, 4, -1};

  EXPECT_FALSE(cuda_graph_contiguous_byte_span(contiguous_request(shape)));
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphTensorResources, StillRejectsNonzeroExtentProductOverflow) {
  const std::array<std::int64_t, 2U> shape{std::numeric_limits<std::int64_t>::max(), 3};

  EXPECT_FALSE(cuda_graph_contiguous_byte_span(contiguous_request(shape)));
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphTensorResources, ComputesCanonicalRoundedBytesForSubByteElements) {
  constexpr std::array<std::int64_t, 1U> shape{4};
  constexpr DLDataType kUInt1{.code = kDLUInt, .bits = 1U, .lanes = 1U};

  const auto bytes = cuda_graph_contiguous_byte_span(contiguous_request(shape, kUInt1));

  HOLOSCAN_ASSERT_HAS_VALUE(bytes) << "Expected contiguous byte span";
  EXPECT_EQ(*bytes, 4U);
}

// REQ: RUNTIME-059
TEST(RuntimeCUDAGraphTensorResources, LaunchMethodsRejectUnknownRegionIndex) {
  const ExecutionPlanState plan;
  std::vector<RuntimeOperatorInstance> operators;
  auto resources = RuntimeCUDAGraphTensorResources::create(
      plan, operators, RuntimeCUDAGraphTensorPoolBinding{}, ProcessGroupId{1U});
  ASSERT_TRUE(resources) << resources.error().message;

  const auto begun = (*resources)->begin_launch(0U, nullptr);
  const auto available = (*resources)->output_available(0U);
  const auto finished = (*resources)->finish_launch(0U);
  ASSERT_FALSE(begun);
  ASSERT_FALSE(available);
  ASSERT_FALSE(finished);
  EXPECT_EQ(begun.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(available.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(finished.error().code, ErrorCode::kInvalidArgument);
  (*resources)->cancel_launch(0U);
}

}  // namespace
}  // namespace holoscan::detail
