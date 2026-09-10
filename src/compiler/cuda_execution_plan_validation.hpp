/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <tuple>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::detail {

struct ExecutionPlanState;

// These semantic keys are the sole canonical ordering authority for CUDA execution records.
// Native handles, driver-returned IDs, and request insertion order never participate.
[[nodiscard]] inline bool canonical_cuda_green_context_pool_order(
    const CudaGreenContextPoolPlanRecord& left,
    const CudaGreenContextPoolPlanRecord& right) noexcept {
  return std::tie(left.process_group.value, left.device.value, left.pool_id) <
         std::tie(right.process_group.value, right.device.value, right.pool_id);
}

[[nodiscard]] inline bool canonical_cuda_green_context_resource_order(
    const CudaGreenContextResourcePlanRecord& left,
    const CudaGreenContextResourcePlanRecord& right) noexcept {
  return std::tie(left.process_group.value, left.device.value, left.green_context_id) <
         std::tie(right.process_group.value, right.device.value, right.green_context_id);
}

[[nodiscard]] inline bool canonical_cuda_green_context_request_order(
    const CudaGreenContextRequestPlanRecord& left,
    const CudaGreenContextRequestPlanRecord& right) noexcept {
  return std::tie(left.authored_name, left.requested.device.value, left.requested.sm_count) <
         std::tie(right.authored_name, right.requested.device.value, right.requested.sm_count);
}

[[nodiscard]] inline bool canonical_cuda_green_context_assignment_order(
    const CudaGreenContextAssignmentPlanRecord& left,
    const CudaGreenContextAssignmentPlanRecord& right) noexcept {
  return std::tie(left.process_group.value, left.device.value, left.partition.value) <
         std::tie(right.process_group.value, right.device.value, right.partition.value);
}

[[nodiscard]] inline bool canonical_cuda_compute_stream_order(
    const CudaComputeStreamPlanRecord& left, const CudaComputeStreamPlanRecord& right) noexcept {
  return std::tie(left.process_group.value, left.device.value, left.owner_id) <
         std::tie(right.process_group.value, right.device.value, right.owner_id);
}

[[nodiscard]] inline bool canonical_tensor_device_resource_order(
    const TensorDeviceResourcePlanRecord& left,
    const TensorDeviceResourcePlanRecord& right) noexcept {
  return std::tie(left.process_group.value, left.device.value) <
         std::tie(right.process_group.value, right.device.value);
}

/// Validate canonical order, unique semantic keys, and every CUDA-plan join without calling CUDA.
[[nodiscard]] expected<void, Error> validate_cuda_execution_plan_records(
    const ExecutionPlanState& state);

}  // namespace holoscan::detail
