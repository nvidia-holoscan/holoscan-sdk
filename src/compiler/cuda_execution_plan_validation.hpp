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
  // Partition owner IDs are human-readable strings, so lexical comparison would put
  // "partition.10" before "partition.2". Keep partition streams numerically ordered and place
  // residual/process-device streams after them.
  const auto left_key = std::tuple{left.process_group.value,
                                   left.device.value,
                                   !left.partition.has_value(),
                                   left.partition ? left.partition->value : 0U};
  const auto right_key = std::tuple{right.process_group.value,
                                    right.device.value,
                                    !right.partition.has_value(),
                                    right.partition ? right.partition->value : 0U};
  return left_key != right_key ? left_key < right_key : left.owner_id < right.owner_id;
}

[[nodiscard]] inline bool canonical_operator_cuda_stream_binding_order(
    const OperatorCudaStreamBindingPlanRecord& left,
    const OperatorCudaStreamBindingPlanRecord& right) noexcept {
  return left.operator_vertex.value < right.operator_vertex.value;
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
