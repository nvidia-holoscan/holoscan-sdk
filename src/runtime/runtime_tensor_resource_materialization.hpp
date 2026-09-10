/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::detail {

struct ExecutionPlanState;
struct RuntimeOperatorInstance;
class RuntimeCudaExecutionResources;

using RuntimeTensorDeviceResource = const TensorDeviceResourcePlanRecord*;

/// Construct a bounded Tensor resource diagnostic without throwing.
[[nodiscard]] Error tensor_resource_error(std::string_view context,
                                          std::string_view identity_kind = {},
                                          std::string_view identity = {}) noexcept;

/// Construct a bounded Tensor resource diagnostic with a numeric identity.
[[nodiscard]] Error tensor_resource_error(std::string_view context, std::string_view identity_kind,
                                          std::uint64_t identity) noexcept;

/// Compare canonical process/device resource keys.
[[nodiscard]] bool canonical_tensor_device_resource_order(
    const TensorDeviceResourcePlanRecord& left,
    const TensorDeviceResourcePlanRecord& right) noexcept;

/// Validate and select the device resource records owned by this process group.
[[nodiscard]] expected<void, Error> materialize_tensor_device_resources(
    const ExecutionPlanState& plan, ProcessGroupId process_group,
    std::vector<RuntimeTensorDeviceResource>& devices);

/// Bind every reconstructed Operator to its exact planned CUDA stream.
[[nodiscard]] expected<void, Error> bind_tensor_operator_streams(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    const RuntimeCudaExecutionResources& cuda_resources);

/// Resolve the producer stream for one compiler-owned output allocation.
[[nodiscard]] expected<cudaStream_t, Error> tensor_output_stream(
    const ExecutionPlanState& plan, const RuntimeCudaExecutionResources& cuda_resources,
    std::span<const RuntimeTensorDeviceResource> devices,
    const TensorAllocationPlanRecord& allocation);

/// Return whether every accepted consumer uses the producer's planned stream.
[[nodiscard]] bool tensor_can_elide_native_readiness(
    const ExecutionPlanState& plan, const RuntimeCudaExecutionResources& cuda_resources,
    const TensorAllocationPlanRecord& allocation, cudaStream_t producer_stream) noexcept;

}  // namespace holoscan::detail
