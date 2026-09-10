/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include <holoscan/core/execution_plan.hpp>

#include "execution_plan_state.hpp"

namespace holoscan::detail {

/// One canonical compiler-owned CUDA service that requires an independent stream.
struct CudaServiceStreamRequirement {
  ProcessGroupId process_group{};
  DeviceId device{};
  std::string owner_id;

  friend bool operator==(const CudaServiceStreamRequirement&,
                         const CudaServiceStreamRequirement&) = default;
};

/// Return the stable service-stream owner for one process-local Channel service.
[[nodiscard]] inline std::string cuda_channel_service_stream_owner(
    const ProcessLocalChannelServicePlanRecord& service, ProcessGroupId process_group,
    DeviceId device) {
  return "service.channel.local.source." + std::to_string(service.source.value()) + ".queue." +
         std::to_string(static_cast<std::uint32_t>(service.queue_policy)) + ".delivery." +
         std::to_string(static_cast<std::uint32_t>(service.delivery_kind)) + ".process." +
         std::to_string(process_group.value) + ".device." + std::to_string(device.value);
}

/// Return the stable service-stream owner for one cross-process Channel service endpoint.
[[nodiscard]] inline std::string cuda_channel_service_stream_owner(
    const CrossProcessChannelServicePlanRecord& service, ProcessGroupId process_group,
    DeviceId device) {
  return "service.channel.cross.source." + std::to_string(service.source.value()) + ".queue." +
         std::to_string(static_cast<std::uint32_t>(service.queue_policy)) + ".process." +
         std::to_string(process_group.value) + ".device." + std::to_string(device.value);
}

/// Derive all CUDA service-stream owners and bind external transfer rows to their owner lineage.
[[nodiscard]] std::vector<CudaServiceStreamRequirement> lower_cuda_service_stream_requirements(
    ExecutionPlanState& state);

/// Reconstruct the canonical immutable service inventory without mutating plan records.
[[nodiscard]] std::vector<CudaServiceStreamRequirement> cuda_service_stream_requirements(
    const ExecutionPlanState& state);

/// Verify that a service stream names exactly one compiler-owned CUDA service requirement.
[[nodiscard]] bool valid_cuda_service_stream_requirement(const ExecutionPlanState& state,
                                                         const CudaComputeStreamPlanRecord& stream);

}  // namespace holoscan::detail
