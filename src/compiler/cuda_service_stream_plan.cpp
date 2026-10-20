/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_service_stream_plan.hpp"

#include <algorithm>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "execution_plan_contracts.hpp"
#include "execution_plan_lookup.hpp"
#include "execution_plan_state.hpp"

namespace holoscan::detail {
namespace {

template <typename Service>
[[nodiscard]] std::vector<DeviceId> service_devices(const Service& service) {
  std::vector<DeviceId> result;
  const auto append = [&](DeviceId device) {
    if (device.valid() && std::find(result.begin(), result.end(), device) == result.end()) {
      result.push_back(device);
    }
  };
  for_each_tensor_service_storage(
      tensor_service_contract_view(service),
      [&](TensorServiceStorageView storage) { append(storage.device); });
  std::sort(result.begin(), result.end(), [](DeviceId lhs, DeviceId rhs) {
    return lhs.value < rhs.value;
  });
  return result;
}

[[nodiscard]] std::string service_owner(std::string_view family, std::string_view service_id,
                                        ProcessGroupId group, DeviceId device) {
  return "service." + std::string(family) + "." + std::string(service_id) + ".process." +
         std::to_string(group.value) + ".device." + std::to_string(device.value);
}

void append_requirement(std::vector<CudaServiceStreamRequirement>& result,
                        ProcessGroupId process_group, DeviceId device, std::string owner_id) {
  if (process_group && device.valid() && !owner_id.empty()) {
    result.push_back(CudaServiceStreamRequirement{
        .process_group = process_group,
        .device = device,
        .owner_id = std::move(owner_id),
    });
  }
}

[[nodiscard]] const TensorInterfacePoolPlanRecord* find_transfer_interface_pool(
    const ExecutionPlanState& state, const TensorTransferPlanRecord& transfer) noexcept {
  if (transfer.field_id.has_value() != transfer.tensor_index.has_value()) {
    return nullptr;
  }
  return transfer.field_id ? find_tensor_interface_pool_by_topic_field(
                                 state, transfer.topic, *transfer.field_id, *transfer.tensor_index)
                           : find_tensor_interface_pool_by_topic(state, transfer.topic);
}

[[nodiscard]] std::vector<CudaServiceStreamRequirement> derive_requirements(
    const ExecutionPlanState& state) {
  std::vector<CudaServiceStreamRequirement> result;
  // Process-local direct-retained and HoloMQ paths retain or transport the existing Tensor
  // backing; neither performs compiler-owned staging or copy work. Add a local requirement only
  // when a future physical service actually owns such work and Runtime can consume its stream.
  for (const auto& service : state.cross_process_channel_services) {
    std::vector<ProcessGroupId> groups = service.target_process_groups;
    if (descriptor_tensor_schema_service(service)) {
      groups.push_back(service.source_process_group);
    }
    std::sort(groups.begin(), groups.end(), [](ProcessGroupId lhs, ProcessGroupId rhs) {
      return lhs.value < rhs.value;
    });
    groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
    for (const ProcessGroupId group : groups) {
      for (const DeviceId device : service_devices(service)) {
        if (cross_process_tensor_service_requires_stream_for_device_in_group(
                service, group, device)) {
          append_requirement(
              result, group, device, cuda_channel_service_stream_owner(service, group, device));
        }
      }
    }
  }
  for (const auto& transfer : state.tensor_transfer_plans) {
    if (!transfer.requires_exportable_copy_pool ||
        transfer.disposition != PlanDisposition::kAccepted) {
      continue;
    }
    const auto* pool = find_transfer_interface_pool(state, transfer);
    if (pool == nullptr || !pool->allocator_process_group || !pool->device.valid()) {
      continue;
    }
    append_requirement(result,
                       pool->allocator_process_group,
                       pool->device,
                       service_owner("tensor.transfer",
                                     "topic." + std::to_string(transfer.topic.value),
                                     pool->allocator_process_group,
                                     pool->device));
  }
  std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.process_group.value, lhs.device.value, lhs.owner_id) <
           std::tie(rhs.process_group.value, rhs.device.value, rhs.owner_id);
  });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

}  // namespace

std::vector<CudaServiceStreamRequirement> lower_cuda_service_stream_requirements(
    ExecutionPlanState& state) {
  for (auto& transfer : state.tensor_transfer_plans) {
    transfer.service_owner_id.clear();
    if (!transfer.requires_exportable_copy_pool ||
        transfer.disposition != PlanDisposition::kAccepted) {
      continue;
    }
    const auto* pool = find_transfer_interface_pool(state, transfer);
    if (pool == nullptr || !pool->allocator_process_group || !pool->device.valid()) {
      continue;
    }
    transfer.interface_process_group = pool->allocator_process_group;
    transfer.interface_device = pool->device;
    transfer.service_owner_id = service_owner("tensor.transfer",
                                              "topic." + std::to_string(transfer.topic.value),
                                              pool->allocator_process_group,
                                              pool->device);
  }
  return derive_requirements(state);
}

std::vector<CudaServiceStreamRequirement> cuda_service_stream_requirements(
    const ExecutionPlanState& state) {
  return derive_requirements(state);
}

bool valid_cuda_service_stream_requirement(const ExecutionPlanState& state,
                                           const CudaComputeStreamPlanRecord& stream) {
  if (stream.partition || !stream.green_context_id) {
    return false;
  }
  const auto requirements = cuda_service_stream_requirements(state);
  return std::binary_search(
      requirements.begin(),
      requirements.end(),
      CudaServiceStreamRequirement{
          .process_group = stream.process_group,
          .device = stream.device,
          .owner_id = stream.owner_id,
      },
      [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.process_group.value, lhs.device.value, lhs.owner_id) <
               std::tie(rhs.process_group.value, rhs.device.value, rhs.owner_id);
      });
}

}  // namespace holoscan::detail
