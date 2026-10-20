/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/operator_cuda_device_obligations.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "compiler/execution_plan_contracts.hpp"

namespace holoscan::detail {

namespace {

// Match tensor_service_storage_requires_operator_cuda_stream for interface pool storage, whether
// the pool backs a raw Tensor or a descriptor-schema field.
[[nodiscard]] bool interface_pool_storage_requires_operator_cuda_stream(
    const TensorInterfacePoolPlanRecord& pool) noexcept {
  switch (pool.memory_kind) {
    case MemoryKind::kPinnedHost:
    case MemoryKind::kCudaDevice:
    case MemoryKind::kCudaManaged:
      return true;
    case MemoryKind::kHost:
    case MemoryKind::kUnknown:
    default:
      return false;
  }
}

}  // namespace

// Keep the physical-record walk together so lowering and validation use the same CUDA-activity
// inventory; the local callbacks share port resolution and device-admission rules across families.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::vector<OperatorCudaDeviceObligation> collect_operator_cuda_device_obligations(
    const OperatorCudaDeviceObligationView& view) {
  std::vector<OperatorCudaDeviceObligation> result;
  const auto port_owner = [&](PortId id) -> VertexId {
    if (!id || id.value() > view.ports.size()) {
      return {};
    }
    const auto& port = view.ports[static_cast<std::size_t>(id.value() - 1U)];
    return port.id == id && port.owner_kind == PortOwnerKind::kOperator ? port.owner : VertexId{};
  };

  std::unordered_map<std::uint64_t, PortId> interface_aliases;
  interface_aliases.reserve(view.interface_bindings.size());
  for (const auto& binding : view.interface_bindings) {
    interface_aliases.emplace(binding.public_port.value(), binding.internal_port);
  }
  const auto interface_owner = [&](PortId id) -> VertexId {
    for (std::size_t depth = 0U; depth <= view.interface_bindings.size(); ++depth) {
      const VertexId owner = port_owner(id);
      if (owner) {
        return owner;
      }
      const auto alias = interface_aliases.find(id.value());
      if (alias == interface_aliases.end()) {
        return {};
      }
      id = alias->second;
    }
    return {};
  };
  const auto append = [&](VertexId owner, PortId source, DeviceId device) {
    if (owner && device.valid()) {
      result.push_back(OperatorCudaDeviceObligation{
          .operator_vertex = owner,
          .source_port = source,
          .device = device,
      });
    }
  };
  const auto append_service = [&](PortId port,
                                  const auto& service,
                                  std::optional<TensorProjectionRecord> projection = std::nullopt) {
    const auto contract = tensor_service_contract_view(service);
    if (projection && !service.tensor_fields.empty()) {
      for (const auto& field : service.tensor_fields) {
        if (field.field_id == projection->field_id &&
            field.tensor_index == projection->tensor_index &&
            tensor_service_storage_requires_operator_cuda_stream(contract, field.memory_kind)) {
          append(port_owner(port), port, field.device);
        }
      }
      return;
    }
    for_each_tensor_service_storage(contract, [&](TensorServiceStorageView storage) {
      if (tensor_service_storage_requires_operator_cuda_stream(contract, storage.memory_kind)) {
        append(port_owner(port), port, storage.device);
      }
    });
  };

  for (const auto& allocation : view.tensor_allocations) {
    append(port_owner(allocation.output), allocation.output, allocation.device);
  }
  for (const auto& allocation : view.tensor_field_allocations) {
    append(port_owner(allocation.output), allocation.output, allocation.device);
  }

  std::unordered_map<std::string_view, const ProcessLocalChannelServicePlanRecord*>
      local_service_by_id;
  local_service_by_id.reserve(view.process_local_services.size());
  for (const auto& service : view.process_local_services) {
    local_service_by_id.emplace(service.service_id, &service);
    if (service.tensor_payload && service.payload_identity.kind == PayloadIdentityKind::kTensor &&
        service.tensor_pool_id.empty()) {
      append_service(service.source, service);
    }
  }
  for (const auto& subscriber : view.process_local_subscribers) {
    const auto service = local_service_by_id.find(subscriber.service_id);
    if (service != local_service_by_id.end() && service->second->source == subscriber.source &&
        service->second->tensor_payload) {
      append_service(subscriber.target, *service->second, subscriber.tensor_projection);
    }
  }

  for (const auto& service : view.cross_process_services) {
    if (!service.tensor_payload) {
      continue;
    }
    if (!service.tensor_fields.empty() || descriptor_tensor_schema_service(service)) {
      append_service(service.source, service);
    }
    for (const auto& subscriber : view.cross_process_subscribers) {
      if (subscriber.service_id == service.service_id && subscriber.source == service.source) {
        append_service(subscriber.target, service, subscriber.tensor_projection);
      }
    }
  }

  for (const auto& pool : view.tensor_interface_pools) {
    if (!pool.imported) {
      append(interface_owner(pool.graph_port), pool.graph_port, pool.device);
    } else if (interface_pool_storage_requires_operator_cuda_stream(pool)) {
      // An imported topic has no owning Operator, so its port never resolves to one. Its
      // consumers read the pool storage and need a stream on the pool's device.
      for (const auto& edge : view.payload_edges) {
        if (edge.source == pool.graph_port) {
          append(interface_owner(edge.target), edge.target, pool.device);
        }
      }
    }
  }
  return result;
}

}  // namespace holoscan::detail
