/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/physical_resource_lowering.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "compiler/detail/checked_arithmetic.hpp"
#include "compiler/execution_plan_contracts.hpp"
#include "compiler/execution_plan_identity.hpp"
#include "compiler/physical_plan_lowering_internal.hpp"

namespace holoscan::detail {
namespace {

struct TensorStreamCandidate {
  std::size_t resource_index{};
  EdgeId edge{};
  VertexId source{};
  VertexId target{};
};

/// Record only strict process-local Tensor chains on the existing process/device stream.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void assign_linear_tensor_streams(
    const PhysicalPlanLoweringResult& result,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    const std::vector<TensorDeviceResourcePlanRecord>& resources,
    std::vector<CudaComputeStreamPlanRecord>& streams) {
  std::vector<TensorStreamCandidate> candidates;
  candidates.reserve(result.payload_edges.size());
  std::vector<std::size_t> incoming(process_placements.size(), 0U);
  std::vector<std::size_t> outgoing(process_placements.size(), 0U);
  std::vector<std::optional<std::size_t>> resource_by_vertex(process_placements.size());
  std::vector<bool> resource_ambiguous(process_placements.size(), false);
  const auto bind_resource = [&](VertexId vertex, std::size_t resource_index) {
    auto& selected = resource_by_vertex[static_cast<std::size_t>(vertex.value - 1U)];
    if (selected && *selected != resource_index) {
      resource_ambiguous[static_cast<std::size_t>(vertex.value - 1U)] = true;
    } else {
      selected = resource_index;
    }
  };
  for (const auto& allocation : result.tensor_allocations) {
    if (!allocation.output || allocation.output.value() > result.ports.size()) {
      continue;
    }
    const auto& port = result.ports[static_cast<std::size_t>(allocation.output.value() - 1U)];
    const auto resource = std::find_if(resources.begin(), resources.end(), [&](const auto& row) {
      return row.process_group == allocation.allocator_process_group &&
             row.device == allocation.device;
    });
    if (port.id == allocation.output && port.owner &&
        port.owner.value <= process_placements.size() && resource != resources.end()) {
      bind_resource(port.owner,
                    static_cast<std::size_t>(std::distance(resources.begin(), resource)));
    }
  }
  for (const auto& allocation : result.tensor_field_allocations) {
    if (!allocation.output || allocation.output.value() > result.ports.size()) {
      continue;
    }
    const auto& port = result.ports[static_cast<std::size_t>(allocation.output.value() - 1U)];
    const auto resource = std::find_if(resources.begin(), resources.end(), [&](const auto& row) {
      return row.process_group == allocation.allocator_process_group &&
             row.device == allocation.device;
    });
    if (port.id == allocation.output && port.owner &&
        port.owner.value <= process_placements.size() && resource != resources.end()) {
      bind_resource(port.owner,
                    static_cast<std::size_t>(std::distance(resources.begin(), resource)));
    }
  }
  for (const auto& pool : result.tensor_interface_pools) {
    const auto resource = std::find_if(resources.begin(), resources.end(), [&](const auto& row) {
      return row.process_group == pool.allocator_process_group && row.device == pool.device;
    });
    if (resource == resources.end()) {
      continue;
    }
    PortId resolved = pool.graph_port;
    for (std::size_t depth = 0U; depth <= result.interface_bindings.size(); ++depth) {
      if (!resolved || resolved.value() > result.ports.size()) {
        break;
      }
      const auto& port = result.ports[static_cast<std::size_t>(resolved.value() - 1U)];
      if (port.id != resolved) {
        break;
      }
      if (port.owner_kind == PortOwnerKind::kOperator && port.owner &&
          port.owner.value <= process_placements.size()) {
        bind_resource(port.owner,
                      static_cast<std::size_t>(std::distance(resources.begin(), resource)));
        break;
      }
      const auto alias = std::find_if(result.interface_bindings.begin(),
                                      result.interface_bindings.end(),
                                      [&](const auto& row) { return row.public_port == resolved; });
      if (alias == result.interface_bindings.end()) {
        break;
      }
      resolved = alias->internal_port;
    }
  }

  for (const auto& edge : result.payload_edges) {
    if (edge.disposition != PlanDisposition::kAccepted ||
        edge.locality != ChannelLocalityKind::kProcessLocal ||
        !is_tensor_backed_payload(edge.payload_identity) || edge.fanout != 1U || !edge.source ||
        !edge.target || edge.source.value() > result.ports.size() ||
        edge.target.value() > result.ports.size()) {
      continue;
    }
    const auto& source = result.ports[static_cast<std::size_t>(edge.source.value() - 1U)];
    const auto& target = result.ports[static_cast<std::size_t>(edge.target.value() - 1U)];
    if (source.id != edge.source || target.id != edge.target ||
        source.owner_kind != PortOwnerKind::kOperator ||
        target.owner_kind != PortOwnerKind::kOperator || !source.owner || !target.owner ||
        source.owner.value > process_placements.size() ||
        target.owner.value > process_placements.size()) {
      continue;
    }
    const auto source_group = process_group_for(process_placements, source.owner);
    const auto target_group = process_group_for(process_placements, target.owner);
    if (!source_group || source_group != target_group) {
      continue;
    }
    ++outgoing[static_cast<std::size_t>(source.owner.value - 1U)];
    ++incoming[static_cast<std::size_t>(target.owner.value - 1U)];
    const auto service = std::find_if(result.process_local_channel_services.begin(),
                                      result.process_local_channel_services.end(),
                                      [&](const auto& row) {
                                        return row.source == edge.source &&
                                               row.queue_policy == edge.queue_policy &&
                                               row.tensor_payload;
                                      });
    const auto allocation = std::find_if(
        result.tensor_allocations.begin(), result.tensor_allocations.end(), [&](const auto& row) {
          return row.output == edge.source && row.allocator_process_group == *source_group;
        });
    const auto field_allocation = std::find_if(
        result.tensor_field_allocations.begin(),
        result.tensor_field_allocations.end(),
        [&](const auto& row) {
          return row.output == edge.source && row.allocator_process_group == *source_group &&
                 requires_explicit_device(row.memory_kind) && row.device.valid();
        });
    if (service == result.process_local_channel_services.end() ||
        service->delivery_kind != ProcessLocalChannelDeliveryKind::kDirectRetained ||
        !service->tensor_payload) {
      continue;
    }
    const bool raw_tensor = edge.payload_identity.kind == PayloadIdentityKind::kTensor;
    const bool descriptor_tensor =
        edge.payload_identity.kind == PayloadIdentityKind::kSchema &&
        edge.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
    const bool descriptor_tensor_fields = descriptor_tensor && !service->tensor_fields.empty() &&
                                          field_allocation != result.tensor_field_allocations.end();
    const auto device = descriptor_tensor_fields ? field_allocation->device
                        : allocation != result.tensor_allocations.end() &&
                                requires_explicit_device(allocation->memory_kind) &&
                                allocation->device.valid()
                            ? allocation->device
                            : DeviceId{};
    const bool storage_contract_matches =
        raw_tensor && allocation != result.tensor_allocations.end()
            ? service->tensor_pool_id == allocation->pool_id &&
                  service->tensor_max_byte_span == allocation->bounds.max_byte_span &&
                  service->tensor_memory_kind == allocation->memory_kind &&
                  service->tensor_device == allocation->device
            : descriptor_tensor &&
                  ((allocation != result.tensor_allocations.end() &&
                    service->tensor_fields.empty() &&
                    service->tensor_pool_id == allocation->pool_id &&
                    service->tensor_max_byte_span == allocation->bounds.max_byte_span &&
                    service->tensor_memory_kind == allocation->memory_kind &&
                    service->tensor_device == allocation->device) ||
                   (descriptor_tensor_fields && service->tensor_pool_id.empty() &&
                    service->tensor_max_byte_span == 0U &&
                    service->tensor_memory_kind == MemoryKind::kUnknown &&
                    !service->tensor_device.valid()));
    if (!device.valid() || !storage_contract_matches) {
      continue;
    }
    const auto resource = std::find_if(resources.begin(), resources.end(), [&](const auto& row) {
      return row.process_group == *source_group && row.device == device;
    });
    if (resource == resources.end()) {
      continue;
    }
    const std::size_t resource_index =
        static_cast<std::size_t>(std::distance(resources.begin(), resource));
    bind_resource(source.owner, resource_index);
    bind_resource(target.owner, resource_index);
    candidates.push_back(TensorStreamCandidate{
        .resource_index = resource_index,
        .edge = edge.edge,
        .source = source.owner,
        .target = target.owner,
    });
  }

  for (const auto& candidate : candidates) {
    const std::size_t source_index = static_cast<std::size_t>(candidate.source.value - 1U);
    const std::size_t target_index = static_cast<std::size_t>(candidate.target.value - 1U);
    if (outgoing[source_index] == 1U && incoming[target_index] == 1U &&
        !resource_ambiguous[source_index] && !resource_ambiguous[target_index] &&
        resource_by_vertex[source_index] == candidate.resource_index &&
        resource_by_vertex[target_index] == candidate.resource_index) {
      streams[candidate.resource_index].fused_linear_edges.push_back(candidate.edge);
    }
  }
  for (auto& stream : streams) {
    std::sort(stream.fused_linear_edges.begin(),
              stream.fused_linear_edges.end(),
              [](EdgeId left, EdgeId right) { return left.value < right.value; });
  }
}

/// Reject an Operator whose Tensor inputs and outputs require distinct compute streams.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void validate_operator_device_obligations(PhysicalPlanLoweringResult& result) {
  std::vector<std::optional<DeviceId>> device_by_vertex;
  std::vector<PortId> source_by_vertex;
  for (const auto& port : result.ports) {
    if (port.owner) {
      device_by_vertex.resize(
          std::max(device_by_vertex.size(), static_cast<std::size_t>(port.owner.value)));
      source_by_vertex.resize(device_by_vertex.size());
    }
  }
  std::unordered_set<std::uint64_t> rejected_vertices;
  const auto bind_device = [&](VertexId owner, PortId port, DeviceId device) {
    if (!owner || !device.valid() || owner.value > device_by_vertex.size()) {
      return;
    }
    auto& selected = device_by_vertex[static_cast<std::size_t>(owner.value - 1U)];
    auto& selected_source = source_by_vertex[static_cast<std::size_t>(owner.value - 1U)];
    if (!selected) {
      selected = device;
      selected_source = port;
      return;
    }
    if (*selected == device || !rejected_vertices.insert(owner.value).second) {
      return;
    }
    reject_physical_plan(result,
                         kTensorDeviceBindingInvalid,
                         DiagnosticSource{.vertex = owner, .port = port},
                         "/vertices/" + std::to_string(owner.value) + "/tensor_device",
                         "one CUDA device across all Operator Tensor obligations",
                         "first_port=" + std::to_string(selected_source.value()) +
                             ",first_device=" + std::to_string(selected->value) +
                             ",conflicting_port=" + std::to_string(port.value()) +
                             ",conflicting_device=" + std::to_string(device.value) +
                             ",reason=operator_device_conflict");
  };
  const auto port_owner = [&](PortId id) -> VertexId {
    if (!id || id.value() > result.ports.size()) {
      return {};
    }
    const auto& port = result.ports[static_cast<std::size_t>(id.value() - 1U)];
    return port.id == id && port.owner_kind == PortOwnerKind::kOperator ? port.owner : VertexId{};
  };
  std::unordered_map<std::uint64_t, PortId> interface_aliases;
  interface_aliases.reserve(result.interface_bindings.size());
  for (const auto& binding : result.interface_bindings) {
    interface_aliases.emplace(binding.public_port.value(), binding.internal_port);
  }
  const auto interface_owner = [&](PortId id) -> VertexId {
    for (std::size_t depth = 0U; depth <= result.interface_bindings.size(); ++depth) {
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

  for (const auto& allocation : result.tensor_allocations) {
    bind_device(port_owner(allocation.output), allocation.output, allocation.device);
  }
  for (const auto& allocation : result.tensor_field_allocations) {
    bind_device(port_owner(allocation.output), allocation.output, allocation.device);
  }
  std::unordered_map<std::string_view, const ProcessLocalChannelServicePlanRecord*>
      process_local_service_by_id;
  process_local_service_by_id.reserve(result.process_local_channel_services.size());
  for (const auto& service : result.process_local_channel_services) {
    process_local_service_by_id.emplace(service.service_id, &service);
  }
  for (const auto& subscriber : result.process_local_channel_subscribers) {
    const auto service = process_local_service_by_id.find(subscriber.service_id);
    if (service != process_local_service_by_id.end() &&
        service->second->source == subscriber.source && service->second->tensor_payload) {
      const auto view = tensor_service_contract_view(*service->second);
      for_each_tensor_service_storage(view, [&](TensorServiceStorageView storage) {
        bind_device(port_owner(subscriber.target), subscriber.target, storage.device);
      });
    }
  }
  for (const auto& service : result.cross_process_channel_services) {
    if (service.tensor_payload) {
      const auto view = tensor_service_contract_view(service);
      const auto bind_target = [&](const PortId target) {
        for_each_tensor_service_storage(view, [&](TensorServiceStorageView storage) {
          bind_device(port_owner(target), target, storage.device);
        });
      };
      if (!view.tensor_fields.empty() || descriptor_tensor_schema_service(view)) {
        for_each_tensor_service_storage(view, [&](TensorServiceStorageView storage) {
          bind_device(port_owner(service.source), service.source, storage.device);
        });
      }
      bind_target(service.target);
      for (const auto& subscriber : result.cross_process_channel_subscribers) {
        if (subscriber.service_id == service.service_id && subscriber.source == service.source &&
            subscriber.target != service.target) {
          bind_target(subscriber.target);
        }
      }
    }
  }
  for (const auto& pool : result.tensor_interface_pools) {
    bind_device(interface_owner(pool.graph_port), pool.graph_port, pool.device);
  }
}

}  // namespace

// Device-resource aggregation is one compile-cold join across all Tensor pool families.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void lower_device_resources(PhysicalPlanLoweringResult& result, PlanGenerationId plan_generation,
                            std::span<const VertexProcessPlacementPlanRecord> process_placements) {
  std::vector<TensorDeviceResourcePlanRecord> device_resources;
  const auto append_device_pool = [&](ProcessGroupId process_group,
                                      DeviceId device,
                                      std::string_view allocator_root,
                                      std::uint32_t slots,
                                      std::uint32_t loans,
                                      std::uint32_t controls,
                                      std::uint32_t readiness,
                                      std::uint32_t releases,
                                      std::uint32_t retirements,
                                      std::uint32_t quarantines) -> bool {
    if (!device.valid()) {
      return true;
    }
    auto found =
        std::find_if(device_resources.begin(), device_resources.end(), [&](const auto& resource) {
          return resource.process_group == process_group && resource.device == device;
        });
    if (found == device_resources.end()) {
      device_resources.push_back(TensorDeviceResourcePlanRecord{
          .plan_generation = plan_generation,
          .run_generation = 0U,
          .process_group = process_group,
          .device = device,
          .allocator_root_id = std::string(allocator_root),
          .materialize_before = LifecycleStage::kArm,
          .fallback_rejected = true,
      });
      found = std::prev(device_resources.end());
    }
    std::uint32_t pool_count{};
    std::uint32_t slot_capacity{};
    std::uint32_t loan_capacity{};
    std::uint32_t control_capacity{};
    std::uint32_t readiness_capacity{};
    std::uint32_t release_capacity{};
    std::uint32_t retirement_capacity{};
    std::uint32_t quarantine_capacity{};
    if (found->allocator_root_id != allocator_root ||
        !checked_add(found->pool_count, 1U, pool_count) ||
        !checked_add(found->slot_capacity, slots, slot_capacity) ||
        !checked_add(found->loan_credit_capacity, loans, loan_capacity) ||
        !checked_add(found->control_capacity, controls, control_capacity) ||
        !checked_add(found->readiness_capacity, readiness, readiness_capacity) ||
        !checked_add(found->release_capacity, releases, release_capacity) ||
        !checked_add(found->retirement_capacity, retirements, retirement_capacity) ||
        !checked_add(found->quarantine_capacity, quarantines, quarantine_capacity)) {
      return false;
    }
    found->pool_count = pool_count;
    found->slot_capacity = slot_capacity;
    found->loan_credit_capacity = loan_capacity;
    found->control_capacity = control_capacity;
    found->readiness_capacity = readiness_capacity;
    found->release_capacity = release_capacity;
    found->retirement_capacity = retirement_capacity;
    found->quarantine_capacity = quarantine_capacity;
    return true;
  };
  bool device_resources_valid = true;
  validate_operator_device_obligations(result);
  for (const auto& pool : result.tensor_allocations) {
    device_resources_valid =
        device_resources_valid && append_device_pool(pool.allocator_process_group,
                                                     pool.device,
                                                     pool.allocator_root_id,
                                                     pool.slot_count,
                                                     pool.max_outstanding_loans,
                                                     pool.control_capacity,
                                                     pool.readiness_capacity,
                                                     pool.release_capacity,
                                                     pool.retirement_capacity,
                                                     pool.quarantine_capacity);
  }
  for (const auto& pool : result.tensor_field_allocations) {
    device_resources_valid =
        device_resources_valid && append_device_pool(pool.allocator_process_group,
                                                     pool.device,
                                                     pool.allocator_root_id,
                                                     pool.slot_count,
                                                     pool.max_outstanding_loans,
                                                     pool.control_capacity,
                                                     pool.readiness_capacity,
                                                     pool.release_capacity,
                                                     pool.retirement_capacity,
                                                     pool.quarantine_capacity);
  }
  for (const auto& pool : result.tensor_interface_pools) {
    device_resources_valid =
        device_resources_valid && append_device_pool(pool.allocator_process_group,
                                                     pool.device,
                                                     pool.allocator_root_id,
                                                     pool.slot_count,
                                                     pool.publisher_loan_capacity,
                                                     pool.control_capacity,
                                                     pool.readiness_capacity,
                                                     pool.release_capacity,
                                                     pool.retirement_capacity,
                                                     pool.quarantine_capacity);
  }
  // A stream-ordered cross-process Tensor service can need a process-local stream without owning
  // a payload pool. Materialize those resources from the same per-group/device predicate used by
  // runtime validation so lowering cannot emit an inadmissible zero-pool resource.
  for (const auto& service : result.cross_process_channel_services) {
    std::vector<DeviceId> candidate_devices;
    const auto append_candidate = [&](DeviceId device) {
      if (device.valid() && std::find(candidate_devices.begin(), candidate_devices.end(), device) ==
                                candidate_devices.end()) {
        candidate_devices.push_back(device);
      }
    };
    for_each_tensor_service_storage(
        tensor_service_contract_view(service),
        [&](TensorServiceStorageView storage) { append_candidate(storage.device); });
    if (candidate_devices.empty()) {
      continue;
    }
    std::vector<ProcessGroupId> process_groups = service.target_process_groups;
    if (service.payload_identity.kind == PayloadIdentityKind::kSchema &&
        service.payload_identity.payload_kind == PayloadKind::kDescriptorBacked) {
      process_groups.push_back(service.source_process_group);
    }
    for (const DeviceId device : candidate_devices) {
      for (const ProcessGroupId process_group : process_groups) {
        if (!cross_process_tensor_service_requires_stream_for_device_in_group(
                service, process_group, device)) {
          continue;
        }
        const auto existing = std::find_if(
            device_resources.begin(), device_resources.end(), [&](const auto& resource) {
              return resource.process_group == process_group && resource.device == device;
            });
        if (existing == device_resources.end()) {
          device_resources.push_back(TensorDeviceResourcePlanRecord{
              .plan_generation = plan_generation,
              .run_generation = 0U,
              .process_group = process_group,
              .device = device,
              .allocator_root_id = tensor_allocator_root_id(plan_generation, process_group, device),
              .materialize_before = LifecycleStage::kArm,
              .fallback_rejected = true,
          });
        }
      }
    }
  }
  if (!device_resources_valid) {
    reject_physical_plan(result, kTensorOverflow);
    return;
  }
  std::sort(
      device_resources.begin(), device_resources.end(), [](const auto& left, const auto& right) {
        return left.process_group == right.process_group
                   ? left.device.value < right.device.value
                   : left.process_group.value < right.process_group.value;
      });
  result.cuda_compute_streams.reserve(device_resources.size());
  for (const auto& resource : device_resources) {
    result.cuda_compute_streams.push_back(CudaComputeStreamPlanRecord{
        .plan_generation = resource.plan_generation,
        .run_generation = resource.run_generation,
        .process_group = resource.process_group,
        .device = resource.device,
        .owner_id = resource.allocator_root_id,
        .partition = std::nullopt,
        .stream_id = tensor_compute_stream_id(
            resource.plan_generation, resource.process_group, resource.device),
        .green_context_id = std::nullopt,
        .materialize_before = resource.materialize_before,
    });
  }
  assign_linear_tensor_streams(
      result, process_placements, device_resources, result.cuda_compute_streams);
  result.tensor_device_resources = std::move(device_resources);
}

}  // namespace holoscan::detail
