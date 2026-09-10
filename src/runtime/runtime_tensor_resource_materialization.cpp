/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_tensor_resource_materialization.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "../compiler/detail/checked_arithmetic.hpp"
#include "../compiler/execution_plan_contracts.hpp"
#include "../compiler/execution_plan_lookup.hpp"
#include "../compiler/execution_plan_state.hpp"
#include "../core/plan_record_contracts.hpp"
#include "graph_exec.hpp"
#include "runtime_cuda_execution_resources.hpp"
#include "runtime_worker.hpp"

namespace holoscan::detail {

namespace {

constexpr std::size_t kTensorResourceErrorCapacity = 192U;

}  // namespace

Error tensor_resource_error(std::string_view context, std::string_view identity_kind,
                            std::string_view identity) noexcept {
  std::array<char, kTensorResourceErrorCapacity> buffer{};
  std::size_t size = 0U;
  const auto append = [&](std::string_view text) noexcept {
    const std::size_t available = buffer.size() - size;
    const std::size_t count = std::min(available, text.size());
    std::copy_n(text.data(), count, buffer.data() + size);
    size += count;
  };
  append("PROFILE_MATERIALIZATION_FAILED: ");
  append(context);
  if (!identity_kind.empty() && !identity.empty()) {
    append(" [");
    append(identity_kind);
    append("=");
    append(identity);
    append("]");
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return Error{ErrorCode::kFailure, std::string(buffer.data(), size)};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    // Error reporting must not turn a bounded setup failure into process termination.
    return Error{ErrorCode::kFailure};
  }
#endif
}

Error tensor_resource_error(std::string_view context, std::string_view identity_kind,
                            std::uint64_t identity) noexcept {
  std::array<char, std::numeric_limits<std::uint64_t>::digits10 + 2U> number{};
  const auto converted = std::to_chars(number.data(), number.data() + number.size(), identity);
  const std::string_view text =
      converted.ec == std::errc()
          ? std::string_view(number.data(), static_cast<std::size_t>(converted.ptr - number.data()))
          : std::string_view("unknown");
  return tensor_resource_error(context, identity_kind, text);
}

namespace {

using DeviceResource = RuntimeTensorDeviceResource;

[[nodiscard]] bool canonical_resource_order(const TensorDeviceResourcePlanRecord& left,
                                            const TensorDeviceResourcePlanRecord& right) noexcept {
  return std::tie(left.process_group.value, left.device.value) <
         std::tie(right.process_group.value, right.device.value);
}

struct ResourceTotals {
  std::uint32_t pools{};
  std::uint32_t slots{};
  std::uint32_t loans{};
  std::uint32_t controls{};
  std::uint32_t readiness{};
  std::uint32_t releases{};
  std::uint32_t retirements{};
  std::uint32_t quarantines{};
};

[[nodiscard]] bool append_totals(ResourceTotals& totals, std::uint32_t slots, std::uint32_t loans,
                                 std::uint32_t controls, std::uint32_t readiness,
                                 std::uint32_t releases, std::uint32_t retirements,
                                 std::uint32_t quarantines) noexcept {
  return checked_add(totals.pools, 1U, totals.pools) &&
         checked_add(totals.slots, slots, totals.slots) &&
         checked_add(totals.loans, loans, totals.loans) &&
         checked_add(totals.controls, controls, totals.controls) &&
         checked_add(totals.readiness, readiness, totals.readiness) &&
         checked_add(totals.releases, releases, totals.releases) &&
         checked_add(totals.retirements, retirements, totals.retirements) &&
         checked_add(totals.quarantines, quarantines, totals.quarantines);
}

struct OperatorStreamSelection {
  cudaStream_t stream{};
  DeviceId device{};
};

[[nodiscard]] bool valid_fused_linear_edges(const ExecutionPlanState& plan,
                                            const CudaComputeStreamPlanRecord& stream) noexcept {
  if (std::adjacent_find(stream.fused_linear_edges.begin(),
                         stream.fused_linear_edges.end(),
                         [](EdgeId left, EdgeId right) { return left.value >= right.value; }) !=
      stream.fused_linear_edges.end()) {
    return false;
  }
  for (const EdgeId edge_id : stream.fused_linear_edges) {
    const auto edge = std::find_if(plan.payload_edges.begin(),
                                   plan.payload_edges.end(),
                                   [&](const auto& row) { return row.edge == edge_id; });
    const bool tensor_backed =
        edge != plan.payload_edges.end() &&
        (edge->payload_identity.kind == PayloadIdentityKind::kTensor ||
         (edge->payload_identity.kind == PayloadIdentityKind::kSchema &&
          edge->payload_identity.payload_kind == PayloadKind::kDescriptorBacked));
    if (!edge_id || edge == plan.payload_edges.end() ||
        edge->disposition != PlanDisposition::kAccepted ||
        edge->locality != ChannelLocalityKind::kProcessLocal || !tensor_backed ||
        edge->fanout != 1U || !edge->source || !edge->target ||
        edge->target.value() > plan.ports.size()) {
      return false;
    }
    const auto* allocation = find_tensor_allocation_by_output(plan, edge->source);
    const auto field_allocations = tensor_field_allocations_by_output(plan, edge->source);
    const bool matching_field_allocation =
        std::any_of(field_allocations.begin(), field_allocations.end(), [&](const auto& row) {
          return row.allocator_process_group == stream.process_group && row.device == stream.device;
        });
    const bool matching_allocation =
        (allocation != nullptr && allocation->allocator_process_group == stream.process_group &&
         allocation->device == stream.device) ||
        (edge->payload_identity.kind == PayloadIdentityKind::kSchema &&
         edge->payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
         matching_field_allocation);
    const auto& target = plan.ports[static_cast<std::size_t>(edge->target.value() - 1U)];
    const auto placement =
        std::find_if(plan.vertex_process_placements.begin(),
                     plan.vertex_process_placements.end(),
                     [&](const auto& row) { return row.vertex == target.owner; });
    if (!matching_allocation || target.id != edge->target ||
        target.owner_kind != PortOwnerKind::kOperator ||
        placement == plan.vertex_process_placements.end() ||
        placement->process_group != stream.process_group) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool valid_device_resource_contract(const ExecutionPlanState& plan,
                                                  std::size_t index) noexcept {
  const auto& resource = plan.tensor_device_resources[index];
  const bool common =
      resource.plan_generation == plan.plan_generation && resource.run_generation == 0U &&
      resource.process_group && resource.device.valid() && !resource.allocator_root_id.empty() &&
      resource.materialize_before == LifecycleStage::kArm &&
      !resource.gpu_resident_backend_selected && resource.fallback_rejected &&
      (index == 0U || canonical_resource_order(plan.tensor_device_resources[index - 1U], resource));
  if (!common) {
    return false;
  }
  if (resource.pool_count != 0U) {
    return resource.slot_capacity != 0U && resource.control_capacity == resource.slot_capacity &&
           resource.readiness_capacity == resource.slot_capacity &&
           resource.release_capacity == resource.slot_capacity &&
           resource.retirement_capacity == resource.slot_capacity &&
           resource.quarantine_capacity == resource.slot_capacity;
  }
  const bool capacities_empty =
      resource.slot_capacity == 0U && resource.loan_credit_capacity == 0U &&
      resource.control_capacity == 0U && resource.readiness_capacity == 0U &&
      resource.release_capacity == 0U && resource.retirement_capacity == 0U &&
      resource.quarantine_capacity == 0U;
  return capacities_empty &&
         std::any_of(plan.cross_process_channel_services.begin(),
                     plan.cross_process_channel_services.end(),
                     [&](const auto& service) {
                       return cross_process_tensor_service_requires_stream_for_device_in_group(
                           service, resource.process_group, resource.device);
                     });
}

template <typename PoolRecord, typename LoanCapacity>
[[nodiscard]] expected<void, Error> append_pool_totals(
    std::span<const PoolRecord> pools, const TensorDeviceResourcePlanRecord& resource,
    ResourceTotals& totals, std::string_view& allocator_root, LoanCapacity loan_capacity,
    std::string_view overflow_detail) {
  for (const auto& pool : pools) {
    if (pool.device != resource.device || pool.allocator_process_group != resource.process_group) {
      continue;
    }
    if (allocator_root.empty()) {
      allocator_root = pool.allocator_root_id;
    }
    if (allocator_root != pool.allocator_root_id || !append_totals(totals,
                                                                   pool.slot_count,
                                                                   loan_capacity(pool),
                                                                   pool.control_capacity,
                                                                   pool.readiness_capacity,
                                                                   pool.release_capacity,
                                                                   pool.retirement_capacity,
                                                                   pool.quarantine_capacity)) {
      return make_unexpected(
          tensor_resource_error(overflow_detail, "resource", resource.allocator_root_id));
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> append_direct_pool_totals(
    const ExecutionPlanState& plan, const TensorDeviceResourcePlanRecord& resource,
    ResourceTotals& totals, std::string_view& allocator_root) {
  return append_pool_totals(
      std::span<const TensorAllocationPlanRecord>{plan.tensor_allocations},
      resource,
      totals,
      allocator_root,
      [](const auto& pool) { return pool.max_outstanding_loans; },
      "direct-pool resource totals overflow");
}

[[nodiscard]] expected<void, Error> append_field_pool_totals(
    const ExecutionPlanState& plan, const TensorDeviceResourcePlanRecord& resource,
    ResourceTotals& totals, std::string_view& allocator_root) {
  return append_pool_totals(
      std::span<const TensorFieldAllocationPlanRecord>{plan.tensor_field_allocations},
      resource,
      totals,
      allocator_root,
      [](const auto& pool) { return pool.max_outstanding_loans; },
      "field-pool resource totals overflow");
}

[[nodiscard]] expected<void, Error> append_interface_pool_totals(
    const ExecutionPlanState& plan, const TensorDeviceResourcePlanRecord& resource,
    ResourceTotals& totals, std::string_view& allocator_root) {
  return append_pool_totals(
      std::span<const TensorInterfacePoolPlanRecord>{plan.tensor_interface_pools},
      resource,
      totals,
      allocator_root,
      [](const auto& pool) { return pool.publisher_loan_capacity; },
      "interface-pool resource totals overflow");
}

[[nodiscard]] bool resource_totals_match(const TensorDeviceResourcePlanRecord& resource,
                                         const ResourceTotals& totals,
                                         std::string_view allocator_root) noexcept {
  const bool root_matches = resource.pool_count == 0U
                                ? allocator_root.empty()
                                : allocator_root == resource.allocator_root_id;
  return root_matches && totals.pools == resource.pool_count &&
         totals.slots == resource.slot_capacity && totals.loans == resource.loan_credit_capacity &&
         totals.controls == resource.control_capacity &&
         totals.readiness == resource.readiness_capacity &&
         totals.releases == resource.release_capacity &&
         totals.retirements == resource.retirement_capacity &&
         totals.quarantines == resource.quarantine_capacity;
}

[[nodiscard]] expected<void, Error> materialize_device_resources(
    const ExecutionPlanState& plan, ProcessGroupId process_group,
    std::vector<DeviceResource>& devices) {
  if (std::any_of(plan.cuda_compute_streams.begin(),
                  plan.cuda_compute_streams.end(),
                  [&](const auto& stream) { return !valid_fused_linear_edges(plan, stream); })) {
    return make_unexpected(tensor_resource_error("compute stream fusion contract mismatch"));
  }
  devices.reserve(plan.tensor_device_resources.size());
  for (std::size_t index = 0U; index < plan.tensor_device_resources.size(); ++index) {
    const auto& resource = plan.tensor_device_resources[index];
    if (!valid_device_resource_contract(plan, index)) {
      return make_unexpected(
          tensor_resource_error("device resource contract mismatch", "resource_index", index));
    }
    ResourceTotals totals;
    std::string_view allocator_root;
    auto direct = append_direct_pool_totals(plan, resource, totals, allocator_root);
    if (!direct) {
      return make_unexpected(std::move(direct.error()));
    }
    auto fields = append_field_pool_totals(plan, resource, totals, allocator_root);
    if (!fields) {
      return make_unexpected(std::move(fields.error()));
    }
    auto interface = append_interface_pool_totals(plan, resource, totals, allocator_root);
    if (!interface) {
      return make_unexpected(std::move(interface.error()));
    }
    if (!resource_totals_match(resource, totals, allocator_root)) {
      return make_unexpected(tensor_resource_error(
          "device resource totals mismatch", "resource", resource.allocator_root_id));
    }
    if (resource.process_group != process_group) {
      continue;
    }

    devices.push_back(&resource);
  }
  return {};
}

[[nodiscard]] cudaStream_t resource_stream(const RuntimeCudaExecutionResources& cuda_resources,
                                           ProcessGroupId group, DeviceId device,
                                           ExecutionPartitionId partition) noexcept {
  if (cudaStream_t stream = cuda_resources.partition_stream(group, device, partition);
      stream != nullptr) {
    return stream;
  }
  return cuda_resources.ordinary_stream(group, device);
}

[[nodiscard]] bool can_elide_native_readiness(const ExecutionPlanState& plan,
                                              const RuntimeCudaExecutionResources& cuda_resources,
                                              const TensorAllocationPlanRecord& allocation,
                                              cudaStream_t producer_stream) noexcept {
  if (producer_stream == nullptr) {
    return false;
  }
  bool found_edge = false;
  for (const auto& edge : plan.payload_edges) {
    if (edge.source != allocation.output || edge.disposition != PlanDisposition::kAccepted) {
      continue;
    }
    found_edge = true;
    const auto* fused_stream = find_cuda_compute_stream_by_fused_edge(
        plan, allocation.allocator_process_group, allocation.device, edge.edge);
    if (fused_stream == nullptr) {
      return false;
    }
    cudaStream_t planned_stream =
        fused_stream->partition
            ? cuda_resources.partition_stream(
                  fused_stream->process_group, fused_stream->device, *fused_stream->partition)
        : fused_stream->green_context_id
            ? nullptr
            : cuda_resources.ordinary_stream(fused_stream->process_group, fused_stream->device);
    if (planned_stream != producer_stream) {
      return false;
    }
    if (edge.locality != ChannelLocalityKind::kProcessLocal || !edge.target ||
        edge.target.value() > plan.ports.size()) {
      return false;
    }
    const auto& target = plan.ports[static_cast<std::size_t>(edge.target.value() - 1U)];
    if (target.id != edge.target || target.owner_kind != PortOwnerKind::kOperator) {
      return false;
    }
    const ProcessGroupId target_group = process_group_for(plan, target.owner);
    const auto* target_partition = find_execution_partition_by_member(plan, target.owner);
    if (!target_group || target_partition == nullptr ||
        resource_stream(cuda_resources, target_group, allocation.device, target_partition->id) !=
            producer_stream) {
      return false;
    }
  }
  return found_edge;
}

[[nodiscard]] bool interface_pool_belongs_to_vertex(const ExecutionPlanState& plan,
                                                    const TensorInterfacePoolPlanRecord& pool,
                                                    VertexId vertex) noexcept {
  PortId resolved = pool.graph_port;
  for (std::size_t depth = 0U; depth <= plan.interface_port_bindings.size(); ++depth) {
    const auto* port = find_port_by_id(plan, resolved);
    if (port != nullptr && port->owner == vertex) {
      return true;
    }
    const auto alias = std::find_if(plan.interface_port_bindings.begin(),
                                    plan.interface_port_bindings.end(),
                                    [&](const auto& row) { return row.public_port == resolved; });
    if (alias == plan.interface_port_bindings.end()) {
      return false;
    }
    resolved = alias->internal_port;
  }
  return false;
}

// Select a stream by joining the plan's exact process, device, and interface obligations.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<OperatorStreamSelection, Error> select_operator_stream(
    const ExecutionPlanState& plan, const RuntimeCudaExecutionResources& cuda_resources,
    const RuntimeOperatorInstance& instance) {
  const ProcessGroupId group = process_group_for(plan, instance.vertex);
  if (!group) {
    return make_unexpected(tensor_resource_error(
        "operator process placement missing", "operator", instance.operator_id));
  }
  cudaStream_t selected_stream = instance.compute_stream;
  DeviceId selected_device = instance.compute_device;
  const auto admit_device = [&](DeviceId device) noexcept -> bool {
    if (!device.valid()) {
      return true;
    }
    cudaStream_t candidate = resource_stream(cuda_resources, group, device, instance.partition);
    if (candidate == nullptr || (selected_stream != nullptr && selected_stream != candidate)) {
      return false;
    }
    selected_stream = candidate;
    selected_device = device;
    return true;
  };
  const auto admit_service_devices = [&](const auto& service) noexcept -> bool {
    const auto view = tensor_service_contract_view(service);
    bool admitted = true;
    const auto admit_if_required = [&](MemoryKind memory_kind, DeviceId device) noexcept {
      return !tensor_service_storage_requires_operator_cuda_stream(view, memory_kind) ||
             admit_device(device);
    };
    for_each_tensor_service_storage(view, [&](TensorServiceStorageView storage) {
      admitted = admitted && admit_if_required(storage.memory_kind, storage.device);
    });
    return admitted;
  };

  for (const auto& allocation : plan.tensor_allocations) {
    const auto* port = find_port_by_id(plan, allocation.output);
    if (port != nullptr && port->owner == instance.vertex && !admit_device(allocation.device)) {
      return make_unexpected(tensor_resource_error(
          "output device resource mismatch", "output", allocation.output.value()));
    }
  }
  for (const auto& allocation : plan.tensor_field_allocations) {
    const auto* port = find_port_by_id(plan, allocation.output);
    if (port != nullptr && port->owner == instance.vertex && !admit_device(allocation.device)) {
      return make_unexpected(tensor_resource_error(
          "field output device resource mismatch", "output", allocation.output.value()));
    }
  }
  for (const auto& subscriber : plan.process_local_channel_subscribers) {
    const auto* port = find_port_by_id(plan, subscriber.target);
    if (port == nullptr || port->owner != instance.vertex) {
      continue;
    }
    const auto* service = find_process_local_channel_service_by_id(plan, subscriber.service_id);
    if (service == nullptr || service->source != subscriber.source) {
      return make_unexpected(tensor_resource_error(
          "subscriber device resource mismatch", "output", subscriber.source.value()));
    }
    if (service->tensor_payload && !admit_service_devices(*service)) {
      return make_unexpected(tensor_resource_error(
          "subscriber device resource mismatch", "output", subscriber.source.value()));
    }
  }
  for (const auto& subscriber : plan.cross_process_channel_subscribers) {
    const auto* port = find_port_by_id(plan, subscriber.target);
    if (port == nullptr || port->owner != instance.vertex) {
      continue;
    }
    const auto* service = find_cross_process_channel_service_by_id(plan, subscriber.service_id);
    if (service == nullptr || service->source != subscriber.source ||
        (service->tensor_payload && !admit_service_devices(*service))) {
      return make_unexpected(
          tensor_resource_error("cross-process subscriber device resource mismatch",
                                "output",
                                subscriber.source.value()));
    }
  }
  for (const auto& service : plan.cross_process_channel_services) {
    const auto* port = find_port_by_id(plan, service.source);
    if (descriptor_tensor_schema_service(service) && port != nullptr &&
        port->owner == instance.vertex && !admit_service_devices(service)) {
      return make_unexpected(
          tensor_resource_error("descriptor-schema publisher device resource mismatch",
                                "output",
                                service.source.value()));
    }
  }
  for (const auto& pool : plan.tensor_interface_pools) {
    if (pool.imported && pool.device.valid() &&
        interface_pool_belongs_to_vertex(plan, pool, instance.vertex) &&
        !admit_device(pool.device)) {
      return make_unexpected(
          tensor_resource_error("interface device resource mismatch", "pool", pool.pool_id));
    }
  }
  return OperatorStreamSelection{.stream = selected_stream, .device = selected_device};
}

[[nodiscard]] expected<void, Error> bind_operator_streams(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    const RuntimeCudaExecutionResources& cuda_resources) {
  for (auto& instance : operators) {
    auto stream = select_operator_stream(plan, cuda_resources, instance);
    if (!stream) {
      return make_unexpected(std::move(stream.error()));
    }
    instance.compute_stream = stream->stream;
    instance.compute_device = stream->device;
  }
  return {};
}

[[nodiscard]] expected<cudaStream_t, Error> output_stream(
    const ExecutionPlanState& plan, const RuntimeCudaExecutionResources& cuda_resources,
    std::span<const DeviceResource> devices, const TensorAllocationPlanRecord& allocation) {
  if (!allocation.device.valid()) {
    return nullptr;
  }
  const auto resource = std::find_if(devices.begin(), devices.end(), [&](const auto* candidate) {
    return candidate->process_group == allocation.allocator_process_group &&
           candidate->device == allocation.device &&
           candidate->allocator_root_id == allocation.allocator_root_id;
  });
  if (resource == devices.end()) {
    return make_unexpected(tensor_resource_error(
        "output device resource missing", "output", allocation.output.value()));
  }
  const auto* port = find_port_by_id(plan, allocation.output);
  if (port == nullptr) {
    return make_unexpected(
        tensor_resource_error("output stream owner missing", "output", allocation.output.value()));
  }
  const auto* partition = find_execution_partition_by_member(plan, port->owner);
  if (partition == nullptr) {
    return make_unexpected(tensor_resource_error(
        "output stream partition missing", "output", allocation.output.value()));
  }
  cudaStream_t stream = resource_stream(
      cuda_resources, allocation.allocator_process_group, allocation.device, partition->id);
  if (stream == nullptr) {
    return make_unexpected(tensor_resource_error(
        "output stream materialization missing", "output", allocation.output.value()));
  }
  return stream;
}

}  // namespace

bool canonical_tensor_device_resource_order(const TensorDeviceResourcePlanRecord& left,
                                            const TensorDeviceResourcePlanRecord& right) noexcept {
  return canonical_resource_order(left, right);
}

expected<void, Error> materialize_tensor_device_resources(
    const ExecutionPlanState& plan, ProcessGroupId process_group,
    std::vector<RuntimeTensorDeviceResource>& devices) {
  return materialize_device_resources(plan, process_group, devices);
}

expected<void, Error> bind_tensor_operator_streams(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    const RuntimeCudaExecutionResources& cuda_resources) {
  return bind_operator_streams(plan, operators, cuda_resources);
}

expected<cudaStream_t, Error> tensor_output_stream(
    const ExecutionPlanState& plan, const RuntimeCudaExecutionResources& cuda_resources,
    std::span<const RuntimeTensorDeviceResource> devices,
    const TensorAllocationPlanRecord& allocation) {
  return output_stream(plan, cuda_resources, devices, allocation);
}

bool tensor_can_elide_native_readiness(const ExecutionPlanState& plan,
                                       const RuntimeCudaExecutionResources& cuda_resources,
                                       const TensorAllocationPlanRecord& allocation,
                                       cudaStream_t producer_stream) noexcept {
  return can_elide_native_readiness(plan, cuda_resources, allocation, producer_stream);
}

}  // namespace holoscan::detail
