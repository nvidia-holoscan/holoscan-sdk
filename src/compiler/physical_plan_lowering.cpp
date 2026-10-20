/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/physical_plan_lowering.hpp"

#include "compiler/channel_service_lowering.hpp"
#include "compiler/external_channel_lowering.hpp"
#include "compiler/physical_plan_lowering_internal.hpp"
#include "compiler/physical_plan_rate_validation.hpp"
#include "compiler/physical_resource_lowering.hpp"
#include "compiler/physical_topology_lowering.hpp"
#include "compiler/tensor_storage_lowering.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "compiler/detail/checked_arithmetic.hpp"
#include "compiler/snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

constexpr std::string_view kTensorDeviceBindingUnused{"PHYSICAL_TENSOR_DEVICE_BINDING_UNUSED"};

}  // namespace

// Keep ordering-sensitive admission, service materialization, and rejection projection in one
// compile-cold orchestrator; the extracted phases own their internal complexity.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PhysicalPlanLoweringResult lower_physical_plan(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const std::optional<Profile>> effective_profiles,
    const RuntimePackageCapabilityView& package_capabilities,
    std::span<const InputSelectionPlanRecord> input_selections,
    std::span<const BracketedWindowPlanRecord> bracketed_windows,
    std::span<const ChannelEndpointRecord> catalog_endpoints,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    std::span<const LifecycleRestartTargetPlanRecord> lifecycle_targets,
    std::span<const ExecutionPartitionRecord> execution_partitions,
    std::span<const TensorDeviceBindingRecord> tensor_device_bindings,
    bool force_holomq_process_local_inline, bool force_holomq_process_local_event_wake,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations,
    std::span<const ResolvedQueueSizing> resolved_queue_sizings,
    std::span<const QueueSizingRecipePlanRecord> queue_sizing_recipes,
    CudaComputeStreamPolicy stream_policy, const bool experimental_split_cut,
    const ProcessLocalEmitAdmissionScope emit_admission,
    const bool force_holomq_process_local_fanout,
    std::span<const LifecycleRestartClosurePlanRecord> lifecycle_closures,
    std::span<const TensorDeviceInheritanceRecord> tensor_device_inheritance) {
  PhysicalPlanLoweringResult result;
  auto topology = lower_physical_topology(result,
                                          snapshot,
                                          effective_profiles,
                                          input_selections,
                                          bracketed_windows,
                                          catalog_endpoints,
                                          process_placements,
                                          tensor_device_bindings,
                                          force_holomq_process_local_inline,
                                          force_holomq_process_local_event_wake,
                                          tensor_byte_span_derivations,
                                          resolved_queue_sizings,
                                          force_holomq_process_local_fanout,
                                          lifecycle_targets,
                                          lifecycle_closures,
                                          tensor_device_inheritance);
  const auto& device_by_port = topology.device_by_port;
  auto& consumed_device_bindings = topology.consumed_device_bindings;
  const auto rejected_tensor_fanout_sources =
      lower_preallocation_channel_services(result,
                                           snapshot,
                                           plan_generation,
                                           process_placements,
                                           topology,
                                           force_holomq_process_local_inline,
                                           tensor_byte_span_derivations);

  // Schema transport is admitted before allocation. A rejected staging contract must not leave
  // an apparently executable projected edge, or ordinary siblings from the same output, behind.
  // Index projected edges once; ordinary graphs need no additional join storage.
  std::unordered_map<std::uint64_t, std::uint64_t> unbound_projection_sources;
  for (const auto& edge : result.payload_edges) {
    if (edge.tensor_projection) {
      unbound_projection_sources.emplace(edge.edge.value, edge.source.value());
    }
  }
  if (!unbound_projection_sources.empty()) {
    for (const auto& subscriber : result.process_local_channel_subscribers) {
      unbound_projection_sources.erase(subscriber.edge.value);
    }
    for (const auto& subscriber : result.cross_process_channel_subscribers) {
      unbound_projection_sources.erase(subscriber.edge.value);
    }
  }
  std::unordered_set<std::uint64_t> rejected_schema_sources;
  for (const auto& [edge, source] : unbound_projection_sources) {
    assert(!result.errors.empty());
    rejected_schema_sources.insert(source);
  }
  if (!rejected_schema_sources.empty()) {
    std::unordered_set<std::uint64_t> rejected_edges;
    for (const auto& edge : result.payload_edges) {
      if (rejected_schema_sources.contains(edge.source.value())) {
        rejected_edges.insert(edge.edge.value);
      }
    }
    const auto rejected_source = [&](const auto& record) {
      return rejected_schema_sources.contains(record.source.value());
    };
    std::erase_if(result.payload_edges, rejected_source);
    std::erase_if(result.channel_locality,
                  [&](const auto& record) { return rejected_edges.contains(record.edge.value); });
    std::erase_if(result.process_local_channel_services, rejected_source);
    std::erase_if(result.process_local_channel_publishers, rejected_source);
    std::erase_if(result.process_local_channel_subscribers, rejected_source);
    std::erase_if(result.cross_process_channel_services, rejected_source);
    std::erase_if(result.cross_process_channel_publishers, rejected_source);
    std::erase_if(result.cross_process_channel_subscribers, rejected_source);
    for (const auto source : rejected_schema_sources) {
      const auto index = static_cast<std::size_t>(source - 1U);
      topology.delivery_by_output[index] = 0U;
      topology.transport_retention_by_output[index] = 0U;
      topology.cache_by_output[index] = 0U;
      topology.window_by_output[index] = 0U;
      topology.reuse_by_output[index] = false;
      topology.drain_by_output[index] = false;
      topology.copy_by_output[index] = false;
      topology.reexport_by_output[index] = false;
    }
    std::erase_if(topology.transport_retention_edges,
                  [&](auto edge) { return rejected_edges.contains(edge); });
  }

  lower_output_storage(result,
                       snapshot,
                       plan_generation,
                       process_placements,
                       execution_partitions,
                       topology,
                       tensor_byte_span_derivations,
                       input_selections,
                       bracketed_windows,
                       resolved_queue_sizings,
                       queue_sizing_recipes,
                       force_holomq_process_local_inline);

  lower_external_channel_services(result,
                                  snapshot,
                                  plan_generation,
                                  effective_profiles,
                                  package_capabilities,
                                  process_placements,
                                  lifecycle_targets,
                                  execution_partitions,
                                  topology);

  // A declared direct Tensor allocation that failed admission cannot authorize either a pooled
  // service or an unpooled pass-through service. Match the earlier profile-rejection behavior by
  // omitting its unrealizable physical edges while retaining the authored Graph and diagnostic.
  std::unordered_set<std::uint64_t> rejected_direct_tensor_sources;
  for (const auto& port : snapshot.ports) {
    const bool descriptor_tensor =
        port.payload_identity.kind == PayloadIdentityKind::kSchema &&
        port.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
    const bool multi_descriptor_tensor =
        descriptor_tensor && port.descriptor_tensor_field_count > 1U;
    const bool field_allocation_declared =
        multi_descriptor_tensor &&
        std::any_of(port.setup.tensor_fields.begin(),
                    port.setup.tensor_fields.end(),
                    [](const auto& field) {
                      return field.tensor_allocation.has_value() ||
                             field.tensor_byte_span_expression.has_value();
                    });
    const bool direct_allocation_declared =
        port.setup.tensor_allocation.has_value() ||
        port.graph_override.tensor_allocation.has_value() || field_allocation_declared ||
        find_port_tensor_byte_span_derivation(tensor_byte_span_derivations, port) != nullptr;
    if (port.direction != PortDirection::kOutput || port.owner_kind != PortOwnerKind::kOperator ||
        (port.payload_identity.kind != PayloadIdentityKind::kTensor && !descriptor_tensor) ||
        !direct_allocation_declared) {
      continue;
    }
    const bool allocation_admitted =
        multi_descriptor_tensor
            ? std::count_if(result.tensor_field_allocations.begin(),
                            result.tensor_field_allocations.end(),
                            [&](const auto& allocation) {
                              return allocation.output == PortId{port.id.value};
                            }) == static_cast<std::ptrdiff_t>(port.descriptor_tensor_field_count)
            : std::any_of(result.tensor_allocations.begin(),
                          result.tensor_allocations.end(),
                          [&](const auto& allocation) {
                            return allocation.output == PortId{port.id.value};
                          });
    const bool candidate_admitted =
        std::ranges::any_of(result.provider_allocation_plans, [&](const auto& provider) {
          return provider.output == PortId(port.id.value) &&
                 provider.geometry_mode == AllocationGeometryMode::kLifecycleCandidate &&
                 provider.tensor_template.has_value() &&
                 provider.disposition == PlanDisposition::kAccepted;
        });
    if (!allocation_admitted && !candidate_admitted) {
      rejected_direct_tensor_sources.insert(port.id.value);
    }
  }
  const auto rejected_placement_sources =
      lower_tensor_channel_services(result,
                                    snapshot,
                                    plan_generation,
                                    process_placements,
                                    topology,
                                    rejected_tensor_fanout_sources,
                                    rejected_direct_tensor_sources,
                                    force_holomq_process_local_inline,
                                    tensor_byte_span_derivations);
  rejected_direct_tensor_sources.insert(rejected_placement_sources.begin(),
                                        rejected_placement_sources.end());
  if (!rejected_direct_tensor_sources.empty()) {
    // Every unrealizable allocation or CUDA placement must already have a compile diagnostic.
    // This guard prevents a future admission branch from silently pruning executable edges.
    assert(!result.errors.empty());
    std::unordered_set<std::uint64_t> rejected_edges;
    for (const auto& edge : result.payload_edges) {
      if (rejected_direct_tensor_sources.contains(edge.source.value())) {
        rejected_edges.insert(edge.edge.value);
      }
    }
    std::erase_if(result.payload_edges, [&](const auto& edge) {
      return rejected_direct_tensor_sources.contains(edge.source.value());
    });
    std::erase_if(result.channel_locality, [&](const auto& locality) {
      return rejected_edges.contains(locality.edge.value);
    });
    const auto rejected_source = [&](const auto& record) {
      return rejected_direct_tensor_sources.contains(record.source.value());
    };
    std::erase_if(result.process_local_channel_services, rejected_source);
    std::erase_if(result.process_local_channel_publishers, rejected_source);
    std::erase_if(result.process_local_channel_subscribers, rejected_source);
    std::erase_if(result.cross_process_channel_services, rejected_source);
    std::erase_if(result.cross_process_channel_publishers, rejected_source);
    std::erase_if(result.cross_process_channel_subscribers, rejected_source);
  }

  // Decide wake attachment only after every compatible edge has joined its immutable service and
  // policy-specific sibling services are adjacent. Suppression is source-family-wide: one
  // unsupported member keeps every sibling service and subscriber attached to the WaitSet.
  finalize_channel_services(result,
                            snapshot,
                            execution_partitions,
                            lifecycle_targets,
                            lifecycle_closures,
                            force_holomq_process_local_event_wake,
                            experimental_split_cut,
                            emit_admission);

  validate_lifecycle_queue_geometry(result, snapshot, queue_sizing_recipes);

  // Aggregate device-backed pools only after final cover validation. The stream policy may change
  // compute-stream ownership, but partition count remains absent from every capacity equation.
  lower_device_resources(result,
                         snapshot,
                         topology,
                         plan_generation,
                         process_placements,
                         execution_partitions,
                         stream_policy);
  std::vector<std::pair<std::uint64_t, DeviceId>> unconsumed_device_bindings;
  unconsumed_device_bindings.reserve(device_by_port.size());
  for (const auto& [port, device] : device_by_port) {
    if (!consumed_device_bindings.contains(port)) {
      unconsumed_device_bindings.emplace_back(port, device);
    }
  }
  std::sort(unconsumed_device_bindings.begin(),
            unconsumed_device_bindings.end(),
            [](const auto& left, const auto& right) { return left.first < right.first; });
  for (const auto& [port, device] : unconsumed_device_bindings) {
    const auto* record = find_port(snapshot, GraphPortId{port});
    reject_physical_plan(
        result,
        kTensorDeviceBindingUnused,
        source_for(snapshot, record == nullptr ? VertexId{} : record->owner, PortId{port}),
        port_pointer(PortId{port}, "/tensor_device_binding"),
        "binding consumed by one admitted Tensor resource",
        "device=" + std::to_string(device.value));
  }

  std::unordered_map<std::string, std::uint32_t> service_generation_capacity;
  for (const auto& generation : result.imported_generation_capacities) {
    auto [found, inserted] = service_generation_capacity.emplace(
        generation.endpoint_or_service_id, generation.endpoint_max_generations);
    if (!inserted) {
      found->second = std::max(found->second, generation.endpoint_max_generations);
    }
  }
  std::uint32_t runtime_generation_capacity = 0U;
  bool generation_capacity_valid = true;
  for (const auto& [service, capacity] : service_generation_capacity) {
    static_cast<void>(service);
    if (!checked_add(runtime_generation_capacity, capacity, runtime_generation_capacity)) {
      generation_capacity_valid = false;
      break;
    }
  }
  if (!generation_capacity_valid ||
      (!result.imported_generation_capacities.empty() && runtime_generation_capacity == 0U)) {
    reject_physical_plan(result, kTensorOverflow);
  } else {
    for (auto& generation : result.imported_generation_capacities) {
      generation.service_max_generations =
          service_generation_capacity.at(generation.endpoint_or_service_id);
      generation.runtime_max_generations = runtime_generation_capacity;
    }
  }

  validate_tensor_provider_geometry(result, snapshot, plan_generation, resolved_queue_sizings);
  return result;
}

}  // namespace holoscan::detail
