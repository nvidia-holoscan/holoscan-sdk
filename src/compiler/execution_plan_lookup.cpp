/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "execution_plan_lookup.hpp"

#include <algorithm>
#include <iterator>

namespace holoscan::detail {
namespace {

template <typename Record>
[[nodiscard]] const Record* find_unique_trigger_by_owner(std::span<const Record> records,
                                                         OperatorInstanceId owner) noexcept {
  const Record* result = nullptr;
  for (const auto& candidate : records) {
    if (candidate.owner.id != owner) {
      continue;
    }
    if (result != nullptr) {
      return nullptr;
    }
    result = &candidate;
  }
  return result;
}

template <typename Record>
[[nodiscard]] std::span<const Record> records_by_owner(std::span<const Record> records,
                                                       OperatorRef owner) noexcept {
  const auto first = std::find_if(records.begin(), records.end(), [owner](const auto& record) {
    return record.owner == owner;
  });
  if (first == records.end()) {
    return {};
  }
  const auto last = std::find_if(
      first, records.end(), [owner](const auto& record) { return record.owner != owner; });
  return {&*first, static_cast<std::size_t>(std::distance(first, last))};
}

}  // namespace

const ExecutionPartitionRecord* find_execution_partition_by_member(const ExecutionPlanState& plan,
                                                                   VertexId member) noexcept {
  const auto found = std::find_if(
      plan.execution_partitions.begin(),
      plan.execution_partitions.end(),
      [member](const auto& partition) {
        return std::binary_search(partition.members.begin(),
                                  partition.members.end(),
                                  member,
                                  [](VertexId lhs, VertexId rhs) { return lhs.value < rhs.value; });
      });
  return found == plan.execution_partitions.end() ? nullptr : &*found;
}

const ExecutionPartitionRecord* find_execution_partition_by_id(const ExecutionPlanState& plan,
                                                               ExecutionPartitionId id) noexcept {
  if (!id || id.value > plan.execution_partitions.size()) {
    return nullptr;
  }
  const auto& partition = plan.execution_partitions[static_cast<std::size_t>(id.value - 1U)];
  return partition.id == id ? &partition : nullptr;
}

const VertexProcessPlacementPlanRecord* find_vertex_process_placement_by_vertex(
    const ExecutionPlanState& plan, VertexId vertex) noexcept {
  const auto found =
      std::find_if(plan.vertex_process_placements.begin(),
                   plan.vertex_process_placements.end(),
                   [vertex](const auto& placement) { return placement.vertex == vertex; });
  return found == plan.vertex_process_placements.end() ? nullptr : &*found;
}

const VertexRecord* find_vertex_by_id(const ExecutionPlanState& plan, VertexId id) noexcept {
  if (!id) {
    return nullptr;
  }
  const auto vertices = plan.graph.vertices();
  const auto found = std::find_if(
      vertices.begin(), vertices.end(), [id](const auto& vertex) { return vertex.id == id; });
  return found == vertices.end() ? nullptr : &*found;
}

const PortPlanRecord* find_port_by_id(const ExecutionPlanState& plan, PortId id) noexcept {
  const auto found = std::find_if(
      plan.ports.begin(), plan.ports.end(), [id](const auto& port) { return port.id == id; });
  return found == plan.ports.end() ? nullptr : &*found;
}

const PortPlanRecord* find_port_by_id_and_direction(const ExecutionPlanState& plan, PortId id,
                                                    PortDirection direction) noexcept {
  const auto* port = find_port_by_id(plan, id);
  return port != nullptr && port->direction == direction ? port : nullptr;
}

const OperatorParameterPlanRecord* find_operator_parameters_by_owner(const ExecutionPlanState& plan,
                                                                     OperatorRef owner) noexcept {
  const auto found =
      std::find_if(plan.operator_parameters.begin(),
                   plan.operator_parameters.end(),
                   [owner](const auto& parameters) { return parameters.owner == owner; });
  return found == plan.operator_parameters.end() ? nullptr : &*found;
}

std::span<const ParameterFieldPlanRecord> parameter_fields_by_owner(const ExecutionPlanState& plan,
                                                                    OperatorRef owner) noexcept {
  return records_by_owner<ParameterFieldPlanRecord>(plan.parameter_fields, owner);
}

std::span<const ParameterDerivedTermPlanRecord> parameter_derived_terms_by_owner(
    const ExecutionPlanState& plan, OperatorRef owner) noexcept {
  return records_by_owner<ParameterDerivedTermPlanRecord>(plan.parameter_derived_terms, owner);
}

const InputSelectionPlanRecord* find_input_selection_by_input(const ExecutionPlanState& plan,
                                                              PortId input) noexcept {
  const auto found = std::find_if(
      plan.input_selections.begin(), plan.input_selections.end(), [input](const auto& selection) {
        return selection.input == input;
      });
  return found == plan.input_selections.end() ? nullptr : &*found;
}

const ClockConversionRecord* find_clock_conversion_by_id(const ExecutionPlanState& plan,
                                                         std::uint64_t id) noexcept {
  if (id == 0U) {
    return nullptr;
  }
  const auto found = std::find_if(plan.clock_conversions.begin(),
                                  plan.clock_conversions.end(),
                                  [id](const auto& conversion) { return conversion.id == id; });
  return found == plan.clock_conversions.end() ? nullptr : &*found;
}

const ClockPlanRecord* find_clock_by_id(const ExecutionPlanState& plan, ClockId id) noexcept {
  if (!id.valid() || id.value > plan.clocks.size()) {
    return nullptr;
  }
  const auto& clock = plan.clocks[static_cast<std::size_t>(id.value - 1U)];
  return clock.id == id ? &clock : nullptr;
}

const TensorInterfacePoolPlanRecord* find_tensor_interface_pool_by_topic(
    const ExecutionPlanState& plan, ExternalTopicId topic) noexcept {
  const auto found =
      std::find_if(plan.tensor_interface_pools.begin(),
                   plan.tensor_interface_pools.end(),
                   [topic](const auto& pool) {
                     return pool.topic == topic && !pool.field_id && !pool.tensor_index;
                   });
  return found == plan.tensor_interface_pools.end() ? nullptr : &*found;
}

const TensorInterfacePoolPlanRecord* find_tensor_interface_pool_by_topic_field(
    const ExecutionPlanState& plan, ExternalTopicId topic, std::uint16_t field_id,
    std::uint16_t tensor_index) noexcept {
  const auto found = std::find_if(plan.tensor_interface_pools.begin(),
                                  plan.tensor_interface_pools.end(),
                                  [topic, field_id, tensor_index](const auto& pool) {
                                    return pool.topic == topic && pool.field_id == field_id &&
                                           pool.tensor_index == tensor_index;
                                  });
  return found == plan.tensor_interface_pools.end() ? nullptr : &*found;
}

const TensorAllocationPlanRecord* find_tensor_allocation_by_output(const ExecutionPlanState& plan,
                                                                   PortId output) noexcept {
  const auto found =
      std::find_if(plan.tensor_allocations.begin(),
                   plan.tensor_allocations.end(),
                   [output](const auto& allocation) { return allocation.output == output; });
  return found == plan.tensor_allocations.end() ? nullptr : &*found;
}

std::span<const TensorFieldAllocationPlanRecord> tensor_field_allocations_by_output(
    const ExecutionPlanState& plan, PortId output) noexcept {
  const auto first = std::lower_bound(
      plan.tensor_field_allocations.begin(),
      plan.tensor_field_allocations.end(),
      output,
      [](const auto& allocation, PortId key) { return allocation.output.value() < key.value(); });
  const auto last = std::upper_bound(
      first, plan.tensor_field_allocations.end(), output, [](PortId key, const auto& allocation) {
        return key.value() < allocation.output.value();
      });
  return {first, static_cast<std::size_t>(std::distance(first, last))};
}

const TensorDeviceResourcePlanRecord* find_tensor_device_resource(const ExecutionPlanState& plan,
                                                                  ProcessGroupId process_group,
                                                                  DeviceId device) noexcept {
  const auto found =
      std::find_if(plan.tensor_device_resources.begin(),
                   plan.tensor_device_resources.end(),
                   [process_group, device](const auto& resource) {
                     return resource.process_group == process_group && resource.device == device;
                   });
  return found == plan.tensor_device_resources.end() ? nullptr : &*found;
}

const CudaComputeStreamPlanRecord* find_cuda_compute_stream_by_partition(
    const ExecutionPlanState& plan, ProcessGroupId process_group, DeviceId device,
    ExecutionPartitionId partition) noexcept {
  const auto found = std::find_if(plan.cuda_compute_streams.begin(),
                                  plan.cuda_compute_streams.end(),
                                  [process_group, device, partition](const auto& stream) {
                                    return stream.process_group == process_group &&
                                           stream.device == device && stream.partition == partition;
                                  });
  return found == plan.cuda_compute_streams.end() ? nullptr : &*found;
}

const CudaComputeStreamPlanRecord* find_cuda_compute_stream_by_fused_edge(
    const ExecutionPlanState& plan, ProcessGroupId process_group, DeviceId device,
    EdgeId edge) noexcept {
  const auto found = std::find_if(
      plan.cuda_compute_streams.begin(),
      plan.cuda_compute_streams.end(),
      [process_group, device, edge](const auto& stream) {
        return stream.process_group == process_group && stream.device == device &&
               std::binary_search(stream.fused_linear_edges.begin(),
                                  stream.fused_linear_edges.end(),
                                  edge,
                                  [](EdgeId lhs, EdgeId rhs) { return lhs.value < rhs.value; });
      });
  return found == plan.cuda_compute_streams.end() ? nullptr : &*found;
}

bool has_cuda_compute_stream_for_device(const ExecutionPlanState& plan,
                                        ProcessGroupId process_group, DeviceId device) noexcept {
  return std::any_of(plan.cuda_compute_streams.begin(),
                     plan.cuda_compute_streams.end(),
                     [process_group, device](const auto& stream) {
                       return stream.process_group == process_group && stream.device == device;
                     });
}

const ProcessLocalChannelServicePlanRecord* find_process_local_channel_service_by_id(
    const ExecutionPlanState& plan, std::string_view service_id) noexcept {
  const auto found =
      std::find_if(plan.process_local_channel_services.begin(),
                   plan.process_local_channel_services.end(),
                   [service_id](const auto& service) { return service.service_id == service_id; });
  return found == plan.process_local_channel_services.end() ? nullptr : &*found;
}

const ProcessLocalChannelPublisherPlanRecord* find_process_local_channel_publisher_by_service_id(
    const ExecutionPlanState& plan, std::string_view service_id) noexcept {
  const auto found = std::find_if(
      plan.process_local_channel_publishers.begin(),
      plan.process_local_channel_publishers.end(),
      [service_id](const auto& publisher) { return publisher.service_id == service_id; });
  return found == plan.process_local_channel_publishers.end() ? nullptr : &*found;
}

const CrossProcessChannelServicePlanRecord* find_cross_process_channel_service_by_id(
    const ExecutionPlanState& plan, std::string_view service_id) noexcept {
  const auto found =
      std::find_if(plan.cross_process_channel_services.begin(),
                   plan.cross_process_channel_services.end(),
                   [service_id](const auto& service) { return service.service_id == service_id; });
  return found == plan.cross_process_channel_services.end() ? nullptr : &*found;
}

const ChannelServiceStaticPlanRecord* find_external_channel_service_by_topic(
    const ExecutionPlanState& plan, ExternalTopicId topic) noexcept {
  const auto found = std::find_if(plan.channel_services.begin(),
                                  plan.channel_services.end(),
                                  [topic](const auto& service) { return service.topic == topic; });
  return found == plan.channel_services.end() ? nullptr : &*found;
}

const ChannelPublisherEndpointPlanRecord* find_external_channel_publisher_by_topic(
    const ExecutionPlanState& plan, ExternalTopicId topic) noexcept {
  const auto found =
      std::find_if(plan.channel_publishers.begin(),
                   plan.channel_publishers.end(),
                   [topic](const auto& publisher) { return publisher.topic == topic; });
  return found == plan.channel_publishers.end() ? nullptr : &*found;
}

const ChannelSubscriberEndpointPlanRecord* find_external_channel_subscriber_by_topic(
    const ExecutionPlanState& plan, ExternalTopicId topic) noexcept {
  const auto found =
      std::find_if(plan.channel_subscribers.begin(),
                   plan.channel_subscribers.end(),
                   [topic](const auto& subscriber) { return subscriber.topic == topic; });
  return found == plan.channel_subscribers.end() ? nullptr : &*found;
}

const ReadinessPlanRecord* find_readiness_by_owner(const ExecutionPlanState& plan,
                                                   VertexId owner) noexcept {
  const auto found = std::find_if(plan.readiness.begin(),
                                  plan.readiness.end(),
                                  [&](const auto& row) { return row.owner == owner; });
  return found == plan.readiness.end() ? nullptr : &*found;
}

const TemporalTriggerPlanRecord* find_temporal_trigger_by_owner(const ExecutionPlanState& plan,
                                                                OperatorInstanceId owner) noexcept {
  const auto found = std::find_if(plan.temporal_triggers.begin(),
                                  plan.temporal_triggers.end(),
                                  [&](const auto& row) { return row.owner.id == owner; });
  return found == plan.temporal_triggers.end() ? nullptr : &*found;
}

const TemporalTriggerPlanRecord* find_unique_temporal_trigger_by_owner(
    const ExecutionPlanState& plan, OperatorInstanceId owner) noexcept {
  return find_unique_trigger_by_owner<TemporalTriggerPlanRecord>(plan.temporal_triggers, owner);
}

const ClockTriggerPlanRecord* find_clock_trigger_by_owner(const ExecutionPlanState& plan,
                                                          OperatorRef owner) noexcept {
  const auto found = std::find_if(plan.clock_triggers.begin(),
                                  plan.clock_triggers.end(),
                                  [&](const auto& row) { return row.owner == owner; });
  return found == plan.clock_triggers.end() ? nullptr : &*found;
}

const PredicateTriggerPlanRecord* find_predicate_trigger_by_owner(const ExecutionPlanState& plan,
                                                                  OperatorRef owner) noexcept {
  const auto found = std::find_if(plan.predicate_triggers.begin(),
                                  plan.predicate_triggers.end(),
                                  [&](const auto& row) { return row.owner == owner; });
  return found == plan.predicate_triggers.end() ? nullptr : &*found;
}

const PredicateTriggerPlanRecord* find_unique_predicate_trigger_by_owner(
    const ExecutionPlanState& plan, OperatorInstanceId owner) noexcept {
  return find_unique_trigger_by_owner<PredicateTriggerPlanRecord>(plan.predicate_triggers, owner);
}

const PredicateCallablePlanRecord* find_predicate_callable_by_id(
    const ExecutionPlanState& plan, std::string_view predicate_id) noexcept {
  const auto found = std::find_if(
      plan.predicate_callables.begin(), plan.predicate_callables.end(), [&](const auto& row) {
        return row.predicate_id == predicate_id;
      });
  return found == plan.predicate_callables.end() ? nullptr : &*found;
}

}  // namespace holoscan::detail
