/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/inline_partition_execution_plan.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "compiler/execution_plan_identity.hpp"
#include "compiler/execution_plan_lookup.hpp"
#include "compiler/process_plan_records.hpp"
#include "core/plan_record_contracts.hpp"
#include "core/tensor_allocation_record_validation.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] const PortPlanRecord* dense_port(const ExecutionPlanState& state,
                                               PortId id) noexcept {
  if (!id || id.value() > state.ports.size()) {
    return nullptr;
  }
  const auto& port = state.ports[static_cast<std::size_t>(id.value() - 1U)];
  return port.id == id ? &port : nullptr;
}

[[nodiscard]] const ExecutableOperatorFactoryRecord* operator_factory(
    const ExecutionPlanState& state, VertexId vertex) noexcept {
  const ExecutableOperatorFactoryRecord* result = nullptr;
  for (const auto& candidate : state.executable_factories.operators) {
    if (candidate.vertex != vertex) {
      continue;
    }
    if (result != nullptr) {
      return nullptr;
    }
    result = &candidate;
  }
  return result;
}

[[nodiscard]] bool default_sleeping_placement(const ThreadPlacement& placement) noexcept {
  return !placement.affinity.has_value() &&
         placement.scheduling.policy == SchedulingPolicy::kOther &&
         !placement.scheduling.priority.has_value() && !placement.scheduling.deadline.has_value() &&
         placement.wake_mode == WakeMode::kSleepingWaitset &&
         placement.irq_isolation == IRQIsolation::kNotRequested &&
         placement.probe_failure_policy == ProbeFailurePolicy::kWarnAndRecordFallback;
}

[[nodiscard]] bool exact_single_default_lane(const ExecutionPlanState& state) noexcept {
  if (state.scheduler_plan.execution_lane_count != 1U ||
      state.scheduler_plan.max_concurrent_activations != 1U ||
      state.scheduler_plan.activation_overlap || state.scheduler_plan.worker_count != 1U ||
      state.scheduler_plan.intra_path_parallelism != 1U || state.execution_lanes.size() != 1U ||
      !state.placement_probes.empty() || !state.partition_placements.empty()) {
    return false;
  }
  const auto& lane = state.execution_lanes.front();
  return lane.execution_lane == 0U && static_cast<bool>(lane.partition) &&
         default_sleeping_placement(lane.accepted_placement);
}

[[nodiscard]] bool operator_parameters_are_supported(const ExecutionPlanState& state,
                                                     OperatorInstanceId owner) noexcept {
  const OperatorParameterPlanRecord* parameters = nullptr;
  for (const auto& candidate : state.operator_parameters) {
    if (candidate.owner.id != owner) {
      continue;
    }
    if (parameters != nullptr) {
      return false;
    }
    parameters = &candidate;
  }
  return parameters != nullptr;
}

[[nodiscard]] bool owned_input(const ExecutionPlanState& state, VertexId vertex,
                               PortId input) noexcept {
  const auto* port = dense_port(state, input);
  return port != nullptr && port->owner == vertex && port->owner_kind == PortOwnerKind::kOperator &&
         port->direction == PortDirection::kInput;
}

[[nodiscard]] bool operator_missing_aggregates_are_supported(const ExecutionPlanState& state,
                                                             VertexId vertex,
                                                             OperatorInstanceId owner) noexcept {
  bool found = false;
  std::array<bool, 3U> phases{};
  for (const auto& candidate : state.missing_aggregates) {
    if (candidate.owner.id != owner) {
      continue;
    }
    const auto phase = static_cast<std::size_t>(candidate.phase);
    if (phase >= phases.size() || phases[phase]) {
      return false;
    }
    phases[phase] = true;
    found = true;
    for (std::size_t index = 0U; index < candidate.inputs.size(); ++index) {
      const auto& input = candidate.inputs[index];
      if (input.ordinal != index || !owned_input(state, vertex, input.input) ||
          input.selector == InputSelectorKind::kBracketedWindow) {
        return false;
      }
    }
  }
  return found;
}

[[nodiscard]] bool raw_tensor_payload(const PayloadIdentityRecord& identity) noexcept {
  return identity.kind == PayloadIdentityKind::kTensor &&
         identity.payload_kind == PayloadKind::kDescriptorBacked && !identity.scalar &&
         !identity.schema && !identity.pod;
}

[[nodiscard]] bool descriptor_tensor_payload(const PayloadIdentityRecord& identity) noexcept {
  return identity.kind == PayloadIdentityKind::kSchema &&
         identity.payload_kind == PayloadKind::kDescriptorBacked && !identity.scalar &&
         identity.schema.has_value() && !identity.pod;
}

[[nodiscard]] bool descriptor_tensor_payload_equal(const PayloadIdentityRecord& left,
                                                   const PayloadIdentityRecord& right) noexcept {
  return descriptor_tensor_payload(left) && descriptor_tensor_payload(right) &&
         payload_identity_exactly_equal(left, right);
}

[[nodiscard]] bool tensor_backed_payload(const PayloadIdentityRecord& identity) noexcept {
  return raw_tensor_payload(identity) || descriptor_tensor_payload(identity);
}

[[nodiscard]] bool supported_payload(const PayloadEdgePlanRecord& edge,
                                     const PortPlanRecord& source,
                                     const PortPlanRecord& target) noexcept {
  const bool pod = edge.payload_identity.kind == PayloadIdentityKind::kPod &&
                   edge.payload_identity.payload_kind == PayloadKind::kInlineTyped &&
                   edge.payload_identity.pod.has_value();
  const bool schema = edge.payload_identity.kind == PayloadIdentityKind::kSchema &&
                      edge.payload_identity.payload_kind == PayloadKind::kSerializedBytes &&
                      edge.payload_identity.schema.has_value() &&
                      source.serialized_payload_capacity_bytes != 0U &&
                      target.serialized_payload_capacity_bytes != 0U;
  const bool tensor = raw_tensor_payload(edge.payload_identity) &&
                      raw_tensor_payload(source.payload_identity) &&
                      raw_tensor_payload(target.payload_identity) &&
                      source.serialized_payload_capacity_bytes == 0U &&
                      target.serialized_payload_capacity_bytes == 0U;
  const bool descriptor_tensor =
      descriptor_tensor_payload_equal(source.payload_identity, edge.payload_identity) &&
      descriptor_tensor_payload_equal(target.payload_identity, edge.payload_identity) &&
      source.serialized_payload_capacity_bytes != 0U &&
      source.serialized_payload_capacity_bytes <= target.serialized_payload_capacity_bytes;
  return pod || schema || tensor || descriptor_tensor;
}

[[nodiscard]] bool simple_channel_edge(const ExecutionPlanState& state,
                                       const PayloadEdgePlanRecord& edge) noexcept {
  if (edge.disposition != PlanDisposition::kAccepted ||
      edge.locality != ChannelLocalityKind::kProcessLocal || edge.queue_depth == 0U ||
      edge.producer_max_emits_per_compute == 0U ||
      (edge.queue_policy != QueuePolicy::kBlocking &&
       edge.queue_policy != QueuePolicy::kDropOldest)) {
    return false;
  }
  const auto* source = dense_port(state, edge.source);
  const auto* target = dense_port(state, edge.target);
  return source != nullptr && target != nullptr && source->owner_kind == PortOwnerKind::kOperator &&
         target->owner_kind == PortOwnerKind::kOperator &&
         source->direction == PortDirection::kOutput &&
         target->direction == PortDirection::kInput && source->owner != target->owner &&
         supported_payload(edge, *source, *target);
}

[[nodiscard]] std::optional<std::size_t> member_index(const std::vector<VertexId>& members,
                                                      VertexId vertex) noexcept {
  const auto found = std::find(members.begin(), members.end(), vertex);
  if (found == members.end()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(std::distance(members.begin(), found));
}

struct TopologyFacts {
  std::vector<VertexId> order;
  std::vector<EdgeId> edges;
  VertexId root{};
};

[[nodiscard]] std::optional<TopologyFacts> stable_topology(const ExecutionPlanState& state,
                                                           const ExecutionPartitionRecord& part) {
  const std::size_t member_count = part.members.size();
  std::vector<std::uint32_t> indegree(member_count, 0U);
  std::vector<std::vector<std::size_t>> successors(member_count);
  std::vector<EdgeId> edges;
  edges.reserve(state.payload_edges.size());

  for (const auto& edge : state.payload_edges) {
    if (!simple_channel_edge(state, edge)) {
      return std::nullopt;
    }
    const auto* source = dense_port(state, edge.source);
    const auto* target = dense_port(state, edge.target);
    if (source == nullptr || target == nullptr) {
      return std::nullopt;
    }
    const auto source_index = member_index(part.members, source->owner);
    const auto target_index = member_index(part.members, target->owner);
    if (!source_index || !target_index || *source_index == *target_index) {
      return std::nullopt;
    }
    // The legacy closed-island record can represent only one exact predecessor
    // and successor per member. General one-root DAGs are owned by the
    // partition-runner plan instead.
    if (indegree[*target_index] != 0U || !successors[*source_index].empty()) {
      return std::nullopt;
    }
    ++indegree[*target_index];
    successors[*source_index].push_back(*target_index);
    edges.push_back(edge.edge);
  }

  std::vector<std::size_t> ready;
  for (std::size_t index = 0U; index < member_count; ++index) {
    if (indegree[index] == 0U) {
      ready.push_back(index);
    }
  }
  if (ready.size() != 1U) {
    return std::nullopt;
  }

  TopologyFacts result;
  result.root = part.members[ready.front()];
  result.order.reserve(member_count);
  while (!ready.empty()) {
    const auto selected =
        std::min_element(ready.begin(), ready.end(), [&](std::size_t lhs, std::size_t rhs) {
          return part.members[lhs].value < part.members[rhs].value;
        });
    const std::size_t current = *selected;
    ready.erase(selected);
    result.order.push_back(part.members[current]);
    for (const std::size_t successor : successors[current]) {
      if (--indegree[successor] == 0U) {
        ready.push_back(successor);
      }
    }
  }
  if (result.order.size() != member_count) {
    return std::nullopt;
  }
  std::sort(
      edges.begin(), edges.end(), [](EdgeId lhs, EdgeId rhs) { return lhs.value < rhs.value; });
  // Runtime uses binary_search for inline Tensor-resource admission; retain
  // this debug assertion next to the sole construction of that edge set.
  assert(std::is_sorted(
      edges.begin(), edges.end(), [](EdgeId lhs, EdgeId rhs) { return lhs.value < rhs.value; }));
  result.edges = std::move(edges);
  return result;
}

struct RootReadinessFacts {
  ClosedIslandReadinessRootKind kind{ClosedIslandReadinessRootKind::kPhysical};
  SourceId source{};
  ClockId clock{};
};

[[nodiscard]] std::optional<RootReadinessFacts> root_readiness(
    const ExecutionPlanState& state, VertexId vertex,
    const TemporalTriggerPlanRecord& trigger) noexcept {
  const auto input_count =
      std::count_if(state.ports.begin(), state.ports.end(), [&](const auto& candidate) {
        return candidate.owner == vertex && candidate.owner_kind == PortOwnerKind::kOperator &&
               candidate.direction == PortDirection::kInput;
      });
  if (trigger.trigger_input || !trigger.declared_inputs.empty() ||
      !trigger.presence_guaranteed_inputs.empty() || !trigger.readiness_source ||
      std::cmp_not_equal(input_count, 0U)) {
    return std::nullopt;
  }
  switch (trigger.kind) {
    case TemporalRecordTriggerKind::kClock: {
      const ClockTriggerPlanRecord* clock = nullptr;
      for (const auto& candidate : state.clock_triggers) {
        if (candidate.owner.id != trigger.owner.id) {
          continue;
        }
        if (clock != nullptr) {
          return std::nullopt;
        }
        clock = &candidate;
      }
      if (clock == nullptr || !trigger.trigger_clock.valid() ||
          clock->clock != trigger.trigger_clock || clock->period_ns != trigger.period_ns) {
        return std::nullopt;
      }
      return RootReadinessFacts{
          .kind = ClosedIslandReadinessRootKind::kClock,
          .source = trigger.readiness_source,
          .clock = trigger.trigger_clock,
      };
    }
    default:
      return std::nullopt;
  }
}

[[nodiscard]] bool owned_inputs(const ExecutionPlanState& state, VertexId vertex,
                                const std::vector<PortId>& inputs) noexcept {
  return std::all_of(inputs.begin(), inputs.end(), [&](PortId input) {
    return owned_input(state, vertex, input);
  });
}

[[nodiscard]] bool input_driven_trigger(const ExecutionPlanState& state, VertexId vertex,
                                        const TemporalTriggerPlanRecord& trigger) noexcept {
  if (!trigger.readiness_source || trigger.trigger_clock.valid() ||
      !owned_inputs(state, vertex, trigger.declared_inputs) ||
      !owned_inputs(state, vertex, trigger.presence_guaranteed_inputs)) {
    return false;
  }

  switch (trigger.kind) {
    case TemporalRecordTriggerKind::kEach:
      return trigger.trigger_input && owned_input(state, vertex, trigger.trigger_input);
    case TemporalRecordTriggerKind::kAnyInput:
      return !trigger.declared_inputs.empty() &&
             (!trigger.trigger_input || owned_input(state, vertex, trigger.trigger_input));
    case TemporalRecordTriggerKind::kPredicate: {
      const auto* predicate = find_unique_predicate_trigger_by_owner(state, trigger.owner.id);
      return predicate != nullptr &&
             predicate->source == PredicateEvaluationSourceKind::kInputUpdate &&
             predicate->driver == trigger.trigger_input && trigger.trigger_input &&
             owned_input(state, vertex, trigger.trigger_input);
    }
    default:
      return false;
  }
}

[[nodiscard]] bool supported_runtime_contracts(const ExecutionPlanState& state) noexcept {
  return state.aligned_triggers.empty() && state.window_triggers.empty() &&
         state.bracketed_windows.empty() && state.channel_services.empty() &&
         state.channel_publishers.empty() && state.channel_subscribers.empty() &&
         state.cross_process_channel_services.empty() &&
         state.cross_process_channel_publishers.empty() &&
         state.cross_process_channel_subscribers.empty() && state.tensor_interface_pools.empty() &&
         state.tensor_inline_records.empty() && state.tensor_transfer_plans.empty();
}

[[nodiscard]] const PayloadEdgePlanRecord* edge_record(const ExecutionPlanState& state,
                                                       EdgeId id) noexcept {
  const PayloadEdgePlanRecord* result = nullptr;
  for (const auto& candidate : state.payload_edges) {
    if (candidate.edge != id) {
      continue;
    }
    if (result != nullptr) {
      return nullptr;
    }
    result = &candidate;
  }
  return result;
}

[[nodiscard]] const TensorAllocationPlanRecord* tensor_allocation(const ExecutionPlanState& state,
                                                                  PortId output) noexcept {
  const TensorAllocationPlanRecord* result = nullptr;
  for (const auto& candidate : state.tensor_allocations) {
    if (candidate.output != output) {
      continue;
    }
    if (result != nullptr) {
      return nullptr;
    }
    result = &candidate;
  }
  return result;
}

[[nodiscard]] bool tensor_field_storage_equal(
    const TensorFieldServiceStorageContract& field,
    const TensorFieldAllocationPlanRecord& allocation) noexcept {
  if (!allocation.layout) {
    return false;
  }
  const auto& layout = *allocation.layout;
  return field.field_id == allocation.field_id && field.tensor_index == allocation.tensor_index &&
         field.max_byte_span == allocation.bounds.max_byte_span &&
         field.memory_kind == allocation.memory_kind && field.device == allocation.device &&
         field.rank == layout.rank && field.dtype.has_value() == layout.dtype.has_value() &&
         (!field.dtype || dl_data_type_exactly_equal(*field.dtype, *layout.dtype));
}

[[nodiscard]] bool supported_tensor_field_allocations(
    const ExecutionPlanState& state, const PortPlanRecord& source,
    const PayloadEdgePlanRecord& edge, ProcessGroupId process_group,
    const ProcessLocalChannelServicePlanRecord& service,
    std::span<const TensorFieldAllocationPlanRecord> allocations) noexcept {
  if (allocations.empty() || service.tensor_fields.size() != allocations.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < allocations.size(); ++index) {
    const auto& allocation = allocations[index];
    if (!validate_tensor_field_allocation_record(allocation, state.plan_generation) ||
        allocation.output != edge.source || allocation.allocator_process_group != process_group ||
        allocation.allocator_root_id !=
            tensor_allocator_root_id(state.plan_generation, process_group, allocation.device) ||
        (allocation.memory_kind != MemoryKind::kHost &&
         allocation.memory_kind != MemoryKind::kCudaDevice) ||
        source.max_emits_per_compute != allocation.max_emits_per_compute ||
        edge.producer_max_emits_per_compute != allocation.max_emits_per_compute ||
        allocation.tensor_index != index ||
        (index != 0U && allocations[index - 1U].field_id >= allocation.field_id) ||
        !tensor_field_storage_equal(service.tensor_fields[index], allocation)) {
      return false;
    }
  }
  return true;
}

// Inline eligibility is a single conservative conjunction over the complete Tensor edge contract.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool supported_tensor_edge(const ExecutionPlanState& state,
                                         const PayloadEdgePlanRecord& edge,
                                         ProcessGroupId process_group) {
  if (!tensor_backed_payload(edge.payload_identity) ||
      edge.disposition != PlanDisposition::kAccepted ||
      edge.locality != ChannelLocalityKind::kProcessLocal || edge.fanout != 1U) {
    return false;
  }
  const auto* source = dense_port(state, edge.source);
  const auto* target = dense_port(state, edge.target);
  const auto* allocation = tensor_allocation(state, edge.source);
  const auto field_allocations = tensor_field_allocations_by_output(state, edge.source);
  const bool raw_tensor = raw_tensor_payload(edge.payload_identity);
  const bool payload_contract_matches =
      raw_tensor
          ? source != nullptr && target != nullptr &&
                raw_tensor_payload(source->payload_identity) &&
                raw_tensor_payload(target->payload_identity)
          : source != nullptr && target != nullptr &&
                descriptor_tensor_payload_equal(source->payload_identity, edge.payload_identity) &&
                descriptor_tensor_payload_equal(target->payload_identity, edge.payload_identity);
  if (source == nullptr || target == nullptr || source->owner_kind != PortOwnerKind::kOperator ||
      target->owner_kind != PortOwnerKind::kOperator ||
      source->direction != PortDirection::kOutput || target->direction != PortDirection::kInput ||
      !payload_contract_matches || (allocation == nullptr && field_allocations.empty()) ||
      (allocation != nullptr && !field_allocations.empty()) ||
      (raw_tensor && allocation == nullptr)) {
    return false;
  }

  const ProcessLocalChannelServicePlanRecord* service = nullptr;
  for (const auto& candidate : state.process_local_channel_services) {
    if (candidate.source != edge.source || candidate.queue_policy != edge.queue_policy) {
      continue;
    }
    if (service != nullptr) {
      return false;
    }
    service = &candidate;
  }
  if (service == nullptr) {
    return false;
  }
  const bool direct_allocation_matches =
      allocation != nullptr &&
      validate_tensor_allocation_record(*allocation, state.plan_generation) &&
      allocation->allocator_process_group == process_group &&
      allocation->pool_id == tensor_output_pool_id(state.plan_generation, edge.source) &&
      allocation->allocator_root_id ==
          tensor_allocator_root_id(state.plan_generation, process_group, allocation->device) &&
      (allocation->memory_kind == MemoryKind::kHost ||
       allocation->memory_kind == MemoryKind::kCudaDevice) &&
      tensor_port_layout_exactly_equal(allocation->layout, source->tensor_layout) &&
      tensor_allocation_satisfies_input(*allocation, target->tensor_layout) &&
      source->max_emits_per_compute == allocation->max_emits_per_compute &&
      edge.producer_max_emits_per_compute == allocation->max_emits_per_compute &&
      (allocation->memory_kind != MemoryKind::kHost || !allocation->device.valid()) &&
      (allocation->memory_kind != MemoryKind::kCudaDevice || allocation->device.valid());
  const bool field_allocations_match =
      allocation == nullptr && descriptor_tensor_payload(edge.payload_identity) &&
      supported_tensor_field_allocations(
          state, *source, edge, process_group, *service, field_allocations);
  if (!direct_allocation_matches && !field_allocations_match) {
    return false;
  }
  const std::string expected_service_id =
      process_local_channel_service_id(state.plan_generation, edge.source, edge.queue_policy);
  const bool storage_contract_matches =
      raw_tensor && allocation != nullptr
          ? service->payload_size_bytes == 0U &&
                service->tensor_max_byte_span == allocation->bounds.max_byte_span &&
                service->tensor_pool_id == allocation->pool_id &&
                service->tensor_memory_kind == allocation->memory_kind &&
                service->tensor_device == allocation->device
          : descriptor_tensor_payload(edge.payload_identity) &&
                service->payload_size_bytes == source->serialized_payload_capacity_bytes &&
                ((direct_allocation_matches && service->tensor_fields.empty() &&
                  service->tensor_max_byte_span == allocation->bounds.max_byte_span &&
                  service->tensor_pool_id == allocation->pool_id &&
                  service->tensor_memory_kind == allocation->memory_kind &&
                  service->tensor_device == allocation->device) ||
                 (field_allocations_match && !service->tensor_fields.empty() &&
                  service->tensor_max_byte_span == 0U && service->tensor_pool_id.empty() &&
                  service->tensor_memory_kind == MemoryKind::kUnknown &&
                  !service->tensor_device.valid()));
  if (service->service_id != expected_service_id ||
      (raw_tensor
           ? !raw_tensor_payload(service->payload_identity)
           : !descriptor_tensor_payload_equal(service->payload_identity, edge.payload_identity)) ||
      !service->tensor_payload || !storage_contract_matches ||
      service->delivery_kind != ProcessLocalChannelDeliveryKind::kDirectRetained ||
      service->cross_process_transport || service->max_subscribers != 1U ||
      service->max_subscriber_queue_depth != edge.queue_depth ||
      service->max_subscriber_borrowed_samples != edge.queue_depth ||
      service->publisher_history_depth != 0U || service->event_wake) {
    return false;
  }

  const ProcessLocalChannelPublisherPlanRecord* publisher = nullptr;
  for (const auto& candidate : state.process_local_channel_publishers) {
    if (candidate.source != edge.source || candidate.service_id != expected_service_id) {
      continue;
    }
    if (publisher != nullptr) {
      return false;
    }
    publisher = &candidate;
  }
  if (publisher == nullptr ||
      publisher->max_loaned_samples != edge.producer_max_emits_per_compute ||
      publisher->replacement_attribution_required !=
          (edge.queue_policy != QueuePolicy::kBlocking)) {
    return false;
  }

  const ProcessLocalChannelSubscriberPlanRecord* subscriber = nullptr;
  for (const auto& candidate : state.process_local_channel_subscribers) {
    if (candidate.edge != edge.edge) {
      continue;
    }
    if (subscriber != nullptr) {
      return false;
    }
    subscriber = &candidate;
  }
  return subscriber != nullptr && subscriber->source == edge.source &&
         subscriber->target == edge.target && subscriber->service_id == expected_service_id &&
         subscriber->requested_queue_depth == edge.queue_depth &&
         subscriber->max_borrowed_samples == edge.queue_depth &&
         subscriber->cache_capacity == edge.cache_capacity &&
         subscriber->readiness_capacity == std::max(edge.queue_depth, edge.cache_capacity) &&
         !subscriber->event_wake;
}

struct TensorResourceTotals {
  std::uint64_t pools{};
  std::uint64_t slots{};
  std::uint64_t loans{};
  std::uint64_t controls{};
  std::uint64_t readiness{};
  std::uint64_t releases{};
  std::uint64_t retirements{};
  std::uint64_t quarantines{};
};

[[nodiscard]] bool add_capacity(std::uint64_t& total, std::uint32_t value) noexcept {
  constexpr auto kMaximum = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
  if (total > kMaximum - value) {
    return false;
  }
  total += value;
  return true;
}

template <typename Allocation>
[[nodiscard]] bool append_allocation_totals(TensorResourceTotals& totals,
                                            const Allocation& allocation) noexcept {
  return add_capacity(totals.pools, 1U) && add_capacity(totals.slots, allocation.slot_count) &&
         add_capacity(totals.loans, allocation.max_outstanding_loans) &&
         add_capacity(totals.controls, allocation.control_capacity) &&
         add_capacity(totals.readiness, allocation.readiness_capacity) &&
         add_capacity(totals.releases, allocation.release_capacity) &&
         add_capacity(totals.retirements, allocation.retirement_capacity) &&
         add_capacity(totals.quarantines, allocation.quarantine_capacity);
}

// The fail-closed proof deliberately joins every Tensor record family in one cold-path predicate.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool supported_tensor_resources(const ExecutionPlanState& state,
                                              ProcessGroupId process_group,
                                              const std::vector<EdgeId>& topology_edges) {
  std::size_t tensor_edge_count{};
  for (const EdgeId edge_id : topology_edges) {
    const auto* edge = edge_record(state, edge_id);
    tensor_edge_count += edge != nullptr && tensor_backed_payload(edge->payload_identity) ? 1U : 0U;
  }
  const bool allocations_empty =
      state.tensor_allocations.empty() && state.tensor_field_allocations.empty();
  if (allocations_empty) {
    return tensor_edge_count == 0U && state.tensor_device_resources.empty();
  }
  if (tensor_edge_count == 0U) {
    return false;
  }
  const auto output_has_one_topology_edge = [&](PortId output) {
    return std::count_if(topology_edges.begin(), topology_edges.end(), [&](EdgeId edge_id) {
             const auto* edge = edge_record(state, edge_id);
             return edge != nullptr && edge->source == output &&
                    tensor_backed_payload(edge->payload_identity);
           }) == 1;
  };
  if (std::ranges::any_of(
          state.tensor_allocations,
          [&](const auto& allocation) {
            return !output_has_one_topology_edge(allocation.output) ||
                   !tensor_field_allocations_by_output(state, allocation.output).empty();
          }) ||
      std::ranges::any_of(state.tensor_field_allocations, [&](const auto& allocation) {
        return !output_has_one_topology_edge(allocation.output) ||
               tensor_allocation(state, allocation.output) != nullptr;
      })) {
    return false;
  }

  std::optional<DeviceId> partition_device;
  const auto admit_memory = [&](MemoryKind memory_kind, DeviceId device) {
    if (memory_kind == MemoryKind::kHost) {
      return !device.valid();
    }
    if (memory_kind != MemoryKind::kCudaDevice || !device.valid()) {
      return false;
    }
    if (partition_device && *partition_device != device) {
      return false;
    }
    partition_device = device;
    return true;
  };
  if (std::ranges::any_of(state.tensor_allocations,
                          [&](const auto& allocation) {
                            return !admit_memory(allocation.memory_kind, allocation.device);
                          }) ||
      std::ranges::any_of(state.tensor_field_allocations,
                          [&](const auto& allocation) {
                            return !admit_memory(allocation.memory_kind, allocation.device);
                          }) ||
      (partition_device ? state.tensor_device_resources.size() != 1U
                        : !state.tensor_device_resources.empty())) {
    return false;
  }

  std::size_t tensor_service_count{};
  for (const auto& service : state.process_local_channel_services) {
    tensor_service_count += service.tensor_payload ? 1U : 0U;
  }
  std::size_t tensor_publisher_count{};
  for (const auto& publisher : state.process_local_channel_publishers) {
    tensor_publisher_count +=
        std::count_if(state.process_local_channel_services.begin(),
                      state.process_local_channel_services.end(),
                      [&](const auto& service) {
                        return service.tensor_payload && service.service_id == publisher.service_id;
                      }) == 1
            ? 1U
            : 0U;
  }
  std::size_t tensor_subscriber_count{};
  for (const auto& subscriber : state.process_local_channel_subscribers) {
    const auto* edge = edge_record(state, subscriber.edge);
    tensor_subscriber_count +=
        edge != nullptr && tensor_backed_payload(edge->payload_identity) ? 1U : 0U;
  }
  if (tensor_service_count != tensor_edge_count || tensor_publisher_count != tensor_edge_count ||
      tensor_subscriber_count != tensor_edge_count) {
    return false;
  }

  for (const EdgeId edge_id : topology_edges) {
    const auto* edge = edge_record(state, edge_id);
    if (edge == nullptr || (tensor_backed_payload(edge->payload_identity) &&
                            !supported_tensor_edge(state, *edge, process_group))) {
      return false;
    }
  }

  for (const auto& allocation : state.tensor_allocations) {
    if (allocation.memory_kind == MemoryKind::kHost) {
      if (allocation.device.valid() || std::any_of(state.tensor_device_resources.begin(),
                                                   state.tensor_device_resources.end(),
                                                   [&](const auto& resource) {
                                                     return resource.process_group ==
                                                                process_group &&
                                                            resource.device == allocation.device;
                                                   })) {
        return false;
      }
      continue;
    }
    if (allocation.memory_kind != MemoryKind::kCudaDevice || !allocation.device.valid()) {
      return false;
    }
    const auto matching = std::count_if(state.tensor_device_resources.begin(),
                                        state.tensor_device_resources.end(),
                                        [&](const auto& resource) {
                                          return resource.process_group == process_group &&
                                                 resource.device == allocation.device;
                                        });
    if (matching != 1) {
      return false;
    }
  }
  for (const auto& allocation : state.tensor_field_allocations) {
    if (allocation.memory_kind == MemoryKind::kHost) {
      if (allocation.device.valid()) {
        return false;
      }
      continue;
    }
    if (allocation.memory_kind != MemoryKind::kCudaDevice || !allocation.device.valid()) {
      return false;
    }
    const auto matching = std::count_if(state.tensor_device_resources.begin(),
                                        state.tensor_device_resources.end(),
                                        [&](const auto& resource) {
                                          return resource.process_group == process_group &&
                                                 resource.device == allocation.device;
                                        });
    if (matching != 1) {
      return false;
    }
  }

  for (const auto& resource : state.tensor_device_resources) {
    if (resource.plan_generation != state.plan_generation || resource.run_generation != 0U ||
        resource.process_group != process_group || !resource.device.valid() ||
        resource.allocator_root_id !=
            tensor_allocator_root_id(state.plan_generation, process_group, resource.device) ||
        resource.materialize_before != LifecycleStage::kArm ||
        resource.gpu_resident_backend_selected || !resource.fallback_rejected) {
      return false;
    }

    std::vector<EdgeId> fused_linear_edges;
    for (const auto& stream : state.cuda_compute_streams) {
      if (stream.process_group != process_group || stream.device != resource.device) {
        continue;
      }
      if (stream.plan_generation != state.plan_generation || stream.run_generation != 0U ||
          stream.owner_id.empty() || stream.stream_id.empty() ||
          stream.materialize_before != LifecycleStage::kArm ||
          std::adjacent_find(stream.fused_linear_edges.begin(),
                             stream.fused_linear_edges.end(),
                             [](EdgeId left, EdgeId right) { return left.value >= right.value; }) !=
              stream.fused_linear_edges.end()) {
        return false;
      }
      fused_linear_edges.insert(fused_linear_edges.end(),
                                stream.fused_linear_edges.begin(),
                                stream.fused_linear_edges.end());
    }
    std::sort(fused_linear_edges.begin(), fused_linear_edges.end(), [](EdgeId left, EdgeId right) {
      return left.value < right.value;
    });
    if (std::adjacent_find(fused_linear_edges.begin(), fused_linear_edges.end()) !=
        fused_linear_edges.end()) {
      return false;
    }

    TensorResourceTotals totals;
    std::size_t allocation_count{};
    for (const auto& allocation : state.tensor_allocations) {
      if (allocation.allocator_process_group != process_group ||
          allocation.device != resource.device) {
        continue;
      }
      ++allocation_count;
      if (allocation.memory_kind != MemoryKind::kCudaDevice ||
          allocation.allocator_root_id != resource.allocator_root_id ||
          !append_allocation_totals(totals, allocation)) {
        return false;
      }
    }
    for (const auto& allocation : state.tensor_field_allocations) {
      if (allocation.allocator_process_group != process_group ||
          allocation.device != resource.device) {
        continue;
      }
      ++allocation_count;
      if (allocation.memory_kind != MemoryKind::kCudaDevice ||
          allocation.allocator_root_id != resource.allocator_root_id ||
          !append_allocation_totals(totals, allocation)) {
        return false;
      }
    }
    if (allocation_count == 0U || totals.pools != resource.pool_count ||
        totals.slots != resource.slot_capacity || totals.loans != resource.loan_credit_capacity ||
        totals.controls != resource.control_capacity ||
        totals.readiness != resource.readiness_capacity ||
        totals.releases != resource.release_capacity ||
        totals.retirements != resource.retirement_capacity ||
        totals.quarantines != resource.quarantine_capacity) {
      return false;
    }

    std::size_t expected_fused_edges{};
    for (const EdgeId edge_id : topology_edges) {
      const auto* edge = edge_record(state, edge_id);
      const auto* allocation = edge == nullptr ? nullptr : tensor_allocation(state, edge->source);
      const auto fields = edge == nullptr ? std::span<const TensorFieldAllocationPlanRecord>{}
                                          : tensor_field_allocations_by_output(state, edge->source);
      const bool direct_uses_resource = allocation != nullptr &&
                                        allocation->memory_kind == MemoryKind::kCudaDevice &&
                                        allocation->device == resource.device;
      const bool field_uses_resource = std::ranges::any_of(fields, [&](const auto& field) {
        return field.memory_kind == MemoryKind::kCudaDevice && field.device == resource.device;
      });
      if (edge == nullptr || !tensor_backed_payload(edge->payload_identity) ||
          (!direct_uses_resource && !field_uses_resource)) {
        continue;
      }
      ++expected_fused_edges;
      if (!std::binary_search(fused_linear_edges.begin(),
                              fused_linear_edges.end(),
                              edge_id,
                              [](EdgeId left, EdgeId right) { return left.value < right.value; })) {
        return false;
      }
    }
    if (expected_fused_edges != fused_linear_edges.size()) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::optional<ProcessGroupId> exact_process_group(
    const ExecutionPlanState& state, const std::vector<VertexId>& members) noexcept {
  if (state.vertex_process_placements.size() != members.size()) {
    return std::nullopt;
  }
  std::optional<ProcessGroupId> result;
  const std::string* launch_group = nullptr;
  for (const VertexId member : members) {
    const VertexProcessPlacementPlanRecord* placement = nullptr;
    for (const auto& candidate : state.vertex_process_placements) {
      if (candidate.vertex != member) {
        continue;
      }
      if (placement != nullptr) {
        return std::nullopt;
      }
      placement = &candidate;
    }
    if (placement == nullptr || !placement->process_group ||
        placement->isolation != ProcessIsolation::kSameProcess ||
        placement->process_role.has_value() || placement->runtime_seal_required) {
      return std::nullopt;
    }
    if (result && *result != placement->process_group) {
      return std::nullopt;
    }
    if (launch_group != nullptr && *launch_group != placement->launch_group) {
      return std::nullopt;
    }
    result = placement->process_group;
    launch_group = &placement->launch_group;
  }
  return result;
}

}  // namespace

// Eligibility is deliberately conservative. Every omitted shape continues through
// the ordinary controller/lane path without changing compile admission.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::optional<InlinePartitionExecutionPlanRecord> derive_inline_partition_execution_plan(
    const ExecutionPlanState& state) {
  // The inline fast path owns ordinary Operator callback sequencing. Fused CUDA Graph regions use
  // the partition runner so Runtime can substitute CUDAGraphRegionOperator for captured members.
  if (!state.cuda_graph_execution_regions.empty() || !state.admitted ||
      state.execution_partitions.size() != 1U ||
      state.execution_partitions.front().execution_lane != 0U ||
      state.execution_partitions.front().members.size() < 3U || state.schedule_edges.size() != 0U ||
      !exact_single_default_lane(state) || !supported_runtime_contracts(state)) {
    return std::nullopt;
  }
  const auto& partition = state.execution_partitions.front();
  if (!partition.id || partition.members.size() != state.executable_factories.operators.size()) {
    return std::nullopt;
  }
  for (const VertexId member : partition.members) {
    const auto* factory = operator_factory(state, member);
    if (factory == nullptr || !operator_parameters_are_supported(state, factory->operator_id) ||
        !operator_missing_aggregates_are_supported(state, member, factory->operator_id)) {
      return std::nullopt;
    }
  }

  const auto process_group = exact_process_group(state, partition.members);
  auto topology = stable_topology(state, partition);
  if (!process_group.has_value() || !topology.has_value()) {
    return std::nullopt;
  }
  const ProcessGroupId resolved_process_group = process_group.value();
  auto& resolved_topology = topology.value();
  if (!supported_tensor_resources(state, resolved_process_group, resolved_topology.edges)) {
    return std::nullopt;
  }

  std::vector<SourceId> readiness_sources;
  readiness_sources.reserve(resolved_topology.order.size());
  std::optional<RootReadinessFacts> root;
  for (const VertexId vertex : resolved_topology.order) {
    const auto* factory = operator_factory(state, vertex);
    if (factory == nullptr) {
      return std::nullopt;
    }
    const auto* trigger = find_unique_temporal_trigger_by_owner(state, factory->operator_id);
    if (trigger == nullptr) {
      return std::nullopt;
    }
    if (vertex == resolved_topology.root) {
      root = root_readiness(state, vertex, *trigger);
      if (!root) {
        return std::nullopt;
      }
    } else if (!input_driven_trigger(state, vertex, *trigger)) {
      return std::nullopt;
    }
    readiness_sources.push_back(trigger->readiness_source);
  }
  if (!root.has_value()) {
    return std::nullopt;
  }
  const RootReadinessFacts resolved_root = root.value_or(RootReadinessFacts{});

  return InlinePartitionExecutionPlanRecord{
      .partition = partition.id,
      .execution_lane = partition.execution_lane,
      .process_group = resolved_process_group,
      .readiness_root = resolved_topology.root,
      .readiness_kind = resolved_root.kind,
      .readiness_source = resolved_root.source,
      .readiness_clock = resolved_root.clock,
      .operators = std::move(resolved_topology.order),
      .readiness_sources = std::move(readiness_sources),
      .edges = std::move(resolved_topology.edges),
  };
}

bool valid_inline_partition_execution_plan(const ExecutionPlanState& state) {
  return state.inline_partition_execution == derive_inline_partition_execution_plan(state);
}

}  // namespace holoscan::detail
