/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/closed_island_activation_plan.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "compiler/process_local_channel_plan_validation.hpp"

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

[[nodiscard]] const ExecutionPartitionRecord* execution_partition_for(
    const ExecutionPlanState& state, VertexId vertex) noexcept {
  const ExecutionPartitionRecord* result = nullptr;
  for (const auto& partition : state.execution_partitions) {
    if (std::find(partition.members.begin(), partition.members.end(), vertex) ==
        partition.members.end()) {
      continue;
    }
    if (result != nullptr) {
      return nullptr;
    }
    result = &partition;
  }
  return result;
}

[[nodiscard]] std::optional<ProcessGroupId> process_group_for(const ExecutionPlanState& state,
                                                              VertexId vertex) noexcept {
  std::optional<ProcessGroupId> result;
  for (const auto& candidate : state.vertex_process_placements) {
    if (candidate.vertex != vertex) {
      continue;
    }
    if (result.has_value() || !candidate.process_group) {
      return std::nullopt;
    }
    result = candidate.process_group;
  }
  return result;
}

[[nodiscard]] std::size_t process_group_operator_count(const ExecutionPlanState& state,
                                                       ProcessGroupId process_group) noexcept {
  return static_cast<std::size_t>(
      std::count_if(state.executable_factories.operators.begin(),
                    state.executable_factories.operators.end(),
                    [&](const auto& candidate) {
                      const auto candidate_group = process_group_for(state, candidate.vertex);
                      return candidate_group && *candidate_group == process_group;
                    }));
}

[[nodiscard]] const TemporalTriggerPlanRecord* temporal_trigger_for(
    const ExecutionPlanState& state, OperatorInstanceId owner) noexcept {
  const TemporalTriggerPlanRecord* result = nullptr;
  for (const auto& candidate : state.temporal_triggers) {
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

[[nodiscard]] bool operator_parameters_are_frozen(const ExecutionPlanState& state,
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
  if (parameters == nullptr || parameters->runtime_updates_enabled ||
      parameters->lifecycle_updates_enabled) {
    return false;
  }
  return std::none_of(state.lifecycle_mutable_fields.begin(),
                      state.lifecycle_mutable_fields.end(),
                      [&](const auto& candidate) { return candidate.owner.id == owner; }) &&
         std::none_of(state.parameter_derived_terms.begin(),
                      state.parameter_derived_terms.end(),
                      [&](const auto& candidate) { return candidate.owner.id == owner; });
}

[[nodiscard]] bool has_advanced_temporal_state(const ExecutionPlanState& state,
                                               OperatorInstanceId owner) noexcept {
  const auto owned = [owner](const auto& candidate) { return candidate.owner.id == owner; };
  return std::any_of(state.aligned_triggers.begin(), state.aligned_triggers.end(), owned) ||
         std::any_of(state.deadlines.begin(), state.deadlines.end(), owned) ||
         std::any_of(state.window_triggers.begin(), state.window_triggers.end(), owned) ||
         std::any_of(state.bracketed_windows.begin(), state.bracketed_windows.end(), owned) ||
         std::any_of(state.predicate_triggers.begin(), state.predicate_triggers.end(), owned);
}

[[nodiscard]] bool exact_successor_contract(const ExecutionPlanState& state,
                                            OperatorInstanceId owner, PortId input) noexcept {
  const auto* input_port = dense_port(state, input);
  if (input_port == nullptr) {
    return false;
  }
  const auto* trigger = temporal_trigger_for(state, owner);
  if (trigger == nullptr || trigger->kind != TemporalRecordTriggerKind::kEach ||
      trigger->trigger_input != input || !trigger->declared_inputs.empty() ||
      trigger->presence_guaranteed_inputs.size() != 1U ||
      trigger->presence_guaranteed_inputs.front() != input || !trigger->readiness_source ||
      trigger->deadline_ns.has_value() || trigger->violation.has_value()) {
    return false;
  }

  const InputSelectionPlanRecord* selection = nullptr;
  for (const auto& candidate : state.input_selections) {
    const auto* port = dense_port(state, candidate.input);
    if (port == nullptr || port->owner != input_port->owner) {
      continue;
    }
    if (selection != nullptr) {
      return false;
    }
    selection = &candidate;
  }
  // OnEach's trigger input lowers as an implicit Exact single-sample selection.
  // An explicit selection row therefore represents additional state and is not
  // part of this conservative closed-island shape.
  return selection == nullptr;
}

struct ReadinessRootFacts {
  ClosedIslandReadinessRootKind kind{ClosedIslandReadinessRootKind::kPhysical};
  SourceId source{};
  ClockId clock{};
};

[[nodiscard]] std::optional<ReadinessRootFacts> physical_readiness_root(
    const ExecutionPlanState& state, VertexId vertex, OperatorInstanceId owner,
    const TemporalTriggerPlanRecord& trigger, ReadinessSourceKind expected_kind,
    ClosedIslandReadinessRootKind root_kind) noexcept {
  const ReadinessPlanRecord* readiness = nullptr;
  for (const auto& candidate : state.readiness) {
    if (candidate.source_id != trigger.readiness_source) {
      continue;
    }
    if (readiness != nullptr) {
      return std::nullopt;
    }
    readiness = &candidate;
  }
  if (readiness == nullptr || readiness->owner != vertex || readiness->kind != expected_kind ||
      !readiness->referenced || !readiness->materialized || !readiness->resolved_clock.valid() ||
      trigger.reference_clock != readiness->resolved_clock) {
    return std::nullopt;
  }

  const auto readable_count = std::count_if(
      state.readable_triggers.begin(), state.readable_triggers.end(), [&](const auto& candidate) {
        return candidate.owner.id == owner;
      });
  if (expected_kind == ReadinessSourceKind::kReadable) {
    const auto readable = std::find_if(
        state.readable_triggers.begin(), state.readable_triggers.end(), [&](const auto& candidate) {
          return candidate.owner.id == owner;
        });
    if (readable_count != 1 || readable == state.readable_triggers.end() ||
        readable->source != trigger.readiness_source ||
        readable->witness_clock != readiness->resolved_clock) {
      return std::nullopt;
    }
  } else if (readable_count != 0) {
    return std::nullopt;
  }

  return ReadinessRootFacts{
      .kind = root_kind,
      .source = trigger.readiness_source,
      .clock = readiness->resolved_clock,
  };
}

[[nodiscard]] std::optional<ReadinessRootFacts> source_readiness_root(
    const ExecutionPlanState& state, VertexId vertex, OperatorInstanceId owner) noexcept {
  const auto* trigger = temporal_trigger_for(state, owner);
  if (trigger == nullptr || trigger->trigger_input || !trigger->declared_inputs.empty() ||
      !trigger->presence_guaranteed_inputs.empty() || !trigger->readiness_source ||
      trigger->deadline_ns.has_value() || trigger->violation.has_value()) {
    return std::nullopt;
  }

  if (trigger->kind == TemporalRecordTriggerKind::kClock) {
    const ClockTriggerPlanRecord* clock = nullptr;
    for (const auto& candidate : state.clock_triggers) {
      if (candidate.owner.id != owner) {
        continue;
      }
      if (clock != nullptr) {
        return std::nullopt;
      }
      clock = &candidate;
    }
    if (clock == nullptr || !trigger->trigger_clock.valid() ||
        clock->clock != trigger->trigger_clock || clock->period_ns != trigger->period_ns) {
      return std::nullopt;
    }
    return ReadinessRootFacts{
        .kind = ClosedIslandReadinessRootKind::kClock,
        .source = trigger->readiness_source,
        .clock = trigger->trigger_clock,
    };
  }

  switch (trigger->kind) {
    case TemporalRecordTriggerKind::kReadable:
      return physical_readiness_root(state,
                                     vertex,
                                     owner,
                                     *trigger,
                                     ReadinessSourceKind::kReadable,
                                     ClosedIslandReadinessRootKind::kPhysical);
    case TemporalRecordTriggerKind::kNotified:
      return physical_readiness_root(state,
                                     vertex,
                                     owner,
                                     *trigger,
                                     ReadinessSourceKind::kNotification,
                                     ClosedIslandReadinessRootKind::kNotification);
    case TemporalRecordTriggerKind::kDoorbell:
      return physical_readiness_root(state,
                                     vertex,
                                     owner,
                                     *trigger,
                                     ReadinessSourceKind::kDoorbell,
                                     ClosedIslandReadinessRootKind::kPhysical);
    case TemporalRecordTriggerKind::kSpin:
      return physical_readiness_root(state,
                                     vertex,
                                     owner,
                                     *trigger,
                                     ReadinessSourceKind::kSpin,
                                     ClosedIslandReadinessRootKind::kPhysical);
    default:
      return std::nullopt;
  }
}

[[nodiscard]] bool exact_process_local_service(const ExecutionPlanState& state,
                                               const PayloadEdgePlanRecord& edge) noexcept {
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
  if (service == nullptr ||
      service->delivery_kind != ProcessLocalChannelDeliveryKind::kDirectRetained ||
      service->cross_process_transport || service->queue_policy != QueuePolicy::kBlocking ||
      service->tensor_payload || service->max_subscribers != 1U ||
      service->max_subscriber_queue_depth != 1U || service->max_subscriber_borrowed_samples != 1U ||
      service->publisher_history_depth != 0U || service->event_wake) {
    return false;
  }

  const ProcessLocalChannelPublisherPlanRecord* publisher = nullptr;
  for (const auto& candidate : state.process_local_channel_publishers) {
    if (candidate.source != edge.source || candidate.service_id != service->service_id) {
      continue;
    }
    if (publisher != nullptr) {
      return false;
    }
    publisher = &candidate;
  }
  if (publisher == nullptr || publisher->max_loaned_samples != 1U ||
      publisher->replacement_attribution_required) {
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
         subscriber->target == edge.target && subscriber->service_id == service->service_id &&
         subscriber->requested_queue_depth == 1U && subscriber->max_borrowed_samples == 1U &&
         subscriber->cache_capacity == 0U && subscriber->readiness_capacity == 1U &&
         !subscriber->event_wake;
}

// Eligibility is deliberately conservative. Every omitted shape continues through
// the ordinary controller/lane path without changing compile admission.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::optional<ClosedIslandActivationPlanRecord> derive_edge(
    const ExecutionPlanState& state, const PayloadEdgePlanRecord& edge) noexcept {
  if (edge.disposition != PlanDisposition::kAccepted ||
      state.lifecycle_transaction_capacity.has_value()) {
    return std::nullopt;
  }
  const auto* source_port = dense_port(state, edge.source);
  const auto* target_port = dense_port(state, edge.target);
  if (source_port == nullptr || target_port == nullptr ||
      source_port->owner_kind != PortOwnerKind::kOperator ||
      target_port->owner_kind != PortOwnerKind::kOperator ||
      source_port->owner == target_port->owner ||
      edge.locality != ChannelLocalityKind::kProcessLocal ||
      edge.queue_policy != QueuePolicy::kBlocking || edge.queue_depth != 1U ||
      edge.producer_max_emits_per_compute != 1U || edge.fanout != 1U || edge.cache_capacity != 0U ||
      source_port->direction != PortDirection::kOutput ||
      source_port->max_emits_per_compute != 1U || target_port->direction != PortDirection::kInput ||
      process_local_channel_delivery_kind(edge) !=
          ProcessLocalChannelDeliveryKind::kDirectRetained) {
    return std::nullopt;
  }

  for (const auto& candidate : state.payload_edges) {
    if (candidate.disposition != PlanDisposition::kAccepted || candidate.edge == edge.edge) {
      continue;
    }
    const auto* candidate_source = dense_port(state, candidate.source);
    const auto* candidate_target = dense_port(state, candidate.target);
    if (candidate_source == nullptr || candidate_target == nullptr ||
        candidate_source->owner == source_port->owner ||
        candidate_source->owner == target_port->owner ||
        candidate_target->owner == source_port->owner ||
        candidate_target->owner == target_port->owner) {
      return std::nullopt;
    }
  }

  const bool supported_payload =
      (edge.payload_identity.kind == PayloadIdentityKind::kPod &&
       edge.payload_identity.payload_kind == PayloadKind::kInlineTyped &&
       edge.payload_identity.pod.has_value()) ||
      (edge.payload_identity.kind == PayloadIdentityKind::kSchema &&
       edge.payload_identity.payload_kind == PayloadKind::kSerializedBytes &&
       edge.payload_identity.schema.has_value());
  if (!supported_payload || !exact_process_local_service(state, edge)) {
    return std::nullopt;
  }

  const auto input_count =
      std::count_if(state.ports.begin(), state.ports.end(), [&](const auto& p) {
        return p.owner == target_port->owner && p.owner_kind == PortOwnerKind::kOperator &&
               p.direction == PortDirection::kInput;
      });
  const auto source_input_count =
      std::count_if(state.ports.begin(), state.ports.end(), [&](const auto& p) {
        return p.owner == source_port->owner && p.owner_kind == PortOwnerKind::kOperator &&
               p.direction == PortDirection::kInput;
      });
  if (input_count != 1 || source_input_count != 0) {
    return std::nullopt;
  }

  const auto* source_factory = operator_factory(state, source_port->owner);
  const auto* target_factory = operator_factory(state, target_port->owner);
  const auto* source_partition = execution_partition_for(state, source_port->owner);
  const auto* target_partition = execution_partition_for(state, target_port->owner);
  const auto source_group = process_group_for(state, source_port->owner);
  const auto target_group = process_group_for(state, target_port->owner);
  if (source_factory == nullptr || target_factory == nullptr || source_partition == nullptr ||
      target_partition == nullptr || source_partition->id != target_partition->id ||
      source_partition->execution_lane != target_partition->execution_lane || !source_group ||
      !target_group || *source_group != *target_group ||
      process_group_operator_count(state, *source_group) != 2U ||
      !operator_parameters_are_frozen(state, source_factory->operator_id) ||
      !operator_parameters_are_frozen(state, target_factory->operator_id) ||
      has_advanced_temporal_state(state, source_factory->operator_id) ||
      has_advanced_temporal_state(state, target_factory->operator_id) ||
      !exact_successor_contract(state, target_factory->operator_id, target_port->id)) {
    return std::nullopt;
  }

  const auto root = source_readiness_root(state, source_port->owner, source_factory->operator_id);
  const auto* successor_trigger = temporal_trigger_for(state, target_factory->operator_id);
  if (!root || successor_trigger == nullptr || !successor_trigger->readiness_source) {
    return std::nullopt;
  }

  return ClosedIslandActivationPlanRecord{
      .source_partition = source_partition->id,
      .successor_partition = target_partition->id,
      .execution_lane = source_partition->execution_lane,
      .process_group = *source_group,
      .source = source_port->owner,
      .successor = target_port->owner,
      .edge = edge.edge,
      .source_output = source_port->id,
      .successor_input = target_port->id,
      .readiness_root = root->kind,
      .readiness_source = root->source,
      .readiness_clock = root->clock,
      .successor_readiness_source = successor_trigger->readiness_source,
  };
}

}  // namespace

std::vector<ClosedIslandActivationPlanRecord> derive_closed_island_activation_plan(
    const ExecutionPlanState& state) {
  std::vector<ClosedIslandActivationPlanRecord> result;
  result.reserve(state.payload_edges.size());
  for (const auto& edge : state.payload_edges) {
    if (auto record = derive_edge(state, edge)) {
      result.push_back(std::move(*record));
    }
  }
  return result;
}

bool valid_closed_island_activation_plan(const ExecutionPlanState& state) {
  return valid_process_local_channel_plan(state) &&
         state.closed_island_activations == derive_closed_island_activation_plan(state);
}

}  // namespace holoscan::detail
