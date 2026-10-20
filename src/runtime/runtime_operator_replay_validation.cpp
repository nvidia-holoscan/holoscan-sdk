/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_operator_replay_validation.hpp"

#include "runtime_operator_plan_validation.hpp"
#include "runtime_partition_plan_validation.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>

#include "../channel/endpoint_catalog_state.hpp"
#include "../compiler/closed_island_activation_plan.hpp"
#include "../compiler/execution_plan_lookup.hpp"
#include "../compiler/inline_partition_execution_plan.hpp"
#include "../core/operator_runtime_access.hpp"
#include "../core/operator_spec_compiler_access.hpp"
#include "../core/plan_record_contracts.hpp"
#include "../core/port_configuration_internal.hpp"
#include "../core/readiness_source_internal.hpp"
#include "../core/source_id_internal.hpp"
#include "../core/temporal_contract_internal.hpp"
#include "dispatcher.hpp"
#include "runtime_clock_reconstruction.hpp"
#include "runtime_work_inventory.hpp"
#include "tensor_boundary_contract_comparison.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] ParamSchemaHash expected_replay_fingerprint(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record,
    std::uint32_t factory_slot, RuntimeReplayMismatchCategory category) noexcept {
  StableHashBuilder hash;
  hash.add("holoscan.runtime.operator-replay.expected/v1");
  for (const std::byte byte : plan.digest.value) {
    hash.add_byte(static_cast<std::uint8_t>(byte));
  }
  hash.add(record.vertex.value);
  hash.add(record.operator_id);
  hash.add(factory_slot);
  hash.add(category);
  return hash.finish();
}

void add_replay_clock_key(StableHashBuilder& hash, const ClockKey& key) noexcept {
  hash.add(key.is_default());
  hash.add(key.name());
}

void add_replay_time_point(StableHashBuilder& hash,
                           const std::optional<TimePoint>& value) noexcept {
  hash.add(value.has_value());
  if (value) {
    hash.add(value->clock_id.value);
    hash.add(value->timestamp_ns);
  }
}

// Hash only pointer-free declaration and normalized contract facts. This is diagnostic identity,
// not an alternate replay predicate: the exact fail-closed comparisons below remain authoritative.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] ParamSchemaHash observed_replay_fingerprint(
    const OperatorSpec& replay, RuntimeReplayMismatchCategory category,
    const TemporalContractSnapshot* temporal = nullptr) {
  StableHashBuilder hash;
  hash.add("holoscan.runtime.operator-replay.observed/v2");
  hash.add(category);
  hash.add(OperatorSpecCompilerAccess::declaration_error(replay).code);
  hash.add(OperatorSpecCompilerAccess::cuda_graph_capture_eligible(replay));

  hash.add_canonical_size(OperatorSpecCompilerAccess::port_count(replay));
  for (std::size_t index = 0U; index < OperatorSpecCompilerAccess::port_count(replay); ++index) {
    const auto port = OperatorSpecCompilerAccess::port(replay, index);
    hash.add(port.direction);
    hash.add(port.name);
    hash.add(port.tensor);
    hash.add(port.optional);
    hash.add(port.payload_identity.kind);
    hash.add(port.payload_identity.payload_kind);
    hash.add(port.serialized_payload_capacity_bytes);
    hash.add(port.queue_depth.has_value());
    if (port.queue_depth) {
      hash.add(*port.queue_depth);
    }
    hash.add(port.max_emits_per_compute);
    hash.add(port.late_joiner_history_depth);
    hash.add(port.rate_error.code);
    hash.add(port.rate_kind);
    hash.add(port.nominal_rate.has_value());
    if (port.nominal_rate) {
      hash.add(port.nominal_rate->numerator_hz);
      hash.add(port.nominal_rate->denominator_hz);
    }
    hash.add(port.expected_rate.has_value());
    if (port.expected_rate) {
      hash.add(port.expected_rate->numerator_hz);
      hash.add(port.expected_rate->denominator_hz);
    }
    hash.add(port.follows_input_declaration != nullptr);
    if (port.follows_input_declaration != nullptr) {
      bool found = false;
      for (std::size_t candidate_index = 0U;
           candidate_index < OperatorSpecCompilerAccess::port_count(replay);
           ++candidate_index) {
        const auto candidate = OperatorSpecCompilerAccess::port(replay, candidate_index);
        if (candidate.declaration != port.follows_input_declaration) {
          continue;
        }
        hash.add(candidate.direction);
        hash.add(candidate.name);
        found = true;
        break;
      }
      hash.add(found);
    }
    hash.add(port.follows_ratio.has_value());
    if (port.follows_ratio) {
      hash.add(port.follows_ratio->numerator);
      hash.add(port.follows_ratio->denominator);
    }
    hash.add(port.lifecycle_binding_stage.has_value());
    if (port.lifecycle_binding_stage) {
      hash.add(*port.lifecycle_binding_stage);
    }
    hash.add(port.max_nominal_rate.has_value());
    if (port.max_nominal_rate) {
      hash.add(port.max_nominal_rate->numerator_hz);
      hash.add(port.max_nominal_rate->denominator_hz);
    }
    hash.add(port.diagnostic_alias);
  }

  hash.add_canonical_size(OperatorSpecCompilerAccess::source_count(replay));
  for (std::size_t index = 0U; index < OperatorSpecCompilerAccess::source_count(replay); ++index) {
    const auto source = OperatorSpecCompilerAccess::source(replay, index);
    hash.add(source.name);
    hash.add(source.kind);
    hash.add(source.capacity);
    hash.add(source.sender_reference_capacity);
    add_replay_clock_key(hash, source.clock);
  }

  hash.add_canonical_size(OperatorSpecCompilerAccess::lifecycle_callback_count(replay));
  for (std::size_t index = 0U; index < OperatorSpecCompilerAccess::lifecycle_callback_count(replay);
       ++index) {
    const auto callback = OperatorSpecCompilerAccess::lifecycle_callback(replay, index);
    hash.add(callback.stage);
    hash.add(callback.required);
    hash.add(callback.method);
  }

  hash.add(temporal != nullptr);
  if (temporal == nullptr) {
    return hash.finish();
  }
  const TemporalTriggerRecord& trigger = temporal->trigger;
  hash.add(trigger.kind);
  hash.add(trigger.port_name);
  add_replay_clock_key(hash, trigger.clock);
  add_replay_time_point(hash, trigger.start_at);
  hash.add(trigger.duration.count());
  hash.add(trigger.tick_policy);
  hash.add(trigger.capacity);
  hash.add_canonical_size(trigger.ports.size());
  for (const std::string& port : trigger.ports) {
    hash.add(port);
  }
  hash.add(source_id_value(trigger.source_id));
  hash.add_canonical_size(trigger.predicate_input_names.size());
  for (const std::string& input : trigger.predicate_input_names) {
    hash.add(input);
  }
  hash.add(trigger.predicate_semantic_version.has_value());
  if (trigger.predicate_semantic_version) {
    hash.add(trigger.predicate_semantic_version->major);
    hash.add(trigger.predicate_semantic_version->minor);
    hash.add(trigger.predicate_semantic_version->patch);
  }
  hash.add(trigger.predicate_firing);
  hash.add(trigger.predicate_clock_source);
  hash.add(trigger.predicate_expects_driver);
  hash.add(trigger.predicate_callable_noexcept);
  hash.add(trigger.predicate_callable_returns_expected);
  hash.add_canonical_size(trigger.predicate_callable_size);
  hash.add_canonical_size(trigger.predicate_callable_alignment);
  hash.add_canonical_size(trigger.predicate_state_size);
  hash.add_canonical_size(trigger.predicate_state_alignment);
  hash.add_canonical_size(trigger.predicate_config_size);
  hash.add_canonical_size(trigger.predicate_config_alignment);
  hash.add(trigger.predicate_config_identity_available);
  hash.add(trigger.predicate_config_schema);
  hash.add(trigger.predicate_config_digest);

  hash.add_canonical_size(temporal->inputs.size());
  for (const TemporalInputRecord& input : temporal->inputs) {
    hash.add(input.port_name);
    hash.add(input.selector);
    hash.add(input.max_age.count());
    hash.add(input.bracket.lookbehind);
    hash.add(input.bracket.lookahead);
    hash.add(input.bracket.max_samples);
    hash.add(input.missing);
    hash.add(input.reuse_max_age.count());
  }
  hash.add_canonical_size(temporal->connected_input_count);
  hash.add(temporal->max_start_delay.has_value());
  if (temporal->max_start_delay) {
    hash.add(temporal->max_start_delay->count());
  }
  hash.add(temporal->violation.has_value());
  if (temporal->violation) {
    hash.add(*temporal->violation);
  }
  hash.add(temporal->synthesized_default);
  return hash.finish();
}

[[nodiscard]] Error replay_mismatch_error(
    ErrorCode code, const char* token, const ExecutionPlanState& plan,
    const ExecutableOperatorFactoryRecord& record, std::uint32_t factory_slot,
    RuntimeReplayMismatchCategory category, const OperatorSpec& replay,
    RuntimeReplayMismatchDetail* detail, const TemporalContractSnapshot* temporal = nullptr) {
  if (detail != nullptr) {
    *detail = RuntimeReplayMismatchDetail{
        .present = true,
        .vertex = record.vertex,
        .operator_id = record.operator_id,
        .factory_slot = factory_slot,
        .category = category,
        .expected = expected_replay_fingerprint(plan, record, factory_slot, category),
        .observed = observed_replay_fingerprint(replay, category, temporal),
    };
  }
  return operator_reconstruction_error(code, token);
}

[[nodiscard]] std::size_t planned_port_count(const ExecutionPlanState& plan,
                                             VertexId owner) noexcept {
  return static_cast<std::size_t>(
      std::count_if(plan.graph.ports().begin(), plan.graph.ports().end(), [&](const auto& port) {
        return port.owner == owner;
      }));
}

[[nodiscard]] std::size_t planned_source_count(const ExecutionPlanState& plan,
                                               VertexId owner) noexcept {
  return static_cast<std::size_t>(
      std::count_if(plan.readiness.begin(), plan.readiness.end(), [&](const auto& source) {
        return source.owner == owner;
      }));
}

[[nodiscard]] bool replay_runtime_codec_valid(
    const OperatorSpecCompilerAccess::PortMetadata& metadata) noexcept {
  const RuntimePortTypeOps* ops = metadata.runtime_type_ops;
  if (ops == nullptr || ops->payload_size == 0U) {
    return false;
  }
  switch (ops->codec_family) {
    case RuntimePayloadCodecFamily::kInlinePod:
      return ops->inline_pod && ops->payload_kind == PayloadKind::kInlineTyped &&
             ops->encode_payload == nullptr && ops->wire_capacity == ops->payload_size &&
             metadata.payload_identity.kind == PayloadIdentityKind::kScalar &&
             metadata.serialized_payload_capacity_bytes == 0U;
    case RuntimePayloadCodecFamily::kCustomPod:
      return ops->inline_pod && ops->payload_kind == PayloadKind::kInlineTyped &&
             ops->encode_payload == nullptr && ops->wire_capacity == ops->payload_size &&
             metadata.payload_identity.kind == PayloadIdentityKind::kPod &&
             metadata.payload_identity.pod.has_value() &&
             metadata.serialized_payload_capacity_bytes == 0U;
    case RuntimePayloadCodecFamily::kGeneratedSchema:
      return !ops->inline_pod && ops->payload_kind == PayloadKind::kSerializedBytes &&
             ops->encode_payload != nullptr &&
             (!ops->persistent_sample_storage ||
              (ops->prepare_sample_storage != nullptr && ops->reset_sample_storage != nullptr &&
               ops->destroy_sample_storage != nullptr && ops->assign_sample_storage != nullptr)) &&
             metadata.payload_identity.kind == PayloadIdentityKind::kSchema &&
             metadata.serialized_payload_capacity_bytes == ops->wire_capacity;
    case RuntimePayloadCodecFamily::kDescriptorTensorSchema:
      return !ops->inline_pod && ops->payload_kind == PayloadKind::kDescriptorBacked &&
             ops->wire_capacity != 0U && ops->descriptor_offset <= ops->wire_capacity &&
             ops->descriptor_tensor_static_count != 0U &&
             ops->descriptor_tensor_static_count <=
                 channel_internal::kMaximumDescriptorTensorFields &&
             static_cast<std::size_t>(ops->descriptor_tensor_static_count) *
                     sizeof(PayloadDescriptor) <=
                 ops->wire_capacity - ops->descriptor_offset &&
             ops->persistent_sample_storage && ops->descriptor_tensor_count != nullptr &&
             ops->descriptor_tensor_at != nullptr &&
             (ops->descriptor_tensor_static_count != 1U ||
              (ops->descriptor_tensor != nullptr &&
               ops->assign_descriptor_tensor_sample != nullptr)) &&
             ops->encode_descriptor_tensor_sample != nullptr &&
             ops->descriptor_tensor_sample_metadata != nullptr &&
             ops->descriptor_tensor_view_metadata_at != nullptr &&
             ops->assign_descriptor_tensors_sample != nullptr &&
             ops->prepare_sample_storage != nullptr && ops->reset_sample_storage != nullptr &&
             ops->destroy_sample_storage != nullptr && ops->destroy_sample != nullptr &&
             ops->sample_payload != nullptr && ops->sample_metadata != nullptr &&
             ops->bind_input != nullptr && ops->unbind_input != nullptr &&
             ops->bind_output != nullptr && ops->unbind_output != nullptr &&
             metadata.payload_identity.kind == PayloadIdentityKind::kSchema &&
             metadata.payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
             metadata.payload_identity.schema.has_value() &&
             metadata.serialized_payload_capacity_bytes == ops->wire_capacity;
    case RuntimePayloadCodecFamily::kUnsupported:
      // Tensor samples use the typed sidecar and immutable backing authority rather than an
      // inline/schema byte codec. Admit only the one Core-owned Tensor operation table; an
      // arbitrary payload whose codec remains unsupported must still fail reconstruction.
      return ops == runtime_port_type_ops<Tensor>() && ops->wire_capacity == 0U &&
             ops->payload_kind == PayloadKind::kDescriptorBacked && !ops->inline_pod &&
             ops->encode_payload == nullptr && ops->construct_sample == nullptr &&
             ops->destroy_sample != nullptr && ops->sample_payload != nullptr &&
             ops->sample_metadata != nullptr && ops->bind_input != nullptr &&
             ops->unbind_input != nullptr && ops->bind_output != nullptr &&
             ops->unbind_output != nullptr &&
             metadata.payload_identity.kind == PayloadIdentityKind::kTensor &&
             metadata.payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
             !metadata.payload_identity.scalar && !metadata.payload_identity.pod &&
             !metadata.payload_identity.schema && metadata.serialized_payload_capacity_bytes == 0U;
    case RuntimePayloadCodecFamily::kProcessLocalObject:
      return ops->wire_capacity == 0U && ops->payload_kind == PayloadKind::kInlineTyped &&
             !ops->inline_pod && ops->encode_payload == nullptr &&
             ops->construct_sample == nullptr && ops->construct_sample_from_value != nullptr &&
             ops->retain_sample_payload_for_deferred_release != nullptr &&
             ops->destroy_sample != nullptr && ops->sample_payload != nullptr &&
             ops->sample_metadata != nullptr && ops->bind_input != nullptr &&
             ops->unbind_input != nullptr && ops->bind_output != nullptr &&
             ops->unbind_output != nullptr &&
             is_py_object_payload_identity(metadata.payload_identity) &&
             metadata.serialized_payload_capacity_bytes == 0U;
  }
  return false;
}

// Compare the complete authored port inventory without weakening exact replay admission.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool replayed_ports_match(const ExecutionPlanState& plan,
                                        const ExecutableOperatorFactoryRecord& record,
                                        const OperatorSpec& replay) {
  const VertexId owner = record.vertex;
  const auto planned_ports = plan.graph.ports();
  std::vector<bool> matched(planned_ports.size(), false);
  for (std::size_t index = 0U; index < OperatorSpecCompilerAccess::port_count(replay); ++index) {
    const auto metadata = OperatorSpecCompilerAccess::port(replay, index);
    std::size_t planned_index = 0U;
    while (planned_index < planned_ports.size() &&
           (matched[planned_index] || planned_ports[planned_index].owner != owner ||
            planned_ports[planned_index].name != metadata.name ||
            planned_ports[planned_index].direction != metadata.direction)) {
      ++planned_index;
    }
    if (planned_index == planned_ports.size() || metadata.declaration == nullptr ||
        metadata.local_type_token == nullptr || metadata.runtime_type_ops == nullptr ||
        !replay_runtime_codec_valid(metadata) ||
        planned_ports[planned_index].optional != metadata.optional ||
        !payload_identity_exactly_equal(planned_ports[planned_index].payload_identity,
                                        metadata.payload_identity)) {
      return false;
    }
    if (metadata.payload_identity.scalar &&
        metadata.payload_identity.scalar->kind == ScalarIdentityKind::kLocalEnum) {
      const ExecutableLocalEnumPortRecord* expected = nullptr;
      for (const auto& candidate : record.local_enum_ports) {
        if (candidate.direction == metadata.direction && candidate.name == metadata.name) {
          if (expected != nullptr) {
            return false;
          }
          expected = &candidate;
        }
      }
      if (expected == nullptr || expected->local_type_token != metadata.local_type_token) {
        return false;
      }
    }
    std::optional<GraphPortId> follows_input;
    if (metadata.follows_input_declaration != nullptr) {
      for (std::size_t candidate_index = 0U;
           candidate_index < OperatorSpecCompilerAccess::port_count(replay);
           ++candidate_index) {
        const auto candidate = OperatorSpecCompilerAccess::port(replay, candidate_index);
        if (candidate.declaration != metadata.follows_input_declaration ||
            candidate.direction != PortDirection::kInput) {
          continue;
        }
        const auto planned_input = std::ranges::find_if(planned_ports, [&](const auto& port) {
          return port.owner == owner && port.owner_kind == PortOwnerKind::kOperator &&
                 port.direction == PortDirection::kInput && port.name == candidate.name;
        });
        if (planned_input != planned_ports.end()) {
          follows_input = planned_input->id;
        }
        break;
      }
    }
    const PortConfigurationRecord candidate_setup =
        setup_port_configuration(metadata, follows_input);
    const PortConfigurationRecord& frozen_setup = planned_ports[planned_index].setup;
    const std::uint64_t port_value = planned_ports[planned_index].id.value;
    if (port_value == 0U || port_value > plan.port_rates.size()) {
      return false;
    }
    const PortRatePlanRecord& rate = plan.port_rates[static_cast<std::size_t>(port_value - 1U)];
    if (rate.port != PortId(port_value) || rate.disposition != PlanDisposition::kAccepted ||
        frozen_setup.queue_depth != candidate_setup.queue_depth ||
        frozen_setup.max_emits_per_compute != candidate_setup.max_emits_per_compute ||
        frozen_setup.late_joiner_history_depth != candidate_setup.late_joiner_history_depth ||
        frozen_setup.rate_kind != candidate_setup.rate_kind ||
        frozen_setup.declared_nominal_rate != candidate_setup.declared_nominal_rate ||
        frozen_setup.declared_expected_rate != candidate_setup.declared_expected_rate ||
        frozen_setup.follows_input != candidate_setup.follows_input ||
        frozen_setup.follows_ratio != candidate_setup.follows_ratio ||
        frozen_setup.lifecycle_binding_stage != candidate_setup.lifecycle_binding_stage ||
        frozen_setup.declared_max_nominal_rate != candidate_setup.declared_max_nominal_rate ||
        rate.kind != candidate_setup.rate_kind ||
        rate.declared_nominal_rate != candidate_setup.declared_nominal_rate ||
        rate.declared_expected_rate != candidate_setup.declared_expected_rate ||
        rate.follows_input !=
            (candidate_setup.follows_input
                 ? std::optional<PortId>{PortId(candidate_setup.follows_input->value)}
                 : std::nullopt) ||
        rate.follows_ratio != candidate_setup.follows_ratio ||
        rate.lifecycle_binding_stage != candidate_setup.lifecycle_binding_stage ||
        rate.declared_max_nominal_rate != candidate_setup.declared_max_nominal_rate) {
      return false;
    }
    matched[planned_index] = true;
  }
  return true;
}

[[nodiscard]] bool replayed_completion_rule_matches(
    const ExecutionPlanState& plan, OperatorInstanceId owner,
    const TemporalContractSnapshot& replay) noexcept {
  const auto planned = std::find_if(plan.operator_completion_rules.begin(),
                                    plan.operator_completion_rules.end(),
                                    [&](const auto& rule) { return rule.owner.id == owner; });
  const std::size_t planned_count = static_cast<std::size_t>(
      std::count_if(plan.operator_completion_rules.begin(),
                    plan.operator_completion_rules.end(),
                    [&](const auto& rule) { return rule.owner.id == owner; }));
  if (!replay.compute_calls) {
    return planned_count == 0U;
  }
  return planned_count == 1U && planned != plan.operator_completion_rules.end() &&
         planned->owner.plan_generation == plan.plan_generation &&
         planned->owner.run_generation == 0U &&
         planned->kind == OperatorCompletionRuleKind::kComputeCalls &&
         planned->limit == replay.compute_calls->limit;
}

[[nodiscard]] const PortRecord* planned_graph_port(const ExecutionPlanState& plan, VertexId owner,
                                                   PortDirection direction,
                                                   std::string_view name) noexcept {
  const auto found =
      std::find_if(plan.graph.ports().begin(), plan.graph.ports().end(), [&](const auto& port) {
        return port.owner == owner && port.direction == direction && port.name == name;
      });
  return found == plan.graph.ports().end() ? nullptr : &*found;
}

[[nodiscard]] std::vector<GraphPortId> replay_alias_component(const ExecutionPlanState& plan,
                                                              GraphPortId initial) {
  std::vector<GraphPortId> component{initial};
  for (std::size_t cursor = 0U; cursor < component.size(); ++cursor) {
    for (const auto& binding : plan.graph.interface_bindings()) {
      GraphPortId adjacent{};
      if (binding.public_port == component[cursor]) {
        adjacent = binding.internal_port;
      } else if (binding.internal_port == component[cursor]) {
        adjacent = binding.public_port;
      }
      if (adjacent && std::find(component.begin(), component.end(), adjacent) == component.end()) {
        component.push_back(adjacent);
      }
    }
  }
  return component;
}

[[nodiscard]] bool replay_input_connected(const ExecutionPlanState& plan, GraphPortId input) {
  const auto aliases = replay_alias_component(plan, input);
  return std::any_of(plan.graph.edges().begin(), plan.graph.edges().end(), [&](const auto& edge) {
    return std::find(aliases.begin(), aliases.end(), edge.target) != aliases.end();
  });
}

[[nodiscard]] std::size_t replay_connected_input_count(const ExecutionPlanState& plan,
                                                       VertexId owner) {
  return static_cast<std::size_t>(
      std::count_if(plan.graph.ports().begin(), plan.graph.ports().end(), [&](const auto& port) {
        return port.owner == owner && port.direction == PortDirection::kInput &&
               replay_input_connected(plan, port.id);
      }));
}

[[nodiscard]] PortId replay_input_id(const ExecutionPlanState& plan, VertexId owner,
                                     std::string_view name) noexcept {
  const PortRecord* const port = planned_graph_port(plan, owner, PortDirection::kInput, name);
  return port == nullptr ? PortId{} : PortId{port->id.value};
}

[[nodiscard]] bool input_backed_trigger_well_formed(
    const ExecutionPlanState& plan, const TemporalTriggerPlanRecord& trigger) noexcept {
  if (!trigger.readiness_source || !trigger.reference_clock.valid() ||
      trigger.owner.plan_generation != plan.plan_generation || trigger.owner.run_generation != 0U) {
    return false;
  }

  switch (trigger.kind) {
    case TemporalRecordTriggerKind::kAnyInput:
      return !trigger.trigger_input && trigger.bounded_state_capacity != 0U;
    case TemporalRecordTriggerKind::kEach:
      return trigger.trigger_input && trigger.bounded_state_capacity != 0U;
    case TemporalRecordTriggerKind::kAligned: {
      const auto found = std::find_if(plan.aligned_triggers.begin(),
                                      plan.aligned_triggers.end(),
                                      [&](const auto& row) { return row.owner == trigger.owner; });
      return found != plan.aligned_triggers.end() && !trigger.trigger_input &&
             !found->members.empty() && found->comparison_clock == trigger.reference_clock;
    }
    case TemporalRecordTriggerKind::kWindow: {
      const auto found = std::find_if(plan.window_triggers.begin(),
                                      plan.window_triggers.end(),
                                      [&](const auto& row) { return row.owner == trigger.owner; });
      return found != plan.window_triggers.end() && trigger.trigger_input &&
             found->anchor == trigger.trigger_input && found->wait_clock == trigger.trigger_clock;
    }
    case TemporalRecordTriggerKind::kPredicate: {
      const PredicateTriggerPlanRecord* const predicate =
          find_predicate_trigger_by_owner(plan, trigger.owner);
      return predicate != nullptr &&
             predicate->source == PredicateEvaluationSourceKind::kInputUpdate &&
             trigger.trigger_input && predicate->driver == trigger.trigger_input &&
             !predicate->clock.valid() && predicate->period_ns == 0 && !predicate->phase;
    }
    case TemporalRecordTriggerKind::kClock:
    case TemporalRecordTriggerKind::kReadable:
    case TemporalRecordTriggerKind::kNotified:
    case TemporalRecordTriggerKind::kDoorbell:
    case TemporalRecordTriggerKind::kSpin:
      return false;
  }
  return false;
}

[[nodiscard]] bool clock_trigger_well_formed(const ExecutionPlanState& plan,
                                             const TemporalTriggerPlanRecord& trigger) noexcept {
  const bool direct_clock = trigger.kind == TemporalRecordTriggerKind::kClock;
  const PredicateTriggerPlanRecord* const predicate =
      trigger.kind == TemporalRecordTriggerKind::kPredicate
          ? find_predicate_trigger_by_owner(plan, trigger.owner)
          : nullptr;
  const bool predicate_clock =
      predicate != nullptr && predicate->source == PredicateEvaluationSourceKind::kClock;
  if ((!direct_clock && !predicate_clock) || !trigger.readiness_source ||
      !trigger.trigger_clock.valid() || trigger.trigger_clock != trigger.reference_clock ||
      trigger.period_ns <= 0 || trigger.owner.plan_generation != plan.plan_generation ||
      trigger.owner.run_generation != 0U) {
    return false;
  }

  const ClockTriggerPlanRecord* const clock = find_clock_trigger_by_owner(plan, trigger.owner);
  if (clock == nullptr || clock->clock != trigger.trigger_clock ||
      clock->period_ns != trigger.period_ns ||
      clock->tick_policy != ClockTickPolicy::kCoalesceLateTicks ||
      clock->initial_tick_index != 0U || clock->initial_handled_index.has_value() ||
      clock->coalesced_range_capacity != 1U || !clock->global_start_barrier ||
      !clock->omitted_phase_index_zero_immediate || !clock->exact_now_eligible ||
      !clock->greatest_due_constant_time || !clock->one_candidate_per_observation ||
      !clock->skipped_indices_have_no_candidate || !clock->scheduled_tick_is_reference ||
      !clock->checked_terminal_no_wrap || !clock->stop_blocks_future_freeze) {
    return false;
  }
  if (clock->phase_origin == ClockPhaseOrigin::kCapturedAtSuccessfulStart) {
    if (clock->explicit_phase || trigger.phase) {
      return false;
    }
  } else if (clock->phase_origin != ClockPhaseOrigin::kExplicitAbsolute || !clock->explicit_phase ||
             !trigger.phase || *clock->explicit_phase != *trigger.phase) {
    return false;
  }
  return predicate == nullptr ||
         (predicate->clock == trigger.trigger_clock && predicate->period_ns == trigger.period_ns &&
          predicate->phase == trigger.phase && !predicate->driver);
}

[[nodiscard]] bool runtime_trigger_well_formed(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record) noexcept {
  const TemporalTriggerPlanRecord* const trigger =
      find_temporal_trigger_by_owner(plan, record.operator_id);
  if (trigger == nullptr || trigger->owner.id != record.operator_id ||
      trigger->owner.plan_generation != plan.plan_generation ||
      trigger->owner.run_generation != 0U) {
    return false;
  }
  if (trigger->kind == TemporalRecordTriggerKind::kClock ||
      trigger->kind == TemporalRecordTriggerKind::kPredicate) {
    const PredicateTriggerPlanRecord* const predicate =
        trigger->kind == TemporalRecordTriggerKind::kPredicate
            ? find_predicate_trigger_by_owner(plan, trigger->owner)
            : nullptr;
    if (trigger->kind == TemporalRecordTriggerKind::kClock ||
        (predicate != nullptr && predicate->source == PredicateEvaluationSourceKind::kClock)) {
      return clock_trigger_well_formed(plan, *trigger);
    }
    return input_backed_trigger_well_formed(plan, *trigger);
  }
  if (trigger->kind == TemporalRecordTriggerKind::kAnyInput ||
      trigger->kind == TemporalRecordTriggerKind::kEach ||
      trigger->kind == TemporalRecordTriggerKind::kAligned ||
      trigger->kind == TemporalRecordTriggerKind::kWindow) {
    return input_backed_trigger_well_formed(plan, *trigger);
  }
  const ReadinessPlanRecord* const source = find_readiness_by_owner(plan, record.vertex);
  return source != nullptr && source->source_id == trigger->readiness_source &&
         source->referenced && source->materialized;
}

[[nodiscard]] constexpr bool supported_profile(Profile profile) noexcept {
  switch (profile) {
    case Profile::kStandard:
    case Profile::kRobotics:
    case Profile::kRealTime:
    case Profile::kRegulated:
    case Profile::kResearch:
      return true;
  }
  return false;
}

[[nodiscard]] constexpr bool storage_matches(ParamStorageTraitsSummary left,
                                             ParamStorageTraitsSummary right) noexcept {
  return left.owns_dynamic_storage == right.owns_dynamic_storage &&
         left.fixed_object_extent == right.fixed_object_extent &&
         left.has_portable_codec == right.has_portable_codec &&
         left.supports_bounded_materialization == right.supports_bounded_materialization &&
         left.inline_bytes == right.inline_bytes;
}

[[nodiscard]] constexpr bool type_shape_matches(ParamFieldTypeShape left,
                                                ParamFieldTypeShape right) noexcept {
  return left.type == right.type && left.bit_width == right.bit_width &&
         left.element_or_rep_type == right.element_or_rep_type &&
         left.element_or_rep_bit_width == right.element_or_rep_bit_width &&
         left.duration_period_numerator == right.duration_period_numerator &&
         left.duration_period_denominator == right.duration_period_denominator &&
         left.optional == right.optional && left.has_stable_shape == right.has_stable_shape;
}

[[nodiscard]] bool parameter_schema_matches_plan(const ExecutionPlanState& plan,
                                                 const OperatorParameterPlanRecord& planned,
                                                 const ParamSchemaDescriptor* schema) noexcept {
  const auto planned_fields = std::count_if(
      plan.parameter_fields.begin(), plan.parameter_fields.end(), [&](const auto& field) {
        return field.owner.id == planned.owner.id;
      });
  if (schema == nullptr) {
    return planned.schema_availability == ParamSchemaAvailability::kOpaqueLocal &&
           planned_fields == 0;
  }
  if (planned.schema_availability != ParamSchemaAvailability::kDeclared ||
      planned.schema_id != schema->stable_id || planned.schema_version != schema->version ||
      planned.structural_hash != schema->structural_hash ||
      planned.contract_hash != schema->contract_hash ||
      planned.metadata_hash != schema->metadata_hash ||
      planned.schema_coverage != schema->coverage || planned.schema_proof != schema->proof ||
      std::cmp_not_equal(planned_fields, schema->fields.size())) {
    return false;
  }
  for (const ParamFieldDescriptor& descriptor : schema->fields) {
    const auto found = std::find_if(
        plan.parameter_fields.begin(), plan.parameter_fields.end(), [&](const auto& field) {
          return field.owner.id == planned.owner.id && field.field == descriptor.id;
        });
    if (found == plan.parameter_fields.end() || found->full_id != descriptor.full_id ||
        found->canonical_name != descriptor.canonical_name ||
        found->logical_type != descriptor.logical_type ||
        !type_shape_matches(found->type_shape, descriptor.type_shape) ||
        found->mutability != descriptor.mutability ||
        found->earliest_invalidated_stage != descriptor.earliest_invalidated_stage ||
        found->persistence != descriptor.persistence ||
        found->visibility != descriptor.visibility ||
        found->external_access != descriptor.external_access ||
        found->sensitivity != descriptor.sensitivity ||
        !storage_matches(found->storage, descriptor.storage)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool parameter_contract_matches_plan(const ExecutionPlanState& plan,
                                                   const ExecutableOperatorFactoryRecord& record,
                                                   const OperatorBase& op) noexcept {
  const OperatorParameterPlanRecord* const planned = find_operator_parameters_by_owner(
      plan, OperatorRef{record.operator_id, plan.plan_generation, 0U});
  if (planned == nullptr || planned->initial_generation != 0U ||
      planned->no_params != OperatorRuntimeAccess::has_no_params(op)) {
    return false;
  }
  const ParamTypeDescriptor& type = OperatorRuntimeAccess::parameter_type(op);
  if (planned->object_size != type.object_size ||
      planned->object_alignment != type.object_alignment ||
      !storage_matches(planned->storage, type.storage) || type.local_type_token == nullptr ||
      type.copy_construct == nullptr || type.destroy == nullptr ||
      type.validate_aggregate == nullptr) {
    return false;
  }
  return parameter_schema_matches_plan(plan, *planned, OperatorRuntimeAccess::parameter_schema(op));
}

[[nodiscard]] bool supported_external_payload(const PayloadIdentityRecord& identity) noexcept {
  switch (identity.kind) {
    case PayloadIdentityKind::kScalar:
      return identity.payload_kind == PayloadKind::kInlineTyped && identity.scalar &&
             identity.scalar->kind != ScalarIdentityKind::kLocalEnum && !identity.pod &&
             !identity.schema;
    case PayloadIdentityKind::kPod:
      return identity.payload_kind == PayloadKind::kInlineTyped && !identity.scalar &&
             identity.pod && !identity.schema;
    case PayloadIdentityKind::kSchema:
      return (identity.payload_kind == PayloadKind::kSerializedBytes ||
              identity.payload_kind == PayloadKind::kDescriptorBacked) &&
             !identity.scalar && !identity.pod && identity.schema;
    case PayloadIdentityKind::kTensor:
      return identity.payload_kind == PayloadKind::kDescriptorBacked && !identity.scalar &&
             !identity.pod && !identity.schema;
  }
  return false;
}

[[nodiscard]] bool same_supported_external_payload(const PayloadIdentityRecord& left,
                                                   const PayloadIdentityRecord& right) noexcept {
  return supported_external_payload(left) && supported_external_payload(right) &&
         payload_identity_exactly_equal(left, right);
}

[[nodiscard]] bool catalog_family_projection_matches(
    const ChannelEndpointRecord& endpoint, const PayloadIdentityRecord& identity,
    const std::optional<TensorBoundaryContractRecord>& topic_tensor,
    const std::span<const DescriptorTensorTopicFieldRecord> topic_tensor_fields,
    const std::span<const DescriptorTensorTopicFieldRecord> service_tensor_fields,
    const InterfaceTensorEmitPolicy service_tensor_emit_policy) noexcept {
  const bool pod = identity.kind == PayloadIdentityKind::kPod;
  const bool schema = identity.kind == PayloadIdentityKind::kSchema;
  const bool descriptor_schema = schema && identity.payload_kind == PayloadKind::kDescriptorBacked;
  const bool tensor = identity.kind == PayloadIdentityKind::kTensor;
  if (endpoint.pod_identity.has_value() != pod ||
      endpoint.service_static.pod_identity.has_value() != pod ||
      endpoint.schema_identity.has_value() != schema ||
      endpoint.service_static.schema_identity.has_value() != schema ||
      endpoint.tensor.has_value() != tensor ||
      endpoint.service_static.tensor.has_value() != tensor || topic_tensor.has_value() != tensor ||
      !descriptor_tensor_fields_equal(endpoint.tensor_fields, topic_tensor_fields) ||
      !descriptor_tensor_fields_equal(endpoint.service_static.tensor_fields, topic_tensor_fields) ||
      !descriptor_tensor_fields_equal(service_tensor_fields, topic_tensor_fields) ||
      descriptor_schema != !topic_tensor_fields.empty() ||
      endpoint.schema_writer_max_serialized_bytes !=
          endpoint.service_static.schema_writer_max_serialized_bytes ||
      (schema != (endpoint.schema_writer_max_serialized_bytes != 0U))) {
    return false;
  }
  if (pod) {
    if (!identity.pod || !endpoint.pod_identity || !endpoint.service_static.pod_identity) {
      return false;
    }
    const auto& projected = endpoint.pod_identity.value();
    const auto& static_projected = endpoint.service_static.pod_identity.value();
    const auto& identity_pod = identity.pod.value();
    return pod_identity_exactly_equal(projected.identity, identity_pod) &&
           pod_identity_exactly_equal(static_projected.identity, identity_pod) &&
           projected.digest_bytes == static_projected.digest_bytes &&
           projected.key == static_projected.key &&
           projected.layout_digest_bytes == static_projected.layout_digest_bytes;
  }
  if (schema) {
    if (!identity.schema || !endpoint.schema_identity || !endpoint.service_static.schema_identity) {
      return false;
    }
    return schema_identity_exactly_equal(endpoint.schema_identity.value(),
                                         identity.schema.value()) &&
           schema_identity_exactly_equal(endpoint.service_static.schema_identity.value(),
                                         identity.schema.value());
  }
  if (tensor) {
    return logical_tensor_requirements_exactly_equal(*endpoint.tensor,
                                                     *endpoint.service_static.tensor) &&
           logical_tensor_requirements_exactly_equal(*endpoint.tensor, *topic_tensor) &&
           endpoint.tensor->foreign_emit_policy == service_tensor_emit_policy;
  }
  return true;
}

[[nodiscard]] bool external_bindings_supported(
    const ExecutionPlanState& plan, const ChannelEndpointCatalogState& catalog) noexcept {
  const auto topics = plan.graph.external_topics();
  if (catalog.endpoints.size() != topics.size()) {
    return false;
  }
  return std::all_of(catalog.endpoints.begin(), catalog.endpoints.end(), [&](const auto& endpoint) {
    const auto topic = std::find_if(topics.begin(), topics.end(), [&](const auto& candidate) {
      return candidate.graph_port.value == endpoint.boundary_port.value();
    });
    if (topic == topics.end()) {
      return false;
    }
    const auto* service = find_external_channel_service_by_topic(plan, topic->id);
    if (service == nullptr) {
      return false;
    }
    const PayloadIdentityRecord& static_payload = endpoint.service_static.payload_identity;
    const bool tensor = endpoint.payload_identity.kind == PayloadIdentityKind::kTensor;
    const bool descriptor_schema =
        endpoint.payload_identity.kind == PayloadIdentityKind::kSchema &&
        endpoint.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
    return endpoint.direction ==
               (topic->imported ? TopicDirection::kImport : TopicDirection::kExport) &&
           endpoint.service == service->service_id &&
           service->tensor_payload == (tensor || descriptor_schema) &&
           catalog_family_projection_matches(endpoint,
                                             endpoint.payload_identity,
                                             topic->tensor,
                                             topic->tensor_fields,
                                             service->tensor_fields,
                                             service->tensor_emit_policy) &&
           same_supported_external_payload(endpoint.payload_identity, static_payload) &&
           same_supported_external_payload(endpoint.payload_identity, topic->payload_identity) &&
           same_supported_external_payload(endpoint.payload_identity, service->payload_identity) &&
           endpoint.schema_writer_max_serialized_bytes == topic->serialized_payload_capacity_bytes;
  });
}

// Authenticate all plan-wide inventories before any operator replay consumes their relationships.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> validate_foundation_plan_impl(const ExecutionPlanState& plan,
                                                                  std::size_t operator_count) {
  const ProfileManifest& profile = plan.profile_manifest;
  // Exact package/schema/policy/ABI/target/capability compatibility was already
  // established by match_runtime_package_capabilities(). Runtime consumes the
  // compiler's accepted profile decision here; it must not select or re-admit a
  // different profile after compilation.
  if (profile.disposition != ProfileAdmissionDisposition::kAccepted ||
      !profile.requested_profile.has_value() || !profile.effective_graph_profile.has_value() ||
      !supported_profile(*profile.requested_profile) ||
      !supported_profile(*profile.effective_graph_profile)) {
    return make_unexpected(operator_reconstruction_error(ErrorCode::kNotSupported,
                                                         "RUNTIME_PROFILE_COMPATIBILITY_FAILED"));
  }
  // Maximum start-delay rows form a closed inventory over trigger rows. Validate that
  // relation before topology-dependent placement so an orphan record in an
  // operator-free plan cannot be hidden by the empty placement disposition.
  if (auto max_start_delays = validate_runtime_max_start_delay_inventory(plan.max_start_delays,
                                                                         plan.temporal_triggers,
                                                                         plan.predicate_triggers,
                                                                         plan.clock_conversions,
                                                                         plan.plan_generation);
      !max_start_delays) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
  if (!has_structural_runtime_topology(plan) &&
      !plan.profile_manifest.materialization_obligations.empty()) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "PROFILE_MATERIALIZATION_FAILED"));
  }
  if (!partition_records_supported(plan)) {
    return make_unexpected(operator_reconstruction_error(ErrorCode::kNotSupported,
                                                         "RUNTIME_PLACEMENT_BINDING_UNAVAILABLE"));
  }
  if (!valid_closed_island_activation_plan(plan)) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
  if (!valid_inline_partition_execution_plan(plan)) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
  const auto* endpoint_catalog = ChannelEndpointCatalogAccess::state(plan.endpoint_catalog);
  const bool external_topics_present = !plan.graph.external_topics().empty();
  const bool catalog_endpoints_present =
      endpoint_catalog != nullptr && !endpoint_catalog->endpoints.empty();
  if ((external_topics_present && !catalog_endpoints_present) ||
      (catalog_endpoints_present && !external_bindings_supported(plan, *endpoint_catalog))) {
    return make_unexpected(operator_reconstruction_error(ErrorCode::kNotSupported,
                                                         "RUNTIME_EXTERNAL_BINDING_UNAVAILABLE"));
  }
  if (operator_count != 0U && (plan.clocks.empty() || !plan.root_default_clock.valid())) {
    return make_unexpected(operator_reconstruction_error(ErrorCode::kNotSupported,
                                                         "RUNTIME_CLOCK_BINDING_UNAVAILABLE"));
  }
  if (operator_count != 0U) {
    const bool needs_physical_factory =
        std::any_of(plan.readiness.begin(), plan.readiness.end(), [](const auto& row) {
          return row.referenced && row.kind != ReadinessSourceKind::kNotification;
        });
    // Source-tree Runtime tests may opt into the finite direct-witness backend through the
    // nonserialized executable inventory. Production plans still require an explicitly bound
    // physical factory, and ambiguous dual-factory test state is rejected before reconstruction.
    const bool has_test_direct_factory =
        plan.executable_factories.test_direct_activation_count != 0U;
    if ((needs_physical_factory && plan.executable_factories.port_binding_factory == nullptr &&
         !has_test_direct_factory) ||
        (has_test_direct_factory && plan.executable_factories.port_binding_factory != nullptr)) {
      return make_unexpected(operator_reconstruction_error(
          ErrorCode::kNotSupported, "RUNTIME_READINESS_BINDING_UNAVAILABLE"));
    }
    const bool triggers_valid =
        std::all_of(plan.executable_factories.operators.begin(),
                    plan.executable_factories.operators.end(),
                    [&](const auto& record) { return runtime_trigger_well_formed(plan, record); });
    if (plan.temporal_triggers.size() != operator_count || !triggers_valid) {
      return make_unexpected(
          operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
    }
  } else if (!plan.readiness.empty() || !plan.temporal_triggers.empty() ||
             !plan.clock_triggers.empty() || !plan.predicate_triggers.empty()) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
  return {};
}

[[nodiscard]] bool replayed_source_matches(const ExecutionPlanState& plan,
                                           const ExecutableOperatorFactoryRecord& record,
                                           const OperatorSpec& replay) noexcept {
  const auto replay_source = OperatorSpecCompilerAccess::source(replay, 0U);
  const ReadinessPlanRecord* const source = find_readiness_by_owner(plan, record.vertex);
  return source != nullptr &&
         (source->kind == ReadinessSourceKind::kReadable ||
          source->kind == ReadinessSourceKind::kNotification ||
          source->kind == ReadinessSourceKind::kDoorbell ||
          source->kind == ReadinessSourceKind::kSpin) &&
         replay_source.kind == source->kind && replay_source.name == source->name &&
         replay_source.capacity == source->requested_capacity &&
         replay_source.sender_reference_capacity == source->requested_sender_reference_capacity &&
         replay_source.clock == source->requested_clock && source->referenced &&
         source->materialized;
}

[[nodiscard]] bool replay_clock_key_matches(const ExecutionPlanState& plan, VertexId owner,
                                            ClockId id, const ClockKey& key) noexcept {
  if (!id.valid() || id.value > plan.clocks.size()) {
    return false;
  }
  // The omitted/default ClockKey is scoped to the Operator's owning Graph and
  // may resolve through an explicit embedded-clock import binding.
  const auto vertex = std::find_if(plan.graph.vertices().begin(),
                                   plan.graph.vertices().end(),
                                   [owner](const auto& row) { return row.id == owner; });
  if (vertex == plan.graph.vertices().end()) {
    return false;
  }
  return resolve_runtime_clock_id(plan, vertex->graph, key) == id;
}

[[nodiscard]] bool replayed_max_start_delay_dependency_matches(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record,
    const OperatorBase& op, const TemporalContractSnapshot& replay,
    const TemporalTriggerPlanRecord& trigger) {
  if (!replay.max_start_delay_dependency || replay.max_start_delay || !trigger.max_start_delay_ns ||
      !trigger.violation || replay.violation != trigger.violation) {
    return false;
  }
  const auto terms = parameter_derived_terms_by_owner(
      plan, OperatorRef{record.operator_id, plan.plan_generation, 0U});
  const auto first = std::find_if(terms.begin(), terms.end(), [](const auto& term) {
    return term.kind == ParameterDerivedTermKind::kContractMaxStartDelay;
  });
  if (first == terms.end() || std::find_if(std::next(first), terms.end(), [](const auto& term) {
                                return term.kind ==
                                       ParameterDerivedTermKind::kContractMaxStartDelay;
                              }) != terms.end()) {
    return false;
  }

  const ParamSchemaDescriptor* const schema = replay.max_start_delay_dependency.root_schema();
  const ParamFieldDescriptor* const field = replay.max_start_delay_dependency.final_field();
  std::int64_t initial_value_ns{};
  if (schema == nullptr || field == nullptr ||
      replay.max_start_delay_dependency.root_type_token !=
          OperatorRuntimeAccess::parameter_type(op).local_type_token ||
      !replay.max_start_delay_dependency.read_initial_ns(
          OperatorRuntimeAccess::parameter_address(op), initial_value_ns)) {
    return false;
  }
  return first->owner.id == record.operator_id &&
         first->owner.plan_generation == plan.plan_generation &&
         first->owner.run_generation == 0U && !first->input && first->field == field->id &&
         first->field_full_id == field->full_id &&
         first->field_path == replay.max_start_delay_dependency.canonical_path() &&
         first->schema_id == schema->stable_id && first->schema_version == schema->version &&
         first->initial_value_ns == initial_value_ns &&
         *trigger.max_start_delay_ns == initial_value_ns;
}

[[nodiscard]] bool replayed_clock_contract_matches(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record,
    const TemporalContractSnapshot& replay) noexcept {
  const TemporalTriggerPlanRecord* const trigger =
      find_temporal_trigger_by_owner(plan, record.operator_id);
  const ClockTriggerPlanRecord* const clock =
      trigger == nullptr ? nullptr : find_clock_trigger_by_owner(plan, trigger->owner);
  return trigger != nullptr && clock != nullptr &&
         replay.trigger.kind == TemporalTriggerKind::kClock &&
         replay.trigger.duration.count() == trigger->period_ns &&
         replay.trigger.tick_policy == clock->tick_policy &&
         replay.trigger.start_at == trigger->phase &&
         replay_clock_key_matches(
             plan, record.vertex, trigger->trigger_clock, replay.trigger.clock) &&
         !replay.synthesized_default;
}

[[nodiscard]] bool replayed_clock_predicate_contract_matches(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record,
    const TemporalContractSnapshot& replay) noexcept {
  const TemporalTriggerPlanRecord* const trigger =
      find_temporal_trigger_by_owner(plan, record.operator_id);
  const PredicateTriggerPlanRecord* const predicate =
      trigger == nullptr ? nullptr : find_predicate_trigger_by_owner(plan, trigger->owner);
  const PredicateCallablePlanRecord* const callable =
      predicate == nullptr ? nullptr : find_predicate_callable_by_id(plan, predicate->id);
  return trigger != nullptr && predicate != nullptr && callable != nullptr &&
         replay.trigger.kind == TemporalTriggerKind::kPredicate &&
         replay.trigger.predicate_clock_source && !replay.trigger.predicate_expects_driver &&
         replay.trigger.predicate && replay.trigger.predicate.name() == predicate->identity &&
         replay.trigger.predicate.callable_kind() == callable->kind &&
         replay.trigger.predicate_firing == predicate->firing &&
         replay.trigger.duration.count() == predicate->period_ns &&
         replay.trigger.start_at == predicate->phase &&
         replay_clock_key_matches(plan, record.vertex, predicate->clock, replay.trigger.clock) &&
         !replay.synthesized_default;
}

[[nodiscard]] TemporalRecordTriggerKind replay_trigger_kind(TemporalTriggerKind kind) noexcept {
  switch (kind) {
    case TemporalTriggerKind::kAnyInput:
      return TemporalRecordTriggerKind::kAnyInput;
    case TemporalTriggerKind::kEach:
      return TemporalRecordTriggerKind::kEach;
    case TemporalTriggerKind::kClock:
      return TemporalRecordTriggerKind::kClock;
    case TemporalTriggerKind::kReadable:
      return TemporalRecordTriggerKind::kReadable;
    case TemporalTriggerKind::kNotified:
      return TemporalRecordTriggerKind::kNotified;
    case TemporalTriggerKind::kDoorbell:
      return TemporalRecordTriggerKind::kDoorbell;
    case TemporalTriggerKind::kSpin:
      return TemporalRecordTriggerKind::kSpin;
    case TemporalTriggerKind::kAligned:
      return TemporalRecordTriggerKind::kAligned;
    case TemporalTriggerKind::kWindow:
      return TemporalRecordTriggerKind::kWindow;
    case TemporalTriggerKind::kPredicate:
      return TemporalRecordTriggerKind::kPredicate;
    case TemporalTriggerKind::kNone:
      break;
  }
  return TemporalRecordTriggerKind::kAnyInput;
}

[[nodiscard]] InputSelectorKind replay_selector_kind(TemporalSelectorKind kind) noexcept {
  switch (kind) {
    case TemporalSelectorKind::kExact:
      return InputSelectorKind::kExact;
    case TemporalSelectorKind::kLatestWithin:
      return InputSelectorKind::kLatestWithin;
    case TemporalSelectorKind::kDrainAll:
      return InputSelectorKind::kDrainAll;
    case TemporalSelectorKind::kBracketedWindow:
      return InputSelectorKind::kBracketedWindow;
  }
  return InputSelectorKind::kExact;
}

[[nodiscard]] TemporalRecordMissingKind replay_missing_kind(TemporalMissingKind kind) noexcept {
  switch (kind) {
    case TemporalMissingKind::kUnspecified:
      return TemporalRecordMissingKind::kUnspecified;
    case TemporalMissingKind::kSkip:
      return TemporalRecordMissingKind::kSkip;
    case TemporalMissingKind::kSafeDefault:
      return TemporalRecordMissingKind::kSafeDefault;
    case TemporalMissingKind::kReuseLast:
      return TemporalRecordMissingKind::kReuseLast;
  }
  return TemporalRecordMissingKind::kUnspecified;
}

[[nodiscard]] bool input_selection_owned_by(const ExecutionPlanState& plan,
                                            const InputSelectionPlanRecord& selection,
                                            VertexId owner) noexcept {
  const auto* found = find_port_by_id(plan, selection.input);
  return found != nullptr && found->owner == owner && found->direction == PortDirection::kInput;
}

[[nodiscard]] bool replayed_latest_within_dependency_matches(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record,
    const OperatorBase& op, const TemporalInputRecord& replay, PortId input,
    const InputSelectionPlanRecord& selection) {
  if (!replay.max_age_dependency) {
    return replay.max_age.count() == selection.max_age_ns;
  }
  if (replay.selector != TemporalSelectorKind::kLatestWithin || replay.max_age.count() != 0) {
    return false;
  }

  const auto terms = parameter_derived_terms_by_owner(
      plan, OperatorRef{record.operator_id, plan.plan_generation, 0U});
  const auto first = std::find_if(terms.begin(), terms.end(), [&](const auto& term) {
    return term.kind == ParameterDerivedTermKind::kLatestWithinMaxAge && term.input == input;
  });
  if (first == terms.end() || std::find_if(std::next(first), terms.end(), [&](const auto& term) {
                                return term.kind == ParameterDerivedTermKind::kLatestWithinMaxAge &&
                                       term.input == input;
                              }) != terms.end()) {
    return false;
  }

  const ParamSchemaDescriptor* const schema = replay.max_age_dependency.root_schema();
  const ParamFieldDescriptor* const field = replay.max_age_dependency.final_field();
  std::int64_t initial_value_ns{};
  if (schema == nullptr || field == nullptr ||
      schema != OperatorRuntimeAccess::parameter_schema(op) ||
      replay.max_age_dependency.root_type_token !=
          OperatorRuntimeAccess::parameter_type(op).local_type_token ||
      !replay.max_age_dependency.read_initial_ns(OperatorRuntimeAccess::parameter_address(op),
                                                 initial_value_ns)) {
    return false;
  }
  return first->owner.id == record.operator_id &&
         first->owner.plan_generation == plan.plan_generation &&
         first->owner.run_generation == 0U && first->input == input && first->field == field->id &&
         first->field_full_id == field->full_id &&
         first->field_path == replay.max_age_dependency.canonical_path() &&
         first->schema_id == schema->stable_id && first->schema_version == schema->version &&
         first->initial_value_ns == initial_value_ns && selection.max_age_ns == initial_value_ns;
}

[[nodiscard]] bool replayed_input_selection_matches(const ExecutionPlanState& plan,
                                                    const ExecutableOperatorFactoryRecord& record,
                                                    const OperatorBase& op,
                                                    const TemporalInputRecord& replay,
                                                    PortId* resolved_input = nullptr) {
  const VertexId owner = record.vertex;
  const PortId input = replay_input_id(plan, owner, replay.port_name);
  const InputSelectionPlanRecord* const selection = find_input_selection_by_input(plan, input);
  const bool bracketed = replay.selector == TemporalSelectorKind::kBracketedWindow;
  if (!input || selection == nullptr ||
      replay_selector_kind(replay.selector) != selection->selector ||
      replay_missing_kind(replay.missing) != selection->missing ||
      !replayed_latest_within_dependency_matches(plan, record, op, replay, input, *selection) ||
      replay.reuse_max_age.count() != selection->reuse_max_age_ns ||
      (bracketed && (replay.bracket.lookbehind != selection->lookbehind ||
                     replay.bracket.lookahead != selection->lookahead)) ||
      (!bracketed && (selection->lookbehind != 0U || selection->lookahead != 0U)) ||
      selection->cache_capacity == 0U || selection->max_selected_samples == 0U ||
      !selection->reference_clock.valid()) {
    return false;
  }

  const bool batch = replay.selector == TemporalSelectorKind::kDrainAll || bracketed;
  if ((batch && selection->cardinality != InputSelectionCardinality::kSampleBatch) ||
      (!batch && selection->cardinality != InputSelectionCardinality::kSingleSample) ||
      (replay.selector == TemporalSelectorKind::kDrainAll &&
       selection->batch_selection_mode != BatchSelectionMode::kConsumeSelected) ||
      (replay.selector == TemporalSelectorKind::kBracketedWindow &&
       (selection->batch_selection_mode != BatchSelectionMode::kNonDestructiveSnapshot ||
        selection->max_selected_samples != replay.bracket.max_samples)) ||
      (!batch && (selection->batch_selection_mode != BatchSelectionMode::kNotApplicable ||
                  selection->max_selected_samples != 1U))) {
    return false;
  }
  if (resolved_input != nullptr) {
    *resolved_input = input;
  }
  return true;
}

[[nodiscard]] const MissingAggregatePlanRecord* planned_missing_aggregate(
    const ExecutionPlanState& plan, OperatorRef owner, MissingSelectionPhase phase) noexcept {
  const auto found = std::find_if(
      plan.missing_aggregates.begin(), plan.missing_aggregates.end(), [&](const auto& row) {
        return row.owner == owner && row.phase == phase;
      });
  return found == plan.missing_aggregates.end() ? nullptr : &*found;
}

[[nodiscard]] bool replayed_missing_aggregate_matches(const ExecutionPlanState& plan,
                                                      OperatorRef owner,
                                                      MissingSelectionPhase phase,
                                                      std::span<const PortId> inputs) noexcept {
  const MissingAggregatePlanRecord* const aggregate = planned_missing_aggregate(plan, owner, phase);
  if (aggregate == nullptr || aggregate->inputs.size() != inputs.size() ||
      !aggregate->evaluate_complete_set || !aggregate->safe_default_dominates ||
      !aggregate->absent_without_safe_default_no_fires ||
      !aggregate->degraded_commits_destructive_selection ||
      !aggregate->no_fire_preserves_provisional_selection ||
      !aggregate->bracketed_window_all_or_nothing) {
    return false;
  }
  for (std::size_t index = 0U; index < inputs.size(); ++index) {
    const InputSelectionPlanRecord* const selection =
        find_input_selection_by_input(plan, inputs[index]);
    const auto& item = aggregate->inputs[index];
    if (selection == nullptr || item.input != inputs[index] || item.ordinal != index ||
        item.selector != selection->selector || item.missing != selection->missing) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] MaxStartDelaySourceCoordinateKind replay_max_start_delay_source(
    TemporalRecordTriggerKind kind, bool predicate_clock_source) noexcept {
  switch (kind) {
    case TemporalRecordTriggerKind::kAnyInput:
    case TemporalRecordTriggerKind::kEach:
      return MaxStartDelaySourceCoordinateKind::kTriggerSample;
    case TemporalRecordTriggerKind::kClock:
      return MaxStartDelaySourceCoordinateKind::kScheduledTick;
    case TemporalRecordTriggerKind::kReadable:
    case TemporalRecordTriggerKind::kNotified:
    case TemporalRecordTriggerKind::kDoorbell:
    case TemporalRecordTriggerKind::kSpin:
      return MaxStartDelaySourceCoordinateKind::kReadinessWitness;
    case TemporalRecordTriggerKind::kAligned:
      return MaxStartDelaySourceCoordinateKind::kAlignedHighOrClosureProof;
    case TemporalRecordTriggerKind::kWindow:
      return MaxStartDelaySourceCoordinateKind::kWindowClose;
    case TemporalRecordTriggerKind::kPredicate:
      return predicate_clock_source ? MaxStartDelaySourceCoordinateKind::kScheduledTick
                                    : MaxStartDelaySourceCoordinateKind::kPredicateDriver;
  }
  return MaxStartDelaySourceCoordinateKind::kTriggerSample;
}

[[nodiscard]] bool replayed_max_start_delay_matches(
    const ExecutionPlanState& plan, const TemporalTriggerPlanRecord& trigger,
    const TemporalContractSnapshot& replay) noexcept {
  const auto first = std::find_if(plan.max_start_delays.begin(),
                                  plan.max_start_delays.end(),
                                  [&](const auto& row) { return row.owner == trigger.owner; });
  const std::size_t count = static_cast<std::size_t>(std::count_if(
      plan.max_start_delays.begin(), plan.max_start_delays.end(), [&](const auto& row) {
        return row.owner == trigger.owner;
      }));
  const bool present = replay.max_start_delay.has_value() && replay.violation.has_value();
  if (!present) {
    return !replay.max_start_delay && !replay.violation && !trigger.max_start_delay_ns &&
           !trigger.violation && count == 0U;
  }
  const bool predicate_clock = trigger.kind == TemporalRecordTriggerKind::kPredicate &&
                               replay.trigger.predicate_clock_source;
  return count == 1U && first != plan.max_start_delays.end() && trigger.max_start_delay_ns &&
         trigger.violation && *trigger.max_start_delay_ns == replay.max_start_delay->count() &&
         *trigger.violation == *replay.violation &&
         first->budget_ns == replay.max_start_delay->count() &&
         first->action == *replay.violation &&
         first->source_coordinate == replay_max_start_delay_source(trigger.kind, predicate_clock) &&
         first->source_clock.valid() && first->conversion_id == trigger.conversion_id &&
         first->callback_count_if_late == (*replay.violation == OnViolation::kFailRun ? 0U : 1U) &&
         first->checked_addition && first->exact_boundary_is_on_time &&
         first->final_invocation_gate && first->consumes_candidate_tag &&
         !first->automatic_output && !first->callback_state_rollback &&
         first->duration_measurement_separate;
}

// Predicate replay is intentionally a single fail-closed comparison across the callable,
// evaluation inputs, and both missing-policy phases. Keeping those coupled facts together avoids
// accidentally accepting a partially reconstructed predicate contract.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool replayed_predicate_contract_matches(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record,
    const TemporalTriggerPlanRecord& trigger, const TemporalContractSnapshot& replay,
    std::span<const PortId> declared_inputs) noexcept {
  const PredicateTriggerPlanRecord* const predicate =
      find_predicate_trigger_by_owner(plan, trigger.owner);
  const bool clock_driven = replay.trigger.predicate_clock_source;
  const PortId replay_driver =
      clock_driven ? PortId{} : replay_input_id(plan, record.vertex, replay.trigger.port_name);
  if (predicate == nullptr || !replay.trigger.predicate ||
      predicate->source != (clock_driven ? PredicateEvaluationSourceKind::kClock
                                         : PredicateEvaluationSourceKind::kInputUpdate) ||
      replay.trigger.predicate_expects_driver == clock_driven ||
      predicate->driver != replay_driver || predicate->driver != trigger.trigger_input ||
      (clock_driven
           ? (predicate->clock != trigger.trigger_clock ||
              predicate->period_ns != trigger.period_ns || predicate->phase != trigger.phase)
           : (predicate->clock.valid() || predicate->period_ns != 0 || predicate->phase)) ||
      predicate->identity != replay.trigger.predicate.name() ||
      predicate->semantic_version != replay.trigger.predicate_semantic_version ||
      predicate->firing != replay.trigger.predicate_firing || predicate->initial_truth ||
      predicate->state_size_bytes != replay.trigger.predicate_state_size ||
      predicate->state_alignment_bytes != replay.trigger.predicate_state_alignment ||
      predicate->has_config != (replay.trigger.predicate_config_size != 0U) ||
      predicate->config_size_bytes != replay.trigger.predicate_config_size ||
      predicate->config_alignment_bytes != replay.trigger.predicate_config_alignment ||
      predicate->config_schema !=
          (predicate->has_config
               ? std::optional<std::string>{replay.trigger.predicate_config_schema}
               : std::nullopt) ||
      predicate->config_digest !=
          (predicate->has_config
               ? std::optional<std::string>{replay.trigger.predicate_config_digest}
               : std::nullopt)) {
    return false;
  }

  const PredicateCallablePlanRecord* const callable =
      find_predicate_callable_by_id(plan, predicate->id);
  if (callable == nullptr || callable->kind != replay.trigger.predicate.callable_kind() ||
      callable->size_bytes != replay.trigger.predicate_callable_size ||
      callable->alignment_bytes != replay.trigger.predicate_callable_alignment ||
      callable->is_noexcept != replay.trigger.predicate_callable_noexcept ||
      callable->result_kind != (replay.trigger.predicate_callable_returns_expected
                                    ? PredicateResultKind::kExpectedBoolError
                                    : PredicateResultKind::kBool)) {
    return false;
  }

  std::vector<const PredicateInputPlanRecord*> predicate_inputs;
  for (const auto& input : plan.predicate_inputs) {
    if (input.predicate_id == predicate->id) {
      predicate_inputs.push_back(&input);
    }
  }
  const std::size_t implicit_driver_count = clock_driven ? 0U : 1U;
  if (replay.trigger.predicate.input_count() != replay.trigger.predicate_input_names.size() ||
      predicate_inputs.size() !=
          replay.trigger.predicate_input_names.size() + implicit_driver_count) {
    return false;
  }
  std::vector<PortId> evaluation_inputs;
  evaluation_inputs.reserve(predicate_inputs.size());
  for (std::size_t index = 0U; index < predicate_inputs.size(); ++index) {
    const bool implicit_driver = !clock_driven && index == 0U;
    const std::size_t replay_input_index = implicit_driver ? 0U : index - implicit_driver_count;
    const PortId expected =
        implicit_driver
            ? predicate->driver
            : replay_input_id(
                  plan, record.vertex, replay.trigger.predicate_input_names[replay_input_index]);
    const PredicateInputPlanRecord& input = *predicate_inputs[index];
    const InputSelectionPlanRecord* const selection = find_input_selection_by_input(plan, expected);
    if (!expected || selection == nullptr || input.input != expected ||
        input.argument_ordinal != index || input.implicit_driver != implicit_driver ||
        input.selector != selection->selector || input.missing != selection->missing ||
        input.reference_clock != selection->reference_clock ||
        input.conversion_id != selection->conversion_id ||
        input.cache_capacity != selection->cache_capacity ||
        input.max_age_ns != selection->max_age_ns ||
        input.reuse_max_age_ns != selection->reuse_max_age_ns) {
      return false;
    }
    evaluation_inputs.push_back(expected);
  }
  if (!replayed_missing_aggregate_matches(
          plan, trigger.owner, MissingSelectionPhase::kPredicateEvaluation, evaluation_inputs)) {
    return false;
  }

  std::vector<PortId> compute_only;
  for (const PortId input : declared_inputs) {
    if (std::find(evaluation_inputs.begin(), evaluation_inputs.end(), input) ==
        evaluation_inputs.end()) {
      compute_only.push_back(input);
    }
  }
  std::sort(compute_only.begin(), compute_only.end(), [](PortId left, PortId right) {
    return left.value() < right.value();
  });
  return replayed_missing_aggregate_matches(
      plan, trigger.owner, MissingSelectionPhase::kPredicateCompute, compute_only);
}

// Replay validation deliberately keeps the complete temporal contract comparison fail-closed in
// one routine; splitting its mutually dependent trigger, selector, and missing-policy checks would
// make it easier to validate only a partial reconstructed contract.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool replayed_temporal_contract_matches(const ExecutionPlanState& plan,
                                                      const ExecutableOperatorFactoryRecord& record,
                                                      OperatorBase& op,
                                                      const TemporalContractSnapshot& replay) {
  const TemporalTriggerPlanRecord* const trigger =
      find_temporal_trigger_by_owner(plan, record.operator_id);
  if (trigger == nullptr || replay.trigger.kind == TemporalTriggerKind::kNone ||
      replay_trigger_kind(replay.trigger.kind) != trigger->kind ||
      replay.connected_input_count != replay_connected_input_count(plan, record.vertex) ||
      replay.synthesized_default != (trigger->origin == TemporalRecordOrigin::kCoreSynthesized) ||
      replay.inputs.size() != trigger->declared_inputs.size()) {
    return false;
  }

  std::vector<PortId> declared_inputs;
  declared_inputs.reserve(replay.inputs.size());
  for (std::size_t index = 0U; index < replay.inputs.size(); ++index) {
    PortId input{};
    if (!replayed_input_selection_matches(plan, record, op, replay.inputs[index], &input) ||
        trigger->declared_inputs[index] != input) {
      return false;
    }
    declared_inputs.push_back(input);
  }
  const std::size_t planned_selection_count = static_cast<std::size_t>(std::count_if(
      plan.input_selections.begin(), plan.input_selections.end(), [&](const auto& selection) {
        return input_selection_owned_by(plan, selection, record.vertex);
      }));
  if (planned_selection_count != replay.inputs.size()) {
    return false;
  }

  PortId expected_trigger_input{};
  std::vector<PortId> expected_presence;
  if (replay.trigger.kind == TemporalTriggerKind::kEach ||
      replay.trigger.kind == TemporalTriggerKind::kWindow ||
      (replay.trigger.kind == TemporalTriggerKind::kPredicate &&
       !replay.trigger.predicate_clock_source)) {
    expected_trigger_input = replay_input_id(plan, record.vertex, replay.trigger.port_name);
    expected_presence.push_back(expected_trigger_input);
  } else if (replay.trigger.kind == TemporalTriggerKind::kAligned) {
    for (const auto& name : replay.trigger.ports) {
      expected_presence.push_back(replay_input_id(plan, record.vertex, name));
    }
  }
  const bool max_start_delay_matched =
      replayed_max_start_delay_matches(plan, *trigger, replay) ||
      replayed_max_start_delay_dependency_matches(plan, record, op, replay, *trigger);
  if (trigger->trigger_input != expected_trigger_input ||
      trigger->presence_guaranteed_inputs != expected_presence || !max_start_delay_matched) {
    return false;
  }

  if (replay.trigger.kind == TemporalTriggerKind::kAnyInput) {
    if (replay.connected_input_count == 0U ||
        trigger->bounded_state_capacity != replay.connected_input_count) {
      return false;
    }
  } else if (replay.trigger.kind == TemporalTriggerKind::kAligned) {
    const auto aligned = std::find_if(plan.aligned_triggers.begin(),
                                      plan.aligned_triggers.end(),
                                      [&](const auto& row) { return row.owner == trigger->owner; });
    if (aligned == plan.aligned_triggers.end() ||
        aligned->members.size() != replay.trigger.ports.size() ||
        aligned->inclusive_window_ns != replay.trigger.duration.count() ||
        !replay_clock_key_matches(
            plan, record.vertex, aligned->comparison_clock, replay.trigger.clock)) {
      return false;
    }
    for (std::size_t index = 0U; index < aligned->members.size(); ++index) {
      if (aligned->members[index].input != expected_presence[index] ||
          aligned->members[index].member_ordinal != index) {
        return false;
      }
    }
  } else if (replay.trigger.kind == TemporalTriggerKind::kWindow) {
    const auto window = std::find_if(plan.window_triggers.begin(),
                                     plan.window_triggers.end(),
                                     [&](const auto& row) { return row.owner == trigger->owner; });
    if (window == plan.window_triggers.end() || window->anchor != expected_trigger_input ||
        window->max_wait_ns != replay.trigger.duration.count() ||
        window->max_pending_windows != replay.trigger.capacity ||
        !replay_clock_key_matches(plan, record.vertex, window->wait_clock, replay.trigger.clock)) {
      return false;
    }
    const std::size_t replayed_bracket_count = static_cast<std::size_t>(
        std::count_if(replay.inputs.begin(), replay.inputs.end(), [](const auto& input) {
          return input.selector == TemporalSelectorKind::kBracketedWindow;
        }));
    const std::size_t planned_bracket_count = static_cast<std::size_t>(std::count_if(
        plan.bracketed_windows.begin(), plan.bracketed_windows.end(), [&](const auto& row) {
          return row.owner == trigger->owner;
        }));
    if (planned_bracket_count != replayed_bracket_count) {
      return false;
    }
  } else if (replay.trigger.kind == TemporalTriggerKind::kPredicate &&
             !replayed_predicate_contract_matches(
                 plan, record, *trigger, replay, declared_inputs)) {
    return false;
  }

  if (replay.trigger.kind != TemporalTriggerKind::kPredicate) {
    std::sort(declared_inputs.begin(), declared_inputs.end(), [](PortId left, PortId right) {
      return left.value() < right.value();
    });
    if (!replayed_missing_aggregate_matches(
            plan, trigger->owner, MissingSelectionPhase::kActivation, declared_inputs)) {
      return false;
    }
  }
  return true;
}

void capture_replay_error(RuntimeDiagnosticDetail* detail, VertexId vertex,
                          DiagnosticOperation operation, const Error& error) noexcept {
  if (detail == nullptr || detail->present) {
    return;
  }
  detail->present = true;
  detail->vertex = vertex;
  detail->operation = operation;
  detail->capture.emplace();
  detail->capture->append(error.code, error.message);
}

[[nodiscard]] expected<void, Error> bind_and_validate_direct_contract(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record, OperatorBase& op,
    const OperatorSpec& replay, RuntimeDiagnosticDetail* failure_detail) {
  const TemporalTriggerPlanRecord* const trigger =
      find_temporal_trigger_by_owner(plan, record.operator_id);
  const ReadinessPlanRecord* const source = find_readiness_by_owner(plan, record.vertex);
  if (trigger == nullptr || source == nullptr ||
      (source->kind != ReadinessSourceKind::kReadable &&
       source->kind != ReadinessSourceKind::kNotification &&
       source->kind != ReadinessSourceKind::kDoorbell &&
       source->kind != ReadinessSourceKind::kSpin) ||
      !source->source_id || !source->referenced || !source->materialized) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }

  const auto max_start_delay = std::find_if(
      plan.max_start_delays.begin(), plan.max_start_delays.end(), [&](const auto& row) {
        return row.owner == trigger->owner;
      });
  const bool has_max_start_delay =
      trigger->max_start_delay_ns.has_value() && trigger->violation.has_value();
  const std::size_t max_start_delay_count = static_cast<std::size_t>(std::count_if(
      plan.max_start_delays.begin(), plan.max_start_delays.end(), [&](const auto& row) {
        return row.owner == trigger->owner;
      }));
  const bool max_start_delay_plan_matches =
      has_max_start_delay
          ? max_start_delay_count == 1U && max_start_delay != plan.max_start_delays.end() &&
                max_start_delay->budget_ns == *trigger->max_start_delay_ns &&
                max_start_delay->action == *trigger->violation &&
                max_start_delay->source_coordinate ==
                    MaxStartDelaySourceCoordinateKind::kReadinessWitness &&
                max_start_delay->source_clock == trigger->reference_clock &&
                max_start_delay->conversion_id == trigger->conversion_id &&
                max_start_delay->callback_count_if_late ==
                    (*trigger->violation == OnViolation::kFailRun ? 0U : 1U) &&
                max_start_delay->checked_addition && max_start_delay->exact_boundary_is_on_time &&
                max_start_delay->final_invocation_gate && max_start_delay->consumes_candidate_tag &&
                !max_start_delay->automatic_output && !max_start_delay->callback_state_rollback &&
                max_start_delay->duration_measurement_separate
          : max_start_delay_count == 0U && !trigger->max_start_delay_ns && !trigger->violation;
  if (!max_start_delay_plan_matches) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }

  const auto replay_source = OperatorSpecCompilerAccess::source(replay, 0U);
  if (replay_source.kind != source->kind || replay_source.declaration == nullptr) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
  switch (source->kind) {
    case ReadinessSourceKind::kReadable: {
      auto& readable = *static_cast<ReadableSource*>(const_cast<void*>(replay_source.declaration));
      ReadinessSourceAccess::bind(readable,
                                  ReadinessSourceAccess::make_ref<ReadinessSourceKind::kReadable>(
                                      source->source_id, plan.plan_generation));
      break;
    }
    case ReadinessSourceKind::kNotification: {
      auto& notification =
          *static_cast<NotificationSource*>(const_cast<void*>(replay_source.declaration));
      ReadinessSourceAccess::bind(
          notification,
          ReadinessSourceAccess::make_ref<ReadinessSourceKind::kNotification>(
              source->source_id, plan.plan_generation));
      break;
    }
    case ReadinessSourceKind::kDoorbell: {
      auto& doorbell = *static_cast<DoorbellSource*>(const_cast<void*>(replay_source.declaration));
      ReadinessSourceAccess::bind(doorbell,
                                  ReadinessSourceAccess::make_ref<ReadinessSourceKind::kDoorbell>(
                                      source->source_id, plan.plan_generation));
      break;
    }
    case ReadinessSourceKind::kSpin: {
      auto& spin = *static_cast<SpinSource*>(const_cast<void*>(replay_source.declaration));
      ReadinessSourceAccess::bind(spin,
                                  ReadinessSourceAccess::make_ref<ReadinessSourceKind::kSpin>(
                                      source->source_id, plan.plan_generation));
      break;
    }
  }

  const std::size_t connected_input_count = replay_connected_input_count(plan, record.vertex);
  auto normalized = ContractTestAccess::normalize(op.contract(), connected_input_count);
  if (!normalized) {
    capture_replay_error(
        failure_detail, record.vertex, DiagnosticOperation::kContract, normalized.error());
  }
  const auto normalized_max_start_delay =
      normalized ? normalized->max_start_delay.value_or(std::chrono::nanoseconds{})
                 : std::chrono::nanoseconds{};
  const std::int64_t planned_max_start_delay = trigger->max_start_delay_ns.value_or(0);
  if (!normalized || replay_trigger_kind(normalized->trigger.kind) != trigger->kind ||
      normalized->trigger.source_id != source->source_id ||
      normalized->connected_input_count != connected_input_count ||
      !replayed_completion_rule_matches(plan, record.operator_id, *normalized) ||
      !replayed_temporal_contract_matches(plan, record, op, *normalized) ||
      (normalized->max_start_delay.has_value() != trigger->max_start_delay_ns.has_value() &&
       !replayed_max_start_delay_dependency_matches(plan, record, op, *normalized, *trigger)) ||
      (normalized->max_start_delay &&
       normalized_max_start_delay.count() != planned_max_start_delay) ||
      normalized->violation != trigger->violation || normalized->synthesized_default) {
    return make_unexpected(
        operator_reconstruction_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
  return {};
}

// Replay checks the complete public declaration against the frozen executable inventory before
// any runtime callback becomes reachable.
// NOLINTBEGIN(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> validate_replayed_operator_impl(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record, OperatorBase& op,
    const OperatorSpec& replay, std::uint32_t factory_slot,
    RuntimeReplayMismatchDetail* replay_mismatch, RuntimeDiagnosticDetail* failure_detail) {
  const auto mismatch = [&](RuntimeReplayMismatchCategory category,
                            ErrorCode code = ErrorCode::kFailure,
                            const char* token = "RUNTIME_OPERATOR_REPLAY_MISMATCH",
                            const TemporalContractSnapshot* temporal =
                                nullptr) -> expected<void, Error> {
    return make_unexpected(replay_mismatch_error(
        code, token, plan, record, factory_slot, category, replay, replay_mismatch, temporal));
  };
  const auto& declaration_error = OperatorSpecCompilerAccess::declaration_error(replay);
  if (declaration_error.code != ErrorCode::kSuccess) {
    capture_replay_error(
        failure_detail, record.vertex, DiagnosticOperation::kSetup, declaration_error);
    return mismatch(RuntimeReplayMismatchCategory::kDeclaration);
  }
  if (OperatorSpecCompilerAccess::cuda_graph_capture_eligible(replay) !=
      record.cuda_graph_capture_eligible) {
    return mismatch(RuntimeReplayMismatchCategory::kDeclaration);
  }

  const std::size_t replay_ports = OperatorSpecCompilerAccess::port_count(replay);
  if (replay_ports != planned_port_count(plan, record.vertex)) {
    return mismatch(RuntimeReplayMismatchCategory::kPorts);
  }
  if (!replayed_ports_match(plan, record, replay)) {
    // The process-local function table is verified against the complete
    // pointer-free payload identity before any lifecycle callback.
    return mismatch(RuntimeReplayMismatchCategory::kPorts,
                    ErrorCode::kNotSupported,
                    "RUNTIME_PORT_BINDING_UNAVAILABLE");
  }

  const std::size_t replay_sources = OperatorSpecCompilerAccess::source_count(replay);
  const TemporalTriggerPlanRecord* const trigger =
      find_temporal_trigger_by_owner(plan, record.operator_id);
  if (trigger == nullptr) {
    return mismatch(RuntimeReplayMismatchCategory::kTemporalContract);
  }
  const PredicateTriggerPlanRecord* const predicate =
      trigger->kind == TemporalRecordTriggerKind::kPredicate
          ? find_predicate_trigger_by_owner(plan, trigger->owner)
          : nullptr;
  const bool clock_source =
      trigger->kind == TemporalRecordTriggerKind::kClock ||
      (predicate != nullptr && predicate->source == PredicateEvaluationSourceKind::kClock);
  const bool direct_source = trigger->kind == TemporalRecordTriggerKind::kReadable ||
                             trigger->kind == TemporalRecordTriggerKind::kNotified ||
                             trigger->kind == TemporalRecordTriggerKind::kDoorbell ||
                             trigger->kind == TemporalRecordTriggerKind::kSpin;
  const bool input_source =
      trigger->kind == TemporalRecordTriggerKind::kAnyInput ||
      trigger->kind == TemporalRecordTriggerKind::kEach ||
      trigger->kind == TemporalRecordTriggerKind::kAligned ||
      trigger->kind == TemporalRecordTriggerKind::kWindow ||
      (predicate != nullptr && predicate->source == PredicateEvaluationSourceKind::kInputUpdate);
  if (((clock_source || input_source) && replay_sources != 0U) ||
      (direct_source && (replay_sources != planned_source_count(plan, record.vertex) ||
                         replay_sources != 1U || !replayed_source_matches(plan, record, replay))) ||
      (!clock_source && !direct_source && !input_source)) {
    return mismatch(RuntimeReplayMismatchCategory::kReadiness);
  }
  std::array<std::optional<OperatorSpecCompilerAccess::LifecycleCallbackMetadata>, 8U>
      replayed_callbacks;
  for (std::size_t index = 0U; index < OperatorSpecCompilerAccess::lifecycle_callback_count(replay);
       ++index) {
    const auto callback = OperatorSpecCompilerAccess::lifecycle_callback(replay, index);
    const std::size_t stage = static_cast<std::uint8_t>(callback.stage);
    if (stage >= replayed_callbacks.size() || replayed_callbacks[stage].has_value()) {
      return mismatch(RuntimeReplayMismatchCategory::kLifecycle);
    }
    replayed_callbacks[stage] = callback;
  }
  const OperatorFactoryBase::LifecycleHookSet replayed_legacy_hooks =
      record.factory->lifecycle_hooks();
  for (std::size_t stage = 0U; stage < replayed_callbacks.size(); ++stage) {
    const LifecycleStage lifecycle_stage = static_cast<LifecycleStage>(stage);
    const auto checkpoint = std::find_if(plan.lifecycle_checkpoints.begin(),
                                         plan.lifecycle_checkpoints.end(),
                                         [&](const auto& item) {
                                           return item.operator_instance.id == record.operator_id &&
                                                  item.stage == lifecycle_stage;
                                         });
    if (checkpoint == plan.lifecycle_checkpoints.end()) {
      return mismatch(RuntimeReplayMismatchCategory::kLifecycle);
    }
    const auto expected = std::find_if(record.lifecycle_callbacks.begin(),
                                       record.lifecycle_callbacks.end(),
                                       [&](const ExecutableLifecycleCallbackRecord& callback) {
                                         return callback.stage == lifecycle_stage;
                                       });
    const bool expected_callback = expected != record.lifecycle_callbacks.end();
    if (checkpoint->callback_present != expected_callback ||
        (expected_callback && checkpoint->required != expected->required)) {
      return mismatch(RuntimeReplayMismatchCategory::kLifecycle);
    }
    const auto& replayed_callback = replayed_callbacks[stage];
    const bool replayed_advanced = replayed_callback.has_value();
    const bool replayed_legacy =
        (lifecycle_stage == LifecycleStage::kStart && replayed_legacy_hooks.start) ||
        (lifecycle_stage == LifecycleStage::kStop && replayed_legacy_hooks.stop);
    const LifecycleMethodId legacy_method = static_cast<LifecycleMethodId>(stage + 1U);
    if (!expected_callback) {
      if (replayed_advanced || replayed_legacy) {
        return mismatch(RuntimeReplayMismatchCategory::kLifecycle);
      }
      continue;
    }
    switch (expected->origin) {
      case ExecutableLifecycleCallbackOrigin::kAdvancedMember: {
        if (replayed_legacy || !replayed_callback.has_value()) {
          return mismatch(RuntimeReplayMismatchCategory::kLifecycle);
        }
        if (replayed_callback->required != expected->required ||
            replayed_callback->method != expected->method) {
          return mismatch(RuntimeReplayMismatchCategory::kLifecycle);
        }
        break;
      }
      case ExecutableLifecycleCallbackOrigin::kLegacyStart:
        if (lifecycle_stage != LifecycleStage::kStart || replayed_advanced || !replayed_legacy ||
            !expected->required || expected->method != legacy_method) {
          return mismatch(RuntimeReplayMismatchCategory::kLifecycle);
        }
        break;
      case ExecutableLifecycleCallbackOrigin::kLegacyStop:
        if (lifecycle_stage != LifecycleStage::kStop || replayed_advanced || !replayed_legacy ||
            !expected->required || expected->method != legacy_method) {
          return mismatch(RuntimeReplayMismatchCategory::kLifecycle);
        }
        break;
    }
  }
  if (clock_source || input_source) {
    const std::size_t connected_input_count = replay_connected_input_count(plan, record.vertex);
    auto normalized = ContractTestAccess::normalize(op.contract(), connected_input_count);
    if (!normalized) {
      capture_replay_error(
          failure_detail, record.vertex, DiagnosticOperation::kContract, normalized.error());
      return mismatch(RuntimeReplayMismatchCategory::kTemporalContract,
                      normalized.error().code,
                      "RUNTIME_OPERATOR_REPLAY_MISMATCH");
    }
    const bool clock_schedule_matched =
        !clock_source ||
        (trigger->kind == TemporalRecordTriggerKind::kClock
             ? replayed_clock_contract_matches(plan, record, *normalized)
             : replayed_clock_predicate_contract_matches(plan, record, *normalized));
    const bool matched = clock_schedule_matched &&
                         replayed_completion_rule_matches(plan, record.operator_id, *normalized) &&
                         replayed_temporal_contract_matches(plan, record, op, *normalized);
    if (!matched) {
      return mismatch(RuntimeReplayMismatchCategory::kTemporalContract,
                      ErrorCode::kFailure,
                      "RUNTIME_OPERATOR_REPLAY_MISMATCH",
                      &*normalized);
    }
  } else {
    if (auto contract = bind_and_validate_direct_contract(plan, record, op, replay, failure_detail);
        !contract) {
      return mismatch(RuntimeReplayMismatchCategory::kTemporalContract,
                      contract.error().code,
                      "RUNTIME_OPERATOR_REPLAY_MISMATCH");
    }
  }
  if (!parameter_contract_matches_plan(plan, record, op)) {
    return mismatch(RuntimeReplayMismatchCategory::kParameterContract,
                    ErrorCode::kFailure,
                    "RUNTIME_PARAMETER_REPLAY_MISMATCH");
  }
  return {};
}
// NOLINTEND(readability-function-cognitive-complexity)

}  // namespace

expected<void, Error> validate_foundation_plan(const ExecutionPlanState& plan,
                                               std::size_t operator_count) {
  return validate_foundation_plan_impl(plan, operator_count);
}

expected<void, Error> validate_replayed_operator(const ExecutionPlanState& plan,
                                                 const ExecutableOperatorFactoryRecord& record,
                                                 OperatorBase& op, const OperatorSpec& replay,
                                                 std::uint32_t factory_slot,
                                                 RuntimeReplayMismatchDetail* replay_mismatch,
                                                 RuntimeDiagnosticDetail* failure_detail) {
  return validate_replayed_operator_impl(
      plan, record, op, replay, factory_slot, replay_mismatch, failure_detail);
}

}  // namespace holoscan::detail
