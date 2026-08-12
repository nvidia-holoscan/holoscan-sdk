/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "graph_exec.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>

#include "../channel/endpoint_catalog_state.hpp"
#include "../compiler/closed_island_activation_plan.hpp"
#include "../compiler/inline_partition_execution_plan.hpp"
#include "../compiler/partition_runner_plan.hpp"
#include "../core/clock_binding_resolution.hpp"
#include "../core/diagnostic_registry.hpp"
#include "../core/lifecycle_capacity_derivation.hpp"
#include "../core/operator_runtime_access.hpp"
#include "../core/operator_spec_compiler_access.hpp"
#include "../core/readiness_source_internal.hpp"
#include "../core/source_id_internal.hpp"
#include "../core/temporal_contract_internal.hpp"
#include "dispatcher.hpp"
#include "lifecycle_executor.hpp"
#include "parameter_control_state.hpp"
#include "runtime_worker.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] Error startup_error(ErrorCode code, const char* token) {
  return Error{code, token};
}

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
[[nodiscard]] ParamSchemaHash observed_replay_fingerprint(
    const OperatorSpec& replay, RuntimeReplayMismatchCategory category,
    const TemporalContractSnapshot* temporal = nullptr) {
  StableHashBuilder hash;
  hash.add("holoscan.runtime.operator-replay.observed/v1");
  hash.add(category);
  hash.add(OperatorSpecCompilerAccess::declaration_error(replay).code);

  hash.add_canonical_size(OperatorSpecCompilerAccess::port_count(replay));
  for (std::size_t index = 0U; index < OperatorSpecCompilerAccess::port_count(replay); ++index) {
    const auto port = OperatorSpecCompilerAccess::port(replay, index);
    hash.add(port.direction);
    hash.add(port.name);
    hash.add(port.tensor);
    hash.add(port.payload_identity.kind);
    hash.add(port.payload_identity.payload_kind);
    hash.add(port.serialized_payload_capacity_bytes);
    hash.add(port.queue_depth);
    hash.add(port.max_emits_per_compute);
    hash.add(port.late_joiner_history_depth);
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
  hash.add(temporal->deadline.has_value());
  if (temporal->deadline) {
    hash.add(temporal->deadline->count());
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
  return startup_error(code, token);
}

[[nodiscard]] bool executable_vertex(VertexKind kind) noexcept {
  return kind == VertexKind::kOperator || kind == VertexKind::kImportedTopic ||
         kind == VertexKind::kExportedTopic;
}

[[nodiscard]] bool accepted_partition_algorithm_supported(
    const PartitioningPlanRecord& partitioning) noexcept {
  if (partitioning.disposition != PlanDisposition::kAccepted ||
      partitioning.interface_version != 1U || !partitioning.requested_algorithm.has_value() ||
      !partitioning.accepted_algorithm.has_value() ||
      partitioning.requested_algorithm != partitioning.accepted_algorithm || partitioning.reason ||
      partitioning.selection_source == PlanSelectionSource::kExplicitFallback) {
    return false;
  }

  const AlgorithmKey& accepted = *partitioning.accepted_algorithm;
  if (accepted.id == "holoscan.partition.serial_baseline" && accepted.version == 1U) {
    return partitioning.maturity == AlgorithmMaturity::kProduction;
  }
  if (accepted.id == "holoscan.partition.source_chain" && accepted.version == 1U) {
    return partitioning.maturity == AlgorithmMaturity::kExperimental;
  }
  return false;
}

[[nodiscard]] bool scheduler_plan_supported(const ExecutionPlanState& plan, bool zero_work) {
  const SchedulerPlanRecord& scheduler = plan.scheduler_plan;
  if (scheduler.ready_policy.id != "holoscan.ready.deterministic_serial" ||
      scheduler.ready_policy.version != 1U || scheduler.reason) {
    return false;
  }
  if (zero_work) {
    return scheduler.execution_lane_count == 0U && scheduler.max_concurrent_activations == 0U &&
           scheduler.worker_count == 0U && scheduler.intra_path_parallelism == 0U &&
           plan.execution_lanes.empty() && plan.placement_probes.empty() &&
           plan.partition_placements.empty();
  }
  const std::uint32_t callback_owner_count =
      static_cast<std::uint32_t>(plan.partition_runners.size());
  return scheduler.execution_lane_count != 0U &&
         scheduler.max_concurrent_activations == callback_owner_count &&
         scheduler.activation_overlap == (callback_owner_count > 1U) &&
         scheduler.worker_count == callback_owner_count &&
         scheduler.intra_path_parallelism == (callback_owner_count == 0U ? 0U : 1U) &&
         plan.execution_lanes.size() == scheduler.execution_lane_count;
}

[[nodiscard]] bool deadline_equal(const std::optional<DeadlineScheduling>& lhs,
                                  const std::optional<DeadlineScheduling>& rhs) noexcept {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  return !lhs || (lhs->runtime_ns == rhs->runtime_ns && lhs->deadline_ns == rhs->deadline_ns &&
                  lhs->period_ns == rhs->period_ns);
}

[[nodiscard]] bool placement_equal(const ThreadPlacement& lhs,
                                   const ThreadPlacement& rhs) noexcept {
  const bool affinity_equal = lhs.affinity.has_value() == rhs.affinity.has_value() &&
                              (!lhs.affinity || lhs.affinity->cores == rhs.affinity->cores);
  return affinity_equal && lhs.scheduling.policy == rhs.scheduling.policy &&
         lhs.scheduling.priority == rhs.scheduling.priority &&
         deadline_equal(lhs.scheduling.deadline, rhs.scheduling.deadline) &&
         lhs.wake_mode == rhs.wake_mode && lhs.irq_isolation == rhs.irq_isolation &&
         lhs.probe_failure_policy == rhs.probe_failure_policy;
}

[[nodiscard]] const ExecutionPartitionRecord* partition_record(const ExecutionPlanState& plan,
                                                               ExecutionPartitionId id) noexcept {
  if (!id || id.value > plan.execution_partitions.size()) {
    return nullptr;
  }
  const auto& partition = plan.execution_partitions[static_cast<std::size_t>(id.value - 1U)];
  return partition.id == id ? &partition : nullptr;
}

[[nodiscard]] std::optional<std::uint8_t> placement_probe_bit(std::string_view probe_id) noexcept {
  if (probe_id == "holoscan.placement.syntax/v1") {
    return 1U << 0U;
  }
  if (probe_id == "holoscan.placement.profile_scope/v1") {
    return 1U << 1U;
  }
  if (probe_id == "holoscan.placement.platform/v1") {
    return 1U << 2U;
  }
  if (probe_id == "holoscan.placement.irq_isolation/v1") {
    return 1U << 3U;
  }
  if (probe_id == "holoscan.placement.apply_readback/v1") {
    return 1U << 4U;
  }
  return std::nullopt;
}

[[nodiscard]] bool placement_probe_status_supported(const PlacementProbePlanRecord& probe,
                                                    std::uint8_t bit) noexcept {
  if (probe.reason) {
    return false;
  }
  switch (bit) {
    case 1U << 0U:
    case 1U << 1U:
    case 1U << 2U:
      return probe.status == PlacementProbeStatus::kPassed && !probe.evidence_id.empty();
    case 1U << 3U:
      return probe.status == PlacementProbeStatus::kNotRequested ||
             ((probe.status == PlacementProbeStatus::kPassed ||
               probe.status == PlacementProbeStatus::kFallbackRecorded) &&
              !probe.evidence_id.empty());
    case 1U << 4U:
      return probe.status == PlacementProbeStatus::kRuntimeRequired && !probe.evidence_id.empty();
    default:
      return false;
  }
}

// Close every partition-keyed placement relationship before any lane is created. These records
// are independently serialized plan families, so nonzero IDs and valid shapes alone do not prove
// that they still describe the same accepted partition inventory.
[[nodiscard]] bool partition_placement_relations_supported(const ExecutionPlanState& plan) {
  constexpr std::uint8_t kCompleteProbeMask = (1U << 5U) - 1U;
  const ThreadPlacement default_placement;
  std::vector<bool> lane_seen(plan.execution_partitions.size());
  for (std::size_t lane_index = 0U; lane_index < plan.execution_lanes.size(); ++lane_index) {
    const auto& lane = plan.execution_lanes[lane_index];
    const auto* partition = partition_record(plan, lane.partition);
    if (lane.execution_lane != lane_index || partition == nullptr ||
        partition->execution_lane != lane.execution_lane ||
        lane_seen[static_cast<std::size_t>(lane.partition.value - 1U)]) {
      return false;
    }
    lane_seen[static_cast<std::size_t>(lane.partition.value - 1U)] = true;
  }
  if (std::find(lane_seen.begin(), lane_seen.end(), false) != lane_seen.end()) {
    return false;
  }

  std::vector<bool> placement_seen(plan.execution_partitions.size());
  for (const auto& placement : plan.partition_placements) {
    const auto* partition = partition_record(plan, placement.partition);
    if (partition == nullptr || !partition->authored_name ||
        *partition->authored_name != placement.authored_name ||
        partition->execution_lane != placement.execution_lane ||
        placement.execution_lane >= plan.execution_lanes.size() ||
        plan.execution_lanes[placement.execution_lane].partition != placement.partition ||
        placement_seen[static_cast<std::size_t>(placement.partition.value - 1U)] ||
        placement.disposition != PlanDisposition::kAccepted || placement.reason ||
        !placement.accepted || !placement.requested.validate() || !placement.accepted->validate() ||
        !placement_equal(*placement.accepted,
                         plan.execution_lanes[placement.execution_lane].accepted_placement)) {
      return false;
    }
    placement_seen[static_cast<std::size_t>(placement.partition.value - 1U)] = true;
  }
  for (std::size_t partition_index = 0U; partition_index < placement_seen.size();
       ++partition_index) {
    const auto& partition = plan.execution_partitions[partition_index];
    if (!placement_seen[partition_index] &&
        !placement_equal(plan.execution_lanes[partition.execution_lane].accepted_placement,
                         default_placement)) {
      return false;
    }
  }

  std::vector<std::uint8_t> probe_masks(plan.execution_partitions.size());
  for (const auto& probe : plan.placement_probes) {
    const auto* partition = partition_record(plan, probe.partition);
    const auto bit = placement_probe_bit(probe.probe_id);
    if (partition == nullptr ||
        !placement_seen[static_cast<std::size_t>(probe.partition.value - 1U)] || !bit ||
        !placement_probe_status_supported(probe, *bit)) {
      return false;
    }
    auto& mask = probe_masks[static_cast<std::size_t>(probe.partition.value - 1U)];
    if ((mask & *bit) != 0U) {
      return false;
    }
    mask = static_cast<std::uint8_t>(mask | *bit);
  }
  for (std::size_t partition_index = 0U; partition_index < placement_seen.size();
       ++partition_index) {
    const std::uint8_t expected = placement_seen[partition_index] ? kCompleteProbeMask : 0U;
    if (probe_masks[partition_index] != expected) {
      return false;
    }
  }
  return true;
}

// Validate the complete dense partition inventory together so no partial mapping can reach runtime.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool partition_records_supported(const ExecutionPlanState& plan) {
  if (!accepted_partition_algorithm_supported(plan.partitioning_plan)) {
    return false;
  }

  std::vector<VertexId> executable_vertices;
  executable_vertices.reserve(plan.graph.vertices().size());
  for (const auto& vertex : plan.graph.vertices()) {
    if (executable_vertex(vertex.kind)) {
      executable_vertices.push_back(vertex.id);
    }
  }
  const bool zero_work = plan.graph.vertices().empty();
  if ((executable_vertices.empty() != zero_work) || !scheduler_plan_supported(plan, zero_work)) {
    return false;
  }
  if (zero_work) {
    return plan.execution_partitions.empty() && plan.partition_decisions.empty() &&
           plan.schedule_edges.empty();
  }
  if (plan.execution_partitions.empty() ||
      plan.execution_partitions.size() > executable_vertices.size() ||
      plan.partition_decisions.size() != executable_vertices.size()) {
    return false;
  }

  std::vector<ExecutionPartitionId> partition_by_vertex(executable_vertices.size());
  for (std::size_t partition_index = 0U; partition_index < plan.execution_partitions.size();
       ++partition_index) {
    const auto& partition = plan.execution_partitions[partition_index];
    if (partition.id.value != partition_index + 1U || partition.members.empty() ||
        partition.execution_lane >= plan.scheduler_plan.execution_lane_count) {
      return false;
    }
    VertexId previous{};
    for (const VertexId member : partition.members) {
      if ((previous && member.value <= previous.value)) {
        return false;
      }
      previous = member;
      const auto found = std::lower_bound(
          executable_vertices.begin(),
          executable_vertices.end(),
          member.value,
          [](VertexId vertex, std::uint64_t value) { return vertex.value < value; });
      if (found == executable_vertices.end() || *found != member) {
        return false;
      }
      const std::size_t vertex_index =
          static_cast<std::size_t>(std::distance(executable_vertices.begin(), found));
      if (partition_by_vertex[vertex_index]) {
        return false;
      }
      partition_by_vertex[vertex_index] = partition.id;
    }
  }

  for (std::size_t vertex_index = 0U; vertex_index < executable_vertices.size(); ++vertex_index) {
    const auto& decision = plan.partition_decisions[vertex_index];
    if (!partition_by_vertex[vertex_index] ||
        decision.vertex != executable_vertices[vertex_index] ||
        decision.partition != partition_by_vertex[vertex_index]) {
      return false;
    }
  }
  return partition_placement_relations_supported(plan);
}

[[nodiscard]] const ExecutionPartitionRecord* planned_partition(const ExecutionPlanState& plan,
                                                                VertexId vertex) noexcept {
  const auto found = std::find_if(
      plan.execution_partitions.begin(),
      plan.execution_partitions.end(),
      [vertex](const auto& partition) {
        return std::binary_search(partition.members.begin(),
                                  partition.members.end(),
                                  vertex,
                                  [](VertexId lhs, VertexId rhs) { return lhs.value < rhs.value; });
      });
  return found == plan.execution_partitions.end() ? nullptr : &*found;
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

[[nodiscard]] bool schema_identity_equal(const SchemaIdentityRecord& lhs,
                                         const SchemaIdentityRecord& rhs) noexcept {
  return lhs.canonical_name == rhs.canonical_name && lhs.version == rhs.version &&
         lhs.schema_sha256 == rhs.schema_sha256 && lhs.id == rhs.id;
}

[[nodiscard]] bool pod_identity_equal(const PodIdentityRecord& lhs,
                                      const PodIdentityRecord& rhs) noexcept {
  if (lhs.canonical_name != rhs.canonical_name || lhs.version != rhs.version ||
      lhs.object_size != rhs.object_size || lhs.object_alignment != rhs.object_alignment ||
      lhs.fields.size() != rhs.fields.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < lhs.fields.size(); ++index) {
    const auto& left = lhs.fields[index];
    const auto& right = rhs.fields[index];
    if (left.name != right.name || left.byte_offset != right.byte_offset ||
        left.scalar_kind != right.scalar_kind || left.element_count != right.element_count) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool replay_payload_identity_equal(const PayloadIdentityRecord& lhs,
                                                 const PayloadIdentityRecord& rhs) noexcept {
  if (lhs.kind != rhs.kind || lhs.payload_kind != rhs.payload_kind ||
      lhs.scalar.has_value() != rhs.scalar.has_value() ||
      lhs.schema.has_value() != rhs.schema.has_value() ||
      lhs.pod.has_value() != rhs.pod.has_value()) {
    return false;
  }
  if (lhs.scalar && lhs.scalar->kind != rhs.scalar->kind) {
    return false;
  }
  if (lhs.schema && !schema_identity_equal(*lhs.schema, *rhs.schema)) {
    return false;
  }
  return !lhs.pod || pod_identity_equal(*lhs.pod, *rhs.pod);
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
             ops->destroy_sample != nullptr && ops->sample_payload != nullptr &&
             ops->sample_metadata != nullptr && ops->bind_input != nullptr &&
             ops->unbind_input != nullptr && ops->bind_output != nullptr &&
             ops->unbind_output != nullptr &&
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
  }
  return false;
}

[[nodiscard]] bool replayed_ports_match(const ExecutionPlanState& plan, VertexId owner,
                                        const OperatorSpec& replay) {
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
        !replay_payload_identity_equal(planned_ports[planned_index].payload_identity,
                                       metadata.payload_identity)) {
      return false;
    }
    matched[planned_index] = true;
  }
  return true;
}

[[nodiscard]] bool plan_ref_matches_legacy_operator(const OperatorRef& ref,
                                                    std::size_t operator_index,
                                                    PlanGenerationId plan_generation) noexcept {
  return ref.id == static_cast<OperatorInstanceId>(operator_index + 1U) &&
         ref.plan_generation == plan_generation && ref.run_generation == 0U;
}

[[nodiscard]] bool valid_plan_operator_ref(const OperatorRef& ref, std::size_t operator_count,
                                           PlanGenerationId plan_generation) noexcept {
  return ref.plan_bound() && ref.plan_generation == plan_generation && ref.id != 0U &&
         ref.id <= operator_count;
}

[[nodiscard]] constexpr bool valid_lifecycle_stage(LifecycleStage stage) noexcept {
  return static_cast<std::uint8_t>(stage) <= static_cast<std::uint8_t>(LifecycleStage::kRelease);
}

[[nodiscard]] constexpr bool valid_restart_stage(LifecycleStage stage) noexcept {
  return static_cast<std::uint8_t>(stage) <= static_cast<std::uint8_t>(LifecycleStage::kStart);
}

[[nodiscard]] RequiredLifecycleCapacity expected_lifecycle_required_capacity(
    const ExecutionPlanState& plan, const LifecycleTransactionCapacityValues& configured) {
  const LifecycleTransactionCapacityValues parameter_required =
      plan.parameter_runtime_capacity ? plan.parameter_runtime_capacity->required_lifecycle_capacity
                                      : LifecycleTransactionCapacityValues{};
  std::vector<std::uint8_t> membership_marks(plan.lifecycle_restart_closures.size());
  return derive_required_lifecycle_capacity(
      configured,
      admitted_lifecycle_restart_capacity(plan.lifecycle_restart_targets,
                                          plan.lifecycle_restart_closures,
                                          plan.plan_generation,
                                          membership_marks),
      parameter_required);
}

[[nodiscard]] bool lifecycle_checkpoints_well_formed(const ExecutionPlanState& plan,
                                                     std::size_t operator_count) {
  constexpr std::size_t kLifecycleStageCount =
      static_cast<std::size_t>(LifecycleStage::kRelease) + 1U;
  if (plan.lifecycle_checkpoints.size() % kLifecycleStageCount != 0U ||
      plan.lifecycle_checkpoints.size() / kLifecycleStageCount != operator_count) {
    return false;
  }

  for (std::size_t stage_index = 0U; stage_index < kLifecycleStageCount; ++stage_index) {
    std::vector<std::uint8_t> seen(operator_count, 0U);
    std::uint32_t previous_level{};
    OperatorInstanceId previous_operator{};
    for (std::size_t operator_index = 0U; operator_index < operator_count; ++operator_index) {
      const auto& checkpoint =
          plan.lifecycle_checkpoints[stage_index * operator_count + operator_index];
      if (!valid_plan_operator_ref(
              checkpoint.operator_instance, operator_count, plan.plan_generation) ||
          checkpoint.stage != static_cast<LifecycleStage>(stage_index) ||
          (checkpoint.required && !checkpoint.callback_present)) {
        return false;
      }
      const std::size_t identity_index = checkpoint.operator_instance.id - 1U;
      if (seen[identity_index] != 0U ||
          (operator_index == 0U && checkpoint.stable_topological_level != 0U) ||
          checkpoint.stable_topological_level < previous_level ||
          checkpoint.stable_topological_level > previous_level + 1U ||
          (checkpoint.stable_topological_level == previous_level && operator_index != 0U &&
           checkpoint.operator_instance.id <= previous_operator)) {
        return false;
      }
      seen[identity_index] = 1U;
      previous_level = checkpoint.stable_topological_level;
      previous_operator = checkpoint.operator_instance.id;
    }
  }
  return true;
}

[[nodiscard]] bool lifecycle_targets_well_formed(const ExecutionPlanState& plan,
                                                 std::size_t operator_count) noexcept {
  if (plan.lifecycle_restart_targets.size() != operator_count) {
    return false;
  }
  for (std::size_t operator_index = 0U; operator_index < operator_count; ++operator_index) {
    const auto& target = plan.lifecycle_restart_targets[operator_index];
    if (!plan_ref_matches_legacy_operator(target.target, operator_index, plan.plan_generation)) {
      return false;
    }
    if (plan.lifecycle_transaction_capacity.has_value()) {
      if (target.disposition == LifecycleRestartTargetDisposition::kDisabled) {
        return false;
      }
    } else if (target.disposition != LifecycleRestartTargetDisposition::kDisabled ||
               target.self_allowed || target.external_control_allowed ||
               target.parameter_origin_allowed || target.allowed_from_stage_mask != 0U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool lifecycle_dependencies_well_formed(const ExecutionPlanState& plan,
                                                      std::size_t operator_count) noexcept {
  return std::all_of(
      plan.lifecycle_dependencies.begin(),
      plan.lifecycle_dependencies.end(),
      [&](const auto& dependency) {
        const bool key_shape = dependency.kind == LifecycleEdgeKind::kBefore
                                   ? dependency.dependency_key.empty()
                                   : dependency.kind == LifecycleEdgeKind::kDependencyKey &&
                                         !dependency.dependency_key.empty();
        return valid_plan_operator_ref(dependency.before, operator_count, plan.plan_generation) &&
               valid_plan_operator_ref(dependency.after, operator_count, plan.plan_generation) &&
               dependency.before.id != dependency.after.id &&
               valid_lifecycle_stage(dependency.stage) && key_shape;
      });
}

[[nodiscard]] bool lifecycle_permissions_well_formed(const ExecutionPlanState& plan,
                                                     std::size_t operator_count) noexcept {
  if (!plan.lifecycle_transaction_capacity && !plan.lifecycle_restart_permissions.empty()) {
    return false;
  }

  OperatorInstanceId previous_requester{};
  OperatorInstanceId previous_target{};
  for (const auto& permission : plan.lifecycle_restart_permissions) {
    if (!valid_plan_operator_ref(permission.requester, operator_count, plan.plan_generation) ||
        !valid_plan_operator_ref(permission.target, operator_count, plan.plan_generation) ||
        permission.requester.id == permission.target.id ||
        std::tuple{permission.requester.id, permission.target.id} <=
            std::tuple{previous_requester, previous_target}) {
      return false;
    }
    previous_requester = permission.requester.id;
    previous_target = permission.target.id;
  }
  return true;
}

[[nodiscard]] bool lifecycle_mutable_fields_well_formed(const ExecutionPlanState& plan,
                                                        std::size_t operator_count) noexcept {
  if (!plan.lifecycle_transaction_capacity && !plan.lifecycle_mutable_fields.empty()) {
    return false;
  }

  OperatorInstanceId previous_owner{};
  ParamFieldId previous_field{};
  for (const auto& field : plan.lifecycle_mutable_fields) {
    if (!valid_plan_operator_ref(field.owner, operator_count, plan.plan_generation) ||
        field.field == 0U || !valid_restart_stage(field.from_stage) ||
        std::tuple{field.owner.id, field.field} <= std::tuple{previous_owner, previous_field}) {
      return false;
    }
    const auto planned_field = std::find_if(
        plan.parameter_fields.begin(), plan.parameter_fields.end(), [&](const auto& row) {
          return row.owner == field.owner && row.field == field.field;
        });
    if (planned_field == plan.parameter_fields.end() ||
        planned_field->mutability != ParamMutability::kLifecycle) {
      return false;
    }
    previous_owner = field.owner.id;
    previous_field = field.field;
  }
  return true;
}

// Keep closure membership, dependency containment, and per-stage widening checks in one
// fail-closed validator because they jointly authenticate the frozen restart transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool lifecycle_closures_well_formed(const ExecutionPlanState& plan,
                                                  std::size_t operator_count) noexcept {
  constexpr std::uint8_t kAllRestartStagesMask = 0x3fU;
  if (plan.lifecycle_restart_closures.size() != operator_count) {
    return false;
  }

  for (std::size_t target_index = 0U; target_index < operator_count; ++target_index) {
    const auto& closure = plan.lifecycle_restart_closures[target_index];
    if (!plan_ref_matches_legacy_operator(closure.target, target_index, plan.plan_generation) ||
        closure.closure.empty() || closure.valid_requested_stage_mask != kAllRestartStagesMask ||
        closure.required_fenced_operators != closure.closure.size()) {
      return false;
    }

    OperatorInstanceId previous{};
    bool contains_target = false;
    for (const OperatorRef member : closure.closure) {
      if (!valid_plan_operator_ref(member, operator_count, plan.plan_generation) ||
          member.id <= previous) {
        return false;
      }
      contains_target = contains_target || member == closure.target;
      previous = member.id;
    }
    if (!contains_target) {
      return false;
    }
    for (const auto& dependency : plan.lifecycle_dependencies) {
      const bool contains_before =
          std::binary_search(closure.closure.begin(),
                             closure.closure.end(),
                             dependency.before,
                             [](auto lhs, auto rhs) { return lhs.id < rhs.id; });
      const bool contains_after = std::binary_search(
          closure.closure.begin(), closure.closure.end(), dependency.after, [](auto lhs, auto rhs) {
            return lhs.id < rhs.id;
          });
      if (contains_before && !contains_after) {
        return false;
      }
    }

    std::uint32_t maximum_quiesce{};
    std::uint32_t maximum_rollback{};
    for (std::size_t stage_index = 0U; stage_index < closure.stage_resolutions.size();
         ++stage_index) {
      const auto& resolution = closure.stage_resolutions[stage_index];
      const LifecycleStage requested = static_cast<LifecycleStage>(stage_index);
      if (resolution.requested_stage != requested ||
          !valid_restart_stage(resolution.effective_stage) ||
          static_cast<std::uint8_t>(resolution.effective_stage) > stage_index ||
          (resolution.effective_stage == requested) !=
              (resolution.reason == LifecycleRestartWideningReason::kNone) ||
          (resolution.effective_stage != requested &&
           resolution.reason != LifecycleRestartWideningReason::kDependencyStage) ||
          resolution.required_structural_event_records != kLifecycleStructuralEventsPerRequest) {
        return false;
      }
      maximum_quiesce = std::max(maximum_quiesce, resolution.required_quiesce_records);
      maximum_rollback = std::max(maximum_rollback, resolution.required_rollback_records);
    }
    if (closure.required_quiesce_records != maximum_quiesce ||
        closure.required_rollback_records != maximum_rollback) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool lifecycle_capacity_well_formed(const ExecutionPlanState& plan) {
  if (!plan.lifecycle_transaction_capacity) {
    return true;
  }
  const auto& capacity = *plan.lifecycle_transaction_capacity;
  const RequiredLifecycleCapacity required =
      expected_lifecycle_required_capacity(plan, capacity.configured);
  if (!lifecycle_capacity_values_positive(capacity.configured) || required.arithmetic_overflow ||
      required.invalid_input ||
      !lifecycle_capacity_values_equal(capacity.required_minimum, required.values) ||
      !lifecycle_capacity_values_satisfy(capacity.configured, capacity.required_minimum) ||
      capacity.derivation_reason != kCanonicalGraphLifecycleCapacityDerivationReason ||
      capacity.resolved_parameter_update_policy == ParamRuntimeUpdatePolicy::kProfileDefault) {
    return false;
  }

  if (!plan.parameter_runtime_capacity) {
    return capacity.resolved_parameter_update_policy == ParamRuntimeUpdatePolicy::kDisabled &&
           !capacity.parameter_origin_enabled &&
           capacity.required_generation_slots_per_operator == 0U &&
           capacity.required_parameter_arena_bytes == 0U &&
           capacity.required_parameter_audit_records == 0U;
  }
  const auto& parameters = *plan.parameter_runtime_capacity;
  return capacity.resolved_parameter_update_policy == parameters.resolved_update_policy &&
         capacity.parameter_origin_enabled == parameters.lifecycle_updates_enabled &&
         capacity.required_generation_slots_per_operator ==
             parameters.generation_slots_per_operator &&
         capacity.required_parameter_arena_bytes == parameters.parameter_arena_bytes &&
         capacity.required_parameter_audit_records == parameters.audit_capacity &&
         lifecycle_capacity_values_satisfy(capacity.required_minimum,
                                           parameters.required_lifecycle_capacity);
}

[[nodiscard]] bool lifecycle_target_authority_well_formed(const ExecutionPlanState& plan) noexcept {
  constexpr std::uint8_t kAllRestartStagesMask = 0x3fU;
  for (const auto& target : plan.lifecycle_restart_targets) {
    const bool has_cross_permission =
        std::any_of(plan.lifecycle_restart_permissions.begin(),
                    plan.lifecycle_restart_permissions.end(),
                    [&](const auto& permission) { return permission.target == target.target; });
    const bool has_mutable_field =
        std::any_of(plan.lifecycle_mutable_fields.begin(),
                    plan.lifecycle_mutable_fields.end(),
                    [&](const auto& field) { return field.owner == target.target; });
    const bool has_authority = target.self_allowed || target.external_control_allowed ||
                               target.parameter_origin_allowed || has_cross_permission;

    if (!plan.lifecycle_transaction_capacity) {
      if (target.disposition != LifecycleRestartTargetDisposition::kDisabled || has_authority ||
          target.allowed_from_stage_mask != 0U) {
        return false;
      }
      continue;
    }
    if (target.disposition == LifecycleRestartTargetDisposition::kNotRequested) {
      if (has_authority || target.allowed_from_stage_mask != 0U) {
        return false;
      }
      continue;
    }
    if (target.disposition != LifecycleRestartTargetDisposition::kAdmitted || !has_authority ||
        target.allowed_from_stage_mask == 0U ||
        (target.allowed_from_stage_mask & ~kAllRestartStagesMask) != 0U ||
        target.parameter_origin_allowed != has_mutable_field) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool lifecycle_plan_well_formed(const ExecutionPlanState& plan,
                                              std::size_t operator_count) {
  if (!lifecycle_checkpoints_well_formed(plan, operator_count) ||
      !lifecycle_targets_well_formed(plan, operator_count) ||
      !lifecycle_dependencies_well_formed(plan, operator_count) ||
      !lifecycle_permissions_well_formed(plan, operator_count) ||
      !lifecycle_mutable_fields_well_formed(plan, operator_count) ||
      !lifecycle_closures_well_formed(plan, operator_count) ||
      !lifecycle_capacity_well_formed(plan) || !lifecycle_target_authority_well_formed(plan)) {
    return false;
  }
  return true;
}

[[nodiscard]] const ReadinessPlanRecord* planned_source(const ExecutionPlanState& plan,
                                                        VertexId owner) noexcept {
  const auto found = std::find_if(plan.readiness.begin(),
                                  plan.readiness.end(),
                                  [&](const auto& row) { return row.owner == owner; });
  return found == plan.readiness.end() ? nullptr : &*found;
}

[[nodiscard]] const TemporalTriggerPlanRecord* planned_trigger(const ExecutionPlanState& plan,
                                                               OperatorInstanceId owner) noexcept {
  const auto found = std::find_if(plan.temporal_triggers.begin(),
                                  plan.temporal_triggers.end(),
                                  [&](const auto& row) { return row.owner.id == owner; });
  return found == plan.temporal_triggers.end() ? nullptr : &*found;
}

[[nodiscard]] const ClockTriggerPlanRecord* planned_clock_trigger(
    const ExecutionPlanState& plan, const OperatorRef owner) noexcept {
  const auto found = std::find_if(plan.clock_triggers.begin(),
                                  plan.clock_triggers.end(),
                                  [&](const auto& row) { return row.owner == owner; });
  return found == plan.clock_triggers.end() ? nullptr : &*found;
}

[[nodiscard]] const PredicateTriggerPlanRecord* planned_predicate_trigger(
    const ExecutionPlanState& plan, const OperatorRef owner) noexcept {
  const auto found = std::find_if(plan.predicate_triggers.begin(),
                                  plan.predicate_triggers.end(),
                                  [&](const auto& row) { return row.owner == owner; });
  return found == plan.predicate_triggers.end() ? nullptr : &*found;
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
          planned_predicate_trigger(plan, trigger.owner);
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
          ? planned_predicate_trigger(plan, trigger.owner)
          : nullptr;
  const bool predicate_clock =
      predicate != nullptr && predicate->source == PredicateEvaluationSourceKind::kClock;
  if ((!direct_clock && !predicate_clock) || !trigger.readiness_source ||
      !trigger.trigger_clock.valid() || trigger.trigger_clock != trigger.reference_clock ||
      trigger.period_ns <= 0 || trigger.owner.plan_generation != plan.plan_generation ||
      trigger.owner.run_generation != 0U) {
    return false;
  }

  const ClockTriggerPlanRecord* const clock = planned_clock_trigger(plan, trigger.owner);
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
  const TemporalTriggerPlanRecord* const trigger = planned_trigger(plan, record.operator_id);
  if (trigger == nullptr || trigger->owner.id != record.operator_id ||
      trigger->owner.plan_generation != plan.plan_generation ||
      trigger->owner.run_generation != 0U) {
    return false;
  }
  if (trigger->kind == TemporalRecordTriggerKind::kClock ||
      trigger->kind == TemporalRecordTriggerKind::kPredicate) {
    const PredicateTriggerPlanRecord* const predicate =
        trigger->kind == TemporalRecordTriggerKind::kPredicate
            ? planned_predicate_trigger(plan, trigger->owner)
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
  const ReadinessPlanRecord* const source = planned_source(plan, record.vertex);
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

[[nodiscard]] const OperatorParameterPlanRecord* planned_parameters(
    const ExecutionPlanState& plan, OperatorInstanceId owner) noexcept {
  const auto found = std::find_if(
      plan.operator_parameters.begin(), plan.operator_parameters.end(), [&](const auto& row) {
        return row.owner.id == owner && row.owner.plan_generation == plan.plan_generation &&
               row.owner.run_generation == 0U;
      });
  return found == plan.operator_parameters.end() ? nullptr : &*found;
}

template <typename Record>
[[nodiscard]] bool parameter_owner_groups_well_formed(std::span<const Record> records,
                                                      std::size_t operator_count,
                                                      PlanGenerationId plan_generation) {
  std::vector<bool> closed_owner(operator_count + 1U);
  OperatorInstanceId active_owner{};
  for (const Record& record : records) {
    const OperatorRef owner = record.owner;
    if (!owner.plan_bound() || owner.plan_generation != plan_generation ||
        owner.id > operator_count) {
      return false;
    }
    if (owner.id == active_owner) {
      continue;
    }
    if (closed_owner[owner.id]) {
      return false;
    }
    if (active_owner != 0U) {
      closed_owner[active_owner] = true;
    }
    active_owner = owner.id;
  }
  return true;
}

[[nodiscard]] std::span<const ParameterFieldPlanRecord> planned_parameter_fields(
    const ExecutionPlanState& plan, OperatorInstanceId owner) noexcept {
  const auto first = std::find_if(plan.parameter_fields.begin(),
                                  plan.parameter_fields.end(),
                                  [&](const auto& field) { return field.owner.id == owner; });
  if (first == plan.parameter_fields.end()) {
    return {};
  }
  const auto last = std::find_if(first, plan.parameter_fields.end(), [&](const auto& field) {
    return field.owner.id != owner;
  });
  return std::span<const ParameterFieldPlanRecord>{&*first, static_cast<std::size_t>(last - first)};
}

[[nodiscard]] std::span<const ParameterDerivedTermPlanRecord> planned_parameter_derived_terms(
    const ExecutionPlanState& plan, OperatorInstanceId owner) noexcept {
  const auto first = std::find_if(plan.parameter_derived_terms.begin(),
                                  plan.parameter_derived_terms.end(),
                                  [&](const auto& term) { return term.owner.id == owner; });
  if (first == plan.parameter_derived_terms.end()) {
    return {};
  }
  const auto last = std::find_if(first, plan.parameter_derived_terms.end(), [&](const auto& term) {
    return term.owner.id != owner;
  });
  return std::span<const ParameterDerivedTermPlanRecord>{&*first,
                                                         static_cast<std::size_t>(last - first)};
}

[[nodiscard]] expected<std::vector<std::string>, Error> planned_operator_path(
    const ExecutionPlanState& plan, VertexId owner) {
  const auto vertex = std::find_if(plan.graph.vertices().begin(),
                                   plan.graph.vertices().end(),
                                   [owner](const auto& row) { return row.id == owner; });
  if (vertex == plan.graph.vertices().end() || vertex->name.empty()) {
    return make_unexpected(
        startup_error(ErrorCode::kFailure, "RUNTIME_PARAMETER_INVENTORY_MISMATCH"));
  }
  const auto graph = std::find_if(plan.graph.graphs().begin(),
                                  plan.graph.graphs().end(),
                                  [&](const auto& row) { return row.id == vertex->graph; });
  if (graph == plan.graph.graphs().end()) {
    return make_unexpected(
        startup_error(ErrorCode::kFailure, "RUNTIME_PARAMETER_INVENTORY_MISMATCH"));
  }
  std::vector<std::string> path;
  if (graph->path.has_value()) {
    // The optional belongs to the immutable snapshot and cannot change between this guard and use.
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    const GraphInstancePath& graph_path = graph->path.value();
    path.assign(graph_path.segments().begin(), graph_path.segments().end());
    // NOLINTEND(bugprone-unchecked-optional-access)
  }
  path.push_back(vertex->name);
  return path;
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
         left.has_stable_shape == right.has_stable_shape;
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
           planned_fields == 0U;
  }
  if (planned.schema_availability != ParamSchemaAvailability::kDeclared ||
      planned.schema_id != schema->stable_id || planned.schema_version != schema->version ||
      planned.structural_hash != schema->structural_hash ||
      planned.contract_hash != schema->contract_hash ||
      planned.metadata_hash != schema->metadata_hash ||
      planned.schema_coverage != schema->coverage || planned.schema_proof != schema->proof ||
      planned_fields != schema->fields.size()) {
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
  const OperatorParameterPlanRecord* const planned = planned_parameters(plan, record.operator_id);
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
      return identity.payload_kind == PayloadKind::kSerializedBytes && !identity.scalar &&
             !identity.pod && identity.schema;
    case PayloadIdentityKind::kTensor:
      return identity.payload_kind == PayloadKind::kDescriptorBacked && !identity.scalar &&
             !identity.pod && !identity.schema;
  }
  return false;
}

[[nodiscard]] bool same_supported_external_payload(const PayloadIdentityRecord& left,
                                                   const PayloadIdentityRecord& right) noexcept {
  return supported_external_payload(left) && supported_external_payload(right) &&
         replay_payload_identity_equal(left, right);
}

[[nodiscard]] bool logical_tensor_equal(const LogicalTensorRequirementsRecord& left,
                                        const LogicalTensorRequirementsRecord& right) noexcept {
  const bool dtype_equal = left.dtype.has_value() == right.dtype.has_value() &&
                           (!left.dtype || (left.dtype->code == right.dtype->code &&
                                            left.dtype->bits == right.dtype->bits &&
                                            left.dtype->lanes == right.dtype->lanes));
  if (left.memory_kind != right.memory_kind || !dtype_equal || left.rank != right.rank ||
      left.max_rank != right.max_rank || left.max_byte_span != right.max_byte_span ||
      left.strides != right.strides || left.foreign_emit_policy != right.foreign_emit_policy ||
      left.max_inline_bytes != right.max_inline_bytes || left.axes.size() != right.axes.size()) {
    return false;
  }
  return std::equal(
      left.axes.begin(), left.axes.end(), right.axes.begin(), [](const auto& lhs, const auto& rhs) {
        return lhs.max_extent == rhs.max_extent &&
               lhs.max_abs_stride_bytes == rhs.max_abs_stride_bytes;
      });
}

[[nodiscard]] bool catalog_family_projection_matches(
    const ChannelEndpointRecord& endpoint, const PayloadIdentityRecord& identity,
    const std::optional<LogicalTensorRequirementsRecord>& topic_tensor,
    const InterfaceTensorEmitPolicy service_tensor_emit_policy) noexcept {
  const bool pod = identity.kind == PayloadIdentityKind::kPod;
  const bool schema = identity.kind == PayloadIdentityKind::kSchema;
  const bool tensor = identity.kind == PayloadIdentityKind::kTensor;
  if (endpoint.pod_identity.has_value() != pod ||
      endpoint.service_static.pod_identity.has_value() != pod ||
      endpoint.schema_identity.has_value() != schema ||
      endpoint.service_static.schema_identity.has_value() != schema ||
      endpoint.tensor.has_value() != tensor ||
      endpoint.service_static.tensor.has_value() != tensor || topic_tensor.has_value() != tensor ||
      endpoint.serialized_payload_capacity_bytes !=
          endpoint.service_static.serialized_payload_capacity_bytes ||
      (schema != (endpoint.serialized_payload_capacity_bytes != 0U))) {
    return false;
  }
  if (pod) {
    if (!identity.pod || !endpoint.pod_identity || !endpoint.service_static.pod_identity) {
      return false;
    }
    const auto& projected = endpoint.pod_identity.value();
    const auto& static_projected = endpoint.service_static.pod_identity.value();
    const auto& identity_pod = identity.pod.value();
    return pod_identity_equal(projected.identity, identity_pod) &&
           pod_identity_equal(static_projected.identity, identity_pod) &&
           projected.digest_bytes == static_projected.digest_bytes &&
           projected.key == static_projected.key &&
           projected.layout_digest_bytes == static_projected.layout_digest_bytes;
  }
  if (schema) {
    if (!identity.schema || !endpoint.schema_identity || !endpoint.service_static.schema_identity) {
      return false;
    }
    return schema_identity_equal(endpoint.schema_identity.value(), identity.schema.value()) &&
           schema_identity_equal(endpoint.service_static.schema_identity.value(),
                                 identity.schema.value());
  }
  if (tensor) {
    return logical_tensor_equal(*endpoint.tensor, *endpoint.service_static.tensor) &&
           logical_tensor_equal(*endpoint.tensor, *topic_tensor) &&
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
    const auto service = std::find_if(plan.channel_services.begin(),
                                      plan.channel_services.end(),
                                      [&](const auto& row) { return row.topic == topic->id; });
    if (service == plan.channel_services.end()) {
      return false;
    }
    const PayloadIdentityRecord& static_payload = endpoint.service_static.payload_identity;
    const bool tensor = endpoint.payload_identity.kind == PayloadIdentityKind::kTensor;
    return endpoint.direction ==
               (topic->imported ? TopicDirection::kImport : TopicDirection::kExport) &&
           endpoint.service == service->service_id && service->tensor_payload == tensor &&
           catalog_family_projection_matches(
               endpoint, endpoint.payload_identity, topic->tensor, service->tensor_emit_policy) &&
           same_supported_external_payload(endpoint.payload_identity, static_payload) &&
           same_supported_external_payload(endpoint.payload_identity, topic->payload_identity) &&
           same_supported_external_payload(endpoint.payload_identity, service->payload_identity) &&
           endpoint.serialized_payload_capacity_bytes == topic->serialized_payload_capacity_bytes;
  });
}

[[nodiscard]] expected<void, Error> validate_foundation_plan(const ExecutionPlanState& plan,
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
    return make_unexpected(
        startup_error(ErrorCode::kNotSupported, "RUNTIME_PROFILE_COMPATIBILITY_FAILED"));
  }
  // Deadline rows form a closed inventory over trigger rows. Validate that
  // relation before topology-dependent placement so an orphan record in an
  // operator-free plan cannot be hidden by the empty placement disposition.
  if (auto deadlines = validate_runtime_deadline_inventory(plan.deadlines,
                                                           plan.temporal_triggers,
                                                           plan.predicate_triggers,
                                                           plan.clock_conversions,
                                                           plan.plan_generation);
      !deadlines) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
  if (plan.graph.vertices().empty() && !plan.profile_manifest.materialization_obligations.empty()) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "PROFILE_MATERIALIZATION_FAILED"));
  }
  if (!partition_records_supported(plan)) {
    return make_unexpected(
        startup_error(ErrorCode::kNotSupported, "RUNTIME_PLACEMENT_BINDING_UNAVAILABLE"));
  }
  if (!valid_closed_island_activation_plan(plan)) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
  if (!valid_inline_partition_execution_plan(plan)) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
  const auto* endpoint_catalog = ChannelEndpointCatalogAccess::state(plan.endpoint_catalog);
  if (endpoint_catalog != nullptr && !endpoint_catalog->endpoints.empty() &&
      !external_bindings_supported(plan, *endpoint_catalog)) {
    return make_unexpected(
        startup_error(ErrorCode::kNotSupported, "RUNTIME_EXTERNAL_BINDING_UNAVAILABLE"));
  }
  if (operator_count != 0U && (plan.clocks.empty() || !plan.root_default_clock.valid())) {
    return make_unexpected(
        startup_error(ErrorCode::kNotSupported, "RUNTIME_CLOCK_BINDING_UNAVAILABLE"));
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
      return make_unexpected(
          startup_error(ErrorCode::kNotSupported, "RUNTIME_READINESS_BINDING_UNAVAILABLE"));
    }
    const bool triggers_valid =
        std::all_of(plan.executable_factories.operators.begin(),
                    plan.executable_factories.operators.end(),
                    [&](const auto& record) { return runtime_trigger_well_formed(plan, record); });
    if (plan.temporal_triggers.size() != operator_count || !triggers_valid) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
    }
  } else if (!plan.readiness.empty() || !plan.temporal_triggers.empty() ||
             !plan.clock_triggers.empty() || !plan.predicate_triggers.empty()) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
  return {};
}

[[nodiscard]] bool replayed_source_matches(const ExecutionPlanState& plan,
                                           const ExecutableOperatorFactoryRecord& record,
                                           const OperatorSpec& replay) noexcept {
  const auto replay_source = OperatorSpecCompilerAccess::source(replay, 0U);
  const ReadinessPlanRecord* const source = planned_source(plan, record.vertex);
  return source != nullptr &&
         (source->kind == ReadinessSourceKind::kSpin ||
          source->kind == ReadinessSourceKind::kNotification) &&
         replay_source.kind == source->kind && replay_source.name == source->name &&
         replay_source.capacity == source->requested_capacity &&
         replay_source.sender_reference_capacity == source->requested_sender_reference_capacity &&
         replay_source.clock == source->requested_clock && source->referenced &&
         source->materialized;
}

[[nodiscard]] ClockId resolve_authored_clock_id(const ExecutionPlanState& plan, GraphId graph,
                                                const ClockKey& key) noexcept {
  const AuthoredClockResolution resolved = resolve_authored_clock(plan.graph.graphs(),
                                                                  plan.graph.clocks(),
                                                                  plan.graph.clock_imports(),
                                                                  plan.graph.clock_bindings(),
                                                                  graph,
                                                                  key);
  if (!resolved.resolved()) {
    return {};
  }
  const auto found = std::find_if(plan.clocks.begin(), plan.clocks.end(), [&](const auto& clock) {
    return clock.graph == resolved.declaration->graph && clock.key == resolved.declaration->key;
  });
  return found == plan.clocks.end() ? ClockId{} : found->id;
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
  return resolve_authored_clock_id(plan, vertex->graph, key) == id;
}

[[nodiscard]] bool replayed_deadline_dependency_matches(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record,
    const OperatorBase& op, const TemporalContractSnapshot& replay,
    const TemporalTriggerPlanRecord& trigger) {
  if (!replay.deadline_dependency || replay.deadline || !trigger.deadline_ns ||
      !trigger.violation || replay.violation != trigger.violation) {
    return false;
  }
  const auto terms = planned_parameter_derived_terms(plan, record.operator_id);
  const auto first = std::find_if(terms.begin(), terms.end(), [](const auto& term) {
    return term.kind == ParameterDerivedTermKind::kContractDeadline;
  });
  if (first == terms.end() || std::find_if(std::next(first), terms.end(), [](const auto& term) {
                                return term.kind == ParameterDerivedTermKind::kContractDeadline;
                              }) != terms.end()) {
    return false;
  }

  const ParamSchemaDescriptor* const schema = replay.deadline_dependency.root_schema();
  const ParamFieldDescriptor* const field = replay.deadline_dependency.final_field();
  std::int64_t initial_value_ns{};
  if (schema == nullptr || field == nullptr ||
      replay.deadline_dependency.root_type_token !=
          OperatorRuntimeAccess::parameter_type(op).local_type_token ||
      !replay.deadline_dependency.read_initial_ns(OperatorRuntimeAccess::parameter_address(op),
                                                  initial_value_ns)) {
    return false;
  }
  return first->owner.id == record.operator_id &&
         first->owner.plan_generation == plan.plan_generation &&
         first->owner.run_generation == 0U && !first->input && first->field == field->id &&
         first->field_full_id == field->full_id &&
         first->field_path == replay.deadline_dependency.canonical_path() &&
         first->schema_id == schema->stable_id && first->schema_version == schema->version &&
         first->initial_value_ns == initial_value_ns && *trigger.deadline_ns == initial_value_ns;
}

[[nodiscard]] bool replayed_clock_contract_matches(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record,
    const TemporalContractSnapshot& replay) noexcept {
  const TemporalTriggerPlanRecord* const trigger = planned_trigger(plan, record.operator_id);
  const ClockTriggerPlanRecord* const clock =
      trigger == nullptr ? nullptr : planned_clock_trigger(plan, trigger->owner);
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
  const TemporalTriggerPlanRecord* const trigger = planned_trigger(plan, record.operator_id);
  const PredicateTriggerPlanRecord* const predicate =
      trigger == nullptr ? nullptr : planned_predicate_trigger(plan, trigger->owner);
  const PredicateCallablePlanRecord* callable = nullptr;
  if (predicate != nullptr) {
    const auto found = std::find_if(
        plan.predicate_callables.begin(), plan.predicate_callables.end(), [&](const auto& row) {
          return row.predicate_id == predicate->id;
        });
    callable = found == plan.predicate_callables.end() ? nullptr : &*found;
  }
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

[[nodiscard]] const InputSelectionPlanRecord* planned_input_selection(
    const ExecutionPlanState& plan, PortId input) noexcept {
  const auto found = std::find_if(plan.input_selections.begin(),
                                  plan.input_selections.end(),
                                  [&](const auto& row) { return row.input == input; });
  return found == plan.input_selections.end() ? nullptr : &*found;
}

[[nodiscard]] bool input_selection_owned_by(const ExecutionPlanState& plan,
                                            const InputSelectionPlanRecord& selection,
                                            VertexId owner) noexcept {
  const auto found = std::find_if(plan.ports.begin(), plan.ports.end(), [&](const auto& port) {
    return port.id == selection.input;
  });
  return found != plan.ports.end() && found->owner == owner &&
         found->direction == PortDirection::kInput;
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

  const auto terms = planned_parameter_derived_terms(plan, record.operator_id);
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
  const InputSelectionPlanRecord* const selection = planned_input_selection(plan, input);
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
    const InputSelectionPlanRecord* const selection = planned_input_selection(plan, inputs[index]);
    const auto& item = aggregate->inputs[index];
    if (selection == nullptr || item.input != inputs[index] || item.ordinal != index ||
        item.selector != selection->selector || item.missing != selection->missing) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] DeadlineSourceCoordinateKind replay_deadline_source(
    TemporalRecordTriggerKind kind, bool predicate_clock_source) noexcept {
  switch (kind) {
    case TemporalRecordTriggerKind::kAnyInput:
    case TemporalRecordTriggerKind::kEach:
      return DeadlineSourceCoordinateKind::kTriggerSample;
    case TemporalRecordTriggerKind::kClock:
      return DeadlineSourceCoordinateKind::kScheduledTick;
    case TemporalRecordTriggerKind::kReadable:
    case TemporalRecordTriggerKind::kNotified:
    case TemporalRecordTriggerKind::kDoorbell:
    case TemporalRecordTriggerKind::kSpin:
      return DeadlineSourceCoordinateKind::kReadinessWitness;
    case TemporalRecordTriggerKind::kAligned:
      return DeadlineSourceCoordinateKind::kAlignedHighOrClosureProof;
    case TemporalRecordTriggerKind::kWindow:
      return DeadlineSourceCoordinateKind::kWindowClose;
    case TemporalRecordTriggerKind::kPredicate:
      return predicate_clock_source ? DeadlineSourceCoordinateKind::kScheduledTick
                                    : DeadlineSourceCoordinateKind::kPredicateDriver;
  }
  return DeadlineSourceCoordinateKind::kTriggerSample;
}

[[nodiscard]] bool replayed_deadline_matches(const ExecutionPlanState& plan,
                                             const TemporalTriggerPlanRecord& trigger,
                                             const TemporalContractSnapshot& replay) noexcept {
  const auto first = std::find_if(plan.deadlines.begin(),
                                  plan.deadlines.end(),
                                  [&](const auto& row) { return row.owner == trigger.owner; });
  const std::size_t count = static_cast<std::size_t>(
      std::count_if(plan.deadlines.begin(), plan.deadlines.end(), [&](const auto& row) {
        return row.owner == trigger.owner;
      }));
  const bool present = replay.deadline.has_value() && replay.violation.has_value();
  if (!present) {
    return !replay.deadline && !replay.violation && !trigger.deadline_ns && !trigger.violation &&
           count == 0U;
  }
  const bool predicate_clock = trigger.kind == TemporalRecordTriggerKind::kPredicate &&
                               replay.trigger.predicate_clock_source;
  return count == 1U && first != plan.deadlines.end() && trigger.deadline_ns && trigger.violation &&
         *trigger.deadline_ns == replay.deadline->count() &&
         *trigger.violation == *replay.violation && first->budget_ns == replay.deadline->count() &&
         first->action == *replay.violation &&
         first->source_coordinate == replay_deadline_source(trigger.kind, predicate_clock) &&
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
      planned_predicate_trigger(plan, trigger.owner);
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

  const auto callable = std::find_if(
      plan.predicate_callables.begin(), plan.predicate_callables.end(), [&](const auto& row) {
        return row.predicate_id == predicate->id;
      });
  if (callable == plan.predicate_callables.end() ||
      callable->kind != replay.trigger.predicate.callable_kind() ||
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
    const InputSelectionPlanRecord* const selection = planned_input_selection(plan, expected);
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
  const TemporalTriggerPlanRecord* const trigger = planned_trigger(plan, record.operator_id);
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
  const bool deadline_matched =
      replayed_deadline_matches(plan, *trigger, replay) ||
      replayed_deadline_dependency_matches(plan, record, op, replay, *trigger);
  if (trigger->trigger_input != expected_trigger_input ||
      trigger->presence_guaranteed_inputs != expected_presence || !deadline_matched) {
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

[[nodiscard]] expected<void, Error> bind_and_validate_direct_contract(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record, OperatorBase& op,
    const OperatorSpec& replay) {
  const TemporalTriggerPlanRecord* const trigger = planned_trigger(plan, record.operator_id);
  const ReadinessPlanRecord* const source = planned_source(plan, record.vertex);
  if (trigger == nullptr || source == nullptr ||
      (source->kind != ReadinessSourceKind::kSpin &&
       source->kind != ReadinessSourceKind::kNotification) ||
      !source->source_id || !source->referenced || !source->materialized) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }

  const auto deadline = std::find_if(plan.deadlines.begin(),
                                     plan.deadlines.end(),
                                     [&](const auto& row) { return row.owner == trigger->owner; });
  const bool has_deadline = trigger->deadline_ns.has_value() && trigger->violation.has_value();
  const std::size_t deadline_count = static_cast<std::size_t>(
      std::count_if(plan.deadlines.begin(), plan.deadlines.end(), [&](const auto& row) {
        return row.owner == trigger->owner;
      }));
  const bool deadline_plan_matches =
      has_deadline
          ? deadline_count == 1U && deadline != plan.deadlines.end() &&
                deadline->budget_ns == *trigger->deadline_ns &&
                deadline->action == *trigger->violation &&
                deadline->source_coordinate == DeadlineSourceCoordinateKind::kReadinessWitness &&
                deadline->source_clock == trigger->reference_clock &&
                deadline->conversion_id == trigger->conversion_id &&
                deadline->callback_count_if_late ==
                    (*trigger->violation == OnViolation::kFailRun ? 0U : 1U) &&
                deadline->checked_addition && deadline->exact_boundary_is_on_time &&
                deadline->final_invocation_gate && deadline->consumes_candidate_tag &&
                !deadline->automatic_output && !deadline->callback_state_rollback &&
                deadline->duration_measurement_separate
          : deadline_count == 0U && !trigger->deadline_ns && !trigger->violation;
  if (!deadline_plan_matches) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }

  const auto replay_source = OperatorSpecCompilerAccess::source(replay, 0U);
  if (replay_source.kind != source->kind || replay_source.declaration == nullptr) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
  if (source->kind == ReadinessSourceKind::kSpin) {
    auto& spin = *static_cast<SpinSource*>(const_cast<void*>(replay_source.declaration));
    ReadinessSourceAccess::bind(spin,
                                ReadinessSourceAccess::make_ref<ReadinessSourceKind::kSpin>(
                                    source->source_id, plan.plan_generation));
  } else {
    auto& notification =
        *static_cast<NotificationSource*>(const_cast<void*>(replay_source.declaration));
    ReadinessSourceAccess::bind(notification,
                                ReadinessSourceAccess::make_ref<ReadinessSourceKind::kNotification>(
                                    source->source_id, plan.plan_generation));
  }

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    const std::size_t connected_input_count = replay_connected_input_count(plan, record.vertex);
    auto normalized = ContractTestAccess::normalize(op.contract(), connected_input_count);
    const auto normalized_deadline = normalized
                                         ? normalized->deadline.value_or(std::chrono::nanoseconds{})
                                         : std::chrono::nanoseconds{};
    const std::int64_t planned_deadline = trigger->deadline_ns.value_or(0);
    if (!normalized || replay_trigger_kind(normalized->trigger.kind) != trigger->kind ||
        normalized->trigger.source_id != source->source_id ||
        normalized->connected_input_count != connected_input_count ||
        !replayed_temporal_contract_matches(plan, record, op, *normalized) ||
        (normalized->deadline.has_value() != trigger->deadline_ns.has_value() &&
         !replayed_deadline_dependency_matches(plan, record, op, *normalized, *trigger)) ||
        (normalized->deadline && normalized_deadline.count() != planned_deadline) ||
        normalized->violation != trigger->violation || normalized->synthesized_default) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
#endif
  return {};
}

// Replay checks the complete public declaration against the frozen executable inventory before
// any runtime callback becomes reachable.
// NOLINTBEGIN(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> validate_replayed_operator(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record, OperatorBase& op,
    const OperatorSpec& replay, std::uint32_t factory_slot,
    RuntimeReplayMismatchDetail* replay_mismatch) {
  const auto mismatch = [&](RuntimeReplayMismatchCategory category,
                            ErrorCode code = ErrorCode::kFailure,
                            const char* token = "RUNTIME_OPERATOR_REPLAY_MISMATCH",
                            const TemporalContractSnapshot* temporal =
                                nullptr) -> expected<void, Error> {
    return make_unexpected(replay_mismatch_error(
        code, token, plan, record, factory_slot, category, replay, replay_mismatch, temporal));
  };
  if (OperatorSpecCompilerAccess::declaration_error(replay).code != ErrorCode::kSuccess) {
    return mismatch(RuntimeReplayMismatchCategory::kDeclaration);
  }

  const std::size_t replay_ports = OperatorSpecCompilerAccess::port_count(replay);
  if (replay_ports != planned_port_count(plan, record.vertex)) {
    return mismatch(RuntimeReplayMismatchCategory::kPorts);
  }
  if (!replayed_ports_match(plan, record.vertex, replay)) {
    // The process-local function table is verified against the complete
    // pointer-free payload identity before any lifecycle callback.
    return mismatch(RuntimeReplayMismatchCategory::kPorts,
                    ErrorCode::kNotSupported,
                    "RUNTIME_PORT_BINDING_UNAVAILABLE");
  }

  const std::size_t replay_sources = OperatorSpecCompilerAccess::source_count(replay);
  const TemporalTriggerPlanRecord* const trigger = planned_trigger(plan, record.operator_id);
  if (trigger == nullptr) {
    return mismatch(RuntimeReplayMismatchCategory::kTemporalContract);
  }
  const PredicateTriggerPlanRecord* const predicate =
      trigger->kind == TemporalRecordTriggerKind::kPredicate
          ? planned_predicate_trigger(plan, trigger->owner)
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
    const bool clock_schedule_matched =
        !clock_source ||
        (trigger->kind == TemporalRecordTriggerKind::kClock
             ? (normalized && replayed_clock_contract_matches(plan, record, *normalized))
             : (normalized &&
                replayed_clock_predicate_contract_matches(plan, record, *normalized)));
    const bool matched = normalized && clock_schedule_matched &&
                         replayed_temporal_contract_matches(plan, record, op, *normalized);
    if (!matched) {
      return mismatch(RuntimeReplayMismatchCategory::kTemporalContract,
                      ErrorCode::kFailure,
                      "RUNTIME_OPERATOR_REPLAY_MISMATCH",
                      normalized ? &*normalized : nullptr);
    }
  } else {
    if (auto contract = bind_and_validate_direct_contract(plan, record, op, replay); !contract) {
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

[[nodiscard]] expected<RuntimeOperatorInstance, Error> reconstruct_operator(
    const ExecutionPlanState& plan, const ExecutableOperatorFactoryRecord& record,
    std::uint32_t factory_slot, RuntimeReplayMismatchDetail* replay_mismatch) {
  if (record.factory == nullptr || record.operator_id == 0U) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_FACTORY_INVALID"));
  }
  // Construction, setup, and contract replay all execute user code on the run-owned worker.
  // Owning control calls could otherwise wait for the coordinator that this reconstruction path
  // has not yet attached, so expose only callback-safe borrowed seams throughout reconstruction.
  const ParamGeneralControlContextScope general_control_scope;
  auto created = record.factory->create();
  if (!created) {
    return make_unexpected(
        startup_error(created.error().code, "RUNTIME_OPERATOR_RECONSTRUCTION_FAILED"));
  }
  if (created.value() == nullptr) {
    return make_unexpected(
        startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_RECONSTRUCTION_FAILED"));
  }
  auto reconstructed_operator = std::move(created).value();

  auto replay = std::make_unique<OperatorSpec>();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    reconstructed_operator->setup(*replay);
    if (auto validated = validate_replayed_operator(
            plan, record, *reconstructed_operator, *replay, factory_slot, replay_mismatch);
        !validated) {
      return make_unexpected(std::move(validated.error()));
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_SETUP_FAILED"));
  }
#endif

  const ExecutionPartitionRecord* const partition = planned_partition(plan, record.vertex);
  if (partition == nullptr) {
    return make_unexpected(
        startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_INVENTORY_MISMATCH"));
  }

  auto tags =
      std::make_unique<TagAllocator>(1U, plan.executable_factories.test_initial_tag_sequence);
  RuntimeOperatorInstance instance{
      .vertex = record.vertex,
      .operator_id = record.operator_id,
      .partition = partition->id,
      .execution_lane = partition->execution_lane,
      .op = std::move(reconstructed_operator),
      .replay_spec = std::move(replay),
      .tags = std::move(tags),
  };
  return instance;
}

struct ReconstructedClock {
  ClockRegistry registry;
  ClockRef default_clock;
};

[[nodiscard]] const ClockPlanRecord* find_clock(const ExecutionPlanState& plan,
                                                ClockId id) noexcept {
  if (!id.valid() || id.value > plan.clocks.size()) {
    return nullptr;
  }
  const ClockPlanRecord& record = plan.clocks[static_cast<std::size_t>(id.value - 1U)];
  return record.id == id ? &record : nullptr;
}

[[nodiscard]] expected<void, Error> validate_clock_import_inventory(
    const ExecutionPlanState& plan) {
  const auto authored_imports = plan.graph.clock_imports();
  const auto authored_bindings = plan.graph.clock_bindings();
  if (authored_imports.size() != authored_bindings.size() ||
      authored_imports.size() != plan.clock_import_resolutions.size()) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }
  std::vector<GraphId> defaults;
  defaults.reserve(plan.graph.clocks().size() + authored_imports.size());
  for (const auto& clock : plan.graph.clocks()) {
    if (clock.is_default) {
      defaults.push_back(clock.graph);
    }
  }
  for (const auto& clock_import : authored_imports) {
    if (clock_import.is_default) {
      defaults.push_back(clock_import.graph);
    }
  }
  std::ranges::sort(defaults, {}, &GraphId::value);
  if (std::ranges::adjacent_find(defaults) != defaults.end()) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }
  for (std::size_t index = 0U; index < authored_imports.size(); ++index) {
    const auto& authored_import = authored_imports[index];
    const auto matches_import = [&](const auto& candidate) {
      return candidate.child_graph == authored_import.graph &&
             candidate.child_clock == authored_import.key;
    };
    const auto binding =
        std::find_if(authored_bindings.begin(), authored_bindings.end(), matches_import);
    const auto binding_count = static_cast<std::size_t>(
        std::count_if(authored_bindings.begin(), authored_bindings.end(), matches_import));
    const ClockId resolved =
        resolve_authored_clock_id(plan, authored_import.graph, authored_import.key);
    if (binding_count != 1U || binding == authored_bindings.end() || !resolved.valid()) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
    const auto& frozen = plan.clock_import_resolutions[index];
    if (frozen.child_graph != authored_import.graph || frozen.child_clock != authored_import.key ||
        frozen.is_default != authored_import.is_default ||
        frozen.parent_graph != binding->parent_graph ||
        frozen.parent_clock != binding->parent_clock || frozen.resolved_clock != resolved) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_clock_inventory(const ExecutionPlanState& plan) {
  const auto authored_clocks = plan.graph.clocks();
  const auto authored_conversions = plan.graph.clock_conversions();
  if (plan.clocks.size() != authored_clocks.size() ||
      plan.executable_factories.clocks.size() != authored_clocks.size() ||
      plan.clock_conversions.size() != authored_conversions.size()) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }

  const auto graph_records = plan.graph.graphs();
  const auto root_graph = std::find_if(graph_records.begin(),
                                       graph_records.end(),
                                       [](const auto& graph) { return !graph.parent.has_value(); });
  if (root_graph == graph_records.end()) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }

  auto imports_valid = validate_clock_import_inventory(plan);
  if (!imports_valid) {
    return imports_valid;
  }

  ClockId expected_root_default;
  for (std::size_t index = 0U; index < authored_clocks.size(); ++index) {
    const auto& authored = authored_clocks[index];
    const auto& planned = plan.clocks[index];
    const auto& executable = plan.executable_factories.clocks[index];
    const bool has_factory = executable.factory != nullptr;
    const bool has_shared = executable.shared != nullptr;
    if (planned.id.value != index + 1U || planned.graph != authored.graph ||
        planned.key != authored.key || planned.shared != authored.shared ||
        planned.is_default != authored.is_default || executable.graph != authored.graph ||
        executable.key != authored.key || has_factory == has_shared ||
        planned.shared != has_shared) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
    if (authored.graph == root_graph->id && authored.is_default) {
      if (expected_root_default.valid()) {
        return make_unexpected(
            startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
      }
      expected_root_default = planned.id;
    }
  }
  if (plan.root_default_clock != expected_root_default) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }

  for (std::size_t index = 0U; index < authored_conversions.size(); ++index) {
    const auto& authored = authored_conversions[index];
    const auto& planned = plan.clock_conversions[index];
    const ClockId from = resolve_authored_clock_id(plan, authored.graph, authored.from);
    const ClockId to = resolve_authored_clock_id(plan, authored.graph, authored.to);
    if (planned.id != index + 1U || planned.plan_generation != plan.plan_generation ||
        planned.graph != authored.graph || planned.from != from || planned.to != to ||
        planned.scale != authored.conversion.scale ||
        planned.offset_ns != authored.conversion.offset_ns ||
        planned.rounding != ClockRoundingMode::kNearestTiesToEven ||
        planned.overflow != ClockOverflowPolicy::kError) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
  }
  return {};
}

[[nodiscard]] ClockKey runtime_clock_key(ClockId id) {
  // ClockRegistry assigns ids by sorted key. A fixed-width private key keeps
  // the immutable plan's dense numeric order even when independently embedded
  // Graphs reuse the same authored clock name.
  constexpr std::string_view kPrefix{"__holoscan_runtime_clock/"};
  std::array<char, 20U> digits{};
  const auto converted = std::to_chars(digits.data(), digits.data() + digits.size(), id.value);
  const std::size_t count = static_cast<std::size_t>(converted.ptr - digits.data());
  std::string name{kPrefix};
  name.append(digits.size() - count, '0');
  name.append(digits.data(), count);
  return ClockKey::named(name);
}

[[nodiscard]] expected<std::shared_ptr<Clock>, Error> reconstruct_clock(
    const ExecutableClockFactoryRecord& executable) {
  if (static_cast<bool>(executable.factory) == static_cast<bool>(executable.shared)) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_FACTORY_INVALID"));
  }
  if (executable.factory == nullptr) {
    return executable.shared;
  }
  auto created = executable.factory->create();
  if (!created) {
    return make_unexpected(
        startup_error(created.error().code, "RUNTIME_CLOCK_RECONSTRUCTION_FAILED"));
  }
  if (created.value() == nullptr) {
    return make_unexpected(
        startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_RECONSTRUCTION_FAILED"));
  }
  return std::shared_ptr<Clock>{std::move(created).value()};
}

// Reconstruct and authenticate the complete plan-local clock catalog as one cold-start unit.
// NOLINTBEGIN(readability-function-cognitive-complexity)
[[nodiscard]] expected<ReconstructedClock, Error> reconstruct_clocks(
    const ExecutionPlanState& plan) {
  if (plan.executable_factories.clocks.size() != plan.clocks.size()) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }
  if (plan.clocks.empty()) {
    if (plan.root_default_clock.valid() || !plan.clock_conversions.empty()) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
    return ReconstructedClock{};
  }

  const ClockPlanRecord* const root_default = find_clock(plan, plan.root_default_clock);
  if (root_default == nullptr || !root_default->is_default) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }

  ClockRegistryBuilder builder;
  std::vector<ClockKey> runtime_keys;
  runtime_keys.reserve(plan.clocks.size());
  for (std::size_t index = 0U; index < plan.clocks.size(); ++index) {
    const ClockPlanRecord& planned = plan.clocks[index];
    const ExecutableClockFactoryRecord& executable = plan.executable_factories.clocks[index];
    if (planned.id.value != index + 1U || planned.graph != executable.graph ||
        planned.key != executable.key || planned.shared != static_cast<bool>(executable.shared)) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
    auto clock = reconstruct_clock(executable);
    if (!clock) {
      return make_unexpected(std::move(clock.error()));
    }
    runtime_keys.push_back(runtime_clock_key(planned.id));
    auto added = builder.add_clock(runtime_keys.back(),
                                   std::move(clock).value(),
                                   ClockCapabilityOptions{
                                       .can_read_now = planned.can_read_now,
                                       .can_wait_until = planned.can_wait_until,
                                       .steady = planned.steady,
                                   });
    if (!added) {
      return make_unexpected(
          startup_error(added.error().code, "RUNTIME_CLOCK_RECONSTRUCTION_FAILED"));
    }
  }

  auto selected_default = builder.set_default(
      runtime_keys[static_cast<std::size_t>(plan.root_default_clock.value - 1U)]);
  if (!selected_default) {
    return make_unexpected(
        startup_error(selected_default.error().code, "RUNTIME_CLOCK_RECONSTRUCTION_FAILED"));
  }

  for (std::size_t index = 0U; index < plan.clock_conversions.size(); ++index) {
    const ClockConversionRecord& conversion = plan.clock_conversions[index];
    const ClockPlanRecord* const from = find_clock(plan, conversion.from);
    const ClockPlanRecord* const to = find_clock(plan, conversion.to);
    if (conversion.id != index + 1U || conversion.plan_generation != plan.plan_generation ||
        from == nullptr || to == nullptr ||
        conversion.rounding != ClockRoundingMode::kNearestTiesToEven ||
        conversion.overflow != ClockOverflowPolicy::kError) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
    auto added =
        builder.add_conversion(runtime_keys[static_cast<std::size_t>(conversion.from.value - 1U)],
                               runtime_keys[static_cast<std::size_t>(conversion.to.value - 1U)],
                               AffineClockConversion{
                                   .scale = conversion.scale,
                                   .offset_ns = conversion.offset_ns,
                               });
    if (!added) {
      return make_unexpected(
          startup_error(added.error().code, "RUNTIME_CLOCK_RECONSTRUCTION_FAILED"));
    }
  }

  auto registry = builder.build();
  if (!registry) {
    return make_unexpected(
        startup_error(registry.error().code, "RUNTIME_CLOCK_RECONSTRUCTION_FAILED"));
  }
  for (const ClockPlanRecord& planned : plan.clocks) {
    auto resolved =
        registry->resolve(runtime_keys[static_cast<std::size_t>(planned.id.value - 1U)]);
    if (!resolved || resolved->id() != planned.id) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
  }
  auto resolved_default = registry->resolve(ClockKey::default_clock());
  if (!resolved_default || resolved_default->id() != plan.root_default_clock) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }
  return ReconstructedClock{
      .registry = std::move(registry).value(),
      .default_clock = std::move(resolved_default).value(),
  };
}
// NOLINTEND(readability-function-cognitive-complexity)

}  // namespace

// Reconstruction validates every frozen inventory before publishing any run-owned state, then
// builds Operators, Parameter bindings, and clocks in dependency order. The linear transaction and
// its fail-closed exits are intentionally kept visible here so no partially reconstructed GraphExec
// can escape through a helper boundary.
// NOLINTBEGIN(readability-function-cognitive-complexity)
expected<std::unique_ptr<GraphExec>, Error> GraphExec::reconstruct(
    std::shared_ptr<const ExecutionPlanState> plan, RunGenerationId run_generation,
    ProcessGroupId process_group, RuntimeReplayMismatchDetail* replay_mismatch) {
  if (replay_mismatch != nullptr) {
    *replay_mismatch = RuntimeReplayMismatchDetail{};
  }
  if (!plan) {
    return make_unexpected(
        startup_error(ErrorCode::kInvalidArgument, "RUNTIME_EXECUTION_PLAN_EMPTY"));
  }
  if (run_generation == 0U) {
    return make_unexpected(
        startup_error(ErrorCode::kInvalidArgument, "RUNTIME_RUN_GENERATION_INVALID"));
  }
  if (process_group && std::none_of(plan->vertex_process_placements.begin(),
                                    plan->vertex_process_placements.end(),
                                    [&](const auto& placement) {
                                      return placement.process_group == process_group;
                                    })) {
    // This remains an Error-only defense in depth for callers below the RunSession boundary.
    // start_run_session() owns the registered, structured startup diagnostic for an invalid
    // explicit process group before executor construction can begin.
    return make_unexpected(
        startup_error(ErrorCode::kInvalidArgument, "RUNTIME_PROCESS_GROUP_SELECTION_INVALID"));
  }
  const auto planned_operator_count = static_cast<std::size_t>(std::count_if(
      plan->graph.vertices().begin(), plan->graph.vertices().end(), [](const auto& vertex) {
        return vertex.kind == VertexKind::kOperator;
      }));
  if (!lifecycle_plan_well_formed(*plan, planned_operator_count)) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_LIFECYCLE_PLAN_INVALID"));
  }

  // Resolve and match the executing package before reconstructing an Operator.
  // Operator reconstruction replays setup(), so performing this cold gate first
  // guarantees that no user callback observes an incompatible package.
  const RuntimePackageCapabilityManifest& generated = runtime_package_capability_manifest();
  const std::string canonical = runtime_package_capability_canonical_json_v1(generated, true);
  auto package = resolve_runtime_package_capability_provider(
      RuntimePackageCapabilityProviderInput{.embedded_bytes = canonical});
  if (!package) {
    return make_unexpected(std::move(package.error()));
  }
  auto matched = match_runtime_package_capabilities(plan->profile_manifest,
                                                    package.value(),
                                                    RuntimePackageCapabilityMatchContext{
                                                        .startup_attempt = 1U,
                                                        .plan_generation = plan->plan_generation,
                                                        .run_generation = run_generation,
                                                    });
  if (!matched) {
    return make_unexpected(std::move(matched.error()));
  }
  if (!matched->compatible()) {
    return make_unexpected(Error{
        ErrorCode::kIncompatible,
        std::string{runtime_package_capability_match_token(matched->result)},
    });
  }
  // Parameter owner-local inventories are consumed as contiguous spans during reconstruction.
  // Authenticate their structural shape before validating runtime backends so a malformed frozen
  // inventory cannot be masked by an unrelated unavailable readiness or placement binding.
  if (plan->operator_parameters.size() != planned_operator_count ||
      !plan->parameter_runtime_capacity.has_value() ||
      !parameter_owner_groups_well_formed(
          std::span{plan->parameter_fields}, planned_operator_count, plan->plan_generation) ||
      !parameter_owner_groups_well_formed(std::span{plan->parameter_derived_terms},
                                          planned_operator_count,
                                          plan->plan_generation)) {
    return make_unexpected(
        startup_error(ErrorCode::kFailure, "RUNTIME_PARAMETER_INVENTORY_MISMATCH"));
  }
  const ParameterRuntimeCapacityPlanRecord parameter_runtime_capacity =
      plan->parameter_runtime_capacity.value_or(ParameterRuntimeCapacityPlanRecord{});
  if (auto supported = validate_foundation_plan(*plan, planned_operator_count); !supported) {
    return make_unexpected(std::move(supported.error()));
  }
  if (plan->executable_factories.operators.size() != planned_operator_count) {
    return make_unexpected(
        startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_INVENTORY_MISMATCH"));
  }
  // Authenticate the compiler-derived partition runner only after the executable inventory has
  // retained its more specific diagnostic, but before reconstructing any Operator or invoking any
  // lifecycle callback. The runner record is a cold GraphExec foundation fact; Runtime must never
  // rebuild or silently repair a mutated copy.
  if (!valid_partition_runner_plan(*plan)) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
  }
  auto clock_inventory = validate_clock_inventory(*plan);
  if (!clock_inventory) {
    return make_unexpected(std::move(clock_inventory.error()));
  }

  auto result = std::unique_ptr<GraphExec>{new GraphExec{std::move(plan), process_group}};
  result->package_capability_report_ = std::move(matched).value();

  // A compiler-accepted empty Graph still passes exact package compatibility
  // and frozen-plan validation. Once those cold gates succeed, it constructs
  // no package-backed Operator or clock that the run can never observe.
  if (result->plan_->graph.vertices().empty()) {
    auto parameter_updates = ParameterUpdateCoordinator::create(
        std::span<const ParameterRuntimeOperatorBinding>{}, parameter_runtime_capacity);
    if (!parameter_updates) {
      return make_unexpected(
          startup_error(parameter_updates.error().code, "RUNTIME_PARAMETER_BINDING_UNAVAILABLE"));
    }
    result->parameter_updates_ = std::move(parameter_updates).value();
    return result;
  }

  result->operators_.reserve(result->plan_->executable_factories.operators.size());
  for (std::size_t index = 0U; index < result->plan_->executable_factories.operators.size();
       ++index) {
    const auto& record = result->plan_->executable_factories.operators[index];
    if (process_group && !worker_owns_vertex(*result->plan_, process_group, record.vertex)) {
      continue;
    }
    auto reconstructed = reconstruct_operator(
        *result->plan_, record, static_cast<std::uint32_t>(index + 1U), replay_mismatch);
    if (!reconstructed) {
      return make_unexpected(std::move(reconstructed.error()));
    }
    result->operators_.push_back(std::move(reconstructed).value());
  }
  std::vector<ParameterRuntimeOperatorBinding> parameter_bindings;
  parameter_bindings.reserve(result->operators_.size());
  for (RuntimeOperatorInstance& instance : result->operators_) {
    const OperatorParameterPlanRecord* const parameter_plan =
        planned_parameters(*result->plan_, instance.operator_id);
    if (parameter_plan == nullptr) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_PARAMETER_INVENTORY_MISMATCH"));
    }
    const ParamTypeDescriptor& parameter_type = OperatorRuntimeAccess::parameter_type(*instance.op);
    instance.static_parameter_header = PinnedParameterHeader{
        .value = OperatorRuntimeAccess::parameter_address(*instance.op),
        .local_type_token = parameter_type.local_type_token,
        .generation = 0U,
        .schema_hash = parameter_plan->contract_hash,
        .slot = 0U,
    };
    instance.parameter_derived_terms =
        planned_parameter_derived_terms(*result->plan_, instance.operator_id);
    instance.runtime_parameter_updates_enabled = parameter_plan->runtime_updates_enabled;
    instance.lifecycle_parameter_updates_enabled = parameter_plan->lifecycle_updates_enabled;
    auto operator_path = planned_operator_path(*result->plan_, instance.vertex);
    if (!operator_path) {
      return make_unexpected(std::move(operator_path.error()));
    }
    parameter_bindings.push_back(ParameterRuntimeOperatorBinding{
        .run_bound_owner =
            OperatorRef{
                .id = instance.operator_id,
                .plan_generation = result->plan_->plan_generation,
                .run_generation = run_generation,
            },
        .operator_path = std::move(operator_path).value(),
        .plan_digest = result->plan_->digest,
        .plan = parameter_plan,
        .fields = planned_parameter_fields(*result->plan_, instance.operator_id),
        .derived_terms = instance.parameter_derived_terms,
        .type = &parameter_type,
        .schema = OperatorRuntimeAccess::parameter_schema(*instance.op),
        .initial_value = OperatorRuntimeAccess::parameter_address(*instance.op),
    });
  }
  auto parameter_updates = ParameterUpdateCoordinator::create(
      parameter_bindings,
      parameter_runtime_capacity,
      static_cast<std::uint32_t>(result->plan_->operator_parameters.size()));
  if (!parameter_updates) {
    return make_unexpected(
        startup_error(parameter_updates.error().code, "RUNTIME_PARAMETER_BINDING_UNAVAILABLE"));
  }
  result->parameter_updates_ = std::move(parameter_updates).value();

  auto clock = reconstruct_clocks(*result->plan_);
  if (!clock) {
    return make_unexpected(std::move(clock.error()));
  }
  result->clock_registry_ = std::move(clock->registry);
  result->default_clock_ = std::move(clock->default_clock);
  return result;
}
// NOLINTEND(readability-function-cognitive-complexity)

GraphExec::~GraphExec() = default;

GraphExec::GraphExec(std::shared_ptr<const ExecutionPlanState> plan, ProcessGroupId process_group)
    : plan_(std::move(plan)), process_group_(process_group) {}

expected<ClockRef, Error> GraphExec::resolve_clock(ClockId id) const noexcept {
  return clock_registry_.resolve(id);
}

expected<void, Error> GraphExec::wait_until(ClockId id, std::int64_t timestamp_ns) const noexcept {
  return clock_registry_.wait_until(id, timestamp_ns);
}

expected<void, Error> GraphExec::initialize_lifecycle(
    RunGenerationId run_generation, const std::atomic_bool* cancellation, ExecutionLaneSet* lanes,
    RuntimeNotificationSourceRegistry* notifications) {
  if (!package_capability_report_ || !package_capability_report_->compatible() ||
      package_capability_report_->context.run_generation != run_generation) {
    return make_unexpected(
        startup_error(ErrorCode::kFailure, "RUNTIME_PACKAGE_CAPABILITY_STATE_INVALID"));
  }

  auto lifecycle = RuntimeLifecycleExecutor::create(*plan_,
                                                    operators_,
                                                    run_generation,
                                                    cancellation,
                                                    this,
                                                    lanes,
                                                    *package_capability_report_,
                                                    notifications);
  if (!lifecycle) {
    return make_unexpected(std::move(lifecycle.error()));
  }
  lifecycle_ = std::move(lifecycle).value();
  return {};
}

bool GraphExec::lifecycle_startup() noexcept {
  return lifecycle_ != nullptr && lifecycle_->startup();
}

LifecycleRestartExecutionOutcome GraphExec::lifecycle_restart(
    const LifecycleRestartExecutionRequest& request) noexcept {
  return lifecycle_ == nullptr ? LifecycleRestartExecutionOutcome{} : lifecycle_->restart(request);
}

bool GraphExec::lifecycle_shutdown() noexcept {
  return lifecycle_ == nullptr || lifecycle_->shutdown();
}

std::string_view GraphExec::lifecycle_failure_token() const noexcept {
  return lifecycle_ == nullptr ? std::string_view{"RUNTIME_LIFECYCLE_CALLBACK_FAILED"}
                               : lifecycle_->failure_token();
}

}  // namespace holoscan::detail
