/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/physical_topology_lowering.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "compiler/detail/checked_arithmetic.hpp"
#include "compiler/execution_plan_contracts.hpp"
#include "compiler/physical_plan_lowering_internal.hpp"
#include "compiler/process_plan_records.hpp"
#include "compiler/rate_plan_lowering.hpp"
#include "compiler/snapshot_lookup.hpp"
#include "core/domain/tensor_layout_validation.hpp"

namespace holoscan::detail {
namespace {

constexpr std::string_view kPortInvalid{"PHYSICAL_PORT_CONTRACT_INVALID"};
constexpr std::string_view kEdgeInvalid{"PHYSICAL_EDGE_CONTRACT_INVALID"};
constexpr std::string_view kBindingInvalid{"PHYSICAL_INTERFACE_BINDING_INVALID"};
constexpr std::string_view kTensorConversionUnsupported{"PHYSICAL_TENSOR_CONVERSION_UNSUPPORTED"};
constexpr std::string_view kTensorDeviceBindingUnexpected{
    "PHYSICAL_TENSOR_DEVICE_BINDING_UNEXPECTED"};
constexpr std::string_view kDeviceUnresolved{"IMPLICIT_DEVICE_ZERO_REJECTED"};
constexpr std::string_view kTensorPoolBoundsInvalid{"PHYSICAL_TENSOR_ALLOCATION_BOUNDS_INVALID"};

[[nodiscard]] bool valid_payload(const PayloadIdentityRecord& payload) noexcept {
  if (!enum_at_most(payload.kind, PayloadIdentityKind::kPod) ||
      !enum_at_most(payload.payload_kind, PayloadKind::kControlEmpty)) {
    return false;
  }
  switch (payload.kind) {
    case PayloadIdentityKind::kScalar:
      return payload.scalar && !payload.schema && !payload.pod &&
             enum_at_most(payload.scalar->kind, ScalarIdentityKind::kLocalEnum) &&
             (payload.scalar->kind == ScalarIdentityKind::kLocalEnum
                  ? payload.scalar->native_object_size == 1U ||
                        payload.scalar->native_object_size == 2U ||
                        payload.scalar->native_object_size == 4U ||
                        payload.scalar->native_object_size == 8U
                  : payload.scalar->native_object_size == 0U);
    case PayloadIdentityKind::kTensor:
      return !payload.scalar && !payload.schema && !payload.pod &&
             payload.payload_kind == PayloadKind::kDescriptorBacked;
    case PayloadIdentityKind::kSchema:
      return !payload.scalar && payload.schema && !payload.pod &&
             valid_schema_identity(*payload.schema);
    case PayloadIdentityKind::kPod:
      return !payload.scalar && !payload.schema && payload.pod && payload.pod->object_size != 0U &&
             payload.pod->object_alignment != 0U && !payload.pod->fields.empty();
  }
  return false;
}

[[nodiscard]] std::uint32_t resolved_queue(const PortRecord& input) noexcept {
  return input.graph_override.queue_depth.value_or(input.setup.queue_depth.value_or(1U));
}

[[nodiscard]] std::uint32_t resolved_burst(const PortRecord& output) noexcept {
  return output.graph_override.max_emits_per_compute.value_or(
      output.setup.max_emits_per_compute.value_or(1U));
}

[[nodiscard]] std::uint32_t resolved_history(const PortRecord& output) noexcept {
  return output.graph_override.late_joiner_history_depth.value_or(
      output.setup.late_joiner_history_depth.value_or(0U));
}

[[nodiscard]] MemoryConversionPolicy resolved_memory_conversion(const EdgeRecord& edge,
                                                                const PortRecord& target) noexcept {
  return edge.graph_override.memory_conversion.value_or(
      edge.add_flow_options.memory_conversion.value_or(
          target.graph_override.memory_conversion.value_or(
              target.setup.memory_conversion.value_or(MemoryConversionPolicy::kDefault))));
}

[[nodiscard]] std::uint32_t cache_for(PortId port,
                                      std::span<const InputSelectionPlanRecord> selections,
                                      std::span<const BracketedWindowPlanRecord> windows) noexcept {
  std::uint32_t result = 0U;
  for (const auto& selection : selections) {
    if (selection.input == port) {
      result = std::max(result, selection.cache_capacity);
    }
  }
  for (const auto& window : windows) {
    if (window.input == port) {
      result = std::max(result, window.cache_capacity);
    }
  }
  return result;
}

[[nodiscard]] bool validate_tensor_layout(
    const std::optional<TensorRepresentation>& layout) noexcept {
  return !layout ||
         ((!layout->memory_kind || enum_at_most(*layout->memory_kind, MemoryKind::kCudaManaged)) &&
          (!layout->dtype || valid_tensor_dtype(*layout->dtype)) &&
          (!layout->strides ||
           enum_at_most(*layout->strides, TensorStridePolicy::kBoundedStrided)));
}

[[nodiscard]] const ChannelEndpointRecord* find_admission_compatible_catalog_endpoint(
    std::span<const ChannelEndpointRecord> endpoints, const ExternalTopicRecord& topic) noexcept {
  const TopicDirection direction =
      topic.imported ? TopicDirection::kImport : TopicDirection::kExport;
  const auto found = std::find_if(endpoints.begin(), endpoints.end(), [&](const auto& endpoint) {
    return endpoint.boundary_port.value() == topic.graph_port.value && endpoint.uri == topic.uri &&
           endpoint.direction == direction &&
           payload_identity_exactly_equal(endpoint.payload_identity, topic.payload_identity) &&
           endpoint.schema_writer_max_serialized_bytes == topic.serialized_payload_capacity_bytes;
  });
  if (found == endpoints.end() || found->service.empty() ||
      !enum_at_most(found->attach.peer, PeerRequirement::kRequired)) {
    return nullptr;
  }
  const auto& max_wait = found->attach.max_wait;
  if (max_wait.has_value() && max_wait.value().count() <= 0) {
    return nullptr;
  }
  return &*found;
}

}  // namespace

std::optional<DeviceId> resolve_tensor_device(PhysicalPlanLoweringResult& result,
                                              const GraphSnapshotData& snapshot,
                                              PhysicalTopologyFacts& topology, PortId port,
                                              MemoryKind memory, DiagnosticSource source,
                                              const bool force_explicit_device) {
  const auto found = topology.device_by_port.find(port.value());
  if (!requires_explicit_device(memory) && !force_explicit_device) {
    if (found != topology.device_by_port.end()) {
      topology.consumed_device_bindings.insert(port.value());
      reject_physical_plan(result,
                           kTensorDeviceBindingUnexpected,
                           source,
                           "/ports/" + std::to_string(port.value()) + "/tensor_device_binding",
                           "absent for non-device storage",
                           "device=" + std::to_string(found->second.value));
      return std::nullopt;
    }
    return DeviceId{};
  }
  if (found == topology.device_by_port.end()) {
    std::string provided{"device_binding=missing"};
    const auto* const port_record = find_port(snapshot, GraphPortId{port.value()});
    if (port_record != nullptr) {
      const auto* const owner = find_vertex(snapshot, port_record->owner);
      if (owner != nullptr) {
        const auto operator_path = canonical_operator_path(snapshot, *owner);
        provided = "operator_path=" + (operator_path ? *operator_path : std::string{"<invalid>"}) +
                   ",output_port=" + port_record->name + "," + provided;
      }
    }
    reject_physical_plan(
        result,
        kDeviceUnresolved,
        source,
        port_pointer(port, "/tensor_device_binding"),
        "CompileOptions::deployment.bind_tensor_output_device with a nonnegative device",
        std::move(provided));
    return std::nullopt;
  }
  topology.consumed_device_bindings.insert(port.value());
  return found->second;
}

ProcessLocalChannelDeliveryKind physical_process_local_delivery_kind(
    const std::set<std::pair<std::uint64_t, std::uint64_t>>& holomq_activation_pairs,
    const PayloadEdgePlanRecord& edge, const PortRecord& source, const PortRecord& target,
    const bool force_holomq_process_local_inline) noexcept {
  return force_holomq_process_local_inline ||
                 holomq_activation_pairs.contains(std::pair{source.owner.value, target.owner.value})
             ? ProcessLocalChannelDeliveryKind::kHoloMQ
             : process_local_channel_delivery_kind(edge);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] PhysicalTopologyFacts lower_physical_topology(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    std::span<const std::optional<Profile>> effective_profiles,
    std::span<const InputSelectionPlanRecord> input_selections,
    std::span<const BracketedWindowPlanRecord> bracketed_windows,
    std::span<const ChannelEndpointRecord> catalog_endpoints,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    std::span<const TensorDeviceBindingRecord> tensor_device_bindings,
    bool force_holomq_process_local_inline, bool force_holomq_process_local_event_wake,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations,
    std::span<const ResolvedQueueSizing> resolved_queue_sizings) {
  result.ports.reserve(snapshot.ports.size());
  std::vector<std::uint32_t> fanout(snapshot.ports.size(), 0U);
  std::unordered_map<std::uint64_t, const ExternalTopicRecord*> imported_by_port;
  std::unordered_map<std::uint64_t, const ExternalTopicRecord*> exported_by_port;
  std::unordered_map<std::uint64_t, const ChannelEndpointRecord*> endpoint_by_port;
  std::unordered_map<std::uint64_t, DeviceId> device_by_port;
  std::unordered_set<std::uint64_t> consumed_device_bindings;
  const auto queue_for_edge = [&](const EdgeRecord& edge, const PortRecord& target) {
    if (const auto* sizing = find_resolved_queue_sizing(resolved_queue_sizings, edge.id);
        sizing != nullptr) {
      // Lifecycle-selectable rates with a finite maximum materialize the admitted ceiling.
      // The selected generation installs its smaller active logical depth before kAllocate.
      return sizing->explicit_depth
                 ? sizing->effective_logical_depth
                 : sizing->admitted_depth_ceiling.value_or(sizing->effective_logical_depth);
    }
    return edge.graph_override.queue_depth.value_or(
        edge.add_flow_options.queue_depth.value_or(resolved_queue(target)));
  };
  const auto queue_for_input = [&](const PortRecord& input) {
    if (!resolved_queue_sizings.empty()) {
      const auto edge =
          std::find_if(snapshot.edges.begin(), snapshot.edges.end(), [&](const auto& item) {
            return resolve_interface_port(snapshot, item.target) == input.id;
          });
      if (edge != snapshot.edges.end()) {
        return queue_for_edge(*edge, input);
      }
    }
    return resolved_queue(input);
  };
  for (std::size_t index = 0; index < tensor_device_bindings.size(); ++index) {
    const auto& binding = tensor_device_bindings[index];
    const auto* port =
        binding.port ? find_port(snapshot, GraphPortId{binding.port.value()}) : nullptr;
    std::string provided;
    if (!binding.port) {
      provided = "port=0,reason=invalid_port";
    } else if (port == nullptr) {
      provided = "port=" + std::to_string(binding.port.value()) + ",reason=unknown_port";
    } else if (port->payload_identity.kind != PayloadIdentityKind::kTensor &&
               !(port->payload_identity.kind == PayloadIdentityKind::kSchema &&
                 port->payload_identity.payload_kind == PayloadKind::kDescriptorBacked)) {
      provided = "port=" + std::to_string(binding.port.value()) + ",reason=non_tensor_port";
    } else if (!binding.device.valid()) {
      provided = "port=" + std::to_string(binding.port.value()) +
                 ",device=" + std::to_string(binding.device.value) + ",reason=invalid_device";
    } else if (device_by_port.contains(binding.port.value())) {
      provided = "port=" + std::to_string(binding.port.value()) +
                 ",device=" + std::to_string(binding.device.value) + ",reason=duplicate_port";
    } else {
      device_by_port.emplace(binding.port.value(), binding.device);
    }
    if (!provided.empty()) {
      reject_physical_plan(
          result,
          kTensorDeviceBindingInvalid,
          source_for(snapshot, port == nullptr ? VertexId{} : port->owner, binding.port),
          "/tensor_device_bindings/" + std::to_string(index),
          "one valid Tensor port with one nonnegative device binding",
          std::move(provided));
    }
  }
  for (const auto& topic : snapshot.external_topics) {
    auto& index = topic.imported ? imported_by_port : exported_by_port;
    const auto* port = topic.graph_port ? find_port(snapshot, topic.graph_port) : nullptr;
    const auto* endpoint = find_admission_compatible_catalog_endpoint(catalog_endpoints, topic);
    if (!topic.id || !topic.vertex || !topic.graph_port || port == nullptr ||
        !valid_payload(topic.payload_identity) || endpoint == nullptr ||
        index.contains(topic.graph_port.value) ||
        endpoint_by_port.contains(topic.graph_port.value)) {
      reject_physical_plan(result,
                           kServiceInvalid,
                           source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}));
      continue;
    }
    index.emplace(topic.graph_port.value, &topic);
    endpoint_by_port.emplace(topic.graph_port.value, endpoint);
  }
  for (const auto& edge : snapshot.edges) {
    const GraphPortId source = resolve_interface_port(snapshot, edge.source);
    if (source && source.value <= fanout.size()) {
      ++fanout[static_cast<std::size_t>(source.value - 1U)];
    }
  }

  for (std::size_t index = 0; index < snapshot.ports.size(); ++index) {
    const auto& port = snapshot.ports[index];
    const PortId plan_id{index + 1U};
    const auto layout = resolved_layout(port);
    const std::uint32_t queue =
        port.direction == PortDirection::kInput ? queue_for_input(port) : 0U;
    const std::uint32_t burst =
        port.direction == PortDirection::kOutput ? resolved_burst(port) : 0U;
    const bool schema = port.payload_identity.kind == PayloadIdentityKind::kSchema;
    if (port.id.value != index + 1U || !port.owner || port.owner.value > snapshot.vertices.size() ||
        !enum_at_most(port.owner_kind, PortOwnerKind::kExportedTopic) ||
        !enum_at_most(port.direction, PortDirection::kOutput) || port.name.empty() ||
        !valid_payload(port.payload_identity) ||
        schema != (port.serialized_payload_capacity_bytes != 0U) ||
        port.generic_loan_supported != (port.generic_loan_slot_size_bytes != 0U) ||
        (port.generic_loan_supported &&
         (port.owner_kind != PortOwnerKind::kOperator || port.direction != PortDirection::kOutput ||
          port.payload_identity.kind == PayloadIdentityKind::kTensor)) ||
        !validate_tensor_layout(layout) ||
        (port.direction == PortDirection::kInput && queue == 0U) ||
        (port.direction == PortDirection::kOutput && burst == 0U)) {
      reject_physical_plan(result, kPortInvalid, source_for(snapshot, port.owner, plan_id));
    }
    auto tensor_contract = resolved_tensor_contract(port);
    if (const auto* derivation =
            find_port_tensor_byte_span_derivation(tensor_byte_span_derivations, port);
        derivation != nullptr) {
      if (!tensor_contract || tensor_contract->max_byte_span ||
          tensor_contract->storage != TensorOutputStorage::kRuntimePool) {
        reject_physical_plan(
            result, kTensorPoolBoundsInvalid, source_for(snapshot, port.owner, plan_id));
      } else {
        tensor_contract->max_byte_span = derivation->resolved_max_byte_span;
        tensor_contract->origins.max_byte_span =
            TensorContractValueOrigin::kCompilerDerivedByteSpan;
        if (tensor_contract->transfer && derivation->resolved_max_staging_byte_span) {
          if (tensor_contract->transfer->max_staging_byte_span &&
              tensor_contract->transfer->max_staging_byte_span !=
                  derivation->resolved_max_staging_byte_span) {
            reject_physical_plan(
                result, kTensorPoolBoundsInvalid, source_for(snapshot, port.owner, plan_id));
          } else if (!tensor_contract->transfer->max_staging_byte_span) {
            tensor_contract->transfer->max_staging_byte_span =
                derivation->resolved_max_staging_byte_span;
            tensor_contract->origins.max_staging_byte_span =
                TensorContractValueOrigin::kCompilerDerivedByteSpan;
          }
        }
      }
    }
    result.ports.push_back(PortPlanRecord{
        .id = plan_id,
        .owner = port.owner,
        .owner_kind = port.owner_kind,
        .direction = port.direction,
        .optional = port.optional,
        .local_name = port.name,
        .payload_identity = port.payload_identity,
        .serialized_payload_capacity_bytes = port.serialized_payload_capacity_bytes,
        .generic_loan_supported = port.generic_loan_supported,
        .generic_loan_slot_size_bytes = port.generic_loan_slot_size_bytes,
        .queue_depth = queue,
        .max_emits_per_compute = burst,
        .late_joiner_history_depth =
            port.direction == PortDirection::kOutput ? resolved_history(port) : 0U,
        .tensor_layout = layout,
        .tensor_contract = std::move(tensor_contract),
        .effective_profile = effective_profile_for_vertex(effective_profiles, port.owner),
    });
  }

  for (const auto& binding : snapshot.interface_bindings) {
    const auto* public_port = find_port(snapshot, binding.public_port);
    const auto* internal_port = find_port(snapshot, binding.internal_port);
    if (public_port == nullptr || internal_port == nullptr ||
        public_port->direction != binding.direction ||
        internal_port->direction != binding.direction ||
        public_port->owner_kind != PortOwnerKind::kGraphInterface ||
        !payload_identity_exactly_equal(public_port->payload_identity,
                                        internal_port->payload_identity)) {
      reject_physical_plan(result,
                           kBindingInvalid,
                           source_for(snapshot,
                                      public_port != nullptr ? public_port->owner : VertexId{},
                                      PortId{binding.public_port.value}));
      continue;
    }
    result.interface_bindings.push_back(InterfacePortBindingPlanRecord{
        .public_port = PortId{binding.public_port.value},
        .internal_port = PortId{binding.internal_port.value},
        .direction = binding.direction,
    });
  }

  std::vector<std::uint32_t> delivery_by_output(snapshot.ports.size(), 0U);
  std::vector<std::uint32_t> transport_retention_by_output(snapshot.ports.size(), 0U);
  std::unordered_set<std::uint64_t> transport_retention_edges;
  std::vector<std::uint32_t> cache_by_output(snapshot.ports.size(), 0U);
  std::vector<std::uint32_t> window_by_output(snapshot.ports.size(), 0U);
  std::vector<bool> reuse_by_output(snapshot.ports.size(), false);
  std::vector<bool> drain_by_output(snapshot.ports.size(), false);
  std::vector<bool> copy_by_output(snapshot.ports.size(), false);
  std::vector<bool> reexport_by_output(snapshot.ports.size(), false);
  for (const auto& edge : snapshot.edges) {
    const GraphPortId source_id = resolve_interface_port(snapshot, edge.source);
    const GraphPortId target_id = resolve_interface_port(snapshot, edge.target);
    const auto* source = find_port(snapshot, source_id);
    const auto* target = find_port(snapshot, target_id);
    if (!source_id || !target_id || source == nullptr || target == nullptr ||
        source->direction != PortDirection::kOutput || target->direction != PortDirection::kInput ||
        !payload_identity_exactly_equal(source->payload_identity, target->payload_identity) ||
        !enum_at_most(edge.add_flow_options.queue_policy, QueuePolicy::kDropOldest) ||
        (edge.graph_override.queue_policy &&
         !enum_at_most(*edge.graph_override.queue_policy, QueuePolicy::kDropOldest))) {
      reject_physical_plan(
          result,
          kEdgeInvalid,
          source_for(
              snapshot, source != nullptr ? source->owner : VertexId{}, PortId{edge.source.value}),
          "/edges/" + std::to_string(edge.id.value));
      result.errors.back().detail = physical_edge_detail(snapshot, target, source);
      result.errors.back().detail->error.message =
          source == nullptr || target == nullptr ? "connection references a missing port"
          : source->direction != PortDirection::kOutput ||
                  target->direction != PortDirection::kInput
              ? "connection requires an output producer and an input consumer"
          : !payload_identity_exactly_equal(source->payload_identity, target->payload_identity)
              ? "producer and consumer payload identities differ"
              : "connection has an invalid queue policy";
      continue;
    }
    if (!effective_profile_for_vertex(effective_profiles, source->owner) ||
        !effective_profile_for_vertex(effective_profiles, target->owner)) {
      // Profile admission already rejected the unresolved owner. Keep the authored Graph edge for
      // diagnostics, but do not claim an accepted physical edge that has no realizable service.
      continue;
    }
    const QueuePolicy policy =
        edge.graph_override.queue_policy.value_or(edge.add_flow_options.queue_policy);
    const std::uint32_t queue = queue_for_edge(edge, *target);
    const std::uint32_t cache =
        cache_for(PortId{edge.target.value}, input_selections, bracketed_windows);
    const std::uint32_t burst = resolved_burst(*source);
    const std::uint32_t derived_minimum = std::max({queue, cache, burst});
    const std::uint32_t maximum = edge.graph_override.max_inflight_samples.value_or(
        edge.add_flow_options.max_inflight_samples.value_or(derived_minimum));
    if (queue == 0U || maximum == 0U || maximum < derived_minimum || burst == 0U) {
      reject_physical_plan(
          result, kQueueInvalid, source_for(snapshot, target->owner, PortId{edge.target.value}));
      continue;
    }
    // An external endpoint rejected by an earlier lowering phase cannot be reclassified as an
    // ordinary process-local edge merely because it is absent from the admitted endpoint index.
    if ((source->owner_kind == PortOwnerKind::kImportedTopic &&
         !imported_by_port.contains(edge.source.value)) ||
        (target->owner_kind == PortOwnerKind::kExportedTopic &&
         !exported_by_port.contains(edge.target.value))) {
      continue;
    }
    ChannelLocalityKind locality = ChannelLocalityKind::kProcessLocal;
    if (imported_by_port.contains(edge.source.value)) {
      locality = ChannelLocalityKind::kExternalImport;
    } else if (exported_by_port.contains(edge.target.value)) {
      locality = ChannelLocalityKind::kExternalExport;
    }
    const auto source_group = process_group_for(process_placements, source->owner);
    const auto target_group = process_group_for(process_placements, target->owner);
    if (source->payload_identity.kind == PayloadIdentityKind::kTensor &&
        locality == ChannelLocalityKind::kProcessLocal && source_group && target_group &&
        *source_group == *target_group &&
        resolved_memory_conversion(edge, *target) != MemoryConversionPolicy::kDefault) {
      // Converter insertion is not implemented. Reject explicit policy instead of discarding it.
      reject_physical_plan(result,
                           kTensorConversionUnsupported,
                           source_for(snapshot, target->owner, PortId{target_id.value}));
      continue;
    }
    result.payload_edges.push_back(PayloadEdgePlanRecord{
        .edge = edge.id,
        .source = PortId{source_id.value},
        .target = PortId{target_id.value},
        .payload_identity = source->payload_identity,
        .queue_policy = policy,
        .queue_depth = queue,
        .max_inflight_samples = maximum,
        .producer_max_emits_per_compute = burst,
        .fanout = fanout[static_cast<std::size_t>(source_id.value - 1U)],
        .cache_capacity = cache,
        .locality = locality,
        .disposition = PlanDisposition::kAccepted,
    });
    result.channel_locality.push_back(ChannelLocalityRecord{
        .edge = edge.id,
        .requested = locality,
        .accepted = locality,
        .requires_runtime_attach = locality != ChannelLocalityKind::kProcessLocal,
    });
    const std::uint32_t delivery_queue = queue;
    std::uint32_t next = 0U;
    if (!checked_add(delivery_by_output[static_cast<std::size_t>(source_id.value - 1U)],
                     delivery_queue,
                     next)) {
      reject_physical_plan(
          result, kQueueInvalid, source_for(snapshot, source->owner, PortId{source_id.value}));
    } else {
      delivery_by_output[static_cast<std::size_t>(source_id.value - 1U)] = next;
    }
    if (!checked_add(
            cache_by_output[static_cast<std::size_t>(source_id.value - 1U)], cache, next)) {
      reject_physical_plan(
          result, kQueueInvalid, source_for(snapshot, source->owner, PortId{source_id.value}));
    } else {
      cache_by_output[static_cast<std::size_t>(source_id.value - 1U)] = next;
    }
    const auto output_index = static_cast<std::size_t>(source_id.value - 1U);
    for (const auto& selection : input_selections) {
      if (selection.input != PortId{target_id.value}) {
        continue;
      }
      reuse_by_output[output_index] = reuse_by_output[output_index] ||
                                      selection.missing == TemporalRecordMissingKind::kReuseLast;
      // The depth-one freshness path preserves ReuseLast beside the ordinary cache row.
      if (policy == QueuePolicy::kDropOldest &&
          selection.selector == InputSelectorKind::kLatestWithin &&
          selection.cache_capacity == 1U &&
          selection.missing == TemporalRecordMissingKind::kReuseLast) {
        if (!checked_add(cache_by_output[output_index], 1U, next)) {
          reject_physical_plan(
              result, kQueueInvalid, source_for(snapshot, source->owner, PortId{source_id.value}));
        } else {
          cache_by_output[output_index] = next;
        }
      }
      drain_by_output[output_index] =
          drain_by_output[output_index] || selection.selector == InputSelectorKind::kDrainAll;
    }
    for (const auto& window : bracketed_windows) {
      if (window.input == PortId{target_id.value}) {
        window_by_output[output_index] =
            std::max(window_by_output[output_index], window.retained_overlap_capacity);
      }
    }
    copy_by_output[output_index] = copy_by_output[output_index] ||
                                   (edge.effective_tensor_payload &&
                                    edge.effective_tensor_payload->interface_tensor_emit_policy !=
                                        InterfaceTensorEmitPolicy::kRequireExportable);
    reexport_by_output[output_index] =
        reexport_by_output[output_index] || exported_by_port.contains(edge.target.value);
  }

  // One callback can publish sibling samples that participate in the same downstream temporal
  // decision. Mixing direct retention with HoloMQ would order those samples by materialization
  // time rather than the callback's publication order. Keep each producer/consumer Operator pair
  // on HoloMQ whenever any edge in that pair requires the general transport path.
  std::set<std::pair<std::uint64_t, std::uint64_t>> holomq_activation_pairs;
  for (const auto& edge : result.payload_edges) {
    if (edge.locality != ChannelLocalityKind::kProcessLocal ||
        process_local_channel_delivery_kind(edge) != ProcessLocalChannelDeliveryKind::kHoloMQ) {
      continue;
    }
    const auto* source = find_port(snapshot, GraphPortId{edge.source.value()});
    const auto* target = find_port(snapshot, GraphPortId{edge.target.value()});
    if (source != nullptr && target != nullptr && source->owner_kind == PortOwnerKind::kOperator &&
        target->owner_kind == PortOwnerKind::kOperator) {
      holomq_activation_pairs.emplace(source->owner.value, target->owner.value);
    }
  }
  for (const auto& edge : result.payload_edges) {
    const bool tensor_backed = is_tensor_backed_payload(edge.payload_identity);
    const auto* source = find_port(snapshot, GraphPortId{edge.source.value()});
    const auto* target = find_port(snapshot, GraphPortId{edge.target.value()});
    if (!tensor_backed || source == nullptr || target == nullptr) {
      continue;
    }
    const auto source_group = process_group_for(process_placements, source->owner);
    const auto target_group = process_group_for(process_placements, target->owner);
    const bool cross_process_edge = source_group && target_group && *source_group != *target_group;
    if (!cross_process_edge &&
        physical_process_local_delivery_kind(
            holomq_activation_pairs, edge, *source, *target, force_holomq_process_local_inline) ==
            ProcessLocalChannelDeliveryKind::kDirectRetained &&
        edge.queue_policy == QueuePolicy::kBlocking) {
      continue;
    }
    // HoloMQ and lossy direct ingress retain an unread queue position independently from a typed
    // Tensor that can keep the prior sample's backing afterward. These are simultaneous live sets.
    auto& transport_retention =
        transport_retention_by_output[static_cast<std::size_t>(edge.source.value() - 1U)];
    std::uint32_t next = 0U;
    if (!checked_add(transport_retention, edge.queue_depth, next)) {
      reject_physical_plan(
          result, kTensorOverflow, source_for(snapshot, source->owner, edge.source));
      continue;
    }
    transport_retention = next;
    transport_retention_edges.insert(edge.edge.value);
  }

  return PhysicalTopologyFacts{
      .imported_by_port = std::move(imported_by_port),
      .exported_by_port = std::move(exported_by_port),
      .endpoint_by_port = std::move(endpoint_by_port),
      .device_by_port = std::move(device_by_port),
      .consumed_device_bindings = std::move(consumed_device_bindings),
      .delivery_by_output = std::move(delivery_by_output),
      .transport_retention_by_output = std::move(transport_retention_by_output),
      .transport_retention_edges = std::move(transport_retention_edges),
      .cache_by_output = std::move(cache_by_output),
      .window_by_output = std::move(window_by_output),
      .reuse_by_output = std::move(reuse_by_output),
      .drain_by_output = std::move(drain_by_output),
      .copy_by_output = std::move(copy_by_output),
      .reexport_by_output = std::move(reexport_by_output),
      .holomq_activation_pairs = std::move(holomq_activation_pairs),
  };
}

}  // namespace holoscan::detail
