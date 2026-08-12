/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/process_local_channel_plan_validation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include <holoscan/core/tensor_schema_output_loan.hpp>
#include <holoscan/schema/schema_package.hpp>

#include "channel/channel_payload_limits.hpp"
#include "compiler/execution_plan_state.hpp"

namespace holoscan::detail {
namespace {

constexpr std::string_view kGeneratedEndpointAlgorithm{"holoscan.auto_endpoint_name"};
constexpr std::uint32_t kGeneratedEndpointAlgorithmVersion = 1U;

[[nodiscard]] bool process_local_payload_equal(const PayloadIdentityRecord& left,
                                               const PayloadIdentityRecord& right) noexcept {
  if (left.kind != right.kind || left.payload_kind != right.payload_kind ||
      left.scalar.has_value() != right.scalar.has_value() ||
      left.schema.has_value() != right.schema.has_value() ||
      left.pod.has_value() != right.pod.has_value()) {
    return false;
  }
  if (left.scalar && left.scalar->kind != right.scalar->kind) {
    return false;
  }
  if (left.schema && (left.schema->canonical_name != right.schema->canonical_name ||
                      left.schema->version != right.schema->version ||
                      left.schema->schema_sha256 != right.schema->schema_sha256 ||
                      left.schema->id != right.schema->id)) {
    return false;
  }
  if (!left.pod) {
    return true;
  }
  if (left.pod->canonical_name != right.pod->canonical_name ||
      left.pod->version != right.pod->version || left.pod->object_size != right.pod->object_size ||
      left.pod->object_alignment != right.pod->object_alignment ||
      left.pod->fields.size() != right.pod->fields.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.pod->fields.size(); ++index) {
    const auto& left_field = left.pod->fields[index];
    const auto& right_field = right.pod->fields[index];
    if (left_field.name != right_field.name || left_field.byte_offset != right_field.byte_offset ||
        left_field.scalar_kind != right_field.scalar_kind ||
        left_field.element_count != right_field.element_count) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::optional<std::uint64_t> channel_payload_size_bytes(
    const PayloadIdentityRecord& identity, std::uint64_t serialized_payload_capacity_bytes,
    const bool process_local_schema_envelope) noexcept {
  if (identity.kind == PayloadIdentityKind::kSchema && identity.schema &&
      serialized_payload_capacity_bytes != 0U) {
    if (identity.payload_kind == PayloadKind::kDescriptorBacked) {
      return serialized_payload_capacity_bytes;
    }
    if (identity.payload_kind != PayloadKind::kSerializedBytes) {
      return std::nullopt;
    }
    if (!process_local_schema_envelope) {
      return serialized_payload_capacity_bytes;
    }
    if (serialized_payload_capacity_bytes >
        std::numeric_limits<std::uint64_t>::max() - kSchemaEnvelopeHeaderSize) {
      return std::nullopt;
    }
    return serialized_payload_capacity_bytes + kSchemaEnvelopeHeaderSize;
  }
  if (identity.kind == PayloadIdentityKind::kPod && identity.pod &&
      identity.pod->object_size != 0U) {
    return identity.pod->object_size;
  }
  if (identity.kind != PayloadIdentityKind::kScalar || !identity.scalar) {
    return std::nullopt;
  }
  switch (identity.scalar->kind) {
    case ScalarIdentityKind::kBoolean:
      return sizeof(bool);
    case ScalarIdentityKind::kInt8:
    case ScalarIdentityKind::kUInt8:
      return 1U;
    case ScalarIdentityKind::kInt16:
    case ScalarIdentityKind::kUInt16:
      return 2U;
    case ScalarIdentityKind::kInt32:
    case ScalarIdentityKind::kUInt32:
    case ScalarIdentityKind::kFloat32:
      return 4U;
    case ScalarIdentityKind::kInt64:
    case ScalarIdentityKind::kUInt64:
    case ScalarIdentityKind::kFloat64:
      return 8U;
    case ScalarIdentityKind::kLocalEnum:
      return std::nullopt;
  }
  return std::nullopt;
}

[[nodiscard]] std::string process_local_service_id(PlanGenerationId generation, PortId source,
                                                   QueuePolicy policy) {
  return "process-local-channel." + std::to_string(generation) + ".source." +
         std::to_string(source.value()) + ".policy." +
         std::to_string(static_cast<std::uint32_t>(policy));
}

[[nodiscard]] const PortPlanRecord* dense_port(const ExecutionPlanState& state,
                                               PortId id) noexcept {
  if (!id || id.value() > state.ports.size()) {
    return nullptr;
  }
  const auto& port = state.ports[static_cast<std::size_t>(id.value() - 1U)];
  return port.id == id ? &port : nullptr;
}

[[nodiscard]] std::optional<PortId> resolve_internal_port(const ExecutionPlanState& state,
                                                          PortId endpoint,
                                                          PortDirection direction) noexcept {
  for (std::size_t depth = 0U; depth <= state.interface_port_bindings.size(); ++depth) {
    const PortPlanRecord* const port = dense_port(state, endpoint);
    if (port == nullptr || port->direction != direction) {
      return std::nullopt;
    }
    if (port->owner_kind == PortOwnerKind::kOperator) {
      return endpoint;
    }
    if (port->owner_kind != PortOwnerKind::kGraphInterface) {
      return std::nullopt;
    }
    const auto binding =
        std::find_if(state.interface_port_bindings.begin(),
                     state.interface_port_bindings.end(),
                     [&](const auto& candidate) { return candidate.public_port == endpoint; });
    if (binding == state.interface_port_bindings.end() || binding->direction != direction) {
      return std::nullopt;
    }
    endpoint = binding->internal_port;
  }
  return std::nullopt;
}

struct ProcessLocalServiceDerivation {
  const ProcessLocalChannelServicePlanRecord* service{};
  const ProcessLocalChannelPublisherPlanRecord* publisher{};
  std::uint32_t subscriber_count{};
  std::uint32_t maximum_queue_depth{};
};

struct TensorServicePoolFacts {
  std::string_view pool_id;
  std::uint64_t max_byte_span{};
  MemoryKind memory_kind{MemoryKind::kUnknown};
  DeviceId device{};
};

struct TensorServicePoolLookup {
  std::optional<TensorServicePoolFacts> facts;
  bool ambiguous{};
};

[[nodiscard]] TensorServicePoolLookup tensor_service_pool(const ExecutionPlanState& state,
                                                          PortId source) noexcept {
  const auto direct = std::find_if(state.tensor_allocations.begin(),
                                   state.tensor_allocations.end(),
                                   [&](const auto& pool) { return pool.output == source; });
  const auto boundary = std::find_if(state.tensor_interface_pools.begin(),
                                     state.tensor_interface_pools.end(),
                                     [&](const auto& pool) { return pool.graph_port == source; });
  const bool has_direct = direct != state.tensor_allocations.end();
  const bool has_boundary = boundary != state.tensor_interface_pools.end();
  if (has_direct && has_boundary) {
    return TensorServicePoolLookup{.ambiguous = true};
  }
  if (!has_direct && !has_boundary) {
    return {};
  }
  if (has_direct) {
    return TensorServicePoolLookup{
        .facts = TensorServicePoolFacts{.pool_id = direct->pool_id,
                                        .max_byte_span = direct->bounds.max_byte_span,
                                        .memory_kind = direct->memory_kind,
                                        .device = direct->device}};
  }
  if (!boundary->logical.max_byte_span) {
    return {};
  }
  const std::uint64_t max_byte_span = boundary->logical.max_byte_span.value_or(0U);
  return TensorServicePoolLookup{.facts =
                                     TensorServicePoolFacts{.pool_id = boundary->pool_id,
                                                            .max_byte_span = max_byte_span,
                                                            .memory_kind = boundary->memory_kind,
                                                            .device = boundary->device}};
}

struct ExternalChannelDerivation {
  const ExternalTopicRecord* topic{};
  const ChannelServiceStaticPlanRecord* service{};
  const ChannelPublisherEndpointPlanRecord* publisher{};
  const ChannelSubscriberEndpointPlanRecord* subscriber{};
  const ChannelEndpointRecord* endpoint{};
  const EdgeRecord* graph_edge{};
  const PayloadEdgePlanRecord* edge{};
  PortId internal_port{};
};

template <typename Record>
[[nodiscard]] bool bind_external_record(std::vector<ExternalChannelDerivation>& derivations,
                                        const Record& record, const Record*& slot) noexcept {
  if (!record.topic || record.topic.value > derivations.size() || slot != nullptr) {
    return false;
  }
  slot = &record;
  return true;
}

[[nodiscard]] ProcessGroupId vertex_process_group(const ExecutionPlanState& state,
                                                  const VertexId vertex) noexcept {
  const auto found =
      std::find_if(state.vertex_process_placements.begin(),
                   state.vertex_process_placements.end(),
                   [&](const auto& placement) { return placement.vertex == vertex; });
  return found == state.vertex_process_placements.end() ? ProcessGroupId{} : found->process_group;
}

[[nodiscard]] bool valid_cross_process_tensor_service(
    const CrossProcessChannelServicePlanRecord& service,
    const TensorServicePoolLookup& pool) noexcept {
  if (!service.tensor_payload || service.payload_size_bytes != 0U || pool.ambiguous) {
    return false;
  }
  if (!pool.facts) {
    return service.tensor_max_byte_span == 0U && service.tensor_pool_id.empty() &&
           service.tensor_memory_kind == MemoryKind::kUnknown && !service.tensor_device.valid();
  }
  return service.tensor_max_byte_span == pool.facts->max_byte_span &&
         service.tensor_pool_id == pool.facts->pool_id &&
         service.tensor_memory_kind == pool.facts->memory_kind &&
         service.tensor_device == pool.facts->device;
}

[[nodiscard]] bool valid_cross_process_inline_service(
    const CrossProcessChannelServicePlanRecord& service,
    const std::optional<std::uint64_t>& payload_size) noexcept {
  return payload_size && service.payload_size_bytes == *payload_size &&
         service.tensor_max_byte_span == 0U && service.tensor_pool_id.empty() &&
         service.tensor_memory_kind == MemoryKind::kUnknown && !service.tensor_device.valid();
}

[[nodiscard]] bool valid_descriptor_tensor_fields(
    const std::vector<TensorFieldServiceStorageContract>& fields,
    const bool cross_process) noexcept {
  if (fields.empty() || fields.size() > kMaximumTensorSchemaLoanFields) {
    return false;
  }
  for (std::size_t index = 0U; index < fields.size(); ++index) {
    const auto& field = fields[index];
    const bool canonical_order = field.tensor_index == index &&
                                 (index == 0U || fields[index - 1U].field_id < field.field_id);
    const bool dtype_valid = !field.dtype || (field.dtype->bits != 0U && field.dtype->lanes != 0U);
    const bool memory_valid = [&]() noexcept {
      switch (field.memory_kind) {
        case MemoryKind::kHost:
          return !field.device.valid();
        case MemoryKind::kPinnedHost:
          return cross_process ? field.device.valid() : !field.device.valid();
        case MemoryKind::kCudaDevice:
        case MemoryKind::kCudaManaged:
          return field.device.valid() &&
                 (!cross_process || field.memory_kind == MemoryKind::kCudaDevice);
        case MemoryKind::kUnknown:
        default:
          return false;
      }
    }();
    if (!canonical_order || field.max_byte_span == 0U || field.pool_id.empty() || !dtype_valid ||
        !memory_valid ||
        (cross_process && (!field.dtype || !field.rank ||
                           field.tensor_emit_policy != InterfaceTensorEmitPolicy::kCopyIfNeeded)) ||
        (!cross_process &&
         field.tensor_emit_policy == InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool valid_descriptor_tensor_fields_against_allocations(
    const ExecutionPlanState& state, const PortId source,
    const std::vector<TensorFieldServiceStorageContract>& fields,
    const bool cross_process) noexcept {
  std::vector<const TensorFieldAllocationPlanRecord*> allocations;
  for (const auto& allocation : state.tensor_field_allocations) {
    if (allocation.output == source) {
      allocations.push_back(&allocation);
    }
  }
  std::sort(allocations.begin(), allocations.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->field_id < rhs->field_id;
  });
  if (allocations.empty()) {
    return fields.empty();
  }
  if (fields.size() != allocations.size() ||
      !valid_descriptor_tensor_fields(fields, cross_process)) {
    return false;
  }
  for (std::size_t index = 0U; index < allocations.size(); ++index) {
    const auto& field = fields[index];
    const auto& allocation = *allocations[index];
    const auto layout = allocation.layout;
    std::optional<DLDataType> allocation_dtype;
    std::optional<std::uint32_t> allocation_rank;
    if (layout.has_value()) {
      const auto& resolved_layout = layout.value();
      allocation_dtype = resolved_layout.dtype;
      allocation_rank = resolved_layout.rank;
    }
    const DLDataType field_dtype = field.dtype.value_or(DLDataType{});
    const DLDataType planned_dtype = allocation_dtype.value_or(DLDataType{});
    const bool dtype_equal = field.dtype.has_value() == allocation_dtype.has_value() &&
                             field_dtype.code == planned_dtype.code &&
                             field_dtype.bits == planned_dtype.bits &&
                             field_dtype.lanes == planned_dtype.lanes;
    // The service pool identifies transport staging; it is intentionally distinct from the
    // producer allocation pool. Compare the storage contract that staging must preserve.
    if (field.field_id != allocation.field_id || field.tensor_index != allocation.tensor_index ||
        field.max_byte_span != allocation.bounds.max_byte_span || field.rank != allocation_rank ||
        !dtype_equal || field.memory_kind != allocation.memory_kind ||
        field.device != allocation.device) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool valid_process_local_descriptor_tensor_fields(
    const ExecutionPlanState& state, const ProcessLocalChannelServicePlanRecord& service) noexcept {
  return valid_descriptor_tensor_fields_against_allocations(
      state, service.source, service.tensor_fields, false);
}

[[nodiscard]] bool valid_cross_process_descriptor_schema_service(
    const ExecutionPlanState& state, const CrossProcessChannelServicePlanRecord& service,
    const std::optional<std::uint64_t>& payload_size) noexcept {
  if (!service.tensor_fields.empty()) {
    return payload_size && service.tensor_payload && service.payload_size_bytes == *payload_size &&
           service.tensor_max_byte_span == 0U && service.tensor_pool_id.empty() &&
           service.tensor_memory_kind == MemoryKind::kUnknown && !service.tensor_device.valid() &&
           service.tensor_emit_policy == InterfaceTensorEmitPolicy::kCopyIfNeeded &&
           valid_descriptor_tensor_fields_against_allocations(
               state, service.source, service.tensor_fields, true);
  }
  return payload_size && service.tensor_payload && service.payload_size_bytes == *payload_size &&
         service.tensor_max_byte_span != 0U && !service.tensor_pool_id.empty() &&
         ((service.tensor_memory_kind == MemoryKind::kHost && !service.tensor_device.valid()) ||
          ((service.tensor_memory_kind == MemoryKind::kPinnedHost ||
            service.tensor_memory_kind == MemoryKind::kCudaDevice) &&
           service.tensor_device.valid())) &&
         service.tensor_emit_policy == InterfaceTensorEmitPolicy::kCopyIfNeeded;
}

[[nodiscard]] std::string cross_process_service_id(const ExecutionPlanState& state, PortId source,
                                                   QueuePolicy policy, EdgeId anchor) {
  return fmt::format("cross-process-channel.{}.source.{}.policy.{}.anchor.{}",
                     state.plan_generation,
                     source.value(),
                     static_cast<std::uint32_t>(policy),
                     anchor.value);
}

[[nodiscard]] bool valid_cross_process_service_member(
    const ExecutionPlanState& state, const PayloadEdgePlanRecord& edge,
    const PortPlanRecord& source, const PortPlanRecord& target,
    const CrossProcessChannelServicePlanRecord& service,
    const std::optional<std::uint64_t>& payload_size, const bool tensor) {
  const std::string expected_service_id =
      cross_process_service_id(state, edge.source, edge.queue_policy, service.edge);
  const bool descriptor_tensor_schema =
      edge.payload_identity.kind == PayloadIdentityKind::kSchema &&
      edge.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
  const ProcessGroupId source_group = vertex_process_group(state, source.owner);
  const ProcessGroupId target_group = vertex_process_group(state, target.owner);
  if (service.source != edge.source || service.service_id != expected_service_id ||
      !process_local_payload_equal(service.payload_identity, edge.payload_identity) ||
      service.tensor_payload != (tensor || descriptor_tensor_schema) ||
      service.queue_policy != edge.queue_policy || service.max_subscribers == 0U ||
      service.max_subscriber_queue_depth < edge.queue_depth ||
      service.max_subscriber_borrowed_samples < edge.queue_depth ||
      service.publisher_history_depth != 0U || !service.event_wake ||
      !service.source_process_group || !service.target_process_group ||
      service.source_process_group == service.target_process_group ||
      service.source_process_group != source_group ||
      !std::binary_search(
          service.target_process_groups.begin(),
          service.target_process_groups.end(),
          target_group,
          [](ProcessGroupId left, ProcessGroupId right) { return left.value < right.value; })) {
    return false;
  }
  const auto tensor_pool =
      tensor ? tensor_service_pool(state, edge.source) : TensorServicePoolLookup{};
  if (tensor) {
    return valid_cross_process_tensor_service(service, tensor_pool);
  }
  return descriptor_tensor_schema
             ? valid_cross_process_descriptor_schema_service(state, service, payload_size)
             : valid_cross_process_inline_service(service, payload_size);
}

[[nodiscard]] bool valid_cross_process_publisher(
    const PayloadEdgePlanRecord& edge, const CrossProcessChannelServicePlanRecord& service,
    const CrossProcessChannelPublisherPlanRecord& publisher) {
  return publisher.edge == edge.edge && publisher.source == edge.source &&
         publisher.service_id == service.service_id &&
         publisher.max_loaned_samples >= edge.producer_max_emits_per_compute &&
         publisher.replacement_attribution_required ==
             (edge.queue_policy != QueuePolicy::kBlocking);
}

[[nodiscard]] bool valid_cross_process_subscriber(
    const PayloadEdgePlanRecord& edge, const CrossProcessChannelServicePlanRecord& service,
    const CrossProcessChannelSubscriberPlanRecord& subscriber) {
  return subscriber.edge == edge.edge && subscriber.source == edge.source &&
         subscriber.target == edge.target && subscriber.service_id == service.service_id &&
         subscriber.requested_queue_depth == edge.queue_depth &&
         subscriber.max_borrowed_samples == edge.queue_depth &&
         subscriber.cache_capacity == edge.cache_capacity &&
         subscriber.readiness_capacity == std::max(edge.queue_depth, edge.cache_capacity) &&
         subscriber.event_wake;
}

[[nodiscard]] bool valid_cross_process_edge_contract(const PayloadEdgePlanRecord& edge,
                                                     const PortPlanRecord* source,
                                                     const PortPlanRecord* target) noexcept {
  return edge.edge && source != nullptr && target != nullptr &&
         source->owner_kind == PortOwnerKind::kOperator &&
         target->owner_kind == PortOwnerKind::kOperator &&
         source->direction == PortDirection::kOutput &&
         target->direction == PortDirection::kInput && edge.queue_depth != 0U &&
         edge.producer_max_emits_per_compute != 0U &&
         source->max_emits_per_compute == edge.producer_max_emits_per_compute &&
         edge.disposition == PlanDisposition::kAccepted;
}

}  // namespace

// This is a whole-plan relational validator: every early return protects a distinct cross-table
// invariant, and publishing a partially validated derived relation would weaken Runtime admission.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool valid_process_local_channel_plan(const ExecutionPlanState& state) {
  constexpr std::size_t kQueuePolicyCount = 2U;
  if (state.ports.size() > std::numeric_limits<std::size_t>::max() / kQueuePolicyCount ||
      state.process_local_channel_services.size() !=
          state.process_local_channel_publishers.size() ||
      state.channel_locality.size() != state.payload_edges.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < state.ports.size(); ++index) {
    const auto& port = state.ports[index];
    const bool schema = port.payload_identity.kind == PayloadIdentityKind::kSchema;
    if (port.id.value() != index + 1U || schema != (port.serialized_payload_capacity_bytes != 0U)) {
      return false;
    }
  }

  // Port IDs are dense and the queue-policy family is closed, so a two-slot
  // table per source gives an exact linear join without a service-by-edge scan.
  std::vector<ProcessLocalServiceDerivation> derivations(state.ports.size() * kQueuePolicyCount);
  const auto graph_edges = state.graph.edges();
  // Compiler-owned plans retain the Graph relation. A few private unit fixtures intentionally
  // exercise the transport constructor from a hand-authored record set without a GraphView; the
  // exact Operator ownership checks below remain mandatory for both forms.
  std::vector<const EdgeRecord*> graph_edge_by_id(graph_edges.size(), nullptr);
  for (const auto& graph_edge : graph_edges) {
    if (!graph_edge.id || graph_edge.id.value > graph_edge_by_id.size()) {
      return false;
    }
    auto& indexed = graph_edge_by_id[static_cast<std::size_t>(graph_edge.id.value - 1U)];
    if (indexed != nullptr) {
      return false;
    }
    indexed = &graph_edge;
  }
  for (std::size_t index = 0U; index < state.process_local_channel_services.size(); ++index) {
    const auto& service = state.process_local_channel_services[index];
    const auto& publisher = state.process_local_channel_publishers[index];
    const PortPlanRecord* const source = dense_port(state, service.source);
    const auto policy = static_cast<std::uint32_t>(service.queue_policy);
    if (source == nullptr || source->owner_kind != PortOwnerKind::kOperator ||
        source->direction != PortDirection::kOutput || policy >= kQueuePolicyCount) {
      return false;
    }
    if (index != 0U) {
      const auto& previous = state.process_local_channel_services[index - 1U];
      if (previous.source.value() > service.source.value() ||
          (previous.source == service.source &&
           static_cast<std::uint32_t>(previous.queue_policy) >= policy)) {
        return false;
      }
    }
    const std::size_t slot =
        static_cast<std::size_t>(service.source.value() - 1U) * kQueuePolicyCount + policy;
    if (derivations[slot].service != nullptr || publisher.source != service.source ||
        publisher.service_id != service.service_id) {
      return false;
    }
    derivations[slot].service = &service;
    derivations[slot].publisher = &publisher;
  }

  std::size_t subscriber_index = 0U;
  EdgeId previous_edge{};
  for (std::size_t edge_index = 0U; edge_index < state.payload_edges.size(); ++edge_index) {
    const auto& edge = state.payload_edges[edge_index];
    const auto& locality = state.channel_locality[edge_index];
    if (previous_edge && edge.edge.value <= previous_edge.value) {
      return false;
    }
    previous_edge = edge.edge;
    if (locality.edge != edge.edge || locality.accepted != edge.locality ||
        locality.requires_runtime_attach !=
            (locality.accepted != ChannelLocalityKind::kProcessLocal)) {
      return false;
    }

    const PortPlanRecord* const source = dense_port(state, edge.source);
    const PortPlanRecord* const target = dense_port(state, edge.target);
    if (source == nullptr || target == nullptr) {
      return false;
    }
    const bool source_schema = source->payload_identity.kind == PayloadIdentityKind::kSchema;
    const bool target_schema = target->payload_identity.kind == PayloadIdentityKind::kSchema;
    if (source_schema != (source->serialized_payload_capacity_bytes != 0U) ||
        target_schema != (target->serialized_payload_capacity_bytes != 0U) ||
        source->serialized_payload_capacity_bytes != target->serialized_payload_capacity_bytes) {
      return false;
    }
    const auto payload_size = channel_payload_size_bytes(
        edge.payload_identity, source->serialized_payload_capacity_bytes, true);
    const bool tensor = edge.payload_identity.kind == PayloadIdentityKind::kTensor;
    const bool descriptor_tensor_schema =
        edge.payload_identity.kind == PayloadIdentityKind::kSchema &&
        edge.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
    if (edge.locality != ChannelLocalityKind::kProcessLocal || (!payload_size && !tensor)) {
      continue;
    }
    const auto policy = static_cast<std::uint32_t>(edge.queue_policy);
    if (!edge.edge || edge.disposition != PlanDisposition::kAccepted || source == nullptr ||
        target == nullptr || source->owner_kind != PortOwnerKind::kOperator ||
        target->owner_kind != PortOwnerKind::kOperator ||
        source->direction != PortDirection::kOutput || target->direction != PortDirection::kInput ||
        policy >= kQueuePolicyCount || edge.queue_depth == 0U ||
        edge.producer_max_emits_per_compute == 0U ||
        source->max_emits_per_compute != edge.producer_max_emits_per_compute ||
        !process_local_payload_equal(source->payload_identity, edge.payload_identity) ||
        !process_local_payload_equal(target->payload_identity, edge.payload_identity)) {
      return false;
    }
    if (!graph_edge_by_id.empty()) {
      if (edge.edge.value > graph_edge_by_id.size()) {
        return false;
      }
      const EdgeRecord* const graph_edge =
          graph_edge_by_id[static_cast<std::size_t>(edge.edge.value - 1U)];
      const auto expected_source =
          graph_edge != nullptr
              ? resolve_internal_port(
                    state, PortId{graph_edge->source.value}, PortDirection::kOutput)
              : std::nullopt;
      const auto expected_target =
          graph_edge != nullptr
              ? resolve_internal_port(
                    state, PortId{graph_edge->target.value}, PortDirection::kInput)
              : std::nullopt;
      if (!expected_source || !expected_target || edge.source != *expected_source ||
          edge.target != *expected_target) {
        return false;
      }
    }

    const std::size_t slot =
        static_cast<std::size_t>(edge.source.value() - 1U) * kQueuePolicyCount + policy;
    auto& derived = derivations[slot];
    if (derived.service == nullptr || derived.publisher == nullptr ||
        subscriber_index >= state.process_local_channel_subscribers.size()) {
      return false;
    }
    const auto& service = *derived.service;
    const auto& publisher = *derived.publisher;
    const auto eligible_delivery_kind = process_local_channel_delivery_kind(edge);
    if (service.delivery_kind == ProcessLocalChannelDeliveryKind::kDirectRetained &&
        eligible_delivery_kind != ProcessLocalChannelDeliveryKind::kDirectRetained) {
      return false;
    }
    const bool expected_event_wake =
        service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ;
    const auto tensor_pool =
        tensor ? tensor_service_pool(state, edge.source) : TensorServicePoolLookup{};
    const std::string expected_service_id =
        process_local_service_id(state.plan_generation, edge.source, edge.queue_policy);
    if (service.service_id != expected_service_id || service.source != edge.source ||
        service.queue_policy != edge.queue_policy ||
        !process_local_payload_equal(service.payload_identity, edge.payload_identity) ||
        service.publisher_history_depth != 0U || service.event_wake != expected_event_wake ||
        publisher.max_loaned_samples != edge.producer_max_emits_per_compute ||
        publisher.replacement_attribution_required !=
            (edge.queue_policy != QueuePolicy::kBlocking)) {
      return false;
    }
    if (tensor) {
      if (!service.tensor_payload || service.payload_size_bytes != 0U || tensor_pool.ambiguous) {
        return false;
      }
      if (tensor_pool.facts) {
        if (service.tensor_max_byte_span != tensor_pool.facts->max_byte_span ||
            service.tensor_pool_id != tensor_pool.facts->pool_id ||
            service.tensor_memory_kind != tensor_pool.facts->memory_kind ||
            service.tensor_device != tensor_pool.facts->device) {
          return false;
        }
      } else if (service.tensor_max_byte_span != 0U || !service.tensor_pool_id.empty() ||
                 service.tensor_memory_kind != MemoryKind::kUnknown ||
                 service.tensor_device.valid()) {
        // Ordinary pass-through Tensor services carry storage authority in the
        // sample itself and must not fabricate allocation-pool evidence.
        return false;
      }
    } else if (descriptor_tensor_schema) {
      if (!service.tensor_payload || !payload_size || service.payload_size_bytes != *payload_size ||
          (service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ &&
           service.payload_size_bytes > channel_internal::kMaximumInlinePayloadBytes) ||
          service.tensor_max_byte_span != 0U || !service.tensor_pool_id.empty() ||
          service.tensor_memory_kind != MemoryKind::kUnknown || service.tensor_device.valid() ||
          (state.admitted && !valid_process_local_descriptor_tensor_fields(state, service))) {
        return false;
      }
    } else if (service.tensor_payload || !payload_size ||
               service.payload_size_bytes != *payload_size || service.tensor_max_byte_span != 0U ||
               !service.tensor_pool_id.empty() ||
               service.tensor_memory_kind != MemoryKind::kUnknown ||
               service.tensor_device.valid()) {
      return false;
    }

    const auto& subscriber = state.process_local_channel_subscribers[subscriber_index++];
    if (subscriber.edge != edge.edge || subscriber.source != edge.source ||
        subscriber.target != edge.target || subscriber.service_id != expected_service_id ||
        subscriber.requested_queue_depth != edge.queue_depth ||
        subscriber.max_borrowed_samples != edge.queue_depth ||
        subscriber.cache_capacity != edge.cache_capacity ||
        subscriber.readiness_capacity != std::max(edge.queue_depth, edge.cache_capacity) ||
        subscriber.event_wake != expected_event_wake ||
        derived.subscriber_count == std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    ++derived.subscriber_count;
    derived.maximum_queue_depth = std::max(derived.maximum_queue_depth, edge.queue_depth);
  }
  if (subscriber_index != state.process_local_channel_subscribers.size()) {
    return false;
  }

  std::size_t derived_service_count = 0U;
  for (const auto& derived : derivations) {
    if (derived.service == nullptr) {
      continue;
    }
    ++derived_service_count;
    if (derived.subscriber_count == 0U ||
        derived.service->max_subscribers != derived.subscriber_count ||
        derived.service->max_subscriber_queue_depth != derived.maximum_queue_depth ||
        derived.service->max_subscriber_borrowed_samples != derived.maximum_queue_depth) {
      return false;
    }
  }
  return derived_service_count == state.process_local_channel_services.size();
}

// Keep the retained Graph, alias inventory, endpoint catalog, and physical records in one
// fail-closed relational validation pass so no independently valid subset can authorize binding.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool valid_external_channel_plan(const ExecutionPlanState& state) {
  const auto topics = state.graph.external_topics();
  const auto* catalog = ChannelEndpointCatalogAccess::state(state.endpoint_catalog);
  if (state.channel_services.size() != topics.size() ||
      state.channel_publishers.size() != topics.size() ||
      state.channel_subscribers.size() != topics.size() ||
      (catalog == nullptr ? 0U : catalog->endpoints.size()) != topics.size()) {
    return false;
  }
  if (topics.empty()) {
    return true;
  }

  const auto graph_ports = state.graph.ports();
  if (state.ports.size() != graph_ports.size()) {
    return false;
  }
  for (const auto& graph_port : graph_ports) {
    if (!graph_port.id || graph_port.id.value > state.ports.size()) {
      return false;
    }
    const PortPlanRecord* const port = dense_port(state, PortId{graph_port.id.value});
    if (port == nullptr || port->owner != graph_port.owner ||
        port->owner_kind != graph_port.owner_kind || port->direction != graph_port.direction ||
        port->local_name != graph_port.name ||
        port->serialized_payload_capacity_bytes != graph_port.serialized_payload_capacity_bytes ||
        !process_local_payload_equal(port->payload_identity, graph_port.payload_identity)) {
      return false;
    }
  }

  // The plan-local alias inventory is part of the external binding authority:
  // an edge endpoint may be a reusable-Graph interface, but Runtime must bind
  // the exact internal Operator port represented by that alias. Prove the
  // inventory against the retained Graph before using it to resolve a topic.
  const auto graph_bindings = state.graph.interface_bindings();
  if (state.interface_port_bindings.size() != graph_bindings.size()) {
    return false;
  }
  std::vector<const InterfacePortBindingPlanRecord*> binding_by_public(state.ports.size(), nullptr);
  for (const auto& binding : state.interface_port_bindings) {
    const PortPlanRecord* const public_port = dense_port(state, binding.public_port);
    const PortPlanRecord* const internal_port = dense_port(state, binding.internal_port);
    if (public_port == nullptr || internal_port == nullptr ||
        public_port->owner_kind != PortOwnerKind::kGraphInterface ||
        public_port->direction != binding.direction ||
        internal_port->direction != binding.direction ||
        public_port->serialized_payload_capacity_bytes !=
            internal_port->serialized_payload_capacity_bytes ||
        !process_local_payload_equal(public_port->payload_identity,
                                     internal_port->payload_identity)) {
      return false;
    }
    auto& indexed = binding_by_public[static_cast<std::size_t>(binding.public_port.value() - 1U)];
    if (indexed != nullptr) {
      return false;
    }
    indexed = &binding;
  }
  for (const auto& binding : graph_bindings) {
    if (!binding.public_port || binding.public_port.value > binding_by_public.size() ||
        !binding.internal_port || binding.internal_port.value > state.ports.size()) {
      return false;
    }
    const auto* planned =
        binding_by_public[static_cast<std::size_t>(binding.public_port.value - 1U)];
    if (planned == nullptr || planned->internal_port != PortId(binding.internal_port.value) ||
        planned->direction != binding.direction) {
      return false;
    }
  }

  const auto resolve_internal_port = [&](PortId endpoint,
                                         PortDirection direction) -> std::optional<PortId> {
    for (std::size_t depth = 0U; depth <= binding_by_public.size(); ++depth) {
      const PortPlanRecord* const port = dense_port(state, endpoint);
      if (port == nullptr || port->direction != direction) {
        return std::nullopt;
      }
      if (port->owner_kind == PortOwnerKind::kOperator) {
        return endpoint;
      }
      if (port->owner_kind != PortOwnerKind::kGraphInterface) {
        return std::nullopt;
      }
      const auto* binding = binding_by_public[static_cast<std::size_t>(endpoint.value() - 1U)];
      if (binding == nullptr || binding->direction != direction) {
        return std::nullopt;
      }
      endpoint = binding->internal_port;
    }
    return std::nullopt;
  };

  const auto graph_edges = state.graph.edges();
  std::vector<const EdgeRecord*> graph_edge_by_id(graph_edges.size(), nullptr);
  for (const auto& edge : graph_edges) {
    if (!edge.id || edge.id.value > graph_edge_by_id.size() || !edge.source || !edge.target ||
        edge.source.value > state.ports.size() || edge.target.value > state.ports.size()) {
      return false;
    }
    auto& indexed = graph_edge_by_id[static_cast<std::size_t>(edge.id.value - 1U)];
    if (indexed != nullptr) {
      return false;
    }
    indexed = &edge;
  }

  std::vector<ExternalChannelDerivation> derivations(topics.size());
  std::vector<ExternalChannelDerivation*> topic_by_port(state.ports.size(), nullptr);
  for (const auto& topic : topics) {
    if (!topic.id || topic.id.value > topics.size() || !topic.graph_port ||
        topic.graph_port.value > state.ports.size()) {
      return false;
    }
    auto& derived = derivations[static_cast<std::size_t>(topic.id.value - 1U)];
    const std::size_t port_index = static_cast<std::size_t>(topic.graph_port.value - 1U);
    if (derived.topic != nullptr || topic_by_port[port_index] != nullptr) {
      return false;
    }
    const PortPlanRecord* port = dense_port(state, PortId{topic.graph_port.value});
    const PortDirection expected_direction =
        topic.imported ? PortDirection::kOutput : PortDirection::kInput;
    const PortOwnerKind expected_owner =
        topic.imported ? PortOwnerKind::kImportedTopic : PortOwnerKind::kExportedTopic;
    if (port == nullptr || port->owner != topic.vertex || port->direction != expected_direction ||
        port->owner_kind != expected_owner ||
        (port->payload_identity.kind == PayloadIdentityKind::kSchema) !=
            (port->serialized_payload_capacity_bytes != 0U) ||
        port->serialized_payload_capacity_bytes != topic.serialized_payload_capacity_bytes ||
        !process_local_payload_equal(port->payload_identity, topic.payload_identity)) {
      return false;
    }
    derived.topic = &topic;
    topic_by_port[port_index] = &derived;
  }

  for (const auto& service : state.channel_services) {
    if (!service.topic || service.topic.value > derivations.size()) {
      return false;
    }
    auto& derived = derivations[static_cast<std::size_t>(service.topic.value - 1U)];
    if (!bind_external_record(derivations, service, derived.service)) {
      return false;
    }
  }
  for (const auto& publisher : state.channel_publishers) {
    if (!publisher.topic || publisher.topic.value > derivations.size()) {
      return false;
    }
    auto& derived = derivations[static_cast<std::size_t>(publisher.topic.value - 1U)];
    if (!bind_external_record(derivations, publisher, derived.publisher)) {
      return false;
    }
  }
  for (const auto& subscriber : state.channel_subscribers) {
    if (!subscriber.topic || subscriber.topic.value > derivations.size()) {
      return false;
    }
    auto& derived = derivations[static_cast<std::size_t>(subscriber.topic.value - 1U)];
    if (!bind_external_record(derivations, subscriber, derived.subscriber)) {
      return false;
    }
  }
  std::set<std::string_view> endpoint_names;
  std::set<std::string_view> endpoint_services;
  for (const auto& endpoint : catalog->endpoints) {
    if (!endpoint.boundary_port || endpoint.boundary_port.value() > topic_by_port.size()) {
      return false;
    }
    ExternalChannelDerivation* const derived =
        topic_by_port[static_cast<std::size_t>(endpoint.boundary_port.value() - 1U)];
    if (derived == nullptr || derived->endpoint != nullptr ||
        !endpoint_names.insert(endpoint.name).second ||
        !endpoint_services.insert(endpoint.service).second) {
      return false;
    }
    derived->endpoint = &endpoint;
  }

  // First derive the sole optional boundary-to-Operator relation from the
  // retained Graph. This makes the Graph edge and any interface-alias chain,
  // rather than a mutable physical edge record, the identity authority.
  for (const auto& edge : graph_edges) {
    ExternalChannelDerivation* derived{};
    PortId authored_internal{};
    PortDirection internal_direction{PortDirection::kInput};
    if (edge.source && edge.source.value <= topic_by_port.size()) {
      auto* candidate = topic_by_port[static_cast<std::size_t>(edge.source.value - 1U)];
      if (candidate != nullptr) {
        if (!candidate->topic->imported) {
          return false;
        }
        derived = candidate;
        authored_internal = PortId{edge.target.value};
        internal_direction = PortDirection::kInput;
      }
    }
    if (edge.target && edge.target.value <= topic_by_port.size()) {
      auto* candidate = topic_by_port[static_cast<std::size_t>(edge.target.value - 1U)];
      if (candidate != nullptr) {
        if (candidate->topic->imported || derived != nullptr) {
          return false;
        }
        derived = candidate;
        authored_internal = PortId{edge.source.value};
        internal_direction = PortDirection::kOutput;
      }
    }
    if (derived == nullptr) {
      continue;
    }
    const auto internal = resolve_internal_port(authored_internal, internal_direction);
    const PortPlanRecord* const boundary_port =
        dense_port(state, PortId{derived->topic->graph_port.value});
    const PortPlanRecord* const authored_port = dense_port(state, authored_internal);
    const PortPlanRecord* const internal_port = internal ? dense_port(state, *internal) : nullptr;
    if (derived->graph_edge != nullptr || !internal || boundary_port == nullptr ||
        authored_port == nullptr || internal_port == nullptr ||
        boundary_port->serialized_payload_capacity_bytes !=
            authored_port->serialized_payload_capacity_bytes ||
        authored_port->serialized_payload_capacity_bytes !=
            internal_port->serialized_payload_capacity_bytes ||
        !process_local_payload_equal(authored_port->payload_identity,
                                     derived->topic->payload_identity) ||
        !process_local_payload_equal(internal_port->payload_identity,
                                     derived->topic->payload_identity)) {
      return false;
    }
    derived->graph_edge = &edge;
    derived->internal_port = *internal;
  }

  for (const auto& edge : state.payload_edges) {
    ExternalChannelDerivation* derived{};
    if (edge.source && edge.source.value() <= topic_by_port.size()) {
      auto* candidate = topic_by_port[static_cast<std::size_t>(edge.source.value() - 1U)];
      if (candidate != nullptr) {
        if (!candidate->topic->imported) {
          return false;
        }
        derived = candidate;
      }
    }
    if (edge.target && edge.target.value() <= topic_by_port.size()) {
      auto* candidate = topic_by_port[static_cast<std::size_t>(edge.target.value() - 1U)];
      if (candidate != nullptr) {
        if (candidate->topic->imported) {
          return false;
        }
        if (derived != nullptr && derived != candidate) {
          return false;
        }
        derived = candidate;
      }
    }
    if (derived != nullptr) {
      if (!edge.edge || edge.edge.value > graph_edge_by_id.size()) {
        return false;
      }
      const EdgeRecord* const graph_edge =
          graph_edge_by_id[static_cast<std::size_t>(edge.edge.value - 1U)];
      if (graph_edge == nullptr) {
        return false;
      }
      const PortId authored_internal = derived->topic->imported ? PortId{graph_edge->target.value}
                                                                : PortId{graph_edge->source.value};
      const PortDirection internal_direction =
          derived->topic->imported ? PortDirection::kInput : PortDirection::kOutput;
      const auto internal = resolve_internal_port(authored_internal, internal_direction);
      const PortId expected_source =
          derived->topic->imported ? PortId{graph_edge->source.value} : derived->internal_port;
      const PortId expected_target =
          derived->topic->imported ? derived->internal_port : PortId{graph_edge->target.value};
      if (derived->edge != nullptr || graph_edge != derived->graph_edge || !internal ||
          *internal != derived->internal_port || edge.source != expected_source ||
          edge.target != expected_target) {
        return false;
      }
      derived->edge = &edge;
    }
  }

  for (const auto& derived : derivations) {
    if (derived.topic == nullptr || derived.service == nullptr || derived.publisher == nullptr ||
        derived.subscriber == nullptr || derived.endpoint == nullptr) {
      return false;
    }
    const auto& topic = *derived.topic;
    const auto& service = *derived.service;
    const auto& publisher = *derived.publisher;
    const auto& subscriber = *derived.subscriber;
    const auto& endpoint = *derived.endpoint;

    if ((derived.graph_edge == nullptr) != (derived.edge == nullptr) ||
        (derived.edge != nullptr && !derived.internal_port)) {
      return false;
    }

    const QueuePolicy queue_policy =
        derived.edge == nullptr ? QueuePolicy::kBlocking : derived.edge->queue_policy;
    const std::uint32_t graph_queue = derived.edge == nullptr ? 1U : derived.edge->queue_depth;
    const std::uint32_t graph_burst =
        derived.edge == nullptr ? 1U : derived.edge->producer_max_emits_per_compute;
    const std::uint32_t publisher_loans =
        topic.imported && topic.bounds.publisher && topic.bounds.publisher->max_loaned_samples
            ? *topic.bounds.publisher->max_loaned_samples
            : graph_burst;
    const std::uint32_t subscriber_queue =
        !topic.imported && topic.bounds.subscriber && topic.bounds.subscriber->queue_depth
            ? *topic.bounds.subscriber->queue_depth
            : graph_queue;
    const std::uint32_t subscriber_borrows =
        !topic.imported && topic.bounds.subscriber && topic.bounds.subscriber->max_borrowed_samples
            ? *topic.bounds.subscriber->max_borrowed_samples
            : subscriber_queue;
    const ChannelLocalityKind expected_locality = topic.imported
                                                      ? ChannelLocalityKind::kExternalImport
                                                      : ChannelLocalityKind::kExternalExport;
    if (derived.edge != nullptr &&
        (derived.edge->locality != expected_locality ||
         derived.edge->disposition != PlanDisposition::kAccepted ||
         derived.edge->queue_depth == 0U || derived.edge->producer_max_emits_per_compute == 0U ||
         !process_local_payload_equal(derived.edge->payload_identity, topic.payload_identity))) {
      return false;
    }

    const bool generated_name = topic.generated_name_requested || !topic.endpoint_name;
    if (endpoint.uri != topic.uri ||
        endpoint.direction !=
            (topic.imported ? TopicDirection::kImport : TopicDirection::kExport) ||
        endpoint.name_origin !=
            (generated_name ? EndpointNameOrigin::kGenerated : EndpointNameOrigin::kExplicit) ||
        generated_name != endpoint.generated_name.has_value() ||
        (generated_name &&
         (endpoint.generated_name->algorithm != kGeneratedEndpointAlgorithm ||
          endpoint.generated_name->version != kGeneratedEndpointAlgorithmVersion)) ||
        (!generated_name && endpoint.name != *topic.endpoint_name) || endpoint.service.empty() ||
        endpoint.service != endpoint.service_static.id ||
        endpoint.service != endpoint.name + ".service" || endpoint.queue_policy != queue_policy ||
        endpoint.service_static.queue_policy != queue_policy ||
        endpoint.service_max_subscriber_depth != subscriber_queue ||
        endpoint.service_max_subscriber_borrowed != subscriber_borrows ||
        endpoint.service_static.max_subscriber_depth != subscriber_queue ||
        endpoint.service_static.max_subscriber_borrowed != subscriber_borrows ||
        endpoint.service_static.publisher_history_depth != 0U ||
        !endpoint.service_static.event_wake ||
        endpoint.serialized_payload_capacity_bytes != topic.serialized_payload_capacity_bytes ||
        endpoint.service_static.serialized_payload_capacity_bytes !=
            topic.serialized_payload_capacity_bytes ||
        !process_local_payload_equal(endpoint.payload_identity, topic.payload_identity) ||
        !process_local_payload_equal(endpoint.service_static.payload_identity,
                                     topic.payload_identity)) {
      return false;
    }

    if (service.topic != topic.id || service.service_id != endpoint.service ||
        publisher.topic != topic.id || publisher.service_id != endpoint.service ||
        subscriber.topic != topic.id || subscriber.service_id != endpoint.service ||
        service.queue_policy != queue_policy ||
        !process_local_payload_equal(service.payload_identity, topic.payload_identity) ||
        service.max_subscriber_queue_depth != subscriber_queue ||
        service.max_subscriber_borrowed_samples != subscriber_borrows ||
        service.publisher_history_depth != 0U || !service.event_wake ||
        service.tensor_payload != (topic.payload_identity.kind == PayloadIdentityKind::kTensor) ||
        service.tensor_emit_policy != (topic.tensor
                                           ? topic.tensor->foreign_emit_policy
                                           : InterfaceTensorEmitPolicy::kRequireExportable) ||
        publisher.max_loaned_samples != publisher_loans ||
        publisher.replacement_attribution_required != (queue_policy != QueuePolicy::kBlocking) ||
        subscriber.requested_queue_depth != subscriber_queue ||
        subscriber.max_borrowed_samples != subscriber_borrows || !subscriber.event_wake) {
      return false;
    }

    if (topic.imported) {
      if (topic.bounds.subscriber || endpoint.bounds.subscriber ||
          endpoint.subscriber_request.has_value() || !endpoint.bounds.publisher ||
          !endpoint.bounds.publisher->max_loaned_samples || !endpoint.publisher_request ||
          !endpoint.publisher_request->max_loaned_samples ||
          *endpoint.bounds.publisher->max_loaned_samples != publisher_loans ||
          *endpoint.publisher_request->max_loaned_samples != publisher_loans) {
        return false;
      }
    } else if (topic.bounds.publisher || endpoint.bounds.publisher ||
               endpoint.publisher_request.has_value() || !endpoint.bounds.subscriber ||
               !endpoint.bounds.subscriber->queue_depth ||
               !endpoint.bounds.subscriber->max_borrowed_samples || !endpoint.subscriber_request ||
               *endpoint.bounds.subscriber->queue_depth != subscriber_queue ||
               *endpoint.bounds.subscriber->max_borrowed_samples != subscriber_borrows ||
               endpoint.subscriber_request->queue_depth != subscriber_queue ||
               endpoint.subscriber_request->max_borrowed_samples != subscriber_borrows) {
      return false;
    }
  }
  return true;
}

// Cross-process validation intentionally evaluates the complete correlated plan artifact.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool valid_cross_process_channel_plan(const ExecutionPlanState& state) {
  if (state.cross_process_channel_services.size() !=
          state.cross_process_channel_publishers.size() ||
      state.channel_locality.size() != state.payload_edges.size()) {
    return false;
  }
  for (const auto& service : state.cross_process_channel_services) {
    if (std::count_if(state.cross_process_channel_services.begin(),
                      state.cross_process_channel_services.end(),
                      [&](const auto& candidate) {
                        return candidate.service_id == service.service_id;
                      }) != 1 ||
        std::count_if(state.cross_process_channel_publishers.begin(),
                      state.cross_process_channel_publishers.end(),
                      [&](const auto& candidate) {
                        return candidate.service_id == service.service_id;
                      }) != 1) {
      return false;
    }
  }
  std::size_t expected_subscribers = 0U;
  for (const auto& edge : state.payload_edges) {
    if (edge.locality != ChannelLocalityKind::kCrossProcess) {
      continue;
    }
    const PortPlanRecord* const source = dense_port(state, edge.source);
    const PortPlanRecord* const target = dense_port(state, edge.target);
    if (!valid_cross_process_edge_contract(edge, source, target)) {
      return false;
    }
    const auto payload_size = channel_payload_size_bytes(
        edge.payload_identity, source->serialized_payload_capacity_bytes, false);
    const bool tensor = edge.payload_identity.kind == PayloadIdentityKind::kTensor;
    if (!payload_size && !tensor) {
      continue;
    }
    ++expected_subscribers;
    const auto subscriber =
        std::find_if(state.cross_process_channel_subscribers.begin(),
                     state.cross_process_channel_subscribers.end(),
                     [&](const auto& candidate) { return candidate.edge == edge.edge; });
    if (subscriber == state.cross_process_channel_subscribers.end()) {
      return false;
    }
    const auto service = std::find_if(
        state.cross_process_channel_services.begin(),
        state.cross_process_channel_services.end(),
        [&](const auto& candidate) { return candidate.service_id == subscriber->service_id; });
    if (service == state.cross_process_channel_services.end() ||
        !valid_cross_process_service_member(
            state, edge, *source, *target, *service, payload_size, tensor) ||
        !valid_cross_process_subscriber(edge, *service, *subscriber)) {
      return false;
    }
  }
  if (expected_subscribers != state.cross_process_channel_subscribers.size()) {
    return false;
  }

  for (const auto& service : state.cross_process_channel_services) {
    std::vector<const CrossProcessChannelSubscriberPlanRecord*> members;
    for (const auto& subscriber : state.cross_process_channel_subscribers) {
      if (subscriber.service_id == service.service_id) {
        members.push_back(&subscriber);
      }
    }
    if (members.empty() || static_cast<std::size_t>(service.max_subscribers) != members.size() ||
        !std::is_sorted(
            service.target_process_groups.begin(),
            service.target_process_groups.end(),
            [](ProcessGroupId left, ProcessGroupId right) { return left.value < right.value; }) ||
        std::adjacent_find(service.target_process_groups.begin(),
                           service.target_process_groups.end()) !=
            service.target_process_groups.end()) {
      return false;
    }
    const auto anchor =
        *std::min_element(members.begin(), members.end(), [](const auto* left, const auto* right) {
          return left->edge.value < right->edge.value;
        });
    const auto anchor_edge =
        std::find_if(state.payload_edges.begin(), state.payload_edges.end(), [&](const auto& edge) {
          return edge.edge == anchor->edge;
        });
    const PortPlanRecord* const anchor_target =
        anchor_edge == state.payload_edges.end() ? nullptr : dense_port(state, anchor_edge->target);
    if (anchor_edge == state.payload_edges.end() || anchor_target == nullptr ||
        service.edge != anchor_edge->edge || service.target != anchor_edge->target ||
        service.target_process_group != vertex_process_group(state, anchor_target->owner)) {
      return false;
    }

    std::uint32_t maximum_queue_depth = 0U;
    std::uint32_t maximum_borrowed_samples = 0U;
    std::uint32_t maximum_loaned_samples = 0U;
    std::vector<ProcessGroupId> target_groups;
    for (const auto* member : members) {
      const auto edge = std::find_if(state.payload_edges.begin(),
                                     state.payload_edges.end(),
                                     [&](const auto& item) { return item.edge == member->edge; });
      const PortPlanRecord* const target =
          edge == state.payload_edges.end() ? nullptr : dense_port(state, edge->target);
      if (edge == state.payload_edges.end() || target == nullptr) {
        return false;
      }
      const auto target_placement =
          std::find_if(state.vertex_process_placements.begin(),
                       state.vertex_process_placements.end(),
                       [&](const auto& placement) { return placement.vertex == target->owner; });
      if (target_placement == state.vertex_process_placements.end()) {
        return false;
      }
      maximum_queue_depth = std::max(maximum_queue_depth, edge->queue_depth);
      maximum_borrowed_samples = std::max(maximum_borrowed_samples, edge->queue_depth);
      maximum_loaned_samples =
          std::max(maximum_loaned_samples, edge->producer_max_emits_per_compute);
      target_groups.push_back(target_placement->process_group);
    }
    std::sort(target_groups.begin(),
              target_groups.end(),
              [](ProcessGroupId left, ProcessGroupId right) { return left.value < right.value; });
    target_groups.erase(std::unique(target_groups.begin(), target_groups.end()),
                        target_groups.end());
    if (service.max_subscriber_queue_depth != maximum_queue_depth ||
        service.max_subscriber_borrowed_samples != maximum_borrowed_samples ||
        service.target_process_groups != target_groups) {
      return false;
    }

    const auto publisher = std::find_if(
        state.cross_process_channel_publishers.begin(),
        state.cross_process_channel_publishers.end(),
        [&](const auto& candidate) { return candidate.service_id == service.service_id; });
    if (publisher == state.cross_process_channel_publishers.end() ||
        !valid_cross_process_publisher(*anchor_edge, service, *publisher) ||
        publisher->max_loaned_samples != maximum_loaned_samples) {
      return false;
    }

    if (service.service_id !=
        cross_process_service_id(state, service.source, service.queue_policy, service.edge)) {
      return false;
    }
  }
  return true;
}

}  // namespace holoscan::detail
