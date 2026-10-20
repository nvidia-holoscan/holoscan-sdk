/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_channel_plan_binding.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../channel/channel_payload_limits.hpp"
#include "../channel/endpoint_catalog_state.hpp"
#include "../channel/pod_codec.hpp"
#include "../compiler/execution_plan_contracts.hpp"
#include "../compiler/execution_plan_lookup.hpp"
#include "../core/tensor_allocation_record_validation.hpp"
#include "tensor_boundary_contract_comparison.hpp"

namespace holoscan::detail {

// Cold plan, layout, endpoint, and process-local Channel contract validation.

namespace {

[[nodiscard]] ChannelQueuePolicy channel_queue_policy(const QueuePolicy policy) noexcept {
  switch (policy) {
    case QueuePolicy::kBlocking:
      return ChannelQueuePolicy::kBlocking;
    case QueuePolicy::kDropOldest:
      return ChannelQueuePolicy::kDropOldest;
    default:
      return ChannelQueuePolicy::kBlocking;
  }
}

[[nodiscard]] bool supported_queue_policy(const QueuePolicy policy) noexcept {
  return policy == QueuePolicy::kBlocking || policy == QueuePolicy::kDropOldest;
}

}  // namespace

bool unique_valid_vertices(const std::span<const VertexId> vertices) noexcept {
  if (vertices.empty()) {
    return false;
  }
  for (auto current = vertices.begin(); current != vertices.end(); ++current) {
    if (!*current || std::find(std::next(current), vertices.end(), *current) != vertices.end()) {
      return false;
    }
  }
  return true;
}

expected<void, Error> validate_import_generation_capacities(
    const std::span<const ImportedGenerationCapacityPlanRecord> records) {
  if (records.empty()) {
    return {};
  }
  std::vector<std::pair<std::string_view, std::uint32_t>> services;
  services.reserve(records.size());
  for (auto current = records.begin(); current != records.end(); ++current) {
    const bool duplicate_topic = std::find_if(records.begin(), current, [&](const auto& prior) {
                                   return prior.topic == current->topic;
                                 }) != current;
    const bool restart_capacity_valid = current->restart_capable_minimum_two
                                            ? current->endpoint_max_generations >= 2U &&
                                                  current->candidate_reservation_capacity == 1U
                                            : current->endpoint_max_generations == 1U &&
                                                  current->candidate_reservation_capacity == 0U;
    if (!current->topic || current->runtime_scope_id.empty() ||
        current->endpoint_or_service_id.empty() || duplicate_topic ||
        current->endpoint_max_generations == 0U || current->service_max_generations == 0U ||
        current->runtime_max_generations == 0U ||
        current->service_max_generations < current->endpoint_max_generations ||
        current->release_record_capacity < current->endpoint_max_generations ||
        current->quarantine_record_capacity < current->endpoint_max_generations ||
        !current->effective_profile.has_value() || current->attach_max_wait_ns.value_or(1U) == 0U ||
        !restart_capacity_valid || !current->no_wrap_aba_generation ||
        !current->fallback_rejected) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const auto service = std::find_if(services.begin(), services.end(), [&](const auto& item) {
      return item.first == current->endpoint_or_service_id;
    });
    if (service == services.end()) {
      services.emplace_back(current->endpoint_or_service_id, current->service_max_generations);
    } else if (service->second != current->service_max_generations) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }
  std::uint64_t runtime_capacity = 0U;
  for (const auto& [service, capacity] : services) {
    static_cast<void>(service);
    runtime_capacity += capacity;
    if (runtime_capacity > std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }
  if (std::any_of(records.begin(), records.end(), [&](const auto& record) {
        return record.runtime_max_generations != runtime_capacity;
      })) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return {};
}

expected<ChannelOptions, Error> process_local_service_options(
    const ProcessLocalChannelServicePlanRecord& service) noexcept {
  if (!service.source || service.service_id.empty() ||
      (!service.tensor_payload && service.payload_size_bytes == 0U) ||
      (!service.tensor_payload &&
       !channel_internal::inline_payload_capacity(service.payload_size_bytes)) ||
      service.max_subscribers == 0U || service.max_subscriber_queue_depth == 0U ||
      service.max_subscriber_borrowed_samples == 0U || service.publisher_history_depth != 0U ||
      (service.cross_process_transport && !service.event_wake) ||
      !supported_queue_policy(service.queue_policy)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  ChannelOptions options{};
  options.queue_policy = channel_queue_policy(service.queue_policy);
  options.max_subscriber_queue_depth = service.max_subscriber_queue_depth;
  options.subscriber_queue_depth = service.max_subscriber_queue_depth;
  options.max_subscriber_borrowed_samples = service.max_subscriber_borrowed_samples;
  options.max_borrowed_samples = service.max_subscriber_borrowed_samples;
  options.publisher_history_depth = service.publisher_history_depth;
  options.expected_recipient_count = service.max_subscribers;
  options.enable_event_wake = service.event_wake;
  options.tensor_transfer_mode = service.cross_process_transport
                                     ? ChannelTensorTransferMode::kBufferHandle
                                     : ChannelTensorTransferMode::kSameProcessSidecar;
  return options;
}

expected<ChannelOptions, Error> external_service_options(
    const ChannelServiceStaticPlanRecord& service,
    const ChannelPublisherEndpointPlanRecord& publisher,
    const ChannelSubscriberEndpointPlanRecord& subscriber) noexcept {
  if (!service.topic || service.service_id.empty() || service.max_subscriber_queue_depth == 0U ||
      service.max_subscriber_borrowed_samples == 0U || service.publisher_history_depth != 0U ||
      !service.event_wake || publisher.topic != service.topic ||
      subscriber.topic != service.topic || publisher.service_id != service.service_id ||
      subscriber.service_id != service.service_id || publisher.max_loaned_samples == 0U ||
      subscriber.requested_queue_depth == 0U || subscriber.max_borrowed_samples == 0U ||
      !subscriber.event_wake || !supported_queue_policy(service.queue_policy)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  ChannelOptions options{};
  options.queue_policy = channel_queue_policy(service.queue_policy);
  options.max_subscriber_queue_depth = service.max_subscriber_queue_depth;
  options.subscriber_queue_depth = subscriber.requested_queue_depth;
  options.max_subscriber_borrowed_samples = service.max_subscriber_borrowed_samples;
  options.max_borrowed_samples = subscriber.max_borrowed_samples;
  options.publisher_history_depth = service.publisher_history_depth;
  options.max_loaned_samples = publisher.max_loaned_samples;
  options.expected_recipient_count = 1U;
  options.enable_event_wake = service.event_wake;
  options.replacement_attribution_required = publisher.replacement_attribution_required;
  options.tensor_transfer_mode = ChannelTensorTransferMode::kBufferHandle;
  return options;
}

expected<std::vector<TensorFieldServiceStorageContract>, Error>
// Resolve every logical descriptor field against one coherent transfer and allocation inventory.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
external_descriptor_tensor_storage_contracts(
    const ExecutionPlanState& plan, const ExternalTopicId topic, const PortId graph_port,
    const std::span<const DescriptorTensorTopicFieldRecord> logical_fields) {
  if (logical_fields.empty()) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  std::vector<TensorFieldServiceStorageContract> result;
  result.reserve(logical_fields.size());
  for (const auto& field : logical_fields) {
    const auto& logical = field.tensor;
    if (!logical.memory_kind || !logical.dtype || !logical.rank || !logical.max_byte_span ||
        *logical.max_byte_span == 0U) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    std::optional<ProcessGroupId> process_group;
    std::optional<DeviceId> device;
    std::size_t transfer_count = 0U;
    for (const auto& transfer : plan.tensor_transfer_plans) {
      if (transfer.topic != topic || transfer.field_id != field.field_id ||
          transfer.tensor_index != field.tensor_index ||
          transfer.disposition != PlanDisposition::kAccepted) {
        continue;
      }
      if (!transfer.interface_process_group ||
          (process_group && *process_group != transfer.interface_process_group) ||
          (device && *device != transfer.interface_device)) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      process_group = transfer.interface_process_group;
      device = transfer.interface_device;
      ++transfer_count;
    }
    if (transfer_count == 0U || !process_group || !device) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const auto* pool =
        find_tensor_interface_pool_by_topic_field(plan, topic, field.field_id, field.tensor_index);
    const bool copy_allowed =
        logical.foreign_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable;
    if (copy_allowed != (pool != nullptr) ||
        (pool != nullptr &&
         (pool->graph_port != graph_port || pool->allocator_process_group != *process_group ||
          pool->device != *device || pool->memory_kind != *logical.memory_kind ||
          pool->emit_policy != logical.foreign_emit_policy || pool->slot_count == 0U))) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    result.push_back(TensorFieldServiceStorageContract{
        .field_id = field.field_id,
        .tensor_index = field.tensor_index,
        .max_byte_span = *logical.max_byte_span,
        .pool_id = pool != nullptr ? pool->pool_id : std::string{},
        .dtype = logical.dtype,
        .rank = logical.rank,
        .memory_kind = *logical.memory_kind,
        .device = *device,
        .tensor_emit_policy = logical.foreign_emit_policy,
    });
  }
  return result;
}
[[nodiscard]] bool descriptor_tensor_identity(const PayloadIdentityRecord& identity) noexcept {
  return identity.kind == PayloadIdentityKind::kSchema &&
         identity.payload_kind == PayloadKind::kDescriptorBacked && !identity.scalar &&
         identity.schema.has_value() && !identity.pod;
}

[[nodiscard]] bool process_local_descriptor_storage_placement(const MemoryKind memory_kind,
                                                              const DeviceId device) noexcept {
  // Pinned DLPack identity is device zero regardless of whether a sibling cross-process arm
  // requires a private CUDA synchronization device in the compiler-owned allocation.
  return (memory_kind == MemoryKind::kHost && !device.valid()) ||
         memory_kind == MemoryKind::kPinnedHost ||
         ((memory_kind == MemoryKind::kCudaDevice || memory_kind == MemoryKind::kCudaManaged) &&
          device.valid());
}

[[nodiscard]] bool singular_descriptor_storage_contract(
    const ProcessLocalChannelServicePlanRecord& service) noexcept {
  return descriptor_tensor_identity(service.payload_identity) && service.tensor_fields.empty() &&
         !service.tensor_pool_id.empty() && service.tensor_max_byte_span != 0U &&
         process_local_descriptor_storage_placement(service.tensor_memory_kind,
                                                    service.tensor_device);
}

[[nodiscard]] bool descriptor_tensor_identity_equal(const PayloadIdentityRecord& left,
                                                    const PayloadIdentityRecord& right) noexcept {
  return descriptor_tensor_identity(left) && descriptor_tensor_identity(right) &&
         payload_identity_exactly_equal(left, right);
}

[[nodiscard]] bool allocation_satisfies_input_layout(
    const TensorAllocationPlanRecord& allocation,
    const std::optional<TensorRepresentation>& input_layout) noexcept {
  if (!input_layout) {
    return true;
  }
  if (input_layout->memory_kind && *input_layout->memory_kind != allocation.memory_kind) {
    return false;
  }
  if (input_layout->dtype &&
      (!allocation.layout ||
       !dl_data_type_exactly_equal(input_layout->dtype, allocation.layout->dtype))) {
    return false;
  }
  if (input_layout->strides &&
      (!allocation.layout || input_layout->strides != allocation.layout->strides)) {
    return false;
  }
  return !input_layout->rank || (allocation.layout && allocation.layout->rank &&
                                 *input_layout->rank == *allocation.layout->rank);
}

namespace {

[[nodiscard]] expected<const PayloadEdgePlanRecord*, Error> resolve_unique_inline_payload_edge(
    const ExecutionPlanState& plan, const InlinePartitionExecutionPlanRecord& inline_plan,
    const PortId source, const std::optional<PortId> target) noexcept {
  const PayloadEdgePlanRecord* matched_edge = nullptr;
  for (const auto& edge : plan.payload_edges) {
    if (edge.source != source || (target && edge.target != *target) ||
        std::find(inline_plan.edges.begin(), inline_plan.edges.end(), edge.edge) ==
            inline_plan.edges.end()) {
      continue;
    }
    if (matched_edge != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    matched_edge = &edge;
  }
  return matched_edge;
}

[[nodiscard]] expected<const TensorAllocationPlanRecord*, Error> resolve_unique_tensor_allocation(
    const ExecutionPlanState& plan, const PortId output) noexcept {
  const TensorAllocationPlanRecord* allocation = nullptr;
  for (const auto& candidate : plan.tensor_allocations) {
    if (candidate.output != output) {
      continue;
    }
    if (allocation != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    allocation = &candidate;
  }
  return allocation;
}

void collect_sorted_field_allocations(
    const ExecutionPlanState& plan, const PortId output,
    std::vector<const TensorFieldAllocationPlanRecord*>& field_allocations) {
  for (const auto& candidate : plan.tensor_field_allocations) {
    if (candidate.output == output) {
      field_allocations.push_back(&candidate);
    }
  }
  std::sort(field_allocations.begin(),
            field_allocations.end(),
            [](const auto* left, const auto* right) { return left->field_id < right->field_id; });
}

struct InlineFieldAllocationContract {
  bool supported{};
  DeviceId device;
};

[[nodiscard]] InlineFieldAllocationContract validate_inline_field_allocations(
    const ExecutionPlanState& plan, const InlinePartitionExecutionPlanRecord& inline_plan,
    const PayloadEdgePlanRecord& edge, const ProcessLocalChannelServicePlanRecord& service,
    const TensorAllocationPlanRecord* direct_allocation,
    const std::span<const TensorFieldAllocationPlanRecord* const> field_allocations) noexcept {
  InlineFieldAllocationContract result{
      .supported = direct_allocation == nullptr && !field_allocations.empty() &&
                   service.tensor_fields.size() == field_allocations.size(),
  };
  for (std::size_t index = 0U; result.supported && index < field_allocations.size(); ++index) {
    const auto& field = *field_allocations[index];
    const auto& storage = service.tensor_fields[index];
    const bool layout_matches = field.layout && storage.rank == field.layout->rank &&
                                dl_data_type_exactly_equal(storage.dtype, field.layout->dtype);
    result.supported =
        validate_tensor_field_allocation_record(field, plan.plan_generation) &&
        field.output == edge.source && field.allocator_process_group == inline_plan.process_group &&
        (field.memory_kind == MemoryKind::kHost || field.memory_kind == MemoryKind::kCudaDevice) &&
        field.tensor_index == index &&
        (index == 0U || field_allocations[index - 1U]->field_id < field.field_id) &&
        storage.field_id == field.field_id && storage.tensor_index == field.tensor_index &&
        storage.max_byte_span == field.bounds.max_byte_span &&
        storage.memory_kind == field.memory_kind && storage.device == field.device &&
        layout_matches &&
        (!field.device.valid() || !result.device.valid() || field.device == result.device);
    if (result.supported && field.device.valid()) {
      result.device = field.device;
    }
  }
  return result;
}

}  // namespace

// Inline admission validates the complete plan, service, allocation, and layout contract.
[[nodiscard]] expected<std::optional<InlineDescriptorTensorContract>, Error>
find_inline_descriptor_tensor_contract(
    const ExecutionPlanState& plan, PortId source, std::optional<PortId> target,
    const ProcessLocalChannelServicePlanRecord& service) noexcept {
  if (!plan.inline_partition_execution) {
    return std::optional<InlineDescriptorTensorContract>{};
  }
  const InlinePartitionExecutionPlanRecord& inline_plan = *plan.inline_partition_execution;
  auto resolved_edge = resolve_unique_inline_payload_edge(plan, inline_plan, source, target);
  if (!resolved_edge) {
    return make_unexpected(std::move(resolved_edge).error());
  }
  const PayloadEdgePlanRecord* const matched_edge = *resolved_edge;
  if (matched_edge == nullptr || !descriptor_tensor_identity(matched_edge->payload_identity)) {
    return std::optional<InlineDescriptorTensorContract>{};
  }

  const auto* source_port =
      find_port_by_id_and_direction(plan, matched_edge->source, PortDirection::kOutput);
  const auto* target_port =
      find_port_by_id_and_direction(plan, matched_edge->target, PortDirection::kInput);
  auto resolved_allocation = resolve_unique_tensor_allocation(plan, matched_edge->source);
  if (!resolved_allocation) {
    return make_unexpected(std::move(resolved_allocation).error());
  }
  const TensorAllocationPlanRecord* const allocation = *resolved_allocation;
  std::vector<const TensorFieldAllocationPlanRecord*> field_allocations;
  collect_sorted_field_allocations(plan, matched_edge->source, field_allocations);
  const auto publisher_count = std::count_if(plan.process_local_channel_publishers.begin(),
                                             plan.process_local_channel_publishers.end(),
                                             [&](const auto& row) {
                                               return row.source == matched_edge->source &&
                                                      row.service_id == service.service_id;
                                             });
  const auto subscriber_count = std::count_if(plan.process_local_channel_subscribers.begin(),
                                              plan.process_local_channel_subscribers.end(),
                                              [&](const auto& row) {
                                                return row.edge == matched_edge->edge &&
                                                       row.source == matched_edge->source &&
                                                       row.target == matched_edge->target &&
                                                       row.service_id == service.service_id;
                                              });
  const bool supported_direct_allocation =
      allocation != nullptr &&
      validate_tensor_allocation_record(*allocation, plan.plan_generation) &&
      allocation->allocator_process_group == inline_plan.process_group &&
      (allocation->memory_kind == MemoryKind::kHost ||
       allocation->memory_kind == MemoryKind::kCudaDevice);
  const InlineFieldAllocationContract field_contract = validate_inline_field_allocations(
      plan, inline_plan, *matched_edge, service, allocation, field_allocations);
  const bool supported_allocation =
      (supported_direct_allocation && field_allocations.empty() && service.tensor_fields.empty()) ||
      field_contract.supported;
  const bool supported_storage =
      supported_direct_allocation
          ? service.tensor_pool_id == allocation->pool_id &&
                service.tensor_max_byte_span == allocation->bounds.max_byte_span &&
                service.tensor_memory_kind == allocation->memory_kind &&
                service.tensor_device == allocation->device
          : service.tensor_pool_id.empty() && service.tensor_max_byte_span == 0U &&
                service.tensor_memory_kind == MemoryKind::kUnknown &&
                !service.tensor_device.valid();
  const bool supported_service =
      source_port != nullptr && service.source == matched_edge->source &&
      descriptor_tensor_identity_equal(service.payload_identity, matched_edge->payload_identity) &&
      service.tensor_payload &&
      service.payload_size_bytes == source_port->serialized_payload_capacity_bytes &&
      supported_storage &&
      service.delivery_kind == ProcessLocalChannelDeliveryKind::kDirectRetained &&
      !service.cross_process_transport && !service.event_wake && service.max_subscribers == 1U;
  if (source_port == nullptr || target_port == nullptr || !supported_allocation ||
      !supported_service || matched_edge->disposition != PlanDisposition::kAccepted ||
      matched_edge->locality != ChannelLocalityKind::kProcessLocal || matched_edge->fanout != 1U ||
      !descriptor_tensor_identity_equal(source_port->payload_identity,
                                        matched_edge->payload_identity) ||
      !descriptor_tensor_identity_equal(target_port->payload_identity,
                                        matched_edge->payload_identity) ||
      source_port->serialized_payload_capacity_bytes == 0U ||
      source_port->serialized_payload_capacity_bytes !=
          target_port->serialized_payload_capacity_bytes ||
      (supported_direct_allocation &&
       !allocation_satisfies_input_layout(*allocation, target_port->tensor_layout)) ||
      publisher_count != 1 || subscriber_count != 1) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (field_contract.supported) {
    return InlineDescriptorTensorContract{
        .source = matched_edge->source,
        .target = matched_edge->target,
        .device = field_contract.device,
    };
  }
  return InlineDescriptorTensorContract{
      .source = matched_edge->source,
      .target = matched_edge->target,
      .max_byte_span = allocation->bounds.max_byte_span,
      .memory_kind = allocation->memory_kind,
      .device = allocation->device,
      .output_layout = allocation->layout,
      .input_layout = target_port->tensor_layout,
  };
}

[[nodiscard]] bool runtime_codec_matches_plan(const RuntimePortTypeOps* ops,
                                              const PortPlanRecord& port,
                                              std::uint64_t service_payload_capacity,
                                              const bool process_local_schema_envelope) noexcept {
  if (ops == nullptr ||
      (ops->wire_capacity == 0U &&
       ops->codec_family != RuntimePayloadCodecFamily::kProcessLocalObject) ||
      ops->wire_capacity > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  const std::uint64_t exact_local_service_capacity =
      ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema &&
              process_local_schema_envelope
          ? ops->wire_capacity + kSchemaEnvelopeHeaderSize
          : ops->wire_capacity;
  switch (ops->codec_family) {
    case RuntimePayloadCodecFamily::kInlinePod:
      return exact_local_service_capacity == service_payload_capacity && ops->inline_pod &&
             ops->payload_kind == PayloadKind::kInlineTyped && ops->encode_payload == nullptr &&
             ops->wire_capacity == ops->payload_size &&
             port.payload_identity.kind == PayloadIdentityKind::kScalar &&
             port.serialized_payload_capacity_bytes == 0U;
    case RuntimePayloadCodecFamily::kCustomPod:
      return exact_local_service_capacity == service_payload_capacity && ops->inline_pod &&
             ops->payload_kind == PayloadKind::kInlineTyped && ops->encode_payload == nullptr &&
             ops->wire_capacity == ops->payload_size &&
             port.payload_identity.kind == PayloadIdentityKind::kPod &&
             port.payload_identity.pod.has_value() &&
             channel_internal::PodFieldwiseCodec::validate(*port.payload_identity.pod)
                 .has_value() &&
             port.serialized_payload_capacity_bytes == 0U;
    case RuntimePayloadCodecFamily::kGeneratedSchema:
      return !ops->inline_pod && ops->payload_kind == PayloadKind::kSerializedBytes &&
             ops->encode_payload != nullptr &&
             (!ops->persistent_sample_storage ||
              (ops->prepare_sample_storage != nullptr && ops->reset_sample_storage != nullptr &&
               ops->destroy_sample_storage != nullptr && ops->assign_sample_storage != nullptr)) &&
             port.payload_identity.kind == PayloadIdentityKind::kSchema &&
             ops->transport_profile_tag == port.payload_identity.transport_profile_tag &&
             port.serialized_payload_capacity_bytes == ops->wire_capacity &&
             (port.direction == PortDirection::kOutput
                  ? exact_local_service_capacity == service_payload_capacity
                  : service_payload_capacity >=
                            (process_local_schema_envelope ? kSchemaEnvelopeHeaderSize : 0U) &&
                        ops->wire_capacity >=
                            service_payload_capacity -
                                (process_local_schema_envelope ? kSchemaEnvelopeHeaderSize : 0U));
    case RuntimePayloadCodecFamily::kProcessLocalObject:
      return process_local_schema_envelope && service_payload_capacity == 0U &&
             ops->wire_capacity == 0U && !ops->inline_pod &&
             ops->payload_kind == PayloadKind::kInlineTyped && ops->encode_payload == nullptr &&
             ops->construct_sample == nullptr && ops->construct_sample_from_value != nullptr &&
             ops->retain_sample_payload_for_deferred_release != nullptr &&
             port.serialized_payload_capacity_bytes == 0U &&
             is_py_object_payload_identity(port.payload_identity);
    case RuntimePayloadCodecFamily::kDescriptorTensorSchema:
    case RuntimePayloadCodecFamily::kUnsupported:
      return false;
  }
  return false;
}

namespace {

[[nodiscard]] bool valid_cross_process_descriptor_field_storage(
    const TensorFieldServiceStorageContract& field) noexcept {
  return !field.pool_id.empty() && field.max_byte_span != 0U && field.dtype && field.rank &&
         ((field.memory_kind == MemoryKind::kHost && !field.device.valid()) ||
          ((field.memory_kind == MemoryKind::kPinnedHost ||
            field.memory_kind == MemoryKind::kCudaDevice) &&
           field.device.valid()));
}

[[nodiscard]] bool descriptor_service_storage_matches_plan(
    const ProcessLocalChannelServicePlanRecord& service,
    const std::uint32_t tensor_count) noexcept {
  if (!service.tensor_fields.empty()) {
    if (service.tensor_fields.size() != tensor_count) {
      return false;
    }
    for (std::size_t index = 0U; index < service.tensor_fields.size(); ++index) {
      const auto& field = service.tensor_fields[index];
      if (field.tensor_index != index || (service.cross_process_transport &&
                                          !valid_cross_process_descriptor_field_storage(field))) {
        return false;
      }
    }
    return true;
  }

  if (!service.cross_process_transport && tensor_count > 1U) {
    return true;
  }
  if (tensor_count != 1U) {
    return false;
  }
  if (service.cross_process_transport) {
    return !service.tensor_pool_id.empty() && service.tensor_max_byte_span != 0U &&
           ((service.tensor_memory_kind == MemoryKind::kHost && !service.tensor_device.valid()) ||
            ((service.tensor_memory_kind == MemoryKind::kPinnedHost ||
              service.tensor_memory_kind == MemoryKind::kCudaDevice) &&
             service.tensor_device.valid()));
  }
  return (!service.tensor_pool_id.empty() && service.tensor_max_byte_span != 0U &&
          process_local_descriptor_storage_placement(service.tensor_memory_kind,
                                                     service.tensor_device)) ||
         (service.tensor_pool_id.empty() && service.tensor_max_byte_span == 0U &&
          service.tensor_memory_kind == MemoryKind::kUnknown && !service.tensor_device.valid());
}

[[nodiscard]] bool runtime_descriptor_tensor_matches_plan(
    const RuntimePortTypeOps& ops, const PortPlanRecord& port,
    const ProcessLocalChannelServicePlanRecord& service) noexcept {
  const std::uint32_t tensor_count = ops.descriptor_tensor_static_count;
  return descriptor_service_storage_matches_plan(service, tensor_count) &&
         service.payload_size_bytes == ops.wire_capacity && ops.wire_capacity != 0U &&
         ops.transport_profile_tag == port.payload_identity.transport_profile_tag &&
         ops.wire_capacity == port.serialized_payload_capacity_bytes &&
         ops.codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema &&
         ops.payload_kind == PayloadKind::kDescriptorBacked && ops.persistent_sample_storage &&
         tensor_count != 0U && tensor_count <= channel_internal::kMaximumDescriptorTensorFields &&
         ops.descriptor_offset <= ops.wire_capacity &&
         static_cast<std::size_t>(tensor_count) * sizeof(PayloadDescriptor) <=
             ops.wire_capacity - ops.descriptor_offset &&
         ops.descriptor_tensor_count != nullptr && ops.descriptor_tensor_at != nullptr &&
         (tensor_count != 1U ||
          (ops.descriptor_tensor != nullptr && ops.assign_descriptor_tensor_sample != nullptr)) &&
         ops.encode_descriptor_tensor_sample != nullptr &&
         ops.descriptor_tensor_sample_metadata != nullptr &&
         ops.descriptor_tensor_view_metadata_at != nullptr &&
         ops.assign_descriptor_tensors_sample != nullptr && ops.prepare_sample_storage != nullptr &&
         ops.reset_sample_storage != nullptr && ops.destroy_sample_storage != nullptr;
}

[[nodiscard]] bool runtime_raw_tensor_matches_plan(
    const RuntimePortTypeOps& ops, const PortPlanRecord& port,
    const ProcessLocalChannelServicePlanRecord& service) noexcept {
  const bool pool_backed = !service.tensor_pool_id.empty();
  const bool supported_pool_placement =
      (service.tensor_memory_kind == MemoryKind::kHost && !service.tensor_device.valid()) ||
      service.tensor_memory_kind == MemoryKind::kPinnedHost ||
      ((service.tensor_memory_kind == MemoryKind::kCudaDevice ||
        service.tensor_memory_kind == MemoryKind::kCudaManaged) &&
       service.tensor_device.valid());
  const bool pool_contract = pool_backed
                                 ? service.tensor_max_byte_span != 0U && supported_pool_placement
                                 : valid_pass_through_tensor_storage(TensorServiceStorageView{
                                       .max_byte_span = service.tensor_max_byte_span,
                                       .pool_id = service.tensor_pool_id,
                                       .memory_kind = service.tensor_memory_kind,
                                       .device = service.tensor_device,
                                   });
  return service.payload_size_bytes == 0U && pool_contract &&
         &ops == runtime_port_type_ops<Tensor>() &&
         ops.codec_family == RuntimePayloadCodecFamily::kUnsupported &&
         ops.sample_size == sizeof(Sample<Tensor>) &&
         ops.sample_alignment == alignof(Sample<Tensor>) &&
         port.payload_identity.kind == PayloadIdentityKind::kTensor &&
         port.serialized_payload_capacity_bytes == 0U;
}

// A projected subscriber takes one Tensor member of the producer's schema sample, so it pairs a
// Tensor port with the producer's own descriptor-schema service rather than a Tensor service. The
// service's descriptor invariants are already matched against the producer in publisher
// materialization, so only the consumer side and the schema-service shape have to hold here.
[[nodiscard]] bool runtime_projected_tensor_matches_plan(
    const RuntimePortTypeOps& ops, const PortPlanRecord& port,
    const ProcessLocalChannelServicePlanRecord& service) noexcept {
  return service.payload_size_bytes != 0U && &ops == runtime_port_type_ops<Tensor>() &&
         ops.codec_family == RuntimePayloadCodecFamily::kUnsupported &&
         ops.sample_size == sizeof(Sample<Tensor>) &&
         ops.sample_alignment == alignof(Sample<Tensor>) &&
         port.payload_identity.kind == PayloadIdentityKind::kTensor &&
         port.serialized_payload_capacity_bytes == 0U;
}

}  // namespace

// Runtime materialization conservatively matches every payload-family and Tensor plan invariant.
[[nodiscard]] bool runtime_tensor_matches_plan(const RuntimePortTypeOps* ops,
                                               const PortPlanRecord& port,
                                               const ProcessLocalChannelServicePlanRecord& service,
                                               const bool projected) noexcept {
  if (ops == nullptr || !service.tensor_payload) {
    return false;
  }
  if (projected) {
    return runtime_projected_tensor_matches_plan(*ops, port, service);
  }
  const bool descriptor_schema =
      port.payload_identity.kind == PayloadIdentityKind::kSchema &&
      port.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
  if (descriptor_schema) {
    return runtime_descriptor_tensor_matches_plan(*ops, port, service);
  }
  return runtime_raw_tensor_matches_plan(*ops, port, service);
}

[[nodiscard]] bool runtime_external_tensor_base_matches_plan(
    const RuntimePortTypeOps* ops, const PortPlanRecord& port,
    const ChannelServiceStaticPlanRecord& service, const ChannelEndpointRecord& endpoint) noexcept {
  const bool supported_memory_kind = endpoint.tensor && endpoint.tensor->memory_kind &&
                                     (*endpoint.tensor->memory_kind == MemoryKind::kHost ||
                                      *endpoint.tensor->memory_kind == MemoryKind::kPinnedHost ||
                                      *endpoint.tensor->memory_kind == MemoryKind::kCudaDevice);
  return service.tensor_payload && service.payload_identity.kind == PayloadIdentityKind::kTensor &&
         service.payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
         supported_memory_kind && endpoint.service_static.tensor &&
         endpoint.tensor->max_byte_span && *endpoint.tensor->max_byte_span != 0U &&
         logical_tensor_requirements_exactly_equal(*endpoint.tensor,
                                                   *endpoint.service_static.tensor) &&
         endpoint.service_static.tensor->max_byte_span == endpoint.tensor->max_byte_span &&
         endpoint.service_static.tensor->strides == endpoint.tensor->strides &&
         ops == runtime_port_type_ops<Tensor>() &&
         ops->codec_family == RuntimePayloadCodecFamily::kUnsupported &&
         ops->sample_size == sizeof(Sample<Tensor>) &&
         ops->sample_alignment == alignof(Sample<Tensor>) &&
         port.payload_identity.kind == PayloadIdentityKind::kTensor &&
         port.payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
         port.serialized_payload_capacity_bytes == 0U;
}

[[nodiscard]] bool runtime_external_tensor_matches_plan(
    const RuntimePortTypeOps* ops, const PortPlanRecord& port,
    const ChannelServiceStaticPlanRecord& service, const ChannelEndpointRecord& endpoint) noexcept {
  const bool descriptor_schema =
      service.payload_identity.kind == PayloadIdentityKind::kSchema &&
      service.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
  if (descriptor_schema) {
    return ops != nullptr && service.tensor_payload && !service.tensor_fields.empty() &&
           !endpoint.tensor && !endpoint.service_static.tensor &&
           descriptor_tensor_fields_equal(endpoint.tensor_fields, service.tensor_fields) &&
           descriptor_tensor_fields_equal(endpoint.service_static.tensor_fields,
                                          service.tensor_fields) &&
           ops->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema &&
           ops->payload_kind == PayloadKind::kDescriptorBacked &&
           ops->transport_profile_tag == service.payload_identity.transport_profile_tag &&
           ops->descriptor_tensor_static_count == service.tensor_fields.size() &&
           ops->persistent_sample_storage &&
           ops->descriptor_tensor_static_count <=
               channel_internal::kMaximumDescriptorTensorFields &&
           ops->wire_capacity != 0U && ops->descriptor_offset <= ops->wire_capacity &&
           static_cast<std::size_t>(ops->descriptor_tensor_static_count) *
                   sizeof(PayloadDescriptor) <=
               ops->wire_capacity - ops->descriptor_offset &&
           ops->wire_capacity == port.serialized_payload_capacity_bytes &&
           ops->wire_capacity == endpoint.schema_writer_max_serialized_bytes &&
           port.payload_identity.kind == PayloadIdentityKind::kSchema &&
           port.payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
           descriptor_tensor_identity_equal(port.payload_identity, service.payload_identity) &&
           descriptor_tensor_identity_equal(endpoint.payload_identity, service.payload_identity) &&
           ops->descriptor_tensor_count != nullptr && ops->descriptor_tensor_at != nullptr &&
           (ops->descriptor_tensor_static_count != 1U ||
            (ops->descriptor_tensor != nullptr &&
             ops->assign_descriptor_tensor_sample != nullptr)) &&
           ops->encode_descriptor_tensor_sample != nullptr &&
           ops->descriptor_tensor_sample_metadata != nullptr &&
           ops->descriptor_tensor_view_metadata_at != nullptr &&
           ops->assign_descriptor_tensors_sample != nullptr &&
           ops->prepare_sample_storage != nullptr && ops->reset_sample_storage != nullptr &&
           ops->destroy_sample_storage != nullptr;
  }
  if (!runtime_external_tensor_base_matches_plan(ops, port, service, endpoint) ||
      !endpoint.tensor) {
    return false;
  }
  return endpoint.tensor->strides == TensorStridePolicy::kContiguousOnly;
}

[[nodiscard]] bool runtime_external_tensor_uses_unsupported_strides(
    const RuntimePortTypeOps* ops, const PortPlanRecord& port,
    const ChannelServiceStaticPlanRecord& service, const ChannelEndpointRecord& endpoint) noexcept {
  const bool descriptor_schema =
      service.payload_identity.kind == PayloadIdentityKind::kSchema &&
      service.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
  if (descriptor_schema) {
    return runtime_external_tensor_matches_plan(ops, port, service, endpoint) &&
           std::any_of(
               service.tensor_fields.begin(), service.tensor_fields.end(), [](const auto& field) {
                 return field.tensor.strides == TensorStridePolicy::kBoundedStrided;
               });
  }
  if (!runtime_external_tensor_base_matches_plan(ops, port, service, endpoint) ||
      !endpoint.tensor) {
    return false;
  }
  return endpoint.tensor->strides == TensorStridePolicy::kBoundedStrided;
}

[[nodiscard]] const EndpointClockHealthPlanRecord* find_endpoint_clock_health(
    const ExecutionPlanState& plan, PortId endpoint) noexcept {
  const auto found = std::find_if(plan.endpoint_clock_health.begin(),
                                  plan.endpoint_clock_health.end(),
                                  [&](const auto& record) { return record.endpoint == endpoint; });
  return found == plan.endpoint_clock_health.end() ? nullptr : &*found;
}

[[nodiscard]] expected<ChannelEndpointCatalog::EndpointView, Error>
find_validated_catalog_endpoint_by_boundary_port(const ExecutionPlanState& plan,
                                                 PortId boundary) noexcept {
  const auto* state = ChannelEndpointCatalogAccess::state(plan.endpoint_catalog);
  if (state == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const auto found =
      std::find_if(state->endpoints.begin(), state->endpoints.end(), [&](const auto& endpoint) {
        return endpoint.boundary_port == boundary;
      });
  if (found == state->endpoints.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return plan.endpoint_catalog.endpoint(found->name);
}

[[nodiscard]] expected<CatalogCaptureClockBinding, Error> bind_catalog_capture_clock(
    const ExecutionPlanState& plan, ChannelEndpointCatalog::EndpointView endpoint,
    ClockId plan_capture_clock) noexcept {
  const ClockId catalog_capture_clock = ChannelEndpointCatalogAccess::capture_clock_id(endpoint);
  if (catalog_capture_clock.valid() != plan_capture_clock.valid()) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (!catalog_capture_clock.valid()) {
    return CatalogCaptureClockBinding{};
  }

  const auto* catalog = ChannelEndpointCatalogAccess::state(plan.endpoint_catalog);
  if (catalog == nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const auto catalog_clock =
      std::find_if(catalog->clocks.begin(), catalog->clocks.end(), [&](const auto& candidate) {
        return candidate.id == catalog_capture_clock.value;
      });
  const auto* plan_clock = find_clock_by_id(plan, plan_capture_clock);
  if (catalog_clock == catalog->clocks.end() || plan_clock == nullptr ||
      catalog_clock->key != plan_clock->key) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return CatalogCaptureClockBinding{
      .catalog = catalog_capture_clock,
      .plan = plan_capture_clock,
  };
}

namespace {

[[nodiscard]] expected<PortId, Error> resolve_internal_interface_port(
    const ExecutionPlanState& plan, PortId endpoint,
    const PortDirection internal_direction) noexcept {
  for (std::size_t depth = 0U; depth <= plan.interface_port_bindings.size(); ++depth) {
    const PortPlanRecord* const port =
        find_port_by_id_and_direction(plan, endpoint, internal_direction);
    if (port == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (port->owner_kind == PortOwnerKind::kOperator) {
      return endpoint;
    }
    if (port->owner_kind != PortOwnerKind::kGraphInterface) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const auto binding =
        std::find_if(plan.interface_port_bindings.begin(),
                     plan.interface_port_bindings.end(),
                     [&](const auto& candidate) { return candidate.public_port == endpoint; });
    if (binding == plan.interface_port_bindings.end() || binding->direction != internal_direction) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    endpoint = binding->internal_port;
  }
  return make_unexpected(Error{ErrorCode::kProtocolError});
}

}  // namespace

// Resolution follows the frozen boundary/interface chain and rejects ambiguity at every hop.
[[nodiscard]] expected<PortId, Error> external_internal_port(const ExecutionPlanState& plan,
                                                             PortId boundary,
                                                             bool imported) noexcept {
  const PortDirection internal_direction =
      imported ? PortDirection::kInput : PortDirection::kOutput;
  PortId result{};
  for (const auto& edge : plan.payload_edges) {
    const bool matches = imported ? edge.source == boundary : edge.target == boundary;
    if (!matches) {
      continue;
    }
    const PortId candidate = imported ? edge.target : edge.source;
    if (result && result != candidate) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    result = candidate;
  }
  if (result) {
    return resolve_internal_interface_port(plan, result, internal_direction);
  }
  const auto binding =
      std::find_if(plan.interface_port_bindings.begin(),
                   plan.interface_port_bindings.end(),
                   [&](const auto& candidate) { return candidate.public_port == boundary; });
  if (binding != plan.interface_port_bindings.end()) {
    return resolve_internal_interface_port(plan, binding->internal_port, internal_direction);
  }
  return make_unexpected(Error{ErrorCode::kNotFound});
}

}  // namespace holoscan::detail
