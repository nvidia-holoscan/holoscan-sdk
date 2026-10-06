/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/channel_service_lowering.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <holoscan/memory/memory_buffer.hpp>

#include "channel/channel_payload_limits.hpp"
#include "compiler/channel_payload_contracts.hpp"
#include "compiler/detail/checked_arithmetic.hpp"
#include "compiler/execution_plan_contracts.hpp"
#include "compiler/execution_plan_identity.hpp"
#include "compiler/physical_plan_lowering_internal.hpp"
#include "compiler/snapshot_lookup.hpp"
#include "compiler/tensor_provider_compatibility.hpp"
#include "core/domain/tensor_layout_validation.hpp"

namespace holoscan::detail {
namespace {

constexpr std::string_view kTensorFanoutInsufficient{
    "PHYSICAL_TENSOR_FANOUT_CAPACITY_INSUFFICIENT"};
constexpr std::string_view kTensorRankUnsupported{"PHYSICAL_TENSOR_RANK_UNSUPPORTED"};
constexpr std::string_view kSchemaDescriptorEnvelopeCapacityExceeded{
    "CHANNEL_SCHEMA_DESCRIPTOR_ENVELOPE_CAPACITY_EXCEEDED"};
constexpr std::string_view kLocalEnumCrossProcessUnsupported{
    "CHANNEL_LOCAL_ENUM_CROSS_PROCESS_UNSUPPORTED"};
constexpr std::uint32_t kCrossProcessTensorMaxRank = 8U;

[[nodiscard]] std::string dtype_summary(const std::optional<DLDataType>& dtype) {
  if (!dtype) {
    return "any";
  }
  return std::to_string(dtype->code) + ":" + std::to_string(dtype->bits) + ":" +
         std::to_string(dtype->lanes);
}

[[nodiscard]] std::string layout_summary(const std::optional<TensorRepresentation>& layout) {
  if (!layout) {
    return "memory=any,dtype=any,rank=any";
  }
  const std::string memory =
      layout->memory_kind ? std::string(memory_kind_name(*layout->memory_kind)) : "any";
  const std::string rank = layout->rank ? std::to_string(*layout->rank) : "any";
  return "memory=" + memory + ",dtype=" + dtype_summary(layout->dtype) + ",rank=" + rank;
}

[[nodiscard]] std::string allocation_layout_summary(const TensorAllocationPlanRecord& allocation) {
  std::optional<TensorRepresentation> layout = allocation.layout;
  if (!layout) {
    layout.emplace();
  }
  layout->memory_kind = allocation.memory_kind;
  return layout_summary(layout);
}

[[nodiscard]] std::string template_layout_summary(
    const TensorProviderTemplatePlanRecord& provider) {
  std::optional<TensorRepresentation> layout = provider.layout;
  if (!layout) {
    layout.emplace();
  }
  layout->memory_kind = provider.memory_kind;
  return layout_summary(layout);
}

[[nodiscard]] std::string boundary_layout_summary(const TensorBoundaryContractRecord& provider) {
  return layout_summary(TensorRepresentation{
      .memory_kind = provider.memory_kind,
      .dtype = provider.dtype,
      .rank = provider.rank,
      .strides = provider.strides,
  });
}

[[nodiscard]] std::string interface_layout_summary(const TensorInterfacePoolPlanRecord& provider) {
  return boundary_layout_summary(provider.logical);
}

struct DescriptorTensorMemoryKindMismatch {
  std::optional<std::uint16_t> field_id;
  std::optional<TensorRepresentation> source;
  std::optional<TensorRepresentation> target;
};

[[nodiscard]] std::optional<DescriptorTensorMemoryKindMismatch>
find_descriptor_tensor_memory_kind_mismatch(const PortRecord& source,
                                            const PortRecord& target) noexcept {
  const auto port_source = resolved_layout(source);
  const auto port_target = resolved_layout(target);
  if (source.descriptor_tensor_field_count == 1U && target.descriptor_tensor_field_count == 1U &&
      port_source && port_source->memory_kind && port_target && port_target->memory_kind &&
      port_source->memory_kind != port_target->memory_kind) {
    return DescriptorTensorMemoryKindMismatch{.source = port_source, .target = port_target};
  }
  for (const auto& target_field : target.setup.tensor_fields) {
    const auto source_field =
        std::find_if(source.setup.tensor_fields.begin(),
                     source.setup.tensor_fields.end(),
                     [&](const auto& candidate) {
                       return candidate.field_id == target_field.field_id &&
                              candidate.tensor_index == target_field.tensor_index;
                     });
    // A present source layout is established before its required memory-kind field is inspected.
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    if (source_field != source.setup.tensor_fields.end() && source_field->tensor_layout &&
        source_field->tensor_layout->memory_kind && target_field.tensor_layout &&
        target_field.tensor_layout->memory_kind &&
        source_field->tensor_layout->memory_kind != target_field.tensor_layout->memory_kind) {
      return DescriptorTensorMemoryKindMismatch{
          .field_id = target_field.field_id,
          .source = source_field->tensor_layout,
          .target = target_field.tensor_layout,
      };
    }
    // NOLINTEND(bugprone-unchecked-optional-access)
  }
  return std::nullopt;
}

[[nodiscard]] bool cross_process_service_compatible(
    const CrossProcessChannelServicePlanRecord& service, PortId source,
    ProcessGroupId source_process_group, const TensorServiceContract& contract) noexcept {
  const auto service_view = tensor_service_contract_view(service);
  const auto contract_view = tensor_service_contract_view(contract);
  return service.source == source && service.source_process_group == source_process_group &&
         service_view.queue_policy == contract_view.queue_policy &&
         tensor_service_payload_storage_exactly_equal(service_view, contract_view) &&
         service_view.publisher_history_depth == contract_view.publisher_history_depth &&
         service_view.event_wake == contract_view.event_wake &&
         service.tensor_emit_policy == contract.tensor_emit_policy;
}

[[nodiscard]] bool append_process_local_channel_binding(
    PhysicalPlanLoweringResult& result, PlanGenerationId generation,
    const PayloadEdgePlanRecord& edge, const TensorServiceContract& contract,
    ProcessLocalChannelDeliveryKind delivery_kind) {
  // Only a direct-retained service hands the consumer the producer's own field storage. A queued
  // HoloMQ service re-materializes the sample from its envelope, and narrowing that back down to
  // one field is not implemented, so reject it here instead of letting it reach the runtime.
  if (edge.tensor_projection && delivery_kind != ProcessLocalChannelDeliveryKind::kDirectRetained) {
    return false;
  }
  auto matching = std::find_if(result.process_local_channel_services.begin(),
                               result.process_local_channel_services.end(),
                               [&](const auto& service) {
                                 return service.source == edge.source &&
                                        service.queue_policy == edge.queue_policy;
                               });
  if (matching == result.process_local_channel_services.end()) {
    const std::string service_id =
        process_local_channel_service_id(generation, edge.source, edge.queue_policy);
    result.process_local_channel_services.push_back(make_process_local_service_record(
        edge.source, service_id, contract, 0U, false, delivery_kind));
    result.process_local_channel_publishers.push_back(ProcessLocalChannelPublisherPlanRecord{
        .source = edge.source,
        .service_id = service_id,
        .max_loaned_samples = edge.producer_max_emits_per_compute,
        .replacement_attribution_required = edge.queue_policy != QueuePolicy::kBlocking,
    });
    matching = std::prev(result.process_local_channel_services.end());
  } else if (edge.tensor_projection &&
             matching->delivery_kind != ProcessLocalChannelDeliveryKind::kDirectRetained) {
    // The reused service must support projection independently of this edge's eligibility.
    return false;
  }

  std::uint32_t subscriber_count{};
  if (!checked_add(matching->max_subscribers, 1U, subscriber_count)) {
    return false;
  }
  matching->max_subscribers = subscriber_count;
  matching->max_subscriber_queue_depth =
      std::max(matching->max_subscriber_queue_depth, edge.queue_depth);
  matching->max_subscriber_borrowed_samples =
      std::max(matching->max_subscriber_borrowed_samples, edge.queue_depth);
  result.process_local_channel_subscribers.push_back(ProcessLocalChannelSubscriberPlanRecord{
      .edge = edge.edge,
      .source = edge.source,
      .target = edge.target,
      .service_id = matching->service_id,
      .requested_queue_depth = edge.queue_depth,
      .max_borrowed_samples = edge.queue_depth,
      .cache_capacity = edge.cache_capacity,
      // Every queued or retained cache candidate can own one unretired
      // Dispatcher witness. Selector-free arrival triggers have no cache,
      // while temporal selectors can retain more samples than the transport
      // queue, so freeze the larger admitted bound for the ingress ring.
      .readiness_capacity = std::max(edge.queue_depth, edge.cache_capacity),
      .event_wake = delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ,
      // Selection belongs to the subscriber; the service retains the source schema identity.
      .tensor_projection = edge.tensor_projection,
  });
  return true;
}

[[nodiscard]] bool append_cross_process_channel_binding(PhysicalPlanLoweringResult& result,
                                                        PlanGenerationId generation,
                                                        const PayloadEdgePlanRecord& edge,
                                                        const TensorServiceContract& contract,
                                                        ProcessGroupId source_process_group,
                                                        ProcessGroupId target_process_group) {
  // A cross-process consumer imports the sample from a transported envelope rather than sharing the
  // producer's pool, so there is no field storage to hand it. Narrowing across a process boundary
  // is not implemented.
  if (edge.tensor_projection) {
    return false;
  }
  auto matching = std::find_if(result.cross_process_channel_services.begin(),
                               result.cross_process_channel_services.end(),
                               [&](const auto& service) {
                                 return cross_process_service_compatible(
                                     service, edge.source, source_process_group, contract);
                               });
  if (matching == result.cross_process_channel_services.end()) {
    const std::string service_id =
        cross_process_channel_service_id(generation, edge.source, edge.queue_policy, edge.edge);
    result.cross_process_channel_services.push_back(
        make_cross_process_service_record(edge.edge,
                                          edge.source,
                                          edge.target,
                                          service_id,
                                          contract,
                                          1U,
                                          source_process_group,
                                          target_process_group));
    result.cross_process_channel_publishers.push_back(CrossProcessChannelPublisherPlanRecord{
        .edge = edge.edge,
        .source = edge.source,
        .service_id = service_id,
        .max_loaned_samples = edge.producer_max_emits_per_compute,
        .replacement_attribution_required = edge.queue_policy != QueuePolicy::kBlocking,
    });
    matching = std::prev(result.cross_process_channel_services.end());
  } else {
    std::uint32_t subscriber_count{};
    if (!checked_add(matching->max_subscribers, 1U, subscriber_count)) {
      return false;
    }
    matching->max_subscribers = subscriber_count;
    matching->max_subscriber_queue_depth =
        std::max(matching->max_subscriber_queue_depth, edge.queue_depth);
    matching->max_subscriber_borrowed_samples =
        std::max(matching->max_subscriber_borrowed_samples, edge.queue_depth);
    if (std::find(matching->target_process_groups.begin(),
                  matching->target_process_groups.end(),
                  target_process_group) == matching->target_process_groups.end()) {
      matching->target_process_groups.push_back(target_process_group);
    }
    auto publisher = std::find_if(
        result.cross_process_channel_publishers.begin(),
        result.cross_process_channel_publishers.end(),
        [&](const auto& candidate) { return candidate.service_id == matching->service_id; });
    if (publisher == result.cross_process_channel_publishers.end()) {
      return false;
    }
    publisher->max_loaned_samples =
        std::max(publisher->max_loaned_samples, edge.producer_max_emits_per_compute);
  }
  result.cross_process_channel_subscribers.push_back(CrossProcessChannelSubscriberPlanRecord{
      .edge = edge.edge,
      .source = edge.source,
      .target = edge.target,
      .service_id = matching->service_id,
      .requested_queue_depth = edge.queue_depth,
      .max_borrowed_samples = edge.queue_depth,
      .cache_capacity = edge.cache_capacity,
      .readiness_capacity = std::max(edge.queue_depth, edge.cache_capacity),
      .event_wake = true,
  });
  return true;
}

void canonicalize_channel_bindings(PhysicalPlanLoweringResult& result) {
  std::sort(result.process_local_channel_services.begin(),
            result.process_local_channel_services.end(),
            [](const auto& left, const auto& right) {
              if (left.source != right.source) {
                return left.source.value() < right.source.value();
              }
              return static_cast<std::uint32_t>(left.queue_policy) <
                     static_cast<std::uint32_t>(right.queue_policy);
            });
  std::sort(result.process_local_channel_publishers.begin(),
            result.process_local_channel_publishers.end(),
            [](const auto& left, const auto& right) {
              if (left.source != right.source) {
                return left.source.value() < right.source.value();
              }
              return left.service_id < right.service_id;
            });
  std::sort(result.process_local_channel_subscribers.begin(),
            result.process_local_channel_subscribers.end(),
            [](const auto& left, const auto& right) { return left.edge.value < right.edge.value; });
  std::sort(result.cross_process_channel_services.begin(),
            result.cross_process_channel_services.end(),
            [](const auto& left, const auto& right) {
              if (left.source != right.source) {
                return left.source.value() < right.source.value();
              }
              if (left.queue_policy != right.queue_policy) {
                return static_cast<std::uint32_t>(left.queue_policy) <
                       static_cast<std::uint32_t>(right.queue_policy);
              }
              return left.service_id < right.service_id;
            });
  std::sort(result.cross_process_channel_publishers.begin(),
            result.cross_process_channel_publishers.end(),
            [](const auto& left, const auto& right) { return left.service_id < right.service_id; });
  std::sort(result.cross_process_channel_subscribers.begin(),
            result.cross_process_channel_subscribers.end(),
            [](const auto& left, const auto& right) { return left.edge.value < right.edge.value; });
  for (auto& service : result.cross_process_channel_services) {
    std::sort(service.target_process_groups.begin(),
              service.target_process_groups.end(),
              [](ProcessGroupId left, ProcessGroupId right) { return left.value < right.value; });
  }
}

}  // namespace

// Preallocation joins every fanout policy before storage admission; keep the compile-cold pass
// whole so rejection and shared-service precedence remain visible in one transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::unordered_set<std::uint64_t> lower_preallocation_channel_services(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    PhysicalTopologyFacts& topology, const bool force_holomq_process_local_inline,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations) {
  // One source can feed multiple queue-policy families. Compatible fanout shares one HoloMQ
  // service while mixed-policy fanout remains mechanically separate. Tensor release-stream
  // authority belongs to the shared backing, so its bound spans every service for one output.
  std::vector<std::uint32_t> descriptor_tensor_fanout(snapshot.ports.size(), 0U);
  std::unordered_set<std::uint64_t> rejected_tensor_fanout_sources;
  for (const auto& edge : result.payload_edges) {
    if (!is_graph_internal_locality(edge.locality) ||
        edge.payload_identity.payload_kind != PayloadKind::kDescriptorBacked || !edge.source ||
        edge.source.value() > descriptor_tensor_fanout.size()) {
      continue;
    }
    const auto* source = find_port(snapshot, GraphPortId{edge.source.value()});
    const auto* target = find_port(snapshot, GraphPortId{edge.target.value()});
    if (source == nullptr || target == nullptr || source->owner_kind != PortOwnerKind::kOperator ||
        target->owner_kind != PortOwnerKind::kOperator) {
      continue;
    }
    const auto source_group = process_group_for(process_placements, source->owner);
    const auto target_group = process_group_for(process_placements, target->owner);
    if (!source_group || !target_group || *source_group != *target_group ||
        rejected_tensor_fanout_sources.contains(edge.source.value())) {
      continue;
    }
    auto& source_fanout =
        descriptor_tensor_fanout[static_cast<std::size_t>(edge.source.value() - 1U)];
    ++source_fanout;
    if (source_fanout > MemoryBuffer::kMaximumFanoutReleaseStreams) {
      rejected_tensor_fanout_sources.insert(edge.source.value());
      reject_physical_plan(
          result, kTensorFanoutInsufficient, source_for(snapshot, source->owner, edge.source));
    }
  }

  result.process_local_channel_services.reserve(result.payload_edges.size());
  result.process_local_channel_publishers.reserve(result.payload_edges.size());
  result.process_local_channel_subscribers.reserve(result.payload_edges.size());
  std::vector<const PayloadEdgePlanRecord*> channel_edges;
  channel_edges.reserve(result.payload_edges.size());
  for (const auto& edge : result.payload_edges) {
    channel_edges.push_back(&edge);
  }
  std::sort(channel_edges.begin(), channel_edges.end(), [](const auto* left, const auto* right) {
    return left->edge.value < right->edge.value;
  });
  for (const auto* const edge_record : channel_edges) {
    const auto& edge = *edge_record;
    if (!is_graph_internal_locality(edge.locality)) {
      continue;
    }
    if (rejected_tensor_fanout_sources.contains(edge.source.value())) {
      continue;
    }
    const auto* source = find_port(snapshot, GraphPortId{edge.source.value()});
    const auto* target = find_port(snapshot, GraphPortId{edge.target.value()});
    if (source == nullptr || target == nullptr || source->owner_kind != PortOwnerKind::kOperator ||
        target->owner_kind != PortOwnerKind::kOperator) {
      reject_physical_plan(
          result,
          kProcessLocalChannelInvalid,
          source_for(snapshot, source != nullptr ? source->owner : VertexId{}, edge.source));
      continue;
    }
    const auto source_group = process_group_for(process_placements, source->owner);
    const auto target_group = process_group_for(process_placements, target->owner);
    if (!source_group || !target_group) {
      if (is_py_object_payload_identity(edge.payload_identity)) {
        reject_physical_plan(
            result, kProcessLocalChannelInvalid, source_for(snapshot, source->owner, edge.source));
      }
      continue;
    }
    const bool cross_process_edge = *source_group != *target_group;
    if (cross_process_edge && edge.payload_identity.kind == PayloadIdentityKind::kScalar &&
        edge.payload_identity.scalar &&
        edge.payload_identity.scalar->kind == ScalarIdentityKind::kLocalEnum) {
      reject_physical_plan(result,
                           kLocalEnumCrossProcessUnsupported,
                           source_for(snapshot, source->owner, edge.source),
                           port_pointer(edge.source, "/payload_identity/scalar_kind"),
                           "process-local typed Graph edge",
                           "kLocalEnum on a cross-process edge");
      continue;
    }
    const auto delivery_kind =
        physical_process_local_delivery_kind(topology.holomq_activation_pairs,
                                             edge,
                                             *source,
                                             *target,
                                             force_holomq_process_local_inline);
    if (is_py_object_payload_identity(edge.payload_identity) &&
        (cross_process_edge || delivery_kind != ProcessLocalChannelDeliveryKind::kDirectRetained)) {
      reject_physical_plan(
          result, kProcessLocalChannelInvalid, source_for(snapshot, source->owner, edge.source));
      continue;
    }
    // A projected edge delivers one Tensor member rather than the serialized root, so its consumer
    // owns no schema envelope to size against the producer's.
    if (!edge.tensor_projection &&
        source->serialized_payload_capacity_bytes > target->serialized_payload_capacity_bytes) {
      reject_physical_plan(
          result, kProcessLocalChannelInvalid, source_for(snapshot, target->owner, edge.target));
      continue;
    }
    const auto payload_size = channel_payload_size_bytes(
        edge.payload_identity, source->serialized_payload_capacity_bytes, !cross_process_edge);
    if (!payload_size) {
      // Pool-backed Tensor descriptors are materialized after Tensor pool admission.
      continue;
    }
    const bool descriptor_tensor_schema =
        edge.payload_identity.kind == PayloadIdentityKind::kSchema &&
        edge.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
    if (descriptor_tensor_schema) {
      if (const auto mismatch = find_descriptor_tensor_memory_kind_mismatch(*source, *target)) {
        reject_physical_plan(
            result,
            kTensorLayoutMismatch,
            source_for(snapshot, target->owner, edge.target),
            port_pointer(
                edge.target,
                mismatch->field_id
                    ? "/tensor_fields/" + std::to_string(*mismatch->field_id) + "/tensor_layout"
                    : "/tensor_layout"),
            layout_summary(mismatch->target),
            layout_summary(mismatch->source));
        continue;
      }
    }
    if (descriptor_tensor_schema &&
        (cross_process_edge || delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ) &&
        *payload_size > channel_internal::kMaximumInlinePayloadBytes) {
      reject_physical_plan(result,
                           kSchemaDescriptorEnvelopeCapacityExceeded,
                           source_for(snapshot, source->owner, edge.source),
                           port_pointer(edge.source, "/serialized_payload_capacity_bytes"),
                           "descriptor envelope no larger than " +
                               std::to_string(channel_internal::kMaximumInlinePayloadBytes) +
                               " bytes",
                           "required=" + std::to_string(*payload_size));
      continue;
    }
    TensorServiceStorageContract descriptor_storage{.payload_size_bytes = *payload_size};
    InterfaceTensorEmitPolicy descriptor_emit_policy{InterfaceTensorEmitPolicy::kRequireExportable};
    if (descriptor_tensor_schema && !cross_process_edge &&
        source->descriptor_tensor_field_count > 1U) {
      for (const auto* const field_ptr : canonical_tensor_fields(source->setup.tensor_fields)) {
        const auto& field = *field_ptr;
        const auto* field_derivation = find_tensor_byte_span_derivation(
            tensor_byte_span_derivations, PortId{source->id.value}, field.field_id);
        const auto field_byte_span =
            field_derivation != nullptr
                ? std::optional<std::uint64_t>{field_derivation->resolved_max_byte_span}
                : (field.tensor_allocation
                       ? std::optional<std::uint64_t>{field.tensor_allocation->max_byte_span}
                       : std::nullopt);
        if (!field.tensor_layout || !field.tensor_layout->memory_kind || !field_byte_span) {
          continue;
        }
        const MemoryKind memory = *field.tensor_layout->memory_kind;
        std::optional<DeviceId> device{DeviceId{}};
        if (memory == MemoryKind::kCudaDevice || memory == MemoryKind::kCudaManaged) {
          device = resolve_tensor_device(result,
                                         snapshot,
                                         topology,
                                         edge.source,
                                         memory,
                                         source_for(snapshot, source->owner, edge.source));
        }
        if (!device) {
          continue;
        }
        descriptor_storage.tensor_fields.push_back(TensorFieldServiceStorageContract{
            .field_id = field.field_id,
            .tensor_index = field.tensor_index,
            .max_byte_span = *field_byte_span,
            .pool_id = tensor_field_output_pool_id(plan_generation, edge.source, field.field_id),
            .dtype = field.tensor_layout->dtype,
            .rank = field.tensor_layout->rank,
            .memory_kind = memory,
            .device = *device,
            .tensor_emit_policy = field.payload_options
                                      ? field.payload_options->interface_tensor_emit_policy
                                      : InterfaceTensorEmitPolicy::kRequireExportable,
        });
      }
    }
    if (descriptor_tensor_schema && cross_process_edge) {
      const bool multi_tensor_schema = source->descriptor_tensor_field_count > 1U;
      if (multi_tensor_schema) {
        if (source->setup.tensor_fields.size() != source->descriptor_tensor_field_count) {
          reject_physical_plan(
              result,
              kSchemaTensorFieldUnbounded,
              source_for(snapshot, source->owner, edge.source),
              port_pointer(edge.source, "/tensor_fields"),
              "one bounded staging contract for every descriptor Tensor field",
              "configured=" + std::to_string(source->setup.tensor_fields.size()) +
                  ",required=" + std::to_string(source->descriptor_tensor_field_count));
          continue;
        }
        bool fields_valid = true;
        descriptor_storage.tensor_fields.reserve(source->setup.tensor_fields.size());
        for (const auto* const field_ptr : canonical_tensor_fields(source->setup.tensor_fields)) {
          const auto& field = *field_ptr;
          const auto* field_derivation = find_tensor_byte_span_derivation(
              tensor_byte_span_derivations, PortId{source->id.value}, field.field_id);
          const auto staging_byte_span =
              field_derivation != nullptr
                  ? field_derivation->resolved_max_staging_byte_span
                  : (field.payload_options ? field.payload_options->max_tensor_byte_span
                                           : std::nullopt);
          const auto field_pointer =
              port_pointer(edge.source, "/tensor_fields/" + std::to_string(field.field_id));
          if (!field.tensor_layout || !field.tensor_layout->memory_kind ||
              !field.tensor_layout->dtype || !field.tensor_layout->rank || !field.payload_options ||
              !staging_byte_span || *staging_byte_span == 0U ||
              field.payload_options->interface_tensor_emit_policy !=
                  InterfaceTensorEmitPolicy::kCopyIfNeeded) {
            reject_physical_plan(result,
                                 kSchemaTensorFieldUnbounded,
                                 source_for(snapshot, source->owner, edge.source),
                                 field_pointer,
                                 "layout, positive max_tensor_byte_span, and kCopyIfNeeded",
                                 "field_id=" + std::to_string(field.field_id));
            fields_valid = false;
            continue;
          }
          const MemoryKind memory = *field.tensor_layout->memory_kind;
          if (memory != MemoryKind::kHost && memory != MemoryKind::kPinnedHost &&
              memory != MemoryKind::kCudaDevice) {
            reject_physical_plan(result,
                                 kSchemaTensorFieldUnsupportedMemory,
                                 source_for(snapshot, source->owner, edge.source),
                                 field_pointer + "/tensor_layout/memory_kind",
                                 "kHost|kPinnedHost|kCudaDevice",
                                 "field_id=" + std::to_string(field.field_id) +
                                     ",memory=" + std::string(memory_kind_name(memory)));
            fields_valid = false;
            continue;
          }
          std::optional<DeviceId> device{DeviceId{}};
          if (memory == MemoryKind::kPinnedHost || memory == MemoryKind::kCudaDevice) {
            device = resolve_tensor_device(result,
                                           snapshot,
                                           topology,
                                           edge.source,
                                           memory,
                                           source_for(snapshot, source->owner, edge.source),
                                           memory == MemoryKind::kPinnedHost);
          }
          if (!device) {
            fields_valid = false;
            continue;
          }
          descriptor_storage.tensor_fields.push_back(TensorFieldServiceStorageContract{
              .field_id = field.field_id,
              .tensor_index = field.tensor_index,
              .max_byte_span = *staging_byte_span,
              .pool_id =
                  tensor_schema_staging_field_pool_id(plan_generation, edge.source, field.field_id),
              .dtype = field.tensor_layout->dtype,
              .rank = field.tensor_layout->rank,
              .memory_kind = memory,
              .device = *device,
              .tensor_emit_policy = field.payload_options->interface_tensor_emit_policy,
          });
        }
        if (!fields_valid) {
          continue;
        }
        descriptor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
      } else {
        const auto layout = resolved_layout(*source);
        const auto* derivation =
            find_port_tensor_byte_span_derivation(tensor_byte_span_derivations, *source);
        const auto* authored_edge = find_edge(snapshot, edge.edge);
        const auto* payload =
            authored_edge != nullptr ? &authored_edge->effective_tensor_payload : nullptr;
        const auto staging_byte_span =
            derivation != nullptr && derivation->resolved_max_staging_byte_span
                ? derivation->resolved_max_staging_byte_span
                : (payload != nullptr && *payload ? (*payload)->max_tensor_byte_span
                                                  : std::nullopt);
        if (!layout || !layout->memory_kind || !layout->dtype || !layout->rank ||
            payload == nullptr || !*payload || !staging_byte_span ||
            (*payload)->interface_tensor_emit_policy != InterfaceTensorEmitPolicy::kCopyIfNeeded ||
            (*layout->memory_kind != MemoryKind::kHost &&
             *layout->memory_kind != MemoryKind::kPinnedHost &&
             *layout->memory_kind != MemoryKind::kCudaDevice)) {
          reject_physical_plan(
              result,
              kTensorProviderUnsupported,
              source_for(snapshot, source->owner, edge.source),
              port_pointer(edge.source, "/payload_options/max_tensor_byte_span"),
              "bounded host, pinned-host, or CUDA-device descriptor-schema staging with "
              "kCopyIfNeeded",
              "cross-process descriptor schema has no admitted staging contract");
          continue;
        }
        const auto device = resolve_tensor_device(result,
                                                  snapshot,
                                                  topology,
                                                  edge.source,
                                                  *layout->memory_kind,
                                                  source_for(snapshot, source->owner, edge.source),
                                                  *layout->memory_kind == MemoryKind::kPinnedHost);
        if (!device) {
          continue;
        }
        descriptor_storage.max_byte_span = *staging_byte_span;
        descriptor_storage.pool_id = tensor_schema_staging_pool_id(plan_generation, edge.source);
        descriptor_storage.memory_kind = *layout->memory_kind;
        descriptor_storage.device = *device;
        descriptor_emit_policy = (*payload)->interface_tensor_emit_policy;
      }
    }

    if (cross_process_edge) {
      const auto contract = make_tensor_service_contract(edge.payload_identity,
                                                         edge.queue_policy,
                                                         edge.queue_depth,
                                                         edge.queue_depth,
                                                         descriptor_tensor_schema,
                                                         std::move(descriptor_storage),
                                                         descriptor_emit_policy);
      if (!append_cross_process_channel_binding(
              result, plan_generation, edge, contract, *source_group, *target_group)) {
        reject_physical_plan(
            result, kProcessLocalChannelInvalid, source_for(snapshot, target->owner, edge.target));
      }
      continue;
    }

    const auto contract = make_tensor_service_contract(edge.payload_identity,
                                                       edge.queue_policy,
                                                       0U,
                                                       0U,
                                                       descriptor_tensor_schema,
                                                       std::move(descriptor_storage),
                                                       descriptor_emit_policy);
    if (!append_process_local_channel_binding(
            result, plan_generation, edge, contract, delivery_kind)) {
      reject_physical_plan(
          result, kProcessLocalChannelInvalid, source_for(snapshot, target->owner, edge.target));
      continue;
    }
  }
  return rejected_tensor_fanout_sources;
}

// Tensor service lowering is one fail-closed join across edge, placement, allocation, and delivery
// records; splitting the guarded cases would obscure their shared diagnostic precedence.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void lower_tensor_channel_services(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    const PhysicalTopologyFacts& topology,
    const std::unordered_set<std::uint64_t>& rejected_tensor_fanout_sources,
    const bool force_holomq_process_local_inline,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations) {
  // A projected edge keeps the producer's schema identity, so it shares the producer's schema
  // service and the loop below never reaches it. Its consumer still declares a Tensor contract, so
  // check the named field against that contract here instead of letting a dtype, rank, or
  // memory-kind mismatch surface as a runtime binding failure.
  for (const auto& edge : result.payload_edges) {
    if (!edge.tensor_projection || !is_graph_internal_locality(edge.locality)) {
      continue;
    }
    const auto* source = find_port(snapshot, GraphPortId{edge.source.value()});
    const auto* target = find_port(snapshot, GraphPortId{edge.target.value()});
    if (source == nullptr || target == nullptr) {
      continue;
    }
    // Resolve the selected field once during admission. Singular schemas may also borrow
    // existing storage or obtain their geometry from a lifecycle provider.
    const auto& projection = *edge.tensor_projection;
    const auto field = std::ranges::find_if(result.tensor_field_allocations, [&](const auto& pool) {
      return pool.output == edge.source && pool.field_id == projection.field_id &&
             pool.tensor_index == projection.tensor_index &&
             pool.disposition == PlanDisposition::kAccepted;
    });
    const auto allocation = std::ranges::find_if(result.tensor_allocations, [&](const auto& pool) {
      return pool.output == edge.source && pool.disposition == PlanDisposition::kAccepted;
    });
    const auto lifecycle =
        std::ranges::find_if(result.provider_allocation_plans, [&](const auto& provider) {
          return provider.output == edge.source &&
                 provider.geometry_mode == AllocationGeometryMode::kLifecycleCandidate &&
                 provider.tensor_template && provider.disposition == PlanDisposition::kAccepted;
        });
    const bool singular =
        source->descriptor_tensor_field_count == 1U && projection.tensor_index == 0U;
    const bool allocation_declared =
        source->setup.tensor_allocation || source->graph_override.tensor_allocation ||
        find_port_tensor_byte_span_derivation(tensor_byte_span_derivations, *source);
    const bool pass_through = singular && !allocation_declared;
    if (field == result.tensor_field_allocations.end() &&
        (!singular || (allocation == result.tensor_allocations.end() &&
                       lifecycle == result.provider_allocation_plans.end() && !pass_through))) {
      reject_physical_plan(
          result,
          kTensorProviderUnsupported,
          source_for(snapshot, target->owner, edge.target),
          port_pointer(edge.source, "/tensor_fields/" + std::to_string(projection.field_id)),
          "an accepted static, lifecycle, or pass-through Tensor provider for the selected field",
          "field_id=" + std::to_string(projection.field_id));
      continue;
    }
    const auto target_layout = resolved_layout(*target);
    const auto target_contract = resolved_tensor_contract(*target);
    TensorMismatches mismatches;
    const auto satisfies = [&](const auto& provider) {
      return tensor_provider_satisfies_input(provider.memory_kind,
                                             provider.bounds,
                                             provider.axes,
                                             provider.layout,
                                             target_layout,
                                             target_contract,
                                             &mismatches);
    };
    bool satisfied = false;
    if (field != result.tensor_field_allocations.end()) {
      satisfied = satisfies(*field);
    } else if (allocation != result.tensor_allocations.end()) {
      satisfied = satisfies(*allocation);
    } else if (lifecycle != result.provider_allocation_plans.end()) {
      satisfied = satisfies(*lifecycle->tensor_template);
    } else {
      const auto contract = resolved_tensor_contract(*source);
      const auto layout = resolved_layout(*source);
      const TensorPoolBounds bounds{
          .max_rank = contract ? contract->max_rank.value_or(0U) : 0U,
          .max_byte_span = contract ? contract->max_byte_span.value_or(0U) : 0U,
      };
      satisfied = tensor_provider_satisfies_input(
          layout ? layout->memory_kind.value_or(MemoryKind::kUnknown) : MemoryKind::kUnknown,
          bounds,
          contract ? std::span<const TensorAxisBounds>{contract->axes}
                   : std::span<const TensorAxisBounds>{},
          layout,
          target_layout,
          target_contract,
          &mismatches);
    }
    if (!satisfied) {
      reject_physical_plan(result,
                           kTensorLayoutMismatch,
                           source_for(snapshot, target->owner, edge.target),
                           port_pointer(edge.target, "/tensor_layout"),
                           layout_summary(target_layout),
                           "selected Tensor provider");
      result.errors.back().detail =
          physical_edge_detail(snapshot, source, target, std::move(mismatches));
    }
  }

  // Tensor services carry descriptor-backed typed samples. Their payload byte
  // span is an admitted logical bound joined to the owning pool; it is not an
  // inline HoloMQ object size and therefore leaves payload_size_bytes at zero.
  for (const auto& edge : result.payload_edges) {
    if (!is_graph_internal_locality(edge.locality) ||
        edge.payload_identity.kind != PayloadIdentityKind::kTensor ||
        rejected_tensor_fanout_sources.contains(edge.source.value())) {
      continue;
    }
    const auto* source = find_port(snapshot, GraphPortId{edge.source.value()});
    const auto* target = find_port(snapshot, GraphPortId{edge.target.value()});
    if (source == nullptr || target == nullptr) {
      continue;
    }
    const auto source_group = process_group_for(process_placements, source->owner);
    const auto target_group = process_group_for(process_placements, target->owner);
    if (!source_group || !target_group) {
      continue;
    }
    const bool cross_process_edge = *source_group != *target_group;
    const auto allocation = std::find_if(
        result.tensor_allocations.begin(), result.tensor_allocations.end(), [&](const auto& pool) {
          return pool.output == edge.source;
        });
    const auto candidate_provider =
        std::ranges::find_if(result.provider_allocation_plans, [&](const auto& provider) {
          return provider.output == edge.source &&
                 provider.geometry_mode == AllocationGeometryMode::kLifecycleCandidate &&
                 provider.tensor_template.has_value() &&
                 provider.disposition == PlanDisposition::kAccepted;
        });
    const TensorProviderTemplatePlanRecord* candidate_template = nullptr;
    if (candidate_provider != result.provider_allocation_plans.end()) {
      const auto& tensor_template = candidate_provider->tensor_template;
      if (tensor_template.has_value()) {
        candidate_template = &tensor_template.value();
      }
    }
    const auto interface_pool =
        std::find_if(result.tensor_interface_pools.begin(),
                     result.tensor_interface_pools.end(),
                     [&](const auto& pool) { return pool.graph_port == edge.source; });
    const bool direct =
        allocation != result.tensor_allocations.end() || candidate_template != nullptr;
    const bool boundary = interface_pool != result.tensor_interface_pools.end();
    const bool direct_allocation_declared =
        source->setup.tensor_allocation.has_value() ||
        source->graph_override.tensor_allocation.has_value() ||
        find_port_tensor_byte_span_derivation(tensor_byte_span_derivations, *source) != nullptr;
    if (direct_allocation_declared && !direct) {
      // A rejected direct allocation cannot fall back to an unpooled pass-through service.
      continue;
    }
    // A pass-through Tensor service remains typed but owns no allocation pool.
    // Empty pool facts distinguish that ordinary forwarding path from direct
    // or interface pool authority without inventing an inline payload size.
    const std::string pool_id = allocation != result.tensor_allocations.end() ? allocation->pool_id
                                : candidate_template != nullptr ? candidate_provider->pool_id
                                : boundary                      ? interface_pool->pool_id
                                                                : std::string{};
    const std::uint64_t max_byte_span =
        allocation != result.tensor_allocations.end() ? allocation->bounds.max_byte_span
        : candidate_template != nullptr               ? candidate_template->bounds.max_byte_span
        : boundary ? interface_pool->logical.max_byte_span.value_or(0U)
                   : 0U;
    const MemoryKind memory = allocation != result.tensor_allocations.end()
                                  ? allocation->memory_kind
                              : candidate_template != nullptr ? candidate_template->memory_kind
                              : boundary                      ? interface_pool->memory_kind
                                                              : MemoryKind::kUnknown;
    const DeviceId device = allocation != result.tensor_allocations.end() ? allocation->device
                            : candidate_template != nullptr ? candidate_template->device
                            : boundary                      ? interface_pool->device
                                                            : DeviceId{};
    const auto delivery_kind =
        physical_process_local_delivery_kind(topology.holomq_activation_pairs,
                                             edge,
                                             *source,
                                             *target,
                                             force_holomq_process_local_inline);

    std::vector<DiagnosticMismatch> mismatches;
    const bool input_compatible =
        allocation != result.tensor_allocations.end()
            ? tensor_provider_satisfies_input(allocation->memory_kind,
                                              allocation->bounds,
                                              allocation->axes,
                                              allocation->layout,
                                              resolved_layout(*target),
                                              resolved_tensor_contract(*target),
                                              &mismatches)
        : candidate_template != nullptr
            ? tensor_provider_satisfies_input(candidate_template->memory_kind,
                                              candidate_template->bounds,
                                              candidate_template->axes,
                                              candidate_template->layout,
                                              resolved_layout(*target),
                                              resolved_tensor_contract(*target),
                                              &mismatches)
        : boundary ? tensor_interface_pool_satisfies_input(*interface_pool,
                                                           resolved_layout(*target),
                                                           resolved_tensor_contract(*target),
                                                           &mismatches)
                   : true;
    if ((direct || boundary) && !input_compatible) {
      reject_physical_plan(
          result,
          kTensorLayoutMismatch,
          source_for(snapshot, target->owner, edge.target),
          port_pointer(edge.target, "/tensor_layout"),
          layout_summary(resolved_layout(*target)),
          allocation != result.tensor_allocations.end() ? allocation_layout_summary(*allocation)
          : candidate_template != nullptr ? template_layout_summary(*candidate_template)
                                          : interface_layout_summary(*interface_pool));
      result.errors.back().detail =
          physical_edge_detail(snapshot, source, target, std::move(mismatches));
      // Omit the inadmissible service; rejected-plan artifacts retain diagnostics without requiring
      // complete runtime-construction relations.
      continue;
    }

    if (cross_process_edge) {
      const bool direct_provider_supported = memory == MemoryKind::kHost ||
                                             memory == MemoryKind::kCudaDevice ||
                                             (memory == MemoryKind::kPinnedHost && device.valid());
      if (direct && !direct_provider_supported) {
        // baseline has no implicit Tensor conversion provider. A process boundary therefore admits
        // only a HoloMQ-exportable source pool whose frozen layout satisfies the target directly.
        reject_physical_plan(result,
                             kTensorProviderUnsupported,
                             source_for(snapshot, target->owner, edge.target),
                             port_pointer(edge.source, "/tensor_layout/memory_kind"),
                             "kHost|kPinnedHost(with device)|kCudaDevice",
                             std::string(memory_kind_name(memory)));
        continue;
      }
      const std::uint32_t direct_max_rank =
          allocation != result.tensor_allocations.end() ? allocation->bounds.max_rank
          : candidate_template != nullptr               ? candidate_template->bounds.max_rank
                                                        : 0U;
      if (direct && direct_max_rank > kCrossProcessTensorMaxRank) {
        reject_physical_plan(result,
                             kTensorRankUnsupported,
                             source_for(snapshot, target->owner, edge.target),
                             port_pointer(edge.source, "/tensor_allocation/max_rank"),
                             "<=" + std::to_string(kCrossProcessTensorMaxRank),
                             std::to_string(direct_max_rank));
        continue;
      }
      const auto contract =
          make_tensor_service_contract(edge.payload_identity,
                                       edge.queue_policy,
                                       edge.queue_depth,
                                       edge.queue_depth,
                                       true,
                                       TensorServiceStorageContract{.max_byte_span = max_byte_span,
                                                                    .pool_id = pool_id,
                                                                    .memory_kind = memory,
                                                                    .device = device});
      if (!append_cross_process_channel_binding(
              result, plan_generation, edge, contract, *source_group, *target_group)) {
        reject_physical_plan(
            result, kProcessLocalChannelInvalid, source_for(snapshot, target->owner, edge.target));
      }
      continue;
    }

    const auto contract =
        make_tensor_service_contract(edge.payload_identity,
                                     edge.queue_policy,
                                     0U,
                                     0U,
                                     true,
                                     TensorServiceStorageContract{.max_byte_span = max_byte_span,
                                                                  .pool_id = pool_id,
                                                                  .memory_kind = memory,
                                                                  .device = device});
    if (!append_process_local_channel_binding(
            result, plan_generation, edge, contract, delivery_kind)) {
      reject_physical_plan(
          result, kProcessLocalChannelInvalid, source_for(snapshot, target->owner, edge.target));
      continue;
    }
  }
}

namespace {

// Event-wake finalization evaluates the complete immutable service inventory as one cold pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void finalize_process_local_event_wakes_impl(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    std::span<const ExecutionPartitionRecord> execution_partitions,
    const bool force_holomq_process_local_event_wake) {
  const std::size_t service_count = result.process_local_channel_services.size();
  std::vector<ProcessLocalHoloMQEventWakeReason> service_wake_reasons(
      service_count, ProcessLocalHoloMQEventWakeReason::kUnclassified);
  std::vector<std::uint32_t> service_member_counts(service_count, 0U);
  std::vector<std::uint32_t> service_source_fanouts(service_count, 0U);
  std::vector<bool> service_fanout_ambiguities(service_count, false);
  for (std::size_t service_index = 0U; service_index < service_count; ++service_index) {
    const auto& service = result.process_local_channel_services[service_index];
    auto wake_reason = service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ
                           ? ProcessLocalHoloMQEventWakeReason::kSuppressed
                           : ProcessLocalHoloMQEventWakeReason::kNotHoloMQ;
    auto& member_count = service_member_counts[service_index];
    auto& source_fanout = service_source_fanouts[service_index];
    for (const auto& subscriber : result.process_local_channel_subscribers) {
      if (subscriber.service_id != service.service_id) {
        continue;
      }
      if (member_count == std::numeric_limits<std::uint32_t>::max()) {
        wake_reason = combine_process_local_holomq_event_wake_reasons(
            wake_reason, ProcessLocalHoloMQEventWakeReason::kReplayAmbiguity);
        continue;
      }
      ++member_count;
      const auto edge = std::find_if(
          result.payload_edges.begin(), result.payload_edges.end(), [&](const auto& candidate) {
            return candidate.edge == subscriber.edge;
          });
      if (edge == result.payload_edges.end()) {
        wake_reason = combine_process_local_holomq_event_wake_reasons(
            wake_reason, ProcessLocalHoloMQEventWakeReason::kReplayAmbiguity);
        continue;
      }
      if (source_fanout == 0U) {
        source_fanout = edge->fanout;
      } else if (source_fanout != edge->fanout) {
        service_fanout_ambiguities[service_index] = true;
      }
      const auto* source = find_port(snapshot, GraphPortId{edge->source.value()});
      const auto* target = find_port(snapshot, GraphPortId{edge->target.value()});
      const auto member_reason =
          source != nullptr && target != nullptr
              ? process_local_holomq_event_wake_reason(*edge,
                                                       service.delivery_kind,
                                                       service.max_subscribers,
                                                       execution_partitions,
                                                       source->owner,
                                                       target->owner)
              : ProcessLocalHoloMQEventWakeReason::kInvalidOperatorOwnership;
      wake_reason = combine_process_local_holomq_event_wake_reasons(wake_reason, member_reason);
    }
    if (member_count != service.max_subscribers) {
      wake_reason = combine_process_local_holomq_event_wake_reasons(
          wake_reason, ProcessLocalHoloMQEventWakeReason::kIncompleteServiceFanout);
    }
    if (service_fanout_ambiguities[service_index]) {
      wake_reason = combine_process_local_holomq_event_wake_reasons(
          wake_reason, ProcessLocalHoloMQEventWakeReason::kReplayAmbiguity);
    }
    service_wake_reasons[service_index] = wake_reason;
  }

  for (std::size_t family_begin = 0U; family_begin < service_count;) {
    std::size_t family_end = family_begin + 1U;
    const PortId source = result.process_local_channel_services[family_begin].source;
    while (family_end < service_count &&
           result.process_local_channel_services[family_end].source == source) {
      ++family_end;
    }
    auto family_wake_reason = ProcessLocalHoloMQEventWakeReason::kSuppressed;
    std::uint64_t family_member_count = 0U;
    std::uint32_t family_source_fanout = 0U;
    bool family_fanout_ambiguity = false;
    bool family_has_holomq = false;
    for (std::size_t service_index = family_begin; service_index < family_end; ++service_index) {
      const auto& service = result.process_local_channel_services[service_index];
      family_member_count += service_member_counts[service_index];
      const std::uint32_t source_fanout = service_source_fanouts[service_index];
      if (source_fanout != 0U) {
        if (family_source_fanout == 0U) {
          family_source_fanout = source_fanout;
        } else if (family_source_fanout != source_fanout) {
          family_fanout_ambiguity = true;
        }
      }
      if (service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ) {
        family_has_holomq = true;
        family_wake_reason = combine_process_local_holomq_event_wake_reasons(
            family_wake_reason, service_wake_reasons[service_index]);
      }
    }
    if (family_has_holomq && family_fanout_ambiguity) {
      family_wake_reason = combine_process_local_holomq_event_wake_reasons(
          family_wake_reason, ProcessLocalHoloMQEventWakeReason::kReplayAmbiguity);
    }
    if (family_has_holomq &&
        (family_source_fanout == 0U || family_member_count != family_source_fanout)) {
      family_wake_reason = combine_process_local_holomq_event_wake_reasons(
          family_wake_reason, ProcessLocalHoloMQEventWakeReason::kIncompleteServiceFanout);
    }
    if (force_holomq_process_local_event_wake &&
        family_wake_reason == ProcessLocalHoloMQEventWakeReason::kSuppressed) {
      family_wake_reason = ProcessLocalHoloMQEventWakeReason::kForcedControl;
    }
    for (std::size_t service_index = family_begin; service_index < family_end; ++service_index) {
      auto& service = result.process_local_channel_services[service_index];
      const bool holomq = service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ;
      service.event_wake_reason =
          holomq ? family_wake_reason : ProcessLocalHoloMQEventWakeReason::kNotHoloMQ;
      service.event_wake =
          holomq && family_wake_reason != ProcessLocalHoloMQEventWakeReason::kSuppressed;
      for (auto& subscriber : result.process_local_channel_subscribers) {
        if (subscriber.service_id == service.service_id) {
          subscriber.event_wake = service.event_wake;
        }
      }
    }
    family_begin = family_end;
  }
}

}  // namespace

void finalize_channel_services(PhysicalPlanLoweringResult& result,
                               const GraphSnapshotData& snapshot,
                               std::span<const ExecutionPartitionRecord> execution_partitions,
                               const bool force_holomq_process_local_event_wake) {
  canonicalize_channel_bindings(result);
  finalize_process_local_event_wakes_impl(
      result, snapshot, execution_partitions, force_holomq_process_local_event_wake);
}

}  // namespace holoscan::detail
