/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/external_channel_lowering.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
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

constexpr std::string_view kTensorLogicalRequirementsInvalid{
    "PHYSICAL_TENSOR_LOGICAL_REQUIREMENTS_INVALID"};
constexpr std::string_view kTensorInterfaceTopologyInvalid{
    "PHYSICAL_TENSOR_INTERFACE_TOPOLOGY_INVALID"};
constexpr std::string_view kDescriptorTensorExternalTopicLoweringUnavailable{
    "PHYSICAL_DESCRIPTOR_TENSOR_EXTERNAL_TOPIC_LOWERING_UNAVAILABLE"};
constexpr std::uint64_t kProfileMaxInlineLogicalBytes = 65'536U;

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

[[nodiscard]] std::string boundary_layout_summary(const TensorBoundaryContractRecord& provider) {
  return layout_summary(TensorRepresentation{
      .memory_kind = provider.memory_kind,
      .dtype = provider.dtype,
      .rank = provider.rank,
      .strides = provider.strides,
  });
}

[[nodiscard]] bool lifecycle_target_admitted(
    std::span<const LifecycleRestartTargetPlanRecord> targets, VertexId vertex) noexcept {
  return std::any_of(targets.begin(), targets.end(), [&](const auto& target) {
    return target.target.id == vertex.value &&
           target.disposition == LifecycleRestartTargetDisposition::kAdmitted;
  });
}

[[nodiscard]] bool topic_restart_capable(
    const GraphSnapshotData& snapshot, const ExternalTopicRecord& topic,
    std::span<const LifecycleRestartTargetPlanRecord> targets) noexcept {
  for (const auto& edge : snapshot.edges) {
    if (topic.imported && edge.source == topic.graph_port) {
      const auto* target = find_port(snapshot, resolve_interface_port(snapshot, edge.target));
      if (target != nullptr && lifecycle_target_admitted(targets, target->owner)) {
        return true;
      }
    }
    if (!topic.imported && edge.target == topic.graph_port) {
      const auto* source = find_port(snapshot, resolve_interface_port(snapshot, edge.source));
      if (source != nullptr && lifecycle_target_admitted(targets, source->owner)) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] std::vector<TensorTransferPlanRecord> tensor_transfer_matrix(
    const ExternalTopicRecord& topic, const TensorBoundaryContractRecord& tensor,
    const std::optional<std::uint16_t> field_id = std::nullopt,
    const std::optional<std::uint16_t> tensor_index = std::nullopt) {
  std::vector<TensorTransferPlanRecord> result;
  if (!tensor.max_byte_span.has_value() ||
      (tensor.foreign_emit_policy == InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded &&
       !tensor.max_inline_bytes.has_value())) {
    return result;
  }
  const std::uint64_t max_byte_span = tensor.max_byte_span.value();
  const std::uint64_t max_staging_byte_span =
      std::min(max_byte_span, tensor.max_staging_byte_span.value_or(max_byte_span));
  const std::uint32_t max_inline_bytes = tensor.max_inline_bytes.value_or(0U);
  constexpr std::array<TensorSourceClass, 7> kClasses{
      TensorSourceClass::kOrdinaryHostContiguous,
      TensorSourceClass::kOrdinaryHostStrided,
      TensorSourceClass::kPinnedHost,
      TensorSourceClass::kCudaDevice,
      TensorSourceClass::kCudaManaged,
      TensorSourceClass::kImportedDescriptor,
      TensorSourceClass::kAlreadyExportable,
  };
  const auto policy = tensor.foreign_emit_policy;
  const auto append_record = [&](TensorSourceClass source_class,
                                 std::uint64_t minimum,
                                 std::uint64_t maximum,
                                 std::optional<TensorTransferRealization>
                                     realization,
                                 std::string_view evidence_id) {
    TensorTransferPlanRecord record{
        .topic = topic.id,
        .field_id = field_id,
        .tensor_index = tensor_index,
        .source_class = source_class,
        .realization = realization,
        .min_logical_bytes = minimum,
        .max_logical_bytes = maximum,
        .evidence_id = std::string(evidence_id),
        .source_must_be_host_ready = realization == TensorTransferRealization::kInline,
        .source_must_be_exportable = realization == TensorTransferRealization::kZeroCopyExportable,
        .source_retained_until_materialization = realization.has_value(),
        .requires_inline_record = realization == TensorTransferRealization::kInline,
        .requires_exportable_copy_pool =
            realization == TensorTransferRealization::kCopiedToExportablePool,
        .disposition = realization ? PlanDisposition::kAccepted : PlanDisposition::kRejected,
    };
    result.push_back(std::move(record));
  };
  for (const auto source_class : kClasses) {
    const bool host_contiguous = source_class == TensorSourceClass::kOrdinaryHostContiguous;
    const bool host_strided = source_class == TensorSourceClass::kOrdinaryHostStrided;
    const bool exportable = source_class == TensorSourceClass::kAlreadyExportable;
    if (exportable) {
      append_record(source_class,
                    0U,
                    max_byte_span,
                    TensorTransferRealization::kZeroCopyExportable,
                    "holoscan.compile.exportable-descriptor/v1");
    } else if (policy == InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded &&
               host_contiguous) {
      const std::uint64_t inline_maximum =
          std::min(max_byte_span, static_cast<std::uint64_t>(max_inline_bytes));
      append_record(source_class,
                    0U,
                    inline_maximum,
                    TensorTransferRealization::kInline,
                    "holoscan.compile.bounded-inline-host/v1");
      if (inline_maximum < max_staging_byte_span) {
        append_record(source_class,
                      inline_maximum + 1U,
                      max_staging_byte_span,
                      TensorTransferRealization::kCopiedToExportablePool,
                      "holoscan.compile.bounded-host-copy/v1");
      }
      const std::uint64_t accepted_maximum = std::max(inline_maximum, max_staging_byte_span);
      if (accepted_maximum < max_byte_span) {
        append_record(source_class,
                      accepted_maximum + 1U,
                      max_byte_span,
                      std::nullopt,
                      "holoscan.compile.tensor-transfer.staging-bound-exceeded/v1");
      }
    } else if (policy != InterfaceTensorEmitPolicy::kRequireExportable &&
               (host_contiguous || host_strided)) {
      append_record(source_class,
                    0U,
                    max_staging_byte_span,
                    TensorTransferRealization::kCopiedToExportablePool,
                    "holoscan.compile.bounded-host-copy/v1");
      if (max_staging_byte_span < max_byte_span) {
        append_record(source_class,
                      max_staging_byte_span + 1U,
                      max_byte_span,
                      std::nullopt,
                      "holoscan.compile.tensor-transfer.staging-bound-exceeded/v1");
      }
    } else {
      append_record(source_class,
                    0U,
                    max_byte_span,
                    std::nullopt,
                    "holoscan.compile.tensor-transfer.unsupported/v1");
    }
  }
  return result;
}

[[nodiscard]] std::vector<TensorTransferPlanRecord> tensor_transfer_matrix(
    const ExternalTopicRecord& topic) {
  return topic.tensor ? tensor_transfer_matrix(topic, *topic.tensor)
                      : std::vector<TensorTransferPlanRecord>{};
}

[[nodiscard]] bool valid_logical_tensor(const TensorBoundaryContractRecord& logical,
                                        bool require_pool) noexcept {
  if ((logical.memory_kind && !enum_at_most(*logical.memory_kind, MemoryKind::kCudaManaged)) ||
      (logical.dtype && !valid_tensor_dtype(*logical.dtype)) ||
      !enum_at_most(logical.strides, TensorStridePolicy::kBoundedStrided) ||
      !enum_at_most(logical.foreign_emit_policy,
                    InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded)) {
    return false;
  }
  const bool inline_policy =
      logical.foreign_emit_policy == InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded;
  if (inline_policy != logical.max_inline_bytes.has_value() ||
      (logical.max_inline_bytes && *logical.max_inline_bytes == 0U)) {
    return false;
  }
  if (logical.max_staging_byte_span && *logical.max_staging_byte_span == 0U) {
    return false;
  }
  if (!require_pool) {
    return true;
  }
  if (!logical.max_rank || !logical.max_byte_span || *logical.max_rank == 0U ||
      *logical.max_byte_span == 0U || (logical.rank && *logical.rank > *logical.max_rank)) {
    return false;
  }
  if (logical.strides == TensorStridePolicy::kBoundedStrided) {
    return logical.axes.size() == *logical.max_rank &&
           std::all_of(logical.axes.begin(), logical.axes.end(), [](const auto& axis) {
             return axis.max_abs_stride_bytes != 0U;
           });
  }
  return logical.axes.size() <= *logical.max_rank;
}

[[nodiscard]] bool logical_span_fits(const TensorBoundaryContractRecord& logical) noexcept {
  if (!logical.max_byte_span || !logical.dtype) {
    return true;
  }
  std::uint64_t bits_per_element = 0U;
  if (!checked_multiply(logical.dtype->bits, logical.dtype->lanes, bits_per_element)) {
    return false;
  }
  const std::uint64_t bytes_per_element = (bits_per_element + 7U) / 8U;
  if (logical.axes.empty()) {
    return bytes_per_element <= *logical.max_byte_span;
  }
  if (std::any_of(logical.axes.begin(), logical.axes.end(), [](const auto& axis) {
        return axis.max_extent == 0U;
      })) {
    return true;
  }
  std::uint64_t span = bytes_per_element;
  if (logical.strides == TensorStridePolicy::kBoundedStrided) {
    span = 0U;
    for (const auto& axis : logical.axes) {
      std::uint64_t contribution = 0U;
      std::uint64_t next = 0U;
      if (!checked_multiply(axis.max_extent - 1U, axis.max_abs_stride_bytes, contribution) ||
          !checked_add(span, contribution, next)) {
        return false;
      }
      span = next;
    }
    std::uint64_t next = 0U;
    if (!checked_add(span, bytes_per_element, next)) {
      return false;
    }
    span = next;
  } else {
    for (const auto& axis : logical.axes) {
      std::uint64_t next = 0U;
      if (!checked_multiply(span, axis.max_extent, next)) {
        return false;
      }
      span = next;
    }
  }
  return span <= *logical.max_byte_span;
}

[[nodiscard]] bool valid_inline_provider(
    const RuntimeTensorInlineRecordCapability* provider) noexcept {
  return provider != nullptr && provider->capability_id == kTensorInlineRecordCapabilityId &&
         provider->capability_version == kTensorInlineRecordCapabilityVersion &&
         provider->record_format_id == kTensorRecordFormatId &&
         provider->record_format_version == kTensorRecordFormatVersion &&
         provider->inline_discriminator == kTensorInlineDiscriminator &&
         provider->opaque_descriptor_discriminator == kTensorOpaqueDescriptorDiscriminator &&
         provider->inline_arm_schema_id == kTensorInlineArmSchemaId &&
         provider->opaque_descriptor_arm_schema_id == kTensorOpaqueDescriptorArmSchemaId &&
         !provider->provider_id.empty() && !provider->service_family_id.empty() &&
         !provider->evidence_id.empty() && provider->fixed_metadata_overhead_bytes != 0U &&
         provider->max_tensor_rank != 0U && provider->max_inline_logical_bytes != 0U &&
         provider->max_fixed_record_bytes >= provider->fixed_metadata_overhead_bytes;
}

}  // namespace

// External endpoint materialization is one compile-cold, fail-closed transaction whose checks
// deliberately preserve diagnostic precedence across the endpoint and Tensor contract families.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void lower_external_channel_services(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation, std::span<const std::optional<Profile>> effective_profiles,
    const RuntimePackageCapabilityView& package_capabilities,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    std::span<const LifecycleRestartTargetPlanRecord> lifecycle_targets,
    std::span<const ExecutionPartitionRecord> execution_partitions,
    PhysicalTopologyFacts& topology) {
  const auto& endpoint_by_port = topology.endpoint_by_port;
  const auto& device_by_port = topology.device_by_port;
  const auto& cache_by_output = topology.cache_by_output;
  const auto& window_by_output = topology.window_by_output;
  const auto& reuse_by_output = topology.reuse_by_output;
  const auto& drain_by_output = topology.drain_by_output;
  for (const auto& authored_topic : snapshot.external_topics) {
    if (is_py_object_payload_identity(authored_topic.payload_identity)) {
      reject_physical_plan(
          result,
          kProcessLocalChannelInvalid,
          source_for(snapshot, authored_topic.vertex, PortId{authored_topic.graph_port.value}));
      continue;
    }
    const auto& topics_by_port =
        authored_topic.imported ? topology.imported_by_port : topology.exported_by_port;
    const auto admitted_topic = topics_by_port.find(authored_topic.graph_port.value);
    const auto* port = find_port(snapshot, authored_topic.graph_port);
    const auto endpoint = endpoint_by_port.find(authored_topic.graph_port.value);
    if (port == nullptr || admitted_topic == topics_by_port.end() ||
        admitted_topic->second != &authored_topic || endpoint == endpoint_by_port.end()) {
      continue;
    }
    ExternalTopicRecord topic = authored_topic;
    if (topic.payload_identity.kind == PayloadIdentityKind::kTensor) {
      // Endpoint lowering may narrow an exported raw-Tensor boundary to a compiler-resolved
      // producer capacity. Physical transfer and interface-pool records consume that same
      // catalog identity rather than the unresolved immutable Graph snapshot.
      topic.tensor = endpoint->second->tensor;
    } else if (topic.payload_identity.kind == PayloadIdentityKind::kSchema &&
               topic.payload_identity.payload_kind == PayloadKind::kDescriptorBacked) {
      // Descriptor fields follow the same rule independently: endpoint lowering resolves each
      // derived producer span, and every physical service/pool must consume those catalog facts.
      topic.tensor_fields = endpoint->second->tensor_fields;
    }
    const bool descriptor_tensor_schema =
        topic.payload_identity.kind == PayloadIdentityKind::kSchema &&
        topic.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
    const auto matching = std::find_if(
        result.payload_edges.begin(), result.payload_edges.end(), [&](const auto& edge) {
          return topic.imported ? edge.source.value() == topic.graph_port.value
                                : edge.target.value() == topic.graph_port.value;
        });
    const std::uint32_t graph_queue =
        matching == result.payload_edges.end() ? 1U : matching->queue_depth;
    const std::uint32_t graph_burst =
        matching == result.payload_edges.end() ? 1U : matching->producer_max_emits_per_compute;
    const ExternalTopicBoundsResolution bounds =
        resolve_external_topic_bounds(topic, graph_queue, graph_burst);
    const std::uint32_t publisher_loans = bounds.publisher_max_loaned_samples;
    const std::uint32_t subscriber_queue = bounds.subscriber_queue_depth;
    const std::uint32_t subscriber_borrows = bounds.subscriber_max_borrowed_samples;
    if (publisher_loans == 0U || subscriber_queue == 0U || subscriber_borrows == 0U ||
        (topic.imported && topic.bounds.subscriber) ||
        (!topic.imported && topic.bounds.publisher)) {
      reject_physical_plan(result,
                           kServiceInvalid,
                           source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}));
      continue;
    }
    const std::string& service_id = endpoint->second->service;
    const QueuePolicy queue_policy =
        matching == result.payload_edges.end() ? QueuePolicy::kBlocking : matching->queue_policy;
    const bool tensor =
        topic.payload_identity.kind == PayloadIdentityKind::kTensor || descriptor_tensor_schema;
    const InterfaceTensorEmitPolicy emit_policy =
        topic.tensor ? topic.tensor->foreign_emit_policy
                     : InterfaceTensorEmitPolicy::kRequireExportable;
    const auto contract = make_tensor_service_contract(topic.payload_identity,
                                                       queue_policy,
                                                       subscriber_queue,
                                                       subscriber_borrows,
                                                       tensor,
                                                       {},
                                                       emit_policy);
    const auto append_endpoint_records = [&]() {
      result.channel_services.push_back(
          make_channel_service_record(topic.id, service_id, contract, topic.tensor_fields));
      result.channel_publishers.push_back(ChannelPublisherEndpointPlanRecord{
          .topic = topic.id,
          .service_id = service_id,
          .max_loaned_samples = publisher_loans,
          .replacement_attribution_required = queue_policy != QueuePolicy::kBlocking,
      });
      result.channel_subscribers.push_back(ChannelSubscriberEndpointPlanRecord{
          .topic = topic.id,
          .service_id = service_id,
          .requested_queue_depth = subscriber_queue,
          .max_borrowed_samples = subscriber_borrows,
          .event_wake = true,
      });
    };
    if (!descriptor_tensor_schema && topic.imported && topic.tensor &&
        valid_logical_tensor(*topic.tensor, true) && logical_span_fits(*topic.tensor) &&
        matching != result.payload_edges.end()) {
      const auto* target = find_port(snapshot, GraphPortId{matching->target.value()});
      std::vector<DiagnosticMismatch> mismatches;
      if (target != nullptr && target->owner_kind == PortOwnerKind::kOperator &&
          !tensor_boundary_provider_satisfies_input(*topic.tensor,
                                                    resolved_layout(*target),
                                                    resolved_tensor_contract(*target),
                                                    &mismatches)) {
        reject_physical_plan(result,
                             kTensorLayoutMismatch,
                             source_for(snapshot, target->owner, matching->target),
                             port_pointer(matching->target, "/tensor_layout"),
                             layout_summary(resolved_layout(*target)),
                             boundary_layout_summary(*topic.tensor));
        result.errors.back().detail = physical_edge_detail(
            snapshot, find_port(snapshot, topic.graph_port), target, std::move(mismatches));
        continue;
      }
    }
    if (!descriptor_tensor_schema) {
      append_endpoint_records();
    }

    const auto effective_profile = effective_profile_for_vertex(effective_profiles, topic.vertex);
    if (!effective_profile) {
      continue;
    }
    const auto append_import_generation_capacity = [&]() {
      if (!topic.imported) {
        return;
      }
      const bool restart_capable = topic_restart_capable(snapshot, topic, lifecycle_targets);
      const std::uint32_t endpoint_capacity = restart_capable ? 2U : 1U;
      std::optional<std::uint64_t> attach_max_wait_ns;
      if (endpoint->second->attach.max_wait) {
        attach_max_wait_ns = static_cast<std::uint64_t>(endpoint->second->attach.max_wait->count());
      }
      result.imported_generation_capacities.push_back(ImportedGenerationCapacityPlanRecord{
          .topic = topic.id,
          .runtime_scope_id = runtime_scope_id(plan_generation),
          .endpoint_or_service_id = service_id,
          .endpoint_max_generations = endpoint_capacity,
          .service_max_generations = endpoint_capacity,
          .runtime_max_generations = 0U,
          .candidate_reservation_capacity = restart_capable ? 1U : 0U,
          .release_record_capacity = endpoint_capacity,
          .quarantine_record_capacity = endpoint_capacity,
          .effective_profile = effective_profile,
          .peer_requirement = endpoint->second->attach.peer,
          .attach_max_wait_ns = attach_max_wait_ns,
          .restart_capable_minimum_two = restart_capable,
      });
    };
    if (!tensor) {
      // Scalar/POD/schema imports use the same finite incarnation ledger as Tensor imports. The
      // payload-specific claims differ, but every externally driven endpoint needs a bounded
      // replacement candidate and a no-wrap generation fence before reset can cut over safely.
      append_import_generation_capacity();
      continue;
    }
    if (descriptor_tensor_schema) {
      const auto source = source_for(snapshot, topic.vertex, PortId{topic.graph_port.value});
      if (topic.tensor_fields.empty() ||
          topic.tensor_fields.size() != port->descriptor_tensor_field_count) {
        reject_physical_plan(
            result,
            kDescriptorTensorExternalTopicLoweringUnavailable,
            source,
            "/external_topics/" + std::to_string(topic.id.value) + "/tensor_fields",
            "one complete canonical contract per descriptor Tensor field",
            "configured=" + std::to_string(topic.tensor_fields.size()) +
                ",required=" + std::to_string(port->descriptor_tensor_field_count));
        continue;
      }
      if (matching == result.payload_edges.end()) {
        reject_physical_plan(result,
                             kTensorInterfaceTopologyInvalid,
                             source,
                             "/external_topics/" + std::to_string(topic.id.value) + "/edge",
                             "one accepted adjacent descriptor-schema edge",
                             "edge=absent");
        continue;
      }
      const PortId adjacent_port_id = topic.imported ? matching->target : matching->source;
      const auto* adjacent_port = find_port(snapshot, GraphPortId{adjacent_port_id.value()});
      if (adjacent_port == nullptr || adjacent_port->owner_kind != PortOwnerKind::kOperator) {
        reject_physical_plan(
            result,
            kTensorInterfaceTopologyInvalid,
            source,
            "/external_topics/" + std::to_string(topic.id.value) + "/adjacent_port",
            "one Operator-owned adjacent descriptor-schema port",
            adjacent_port == nullptr ? "adjacent_port=absent" : "adjacent_port=non_operator");
        continue;
      }
      const auto process_group = process_group_for(process_placements, adjacent_port->owner);
      if (!process_group) {
        reject_physical_plan(result,
                             kTensorProcessGroupUnresolved,
                             source,
                             port_pointer(adjacent_port_id, "/owner_process_group"),
                             "one accepted process group",
                             "owner=" + std::to_string(adjacent_port->owner.value));
        continue;
      }
      if (accepted_cover_count(execution_partitions, adjacent_port->owner) != 1U) {
        reject_physical_plan(result, kResourceCoverInvalid, source);
        continue;
      }

      const PortId graph_port{topic.graph_port.value};
      const auto port_index = static_cast<std::size_t>(topic.graph_port.value - 1U);
      bool fields_valid = true;
      for (const auto& field : topic.tensor_fields) {
        const auto field_pointer = "/external_topics/" + std::to_string(topic.id.value) +
                                   "/tensor_fields/" + std::to_string(field.field_id);
        const auto& logical = field.tensor;
        const bool complete = logical.memory_kind && logical.dtype && logical.rank &&
                              valid_logical_tensor(logical, true) && logical_span_fits(logical);
        const bool supported_policy =
            logical.foreign_emit_policy !=
                InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded &&
            (!logical.memory_conversion ||
             *logical.memory_conversion == MemoryConversionPolicy::kDefault ||
             *logical.memory_conversion == MemoryConversionPolicy::kDisable) &&
            (logical.foreign_emit_policy == InterfaceTensorEmitPolicy::kRequireExportable ||
             logical.strides == TensorStridePolicy::kContiguousOnly);
        if (!complete || !supported_policy) {
          reject_physical_plan(
              result,
              kTensorLogicalRequirementsInvalid,
              source,
              field_pointer,
              "complete finite host, pinned-host, or CUDA-device field requirements with "
              "require-exportable or contiguous copy-if-needed transfer",
              !complete ? "incomplete_or_malformed_contract"
                        : "unsupported_transfer_or_conversion_policy");
          fields_valid = false;
          continue;
        }
        const MemoryKind memory = *logical.memory_kind;
        if (memory != MemoryKind::kHost && memory != MemoryKind::kPinnedHost &&
            memory != MemoryKind::kCudaDevice) {
          reject_physical_plan(result,
                               kSchemaTensorFieldUnsupportedMemory,
                               source,
                               field_pointer + "/memory_kind",
                               "kHost|kPinnedHost|kCudaDevice",
                               std::string(memory_kind_name(memory)));
          fields_valid = false;
          continue;
        }
        PortId device_binding_port = graph_port;
        const bool endpoint_uses_device_binding =
            memory == MemoryKind::kPinnedHost || memory == MemoryKind::kCudaDevice;
        if (!topic.imported && endpoint_uses_device_binding &&
            !device_by_port.contains(graph_port.value()) &&
            device_by_port.contains(adjacent_port_id.value())) {
          device_binding_port = adjacent_port_id;
        }
        std::optional<DeviceId> device{DeviceId{}};
        if (endpoint_uses_device_binding) {
          device = resolve_tensor_device(result,
                                         snapshot,
                                         topology,
                                         device_binding_port,
                                         memory,
                                         source,
                                         memory == MemoryKind::kPinnedHost);
        }
        if (!device) {
          fields_valid = false;
          continue;
        }
        auto transfer_records =
            tensor_transfer_matrix(topic, logical, field.field_id, field.tensor_index);
        for (auto& transfer : transfer_records) {
          transfer.interface_process_group = *process_group;
          transfer.interface_device = *device;
        }
        result.tensor_transfer_plans.insert(result.tensor_transfer_plans.end(),
                                            std::make_move_iterator(transfer_records.begin()),
                                            std::make_move_iterator(transfer_records.end()));
        if (logical.foreign_emit_policy == InterfaceTensorEmitPolicy::kRequireExportable) {
          continue;
        }

        const std::string pool_id =
            tensor_interface_field_pool_id(plan_generation, topic.id, field.field_id);
        std::vector<TensorSlotClaimReason> retention_reasons{
            TensorSlotClaimReason::kSubscriberBorrowOrImport};
        if (cache_by_output[port_index] != 0U) {
          retention_reasons.push_back(TensorSlotClaimReason::kTemporalCache);
        }
        if (window_by_output[port_index] != 0U) {
          retention_reasons.push_back(TensorSlotClaimReason::kPendingOrSelectedWindow);
        }
        if (reuse_by_output[port_index]) {
          retention_reasons.push_back(TensorSlotClaimReason::kReuseLast);
        }
        if (drain_by_output[port_index]) {
          retention_reasons.push_back(TensorSlotClaimReason::kDrainAllActivation);
        }
        retention_reasons.push_back(TensorSlotClaimReason::kCopyOrRecorderMaterialization);
        std::vector<TensorSlotClaimReason> delivery_reasons{
            TensorSlotClaimReason::kAcceptedPublication,
            TensorSlotClaimReason::kHolomqTransportOrFanout,
        };
        if (memory == MemoryKind::kPinnedHost || memory == MemoryKind::kCudaDevice) {
          delivery_reasons.push_back(TensorSlotClaimReason::kConsumerCudaRelease);
        }
        if (!topic.imported) {
          delivery_reasons.push_back(TensorSlotClaimReason::kRelayOrReexport);
        }
        delivery_reasons.push_back(TensorSlotClaimReason::kDlpackOrFrameworkBorrow);
        std::uint32_t interface_retention = subscriber_borrows;
        if (!checked_add(interface_retention, cache_by_output[port_index], interface_retention)) {
          reject_physical_plan(result, kTensorOverflow, source, field_pointer);
          fields_valid = false;
          continue;
        }
        const std::array contributors{
            TensorClaimContributor{.name = "producer",
                                   .maximum = publisher_loans,
                                   .reasons = {TensorSlotClaimReason::kProducerLoan,
                                               TensorSlotClaimReason::kProducerWriter,
                                               TensorSlotClaimReason::kProducerReadiness}},
            TensorClaimContributor{.name = "delivery",
                                   .maximum = subscriber_queue,
                                   .reasons = std::move(delivery_reasons)},
            TensorClaimContributor{.name = "retention",
                                   .maximum = interface_retention,
                                   .reasons = std::move(retention_reasons)},
            TensorClaimContributor{.name = "retirement",
                                   .maximum = std::max(publisher_loans, subscriber_borrows),
                                   .reasons = {TensorSlotClaimReason::kCancellation,
                                               TensorSlotClaimReason::kQuarantine,
                                               TensorSlotClaimReason::kRetirementService}},
        };
        const auto slot_count = append_tensor_claim_contributors(
            pool_id, contributors, result.pool_capacity_contributors);
        std::uint64_t aggregate_bytes = 0U;
        if (!slot_count || *slot_count == 0U ||
            !checked_multiply(*slot_count, *logical.max_byte_span, aggregate_bytes)) {
          reject_physical_plan(result, kTensorOverflow, source, field_pointer);
          fields_valid = false;
          continue;
        }
        result.tensor_interface_pools.push_back(TensorInterfacePoolPlanRecord{
            .topic = topic.id,
            .graph_port = graph_port,
            .field_id = field.field_id,
            .tensor_index = field.tensor_index,
            .plan_generation = plan_generation,
            .run_generation = 0U,
            .pool_id = pool_id,
            .allocator_root_id = tensor_allocator_root_id(plan_generation, *process_group, *device),
            .allocator_process_group = *process_group,
            .exporter_subpool_id = pool_id + ".exporter-generation",
            .logical = logical,
            .memory_kind = memory,
            .device = *device,
            .slot_count = *slot_count,
            .aggregate_byte_capacity = aggregate_bytes,
            .publisher_loan_capacity = publisher_loans,
            .control_capacity = *slot_count,
            .readiness_capacity = *slot_count,
            .release_capacity = *slot_count,
            .retirement_capacity = *slot_count,
            .quarantine_capacity = *slot_count,
            .emit_policy = logical.foreign_emit_policy,
            .imported = topic.imported,
            .effective_profile = effective_profile,
        });
      }
      if (!fields_valid) {
        continue;
      }
      append_endpoint_records();
      append_import_generation_capacity();
      continue;
    }
    if (!topic.tensor || !topic.tensor->memory_kind || !valid_logical_tensor(*topic.tensor, true) ||
        !logical_span_fits(*topic.tensor)) {
      const std::string provided =
          !topic.tensor
              ? "absent"
              : (!topic.tensor->memory_kind ? "memory_kind=absent"
                                            : (!valid_logical_tensor(*topic.tensor, true)
                                                   ? "malformed_logical_requirements"
                                                   : "logical_span_exceeds_bound_or_overflows"));
      reject_physical_plan(result,
                           kTensorLogicalRequirementsInvalid,
                           source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}),
                           "/external_topics/" + std::to_string(topic.id.value) + "/tensor",
                           "complete finite logical Tensor requirements",
                           provided);
      continue;
    }
    if (emit_policy == InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded) {
      const auto* provider = package_capabilities.tensor_inline_record();
      if (!valid_inline_provider(provider) || !topic.tensor->max_inline_bytes ||
          !topic.tensor->max_rank || *topic.tensor->max_rank > provider->max_tensor_rank) {
        reject_physical_plan(result,
                             kInlineProviderUnsupported,
                             source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}));
        continue;
      }
      const std::uint64_t profile_max = kProfileMaxInlineLogicalBytes;
      const std::uint64_t effective_max = std::min(provider->max_inline_logical_bytes, profile_max);
      const std::uint64_t accepted_max = *topic.tensor->max_inline_bytes;
      std::uint64_t fixed_bucket_bytes = 0U;
      if (accepted_max > effective_max ||
          !checked_add(provider->fixed_metadata_overhead_bytes, accepted_max, fixed_bucket_bytes) ||
          fixed_bucket_bytes > provider->max_fixed_record_bytes) {
        reject_physical_plan(result,
                             kInlineProviderUnsupported,
                             source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}));
        continue;
      }
      result.tensor_inline_records.push_back(TensorInlineRecordPlanRecord{
          .topic = topic.id,
          .service_id = service_id,
          .fixed_record_bucket_id = tensor_inline_record_bucket_id(plan_generation, topic.id),
          .provider_id = provider->provider_id,
          .service_family_id = provider->service_family_id,
          .capability_id = provider->capability_id,
          .capability_version = provider->capability_version,
          .capability_evidence_id = provider->evidence_id,
          .record_format_id = provider->record_format_id,
          .record_format_version = provider->record_format_version,
          .inline_discriminator = provider->inline_discriminator,
          .opaque_descriptor_discriminator = provider->opaque_descriptor_discriminator,
          .fixed_metadata_overhead_bytes = provider->fixed_metadata_overhead_bytes,
          .provider_max_tensor_rank = provider->max_tensor_rank,
          .provider_max_inline_logical_bytes = provider->max_inline_logical_bytes,
          .profile_max_inline_logical_bytes = profile_max,
          .effective_max_inline_logical_bytes = effective_max,
          .accepted_max_inline_logical_bytes = accepted_max,
          .provider_max_fixed_record_bytes = provider->max_fixed_record_bytes,
          .fixed_record_bucket_bytes = fixed_bucket_bytes,
          .inline_arm_schema_id = provider->inline_arm_schema_id,
          .opaque_descriptor_arm_schema_id = provider->opaque_descriptor_arm_schema_id,
          .effective_profile = effective_profile,
          .disposition = PlanDisposition::kAccepted,
      });
    }
    auto transfer_records = tensor_transfer_matrix(topic);
    result.tensor_transfer_plans.insert(result.tensor_transfer_plans.end(),
                                        std::make_move_iterator(transfer_records.begin()),
                                        std::make_move_iterator(transfer_records.end()));
    append_import_generation_capacity();
    const bool materialization_pool = emit_policy != InterfaceTensorEmitPolicy::kRequireExportable;
    if (!materialization_pool) {
      continue;
    }
    const MemoryKind memory = topic.tensor->memory_kind.value_or(MemoryKind::kUnknown);
    if (memory == MemoryKind::kUnknown) {
      reject_physical_plan(
          result,
          kTensorMemoryKindUnresolved,
          source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}),
          "/external_topics/" + std::to_string(topic.id.value) + "/tensor/memory_kind",
          "kHost|kPinnedHost|kCudaDevice|kCudaManaged",
          std::string(memory_kind_name(memory)));
      continue;
    }
    if (matching == result.payload_edges.end()) {
      reject_physical_plan(result,
                           kTensorInterfaceTopologyInvalid,
                           source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}),
                           "/external_topics/" + std::to_string(topic.id.value) + "/edge",
                           "one accepted adjacent Tensor edge",
                           "edge=absent");
      continue;
    }
    const PortId adjacent_port_id = topic.imported ? matching->target : matching->source;
    const auto* adjacent_port = find_port(snapshot, GraphPortId{adjacent_port_id.value()});
    if (adjacent_port == nullptr || adjacent_port->owner_kind != PortOwnerKind::kOperator) {
      reject_physical_plan(
          result,
          kTensorInterfaceTopologyInvalid,
          source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}),
          "/external_topics/" + std::to_string(topic.id.value) + "/adjacent_port",
          "one Operator-owned adjacent Tensor port",
          adjacent_port == nullptr ? "adjacent_port=absent" : "adjacent_port=non_operator");
      continue;
    }
    const std::string pool_id = tensor_interface_pool_id(plan_generation, topic.id);
    const auto process_group = process_group_for(process_placements, adjacent_port->owner);
    if (!process_group) {
      reject_physical_plan(result,
                           kTensorProcessGroupUnresolved,
                           source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}),
                           port_pointer(adjacent_port_id, "/owner_process_group"),
                           "one accepted process group",
                           "owner=" + std::to_string(adjacent_port->owner.value));
      continue;
    }
    const PortId graph_port{topic.graph_port.value};
    const auto source = source_for(snapshot, topic.vertex, graph_port);
    if (accepted_cover_count(execution_partitions, adjacent_port->owner) != 1U) {
      reject_physical_plan(result, kResourceCoverInvalid, source);
      continue;
    }
    PortId device_binding_port = graph_port;
    const bool endpoint_uses_device_binding =
        memory == MemoryKind::kPinnedHost || memory == MemoryKind::kCudaDevice;
    if (!topic.imported && endpoint_uses_device_binding &&
        !device_by_port.contains(graph_port.value()) &&
        device_by_port.contains(adjacent_port_id.value())) {
      // Public placement names the producing Output<Tensor>, not the synthetic catalog port.
      // Reuse that exact cold binding for the endpoint-owned staging pool.
      device_binding_port = adjacent_port_id;
    }
    const auto device = resolve_tensor_device(result,
                                              snapshot,
                                              topology,
                                              device_binding_port,
                                              memory,
                                              source,
                                              memory == MemoryKind::kPinnedHost);
    if (!device) {
      continue;
    }
    const auto port_index = static_cast<std::size_t>(topic.graph_port.value - 1U);
    std::vector<TensorSlotClaimReason> retention_reasons{
        TensorSlotClaimReason::kSubscriberBorrowOrImport};
    if (cache_by_output[port_index] != 0U) {
      retention_reasons.push_back(TensorSlotClaimReason::kTemporalCache);
    }
    if (window_by_output[port_index] != 0U) {
      retention_reasons.push_back(TensorSlotClaimReason::kPendingOrSelectedWindow);
    }
    if (reuse_by_output[port_index]) {
      retention_reasons.push_back(TensorSlotClaimReason::kReuseLast);
    }
    if (drain_by_output[port_index]) {
      retention_reasons.push_back(TensorSlotClaimReason::kDrainAllActivation);
    }
    if (emit_policy != InterfaceTensorEmitPolicy::kRequireExportable) {
      retention_reasons.push_back(TensorSlotClaimReason::kCopyOrRecorderMaterialization);
    }
    std::vector<TensorSlotClaimReason> delivery_reasons{
        TensorSlotClaimReason::kAcceptedPublication,
        TensorSlotClaimReason::kHolomqTransportOrFanout,
    };
    if (memory == MemoryKind::kPinnedHost || memory == MemoryKind::kCudaDevice ||
        memory == MemoryKind::kCudaManaged) {
      delivery_reasons.push_back(TensorSlotClaimReason::kConsumerCudaRelease);
    }
    if (!topic.imported) {
      delivery_reasons.push_back(TensorSlotClaimReason::kRelayOrReexport);
    }
    delivery_reasons.push_back(TensorSlotClaimReason::kDlpackOrFrameworkBorrow);
    // Queue occupancy and borrowed/materialized storage are simultaneous live sets: taking one
    // sample can free a transport queue position while its backing remains retained by the caller,
    // allowing another distinct sample to enter the queue.
    std::uint32_t interface_retention = subscriber_borrows;
    if (!checked_add(interface_retention, cache_by_output[port_index], interface_retention)) {
      reject_physical_plan(result,
                           kTensorOverflow,
                           source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}));
      continue;
    }
    const std::array contributors{
        TensorClaimContributor{.name = "producer",
                               .maximum = publisher_loans,
                               .reasons = {TensorSlotClaimReason::kProducerLoan,
                                           TensorSlotClaimReason::kProducerWriter,
                                           TensorSlotClaimReason::kProducerReadiness}},
        TensorClaimContributor{.name = "delivery",
                               .maximum = subscriber_queue,
                               .reasons = std::move(delivery_reasons)},
        TensorClaimContributor{.name = "retention",
                               .maximum = interface_retention,
                               .reasons = std::move(retention_reasons)},
        TensorClaimContributor{.name = "retirement",
                               .maximum = std::max(publisher_loans, subscriber_borrows),
                               .reasons = {TensorSlotClaimReason::kCancellation,
                                           TensorSlotClaimReason::kQuarantine,
                                           TensorSlotClaimReason::kRetirementService}},
    };
    const auto slot_count =
        append_tensor_claim_contributors(pool_id, contributors, result.pool_capacity_contributors);
    if (!slot_count || *slot_count == 0U) {
      reject_physical_plan(result,
                           kTensorOverflow,
                           source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}));
      continue;
    }
    std::uint64_t aggregate_bytes = 0U;
    if (!checked_multiply(*slot_count, *topic.tensor->max_byte_span, aggregate_bytes)) {
      reject_physical_plan(result,
                           kTensorOverflow,
                           source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}));
      continue;
    }
    result.tensor_interface_pools.push_back(TensorInterfacePoolPlanRecord{
        .topic = topic.id,
        .graph_port = PortId{topic.graph_port.value},
        .plan_generation = plan_generation,
        .run_generation = 0U,
        .pool_id = pool_id,
        .allocator_root_id = tensor_allocator_root_id(plan_generation, *process_group, *device),
        .allocator_process_group = *process_group,
        .exporter_subpool_id = pool_id + ".exporter-generation",
        .logical = *topic.tensor,
        .memory_kind = memory,
        .device = *device,
        .slot_count = *slot_count,
        .aggregate_byte_capacity = aggregate_bytes,
        .publisher_loan_capacity = publisher_loans,
        .control_capacity = *slot_count,
        .readiness_capacity = *slot_count,
        .release_capacity = *slot_count,
        .retirement_capacity = *slot_count,
        .quarantine_capacity = *slot_count,
        .emit_policy = emit_policy,
        .imported = topic.imported,
        .effective_profile = effective_profile,
    });
  }
}

}  // namespace holoscan::detail
