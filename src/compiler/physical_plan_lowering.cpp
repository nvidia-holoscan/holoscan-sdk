/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/physical_plan_lowering.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <holoscan/memory/memory_buffer.hpp>
#include <holoscan/schema/schema_package.hpp>

#include "channel/channel_payload_limits.hpp"
#include "compiler/detail/checked_arithmetic.hpp"
#include "compiler/detail/snapshot_lookup.hpp"
#include "compiler/process_plan_records.hpp"
#include "core/tensor_allocation_record_validation.hpp"

namespace holoscan::detail {
namespace {

constexpr std::string_view kPortInvalid{"PHYSICAL_PORT_CONTRACT_INVALID"};
constexpr std::string_view kEdgeInvalid{"PHYSICAL_EDGE_CONTRACT_INVALID"};
constexpr std::string_view kBindingInvalid{"PHYSICAL_INTERFACE_BINDING_INVALID"};
constexpr std::string_view kQueueInvalid{"PHYSICAL_QUEUE_CAPACITY_INVALID"};
constexpr std::string_view kTensorOverflow{"PHYSICAL_TENSOR_CAPACITY_OVERFLOW"};
constexpr std::string_view kTensorFanoutInsufficient{
    "PHYSICAL_TENSOR_FANOUT_CAPACITY_INSUFFICIENT"};
constexpr std::string_view kTensorWidening{"PHYSICAL_TENSOR_ALLOCATION_WIDENING"};
constexpr std::string_view kTensorConversionUnsupported{"PHYSICAL_TENSOR_CONVERSION_UNSUPPORTED"};
constexpr std::string_view kTensorDeviceBindingInvalid{"PHYSICAL_TENSOR_DEVICE_BINDING_INVALID"};
constexpr std::string_view kTensorDeviceBindingUnexpected{
    "PHYSICAL_TENSOR_DEVICE_BINDING_UNEXPECTED"};
constexpr std::string_view kTensorDeviceBindingUnused{"PHYSICAL_TENSOR_DEVICE_BINDING_UNUSED"};
constexpr std::string_view kTensorAllocationBoundsInvalid{
    "PHYSICAL_TENSOR_ALLOCATION_BOUNDS_INVALID"};
constexpr std::string_view kTensorMemoryKindUnresolved{"PHYSICAL_TENSOR_MEMORY_KIND_UNRESOLVED"};
constexpr std::string_view kTensorProcessGroupUnresolved{
    "PHYSICAL_TENSOR_PROCESS_GROUP_UNRESOLVED"};
constexpr std::string_view kTensorLogicalRequirementsInvalid{
    "PHYSICAL_TENSOR_LOGICAL_REQUIREMENTS_INVALID"};
constexpr std::string_view kTensorInterfaceTopologyInvalid{
    "PHYSICAL_TENSOR_INTERFACE_TOPOLOGY_INVALID"};
constexpr std::string_view kTensorLayoutMismatch{"PHYSICAL_TENSOR_LAYOUT_MISMATCH"};
constexpr std::string_view kTensorProviderUnsupported{"PHYSICAL_TENSOR_PROVIDER_UNSUPPORTED"};
constexpr std::string_view kTensorRankUnsupported{"PHYSICAL_TENSOR_RANK_UNSUPPORTED"};
constexpr std::string_view kSchemaTensorFieldUnbounded{"CHANNEL_SCHEMA_TENSOR_FIELD_UNBOUNDED"};
constexpr std::string_view kSchemaTensorFieldUnsupportedMemory{
    "CHANNEL_SCHEMA_TENSOR_FIELD_UNSUPPORTED_MEMORY"};
constexpr std::string_view kSchemaTensorBufferCountExceeded{
    "CHANNEL_SCHEMA_TENSOR_BUFFER_COUNT_EXCEEDED"};
constexpr std::string_view kSchemaDescriptorEnvelopeCapacityExceeded{
    "CHANNEL_SCHEMA_DESCRIPTOR_ENVELOPE_CAPACITY_EXCEEDED"};
constexpr std::string_view kDeviceUnresolved{"IMPLICIT_DEVICE_ZERO_REJECTED"};
constexpr std::string_view kResourceCoverInvalid{"PARTITION_RESOURCE_REALIZATION_FAILED"};
constexpr std::string_view kServiceInvalid{"PHYSICAL_SERVICE_CONTRACT_INVALID"};
constexpr std::string_view kProcessLocalChannelInvalid{"PROCESS_LOCAL_CHANNEL_PLAN_INVALID"};
constexpr std::string_view kReadinessInvalid{"PHYSICAL_READINESS_CAPACITY_INSUFFICIENT"};
constexpr std::string_view kLoanEvidenceUnavailable{"GENERIC_LOAN_TRAIT_EVIDENCE_UNAVAILABLE"};
constexpr std::string_view kInlineProviderUnsupported{
    "PHYSICAL_TENSOR_INLINE_PROVIDER_UNSUPPORTED"};
constexpr std::uint64_t kEa1ProfileMaxInlineLogicalBytes = 65'536U;
constexpr std::uint32_t kCrossProcessTensorMaxRank = 8U;

template <typename Enum>
[[nodiscard]] constexpr bool enum_at_most(Enum value, Enum maximum) noexcept {
  using Underlying = std::underlying_type_t<Enum>;
  return static_cast<Underlying>(value) <= static_cast<Underlying>(maximum);
}

[[nodiscard]] DiagnosticSource source_for(const GraphSnapshotData& snapshot, VertexId vertex = {},
                                          PortId port = {}) noexcept {
  return DiagnosticSource{.graph_generation = snapshot.generation, .vertex = vertex, .port = port};
}

void reject(PhysicalPlanLoweringResult& output, std::string_view token,
            DiagnosticSource source = {}, std::string json_pointer = {}, std::string required = {},
            std::string provided = {}) {
  output.errors.push_back(PhysicalPlanDiagnostic{
      .token = std::string(token),
      .source = source,
      .json_pointer = std::move(json_pointer),
      .required = std::move(required),
      .provided = std::move(provided),
  });
}

[[nodiscard]] constexpr std::string_view memory_kind_name(MemoryKind memory) noexcept {
  switch (memory) {
    case MemoryKind::kUnknown:
      return "kUnknown";
    case MemoryKind::kHost:
      return "kHost";
    case MemoryKind::kPinnedHost:
      return "kPinnedHost";
    case MemoryKind::kCudaDevice:
      return "kCudaDevice";
    case MemoryKind::kCudaManaged:
      return "kCudaManaged";
  }
  return "invalid";
}

[[nodiscard]] std::string port_pointer(PortId port, std::string_view suffix) {
  return "/ports/" + std::to_string(port.value()) + std::string(suffix);
}

[[nodiscard]] std::vector<const TensorFieldConfigurationRecord*> canonical_tensor_fields(
    const std::vector<TensorFieldConfigurationRecord>& fields) {
  std::vector<const TensorFieldConfigurationRecord*> canonical;
  canonical.reserve(fields.size());
  for (const auto& field : fields) {
    canonical.push_back(&field);
  }
  std::sort(canonical.begin(), canonical.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->field_id < rhs->field_id;
  });
  return canonical;
}

[[nodiscard]] std::string dtype_summary(const std::optional<DLDataType>& dtype) {
  if (!dtype) {
    return "any";
  }
  return std::to_string(dtype->code) + ":" + std::to_string(dtype->bits) + ":" +
         std::to_string(dtype->lanes);
}

[[nodiscard]] std::string layout_summary(const std::optional<TensorPortLayout>& layout) {
  if (!layout) {
    return "memory=any,dtype=any,rank=any";
  }
  const std::string memory =
      layout->memory_kind ? std::string(memory_kind_name(*layout->memory_kind)) : "any";
  const std::string rank = layout->rank ? std::to_string(*layout->rank) : "any";
  return "memory=" + memory + ",dtype=" + dtype_summary(layout->dtype) + ",rank=" + rank;
}

[[nodiscard]] std::string allocation_layout_summary(const TensorAllocationPlanRecord& allocation) {
  std::optional<TensorPortLayout> layout = allocation.layout;
  if (!layout) {
    layout.emplace();
  }
  layout->memory_kind = allocation.memory_kind;
  return layout_summary(layout);
}

[[nodiscard]] bool valid_dtype(DLDataType dtype) noexcept {
  if (dtype.bits == 0U || dtype.lanes == 0U) {
    return false;
  }
  switch (dtype.code) {
    case kDLInt:
    case kDLUInt:
    case kDLFloat:
    case kDLOpaqueHandle:
    case kDLBfloat:
    case kDLComplex:
    case kDLBool:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] bool payload_equal(const PayloadIdentityRecord& lhs,
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
  if (lhs.schema && (lhs.schema->canonical_name != rhs.schema->canonical_name ||
                     lhs.schema->version != rhs.schema->version ||
                     lhs.schema->schema_sha256 != rhs.schema->schema_sha256 ||
                     lhs.schema->id != rhs.schema->id)) {
    return false;
  }
  if (!lhs.pod) {
    return true;
  }
  if (lhs.pod->canonical_name != rhs.pod->canonical_name || lhs.pod->version != rhs.pod->version ||
      lhs.pod->object_size != rhs.pod->object_size ||
      lhs.pod->object_alignment != rhs.pod->object_alignment ||
      lhs.pod->fields.size() != rhs.pod->fields.size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.pod->fields.size(); ++index) {
    const auto& left = lhs.pod->fields[index];
    const auto& right = rhs.pod->fields[index];
    if (left.name != right.name || left.byte_offset != right.byte_offset ||
        left.scalar_kind != right.scalar_kind || left.element_count != right.element_count) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool valid_payload(const PayloadIdentityRecord& payload) noexcept {
  if (!enum_at_most(payload.kind, PayloadIdentityKind::kPod) ||
      !enum_at_most(payload.payload_kind, PayloadKind::kControlEmpty)) {
    return false;
  }
  switch (payload.kind) {
    case PayloadIdentityKind::kScalar:
      return payload.scalar && !payload.schema && !payload.pod &&
             enum_at_most(payload.scalar->kind, ScalarIdentityKind::kLocalEnum);
    case PayloadIdentityKind::kTensor:
      return !payload.scalar && !payload.schema && !payload.pod &&
             payload.payload_kind == PayloadKind::kDescriptorBacked;
    case PayloadIdentityKind::kSchema:
      return !payload.scalar && payload.schema && !payload.pod &&
             !payload.schema->canonical_name.empty() && payload.schema->version != 0U;
    case PayloadIdentityKind::kPod:
      return !payload.scalar && !payload.schema && payload.pod && payload.pod->object_size != 0U &&
             payload.pod->object_alignment != 0U && !payload.pod->fields.empty();
  }
  return false;
}

[[nodiscard]] bool is_tensor_backed_payload(const PayloadIdentityRecord& payload) noexcept {
  return payload.kind == PayloadIdentityKind::kTensor ||
         (payload.kind == PayloadIdentityKind::kSchema &&
          payload.payload_kind == PayloadKind::kDescriptorBacked);
}

[[nodiscard]] std::optional<Profile> profile_for(std::span<const std::optional<Profile>> profiles,
                                                 VertexId owner) noexcept {
  if (!owner || owner.value > profiles.size()) {
    return std::nullopt;
  }
  return profiles[static_cast<std::size_t>(owner.value - 1U)];
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

[[nodiscard]] std::optional<TensorPortLayout> resolved_layout(const PortRecord& port) {
  return port.graph_override.tensor_layout ? port.graph_override.tensor_layout
                                           : port.setup.tensor_layout;
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

[[nodiscard]] bool validate_tensor_layout(const std::optional<TensorPortLayout>& layout) noexcept {
  return !layout ||
         ((!layout->memory_kind || enum_at_most(*layout->memory_kind, MemoryKind::kCudaManaged)) &&
          (!layout->dtype || valid_dtype(*layout->dtype)));
}

[[nodiscard]] bool dtype_equal(DLDataType left, DLDataType right) noexcept {
  return left.code == right.code && left.bits == right.bits && left.lanes == right.lanes;
}

[[nodiscard]] bool direct_tensor_satisfies_input(
    const TensorAllocationPlanRecord& allocation,
    const std::optional<TensorPortLayout>& input_layout) noexcept {
  if (!input_layout) {
    return true;
  }
  if (input_layout->memory_kind && *input_layout->memory_kind != allocation.memory_kind) {
    return false;
  }
  if (input_layout->dtype && (!allocation.layout || !allocation.layout->dtype ||
                              !dtype_equal(*input_layout->dtype, *allocation.layout->dtype))) {
    return false;
  }
  return !input_layout->rank || (allocation.layout && allocation.layout->rank &&
                                 *input_layout->rank == *allocation.layout->rank);
}

[[nodiscard]] std::string stable_id(std::string_view prefix, PlanGenerationId generation,
                                    std::uint64_t local) {
  return std::string(prefix) + "." + std::to_string(generation) + "." + std::to_string(local);
}

[[nodiscard]] bool allocation_tightens(const TensorAllocationBounds& base,
                                       const TensorAllocationBounds& candidate) noexcept {
  return candidate.max_rank != 0U && candidate.max_byte_span != 0U &&
         candidate.max_rank <= base.max_rank && candidate.max_byte_span <= base.max_byte_span;
}

[[nodiscard]] std::optional<std::uint64_t> scalar_size_bytes(
    const PayloadIdentityRecord& identity) noexcept {
  if (identity.kind != PayloadIdentityKind::kScalar || !identity.scalar) {
    return std::nullopt;
  }
  switch (identity.scalar->kind) {
    case ScalarIdentityKind::kBoolean:
      // Process-local typed storage contains a native C++ bool. Its canonical
      // catalog wire representation is separately fixed at one byte by the
      // catalog codec and must not redefine the native object width here.
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
  return scalar_size_bytes(identity);
}

[[nodiscard]] std::optional<GenericLoanTraitEvidence> generic_loan_trait_evidence(
    const PortPlanRecord& port) noexcept {
  if (!port.generic_loan_supported || port.generic_loan_slot_size_bytes == 0U) {
    return std::nullopt;
  }
  switch (port.payload_identity.kind) {
    case PayloadIdentityKind::kScalar:
      return GenericLoanTraitEvidence::kStaticallyBoundedScalar;
    case PayloadIdentityKind::kPod:
      return GenericLoanTraitEvidence::kNothrowDefaultConstructiblePOD;
    case PayloadIdentityKind::kSchema:
      return GenericLoanTraitEvidence::kNothrowDefaultConstructibleSchema;
    case PayloadIdentityKind::kTensor:
      break;
  }
  return std::nullopt;
}

[[nodiscard]] std::string process_local_service_id(PlanGenerationId generation, PortId source,
                                                   QueuePolicy policy) {
  return fmt::format("process-local-channel.{}.source.{}.policy.{}",
                     generation,
                     source.value(),
                     static_cast<std::uint32_t>(policy));
}

[[nodiscard]] std::string cross_process_service_id(PlanGenerationId generation, PortId source,
                                                   QueuePolicy policy, EdgeId anchor) {
  return fmt::format("cross-process-channel.{}.source.{}.policy.{}.anchor.{}",
                     generation,
                     source.value(),
                     static_cast<std::uint32_t>(policy),
                     anchor.value);
}

[[nodiscard]] bool cross_process_service_compatible(
    const CrossProcessChannelServicePlanRecord& service, PortId source,
    ProcessGroupId source_process_group, const TensorServiceContract& contract) noexcept {
  return service.source == source && service.source_process_group == source_process_group &&
         service.queue_policy == contract.queue_policy &&
         payload_equal(service.payload_identity, contract.payload_identity) &&
         service.payload_size_bytes == contract.payload_size_bytes &&
         service.tensor_payload == contract.tensor_payload &&
         service.tensor_max_byte_span == contract.tensor_max_byte_span &&
         service.tensor_pool_id == contract.tensor_pool_id &&
         service.tensor_memory_kind == contract.tensor_memory_kind &&
         service.tensor_device == contract.tensor_device &&
         service.tensor_fields == contract.tensor_fields &&
         service.publisher_history_depth == contract.publisher_history_depth &&
         service.event_wake == contract.event_wake &&
         service.tensor_emit_policy == contract.tensor_emit_policy;
}

[[nodiscard]] bool append_cross_process_channel_binding(PhysicalPlanLoweringResult& result,
                                                        PlanGenerationId generation,
                                                        const PayloadEdgePlanRecord& edge,
                                                        const TensorServiceContract& contract,
                                                        ProcessGroupId source_process_group,
                                                        ProcessGroupId target_process_group) {
  auto matching = std::find_if(result.cross_process_channel_services.begin(),
                               result.cross_process_channel_services.end(),
                               [&](const auto& service) {
                                 return cross_process_service_compatible(
                                     service, edge.source, source_process_group, contract);
                               });
  if (matching == result.cross_process_channel_services.end()) {
    const std::string service_id =
        cross_process_service_id(generation, edge.source, edge.queue_policy, edge.edge);
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

[[nodiscard]] constexpr bool is_graph_internal_locality(
    const ChannelLocalityKind locality) noexcept {
  // The process-placement reconciliation may run before or after physical service lowering.
  // Process-group identity below remains authoritative for local versus cross-process emission.
  return locality == ChannelLocalityKind::kProcessLocal ||
         locality == ChannelLocalityKind::kCrossProcess;
}

struct ClaimContributor {
  std::string_view name;
  std::uint32_t maximum{};
  std::vector<TensorSlotClaimReason> reasons;
};

[[nodiscard]] std::array<ClaimContributor, 5> tensor_claim_contributors(
    MemoryKind memory, std::uint32_t producer, std::uint32_t delivery,
    std::uint32_t transport_retention, std::uint32_t retention, std::uint32_t window,
    std::uint32_t retirement, bool has_cache, bool reuse, bool drain, bool copy, bool reexport) {
  std::vector<TensorSlotClaimReason> delivery_reasons;
  if (delivery != 0U) {
    delivery_reasons = {TensorSlotClaimReason::kAcceptedPublication,
                        TensorSlotClaimReason::kHolomqTransportOrFanout};
    if (transport_retention == 0U) {
      delivery_reasons.push_back(TensorSlotClaimReason::kSubscriberBorrowOrImport);
      if (memory == MemoryKind::kCudaDevice || memory == MemoryKind::kCudaManaged) {
        delivery_reasons.push_back(TensorSlotClaimReason::kConsumerCudaRelease);
      }
      if (reexport) {
        delivery_reasons.push_back(TensorSlotClaimReason::kRelayOrReexport);
      }
      delivery_reasons.push_back(TensorSlotClaimReason::kDlpackOrFrameworkBorrow);
    }
  }
  std::vector<TensorSlotClaimReason> transport_retention_reasons;
  if (transport_retention != 0U) {
    transport_retention_reasons.push_back(TensorSlotClaimReason::kSubscriberBorrowOrImport);
    if (memory == MemoryKind::kCudaDevice || memory == MemoryKind::kCudaManaged) {
      transport_retention_reasons.push_back(TensorSlotClaimReason::kConsumerCudaRelease);
    }
    if (reexport) {
      transport_retention_reasons.push_back(TensorSlotClaimReason::kRelayOrReexport);
    }
    transport_retention_reasons.push_back(TensorSlotClaimReason::kDlpackOrFrameworkBorrow);
  }
  std::vector<TensorSlotClaimReason> retention_reasons;
  if (has_cache) {
    retention_reasons.push_back(TensorSlotClaimReason::kTemporalCache);
  }
  if (window != 0U) {
    retention_reasons.push_back(TensorSlotClaimReason::kPendingOrSelectedWindow);
  }
  if (reuse) {
    retention_reasons.push_back(TensorSlotClaimReason::kReuseLast);
  }
  if (drain) {
    retention_reasons.push_back(TensorSlotClaimReason::kDrainAllActivation);
  }
  if (copy) {
    retention_reasons.push_back(TensorSlotClaimReason::kCopyOrRecorderMaterialization);
  }
  return {
      ClaimContributor{.name = "producer",
                       .maximum = producer,
                       .reasons = {TensorSlotClaimReason::kProducerLoan,
                                   TensorSlotClaimReason::kProducerWriter,
                                   TensorSlotClaimReason::kProducerReadiness}},
      ClaimContributor{
          .name = "delivery", .maximum = delivery, .reasons = std::move(delivery_reasons)},
      ClaimContributor{.name = "transport-retention",
                       .maximum = transport_retention,
                       .reasons = std::move(transport_retention_reasons)},
      ClaimContributor{
          .name = "retention", .maximum = retention, .reasons = std::move(retention_reasons)},
      ClaimContributor{.name = "retirement",
                       .maximum = retirement,
                       .reasons = {TensorSlotClaimReason::kCancellation,
                                   TensorSlotClaimReason::kQuarantine,
                                   TensorSlotClaimReason::kRetirementService}},
  };
}

[[nodiscard]] std::optional<std::uint32_t> append_claim_contributors(
    std::string_view pool_id, std::span<const ClaimContributor> contributors,
    std::vector<PoolCapacityContributorRecord>& output) {
  std::uint32_t total = 0U;
  for (const auto& contributor : contributors) {
    if (contributor.maximum == 0U || contributor.reasons.empty()) {
      continue;
    }
    std::uint32_t next = 0U;
    if (!checked_add(total, contributor.maximum, next)) {
      return std::nullopt;
    }
    total = next;
    output.push_back(PoolCapacityContributorRecord{
        .pool_id = std::string(pool_id),
        .contributor_id = std::string(pool_id) + "." + std::string(contributor.name),
        .maximum_distinct_slots = contributor.maximum,
        .reasons = contributor.reasons,
    });
  }
  return total;
}

[[nodiscard]] const ChannelEndpointRecord* find_catalog_endpoint(
    std::span<const ChannelEndpointRecord> endpoints, const ExternalTopicRecord& topic) noexcept {
  const TopicDirection direction =
      topic.imported ? TopicDirection::kImport : TopicDirection::kExport;
  const auto found = std::find_if(endpoints.begin(), endpoints.end(), [&](const auto& endpoint) {
    return endpoint.boundary_port.value() == topic.graph_port.value && endpoint.uri == topic.uri &&
           endpoint.direction == direction &&
           payload_equal(endpoint.payload_identity, topic.payload_identity) &&
           endpoint.serialized_payload_capacity_bytes == topic.serialized_payload_capacity_bytes;
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

[[nodiscard]] std::optional<ProcessGroupId> process_group_for(
    std::span<const VertexProcessPlacementPlanRecord> placements, VertexId vertex) noexcept {
  const auto found = std::find_if(placements.begin(), placements.end(), [&](const auto& placement) {
    return placement.vertex == vertex;
  });
  if (found == placements.end() || !found->process_group) {
    return std::nullopt;
  }
  return found->process_group;
}

[[nodiscard]] std::string allocator_root_id(PlanGenerationId generation,
                                            ProcessGroupId process_group, DeviceId device) {
  std::string result = "tensor-allocator." + std::to_string(generation) + ".process." +
                       std::to_string(process_group.value) + ".device.";
  result += device.valid() ? std::to_string(device.value) : "host";
  return result;
}

[[nodiscard]] std::string tensor_stream_id(PlanGenerationId generation,
                                           ProcessGroupId process_group, DeviceId device) {
  return "tensor-stream." + std::to_string(generation) + ".process." +
         std::to_string(process_group.value) + ".device." + std::to_string(device.value);
}

[[nodiscard]] bool requires_explicit_device(MemoryKind memory) noexcept {
  return memory == MemoryKind::kCudaDevice || memory == MemoryKind::kCudaManaged;
}

struct TensorStreamCandidate {
  std::size_t resource_index{};
  EdgeId edge{};
  VertexId source{};
  VertexId target{};
};

/// Record only strict process-local Tensor chains on the existing process/device stream.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void assign_linear_tensor_streams(
    const PhysicalPlanLoweringResult& result,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    std::vector<TensorDeviceResourcePlanRecord>& resources) {
  std::vector<TensorStreamCandidate> candidates;
  candidates.reserve(result.payload_edges.size());
  std::vector<std::size_t> incoming(process_placements.size(), 0U);
  std::vector<std::size_t> outgoing(process_placements.size(), 0U);
  std::vector<std::optional<std::size_t>> resource_by_vertex(process_placements.size());
  std::vector<bool> resource_ambiguous(process_placements.size(), false);
  const auto bind_resource = [&](VertexId vertex, std::size_t resource_index) {
    auto& selected = resource_by_vertex[static_cast<std::size_t>(vertex.value - 1U)];
    if (selected && *selected != resource_index) {
      resource_ambiguous[static_cast<std::size_t>(vertex.value - 1U)] = true;
    } else {
      selected = resource_index;
    }
  };
  for (const auto& allocation : result.tensor_allocations) {
    if (!allocation.output || allocation.output.value() > result.ports.size()) {
      continue;
    }
    const auto& port = result.ports[static_cast<std::size_t>(allocation.output.value() - 1U)];
    const auto resource = std::find_if(resources.begin(), resources.end(), [&](const auto& row) {
      return row.process_group == allocation.allocator_process_group &&
             row.device == allocation.device;
    });
    if (port.id == allocation.output && port.owner &&
        port.owner.value <= process_placements.size() && resource != resources.end()) {
      bind_resource(port.owner,
                    static_cast<std::size_t>(std::distance(resources.begin(), resource)));
    }
  }
  for (const auto& allocation : result.tensor_field_allocations) {
    if (!allocation.output || allocation.output.value() > result.ports.size()) {
      continue;
    }
    const auto& port = result.ports[static_cast<std::size_t>(allocation.output.value() - 1U)];
    const auto resource = std::find_if(resources.begin(), resources.end(), [&](const auto& row) {
      return row.process_group == allocation.allocator_process_group &&
             row.device == allocation.device;
    });
    if (port.id == allocation.output && port.owner &&
        port.owner.value <= process_placements.size() && resource != resources.end()) {
      bind_resource(port.owner,
                    static_cast<std::size_t>(std::distance(resources.begin(), resource)));
    }
  }
  for (const auto& pool : result.tensor_interface_pools) {
    const auto resource = std::find_if(resources.begin(), resources.end(), [&](const auto& row) {
      return row.process_group == pool.allocator_process_group && row.device == pool.device;
    });
    if (resource == resources.end()) {
      continue;
    }
    PortId resolved = pool.graph_port;
    for (std::size_t depth = 0U; depth <= result.interface_bindings.size(); ++depth) {
      if (!resolved || resolved.value() > result.ports.size()) {
        break;
      }
      const auto& port = result.ports[static_cast<std::size_t>(resolved.value() - 1U)];
      if (port.id != resolved) {
        break;
      }
      if (port.owner_kind == PortOwnerKind::kOperator && port.owner &&
          port.owner.value <= process_placements.size()) {
        bind_resource(port.owner,
                      static_cast<std::size_t>(std::distance(resources.begin(), resource)));
        break;
      }
      const auto alias = std::find_if(result.interface_bindings.begin(),
                                      result.interface_bindings.end(),
                                      [&](const auto& row) { return row.public_port == resolved; });
      if (alias == result.interface_bindings.end()) {
        break;
      }
      resolved = alias->internal_port;
    }
  }

  for (const auto& edge : result.payload_edges) {
    if (edge.disposition != PlanDisposition::kAccepted ||
        edge.locality != ChannelLocalityKind::kProcessLocal ||
        !is_tensor_backed_payload(edge.payload_identity) || edge.fanout != 1U || !edge.source ||
        !edge.target || edge.source.value() > result.ports.size() ||
        edge.target.value() > result.ports.size()) {
      continue;
    }
    const auto& source = result.ports[static_cast<std::size_t>(edge.source.value() - 1U)];
    const auto& target = result.ports[static_cast<std::size_t>(edge.target.value() - 1U)];
    if (source.id != edge.source || target.id != edge.target ||
        source.owner_kind != PortOwnerKind::kOperator ||
        target.owner_kind != PortOwnerKind::kOperator || !source.owner || !target.owner ||
        source.owner.value > process_placements.size() ||
        target.owner.value > process_placements.size()) {
      continue;
    }
    const auto source_group = process_group_for(process_placements, source.owner);
    const auto target_group = process_group_for(process_placements, target.owner);
    if (!source_group || source_group != target_group) {
      continue;
    }
    ++outgoing[static_cast<std::size_t>(source.owner.value - 1U)];
    ++incoming[static_cast<std::size_t>(target.owner.value - 1U)];
    const auto service = std::find_if(result.process_local_channel_services.begin(),
                                      result.process_local_channel_services.end(),
                                      [&](const auto& row) {
                                        return row.source == edge.source &&
                                               row.queue_policy == edge.queue_policy &&
                                               row.tensor_payload;
                                      });
    const auto allocation = std::find_if(
        result.tensor_allocations.begin(), result.tensor_allocations.end(), [&](const auto& row) {
          return row.output == edge.source && row.allocator_process_group == *source_group;
        });
    const auto field_allocation = std::find_if(
        result.tensor_field_allocations.begin(),
        result.tensor_field_allocations.end(),
        [&](const auto& row) {
          return row.output == edge.source && row.allocator_process_group == *source_group &&
                 requires_explicit_device(row.memory_kind) && row.device.valid();
        });
    if (service == result.process_local_channel_services.end() ||
        service->delivery_kind != ProcessLocalChannelDeliveryKind::kDirectRetained ||
        !service->tensor_payload) {
      continue;
    }
    const bool raw_tensor = edge.payload_identity.kind == PayloadIdentityKind::kTensor;
    const bool descriptor_tensor =
        edge.payload_identity.kind == PayloadIdentityKind::kSchema &&
        edge.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
    const bool descriptor_tensor_fields = descriptor_tensor && !service->tensor_fields.empty() &&
                                          field_allocation != result.tensor_field_allocations.end();
    const auto device = descriptor_tensor_fields ? field_allocation->device
                        : allocation != result.tensor_allocations.end() &&
                                requires_explicit_device(allocation->memory_kind) &&
                                allocation->device.valid()
                            ? allocation->device
                            : DeviceId{};
    const bool storage_contract_matches =
        raw_tensor && allocation != result.tensor_allocations.end()
            ? service->tensor_pool_id == allocation->pool_id &&
                  service->tensor_max_byte_span == allocation->bounds.max_byte_span &&
                  service->tensor_memory_kind == allocation->memory_kind &&
                  service->tensor_device == allocation->device
            : descriptor_tensor &&
                  ((allocation != result.tensor_allocations.end() &&
                    service->tensor_fields.empty()) ||
                   descriptor_tensor_fields) &&
                  service->tensor_pool_id.empty() && service->tensor_max_byte_span == 0U &&
                  service->tensor_memory_kind == MemoryKind::kUnknown &&
                  !service->tensor_device.valid();
    if (!device.valid() || !storage_contract_matches) {
      continue;
    }
    const auto resource = std::find_if(resources.begin(), resources.end(), [&](const auto& row) {
      return row.process_group == *source_group && row.device == device;
    });
    if (resource == resources.end()) {
      continue;
    }
    const std::size_t resource_index =
        static_cast<std::size_t>(std::distance(resources.begin(), resource));
    bind_resource(source.owner, resource_index);
    bind_resource(target.owner, resource_index);
    candidates.push_back(TensorStreamCandidate{
        .resource_index = resource_index,
        .edge = edge.edge,
        .source = source.owner,
        .target = target.owner,
    });
  }

  for (const auto& candidate : candidates) {
    const std::size_t source_index = static_cast<std::size_t>(candidate.source.value - 1U);
    const std::size_t target_index = static_cast<std::size_t>(candidate.target.value - 1U);
    if (outgoing[source_index] == 1U && incoming[target_index] == 1U &&
        !resource_ambiguous[source_index] && !resource_ambiguous[target_index] &&
        resource_by_vertex[source_index] == candidate.resource_index &&
        resource_by_vertex[target_index] == candidate.resource_index) {
      resources[candidate.resource_index].fused_linear_edges.push_back(candidate.edge);
    }
  }
  for (auto& resource : resources) {
    std::sort(resource.fused_linear_edges.begin(),
              resource.fused_linear_edges.end(),
              [](EdgeId left, EdgeId right) { return left.value < right.value; });
  }
}

/// Reject an Operator whose Tensor inputs and outputs require distinct compute streams.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void validate_operator_device_obligations(PhysicalPlanLoweringResult& result) {
  std::vector<std::optional<DeviceId>> device_by_vertex;
  std::vector<PortId> source_by_vertex;
  for (const auto& port : result.ports) {
    if (port.owner) {
      device_by_vertex.resize(
          std::max(device_by_vertex.size(), static_cast<std::size_t>(port.owner.value)));
      source_by_vertex.resize(device_by_vertex.size());
    }
  }
  std::unordered_set<std::uint64_t> rejected_vertices;
  const auto bind_device = [&](VertexId owner, PortId port, DeviceId device) {
    if (!owner || !device.valid() || owner.value > device_by_vertex.size()) {
      return;
    }
    auto& selected = device_by_vertex[static_cast<std::size_t>(owner.value - 1U)];
    auto& selected_source = source_by_vertex[static_cast<std::size_t>(owner.value - 1U)];
    if (!selected) {
      selected = device;
      selected_source = port;
      return;
    }
    if (*selected == device || !rejected_vertices.insert(owner.value).second) {
      return;
    }
    reject(result,
           kTensorDeviceBindingInvalid,
           DiagnosticSource{.vertex = owner, .port = port},
           "/vertices/" + std::to_string(owner.value) + "/tensor_device",
           "one CUDA device across all Operator Tensor obligations",
           "first_port=" + std::to_string(selected_source.value()) +
               ",first_device=" + std::to_string(selected->value) +
               ",conflicting_port=" + std::to_string(port.value()) + ",conflicting_device=" +
               std::to_string(device.value) + ",reason=operator_device_conflict");
  };
  const auto port_owner = [&](PortId id) -> VertexId {
    if (!id || id.value() > result.ports.size()) {
      return {};
    }
    const auto& port = result.ports[static_cast<std::size_t>(id.value() - 1U)];
    return port.id == id && port.owner_kind == PortOwnerKind::kOperator ? port.owner : VertexId{};
  };
  std::unordered_map<std::uint64_t, PortId> interface_aliases;
  interface_aliases.reserve(result.interface_bindings.size());
  for (const auto& binding : result.interface_bindings) {
    interface_aliases.emplace(binding.public_port.value(), binding.internal_port);
  }
  const auto interface_owner = [&](PortId id) -> VertexId {
    for (std::size_t depth = 0U; depth <= result.interface_bindings.size(); ++depth) {
      const VertexId owner = port_owner(id);
      if (owner) {
        return owner;
      }
      const auto alias = interface_aliases.find(id.value());
      if (alias == interface_aliases.end()) {
        return {};
      }
      id = alias->second;
    }
    return {};
  };

  for (const auto& allocation : result.tensor_allocations) {
    bind_device(port_owner(allocation.output), allocation.output, allocation.device);
  }
  for (const auto& allocation : result.tensor_field_allocations) {
    bind_device(port_owner(allocation.output), allocation.output, allocation.device);
  }
  std::unordered_map<std::string_view, const ProcessLocalChannelServicePlanRecord*>
      process_local_service_by_id;
  process_local_service_by_id.reserve(result.process_local_channel_services.size());
  for (const auto& service : result.process_local_channel_services) {
    process_local_service_by_id.emplace(service.service_id, &service);
  }
  for (const auto& subscriber : result.process_local_channel_subscribers) {
    const auto service = process_local_service_by_id.find(subscriber.service_id);
    if (service != process_local_service_by_id.end() &&
        service->second->source == subscriber.source && service->second->tensor_payload) {
      if (service->second->tensor_fields.empty()) {
        bind_device(
            port_owner(subscriber.target), subscriber.target, service->second->tensor_device);
      } else {
        for (const auto& field : service->second->tensor_fields) {
          bind_device(port_owner(subscriber.target), subscriber.target, field.device);
        }
      }
    }
  }
  for (const auto& service : result.cross_process_channel_services) {
    if (service.tensor_payload) {
      const auto bind_target = [&](const PortId target) {
        if (service.tensor_fields.empty()) {
          bind_device(port_owner(target), target, service.tensor_device);
        } else {
          for (const auto& field : service.tensor_fields) {
            bind_device(port_owner(target), target, field.device);
          }
        }
      };
      if (service.tensor_fields.empty()) {
        if (service.payload_identity.kind == PayloadIdentityKind::kSchema &&
            service.payload_identity.payload_kind == PayloadKind::kDescriptorBacked) {
          bind_device(port_owner(service.source), service.source, service.tensor_device);
        }
      } else {
        for (const auto& field : service.tensor_fields) {
          bind_device(port_owner(service.source), service.source, field.device);
        }
      }
      bind_target(service.target);
      for (const auto& subscriber : result.cross_process_channel_subscribers) {
        if (subscriber.service_id == service.service_id && subscriber.source == service.source &&
            subscriber.target != service.target) {
          bind_target(subscriber.target);
        }
      }
    }
  }
  for (const auto& pool : result.tensor_interface_pools) {
    bind_device(interface_owner(pool.graph_port), pool.graph_port, pool.device);
  }
}

[[nodiscard]] std::size_t accepted_cover_count(std::span<const ExecutionPartitionRecord> partitions,
                                               VertexId owner) noexcept {
  std::size_t count = 0U;
  for (const auto& partition : partitions) {
    count += static_cast<std::size_t>(
        std::count(partition.members.begin(), partition.members.end(), owner));
  }
  return count;
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
    const ExternalTopicRecord& topic) {
  std::vector<TensorTransferPlanRecord> result;
  if (!topic.tensor.has_value()) {
    return result;
  }
  const auto& tensor = topic.tensor.value();
  if (!tensor.max_byte_span.has_value() ||
      (tensor.foreign_emit_policy == InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded &&
       !tensor.max_inline_bytes.has_value())) {
    return result;
  }
  const std::uint64_t max_byte_span = tensor.max_byte_span.value();
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
      if (inline_maximum < max_byte_span) {
        append_record(source_class,
                      inline_maximum + 1U,
                      max_byte_span,
                      TensorTransferRealization::kCopiedToExportablePool,
                      "holoscan.compile.bounded-host-copy/v1");
      }
    } else if (policy != InterfaceTensorEmitPolicy::kRequireExportable &&
               (host_contiguous || host_strided)) {
      append_record(source_class,
                    0U,
                    max_byte_span,
                    TensorTransferRealization::kCopiedToExportablePool,
                    "holoscan.compile.bounded-host-copy/v1");
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

[[nodiscard]] bool valid_logical_tensor(const LogicalTensorRequirementsRecord& logical,
                                        bool require_pool) noexcept {
  if ((logical.memory_kind && !enum_at_most(*logical.memory_kind, MemoryKind::kCudaManaged)) ||
      (logical.dtype && !valid_dtype(*logical.dtype)) ||
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

[[nodiscard]] bool logical_span_fits(const LogicalTensorRequirementsRecord& logical) noexcept {
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

// Keep the physical resource joins and capacity derivation in one fail-closed pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PhysicalPlanLoweringResult lower_physical_plan(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const std::optional<Profile>> effective_profiles,
    const RuntimePackageCapabilityView& package_capabilities,
    std::span<const InputSelectionPlanRecord> input_selections,
    std::span<const BracketedWindowPlanRecord> bracketed_windows,
    std::span<const ChannelEndpointRecord> catalog_endpoints,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    std::span<const LifecycleRestartTargetPlanRecord> lifecycle_targets,
    std::span<const ExecutionPartitionRecord> execution_partitions,
    std::span<const TensorDeviceBindingRecord> tensor_device_bindings,
    bool force_holomq_process_local_inline) {
  PhysicalPlanLoweringResult result;
  result.ports.reserve(snapshot.ports.size());
  std::vector<std::uint32_t> fanout(snapshot.ports.size(), 0U);
  std::unordered_map<std::uint64_t, const ExternalTopicRecord*> imported_by_port;
  std::unordered_map<std::uint64_t, const ExternalTopicRecord*> exported_by_port;
  std::unordered_map<std::uint64_t, const ChannelEndpointRecord*> endpoint_by_port;
  std::unordered_map<std::uint64_t, DeviceId> device_by_port;
  std::unordered_set<std::uint64_t> consumed_device_bindings;
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
      reject(result,
             kTensorDeviceBindingInvalid,
             source_for(snapshot, port == nullptr ? VertexId{} : port->owner, binding.port),
             "/tensor_device_bindings/" + std::to_string(index),
             "one valid Tensor port with one nonnegative device binding",
             std::move(provided));
    }
  }
  const auto resolve_device = [&](PortId port,
                                  MemoryKind memory,
                                  DiagnosticSource source,
                                  const bool force_explicit_device =
                                      false) -> std::optional<DeviceId> {
    const auto found = device_by_port.find(port.value());
    if (!requires_explicit_device(memory) && !force_explicit_device) {
      if (found != device_by_port.end()) {
        consumed_device_bindings.insert(port.value());
        reject(result,
               kTensorDeviceBindingUnexpected,
               source,
               port_pointer(port, "/tensor_device_binding"),
               "absent for non-device storage",
               "device=" + std::to_string(found->second.value));
        return std::nullopt;
      }
      return DeviceId{};
    }
    if (found == device_by_port.end()) {
      reject(result, kDeviceUnresolved, source);
      return std::nullopt;
    }
    consumed_device_bindings.insert(port.value());
    return found->second;
  };
  for (const auto& topic : snapshot.external_topics) {
    auto& index = topic.imported ? imported_by_port : exported_by_port;
    const auto* endpoint = find_catalog_endpoint(catalog_endpoints, topic);
    if (!topic.id || !topic.vertex || !topic.graph_port ||
        find_port(snapshot, topic.graph_port) == nullptr ||
        !index.emplace(topic.graph_port.value, &topic).second ||
        !valid_payload(topic.payload_identity) || endpoint == nullptr ||
        !endpoint_by_port.emplace(topic.graph_port.value, endpoint).second) {
      reject(result,
             kServiceInvalid,
             source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}));
    }
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
    const std::uint32_t queue = port.direction == PortDirection::kInput ? resolved_queue(port) : 0U;
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
      reject(result, kPortInvalid, source_for(snapshot, port.owner, plan_id));
    }
    result.ports.push_back(PortPlanRecord{
        .id = plan_id,
        .owner = port.owner,
        .owner_kind = port.owner_kind,
        .direction = port.direction,
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
        .effective_profile = profile_for(effective_profiles, port.owner),
    });
  }

  for (const auto& binding : snapshot.interface_bindings) {
    const auto* public_port = find_port(snapshot, binding.public_port);
    const auto* internal_port = find_port(snapshot, binding.internal_port);
    if (public_port == nullptr || internal_port == nullptr ||
        public_port->direction != binding.direction ||
        internal_port->direction != binding.direction ||
        public_port->owner_kind != PortOwnerKind::kGraphInterface ||
        !payload_equal(public_port->payload_identity, internal_port->payload_identity)) {
      reject(result,
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
        !payload_equal(source->payload_identity, target->payload_identity) ||
        !enum_at_most(edge.add_flow_options.queue_policy, QueuePolicy::kDropOldest) ||
        (edge.graph_override.queue_policy &&
         !enum_at_most(*edge.graph_override.queue_policy, QueuePolicy::kDropOldest))) {
      reject(
          result,
          kEdgeInvalid,
          source_for(
              snapshot, source != nullptr ? source->owner : VertexId{}, PortId{edge.source.value}));
      continue;
    }
    if (!profile_for(effective_profiles, source->owner) ||
        !profile_for(effective_profiles, target->owner)) {
      // Profile admission already rejected the unresolved owner. Keep the authored Graph edge for
      // diagnostics, but do not claim an accepted physical edge that has no realizable service.
      continue;
    }
    const QueuePolicy policy =
        edge.graph_override.queue_policy.value_or(edge.add_flow_options.queue_policy);
    const std::uint32_t queue = edge.graph_override.queue_depth.value_or(
        edge.add_flow_options.queue_depth.value_or(resolved_queue(*target)));
    const std::uint32_t cache =
        cache_for(PortId{edge.target.value}, input_selections, bracketed_windows);
    const std::uint32_t burst = resolved_burst(*source);
    const std::uint32_t derived_minimum = std::max({queue, cache, burst});
    const std::uint32_t maximum = edge.graph_override.max_inflight_samples.value_or(
        edge.add_flow_options.max_inflight_samples.value_or(derived_minimum));
    if (queue == 0U || maximum == 0U || maximum < derived_minimum || burst == 0U) {
      reject(result, kQueueInvalid, source_for(snapshot, target->owner, PortId{edge.target.value}));
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
      reject(result,
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
      reject(result, kQueueInvalid, source_for(snapshot, source->owner, PortId{source_id.value}));
    } else {
      delivery_by_output[static_cast<std::size_t>(source_id.value - 1U)] = next;
    }
    if (!checked_add(
            cache_by_output[static_cast<std::size_t>(source_id.value - 1U)], cache, next)) {
      reject(result, kQueueInvalid, source_for(snapshot, source->owner, PortId{source_id.value}));
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
  const auto process_local_delivery_kind = [&](const PayloadEdgePlanRecord& edge,
                                               const PortRecord& source,
                                               const PortRecord& target) noexcept {
    return force_holomq_process_local_inline || holomq_activation_pairs.contains(std::pair{
                                                    source.owner.value, target.owner.value})
               ? ProcessLocalChannelDeliveryKind::kHoloMQ
               : process_local_channel_delivery_kind(edge);
  };

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
        process_local_delivery_kind(edge, *source, *target) ==
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
      reject(result, kTensorOverflow, source_for(snapshot, source->owner, edge.source));
      continue;
    }
    transport_retention = next;
  }

  // One source can feed multiple queue-policy families. HoloMQ service-static
  // configuration includes the policy, so compatible fanout shares one
  // service while mixed policy fanout remains mechanically separate. Endpoint
  // records retain edge-local queue/cache bounds and never redefine maxima.
  // Tensor release-stream authority belongs to the shared backing, however,
  // so its bound applies across every service sourced by the same output.
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
      reject(result, kTensorFanoutInsufficient, source_for(snapshot, source->owner, edge.source));
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
      reject(result,
             kProcessLocalChannelInvalid,
             source_for(snapshot, source != nullptr ? source->owner : VertexId{}, edge.source));
      continue;
    }
    const auto source_group = process_group_for(process_placements, source->owner);
    const auto target_group = process_group_for(process_placements, target->owner);
    if (!source_group || !target_group) {
      continue;
    }
    const bool cross_process_edge = *source_group != *target_group;
    if (source->serialized_payload_capacity_bytes != target->serialized_payload_capacity_bytes) {
      reject(result, kProcessLocalChannelInvalid, source_for(snapshot, target->owner, edge.target));
      continue;
    }
    const auto payload_size = channel_payload_size_bytes(
        edge.payload_identity, source->serialized_payload_capacity_bytes, !cross_process_edge);
    if (!payload_size) {
      // Pool-backed Tensor descriptors are materialized after Tensor pool
      // admission. A local enum without fixed-width evidence remains
      // ineligible for this fixed inline-payload service family.
      continue;
    }
    const bool descriptor_tensor_schema =
        edge.payload_identity.kind == PayloadIdentityKind::kSchema &&
        edge.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
    const auto delivery_kind = process_local_delivery_kind(edge, *source, *target);
    if (descriptor_tensor_schema &&
        (cross_process_edge || delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ) &&
        *payload_size > channel_internal::kMaximumInlinePayloadBytes) {
      reject(result,
             kSchemaDescriptorEnvelopeCapacityExceeded,
             source_for(snapshot, source->owner, edge.source),
             port_pointer(edge.source, "/serialized_payload_capacity_bytes"),
             "descriptor envelope no larger than " +
                 std::to_string(channel_internal::kMaximumInlinePayloadBytes) + " bytes",
             "required=" + std::to_string(*payload_size));
      continue;
    }
    TensorServiceStorageContract descriptor_storage{.payload_size_bytes = *payload_size};
    InterfaceTensorEmitPolicy descriptor_emit_policy{InterfaceTensorEmitPolicy::kRequireExportable};
    if (descriptor_tensor_schema && !cross_process_edge &&
        source->descriptor_tensor_field_count > 1U) {
      for (const auto* const field_ptr : canonical_tensor_fields(source->setup.tensor_fields)) {
        const auto& field = *field_ptr;
        if (!field.tensor_layout || !field.tensor_layout->memory_kind || !field.tensor_allocation) {
          continue;
        }
        const MemoryKind memory = *field.tensor_layout->memory_kind;
        std::optional<DeviceId> device{DeviceId{}};
        if (memory == MemoryKind::kCudaDevice || memory == MemoryKind::kCudaManaged) {
          device =
              resolve_device(edge.source, memory, source_for(snapshot, source->owner, edge.source));
        }
        if (!device) {
          continue;
        }
        descriptor_storage.tensor_fields.push_back(TensorFieldServiceStorageContract{
            .field_id = field.field_id,
            .tensor_index = field.tensor_index,
            .max_byte_span = field.tensor_allocation->max_byte_span,
            .pool_id = fmt::format("tensor-field-output-pool.{}.{}.field.{}",
                                   plan_generation,
                                   edge.source.value(),
                                   field.field_id),
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
          reject(result,
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
          const auto field_pointer =
              port_pointer(edge.source, "/tensor_fields/" + std::to_string(field.field_id));
          if (!field.tensor_layout || !field.tensor_layout->memory_kind ||
              !field.tensor_layout->dtype || !field.tensor_layout->rank || !field.payload_options ||
              !field.payload_options->max_tensor_byte_span ||
              *field.payload_options->max_tensor_byte_span == 0U ||
              field.payload_options->interface_tensor_emit_policy !=
                  InterfaceTensorEmitPolicy::kCopyIfNeeded) {
            reject(result,
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
            reject(result,
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
            device = resolve_device(edge.source,
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
              .max_byte_span = *field.payload_options->max_tensor_byte_span,
              .pool_id = fmt::format("tensor-schema-staging-pool.{}.{}.field.{}",
                                     plan_generation,
                                     edge.source.value(),
                                     field.field_id),
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
        const auto* authored_edge = find_edge(snapshot, edge.edge);
        const auto* payload =
            authored_edge != nullptr ? &authored_edge->effective_tensor_payload : nullptr;
        if (!layout || !layout->memory_kind || !layout->dtype || !layout->rank ||
            payload == nullptr || !*payload || !(*payload)->max_tensor_byte_span ||
            (*payload)->interface_tensor_emit_policy != InterfaceTensorEmitPolicy::kCopyIfNeeded ||
            (*layout->memory_kind != MemoryKind::kHost &&
             *layout->memory_kind != MemoryKind::kPinnedHost &&
             *layout->memory_kind != MemoryKind::kCudaDevice)) {
          reject(result,
                 kTensorProviderUnsupported,
                 source_for(snapshot, source->owner, edge.source),
                 port_pointer(edge.source, "/payload_options/max_tensor_byte_span"),
                 "bounded host, pinned-host, or CUDA-device descriptor-schema staging with "
                 "kCopyIfNeeded",
                 "cross-process descriptor schema has no admitted staging contract");
          continue;
        }
        const auto device = resolve_device(edge.source,
                                           *layout->memory_kind,
                                           source_for(snapshot, source->owner, edge.source),
                                           *layout->memory_kind == MemoryKind::kPinnedHost);
        if (!device) {
          continue;
        }
        descriptor_storage.max_byte_span = *(*payload)->max_tensor_byte_span;
        descriptor_storage.pool_id =
            stable_id("tensor-schema-staging-pool", plan_generation, edge.source.value());
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
        reject(
            result, kProcessLocalChannelInvalid, source_for(snapshot, target->owner, edge.target));
      }
      continue;
    }

    const auto matching = std::find_if(result.process_local_channel_services.begin(),
                                       result.process_local_channel_services.end(),
                                       [&](const auto& service) {
                                         return service.source == edge.source &&
                                                service.queue_policy == edge.queue_policy;
                                       });
    std::size_t service_index{};
    if (matching == result.process_local_channel_services.end()) {
      const std::string service_id =
          process_local_service_id(plan_generation, edge.source, edge.queue_policy);
      const auto contract = make_tensor_service_contract(edge.payload_identity,
                                                         edge.queue_policy,
                                                         0U,
                                                         0U,
                                                         descriptor_tensor_schema,
                                                         std::move(descriptor_storage),
                                                         descriptor_emit_policy);
      result.process_local_channel_services.push_back(make_process_local_service_record(
          edge.source, service_id, contract, 0U, false, delivery_kind));
      result.process_local_channel_publishers.push_back(ProcessLocalChannelPublisherPlanRecord{
          .source = edge.source,
          .service_id = service_id,
          .max_loaned_samples = edge.producer_max_emits_per_compute,
          .replacement_attribution_required = edge.queue_policy != QueuePolicy::kBlocking,
      });
      service_index = result.process_local_channel_services.size() - 1U;
    } else {
      service_index = static_cast<std::size_t>(
          std::distance(result.process_local_channel_services.begin(), matching));
    }

    auto& service = result.process_local_channel_services[service_index];
    std::uint32_t subscriber_count{};
    if (!checked_add(service.max_subscribers, 1U, subscriber_count)) {
      reject(result, kProcessLocalChannelInvalid, source_for(snapshot, target->owner, edge.target));
      continue;
    }
    service.max_subscribers = subscriber_count;
    service.max_subscriber_queue_depth =
        std::max(service.max_subscriber_queue_depth, edge.queue_depth);
    service.max_subscriber_borrowed_samples =
        std::max(service.max_subscriber_borrowed_samples, edge.queue_depth);
    result.process_local_channel_subscribers.push_back(ProcessLocalChannelSubscriberPlanRecord{
        .edge = edge.edge,
        .source = edge.source,
        .target = edge.target,
        .service_id = service.service_id,
        .requested_queue_depth = edge.queue_depth,
        .max_borrowed_samples = edge.queue_depth,
        .cache_capacity = edge.cache_capacity,
        // Every queued or retained cache candidate can own one unretired
        // Dispatcher witness. Selector-free arrival triggers have no cache,
        // while temporal selectors can retain more samples than the transport
        // queue, so freeze the larger admitted bound for the ingress ring.
        .readiness_capacity = std::max(edge.queue_depth, edge.cache_capacity),
        .event_wake = delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ,
    });
  }
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

  std::unordered_set<std::uint64_t> cross_process_tensor_sources;
  cross_process_tensor_sources.reserve(result.payload_edges.size());
  for (const auto& edge : result.payload_edges) {
    const bool tensor_backed = is_tensor_backed_payload(edge.payload_identity);
    if (!tensor_backed) {
      continue;
    }
    const auto* source = find_port(snapshot, GraphPortId{edge.source.value()});
    if (edge.locality == ChannelLocalityKind::kExternalExport) {
      if (source != nullptr && process_group_for(process_placements, source->owner)) {
        cross_process_tensor_sources.insert(edge.source.value());
      }
      continue;
    }
    if (!is_graph_internal_locality(edge.locality)) {
      continue;
    }
    const auto* target = find_port(snapshot, GraphPortId{edge.target.value()});
    if (source == nullptr || target == nullptr) {
      continue;
    }
    const auto source_group = process_group_for(process_placements, source->owner);
    const auto target_group = process_group_for(process_placements, target->owner);
    if (source_group && target_group && source_group != target_group) {
      cross_process_tensor_sources.insert(edge.source.value());
    }
  }

  for (const auto& port : result.ports) {
    if (port.direction != PortDirection::kOutput) {
      continue;
    }
    const auto index = static_cast<std::size_t>(port.id.value() - 1U);
    const bool direct_tensor = port.payload_identity.kind == PayloadIdentityKind::kTensor;
    const bool descriptor_tensor_schema =
        port.payload_identity.kind == PayloadIdentityKind::kSchema &&
        port.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
    const auto& authored = snapshot.ports[index];
    const bool multi_descriptor_tensor_schema =
        descriptor_tensor_schema && authored.descriptor_tensor_field_count > 1U;
    const auto& setup = authored.setup.tensor_allocation;
    const auto& override = authored.graph_override.tensor_allocation;
    const bool descriptor_tensor_loan =
        descriptor_tensor_schema &&
        (multi_descriptor_tensor_schema || setup.has_value() || override.has_value());
    const bool tensor_publication_authority = direct_tensor || descriptor_tensor_loan;
    const std::uint32_t retirement = std::max(1U, port.max_emits_per_compute);
    if (!tensor_publication_authority) {
      if (!port.effective_profile) {
        continue;
      }
      if (port.generic_loan_supported) {
        const auto evidence = generic_loan_trait_evidence(port);
        if (!evidence) {
          reject(result, kLoanEvidenceUnavailable, source_for(snapshot, port.owner, port.id));
          continue;
        }
        std::uint32_t slots = port.max_emits_per_compute;
        if (!checked_add(slots, delivery_by_output[index], slots) ||
            !checked_add(slots, cache_by_output[index], slots) ||
            !checked_add(slots, port.late_joiner_history_depth, slots) ||
            !checked_add(slots, retirement, slots)) {
          reject(result, kQueueInvalid, source_for(snapshot, port.owner, port.id));
        } else {
          std::uint64_t aggregate_bytes = 0U;
          if (!checked_multiply(slots, port.generic_loan_slot_size_bytes, aggregate_bytes)) {
            reject(result, kQueueInvalid, source_for(snapshot, port.owner, port.id));
            continue;
          }
          result.generic_output_loans.push_back(GenericOutputLoanPlanRecord{
              .output = port.id,
              .plan_generation = plan_generation,
              .run_generation = 0U,
              .pool_id = stable_id("generic-output-pool", plan_generation, port.id.value()),
              .trait_evidence = *evidence,
              .slot_size_bytes = port.generic_loan_slot_size_bytes,
              .aggregate_byte_capacity = aggregate_bytes,
              .slot_count = slots,
              .max_live_loans = port.max_emits_per_compute,
              .max_emits_per_compute = port.max_emits_per_compute,
              .retirement_capacity = retirement,
              .effective_profile = port.effective_profile,
              .disposition = PlanDisposition::kAccepted,
          });
        }
      }
    }
    result.publication_outcomes.push_back(PublicationOutcomePlanRecord{
        .output = port.id,
        .tensor = tensor_publication_authority,
        .pre_submit_authority = tensor_publication_authority
                                    ? PreSubmitAuthorityDisposition::kConsumedToRetirement
                                    : PreSubmitAuthorityDisposition::kRetainedSameActivation,
        .pre_submit_assigns_sequence = false,
        .post_submit_assigns_sequence = true,
        .post_submit_consumes_authority = true,
        .retain_claims_until_retirement = true,
        .runtime_retry_enabled = false,
    });

    if (!tensor_publication_authority) {
      continue;
    }
    if (multi_descriptor_tensor_schema) {
      if (!port.effective_profile) {
        continue;
      }
      const auto source = source_for(snapshot, port.owner, port.id);
      if (authored.setup.tensor_fields.size() != authored.descriptor_tensor_field_count) {
        reject(result,
               kSchemaTensorFieldUnbounded,
               source,
               port_pointer(port.id, "/tensor_fields"),
               "one bounded allocation contract for every descriptor Tensor field",
               "configured=" + std::to_string(authored.setup.tensor_fields.size()) +
                   ",required=" + std::to_string(authored.descriptor_tensor_field_count));
        continue;
      }
      const auto process_group = process_group_for(process_placements, port.owner);
      if (!process_group) {
        reject(result,
               kTensorProcessGroupUnresolved,
               source,
               port_pointer(port.id, "/owner_process_group"),
               "one accepted process group",
               "owner=" + std::to_string(port.owner.value));
        continue;
      }
      if (accepted_cover_count(execution_partitions, port.owner) != 1U) {
        reject(result, kResourceCoverInvalid, source);
        continue;
      }
      std::uint32_t retention = 0U;
      if (!checked_add(cache_by_output[index], port.late_joiner_history_depth, retention)) {
        reject(result, kTensorOverflow, source);
        continue;
      }

      bool fields_valid = true;
      std::uint32_t total_slots = 0U;
      std::vector<TensorFieldAllocationPlanRecord> field_allocations;
      std::vector<PoolCapacityContributorRecord> field_contributors;
      field_allocations.reserve(authored.setup.tensor_fields.size());
      for (const auto* const field_ptr : canonical_tensor_fields(authored.setup.tensor_fields)) {
        const auto& field = *field_ptr;
        const auto field_pointer =
            port_pointer(port.id, "/tensor_fields/" + std::to_string(field.field_id));
        if (field.payload_options &&
            (field.payload_options->max_inline_bytes ||
             field.payload_options->interface_tensor_emit_policy ==
                 InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded)) {
          reject(result,
                 kInlineProviderUnsupported,
                 source,
                 field_pointer + "/payload_options",
                 "descriptor carriage for every multi-Tensor field",
                 "field_id=" + std::to_string(field.field_id) +
                     ",reason=mixed_inline_descriptor_carriage");
          fields_valid = false;
          continue;
        }
        if (!field.tensor_allocation || !field.tensor_layout || !field.tensor_layout->memory_kind ||
            field.tensor_allocation->max_rank == 0U ||
            field.tensor_allocation->max_byte_span == 0U ||
            (field.tensor_layout->rank &&
             *field.tensor_layout->rank > field.tensor_allocation->max_rank)) {
          reject(result,
                 kSchemaTensorFieldUnbounded,
                 source,
                 field_pointer,
                 "layout and positive max_rank/max_byte_span allocation bounds",
                 "field_id=" + std::to_string(field.field_id));
          fields_valid = false;
          continue;
        }
        const MemoryKind memory = *field.tensor_layout->memory_kind;
        if (memory != MemoryKind::kHost && memory != MemoryKind::kPinnedHost &&
            memory != MemoryKind::kCudaDevice && memory != MemoryKind::kCudaManaged) {
          reject(result,
                 kSchemaTensorFieldUnsupportedMemory,
                 source,
                 field_pointer + "/tensor_layout/memory_kind",
                 "kHost|kPinnedHost|kCudaDevice|kCudaManaged",
                 "field_id=" + std::to_string(field.field_id) +
                     ",memory=" + std::string(memory_kind_name(memory)));
          fields_valid = false;
          continue;
        }
        const bool cross_process_pinned = memory == MemoryKind::kPinnedHost &&
                                          cross_process_tensor_sources.contains(port.id.value());
        std::optional<DeviceId> device{DeviceId{}};
        if (memory == MemoryKind::kCudaDevice || memory == MemoryKind::kCudaManaged ||
            cross_process_pinned) {
          device = resolve_device(port.id, memory, source, cross_process_pinned);
        }
        if (!device) {
          fields_valid = false;
          continue;
        }
        const std::string pool_id = fmt::format("tensor-field-output-pool.{}.{}.field.{}",
                                                plan_generation,
                                                port.id.value(),
                                                field.field_id);
        const auto contributors = tensor_claim_contributors(
            memory,
            port.max_emits_per_compute,
            delivery_by_output[index],
            transport_retention_by_output[index],
            retention,
            window_by_output[index],
            retirement,
            cache_by_output[index] != 0U || port.late_joiner_history_depth != 0U,
            reuse_by_output[index],
            drain_by_output[index],
            copy_by_output[index],
            reexport_by_output[index]);
        const auto slot_count =
            append_claim_contributors(pool_id, contributors, field_contributors);
        std::uint32_t next_total = 0U;
        if (!slot_count || *slot_count == 0U ||
            !checked_add(total_slots, slot_count.value_or(0U), next_total)) {
          reject(result, kTensorOverflow, source, field_pointer);
          fields_valid = false;
          continue;
        }
        total_slots = next_total;
        std::uint64_t aggregate_bytes = 0U;
        if (!checked_multiply(
                *slot_count, field.tensor_allocation->max_byte_span, aggregate_bytes)) {
          reject(result, kTensorOverflow, source, field_pointer);
          fields_valid = false;
          continue;
        }
        field_allocations.push_back(TensorFieldAllocationPlanRecord{
            .output = port.id,
            .field_id = field.field_id,
            .tensor_index = field.tensor_index,
            .plan_generation = plan_generation,
            .run_generation = 0U,
            .pool_id = pool_id,
            .allocator_root_id = allocator_root_id(plan_generation, *process_group, *device),
            .allocator_process_group = *process_group,
            .bounds = *field.tensor_allocation,
            .layout = field.tensor_layout,
            .memory_kind = memory,
            .device = *device,
            .slot_count = *slot_count,
            .aggregate_byte_capacity = aggregate_bytes,
            .max_outstanding_loans = port.max_emits_per_compute,
            .max_emits_per_compute = port.max_emits_per_compute,
            .control_capacity = *slot_count,
            .readiness_capacity = *slot_count,
            .release_capacity = *slot_count,
            .retirement_capacity = *slot_count,
            .quarantine_capacity = *slot_count,
            .effective_profile = port.effective_profile,
            .disposition = PlanDisposition::kAccepted,
        });
      }
      if (!fields_valid) {
        continue;
      }
      if (total_slots > kMaximumDescriptorTensorFieldPoolSlots) {
        reject(result,
               kSchemaTensorBufferCountExceeded,
               source,
               port_pointer(port.id, "/tensor_fields"),
               "total_buffer_count<=" + std::to_string(kMaximumDescriptorTensorFieldPoolSlots),
               "total_buffer_count=" + std::to_string(total_slots));
        continue;
      }
      result.pool_capacity_contributors.insert(result.pool_capacity_contributors.end(),
                                               std::make_move_iterator(field_contributors.begin()),
                                               std::make_move_iterator(field_contributors.end()));
      result.tensor_field_allocations.insert(result.tensor_field_allocations.end(),
                                             std::make_move_iterator(field_allocations.begin()),
                                             std::make_move_iterator(field_allocations.end()));
      continue;
    }
    if (!setup && !override) {
      continue;
    }
    if (!port.effective_profile) {
      // Profile admission already records why this owner has no effective profile. Preserve the
      // partial physical plan without duplicating that root-cause diagnostic or granting a pool.
      continue;
    }
    if (!setup || (override && !allocation_tightens(*setup, *override))) {
      reject(result, kTensorWidening, source_for(snapshot, port.owner, port.id));
      continue;
    }
    const TensorAllocationBounds bounds = override.value_or(*setup);
    if (bounds.max_rank == 0U || bounds.max_byte_span == 0U) {
      reject(result,
             kTensorAllocationBoundsInvalid,
             source_for(snapshot, port.owner, port.id),
             port_pointer(port.id, "/tensor_allocation"),
             "max_rank>0,max_byte_span>0",
             "max_rank=" + std::to_string(bounds.max_rank) +
                 ",max_byte_span=" + std::to_string(bounds.max_byte_span));
      continue;
    }
    const MemoryKind memory = port.tensor_layout && port.tensor_layout->memory_kind
                                  ? *port.tensor_layout->memory_kind
                                  : MemoryKind::kUnknown;
    if (memory == MemoryKind::kUnknown) {
      reject(result,
             kTensorMemoryKindUnresolved,
             source_for(snapshot, port.owner, port.id),
             port_pointer(port.id, "/tensor_layout/memory_kind"),
             "kHost|kPinnedHost|kCudaDevice|kCudaManaged",
             std::string(memory_kind_name(memory)));
      continue;
    }
    const std::string pool_id = stable_id("tensor-output-pool", plan_generation, port.id.value());
    const auto process_group = process_group_for(process_placements, port.owner);
    if (!process_group) {
      reject(result,
             kTensorProcessGroupUnresolved,
             source_for(snapshot, port.owner, port.id),
             port_pointer(port.id, "/owner_process_group"),
             "one accepted process group",
             "owner=" + std::to_string(port.owner.value));
      continue;
    }
    const auto source = source_for(snapshot, port.owner, port.id);
    if (accepted_cover_count(execution_partitions, port.owner) != 1U) {
      reject(result, kResourceCoverInvalid, source);
      continue;
    }
    // HoloMQ's pinned-host generation protocol uses CUDA stream memory operations even though
    // public DLPack metadata has no physical device ordinal. Freeze the private synchronization
    // device only when this output actually crosses a process boundary.
    const bool cross_process_pinned =
        memory == MemoryKind::kPinnedHost && cross_process_tensor_sources.contains(port.id.value());
    const auto device = resolve_device(port.id, memory, source, cross_process_pinned);
    if (!device) {
      continue;
    }
    std::uint32_t retention = 0U;
    if (!checked_add(cache_by_output[index], port.late_joiner_history_depth, retention)) {
      reject(result, kTensorOverflow, source_for(snapshot, port.owner, port.id));
      continue;
    }
    const auto contributors = tensor_claim_contributors(
        memory,
        port.max_emits_per_compute,
        delivery_by_output[index],
        transport_retention_by_output[index],
        retention,
        window_by_output[index],
        retirement,
        cache_by_output[index] != 0U || port.late_joiner_history_depth != 0U,
        reuse_by_output[index],
        drain_by_output[index],
        copy_by_output[index],
        reexport_by_output[index]);
    const auto slot_count =
        append_claim_contributors(pool_id, contributors, result.pool_capacity_contributors);
    if (!slot_count || *slot_count == 0U) {
      reject(result, kTensorOverflow, source_for(snapshot, port.owner, port.id));
      continue;
    }
    std::uint64_t aggregate_bytes = 0U;
    if (!checked_multiply(*slot_count, bounds.max_byte_span, aggregate_bytes)) {
      reject(result, kTensorOverflow, source_for(snapshot, port.owner, port.id));
      continue;
    }
    result.tensor_allocations.push_back(TensorAllocationPlanRecord{
        .output = port.id,
        .plan_generation = plan_generation,
        .run_generation = 0U,
        .pool_id = pool_id,
        .allocator_root_id = allocator_root_id(plan_generation, *process_group, *device),
        .allocator_process_group = *process_group,
        .bounds = bounds,
        .layout = port.tensor_layout,
        .memory_kind = memory,
        .device = *device,
        .slot_count = *slot_count,
        .aggregate_byte_capacity = aggregate_bytes,
        .max_outstanding_loans = port.max_emits_per_compute,
        .max_emits_per_compute = port.max_emits_per_compute,
        .control_capacity = *slot_count,
        .readiness_capacity = *slot_count,
        .release_capacity = *slot_count,
        .retirement_capacity = *slot_count,
        .quarantine_capacity = *slot_count,
        .effective_profile = port.effective_profile,
        .disposition = PlanDisposition::kAccepted,
    });
  }

  for (const auto& topic : snapshot.external_topics) {
    const auto* port = find_port(snapshot, topic.graph_port);
    const auto endpoint = endpoint_by_port.find(topic.graph_port.value);
    if (port == nullptr || endpoint == endpoint_by_port.end()) {
      continue;
    }
    const auto matching = std::find_if(
        result.payload_edges.begin(), result.payload_edges.end(), [&](const auto& edge) {
          return topic.imported ? edge.source.value() == topic.graph_port.value
                                : edge.target.value() == topic.graph_port.value;
        });
    const std::uint32_t graph_queue =
        matching == result.payload_edges.end() ? 1U : matching->queue_depth;
    const std::uint32_t graph_burst =
        matching == result.payload_edges.end() ? 1U : matching->producer_max_emits_per_compute;
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
    if (publisher_loans == 0U || subscriber_queue == 0U || subscriber_borrows == 0U ||
        (topic.imported && topic.bounds.subscriber) ||
        (!topic.imported && topic.bounds.publisher)) {
      reject(result,
             kServiceInvalid,
             source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}));
      continue;
    }
    const std::string& service_id = endpoint->second->service;
    const QueuePolicy queue_policy =
        matching == result.payload_edges.end() ? QueuePolicy::kBlocking : matching->queue_policy;
    const bool tensor = topic.payload_identity.kind == PayloadIdentityKind::kTensor;
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
    result.channel_services.push_back(make_channel_service_record(topic.id, service_id, contract));
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

    const auto effective_profile = profile_for(effective_profiles, topic.vertex);
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
          .runtime_scope_id = stable_id("runtime-scope", plan_generation, 1U),
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
    if (!topic.tensor || !topic.tensor->memory_kind || !valid_logical_tensor(*topic.tensor, true) ||
        !logical_span_fits(*topic.tensor)) {
      const std::string provided =
          !topic.tensor
              ? "absent"
              : (!topic.tensor->memory_kind ? "memory_kind=absent"
                                            : (!valid_logical_tensor(*topic.tensor, true)
                                                   ? "malformed_logical_requirements"
                                                   : "logical_span_exceeds_bound_or_overflows"));
      reject(result,
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
        reject(result,
               kInlineProviderUnsupported,
               source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}));
        continue;
      }
      const std::uint64_t profile_max = kEa1ProfileMaxInlineLogicalBytes;
      const std::uint64_t effective_max = std::min(provider->max_inline_logical_bytes, profile_max);
      const std::uint64_t accepted_max = *topic.tensor->max_inline_bytes;
      std::uint64_t fixed_bucket_bytes = 0U;
      if (accepted_max > effective_max ||
          !checked_add(provider->fixed_metadata_overhead_bytes, accepted_max, fixed_bucket_bytes) ||
          fixed_bucket_bytes > provider->max_fixed_record_bytes) {
        reject(result,
               kInlineProviderUnsupported,
               source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}));
        continue;
      }
      result.tensor_inline_records.push_back(TensorInlineRecordPlanRecord{
          .topic = topic.id,
          .service_id = service_id,
          .fixed_record_bucket_id =
              stable_id("tensor-inline-record-bucket", plan_generation, topic.id.value),
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
      reject(result,
             kTensorMemoryKindUnresolved,
             source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}),
             "/external_topics/" + std::to_string(topic.id.value) + "/tensor/memory_kind",
             "kHost|kPinnedHost|kCudaDevice|kCudaManaged",
             std::string(memory_kind_name(memory)));
      continue;
    }
    if (matching == result.payload_edges.end()) {
      reject(result,
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
      reject(result,
             kTensorInterfaceTopologyInvalid,
             source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}),
             "/external_topics/" + std::to_string(topic.id.value) + "/adjacent_port",
             "one Operator-owned adjacent Tensor port",
             adjacent_port == nullptr ? "adjacent_port=absent" : "adjacent_port=non_operator");
      continue;
    }
    const std::string pool_id = stable_id("tensor-interface-pool", plan_generation, topic.id.value);
    const auto process_group = process_group_for(process_placements, adjacent_port->owner);
    if (!process_group) {
      reject(result,
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
      reject(result, kResourceCoverInvalid, source);
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
    const auto device =
        resolve_device(device_binding_port, memory, source, memory == MemoryKind::kPinnedHost);
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
      reject(result,
             kTensorOverflow,
             source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}));
      continue;
    }
    const std::array contributors{
        ClaimContributor{.name = "producer",
                         .maximum = publisher_loans,
                         .reasons = {TensorSlotClaimReason::kProducerLoan,
                                     TensorSlotClaimReason::kProducerWriter,
                                     TensorSlotClaimReason::kProducerReadiness}},
        ClaimContributor{.name = "delivery",
                         .maximum = subscriber_queue,
                         .reasons = std::move(delivery_reasons)},
        ClaimContributor{.name = "retention",
                         .maximum = interface_retention,
                         .reasons = std::move(retention_reasons)},
        ClaimContributor{.name = "retirement",
                         .maximum = std::max(publisher_loans, subscriber_borrows),
                         .reasons = {TensorSlotClaimReason::kCancellation,
                                     TensorSlotClaimReason::kQuarantine,
                                     TensorSlotClaimReason::kRetirementService}},
    };
    const auto slot_count =
        append_claim_contributors(pool_id, contributors, result.pool_capacity_contributors);
    if (!slot_count || *slot_count == 0U) {
      reject(result,
             kTensorOverflow,
             source_for(snapshot, topic.vertex, PortId{topic.graph_port.value}));
      continue;
    }
    std::uint64_t aggregate_bytes = 0U;
    if (!checked_multiply(*slot_count, *topic.tensor->max_byte_span, aggregate_bytes)) {
      reject(result,
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
        .allocator_root_id = allocator_root_id(plan_generation, *process_group, *device),
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
    const auto interface_pool =
        std::find_if(result.tensor_interface_pools.begin(),
                     result.tensor_interface_pools.end(),
                     [&](const auto& pool) { return pool.graph_port == edge.source; });
    const bool direct = allocation != result.tensor_allocations.end();
    const bool boundary = interface_pool != result.tensor_interface_pools.end();
    const bool direct_allocation_declared = source->setup.tensor_allocation.has_value() ||
                                            source->graph_override.tensor_allocation.has_value();
    if (direct_allocation_declared && !direct) {
      // A rejected direct allocation cannot fall back to an unpooled pass-through service.
      continue;
    }
    // A pass-through Tensor service remains typed but owns no allocation pool.
    // Empty pool facts distinguish that ordinary forwarding path from direct
    // or interface pool authority without inventing an inline payload size.
    const std::string pool_id =
        direct ? allocation->pool_id : (boundary ? interface_pool->pool_id : std::string{});
    const std::uint64_t max_byte_span =
        direct ? allocation->bounds.max_byte_span
               : (boundary ? *interface_pool->logical.max_byte_span : 0U);
    const MemoryKind memory = direct
                                  ? allocation->memory_kind
                                  : (boundary ? interface_pool->memory_kind : MemoryKind::kUnknown);
    const DeviceId device =
        direct ? allocation->device : (boundary ? interface_pool->device : DeviceId{});
    const auto delivery_kind = process_local_delivery_kind(edge, *source, *target);

    if (direct && !direct_tensor_satisfies_input(*allocation, resolved_layout(*target))) {
      reject(result,
             kTensorLayoutMismatch,
             source_for(snapshot, target->owner, edge.target),
             port_pointer(edge.target, "/tensor_layout"),
             layout_summary(resolved_layout(*target)),
             allocation_layout_summary(*allocation));
      continue;
    }

    if (cross_process_edge) {
      const bool direct_provider_supported = memory == MemoryKind::kHost ||
                                             memory == MemoryKind::kCudaDevice ||
                                             (memory == MemoryKind::kPinnedHost && device.valid());
      if (direct && !direct_provider_supported) {
        // EA1 has no implicit Tensor conversion provider. A process boundary therefore admits
        // only a HoloMQ-exportable source pool whose frozen layout satisfies the target directly.
        reject(result,
               kTensorProviderUnsupported,
               source_for(snapshot, target->owner, edge.target),
               port_pointer(edge.source, "/tensor_layout/memory_kind"),
               "kHost|kPinnedHost(with device)|kCudaDevice",
               std::string(memory_kind_name(memory)));
        continue;
      }
      if (direct && allocation->bounds.max_rank > kCrossProcessTensorMaxRank) {
        reject(result,
               kTensorRankUnsupported,
               source_for(snapshot, target->owner, edge.target),
               port_pointer(edge.source, "/tensor_allocation/max_rank"),
               "<=" + std::to_string(kCrossProcessTensorMaxRank),
               std::to_string(allocation->bounds.max_rank));
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
        reject(
            result, kProcessLocalChannelInvalid, source_for(snapshot, target->owner, edge.target));
      }
      continue;
    }

    const auto matching = std::find_if(result.process_local_channel_services.begin(),
                                       result.process_local_channel_services.end(),
                                       [&](const auto& service) {
                                         return service.source == edge.source &&
                                                service.queue_policy == edge.queue_policy;
                                       });
    std::size_t service_index{};
    if (matching == result.process_local_channel_services.end()) {
      const std::string service_id =
          process_local_service_id(plan_generation, edge.source, edge.queue_policy);
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
      result.process_local_channel_services.push_back(make_process_local_service_record(
          edge.source, service_id, contract, 0U, false, delivery_kind));
      result.process_local_channel_publishers.push_back(ProcessLocalChannelPublisherPlanRecord{
          .source = edge.source,
          .service_id = service_id,
          .max_loaned_samples = edge.producer_max_emits_per_compute,
          .replacement_attribution_required = edge.queue_policy != QueuePolicy::kBlocking,
      });
      service_index = result.process_local_channel_services.size() - 1U;
    } else {
      service_index = static_cast<std::size_t>(
          std::distance(result.process_local_channel_services.begin(), matching));
    }
    auto& service = result.process_local_channel_services[service_index];
    std::uint32_t subscriber_count{};
    if (!checked_add(service.max_subscribers, 1U, subscriber_count)) {
      reject(result, kProcessLocalChannelInvalid, source_for(snapshot, target->owner, edge.target));
      continue;
    }
    service.max_subscribers = subscriber_count;
    service.max_subscriber_queue_depth =
        std::max(service.max_subscriber_queue_depth, edge.queue_depth);
    service.max_subscriber_borrowed_samples =
        std::max(service.max_subscriber_borrowed_samples, edge.queue_depth);
    result.process_local_channel_subscribers.push_back(ProcessLocalChannelSubscriberPlanRecord{
        .edge = edge.edge,
        .source = edge.source,
        .target = edge.target,
        .service_id = service.service_id,
        .requested_queue_depth = edge.queue_depth,
        .max_borrowed_samples = edge.queue_depth,
        .cache_capacity = edge.cache_capacity,
        .readiness_capacity = std::max(edge.queue_depth, edge.cache_capacity),
        .event_wake = delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ,
    });
  }
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

  // Aggregate device-backed pools by process/device only after final cover
  // validation. The accepted partition count is intentionally absent from the
  // key and capacity equations.
  std::vector<TensorDeviceResourcePlanRecord> device_resources;
  const auto append_device_pool = [&](ProcessGroupId process_group,
                                      DeviceId device,
                                      std::string_view allocator_root,
                                      std::uint32_t slots,
                                      std::uint32_t loans,
                                      std::uint32_t controls,
                                      std::uint32_t readiness,
                                      std::uint32_t releases,
                                      std::uint32_t retirements,
                                      std::uint32_t quarantines) -> bool {
    if (!device.valid()) {
      return true;
    }
    auto found =
        std::find_if(device_resources.begin(), device_resources.end(), [&](const auto& resource) {
          return resource.process_group == process_group && resource.device == device;
        });
    if (found == device_resources.end()) {
      device_resources.push_back(TensorDeviceResourcePlanRecord{
          .plan_generation = plan_generation,
          .run_generation = 0U,
          .process_group = process_group,
          .device = device,
          .allocator_root_id = std::string(allocator_root),
          .stream_id = tensor_stream_id(plan_generation, process_group, device),
          .pool_count = 0U,
          .slot_capacity = 0U,
          .loan_credit_capacity = 0U,
          .control_capacity = 0U,
          .readiness_capacity = 0U,
          .release_capacity = 0U,
          .retirement_capacity = 0U,
          .quarantine_capacity = 0U,
          .materialize_before = LifecycleStage::kArm,
          .per_partition_stream = false,
          .per_partition_event = false,
          .per_partition_green_context = false,
          .gpu_resident_backend_selected = false,
          .fallback_rejected = true,
      });
      found = std::prev(device_resources.end());
    }
    std::uint32_t pool_count{};
    std::uint32_t slot_capacity{};
    std::uint32_t loan_capacity{};
    std::uint32_t control_capacity{};
    std::uint32_t readiness_capacity{};
    std::uint32_t release_capacity{};
    std::uint32_t retirement_capacity{};
    std::uint32_t quarantine_capacity{};
    if (found->allocator_root_id != allocator_root ||
        !checked_add(found->pool_count, 1U, pool_count) ||
        !checked_add(found->slot_capacity, slots, slot_capacity) ||
        !checked_add(found->loan_credit_capacity, loans, loan_capacity) ||
        !checked_add(found->control_capacity, controls, control_capacity) ||
        !checked_add(found->readiness_capacity, readiness, readiness_capacity) ||
        !checked_add(found->release_capacity, releases, release_capacity) ||
        !checked_add(found->retirement_capacity, retirements, retirement_capacity) ||
        !checked_add(found->quarantine_capacity, quarantines, quarantine_capacity)) {
      return false;
    }
    found->pool_count = pool_count;
    found->slot_capacity = slot_capacity;
    found->loan_credit_capacity = loan_capacity;
    found->control_capacity = control_capacity;
    found->readiness_capacity = readiness_capacity;
    found->release_capacity = release_capacity;
    found->retirement_capacity = retirement_capacity;
    found->quarantine_capacity = quarantine_capacity;
    return true;
  };
  bool device_resources_valid = true;
  validate_operator_device_obligations(result);
  for (const auto& pool : result.tensor_allocations) {
    device_resources_valid =
        device_resources_valid && append_device_pool(pool.allocator_process_group,
                                                     pool.device,
                                                     pool.allocator_root_id,
                                                     pool.slot_count,
                                                     pool.max_outstanding_loans,
                                                     pool.control_capacity,
                                                     pool.readiness_capacity,
                                                     pool.release_capacity,
                                                     pool.retirement_capacity,
                                                     pool.quarantine_capacity);
  }
  for (const auto& pool : result.tensor_field_allocations) {
    device_resources_valid =
        device_resources_valid && append_device_pool(pool.allocator_process_group,
                                                     pool.device,
                                                     pool.allocator_root_id,
                                                     pool.slot_count,
                                                     pool.max_outstanding_loans,
                                                     pool.control_capacity,
                                                     pool.readiness_capacity,
                                                     pool.release_capacity,
                                                     pool.retirement_capacity,
                                                     pool.quarantine_capacity);
  }
  for (const auto& pool : result.tensor_interface_pools) {
    device_resources_valid =
        device_resources_valid && append_device_pool(pool.allocator_process_group,
                                                     pool.device,
                                                     pool.allocator_root_id,
                                                     pool.slot_count,
                                                     pool.publisher_loan_capacity,
                                                     pool.control_capacity,
                                                     pool.readiness_capacity,
                                                     pool.release_capacity,
                                                     pool.retirement_capacity,
                                                     pool.quarantine_capacity);
  }
  // A CUDA-device or pinned-host BufferHandle consumer needs a process-local stream even though it
  // owns no payload pool. Materialize one stream-only resource per target process/device so
  // selection can insert the producer-generation wait before the callback receives its Tensor.
  for (const auto& service : result.cross_process_channel_services) {
    std::vector<DeviceId> stream_devices;
    if (!service.tensor_fields.empty()) {
      for (const auto& field : service.tensor_fields) {
        const bool stream_ordered = field.memory_kind == MemoryKind::kCudaDevice ||
                                    field.memory_kind == MemoryKind::kPinnedHost;
        if (stream_ordered && field.device.valid() &&
            std::find(stream_devices.begin(), stream_devices.end(), field.device) ==
                stream_devices.end()) {
          stream_devices.push_back(field.device);
        }
      }
    } else {
      const bool stream_ordered = service.tensor_memory_kind == MemoryKind::kCudaDevice ||
                                  service.tensor_memory_kind == MemoryKind::kPinnedHost;
      if (stream_ordered && service.tensor_device.valid()) {
        stream_devices.push_back(service.tensor_device);
      }
    }
    if (!service.tensor_payload || stream_devices.empty()) {
      continue;
    }
    std::vector<ProcessGroupId> process_groups = service.target_process_groups;
    if (service.payload_identity.kind == PayloadIdentityKind::kSchema &&
        service.payload_identity.payload_kind == PayloadKind::kDescriptorBacked) {
      process_groups.push_back(service.source_process_group);
    }
    for (const DeviceId device : stream_devices) {
      for (const ProcessGroupId process_group : process_groups) {
        if (!process_group) {
          continue;
        }
        const auto existing = std::find_if(
            device_resources.begin(), device_resources.end(), [&](const auto& resource) {
              return resource.process_group == process_group && resource.device == device;
            });
        if (existing == device_resources.end()) {
          device_resources.push_back(TensorDeviceResourcePlanRecord{
              .plan_generation = plan_generation,
              .run_generation = 0U,
              .process_group = process_group,
              .device = device,
              .allocator_root_id = allocator_root_id(plan_generation, process_group, device),
              .stream_id = tensor_stream_id(plan_generation, process_group, device),
              .materialize_before = LifecycleStage::kArm,
              .fallback_rejected = true,
          });
        }
      }
    }
  }
  if (!device_resources_valid) {
    reject(result, kTensorOverflow);
  } else {
    std::sort(
        device_resources.begin(), device_resources.end(), [](const auto& left, const auto& right) {
          return left.process_group == right.process_group
                     ? left.device.value < right.device.value
                     : left.process_group.value < right.process_group.value;
        });
    assign_linear_tensor_streams(result, process_placements, device_resources);
    result.tensor_device_resources = std::move(device_resources);
  }
  std::vector<std::pair<std::uint64_t, DeviceId>> unconsumed_device_bindings;
  unconsumed_device_bindings.reserve(device_by_port.size());
  for (const auto& [port, device] : device_by_port) {
    if (!consumed_device_bindings.contains(port)) {
      unconsumed_device_bindings.emplace_back(port, device);
    }
  }
  std::sort(unconsumed_device_bindings.begin(),
            unconsumed_device_bindings.end(),
            [](const auto& left, const auto& right) { return left.first < right.first; });
  for (const auto& [port, device] : unconsumed_device_bindings) {
    const auto* record = find_port(snapshot, GraphPortId{port});
    reject(result,
           kTensorDeviceBindingUnused,
           source_for(snapshot, record == nullptr ? VertexId{} : record->owner, PortId{port}),
           port_pointer(PortId{port}, "/tensor_device_binding"),
           "binding consumed by one admitted Tensor resource",
           "device=" + std::to_string(device.value));
  }

  std::unordered_map<std::string, std::uint32_t> service_generation_capacity;
  for (const auto& generation : result.imported_generation_capacities) {
    auto [found, inserted] = service_generation_capacity.emplace(
        generation.endpoint_or_service_id, generation.endpoint_max_generations);
    if (!inserted) {
      found->second = std::max(found->second, generation.endpoint_max_generations);
    }
  }
  std::uint32_t runtime_generation_capacity = 0U;
  bool generation_capacity_valid = true;
  for (const auto& [service, capacity] : service_generation_capacity) {
    static_cast<void>(service);
    if (!checked_add(runtime_generation_capacity, capacity, runtime_generation_capacity)) {
      generation_capacity_valid = false;
      break;
    }
  }
  if (!generation_capacity_valid ||
      (!result.imported_generation_capacities.empty() && runtime_generation_capacity == 0U)) {
    reject(result, kTensorOverflow);
  } else {
    for (auto& generation : result.imported_generation_capacities) {
      generation.service_max_generations =
          service_generation_capacity.at(generation.endpoint_or_service_id);
      generation.runtime_max_generations = runtime_generation_capacity;
    }
  }

  for (const auto& tensor : result.tensor_allocations) {
    if (!validate_tensor_allocation_record(tensor, plan_generation)) {
      reject(result, kReadinessInvalid, source_for(snapshot, {}, tensor.output));
    }
  }
  for (const auto& tensor : result.tensor_field_allocations) {
    if (!validate_tensor_field_allocation_record(tensor, plan_generation) ||
        !validate_tensor_field_capacity_contributors(tensor, result.pool_capacity_contributors)) {
      reject(result,
             kReadinessInvalid,
             source_for(snapshot, {}, tensor.output),
             port_pointer(tensor.output, "/tensor_fields/" + std::to_string(tensor.field_id)));
    }
  }
  for (const auto& tensor : result.tensor_interface_pools) {
    if (tensor.readiness_capacity < tensor.publisher_loan_capacity ||
        tensor.control_capacity < tensor.slot_count ||
        tensor.release_capacity < tensor.slot_count ||
        tensor.retirement_capacity < tensor.slot_count ||
        tensor.quarantine_capacity < tensor.slot_count) {
      reject(result, kReadinessInvalid, source_for(snapshot, {}, tensor.graph_port));
    }
  }
  return result;
}

}  // namespace holoscan::detail
