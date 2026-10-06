/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/execution_plan.hpp>

#include "../channel/channel_payload_limits.hpp"
#include "../channel/endpoint_catalog_state.hpp"
#include "../core/plan_record_contracts.hpp"
#include "../runtime/executable_factory_inventory.hpp"
#include "../runtime/runtime_diagnostic_sink.hpp"
#include "parameter_generation_zero.hpp"
#include "process_plan_records.hpp"

namespace holoscan::detail {

/// Select the private runtime delivery mechanism for one process-local service.
enum class ProcessLocalChannelDeliveryKind : std::uint8_t {
  kHoloMQ = 0,
  kDirectRetained = 1,
};

/// Classify the cold compiler proof for one process-local HoloMQ event wake.
enum class ProcessLocalHoloMQEventWakeReason : std::uint8_t {
  kSuppressed = 0,
  kNotHoloMQ = 1,
  kEdgeNotAccepted = 2,
  kNonProcessLocal = 3,
  kEmptyService = 4,
  kIncompleteServiceFanout = 5,
  kUnsupportedPayloadContract = 6,
  kInvalidOperatorOwnership = 7,
  kOwnerLoop = 8,
  kCallbackShape = 9,
  kUnsupportedQueuePolicy = 10,
  kPartitionAmbiguity = 11,
  kSplitExecutionPartition = 12,
  kReplayAmbiguity = 13,
  kUnclassified = 14,
  kForcedControl = 15,
};

/// Return the stable diagnostic token for one event-wake reason.
[[nodiscard]] constexpr std::string_view process_local_holomq_event_wake_reason_token(
    const ProcessLocalHoloMQEventWakeReason reason) noexcept {
  switch (reason) {
    case ProcessLocalHoloMQEventWakeReason::kSuppressed:
      return "suppressed";
    case ProcessLocalHoloMQEventWakeReason::kNotHoloMQ:
      return "not_holomq";
    case ProcessLocalHoloMQEventWakeReason::kEdgeNotAccepted:
      return "edge_not_accepted";
    case ProcessLocalHoloMQEventWakeReason::kNonProcessLocal:
      return "non_process_local";
    case ProcessLocalHoloMQEventWakeReason::kEmptyService:
      return "empty_service";
    case ProcessLocalHoloMQEventWakeReason::kIncompleteServiceFanout:
      return "incomplete_service_fanout";
    case ProcessLocalHoloMQEventWakeReason::kUnsupportedPayloadContract:
      return "unsupported_payload_contract";
    case ProcessLocalHoloMQEventWakeReason::kInvalidOperatorOwnership:
      return "invalid_operator_ownership";
    case ProcessLocalHoloMQEventWakeReason::kOwnerLoop:
      return "owner_loop";
    case ProcessLocalHoloMQEventWakeReason::kCallbackShape:
      return "callback_shape";
    case ProcessLocalHoloMQEventWakeReason::kUnsupportedQueuePolicy:
      return "unsupported_queue_policy";
    case ProcessLocalHoloMQEventWakeReason::kPartitionAmbiguity:
      return "partition_ambiguity";
    case ProcessLocalHoloMQEventWakeReason::kSplitExecutionPartition:
      return "split_execution_partition";
    case ProcessLocalHoloMQEventWakeReason::kReplayAmbiguity:
      return "replay_ambiguity";
    case ProcessLocalHoloMQEventWakeReason::kUnclassified:
      return "unclassified";
    case ProcessLocalHoloMQEventWakeReason::kForcedControl:
      return "forced_control";
  }
  return "unclassified";
}

/// Select one deterministic retained-wake reason across every service member.
[[nodiscard]] constexpr ProcessLocalHoloMQEventWakeReason
combine_process_local_holomq_event_wake_reasons(
    const ProcessLocalHoloMQEventWakeReason current,
    const ProcessLocalHoloMQEventWakeReason candidate) noexcept {
  const auto retained = [](const ProcessLocalHoloMQEventWakeReason reason) {
    return reason != ProcessLocalHoloMQEventWakeReason::kSuppressed &&
           reason != ProcessLocalHoloMQEventWakeReason::kNotHoloMQ;
  };
  if (!retained(candidate)) {
    return current;
  }
  if (!retained(current) ||
      static_cast<std::uint8_t>(candidate) < static_cast<std::uint8_t>(current)) {
    return candidate;
  }
  return current;
}

/// Freeze one field-specific descriptor-schema storage contract.
struct TensorFieldServiceStorageContract {
  std::uint16_t field_id{};
  std::uint16_t tensor_index{};
  std::uint64_t max_byte_span{};
  std::string pool_id;
  std::optional<DLDataType> dtype;
  std::optional<std::uint32_t> rank;
  MemoryKind memory_kind{MemoryKind::kUnknown};
  DeviceId device{};
  InterfaceTensorEmitPolicy tensor_emit_policy{InterfaceTensorEmitPolicy::kRequireExportable};

  friend bool operator==(const TensorFieldServiceStorageContract& left,
                         const TensorFieldServiceStorageContract& right) noexcept {
    return left.field_id == right.field_id && left.tensor_index == right.tensor_index &&
           left.max_byte_span == right.max_byte_span && left.pool_id == right.pool_id &&
           dl_data_type_exactly_equal(left.dtype, right.dtype) && left.rank == right.rank &&
           left.memory_kind == right.memory_kind && left.device == right.device &&
           left.tensor_emit_policy == right.tensor_emit_policy;
  }
};

/// Freeze one accepted reusable-Graph clock import and its exact resolved domain.
///
/// The authored Graph snapshot remains the public provenance record. This private
/// duplicate is deliberate startup-authentication evidence: Runtime compares it
/// with the reconstructed snapshot before lifecycle so a stale or corrupted
/// import/binding inventory cannot silently resolve to another valid clock.
struct ClockImportResolutionPlanRecord {
  GraphId child_graph{};
  ClockKey child_clock;
  bool is_default{};
  GraphId parent_graph{};
  ClockKey parent_clock;
  ClockId resolved_clock{};

  friend bool operator==(const ClockImportResolutionPlanRecord&,
                         const ClockImportResolutionPlanRecord&) = default;
};

/// Classify the compiler-proven readiness root for one closed activation island.
enum class ClosedIslandReadinessRootKind : std::uint8_t {
  kPhysical = 0,
  kClock = 1,
  kNotification = 2,
};

/// Freeze one exact two-Operator activation island admitted for composite dispatch.
///
/// This record is private Runtime construction authority. It is derived from
/// canonical plan relations, omitted from every public projection and digest,
/// and reconstructed exactly before Runtime creates user or backend state.
struct ClosedIslandActivationPlanRecord {
  // Current derivation admits same-partition closed islands only:
  // source_partition == successor_partition. derive_edge rejects asymmetric
  // candidates before this private record is published.
  ExecutionPartitionId source_partition{};
  ExecutionPartitionId successor_partition{};
  std::uint32_t execution_lane{};
  ProcessGroupId process_group{};
  VertexId source{};
  VertexId successor{};
  EdgeId edge{};
  PortId source_output{};
  PortId successor_input{};
  ClosedIslandReadinessRootKind readiness_root{ClosedIslandReadinessRootKind::kPhysical};
  SourceId readiness_source{};
  ClockId readiness_clock{};
  SourceId successor_readiness_source{};

  friend bool operator==(const ClosedIslandActivationPlanRecord&,
                         const ClosedIslandActivationPlanRecord&) = default;
};

/// Freeze one exact partition whose Operators may execute on the Runtime controller thread.
///
/// This record is private Runtime construction authority. Compilation derives
/// its stable topological order from canonical accepted relations, omits it
/// from every public projection and digest, and reconstructs it exactly before
/// Runtime creates user or backend state.
struct InlinePartitionExecutionPlanRecord {
  ExecutionPartitionId partition{};
  std::uint32_t execution_lane{};
  ProcessGroupId process_group{};
  VertexId readiness_root{};
  ClosedIslandReadinessRootKind readiness_kind{ClosedIslandReadinessRootKind::kPhysical};
  SourceId readiness_source{};
  ClockId readiness_clock{};
  std::vector<VertexId> operators;
  std::vector<SourceId> readiness_sources;
  std::vector<EdgeId> edges;

  friend bool operator==(const InlinePartitionExecutionPlanRecord&,
                         const InlinePartitionExecutionPlanRecord&) = default;
};

/// Identify one CUDA Graph region within its owning Graph.
struct CUDAGraphRegionPlanRef {
  GraphId graph{};
  RegionId region{};

  friend bool operator==(const CUDAGraphRegionPlanRef&, const CUDAGraphRegionPlanRef&) = default;
};

/// Freeze one callback owner within a partition-local execution runner.
///
/// The record joins the canonical Graph vertex to the reconstructed Operator
/// identity and its already admitted temporal source. Vector order in the
/// enclosing runner record is the stable topological callback order.
struct PartitionRunnerOperatorPlanRecord {
  VertexId vertex{};
  OperatorInstanceId operator_id{};
  /// Present when Runtime must construct `CUDAGraphRegionOperator` instead of the source factory.
  std::optional<CUDAGraphRegionPlanRef> fused_cuda_graph_region;
  SourceId readiness_source{};  ///< Zero when no temporal trigger owns the topology source.
  /// Absent when the topology source has no temporal-trigger record.
  std::optional<TemporalRecordTriggerKind> trigger_kind;
  bool topology_root{};  ///< True when no payload edge from another runner member precedes it.

  friend bool operator==(const PartitionRunnerOperatorPlanRecord&,
                         const PartitionRunnerOperatorPlanRecord&) = default;
};

/// Freeze one private partition-local execution-runner identity.
///
/// This record is Runtime construction authority only. It is reconstructed
/// exactly from accepted public plan relations before backend state exists,
/// and is intentionally absent from public projections, canonical JSON, and
/// the plan digest.
struct PartitionRunnerPlanRecord {
  std::uint32_t ordinal{};
  ExecutionPartitionId partition{};
  std::uint32_t execution_lane{};
  ProcessGroupId process_group{};
  std::vector<PartitionRunnerOperatorPlanRecord> operators;
  std::vector<EdgeId> internal_edges;
  std::vector<EdgeId> incoming_boundary_edges;
  std::vector<EdgeId> outgoing_boundary_edges;

  friend bool operator==(const PartitionRunnerPlanRecord&,
                         const PartitionRunnerPlanRecord&) = default;
};

/// Preserve one resolved Tensor connection in a CUDA Graph region.
struct CUDAGraphCaptureEdgeRecord {
  EdgeId edge{};
  VertexId source{};
  PortId source_port{};
  VertexId target{};
  PortId target_port{};

  friend constexpr bool operator==(const CUDAGraphCaptureEdgeRecord&,
                                   const CUDAGraphCaptureEdgeRecord&) noexcept = default;
};

/// Preserve the accepted trigger form for one CUDA Graph region member.
///
/// This is a reduced scheduling-contract summary, not a copy of the member's complete Contract.
/// A source without ingress may use OnClock, OnReadable, OnNotified, OnDoorbell, or OnSpin. Other
/// members may use OnEach or normalized OnAnyInput on their sole expected input. Future
/// multi-input support can extend this private record without changing the public SDK API.
/// Operator completion remains authoritative in the separate OperatorCompletionPlanRecord and is
/// joined to an inputless capture source through the shared OperatorRef owner.
struct CUDAGraphCaptureMemberContractRecord {
  OperatorRef owner{};
  TemporalRecordTriggerKind trigger{TemporalRecordTriggerKind::kAnyInput};
  PortId trigger_input{};
  bool synthesized_default{};
};

/// Bind one captured operator identity to its Graph vertex.
struct CUDAGraphCaptureMemberRecord {
  OperatorRef operator_ref{};
  VertexId vertex{};
};

/// Describe how later compiler stages will capture one validated CUDA Graph region.
struct CUDAGraphCapturePlanRecord {
  GraphId graph{};
  RegionId region{};
  std::string region_name;
  OperatorRef source{};
  OperatorRef sink{};
  std::vector<CUDAGraphCaptureMemberRecord> members;
  std::vector<CUDAGraphCaptureEdgeRecord> internal_edges;
  std::optional<CUDAGraphCaptureEdgeRecord> ingress;
  std::optional<CUDAGraphCaptureEdgeRecord> egress;
  std::vector<CUDAGraphCaptureMemberContractRecord> contracts;
};

/// Classify one fixed buffer used while capturing and launching a fused CUDA Graph region.
enum class CUDAGraphCaptureBufferKind : std::uint8_t {
  kIngress = 0,
  kInternal = 1,
  kEgress = 2,
};

/// Preserve the fixed Tensor allocation required for one CUDA Graph region edge.
///
/// `fixed_allocation_id` identifies the single allocation that Runtime will retain for the
/// RunSession. Two chained regions reference the same identity for their shared edge. The ordinary
/// pool and allocator identities provide checked capacity and placement; they do not identify the
/// address replayed by the CUDA Graph. `fixed_shape` is the exact logical shape used for capture
/// and every later activation; `axes` remains the allocation's maximum capacity evidence.
struct CUDAGraphCaptureBufferPlanRecord {
  CUDAGraphCaptureBufferKind kind{CUDAGraphCaptureBufferKind::kInternal};
  EdgeId edge{};
  PortId source_port{};
  PortId target_port{};
  PayloadIdentityRecord payload_identity;
  TensorRepresentation layout;
  TensorPoolBounds bounds;
  std::vector<TensorAxisBounds> axes;
  std::vector<std::uint64_t> fixed_shape;
  std::uint32_t active_slot_count{};
  std::uint64_t aggregate_byte_capacity{};
  MemoryKind memory_kind{MemoryKind::kUnknown};
  DeviceId device{};  ///< CUDA ordinal for device memory; unresolved for pinned-host boundaries.
  std::string fixed_allocation_id;
  std::string pool_id;
  std::string allocator_root_id;

  friend bool operator==(const CUDAGraphCaptureBufferPlanRecord& left,
                         const CUDAGraphCaptureBufferPlanRecord& right) noexcept {
    return left.kind == right.kind && left.edge == right.edge &&
           left.source_port == right.source_port && left.target_port == right.target_port &&
           payload_identity_exactly_equal(left.payload_identity, right.payload_identity) &&
           tensor_port_layout_exactly_equal(left.layout, right.layout) &&
           left.bounds.max_rank == right.bounds.max_rank &&
           left.bounds.max_byte_span == right.bounds.max_byte_span && left.axes == right.axes &&
           left.fixed_shape == right.fixed_shape &&
           left.active_slot_count == right.active_slot_count &&
           left.aggregate_byte_capacity == right.aggregate_byte_capacity &&
           left.memory_kind == right.memory_kind && left.device == right.device &&
           left.fixed_allocation_id == right.fixed_allocation_id && left.pool_id == right.pool_id &&
           left.allocator_root_id == right.allocator_root_id;
  }
};

/// Describe one validated region after it is replaced in the execution plan by a fused operator.
///
/// The fused operator uses the region source's logical vertex identity and temporal contract. Its
/// Runtime implementation is `CUDAGraphRegionOperator`; the original member factories remain in
/// the capture plan so startup can call each member's `compute()` once under stream capture.
struct CUDAGraphExecutionRegionPlanRecord {
  GraphId graph{};
  RegionId region{};
  VertexId fused_operator{};
  ExecutionPartitionId partition{};
  std::uint32_t execution_lane{};
  ProcessGroupId process_group{};
  DeviceId device{};
  std::string stream_id;
  std::optional<std::string> green_context_id;
  std::optional<CUDAGraphCaptureEdgeRecord> ingress;
  std::optional<CUDAGraphCaptureEdgeRecord> egress;
  std::vector<CUDAGraphCaptureBufferPlanRecord> buffers;

  friend bool operator==(const CUDAGraphExecutionRegionPlanRecord&,
                         const CUDAGraphExecutionRegionPlanRecord&) = default;
};

/// Derive the process-local edge families authorized to bypass HoloMQ transport.
[[nodiscard]] inline ProcessLocalChannelDeliveryKind process_local_channel_delivery_kind(
    const PayloadEdgePlanRecord& edge) noexcept {
  // REQ: CHAN-096
  // REQ: CHAN-097
  // REQ: CHAN-098
  // REQ: CHAN-099
  // REQ: CHAN-116
  // REQ: CHAN-117
  // REQ: CHAN-118
  // REQ: CHAN-119
  const bool direct_inline_payload =
      edge.payload_identity.payload_kind == PayloadKind::kInlineTyped &&
      (edge.payload_identity.kind == PayloadIdentityKind::kScalar ||
       edge.payload_identity.kind == PayloadIdentityKind::kPod);
  const bool direct_generated_schema =
      edge.payload_identity.payload_kind == PayloadKind::kSerializedBytes &&
      edge.payload_identity.kind == PayloadIdentityKind::kSchema;
  const bool direct_process_local_object =
      edge.payload_identity.payload_kind == PayloadKind::kInlineTyped &&
      is_py_object_payload_identity(edge.payload_identity);
  const bool direct_tensor = edge.payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
                             edge.payload_identity.kind == PayloadIdentityKind::kTensor;
  const bool direct_descriptor_tensor_schema =
      edge.payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
      edge.payload_identity.kind == PayloadIdentityKind::kSchema;
  const bool supported_direct_queue_policy =
      edge.queue_policy == QueuePolicy::kBlocking || edge.queue_policy == QueuePolicy::kDropOldest;
  return edge.disposition == PlanDisposition::kAccepted &&
                 edge.locality == ChannelLocalityKind::kProcessLocal && edge.fanout == 1U &&
                 supported_direct_queue_policy &&
                 (direct_inline_payload || direct_generated_schema || direct_tensor ||
                  direct_descriptor_tensor_schema || direct_process_local_object)
             ? ProcessLocalChannelDeliveryKind::kDirectRetained
             : ProcessLocalChannelDeliveryKind::kHoloMQ;
}

/// Return whether one member of a frozen HoloMQ service is covered by the synchronous
/// activation-cut protocol.
///
/// Callers must evaluate every subscriber in the completed source family and suppress wake
/// attachment only when every member returns `kSuppressed`. A fanout family is admitted only for
/// scalar/POD values, bounded generated schemas, raw Tensor, or bounded descriptor Tensor schemas
/// and only when its policy-specific sibling services collectively contain the source's complete
/// fanout. Multiple source outputs share the same serialized post-compute activation cut, and each
/// eligible source cut materializes its contribution to a multi-input target before the next
/// temporal candidate freeze. Temporal and lifecycle records are intentionally absent because
/// selectors consume typed readiness only after that cut.
// This ordered classifier preserves one stable reason precedence across every eligibility fact;
// splitting the decisions would obscure that diagnostic contract.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] inline ProcessLocalHoloMQEventWakeReason process_local_holomq_event_wake_reason(
    const PayloadEdgePlanRecord& edge, const ProcessLocalChannelDeliveryKind delivery_kind,
    const std::uint32_t service_max_subscribers,
    const std::span<const ExecutionPartitionRecord> execution_partitions,
    const VertexId source_owner, const VertexId target_owner) noexcept {
  const bool inline_value = edge.payload_identity.payload_kind == PayloadKind::kInlineTyped &&
                            ((edge.payload_identity.kind == PayloadIdentityKind::kScalar &&
                              edge.payload_identity.scalar.has_value()) ||
                             (edge.payload_identity.kind == PayloadIdentityKind::kPod &&
                              edge.payload_identity.pod.has_value()));
  const bool generated_schema =
      edge.payload_identity.payload_kind == PayloadKind::kSerializedBytes &&
      edge.payload_identity.kind == PayloadIdentityKind::kSchema &&
      edge.payload_identity.schema.has_value();
  const bool raw_tensor = edge.payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
                          edge.payload_identity.kind == PayloadIdentityKind::kTensor;
  const bool descriptor_tensor_schema =
      edge.payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
      edge.payload_identity.kind == PayloadIdentityKind::kSchema &&
      edge.payload_identity.schema.has_value();
  if (delivery_kind != ProcessLocalChannelDeliveryKind::kHoloMQ) {
    return ProcessLocalHoloMQEventWakeReason::kNotHoloMQ;
  }
  if (edge.disposition != PlanDisposition::kAccepted) {
    return ProcessLocalHoloMQEventWakeReason::kEdgeNotAccepted;
  }
  if (edge.locality != ChannelLocalityKind::kProcessLocal) {
    return ProcessLocalHoloMQEventWakeReason::kNonProcessLocal;
  }
  if (service_max_subscribers == 0U) {
    return ProcessLocalHoloMQEventWakeReason::kEmptyService;
  }
  if (!inline_value && !generated_schema && !raw_tensor && !descriptor_tensor_schema) {
    return ProcessLocalHoloMQEventWakeReason::kUnsupportedPayloadContract;
  }
  if (!source_owner || !target_owner) {
    return ProcessLocalHoloMQEventWakeReason::kInvalidOperatorOwnership;
  }
  if (source_owner == target_owner) {
    return ProcessLocalHoloMQEventWakeReason::kOwnerLoop;
  }
  if (edge.queue_policy != QueuePolicy::kBlocking &&
      edge.queue_policy != QueuePolicy::kDropOldest) {
    return ProcessLocalHoloMQEventWakeReason::kUnsupportedQueuePolicy;
  }

  auto source_partition = ExecutionPartitionId{};
  auto target_partition = ExecutionPartitionId{};
  for (const auto& partition : execution_partitions) {
    const auto source_membership =
        std::count(partition.members.begin(), partition.members.end(), source_owner);
    const auto target_membership =
        std::count(partition.members.begin(), partition.members.end(), target_owner);
    if (!partition.id || source_membership > 1 || target_membership > 1 ||
        (source_membership != 0 && source_partition) ||
        (target_membership != 0 && target_partition)) {
      return ProcessLocalHoloMQEventWakeReason::kPartitionAmbiguity;
    }
    if (source_membership == 1) {
      source_partition = partition.id;
    }
    if (target_membership == 1) {
      target_partition = partition.id;
    }
  }
  if (!source_partition || !target_partition) {
    return ProcessLocalHoloMQEventWakeReason::kPartitionAmbiguity;
  }
  return source_partition == target_partition
             ? ProcessLocalHoloMQEventWakeReason::kSuppressed
             : ProcessLocalHoloMQEventWakeReason::kSplitExecutionPartition;
}

/// Freeze one private process-local Channel service shared by compatible fanout edges.
///
/// Compiler lowering creates one record per source-port and queue-policy pair.
/// The record is Runtime construction authority only: it is deliberately absent
/// from the public ExecutionPlan header, accessors, canonical JSON, and digest.
struct ProcessLocalChannelServicePlanRecord {
  PortId source{};
  std::string service_id;
  PayloadIdentityRecord payload_identity;
  std::uint64_t payload_size_bytes{};            ///< Required logical payload frame bytes.
  std::uint64_t selected_inline_bucket_bytes{};  ///< Exact HoloMQ bucket, or zero when direct.
  bool tensor_payload{};
  std::uint64_t tensor_max_byte_span{};
  std::string tensor_pool_id;
  MemoryKind tensor_memory_kind{MemoryKind::kUnknown};
  DeviceId tensor_device{};
  std::vector<TensorFieldServiceStorageContract> tensor_fields;
  ProcessLocalChannelDeliveryKind delivery_kind{ProcessLocalChannelDeliveryKind::kHoloMQ};
  bool cross_process_transport{};
  QueuePolicy queue_policy{QueuePolicy::kBlocking};
  std::uint32_t max_subscribers{};
  std::uint32_t max_subscriber_queue_depth{};
  std::uint32_t max_subscriber_borrowed_samples{};
  std::uint32_t publisher_history_depth{};
  bool event_wake{};
  ProcessLocalHoloMQEventWakeReason event_wake_reason{
      ProcessLocalHoloMQEventWakeReason::kUnclassified};
};

/// Freeze one private publisher-local request for a process-local Channel service.
struct ProcessLocalChannelPublisherPlanRecord {
  PortId source{};
  std::string service_id;
  std::uint32_t max_loaned_samples{};
  bool replacement_attribution_required{};
};

/// Freeze one private subscriber-local request for a process-local payload edge.
struct ProcessLocalChannelSubscriberPlanRecord {
  EdgeId edge{};
  PortId source{};
  PortId target{};
  std::string service_id;
  std::uint32_t requested_queue_depth{};
  std::uint32_t max_borrowed_samples{};
  std::uint32_t cache_capacity{};
  std::uint32_t readiness_capacity{};
  bool event_wake{};
  /// Producer Tensor field this subscriber takes when its edge narrows the schema payload to a
  /// Tensor. Narrowing is per subscriber: the shared service still carries the whole sample, and
  /// sibling subscribers on the same service continue to receive the schema root.
  std::optional<TensorProjectionRecord> tensor_projection;
};

/// Freeze one private cross-process Channel service for compatible fanout arms.
///
/// The anchor edge and target retain deterministic provenance. Every compatible
/// edge from the same source and queue-policy family shares this service, while
/// subscriber records retain exact per-edge targets and bounds.
/// Like the process-local records, these are Runtime construction authority
/// only and stay out of the public plan surface, canonical JSON, and digest.
struct CrossProcessChannelServicePlanRecord {
  EdgeId edge{};
  PortId source{};
  PortId target{};
  std::string service_id;
  PayloadIdentityRecord payload_identity;
  std::uint64_t payload_size_bytes{};            ///< Required logical payload frame bytes.
  std::uint64_t selected_inline_bucket_bytes{};  ///< Exact HoloMQ bucket specialization.
  bool tensor_payload{};
  std::uint64_t tensor_max_byte_span{};
  std::string tensor_pool_id;
  MemoryKind tensor_memory_kind{MemoryKind::kUnknown};
  DeviceId tensor_device{};
  std::vector<TensorFieldServiceStorageContract> tensor_fields;
  QueuePolicy queue_policy{QueuePolicy::kBlocking};
  std::uint32_t max_subscribers{};
  std::uint32_t max_subscriber_queue_depth{};
  std::uint32_t max_subscriber_borrowed_samples{};
  std::uint32_t publisher_history_depth{};
  bool event_wake{};
  InterfaceTensorEmitPolicy tensor_emit_policy{InterfaceTensorEmitPolicy::kRequireExportable};
  ProcessGroupId source_process_group{};
  ProcessGroupId target_process_group{};
  std::vector<ProcessGroupId> target_process_groups;
};

/// Freeze one private publisher-local request for a cross-process Channel edge.
struct CrossProcessChannelPublisherPlanRecord {
  EdgeId edge{};
  PortId source{};
  std::string service_id;
  std::uint32_t max_loaned_samples{};
  bool replacement_attribution_required{};
};

/// Freeze one private subscriber-local request for a cross-process Channel edge.
struct CrossProcessChannelSubscriberPlanRecord {
  EdgeId edge{};
  PortId source{};
  PortId target{};
  std::string service_id;
  std::uint32_t requested_queue_depth{};
  std::uint32_t max_borrowed_samples{};
  std::uint32_t cache_capacity{};
  std::uint32_t readiness_capacity{};
  bool event_wake{};
};

/// Common immutable payload, queue, and Tensor facts shared by Channel service record families.
struct TensorServiceContract {
  PayloadIdentityRecord payload_identity;
  std::uint64_t payload_size_bytes{};
  std::uint64_t selected_inline_bucket_bytes{};
  bool tensor_payload{};
  std::uint64_t tensor_max_byte_span{};
  std::string tensor_pool_id;
  MemoryKind tensor_memory_kind{MemoryKind::kUnknown};
  DeviceId tensor_device{};
  std::vector<TensorFieldServiceStorageContract> tensor_fields;
  QueuePolicy queue_policy{QueuePolicy::kBlocking};
  std::uint32_t max_subscriber_queue_depth{};
  std::uint32_t max_subscriber_borrowed_samples{};
  std::uint32_t publisher_history_depth{};
  bool event_wake{};
  InterfaceTensorEmitPolicy tensor_emit_policy{InterfaceTensorEmitPolicy::kRequireExportable};
};

/// Optional physical Tensor facts carried only by internal service records.
struct TensorServiceStorageContract {
  std::uint64_t payload_size_bytes{};
  std::uint64_t max_byte_span{};
  std::string pool_id;
  MemoryKind memory_kind{MemoryKind::kUnknown};
  DeviceId device{};
  std::vector<TensorFieldServiceStorageContract> tensor_fields;
};

/// Build the one common service contract projected into every private service family.
// Payload identity and storage are ownership sinks transferred into the returned contract.
// NOLINTNEXTLINE(performance-unnecessary-value-param)
[[nodiscard]] inline TensorServiceContract make_tensor_service_contract(
    PayloadIdentityRecord payload_identity, QueuePolicy queue_policy,
    std::uint32_t max_subscriber_queue_depth, std::uint32_t max_subscriber_borrowed_samples,
    bool tensor_payload, TensorServiceStorageContract storage = {},
    InterfaceTensorEmitPolicy tensor_emit_policy = InterfaceTensorEmitPolicy::kRequireExportable) {
  return TensorServiceContract{
      .payload_identity = std::move(payload_identity),
      .payload_size_bytes = storage.payload_size_bytes,
      .selected_inline_bucket_bytes =
          channel_internal::inline_payload_bucket_size(storage.payload_size_bytes).value_or(0U),
      .tensor_payload = tensor_payload,
      .tensor_max_byte_span = storage.max_byte_span,
      .tensor_pool_id = std::move(storage.pool_id),
      .tensor_memory_kind = storage.memory_kind,
      .tensor_device = storage.device,
      .tensor_fields = std::move(storage.tensor_fields),
      .queue_policy = queue_policy,
      .max_subscriber_queue_depth = max_subscriber_queue_depth,
      .max_subscriber_borrowed_samples = max_subscriber_borrowed_samples,
      .publisher_history_depth = 0U,
      .event_wake = true,
      .tensor_emit_policy = tensor_emit_policy,
  };
}

/// Reconstruct the common service contract from one frozen cross-process record.
[[nodiscard]] inline TensorServiceContract make_tensor_service_contract(
    const CrossProcessChannelServicePlanRecord& service) {
  return TensorServiceContract{
      .payload_identity = service.payload_identity,
      .payload_size_bytes = service.payload_size_bytes,
      .selected_inline_bucket_bytes = service.selected_inline_bucket_bytes,
      .tensor_payload = service.tensor_payload,
      .tensor_max_byte_span = service.tensor_max_byte_span,
      .tensor_pool_id = service.tensor_pool_id,
      .tensor_memory_kind = service.tensor_memory_kind,
      .tensor_device = service.tensor_device,
      .tensor_fields = service.tensor_fields,
      .queue_policy = service.queue_policy,
      .max_subscriber_queue_depth = service.max_subscriber_queue_depth,
      .max_subscriber_borrowed_samples = service.max_subscriber_borrowed_samples,
      .publisher_history_depth = service.publisher_history_depth,
      .event_wake = service.event_wake,
      .tensor_emit_policy = service.tensor_emit_policy,
  };
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
[[nodiscard]] inline ChannelServiceStaticPlanRecord make_channel_service_record(
    ExternalTopicId topic, std::string service_id, const TensorServiceContract& contract,
    std::vector<DescriptorTensorTopicFieldRecord> tensor_fields = {}) {
  return ChannelServiceStaticPlanRecord{
      .topic = topic,
      .service_id = std::move(service_id),
      .payload_identity = contract.payload_identity,
      .tensor_fields = std::move(tensor_fields),
      .queue_policy = contract.queue_policy,
      .max_subscriber_queue_depth = contract.max_subscriber_queue_depth,
      .max_subscriber_borrowed_samples = contract.max_subscriber_borrowed_samples,
      .publisher_history_depth = contract.publisher_history_depth,
      .event_wake = contract.event_wake,
      .tensor_payload = contract.tensor_payload,
      .tensor_emit_policy = contract.tensor_emit_policy,
  };
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
[[nodiscard]] inline ProcessLocalChannelServicePlanRecord make_process_local_service_record(
    PortId source, std::string service_id, const TensorServiceContract& contract,
    std::uint32_t max_subscribers, bool cross_process_transport = false,
    ProcessLocalChannelDeliveryKind delivery_kind = ProcessLocalChannelDeliveryKind::kHoloMQ) {
  return ProcessLocalChannelServicePlanRecord{
      .source = source,
      .service_id = std::move(service_id),
      .payload_identity = contract.payload_identity,
      .payload_size_bytes = contract.payload_size_bytes,
      .selected_inline_bucket_bytes = delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ
                                          ? contract.selected_inline_bucket_bytes
                                          : 0U,
      .tensor_payload = contract.tensor_payload,
      .tensor_max_byte_span = contract.tensor_max_byte_span,
      .tensor_pool_id = contract.tensor_pool_id,
      .tensor_memory_kind = contract.tensor_memory_kind,
      .tensor_device = contract.tensor_device,
      .tensor_fields = contract.tensor_fields,
      .delivery_kind = delivery_kind,
      .cross_process_transport = cross_process_transport,
      .queue_policy = contract.queue_policy,
      .max_subscribers = max_subscribers,
      .max_subscriber_queue_depth = contract.max_subscriber_queue_depth,
      .max_subscriber_borrowed_samples = contract.max_subscriber_borrowed_samples,
      .publisher_history_depth = contract.publisher_history_depth,
      .event_wake = delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ,
  };
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
[[nodiscard]] inline CrossProcessChannelServicePlanRecord make_cross_process_service_record(
    EdgeId edge, PortId source, PortId target, std::string service_id,
    const TensorServiceContract& contract, std::uint32_t max_subscribers,
    ProcessGroupId source_process_group, ProcessGroupId target_process_group) {
  return CrossProcessChannelServicePlanRecord{
      .edge = edge,
      .source = source,
      .target = target,
      .service_id = std::move(service_id),
      .payload_identity = contract.payload_identity,
      .payload_size_bytes = contract.payload_size_bytes,
      .selected_inline_bucket_bytes = contract.selected_inline_bucket_bytes,
      .tensor_payload = contract.tensor_payload,
      .tensor_max_byte_span = contract.tensor_max_byte_span,
      .tensor_pool_id = contract.tensor_pool_id,
      .tensor_memory_kind = contract.tensor_memory_kind,
      .tensor_device = contract.tensor_device,
      .tensor_fields = contract.tensor_fields,
      .queue_policy = contract.queue_policy,
      .max_subscribers = max_subscribers,
      .max_subscriber_queue_depth = contract.max_subscriber_queue_depth,
      .max_subscriber_borrowed_samples = contract.max_subscriber_borrowed_samples,
      .publisher_history_depth = contract.publisher_history_depth,
      .event_wake = contract.event_wake,
      .tensor_emit_policy = contract.tensor_emit_policy,
      .source_process_group = source_process_group,
      .target_process_group = target_process_group,
      .target_process_groups = {target_process_group},
  };
}

/// @brief Deep-own one immutable accepted or rejected execution-plan artifact.
///
/// The state retains the frozen Graph snapshot plus normalized plan records.
/// It intentionally contains no live Runtime/backend object, scheduler thread,
/// channel endpoint, transport handle, or mutable authoring storage.
struct ExecutionPartitionStringLookupEntry {
  std::string key;              ///< Exact authored name or canonical Operator path.
  std::uint32_t table_index{};  ///< One-based index into execution_partitions.

  friend bool operator==(const ExecutionPartitionStringLookupEntry&,
                         const ExecutionPartitionStringLookupEntry&) = default;
};

struct ExecutionPlanState {
  GraphView graph;                          ///< Deep-owning canonical compiler snapshot.
  GraphAuthoringIdentity graph_identity{};  ///< Source instance/version/content identity.
  PlanDigest digest{};                      ///< Digest of normalized plan semantics.
  bool admitted{};                          ///< Total compile-admission disposition.
  std::string json;                         ///< Canonical schema-versioned plan JSON.
  std::vector<Diagnostic> diagnostics;      ///< Canonically ordered bounded diagnostics.

  ProfileManifest profile_manifest;                          ///< Total profile-admission evidence.
  PartitioningPlanRecord partitioning_plan;                  ///< Partition selector evidence.
  std::vector<PartitionDecisionRecord> partition_decisions;  ///< Per-vertex decisions.
  std::vector<ExecutionPartitionRecord> execution_partitions;  ///< Disjoint cover.
  /// Dense and sorted immutable lookup indices derived by the authoritative cover publisher.
  std::vector<std::uint32_t> execution_partition_index_by_id;
  std::vector<std::uint32_t> execution_partition_index_by_operator_id;
  std::vector<std::uint32_t> execution_partition_index_by_root_operator;
  std::vector<ExecutionPartitionStringLookupEntry> execution_partition_index_by_authored_name;
  std::vector<ExecutionPartitionStringLookupEntry> execution_partition_index_by_operator_path;
  std::vector<ScheduleEdgeRecord> schedule_edges;  ///< Cross-partition graph edges.
  std::vector<ClosedIslandActivationPlanRecord>
      closed_island_activations;  ///< Private exact composite-dispatch authority.
  std::optional<InlinePartitionExecutionPlanRecord>
      inline_partition_execution;  ///< Private exact controller-owned partition authority.
  std::vector<PartitionRunnerPlanRecord>
      partition_runners;               ///< Private immutable partition-local runner authority.
  SchedulerPlanRecord scheduler_plan;  ///< Deterministic scheduling evidence.
  std::vector<PlacementProbePlanRecord> placement_probes;          ///< Compile/runtime proofs.
  std::vector<PartitionPlacementPlanRecord> partition_placements;  ///< Accepted partition policies.
  std::vector<SupportThreadAffinityPlanRecord>
      support_thread_affinities;  ///< Canonical support-thread affinity admission evidence.
  std::vector<ExecutionLanePlanRecord> execution_lanes;  ///< Dense Runtime lane mapping.
  std::vector<ProcessRolePlanRecord> process_roles;      ///< Normalized semantic process roles.
  std::vector<ProcessSelectorPlanRecord>
      process_selectors;  ///< Canonical object-selector resolutions.
  std::vector<VertexProcessPlacementPlanRecord>
      vertex_process_placements;               ///< Total compile-admitted process groups.
  std::vector<PortPlanRecord> ports;           ///< Canonical physical port contracts.
  std::vector<PortRatePlanRecord> port_rates;  ///< Pointer-free rate declarations and domains.
  std::vector<AccumulationHorizonPlanRecord>
      accumulation_horizons;  ///< Exact finite sizing horizons.
  std::vector<HardFloorContributorPlanRecord>
      hard_floor_contributors;  ///< Reviewable hard-floor composition.
  std::vector<QueueSizingRecipePlanRecord>
      queue_sizing_recipes;  ///< Immutable logical edge-sizing recipes.
  std::vector<InterfacePortBindingPlanRecord>
      interface_port_bindings;                          ///< Plan-local interface aliases.
  std::vector<PayloadEdgePlanRecord> payload_edges;     ///< Resolved queue/cache edges.
  std::vector<ChannelLocalityRecord> channel_locality;  ///< Requested/accepted locality.
  std::vector<ChannelServiceStaticPlanRecord>
      channel_services;  ///< Symmetric service-static requirements.
  std::vector<ChannelPublisherEndpointPlanRecord>
      channel_publishers;  ///< Publisher-local endpoint requests.
  std::vector<ChannelSubscriberEndpointPlanRecord>
      channel_subscribers;  ///< Subscriber-local endpoint requests.
  std::vector<ProcessLocalChannelServicePlanRecord>
      process_local_channel_services;  ///< Internal service-static requirements.
  std::vector<ProcessLocalChannelPublisherPlanRecord>
      process_local_channel_publishers;  ///< Internal publisher-local requests.
  std::vector<ProcessLocalChannelSubscriberPlanRecord>
      process_local_channel_subscribers;         ///< Internal edge-local subscriber requests.
  bool force_process_local_holomq_event_wake{};  ///< Source-private exact A/B control.
  std::vector<CrossProcessChannelServicePlanRecord>
      cross_process_channel_services;  ///< Per-edge cross-process service contracts.
  std::vector<CrossProcessChannelPublisherPlanRecord>
      cross_process_channel_publishers;  ///< Cross-process publisher-local requests.
  std::vector<CrossProcessChannelSubscriberPlanRecord>
      cross_process_channel_subscribers;  ///< Cross-process subscriber-local requests.
  std::vector<GenericOutputLoanPlanRecord>
      generic_output_loans;  ///< Non-Tensor constructed-slot capabilities.
  std::vector<TensorAllocationPlanRecord>
      tensor_allocations;  ///< Explicit direct Tensor allocation capabilities.
  std::vector<TensorByteSpanDerivationPlanRecord>
      tensor_byte_span_derivations;  ///< Compile-evaluated Tensor slot-capacity evidence.
  std::vector<TensorMemoryKindDerivationPlanRecord>
      tensor_memory_kind_derivations;  ///< Compile-evaluated Tensor memory-kind evidence.
  std::vector<TensorFieldAllocationPlanRecord>
      tensor_field_allocations;  ///< Allocations ordered canonically by (output, field_id).
  std::vector<TensorInterfacePoolPlanRecord>
      tensor_interface_pools;  ///< External Tensor pool contracts.
  std::vector<TensorDeviceResourcePlanRecord>
      tensor_device_resources;  ///< Process/device pool and stream materialization.
  std::vector<CudaGreenContextPoolPlanRecord>
      cuda_green_context_pools;  ///< Process/device Green Context pools.
  std::vector<CudaGreenContextResourcePlanRecord>
      cuda_green_context_resources;  ///< Structured context resources.
  std::vector<CudaGreenContextRequestPlanRecord>
      cuda_green_context_requests;  ///< Accepted and rejected author requests.
  std::vector<CudaGreenContextAssignmentPlanRecord>
      cuda_green_context_assignments;  ///< Partition-to-context joins.
  std::vector<CudaComputeStreamPlanRecord>
      cuda_compute_streams;  ///< Generalized partition and service streams.
  CudaComputeStreamPolicy cuda_compute_stream_policy{
      CudaComputeStreamPolicy::kPerExecutionPartition};  ///< Frozen ordinary sharing policy.
  std::vector<OperatorCudaStreamBindingPlanRecord>
      operator_cuda_stream_bindings;  ///< Exact CUDA-active Operator stream authority.
  std::vector<TensorInlineRecordPlanRecord>
      tensor_inline_records;  ///< Generated-provider inline record buckets.
  std::vector<ImportedGenerationCapacityPlanRecord>
      imported_generation_capacities;  ///< Finite restart/ABA retention.
  std::vector<TensorTransferPlanRecord>
      tensor_transfer_plans;  ///< Closed source-class/realization matrix.
  std::vector<PoolCapacityContributorRecord>
      pool_capacity_contributors;  ///< Unique-slot derivation evidence.
  std::vector<PoolCapacityDependencyPlanRecord>
      pool_capacity_dependencies;  ///< Stable many-to-many logical-claim to pool joins.
  std::vector<ProviderAllocationPlanRecord>
      provider_allocation_plans;  ///< Immutable provider dependency-range recipes.
  std::vector<PublicationOutcomePlanRecord>
      publication_outcomes;  ///< Pre/post-submit authority contracts.
  std::vector<CUDAGraphCapturePlanRecord>
      cuda_graph_capture_plans;  ///< Validated region capture plans.
  std::vector<CUDAGraphExecutionRegionPlanRecord>
      cuda_graph_execution_regions;     ///< Physically admitted fused-region plans.
  PlanGenerationId plan_generation{};   ///< Nonzero identity for all plan-bound records.
  ClockId root_default_clock{};         ///< Root Graph default, or invalid when absent.
  std::vector<ClockPlanRecord> clocks;  ///< Dense plan-local clock catalog.
  std::vector<ClockImportResolutionPlanRecord>
      clock_import_resolutions;  ///< Exact cold-start import/binding authentication.
  std::vector<ClockConversionRecord> clock_conversions;  ///< Direct authored conversions.
  std::vector<EndpointClockHealthPlanRecord>
      endpoint_clock_health;                   ///< Immutable endpoint health contracts and bounds.
  std::vector<ReadinessPlanRecord> readiness;  ///< Logical source admission records.
  std::vector<OperatorCompletionPlanRecord>
      operator_completion_rules;  ///< Operator-owned natural completion policies.
  std::vector<TemporalTriggerPlanRecord> temporal_triggers;      ///< Total temporal triggers.
  std::vector<AlignedTriggerPlanRecord> aligned_triggers;        ///< Oldest-head aligned joins.
  std::vector<ClockTriggerPlanRecord> clock_triggers;            ///< Fixed-phase clock schedules.
  std::vector<DeadlinePlanRecord> deadlines;                     ///< Latest-start policies.
  std::vector<ReadableTriggerPlanRecord> readable_triggers;      ///< Level-readable triggers.
  std::vector<MissingAggregatePlanRecord> missing_aggregates;    ///< Phase-local precedence.
  std::vector<WindowTriggerPlanRecord> window_triggers;          ///< Window state machines.
  std::vector<BracketedWindowPlanRecord> bracketed_windows;      ///< Window retention facts.
  std::vector<InputSelectionPlanRecord> input_selections;        ///< Resolved selectors.
  std::vector<PredicateTriggerPlanRecord> predicate_triggers;    ///< Predicate semantics.
  std::vector<PredicateInputPlanRecord> predicate_inputs;        ///< Ordered evaluator inputs.
  std::vector<PredicateCallablePlanRecord> predicate_callables;  ///< Callable admission facts.

  std::vector<LifecycleCheckpointPlanRecord> lifecycle_checkpoints;   ///< Stage checkpoints.
  std::vector<LifecycleDependencyPlanRecord> lifecycle_dependencies;  ///< Same-stage DAG.
  std::vector<LifecycleRestartPermissionPlanRecord>
      lifecycle_restart_permissions;  ///< Cross-target restart authority.
  std::vector<LifecycleRestartTargetPlanRecord>
      lifecycle_restart_targets;  ///< Per-Operator restart dispositions.
  std::vector<LifecycleRestartClosurePlanRecord>
      lifecycle_restart_closures;  ///< Per-target closure and stage widening.
  std::vector<LifecycleMutableFieldPlanRecord>
      lifecycle_mutable_fields;  ///< Parameter invalidation stages.
  std::optional<LifecycleTransactionCapacityPlanRecord>
      lifecycle_transaction_capacity;  ///< Active transaction capacities when enabled.

  std::vector<OperatorHostPlanRecord> operator_hosts;  ///< Pointer-free callback-host plan facts.
  std::vector<OperatorParameterPlanRecord> operator_parameters;  ///< Generation-zero inventory.
  std::vector<ParameterFieldPlanRecord> parameter_fields;        ///< Canonical field catalog.
  ParameterCatalogView parameter_catalog;  ///< Public policy-filtered plan-bound catalog.
  std::vector<ParameterGenerationZeroOperatorRecord>
      parameter_generation_zero_operators;  ///< Protected aggregate provenance/digests.
  std::vector<ParameterGenerationZeroFieldRecord>
      parameter_generation_zero_fields;  ///< Protected field provenance/digests.
  std::vector<ParameterDerivedTermPlanRecord>
      parameter_derived_terms;  ///< Bounded temporal Parameter dependencies.
  std::optional<ParameterRuntimeCapacityPlanRecord>
      parameter_runtime_capacity;  ///< Finite Parameter storage/evidence capacities.

  // Process-local executable state is excluded from every public projection,
  // serializer, digest, and diagnostic. Runtime replays these factories against
  // `graph` before user lifecycle after the source Graph has been destroyed.
  ExecutableFactoryInventory executable_factories;

  // A private plan-owned observability binding lets blocking and asynchronous
  // entry points publish the same terminal structural sequence. It is omitted
  // from serialization and identity, never consulted by execution policy, and
  // normally absent unless the embedding runtime configures a sink.
  std::shared_ptr<RuntimeStructuralDiagnosticSink> runtime_diagnostic_sink;

  ChannelEndpointCatalog endpoint_catalog;  ///< Immutable standalone-client handoff.
};

/// One cold inspection row for a process-local HoloMQ service's event-wake proof.
struct ProcessLocalHoloMQEventWakeApplicabilityRecord {
  PortId source{};
  std::string service_id;
  QueuePolicy queue_policy{QueuePolicy::kBlocking};
  std::uint32_t planned_subscriber_count{};
  std::size_t subscriber_record_count{};
  std::size_t payload_edge_count{};
  bool event_wake{};
  ProcessLocalHoloMQEventWakeReason reason{ProcessLocalHoloMQEventWakeReason::kUnclassified};
};

/// Inspect private event-wake evidence without publishing it into Runtime bindings.
///
/// This opt-in cold helper allocates its result and scans private plan records. Compiler and
/// Runtime hot paths do not invoke it.
[[nodiscard]] inline std::vector<ProcessLocalHoloMQEventWakeApplicabilityRecord>
inspect_process_local_holomq_event_wake_applicability(const ExecutionPlanState& state) {
  std::vector<ProcessLocalHoloMQEventWakeApplicabilityRecord> result;
  result.reserve(state.process_local_channel_services.size());
  for (const auto& service : state.process_local_channel_services) {
    if (service.delivery_kind != ProcessLocalChannelDeliveryKind::kHoloMQ) {
      continue;
    }
    std::size_t subscriber_count = 0U;
    std::size_t edge_count = 0U;
    for (const auto& subscriber : state.process_local_channel_subscribers) {
      if (subscriber.service_id != service.service_id) {
        continue;
      }
      ++subscriber_count;
      const auto edge = std::find_if(
          state.payload_edges.begin(), state.payload_edges.end(), [&](const auto& candidate) {
            return candidate.edge == subscriber.edge &&
                   candidate.locality == ChannelLocalityKind::kProcessLocal &&
                   candidate.disposition == PlanDisposition::kAccepted;
          });
      if (edge != state.payload_edges.end()) {
        ++edge_count;
      }
    }
    result.push_back(ProcessLocalHoloMQEventWakeApplicabilityRecord{
        .source = service.source,
        .service_id = service.service_id,
        .queue_policy = service.queue_policy,
        .planned_subscriber_count = service.max_subscribers,
        .subscriber_record_count = subscriber_count,
        .payload_edge_count = edge_count,
        .event_wake = service.event_wake,
        .reason = service.event_wake_reason,
    });
  }
  return result;
}

/// @brief Provide compiler-only construction and inspection of plan state.
class ExecutionPlanAccess {
 public:
  /// @brief Publish one complete immutable execution-plan state.
  /// @param state Deep-owning plan state, or null for an empty plan.
  /// @return Public copyable immutable plan facade.
  // The shared state is an ownership sink and may be transferred by the compiler.
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  [[nodiscard]] static ExecutionPlan make(
      std::shared_ptr<const ExecutionPlanState> state) noexcept {
    return ExecutionPlan{std::move(state)};
  }

  /// @brief Return the private immutable state behind a public plan.
  /// @param plan Plan facade to inspect.
  /// @return State pointer, or null for an empty plan.
  [[nodiscard]] static const ExecutionPlanState* state(const ExecutionPlan& plan) noexcept {
    return plan.state_.get();
  }

  /// @brief Retain the immutable private plan state for one RunSession.
  /// @param plan Accepted plan facade to retain independently of the caller wrapper.
  /// @return Shared immutable state, or null for an empty plan.
  [[nodiscard]] static std::shared_ptr<const ExecutionPlanState> retain(
      const ExecutionPlan& plan) noexcept {
    return plan.state_;
  }
};

}  // namespace holoscan::detail
