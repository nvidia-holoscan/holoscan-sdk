/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/execution_plan.hpp>

#include "../channel/endpoint_catalog_state.hpp"
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
    const bool dtype_equal = left.dtype.has_value() == right.dtype.has_value() &&
                             (!left.dtype || (left.dtype->code == right.dtype->code &&
                                              left.dtype->bits == right.dtype->bits &&
                                              left.dtype->lanes == right.dtype->lanes));
    return left.field_id == right.field_id && left.tensor_index == right.tensor_index &&
           left.max_byte_span == right.max_byte_span && left.pool_id == right.pool_id &&
           dtype_equal && left.rank == right.rank && left.memory_kind == right.memory_kind &&
           left.device == right.device && left.tensor_emit_policy == right.tensor_emit_policy;
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

/// Freeze one callback owner within a partition-local execution runner.
///
/// The record joins the canonical Graph vertex to the reconstructed Operator
/// identity and its already admitted temporal source. Vector order in the
/// enclosing runner record is the stable topological callback order.
struct PartitionRunnerOperatorPlanRecord {
  VertexId vertex{};
  OperatorInstanceId operator_id{};
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
                  direct_descriptor_tensor_schema)
             ? ProcessLocalChannelDeliveryKind::kDirectRetained
             : ProcessLocalChannelDeliveryKind::kHoloMQ;
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
  std::uint64_t payload_size_bytes{};
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
  std::uint64_t payload_size_bytes{};
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

[[nodiscard]] inline bool descriptor_tensor_schema_service(
    const CrossProcessChannelServicePlanRecord& service) noexcept {
  return service.tensor_payload && service.payload_identity.kind == PayloadIdentityKind::kSchema &&
         service.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
}

[[nodiscard]] inline bool cross_process_tensor_service_requires_stream_in_group(
    const CrossProcessChannelServicePlanRecord& service, ProcessGroupId process_group) noexcept {
  const bool target_group = std::find(service.target_process_groups.begin(),
                                      service.target_process_groups.end(),
                                      process_group) != service.target_process_groups.end();
  const bool participating_group = target_group || (descriptor_tensor_schema_service(service) &&
                                                    service.source_process_group == process_group);
  if (!service.tensor_payload || !participating_group) {
    return false;
  }
  if (!service.tensor_fields.empty()) {
    return std::any_of(
        service.tensor_fields.begin(), service.tensor_fields.end(), [](const auto& field) {
          return (field.memory_kind == MemoryKind::kCudaDevice ||
                  field.memory_kind == MemoryKind::kPinnedHost) &&
                 field.device.valid();
        });
  }
  const bool stream_ordered = service.tensor_memory_kind == MemoryKind::kCudaDevice ||
                              service.tensor_memory_kind == MemoryKind::kPinnedHost;
  return stream_ordered && service.tensor_device.valid();
}

[[nodiscard]] inline bool cross_process_tensor_service_requires_stream_for_device_in_group(
    const CrossProcessChannelServicePlanRecord& service, ProcessGroupId process_group,
    DeviceId device) noexcept {
  if (!device.valid() ||
      !cross_process_tensor_service_requires_stream_in_group(service, process_group)) {
    return false;
  }
  if (!service.tensor_fields.empty()) {
    return std::any_of(
        service.tensor_fields.begin(), service.tensor_fields.end(), [&](const auto& field) {
          return (field.memory_kind == MemoryKind::kCudaDevice ||
                  field.memory_kind == MemoryKind::kPinnedHost) &&
                 field.device == device;
        });
  }
  return service.tensor_device == device;
}

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
[[nodiscard]] inline TensorServiceContract make_tensor_service_contract(
    PayloadIdentityRecord payload_identity, QueuePolicy queue_policy,
    std::uint32_t max_subscriber_queue_depth, std::uint32_t max_subscriber_borrowed_samples,
    bool tensor_payload, TensorServiceStorageContract storage = {},
    InterfaceTensorEmitPolicy tensor_emit_policy = InterfaceTensorEmitPolicy::kRequireExportable) {
  return TensorServiceContract{
      .payload_identity = std::move(payload_identity),
      .payload_size_bytes = storage.payload_size_bytes,
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

[[nodiscard]] inline ChannelServiceStaticPlanRecord make_channel_service_record(
    ExternalTopicId topic, std::string service_id, const TensorServiceContract& contract) {
  return ChannelServiceStaticPlanRecord{
      .topic = topic,
      .service_id = std::move(service_id),
      .payload_identity = contract.payload_identity,
      .queue_policy = contract.queue_policy,
      .max_subscriber_queue_depth = contract.max_subscriber_queue_depth,
      .max_subscriber_borrowed_samples = contract.max_subscriber_borrowed_samples,
      .publisher_history_depth = contract.publisher_history_depth,
      .event_wake = contract.event_wake,
      .tensor_payload = contract.tensor_payload,
      .tensor_emit_policy = contract.tensor_emit_policy,
  };
}

[[nodiscard]] inline ProcessLocalChannelServicePlanRecord make_process_local_service_record(
    PortId source, std::string service_id, const TensorServiceContract& contract,
    std::uint32_t max_subscribers, bool cross_process_transport = false,
    ProcessLocalChannelDeliveryKind delivery_kind = ProcessLocalChannelDeliveryKind::kHoloMQ) {
  return ProcessLocalChannelServicePlanRecord{
      .source = source,
      .service_id = std::move(service_id),
      .payload_identity = contract.payload_identity,
      .payload_size_bytes = contract.payload_size_bytes,
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
  std::vector<ScheduleEdgeRecord> schedule_edges;              ///< Cross-partition graph edges.
  std::vector<ClosedIslandActivationPlanRecord>
      closed_island_activations;  ///< Private exact composite-dispatch authority.
  std::optional<InlinePartitionExecutionPlanRecord>
      inline_partition_execution;  ///< Private exact controller-owned partition authority.
  std::vector<PartitionRunnerPlanRecord>
      partition_runners;               ///< Private immutable partition-local runner authority.
  SchedulerPlanRecord scheduler_plan;  ///< Deterministic scheduling evidence.
  std::vector<PlacementProbePlanRecord> placement_probes;          ///< Compile/runtime proofs.
  std::vector<PartitionPlacementPlanRecord> partition_placements;  ///< Accepted partition policies.
  std::vector<ExecutionLanePlanRecord> execution_lanes;            ///< Dense Runtime lane mapping.
  std::vector<ProcessRolePlanRecord> process_roles;  ///< Normalized semantic process roles.
  std::vector<ProcessSelectorPlanRecord>
      process_selectors;  ///< Canonical object-selector resolutions.
  std::vector<VertexProcessPlacementPlanRecord>
      vertex_process_placements;      ///< Total compile-admitted process groups.
  std::vector<PortPlanRecord> ports;  ///< Canonical physical port contracts.
  std::vector<SchemaPackagePlanRecord>
      schema_packages;  ///< Deep-owned generated-package diagnostic provenance.
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
      process_local_channel_subscribers;  ///< Internal edge-local subscriber requests.
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
  std::vector<TensorFieldAllocationPlanRecord>
      tensor_field_allocations;  ///< Per-field descriptor-schema allocations.
  std::vector<TensorInterfacePoolPlanRecord>
      tensor_interface_pools;  ///< External Tensor pool contracts.
  std::vector<TensorDeviceResourcePlanRecord>
      tensor_device_resources;  ///< Process/device pool and stream materialization.
  std::vector<TensorInlineRecordPlanRecord>
      tensor_inline_records;  ///< Generated-provider inline record buckets.
  std::vector<ImportedGenerationCapacityPlanRecord>
      imported_generation_capacities;  ///< Finite restart/ABA retention.
  std::vector<TensorTransferPlanRecord>
      tensor_transfer_plans;  ///< Closed source-class/realization matrix.
  std::vector<PoolCapacityContributorRecord>
      pool_capacity_contributors;  ///< Unique-slot derivation evidence.
  std::vector<PublicationOutcomePlanRecord>
      publication_outcomes;             ///< Pre/post-submit authority contracts.
  PlanGenerationId plan_generation{};   ///< Nonzero identity for all plan-bound records.
  ClockId root_default_clock{};         ///< Root Graph default, or invalid when absent.
  std::vector<ClockPlanRecord> clocks;  ///< Dense plan-local clock catalog.
  std::vector<ClockImportResolutionPlanRecord>
      clock_import_resolutions;  ///< Exact cold-start import/binding authentication.
  std::vector<ClockConversionRecord> clock_conversions;  ///< Direct authored conversions.
  std::vector<EndpointClockHealthPlanRecord>
      endpoint_clock_health;                   ///< Immutable endpoint health contracts and bounds.
  std::vector<ReadinessPlanRecord> readiness;  ///< Logical source admission records.
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

/// @brief Provide compiler-only construction and inspection of plan state.
class ExecutionPlanAccess {
 public:
  /// @brief Publish one complete immutable execution-plan state.
  /// @param state Deep-owning plan state, or null for an empty plan.
  /// @return Public copyable immutable plan facade.
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
