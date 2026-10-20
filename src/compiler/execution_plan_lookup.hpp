/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>
#include <string_view>

#include "execution_plan_state.hpp"

namespace holoscan::detail {

/// Find the final execution partition containing an exact canonical vertex.
///
/// Accepted final plans contain a disjoint partition cover and store each partition's members in
/// ascending vertex order. This policy-free lookup assumes those invariants have already been
/// validated and does not diagnose duplicate membership in corrupted input. Unlike partition IDs,
/// vertex membership has no dense vector-position projection, so this lookup searches partition
/// records and then uses the canonical member ordering within each record.
///
/// @return A non-owning pointer to the containing partition, or \c nullptr when absent.
[[nodiscard]] const ExecutionPartitionRecord* find_execution_partition_by_member(
    const ExecutionPlanState& plan, VertexId member) noexcept;

/// Find the only execution partition containing an exact canonical vertex.
///
/// @return The containing partition, or \c nullptr when membership is absent or duplicated.
[[nodiscard]] const ExecutionPartitionRecord* find_unique_execution_partition_by_member(
    const ExecutionPlanState& plan, VertexId member) noexcept;

/// Find the only CUDA Graph capture plan with the complete graph and region key.
///
/// @return The matching plan, or \c nullptr when the key is absent or duplicated.
[[nodiscard]] const CUDAGraphCapturePlanRecord* find_unique_cuda_graph_capture_plan_by_region(
    const ExecutionPlanState& plan, GraphId graph, RegionId region) noexcept;

/// Return whether an admitted CUDA Graph region in one process group captures an output's address.
[[nodiscard]] bool cuda_graph_uses_fixed_output_address(const ExecutionPlanState& plan,
                                                        ProcessGroupId process_group,
                                                        PortId output) noexcept;

/// Return the CUDA device that consumes or produces a fixed CUDA Graph output address in one
/// process group. Conflicting region devices return an invalid device.
[[nodiscard]] DeviceId cuda_graph_fixed_output_device(const ExecutionPlanState& plan,
                                                      ProcessGroupId process_group,
                                                      PortId output) noexcept;

/// Find the only admitted CUDA Graph execution region containing an Operator vertex.
///
/// Membership is resolved through the region's capture plan. Missing or duplicate plan records
/// return \c nullptr so callers do not select an arbitrary fused region from malformed state.
[[nodiscard]] const CUDAGraphExecutionRegionPlanRecord*
find_unique_cuda_graph_execution_region_by_member(const ExecutionPlanState& plan,
                                                  VertexId member) noexcept;

/// Return whether an edge executes inside an admitted fused CUDA Graph region.
///
/// A missing or ambiguous capture plan is not treated as an admitted internal edge.
[[nodiscard]] bool is_cuda_graph_internal_execution_edge(const ExecutionPlanState& plan,
                                                         EdgeId edge) noexcept;

/// Find a final execution partition by its exact dense partition ID.
///
/// Accepted final plans store partition ID \c N at vector position \c N-1. Duplicate and ordering
/// validation remains the validator's responsibility; this lookup deliberately uses that dense
/// final-plan invariant for constant-time consumption.
///
/// @return A non-owning pointer to the dense matching record, or \c nullptr when the ID is zero,
/// out of range, or does not match its canonical vector position.
[[nodiscard]] const ExecutionPartitionRecord* find_execution_partition_by_id(
    const ExecutionPlanState& plan, ExecutionPartitionId id) noexcept;

/// Find final process placement by its exact canonical vertex key.
///
/// The accepted final plan must already have been validated to contain exactly one placement for
/// each executable vertex. This policy-free lookup does not diagnose duplicates.
///
/// @return A non-owning pointer to the matching placement, or \c nullptr when absent.
[[nodiscard]] const VertexProcessPlacementPlanRecord* find_vertex_process_placement_by_vertex(
    const ExecutionPlanState& plan, VertexId vertex) noexcept;

/// Find the only process-placement record for an exact canonical vertex.
///
/// @return The matching placement, or \c nullptr when placement is absent or duplicated.
[[nodiscard]] const VertexProcessPlacementPlanRecord*
find_unique_vertex_process_placement_by_vertex(const ExecutionPlanState& plan,
                                               VertexId vertex) noexcept;

/// Find a final canonical Graph vertex by its globally unique plan-local ID.
///
/// The accepted final plan must already have been validated to contain at most one vertex for the
/// ID. This lookup preserves the canonical Graph record rather than projecting placement state.
///
/// @return A non-owning pointer to the matching vertex, or \c nullptr when absent.
[[nodiscard]] const VertexRecord* find_vertex_by_id(const ExecutionPlanState& plan,
                                                    VertexId id) noexcept;

/// Find a final port record by its globally unique plan-local port ID.
///
/// The accepted final plan must already have been validated to contain at most one record for the
/// ID. Direction and ownership remain caller-visible contract checks rather than part of the key.
///
/// @return A non-owning pointer to the matching port, or \c nullptr when absent.
[[nodiscard]] const PortPlanRecord* find_port_by_id(const ExecutionPlanState& plan,
                                                    PortId id) noexcept;

/// Find a final port by ID and verify its declared direction.
///
/// This is a contract-checking convenience over \ref find_port_by_id; direction is not a second
/// identity key because port IDs are globally unique within the final plan.
[[nodiscard]] const PortPlanRecord* find_port_by_id_and_direction(const ExecutionPlanState& plan,
                                                                  PortId id,
                                                                  PortDirection direction) noexcept;

/// Find generation-zero Operator parameters by the complete plan-bound owner key.
///
/// Accepted final plans contain exactly one record per Operator. This lookup deliberately does not
/// weaken the key to an instance ID or infer the enclosing plan/run generations.
[[nodiscard]] const OperatorParameterPlanRecord* find_operator_parameters_by_owner(
    const ExecutionPlanState& plan, OperatorRef owner) noexcept;

/// Return the contiguous parameter-field catalog for one complete plan-bound Operator owner.
///
/// Accepted final plans keep each owner's records contiguous. Validation remains responsible for
/// rejecting duplicate or interleaved owner groups before this accessor is used.
///
/// @return A borrowed span preserving canonical field order, or an empty span when absent.
[[nodiscard]] std::span<const ParameterFieldPlanRecord> parameter_fields_by_owner(
    const ExecutionPlanState& plan, OperatorRef owner) noexcept;

/// Return the contiguous parameter-derived terms for one complete plan-bound Operator owner.
///
/// Accepted final plans keep each owner's records contiguous. Validation remains responsible for
/// rejecting duplicate or interleaved owner groups before this accessor is used.
///
/// @return A borrowed span preserving canonical term order, or an empty span when absent.
[[nodiscard]] std::span<const ParameterDerivedTermPlanRecord> parameter_derived_terms_by_owner(
    const ExecutionPlanState& plan, OperatorRef owner) noexcept;

/// Find an input-selection contract by its globally unique input port ID.
[[nodiscard]] const InputSelectionPlanRecord* find_input_selection_by_input(
    const ExecutionPlanState& plan, PortId input) noexcept;

/// Find a clock conversion by its exact nonzero plan-local conversion ID.
[[nodiscard]] const ClockConversionRecord* find_clock_conversion_by_id(
    const ExecutionPlanState& plan, std::uint64_t id) noexcept;

/// Find a plan clock by its exact nonzero plan-local clock ID.
[[nodiscard]] const ClockPlanRecord* find_clock_by_id(const ExecutionPlanState& plan,
                                                      ClockId id) noexcept;

/// Find a singular raw-Tensor interface pool by its exact canonical topic ID.
///
/// Descriptor-backed schema topics may own several field/index-specific pools and must use
/// \ref find_tensor_interface_pool_by_topic_field instead. Validation remains responsible for
/// diagnosing duplicate or conflicting pool records before this accessor is used.
[[nodiscard]] const TensorInterfacePoolPlanRecord* find_tensor_interface_pool_by_topic(
    const ExecutionPlanState& plan, ExternalTopicId topic) noexcept;

/// Find one descriptor Tensor interface pool by its complete topic/field/index identity.
[[nodiscard]] const TensorInterfacePoolPlanRecord* find_tensor_interface_pool_by_topic_field(
    const ExecutionPlanState& plan, ExternalTopicId topic, std::uint16_t field_id,
    std::uint16_t tensor_index) noexcept;

/// Find one direct Tensor allocation by its exact output port key.
[[nodiscard]] const TensorAllocationPlanRecord* find_tensor_allocation_by_output(
    const ExecutionPlanState& plan, PortId output) noexcept;

/// Return the contiguous, canonically ordered Tensor-field allocations for one output.
///
/// Accepted final plans validate this vector's ascending @c (output,field_id) order before
/// publication. This preserves the one-to-many relation instead of hiding it behind a first-match
/// pointer.
[[nodiscard]] std::span<const TensorFieldAllocationPlanRecord> tensor_field_allocations_by_output(
    const ExecutionPlanState& plan, PortId output) noexcept;

/// Find the aggregate Tensor resource for one exact process/device pair.
///
/// Accepted records are unique by the complete pair. This lookup does not infer device zero or
/// diagnose duplicate resources in unvalidated input.
[[nodiscard]] const TensorDeviceResourcePlanRecord* find_tensor_device_resource(
    const ExecutionPlanState& plan, ProcessGroupId process_group, DeviceId device) noexcept;

/// Find a partition-owned CUDA compute stream by its complete process/device/partition key.
[[nodiscard]] const CudaComputeStreamPlanRecord* find_cuda_compute_stream_by_partition(
    const ExecutionPlanState& plan, ProcessGroupId process_group, DeviceId device,
    ExecutionPartitionId partition) noexcept;

/// Find the CUDA compute stream that canonically owns one fused edge on a process/device.
///
/// Final validation proves that a fused edge is assigned at most once for the selected
/// process/device. This lookup does not diagnose duplicate ownership in corrupted input.
[[nodiscard]] const CudaComputeStreamPlanRecord* find_cuda_compute_stream_by_fused_edge(
    const ExecutionPlanState& plan, ProcessGroupId process_group, DeviceId device,
    EdgeId edge) noexcept;

/// Return whether any CUDA compute stream exists for an exact process/device pair.
///
/// A process/device may own several partition or service streams. This query preserves that
/// one-to-many relation instead of returning an arbitrary first match.
[[nodiscard]] bool has_cuda_compute_stream_for_device(const ExecutionPlanState& plan,
                                                      ProcessGroupId process_group,
                                                      DeviceId device) noexcept;

/// Find a process-local Channel service by its exact stable service ID.
[[nodiscard]] const ProcessLocalChannelServicePlanRecord* find_process_local_channel_service_by_id(
    const ExecutionPlanState& plan, std::string_view service_id) noexcept;

/// Find the unique process-local publisher request for a service ID.
[[nodiscard]] const ProcessLocalChannelPublisherPlanRecord*
find_process_local_channel_publisher_by_service_id(const ExecutionPlanState& plan,
                                                   std::string_view service_id) noexcept;

/// Find a cross-process Channel service by its exact stable service ID.
[[nodiscard]] const CrossProcessChannelServicePlanRecord* find_cross_process_channel_service_by_id(
    const ExecutionPlanState& plan, std::string_view service_id) noexcept;

/// Find the external Channel service contract for one canonical topic.
[[nodiscard]] const ChannelServiceStaticPlanRecord* find_external_channel_service_by_topic(
    const ExecutionPlanState& plan, ExternalTopicId topic) noexcept;

/// Find the external Channel publisher contract for one canonical topic.
[[nodiscard]] const ChannelPublisherEndpointPlanRecord* find_external_channel_publisher_by_topic(
    const ExecutionPlanState& plan, ExternalTopicId topic) noexcept;

/// Find the external Channel subscriber contract for one canonical topic.
[[nodiscard]] const ChannelSubscriberEndpointPlanRecord* find_external_channel_subscriber_by_topic(
    const ExecutionPlanState& plan, ExternalTopicId topic) noexcept;

/// Find readiness by the exact \c ReadinessPlanRecord::owner vertex key.
///
/// The accepted final plan must already have been validated to contain at most one readiness
/// record for the owner. This policy-free lookup does not diagnose duplicates and returns the
/// first insertion-order match for unvalidated input.
///
/// @return A non-owning pointer to the matching record, or \c nullptr when the owner is absent.
[[nodiscard]] const ReadinessPlanRecord* find_readiness_by_owner(const ExecutionPlanState& plan,
                                                                 VertexId owner) noexcept;

/// Find a temporal trigger by the exact \c TemporalTriggerPlanRecord::owner.id key.
///
/// The accepted final plan must already have been validated to contain at most one temporal
/// trigger for the Operator instance. This policy-free lookup does not diagnose duplicates and
/// returns the first insertion-order match for unvalidated input.
///
/// @return A non-owning pointer to the matching record, or \c nullptr when the owner is absent.
[[nodiscard]] const TemporalTriggerPlanRecord* find_temporal_trigger_by_owner(
    const ExecutionPlanState& plan, OperatorInstanceId owner) noexcept;

/// Find exactly one intermediate temporal trigger by Operator instance ID.
///
/// Unlike the final-plan lookup above, this compiler-side operation fails closed when duplicate
/// records still exist during derivation. It returns c nullptr for either absence or ambiguity.
[[nodiscard]] const TemporalTriggerPlanRecord* find_unique_temporal_trigger_by_owner(
    const ExecutionPlanState& plan, OperatorInstanceId owner) noexcept;

/// Find a clock trigger by its complete \c ClockTriggerPlanRecord::owner key.
///
/// The accepted final plan must already have been validated to contain at most one clock trigger
/// for the complete plan/run-qualified Operator reference. This policy-free lookup does not
/// diagnose duplicates and returns the first insertion-order match for unvalidated input.
///
/// @return A non-owning pointer to the matching record, or \c nullptr when the owner is absent.
[[nodiscard]] const ClockTriggerPlanRecord* find_clock_trigger_by_owner(
    const ExecutionPlanState& plan, OperatorRef owner) noexcept;

/// Find a predicate trigger by its complete \c PredicateTriggerPlanRecord::owner key.
///
/// The accepted final plan must already have been validated to contain at most one predicate
/// trigger for the complete plan/run-qualified Operator reference. This policy-free lookup does
/// not diagnose duplicates and returns the first insertion-order match for unvalidated input.
///
/// @return A non-owning pointer to the matching record, or \c nullptr when the owner is absent.
[[nodiscard]] const PredicateTriggerPlanRecord* find_predicate_trigger_by_owner(
    const ExecutionPlanState& plan, OperatorRef owner) noexcept;

/// Find exactly one intermediate predicate trigger by Operator instance ID.
///
/// Intermediate derivation rejects duplicate owner records instead of choosing an insertion-order
/// match. It returns c nullptr for either absence or ambiguity.
[[nodiscard]] const PredicateTriggerPlanRecord* find_unique_predicate_trigger_by_owner(
    const ExecutionPlanState& plan, OperatorInstanceId owner) noexcept;

/// Find a predicate callable by its exact \c PredicateCallablePlanRecord::predicate_id key.
///
/// The accepted final plan must already have been validated to contain at most one callable for
/// the predicate ID. This policy-free lookup does not diagnose duplicates and returns the first
/// insertion-order match for unvalidated input. The borrowed ID is not retained.
///
/// @return A non-owning pointer to the matching record, or \c nullptr when the ID is absent.
[[nodiscard]] const PredicateCallablePlanRecord* find_predicate_callable_by_id(
    const ExecutionPlanState& plan, std::string_view predicate_id) noexcept;

}  // namespace holoscan::detail
