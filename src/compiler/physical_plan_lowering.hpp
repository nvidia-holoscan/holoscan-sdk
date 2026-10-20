/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/execution_plan.hpp>

#include "channel/endpoint_catalog_state.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/plan_lowering_diagnostic.hpp"
#include "core/graph_snapshot.hpp"
#include "process_plan_records.hpp"
#include "rate_plan_lowering.hpp"
#include "runtime_package_capabilities.hpp"

namespace holoscan::detail {

/// @brief Physical-plan spelling of the common lowering diagnostic value.
using PhysicalPlanDiagnostic = PlanLoweringDiagnostic;

/// @brief Own canonical physical topology, capacity, and publication records.
struct PhysicalPlanLoweringResult {
  std::vector<PortPlanRecord> ports;
  std::vector<InterfacePortBindingPlanRecord> interface_bindings;
  std::vector<PayloadEdgePlanRecord> payload_edges;
  std::vector<ChannelLocalityRecord> channel_locality;
  std::vector<ChannelServiceStaticPlanRecord> channel_services;
  std::vector<ChannelPublisherEndpointPlanRecord> channel_publishers;
  std::vector<ChannelSubscriberEndpointPlanRecord> channel_subscribers;
  std::vector<ProcessLocalChannelServicePlanRecord> process_local_channel_services;
  std::vector<ProcessLocalChannelPublisherPlanRecord> process_local_channel_publishers;
  std::vector<ProcessLocalChannelSubscriberPlanRecord> process_local_channel_subscribers;
  std::vector<CrossProcessChannelServicePlanRecord> cross_process_channel_services;
  std::vector<CrossProcessChannelPublisherPlanRecord> cross_process_channel_publishers;
  std::vector<CrossProcessChannelSubscriberPlanRecord> cross_process_channel_subscribers;
  std::vector<GenericOutputLoanPlanRecord> generic_output_loans;
  std::vector<TensorAllocationPlanRecord> tensor_allocations;
  std::vector<TensorFieldAllocationPlanRecord> tensor_field_allocations;
  std::vector<TensorInterfacePoolPlanRecord> tensor_interface_pools;
  std::vector<TensorDeviceResourcePlanRecord> tensor_device_resources;
  std::vector<CudaComputeStreamPlanRecord> cuda_compute_streams;
  std::vector<OperatorCudaStreamBindingPlanRecord> operator_cuda_stream_bindings;
  std::vector<TensorInlineRecordPlanRecord> tensor_inline_records;
  std::vector<ImportedGenerationCapacityPlanRecord> imported_generation_capacities;
  std::vector<TensorTransferPlanRecord> tensor_transfer_plans;
  std::vector<PoolCapacityContributorRecord> pool_capacity_contributors;
  std::vector<PoolCapacityDependencyPlanRecord> pool_capacity_dependencies;
  std::vector<ProviderAllocationPlanRecord> provider_allocation_plans;
  std::vector<PublicationOutcomePlanRecord> publication_outcomes;
  std::vector<PhysicalPlanDiagnostic> errors;
};

/// @brief Record the explicit Operator device evidence considered for one unbound connected
/// pass-through CUDA output.
struct TensorDeviceInheritanceRecord {
  PortId port{};  ///< Output that inherited placement, or was left unplaced.
  /// Evidence sorted by source: `output:<port>` or `green_context_partition:<name>`.
  std::vector<std::pair<std::string, DeviceId>> evidence;
  bool inherited{};  ///< Every source named one device, which the output inherited.
};

/// @brief Lower canonical Graph topology into bounded Channel and pool contracts.
/// @param snapshot Canonical pointer-free Graph snapshot.
/// @param plan_generation Nonzero plan generation copied into plan-bound capability records.
/// @param effective_profiles Vertex-indexed effective profile results.
/// @param package_capabilities Exact generated package/provider evidence.
/// @param input_selections Final temporal selection/cache bounds indexed by plan-local PortId.
/// @param bracketed_windows Final bracketed-window overlap and cache requirements.
/// @param catalog_endpoints Canonical endpoint records that own service identity.
/// @param process_placements Total vertex-to-process-group mapping.
/// @param lifecycle_targets Final lifecycle restart dispositions used for generation capacity.
/// @param lifecycle_closures Final restart closures; their members are restartable with their
/// admitted target.
/// @param execution_partitions Final accepted exact vertex cover used to validate resource
/// ownership and, under the partition stream policy, select the compute-stream owner.
/// @param tensor_device_bindings Explicit device evidence for device-visible Tensor ports.
/// @param tensor_device_inheritance Evidence behind inherited or unplaced pass-through outputs,
/// used to explain unresolved and conflicting device diagnostics.
/// @return Total inspectable physical records plus stable fail-closed diagnostics.
/// @note This cold pass creates no endpoint, queue, pool, CUDA event, thread, or backend object.
[[nodiscard]] PhysicalPlanLoweringResult lower_physical_plan(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const std::optional<Profile>> effective_profiles,
    const RuntimePackageCapabilityView& package_capabilities,
    std::span<const InputSelectionPlanRecord> input_selections,
    std::span<const BracketedWindowPlanRecord> bracketed_windows,
    std::span<const ChannelEndpointRecord> catalog_endpoints = {},
    std::span<const VertexProcessPlacementPlanRecord> process_placements = {},
    std::span<const LifecycleRestartTargetPlanRecord> lifecycle_targets = {},
    std::span<const ExecutionPartitionRecord> execution_partitions = {},
    std::span<const TensorDeviceBindingRecord> tensor_device_bindings = {},
    bool force_holomq_process_local_inline = false,
    bool force_holomq_process_local_event_wake = false,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations = {},
    std::span<const ResolvedQueueSizing> resolved_queue_sizings = {},
    std::span<const QueueSizingRecipePlanRecord> queue_sizing_recipes = {},
    CudaComputeStreamPolicy stream_policy = CudaComputeStreamPolicy::kPerExecutionPartition,
    bool experimental_split_cut = false,
    ProcessLocalEmitAdmissionScope emit_admission = ProcessLocalEmitAdmissionScope::kNone,
    bool force_holomq_process_local_fanout = false,
    std::span<const LifecycleRestartClosurePlanRecord> lifecycle_closures = {},
    std::span<const TensorDeviceInheritanceRecord> tensor_device_inheritance = {});

}  // namespace holoscan::detail
