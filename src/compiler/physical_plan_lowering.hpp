/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/core/execution_plan.hpp>

#include "channel/endpoint_catalog_state.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/graph_snapshot.hpp"
#include "process_plan_records.hpp"
#include "runtime_package_capabilities.hpp"

namespace holoscan::detail {

/// @brief Preserve one stable physical-plan diagnostic and its canonical source.
struct PhysicalPlanDiagnostic {
  std::string token;         ///< Stable diagnostic token registered by Core.
  DiagnosticSource source;   ///< Canonical graph/vertex/port source identity.
  std::string json_pointer;  ///< Canonical location of the rejected physical fact.
  std::string required;      ///< Bounded accepted-side evidence.
  std::string provided;      ///< Bounded rejected-side evidence.
};

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
  std::vector<TensorInlineRecordPlanRecord> tensor_inline_records;
  std::vector<ImportedGenerationCapacityPlanRecord> imported_generation_capacities;
  std::vector<TensorTransferPlanRecord> tensor_transfer_plans;
  std::vector<PoolCapacityContributorRecord> pool_capacity_contributors;
  std::vector<PublicationOutcomePlanRecord> publication_outcomes;
  std::vector<PhysicalPlanDiagnostic> errors;
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
/// @param execution_partitions Final accepted exact vertex cover used only to
/// validate resource ownership, never to multiply physical resources.
/// @param tensor_device_bindings Explicit device evidence for device-visible Tensor ports.
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
    bool force_holomq_process_local_inline = false);

}  // namespace holoscan::detail
