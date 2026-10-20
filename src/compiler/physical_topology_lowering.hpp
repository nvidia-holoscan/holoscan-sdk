/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>

#include "channel/endpoint_catalog_state.hpp"
#include "compiler/physical_plan_lowering.hpp"

namespace holoscan::detail {

struct ResolvedQueueSizing;

/// @brief Own topology indexes, dense capacity facts, and binding-consumption evidence.
struct PhysicalTopologyFacts {
  std::unordered_map<std::uint64_t, const ExternalTopicRecord*> imported_by_port;
  std::unordered_map<std::uint64_t, const ExternalTopicRecord*> exported_by_port;
  std::unordered_map<std::uint64_t, const ChannelEndpointRecord*> endpoint_by_port;
  std::unordered_map<std::uint64_t, DeviceId> device_by_port;
  /// Evidence behind each inherited or unplaced pass-through output, for diagnostics.
  std::unordered_map<std::uint64_t, TensorDeviceInheritanceRecord> device_inheritance_by_port;
  std::unordered_set<std::uint64_t> consumed_device_bindings;
  std::vector<std::uint32_t> delivery_by_output;
  std::vector<std::uint32_t> transport_retention_by_output;
  std::unordered_set<std::uint64_t> transport_retention_edges;
  /// Process-local HoloMQ edges whose subscriber stages Tensor arrivals; a subset of the above.
  std::unordered_set<std::uint64_t> ingress_staging_edges;
  /// Single-slot holders per output that no queue depth sizes: one endpoint head per transported
  /// subscriber and one reservation slot per subscriber that stages. Included in
  /// transport_retention_by_output; a recipe-sized pool budgets them as a fixed contributor.
  std::vector<std::uint32_t> fixed_transport_by_output;
  /// Private depth-one ReuseLast fallback slots per output. Included in
  /// cache_by_output; a recipe-sized pool budgets them as a fixed contributor.
  std::vector<std::uint32_t> fixed_retention_by_output;
  std::vector<std::uint32_t> cache_by_output;
  std::vector<std::uint32_t> window_by_output;
  std::vector<bool> reuse_by_output;
  std::vector<bool> drain_by_output;
  std::vector<bool> copy_by_output;
  std::vector<bool> reexport_by_output;
  std::set<std::pair<std::uint64_t, std::uint64_t>> holomq_activation_pairs;
  /// Source output ports whose whole drop-oldest fanout family uses direct retention.
  std::unordered_set<std::uint64_t> direct_fanout_sources;
};

/// @brief Resolve the device for one Tensor storage obligation and consume explicit evidence.
[[nodiscard]] std::optional<DeviceId> resolve_tensor_device(PhysicalPlanLoweringResult& result,
                                                            const GraphSnapshotData& snapshot,
                                                            PhysicalTopologyFacts& topology,
                                                            PortId port, MemoryKind memory,
                                                            DiagnosticSource source,
                                                            bool force_explicit_device = false);

/// @brief Add placement for connected pass-through CUDA outputs from their owner's unique
/// explicit output or authored Green Context device. Never propagate input placement.
/// @return The evidence considered for each such output, whether inherited or left unplaced.
[[nodiscard]] std::vector<TensorDeviceInheritanceRecord> inherit_pass_through_tensor_devices(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    std::vector<TensorDeviceBindingRecord>& bindings);

/// @brief Return the effective process-local delivery kind for an Operator pair.
[[nodiscard]] ProcessLocalChannelDeliveryKind physical_process_local_delivery_kind(
    const std::set<std::pair<std::uint64_t, std::uint64_t>>& holomq_activation_pairs,
    const std::unordered_set<std::uint64_t>& direct_fanout_sources,
    const PayloadEdgePlanRecord& edge, const PortRecord& source, const PortRecord& target,
    bool force_holomq_process_local_inline) noexcept;

/// @brief Lower ports, bindings, edges, locality, and dense per-output capacity facts.
[[nodiscard]] PhysicalTopologyFacts lower_physical_topology(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    std::span<const std::optional<Profile>> effective_profiles,
    std::span<const InputSelectionPlanRecord> input_selections,
    std::span<const BracketedWindowPlanRecord> bracketed_windows,
    std::span<const ChannelEndpointRecord> catalog_endpoints,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    std::span<const TensorDeviceBindingRecord> tensor_device_bindings,
    bool force_holomq_process_local_inline, bool force_holomq_process_local_event_wake,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations,
    std::span<const ResolvedQueueSizing> resolved_queue_sizings,
    bool force_holomq_process_local_fanout = false,
    std::span<const LifecycleRestartTargetPlanRecord> lifecycle_targets = {},
    std::span<const LifecycleRestartClosurePlanRecord> lifecycle_closures = {},
    std::span<const TensorDeviceInheritanceRecord> tensor_device_inheritance = {});

}  // namespace holoscan::detail
