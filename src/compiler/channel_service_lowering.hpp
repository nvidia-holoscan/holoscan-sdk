/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <span>
#include <unordered_set>

#include "compiler/physical_plan_lowering.hpp"
#include "compiler/physical_topology_lowering.hpp"

namespace holoscan::detail {

/// @brief Lower services whose storage contract is known before Tensor pool allocation.
[[nodiscard]] std::unordered_set<std::uint64_t> lower_preallocation_channel_services(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    PhysicalTopologyFacts& topology, bool force_holomq_process_local_inline,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations);

/// @brief Lower raw Tensor services after direct and interface pools are frozen.
/// @return Sources rejected because their pass-through CUDA placement is unresolved.
[[nodiscard]] std::unordered_set<std::uint64_t> lower_tensor_channel_services(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    PhysicalTopologyFacts& topology,
    const std::unordered_set<std::uint64_t>& rejected_tensor_fanout_sources,
    const std::unordered_set<std::uint64_t>& rejected_direct_tensor_sources,
    bool force_holomq_process_local_inline,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations);

/// @brief Canonicalize channel records and freeze process-local event-wake policy.
///
/// Each HoloMQ service that `emit_admission` covers and ProcessLocalEmitAdmissionFacts admits is
/// delivered inside the producer's emit() and needs no event wake.
void finalize_channel_services(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    std::span<const ExecutionPartitionRecord> execution_partitions,
    std::span<const LifecycleRestartTargetPlanRecord> lifecycle_targets,
    std::span<const LifecycleRestartClosurePlanRecord> lifecycle_closures,
    bool force_holomq_process_local_event_wake, bool experimental_split_cut = false,
    ProcessLocalEmitAdmissionScope emit_admission = ProcessLocalEmitAdmissionScope::kNone);

}  // namespace holoscan::detail
