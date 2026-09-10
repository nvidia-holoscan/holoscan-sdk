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
void lower_tensor_channel_services(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    const PhysicalTopologyFacts& topology,
    const std::unordered_set<std::uint64_t>& rejected_tensor_fanout_sources,
    bool force_holomq_process_local_inline,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations);

/// @brief Canonicalize channel records and freeze process-local event-wake policy.
void finalize_channel_services(PhysicalPlanLoweringResult& result,
                               const GraphSnapshotData& snapshot,
                               std::span<const ExecutionPartitionRecord> execution_partitions,
                               bool force_holomq_process_local_event_wake);

}  // namespace holoscan::detail
