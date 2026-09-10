/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>

#include "compiler/physical_plan_lowering.hpp"
#include "compiler/physical_topology_lowering.hpp"

namespace holoscan::detail {

/// @brief Lower publication authority and compiler-owned output storage.
void lower_output_storage(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    std::span<const ExecutionPartitionRecord> execution_partitions, PhysicalTopologyFacts& topology,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations,
    std::span<const InputSelectionPlanRecord> input_selections,
    std::span<const BracketedWindowPlanRecord> bracketed_windows,
    std::span<const ResolvedQueueSizing> resolved_queue_sizings,
    std::span<const QueueSizingRecipePlanRecord> queue_sizing_recipes,
    bool force_holomq_process_local_inline);

}  // namespace holoscan::detail
