/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>

#include "compiler/physical_plan_lowering.hpp"
#include "compiler/physical_topology_lowering.hpp"

namespace holoscan::detail {

/// @brief Aggregate admitted Tensor pools into process/device resources and compute streams.
/// The snapshot and topology name ports and device sources in operator device conflicts.
void lower_device_resources(PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
                            const PhysicalTopologyFacts& topology, PlanGenerationId plan_generation,
                            std::span<const VertexProcessPlacementPlanRecord> process_placements,
                            std::span<const ExecutionPartitionRecord> execution_partitions,
                            CudaComputeStreamPolicy stream_policy);

}  // namespace holoscan::detail
