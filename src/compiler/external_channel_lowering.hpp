/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <span>

#include "compiler/physical_plan_lowering.hpp"
#include "compiler/physical_topology_lowering.hpp"

namespace holoscan::detail {

/// @brief Lower external endpoint services and their bounded Tensor storage.
void lower_external_channel_services(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation, std::span<const std::optional<Profile>> effective_profiles,
    const RuntimePackageCapabilityView& package_capabilities,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    std::span<const LifecycleRestartTargetPlanRecord> lifecycle_targets,
    std::span<const ExecutionPartitionRecord> execution_partitions,
    PhysicalTopologyFacts& topology);

}  // namespace holoscan::detail
