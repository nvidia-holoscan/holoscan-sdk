/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>

#include "compiler/physical_plan_lowering.hpp"

namespace holoscan::detail {

void validate_lifecycle_queue_geometry(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    std::span<const QueueSizingRecipePlanRecord> queue_sizing_recipes);

void validate_tensor_provider_geometry(PhysicalPlanLoweringResult& result,
                                       const GraphSnapshotData& snapshot,
                                       PlanGenerationId plan_generation,
                                       std::span<const ResolvedQueueSizing> resolved_queue_sizings);

}  // namespace holoscan::detail
