/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>

#include "compiler/physical_plan_lowering.hpp"

namespace holoscan::detail {

/// @brief Aggregate admitted Tensor pools into process/device resources and compute streams.
void lower_device_resources(PhysicalPlanLoweringResult& result, PlanGenerationId plan_generation,
                            std::span<const VertexProcessPlacementPlanRecord> process_placements,
                            std::span<const ExecutionPartitionRecord> execution_partitions,
                            CudaComputeStreamPolicy stream_policy);

}  // namespace holoscan::detail
