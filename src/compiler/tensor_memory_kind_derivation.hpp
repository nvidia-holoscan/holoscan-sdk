/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>
#include <vector>

#include <holoscan/core/execution_plan.hpp>

#include "core/graph_snapshot.hpp"
#include "plan_lowering_diagnostic.hpp"
#include "runtime/executable_factory_inventory.hpp"

namespace holoscan::detail {

/// @brief Own compile-evaluated Tensor memory kinds and bounded diagnostics.
struct TensorMemoryKindDerivationLowering {
  std::vector<TensorMemoryKindDerivationPlanRecord> records;  ///< Canonical output evidence.
  std::vector<PlanLoweringDiagnostic> diagnostics;            ///< Deterministic rejection evidence.
};

/// @brief Return whether a Graph requires a mutable memory-kind resolution snapshot.
/// @param snapshot Canonical Graph declaration.
/// @return `true` when any raw or descriptor Tensor output derives its memory kind.
[[nodiscard]] bool has_tensor_memory_kind_derivations(const GraphSnapshotData& snapshot) noexcept;

/// @brief Resolve immutable Parameter-derived Tensor output memory kinds.
/// @param factories Generation-zero factory inventory after Parameter overrides are applied.
/// @param snapshot Mutable physical-lowering copy of the canonical Graph declaration.
/// @param plan_generation Nonzero plan generation for emitted evidence.
/// @param parameter_fields Pointer-free Parameter declaration plan used for identity validation.
/// @return Total derivation evidence and fail-closed diagnostics.
///
/// The source Graph snapshot retained by ExecutionPlan is not passed here. The compiler supplies a
/// private copy so the resolved concrete kind can flow through all existing physical lowerers while
/// normalized Graph identity continues to describe the unresolved declaration.
[[nodiscard]] TensorMemoryKindDerivationLowering lower_tensor_memory_kind_derivations(
    const ExecutableFactoryInventory& factories, GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation, std::span<const ParameterFieldPlanRecord> parameter_fields);

}  // namespace holoscan::detail
