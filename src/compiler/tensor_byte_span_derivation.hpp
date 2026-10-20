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

/// @brief Own compile-evaluated Tensor slot capacities and bounded diagnostics.
struct TensorByteSpanDerivationLowering {
  std::vector<TensorByteSpanDerivationPlanRecord> records;  ///< Canonical output-ordered evidence.
  std::vector<PlanLoweringDiagnostic> diagnostics;          ///< Deterministic rejection evidence.
};

/// @brief Resolve immutable Tensor output-pool byte-span expressions.
/// @param factories Generation-zero factory inventory after Parameter overrides are applied.
/// @param snapshot Canonical pointer-free Graph declaration state.
/// @param plan_generation Nonzero plan generation for emitted evidence.
/// @param parameter_fields Pointer-free Parameter declaration plan used for identity validation.
/// @return Total derivation evidence and fail-closed diagnostics.
[[nodiscard]] TensorByteSpanDerivationLowering lower_tensor_byte_span_derivations(
    const ExecutableFactoryInventory& factories, const GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation, std::span<const ParameterFieldPlanRecord> parameter_fields);

}  // namespace holoscan::detail
