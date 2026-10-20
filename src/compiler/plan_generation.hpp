/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compiler/execution_plan_state.hpp"

namespace holoscan::detail {

/// Derive the deterministic nonzero plan generation from canonical semantic plan bytes.
///
/// The state's artifact digest must have been finalized while every plan-generation
/// field used the compiler's fixed canonicalization seed. Protected resolved
/// Parameter value records are incorporated without exposing their digest bytes
/// in public JSON. Consequently, equivalent request spellings that lower to the
/// same semantic records share one identity, while addresses, authoring order,
/// logging state, and unused valid inputs do not contribute.
[[nodiscard]] PlanGenerationId derive_plan_generation_id(
    const ExecutionPlanState& canonical_state) noexcept;

/// Rebind every plan-generation-bearing record to one final identity.
///
/// This cold compiler pass changes provenance fields and the Parameter-catalog
/// compatibility digest only. All normalized semantic records remain unchanged.
void rebind_plan_generation(ExecutionPlanState& state, PlanGenerationId plan_generation);

}  // namespace holoscan::detail
