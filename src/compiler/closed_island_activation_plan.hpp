/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "compiler/execution_plan_state.hpp"

namespace holoscan::detail {

/// Derive the exact private authority for eligible two-Operator activation islands.
///
/// Ineligible partitions are omitted rather than rejected. The derivation is a
/// pure cold-path projection over already lowered canonical plan relations.
[[nodiscard]] std::vector<ClosedIslandActivationPlanRecord> derive_closed_island_activation_plan(
    const ExecutionPlanState& state);

/// Reconstruct and compare the complete private closed-island authority.
///
/// This check also validates the process-local Channel relations on which the
/// derivation depends. It performs no Runtime or backend mutation.
[[nodiscard]] bool valid_closed_island_activation_plan(const ExecutionPlanState& state);

}  // namespace holoscan::detail
