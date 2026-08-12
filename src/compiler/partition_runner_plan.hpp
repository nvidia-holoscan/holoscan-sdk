/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <vector>

#include "compiler/execution_plan_state.hpp"

namespace holoscan::detail {

/// Derive the complete private partition-local runner inventory.
///
/// Rejected and zero-work plans produce an empty inventory. An admitted
/// non-empty plan that cannot be reconstructed exactly produces no value.
[[nodiscard]] std::optional<std::vector<PartitionRunnerPlanRecord>> derive_partition_runner_plan(
    const ExecutionPlanState& state);

/// Reconstruct and compare all private partition-local runner authority.
///
/// The check performs no Runtime or backend mutation.
[[nodiscard]] bool valid_partition_runner_plan(const ExecutionPlanState& state);

}  // namespace holoscan::detail
