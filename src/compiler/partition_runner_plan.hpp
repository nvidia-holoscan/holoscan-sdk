/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <vector>

#include "compiler/execution_plan_state.hpp"

namespace holoscan::detail {

/// Describe incompatible exact affinity requests for one controller-owned partition runner.
struct InlinePartitionControllerAffinityConflict {
  ProcessGroupId process_group{};
  ExecutionPartitionId partition{};
  std::uint32_t execution_lane{};
  CPUAffinity controller_affinity;
  CPUAffinity partition_affinity;
};

/// Derive the complete private partition-local runner inventory.
///
/// Rejected and zero-work plans produce an empty inventory. An admitted
/// non-empty plan that cannot be reconstructed exactly produces no value.
[[nodiscard]] std::optional<std::vector<PartitionRunnerPlanRecord>> derive_partition_runner_plan(
    const ExecutionPlanState& state);

/// Find the first process whose sole runner cannot share its controller thread's exact affinity.
///
/// A process-local runner executes inline when it is the only runner in that process. If both the
/// controller and that runner carry explicit affinity, one physical thread can honor both accepted
/// contracts only when the normalized CPU sets are identical.
[[nodiscard]] std::optional<InlinePartitionControllerAffinityConflict>
find_inline_partition_controller_affinity_conflict(const ExecutionPlanState& state);

/// Reconstruct and compare all private partition-local runner authority.
///
/// The check performs no Runtime or backend mutation.
[[nodiscard]] bool valid_partition_runner_plan(const ExecutionPlanState& state);

}  // namespace holoscan::detail
