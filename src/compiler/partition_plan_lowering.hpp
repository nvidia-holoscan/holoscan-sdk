/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>

#include "../core/graph_snapshot.hpp"
#include "execution_plan_state.hpp"

namespace holoscan::detail {

/// Lower one canonical Graph snapshot through the selected private EA1 partition planner.
[[nodiscard]] expected<void, Error> lower_partition_plan(
    const GraphSnapshotData& snapshot, Profile root_profile,
    std::span<const std::optional<Profile>> effective_profiles,
    std::optional<std::string_view> captured_environment_value,
    std::span<const std::uint32_t> lane_by_vertex,
    std::span<const std::uint32_t> process_group_by_vertex, ExecutionPlanState& output);

/// Validate the published partition cover against the temporal sources selected by contracts.
///
/// Partition proposal runs before Parameter-resolved temporal replay, so no planner turns
/// declaration-only readiness sources into execution boundaries. This cold post-replay pass
/// rejects every implicit Graph or authored partition that contains more than one actual
/// execution root.
[[nodiscard]] expected<void, Error> validate_partition_execution_roots(
    const GraphSnapshotData& snapshot, ExecutionPlanState& output);

}  // namespace holoscan::detail
