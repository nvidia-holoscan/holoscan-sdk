/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>

#include "../core/graph_snapshot.hpp"
#include "execution_plan_state.hpp"

namespace holoscan::detail {

struct TensorDeviceBindingRecord;

/// Lower one canonical Graph snapshot through the selected private baseline partition planner.
[[nodiscard]] expected<void, Error> lower_partition_plan(
    const GraphSnapshotData& snapshot, Profile root_profile,
    std::span<const std::optional<Profile>> effective_profiles,
    bool removed_environment_selector_present,
    std::span<const std::uint32_t> process_group_by_vertex, const DeploymentBinding& deployment,
    std::span<const TensorDeviceBindingRecord> tensor_device_bindings, ExecutionPlanState& output);

/// Validate the published partition cover against the temporal sources selected by contracts.
///
/// Automatic planning already consumes selected readiness roots. This additional cold guard
/// preserves the complete-manual exact-cover restriction without re-planning.
[[nodiscard]] expected<void, Error> validate_partition_execution_roots(
    const GraphSnapshotData& snapshot, ExecutionPlanState& output);

}  // namespace holoscan::detail
