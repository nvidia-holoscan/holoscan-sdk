/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "../compiler/execution_plan_state.hpp"
#include "../time/clock_registry_internal.hpp"

namespace holoscan::detail {

/// Own the complete run-local clock catalog produced by cold reconstruction.
struct RuntimeClockReconstruction {
  ClockRegistry registry;
  ClockRef default_clock;
};

/// Resolve one Graph-scoped authored clock key into the immutable plan catalog.
[[nodiscard]] ClockId resolve_runtime_clock_id(const ExecutionPlanState& plan, GraphId graph,
                                               const ClockKey& key) noexcept;

/// Authenticate the complete frozen clock inventory before any runtime factory is invoked.
[[nodiscard]] expected<void, Error> validate_runtime_clock_reconstruction(
    const ExecutionPlanState& plan);

/// Materialize and authenticate the complete run-local clock catalog as one cold-start unit.
[[nodiscard]] expected<RuntimeClockReconstruction, Error> reconstruct_runtime_clocks(
    const ExecutionPlanState& plan);

}  // namespace holoscan::detail
