/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "compiler/execution_plan_state.hpp"
#include "core/graph_snapshot.hpp"

namespace holoscan::detail {

/// Synthetic canonical path component for the root Graph.
///
/// Percent-encoding always escapes '%', so this value cannot collide with an encoded
/// user-authored Graph path segment.
inline constexpr std::string_view kRootGraphPathComponent{"%root"};

/// Hold one atomically validated clock catalog, direct conversions, and readiness table.
///
/// Every numeric clock identity in this result belongs to `plan_generation`.
/// The compiler publishes the result only after validating every declaration,
/// so an error cannot expose a partial catalog or source table.
struct ClockAndReadinessLoweringResult {
  PlanGenerationId plan_generation{};
  ClockId root_default_clock{};
  std::vector<ClockPlanRecord> clocks;
  std::vector<ClockImportResolutionPlanRecord> clock_import_resolutions;
  std::vector<ClockConversionRecord> clock_conversions;
  std::vector<ReadinessPlanRecord> readiness;
  std::vector<DiagnosticSource> unused_source_warnings;
};

/// Find the default clock for one Graph in an already lowered plan-local clock catalog.
[[nodiscard]] ClockId find_plan_default_clock_id(std::span<const ClockPlanRecord> clocks,
                                                 GraphId graph) noexcept;

/// Resolve one authored clock reference into an already lowered plan-local clock catalog.
[[nodiscard]] ClockId resolve_plan_clock_id(const GraphSnapshotData& snapshot,
                                            std::span<const ClockPlanRecord> clocks, GraphId graph,
                                            const ClockKey& key) noexcept;

/// Lower clocks, direct conversions, and readiness declarations into one plan-local identity space.
///
/// On failure the result is absent, so callers cannot accidentally publish a
/// partial clock catalog, conversion table, or source table.
[[nodiscard]] expected<ClockAndReadinessLoweringResult, Error> lower_clocks_and_readiness(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const std::optional<Profile>> effective_vertex_profiles);

}  // namespace holoscan::detail
