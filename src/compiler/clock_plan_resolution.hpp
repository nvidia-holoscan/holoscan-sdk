/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <span>

#include <holoscan/core/execution_plan.hpp>

#include "core/clock_binding_resolution.hpp"
#include "core/graph_snapshot.hpp"

namespace holoscan::detail {

/// Resolve one exact graph-local clock declaration in the lowered plan inventory.
[[nodiscard]] inline ClockId find_clock_plan_id(std::span<const ClockPlanRecord> clocks,
                                                GraphId graph, const ClockKey& key) noexcept {
  const auto found = std::find_if(clocks.begin(), clocks.end(), [&](const auto& clock) {
    return clock.graph == graph && clock.key == key;
  });
  return found == clocks.end() ? ClockId{} : found->id;
}

/// Resolve the default clock for one graph in the lowered plan inventory.
[[nodiscard]] inline ClockId find_default_clock_plan_id(std::span<const ClockPlanRecord> clocks,
                                                        GraphId graph) noexcept {
  const auto found = std::find_if(clocks.begin(), clocks.end(), [&](const auto& clock) {
    return clock.graph == graph && clock.is_default;
  });
  return found == clocks.end() ? ClockId{} : found->id;
}

/// Resolve an authored local/imported clock key to its canonical lowered plan identity.
[[nodiscard]] inline ClockId resolve_clock_plan_id(const GraphSnapshotData& snapshot,
                                                   std::span<const ClockPlanRecord> clocks,
                                                   GraphId graph, const ClockKey& key) noexcept {
  const AuthoredClockResolution resolved = resolve_authored_clock(snapshot.graphs,
                                                                  snapshot.clocks,
                                                                  snapshot.clock_imports,
                                                                  snapshot.clock_bindings,
                                                                  graph,
                                                                  key);
  return resolved.resolved()
             ? find_clock_plan_id(clocks, resolved.declaration->graph, resolved.declaration->key)
             : ClockId{};
}

}  // namespace holoscan::detail
