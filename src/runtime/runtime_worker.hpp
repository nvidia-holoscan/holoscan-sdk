/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/run_session.hpp>

#include "../compiler/execution_plan_lookup.hpp"
#include "../compiler/execution_plan_state.hpp"

namespace holoscan::detail {

/// Return the compile-admitted process group that owns a canonical vertex.
[[nodiscard]] inline ProcessGroupId process_group_for(const ExecutionPlanState& plan,
                                                      VertexId vertex) noexcept {
  const auto* found = find_vertex_process_placement_by_vertex(plan, vertex);
  return found == nullptr ? ProcessGroupId{} : found->process_group;
}

/// Return whether a canonical vertex belongs to the selected worker.
[[nodiscard]] inline bool worker_owns_vertex(const ExecutionPlanState& plan,
                                             ProcessGroupId process_group,
                                             VertexId vertex) noexcept {
  return process_group && process_group_for(plan, vertex) == process_group;
}

/// Start one source-private process-group worker from an accepted plan.
///
/// Deployment/runtime infrastructure selects only a compiler-produced group.
/// This is deliberately not an installed application-author API.
[[nodiscard]] HOLOSCAN_CORE_EXPORT RunSession run_async_worker(const ExecutionPlan& plan,
                                                               ProcessGroupId process_group);

}  // namespace holoscan::detail
