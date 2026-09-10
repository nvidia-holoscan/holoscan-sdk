/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compiler/execution_plan_state.hpp"

namespace holoscan::detail {

/// Return whether the frozen Graph contains any structural topology.
[[nodiscard]] inline bool has_structural_runtime_topology(const ExecutionPlanState& plan) noexcept {
  return !plan.graph.vertices().empty();
}

/// Return whether the complete plan contains any callback-owning Operator factory.
[[nodiscard]] inline bool has_plan_callback_owners(const ExecutionPlanState& plan) noexcept {
  return !plan.executable_factories.operators.empty();
}

}  // namespace holoscan::detail
