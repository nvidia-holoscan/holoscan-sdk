/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "compiler/execution_plan_state.hpp"

namespace holoscan::detail::plan_artifact_rate_serialization_internal {

[[nodiscard]] bool rate_plan_records_valid(const ExecutionPlanState& state) noexcept;
[[nodiscard]] bool pool_capacity_plan_records_valid(const ExecutionPlanState& state) noexcept;

void append_rate_plan_sections(std::string& output, const ExecutionPlanState& state);
void append_pool_capacity_plan_sections(std::string& output, const ExecutionPlanState& state);

}  // namespace holoscan::detail::plan_artifact_rate_serialization_internal
