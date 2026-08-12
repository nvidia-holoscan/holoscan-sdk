/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include "compiler/execution_plan_state.hpp"

namespace holoscan::detail {

/// Derive private controller-owned execution authority for one conservative serial partition.
///
/// Ineligible plans return no record and continue through the ordinary
/// controller/lane path. The derivation is a pure cold-path projection over
/// already lowered canonical plan relations.
[[nodiscard]] std::optional<InlinePartitionExecutionPlanRecord>
derive_inline_partition_execution_plan(const ExecutionPlanState& state);

/// Reconstruct and compare the complete private controller-owned partition authority.
///
/// The check performs no Runtime or backend mutation.
[[nodiscard]] bool valid_inline_partition_execution_plan(const ExecutionPlanState& state);

}  // namespace holoscan::detail
