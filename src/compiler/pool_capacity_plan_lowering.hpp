/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>
#include <string_view>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>

#include "compiler/rate_plan_lowering.hpp"

namespace holoscan::detail {

/// @brief Hold checked active and admitted-ceiling entries for one stable provider pool.
struct EvaluatedPoolCapacity {
  std::uint32_t required_entries{};
  std::uint32_t admitted_entry_ceiling{};
};

/// @brief Evaluate one pool's immutable dependency algebra without rebuilding queue sizing.
[[nodiscard]] expected<EvaluatedPoolCapacity, Error> evaluate_pool_capacity_plan(
    std::string_view pool_id, std::span<const PoolCapacityDependencyPlanRecord> dependencies,
    std::span<const PoolCapacityContributorRecord> predecessor_contributors,
    std::span<const ResolvedQueueSizing> queue_sizings);

}  // namespace holoscan::detail
