/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <span>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "partition_problem_lowering.hpp"
#include "partitioning/partitioning_pass.hpp"

namespace holoscan::detail {

/// Build the accepted cover for an exact developer-authored partition inventory.
[[nodiscard]] expected<partitioning::PartitioningPassResult, Error> build_manual_partition_cover(
    const GraphSnapshotData& snapshot, const PartitionProblemStorage& storage,
    std::span<const std::uint32_t> process_group_by_vertex,
    DiagnosticSource* failure_source = nullptr);

/// Attach external endpoint vertices to a constrained automatic cover.
[[nodiscard]] expected<void, Error> attach_constrained_partition_synthetics(
    const GraphSnapshotData& snapshot, const PartitionProblemStorage& storage,
    std::span<const std::uint32_t> process_group_by_vertex,
    partitioning::ValidatedPartitionPlan& plan);

/// Revalidate the complete accepted cover before publishing public evidence.
[[nodiscard]] expected<void, Error> validate_final_accepted_partition_cover(
    const GraphSnapshotData& snapshot, std::span<const std::uint32_t> process_group_by_vertex,
    const partitioning::ValidatedPartitionPlan& plan);

}  // namespace holoscan::detail
