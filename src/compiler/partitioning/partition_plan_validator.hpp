/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "partition_types.hpp"

namespace holoscan::detail::partitioning {

/// Validate that exact authored membership has at most one normalized device obligation.
/// Empty obligations remain unconstrained; members index the supplied normalized vertex views.
[[nodiscard]] expected<void, Error> validate_authored_partition_device_domain(
    std::span<const PartitionVertexView> vertices, std::span<const VertexIndex> members);

/// Validate an untrusted assignment-only proposal and assign canonical partition identities.
[[nodiscard]] expected<ValidatedPartitionPlan, Error> validate_partition_proposal(
    const PartitionProblem& problem, const PartitionPlannerDescriptor& selected_planner,
    const PartitionProposal& proposal);

}  // namespace holoscan::detail::partitioning
