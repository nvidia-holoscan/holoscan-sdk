/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "partition_types.hpp"

namespace holoscan::detail::partitioning {

/// Validate an untrusted assignment-only proposal and assign canonical partition identities.
[[nodiscard]] expected<ValidatedPartitionPlan, Error> validate_partition_proposal(
    const PartitionProblem& problem, const PartitionPlannerDescriptor& selected_planner,
    const PartitionProposal& proposal);

}  // namespace holoscan::detail::partitioning
