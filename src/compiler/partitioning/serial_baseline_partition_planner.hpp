/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "partition_planner.hpp"

namespace holoscan::detail::partitioning {

/// Produce the coarsest cover already established by mandatory execution-domain identity.
class SerialBaselinePartitionPlanner final : public PartitionPlanner {
 public:
  [[nodiscard]] PartitionPlannerDescriptor descriptor() const noexcept override;
  [[nodiscard]] expected<PartitionProposal, Error> propose(
      const PartitionProblem& problem) const override;
};

[[nodiscard]] std::unique_ptr<PartitionPlanner> make_serial_baseline_partition_planner();

}  // namespace holoscan::detail::partitioning
