/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "partition_planner.hpp"

namespace holoscan::detail::partitioning {

/// Produce deterministic source/fan-in/fan-out chain partitions for controlled EA1 evidence.
class SourceChainPartitionPlanner final : public PartitionPlanner {
 public:
  [[nodiscard]] PartitionPlannerDescriptor descriptor() const noexcept override;
  [[nodiscard]] expected<PartitionProposal, Error> propose(
      const PartitionProblem& problem) const override;
};

[[nodiscard]] std::unique_ptr<PartitionPlanner> make_source_chain_partition_planner();

}  // namespace holoscan::detail::partitioning
