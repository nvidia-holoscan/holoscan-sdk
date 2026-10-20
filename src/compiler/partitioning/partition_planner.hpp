/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "partition_types.hpp"

namespace holoscan::detail::partitioning {

/// Private compile-time strategy interface for assignment-only partition proposals.
class PartitionPlanner {
 public:
  virtual ~PartitionPlanner() = default;

  /// Return stable identity, maturity, interface, and capability requirements.
  [[nodiscard]] virtual PartitionPlannerDescriptor descriptor() const noexcept = 0;

  /// Propose a bounded assignment without assigning runtime resources or final public IDs.
  [[nodiscard]] virtual expected<PartitionProposal, Error> propose(
      const PartitionProblem& problem) const = 0;
};

}  // namespace holoscan::detail::partitioning
