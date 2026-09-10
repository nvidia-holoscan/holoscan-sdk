/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "partition_planner.hpp"

namespace holoscan::detail::partitioning {

using PartitionPlannerFactory = std::unique_ptr<PartitionPlanner> (*)();

/// Bind one immutable planner descriptor to one explicit factory.
struct PartitionPlannerRegistration {
  PartitionPlannerDescriptor descriptor;
  PartitionPlannerFactory create{};
};

/// Immutable constructor-injected catalog for private built-in partition planners.
class PartitionPlannerCatalog {
 public:
  /// Validate and own an explicit planner registration table.
  [[nodiscard]] static expected<PartitionPlannerCatalog, Error> create(
      std::span<const PartitionPlannerRegistration> registrations,
      PartitionPlannerCapabilities available_capabilities = kAllPartitionPlannerCapabilities);

  /// Create the exact baseline built-in baseline and source-chain catalog.
  [[nodiscard]] static expected<PartitionPlannerCatalog, Error> create_default_builtins();

  /// Construct a fresh planner for an exact algorithm key.
  [[nodiscard]] expected<std::unique_ptr<PartitionPlanner>, Error> make(
      const AlgorithmKey& algorithm) const;

 private:
  struct Entry {
    std::string algorithm_id;
    std::uint32_t algorithm_version{};
    std::uint32_t interface_version{};
    AlgorithmMaturity maturity{AlgorithmMaturity::kProduction};
    PartitionPlannerCapabilities required_capabilities;
    PartitionPlannerFactory create{};
  };

  explicit PartitionPlannerCatalog(std::vector<Entry> entries) : entries_(std::move(entries)) {}

  std::vector<Entry> entries_;
};

}  // namespace holoscan::detail::partitioning
