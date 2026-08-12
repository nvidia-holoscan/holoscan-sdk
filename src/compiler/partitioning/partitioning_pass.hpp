/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <span>
#include <string_view>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "partition_plan_validator.hpp"
#include "partition_planner_catalog.hpp"

namespace holoscan::detail::partitioning {

/// Carry compile-owned selector inputs after the process environment was captured exactly once.
struct PartitionSelectionInput {
  std::optional<std::string_view> captured_environment_value;
  std::optional<AlgorithmKey> compiler_service_request;
  std::optional<AlgorithmKey> test_override_request;
  Profile root_profile{Profile::kStandard};
  std::span<const std::optional<Profile>> effective_profiles;
  std::span<const RegionProfileRecord> effective_regions;
};

/// Carry validated logical partition output and its read-only selection provenance.
struct PartitioningPassResult {
  AlgorithmKey requested_algorithm;
  AlgorithmMaturity maturity{AlgorithmMaturity::kProduction};
  PlanSelectionSource selection_source{PlanSelectionSource::kCoreDefault};
  bool experimental_warning_required{};
  ValidatedPartitionPlan plan;
};

/// Select, invoke, and validate one private planner without fallback or runtime resource lowering.
class PartitioningPass {
 public:
  explicit PartitioningPass(const PartitionPlannerCatalog& catalog) : catalog_(catalog) {}

  [[nodiscard]] expected<PartitioningPassResult, Error> run(
      const PartitionSelectionInput& selection, const PartitionProblem& problem) const;

 private:
  const PartitionPlannerCatalog& catalog_;
};

}  // namespace holoscan::detail::partitioning
