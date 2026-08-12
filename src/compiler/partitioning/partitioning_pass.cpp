/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "partitioning_pass.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/logger/logger.hpp>

namespace holoscan::detail::partitioning {
namespace {

[[nodiscard]] unexpected<Error> selection_error(ErrorCode code, std::string_view token) {
  return make_unexpected(Error{code, std::string(token)});
}

[[nodiscard]] bool environment_profile_permitted(Profile profile) noexcept {
  return profile == Profile::kStandard || profile == Profile::kResearch;
}

[[nodiscard]] expected<AlgorithmKey, Error> parse_environment_value(std::string_view value) {
  if (value == "serial_baseline/v1") {
    return serial_baseline_algorithm();
  }
  if (value == "source_chain/v1") {
    return source_chain_algorithm();
  }
  return selection_error(ErrorCode::kInvalidArgument, "PARTITION_SELECTION_ENV_VALUE_INVALID");
}

}  // namespace

expected<PartitioningPassResult, Error> PartitioningPass::run(
    const PartitionSelectionInput& selection, const PartitionProblem& problem) const {
  std::optional<AlgorithmKey> environment_request;
  if (selection.captured_environment_value.has_value()) {
    auto parsed = parse_environment_value(*selection.captured_environment_value);
    if (!parsed) {
      return forward_error(std::move(parsed));
    }
    environment_request = parsed.value();
  }

  const std::uint32_t explicit_source_count =
      static_cast<std::uint32_t>(environment_request.has_value()) +
      static_cast<std::uint32_t>(selection.compiler_service_request.has_value()) +
      static_cast<std::uint32_t>(selection.test_override_request.has_value());
  if (explicit_source_count > 1U) {
    return selection_error(ErrorCode::kInvalidArgument, "PARTITION_SELECTION_CONFLICT");
  }

  AlgorithmKey selected_algorithm = serial_baseline_algorithm();
  PlanSelectionSource selection_source = PlanSelectionSource::kCoreDefault;
  if (environment_request.has_value()) {
    if (!environment_profile_permitted(selection.root_profile)) {
      return selection_error(ErrorCode::kPermissionDenied, "PARTITION_SELECTION_PROFILE_FORBIDDEN");
    }
    for (const std::optional<Profile> profile : selection.effective_profiles) {
      if (!profile.has_value() || !environment_profile_permitted(*profile)) {
        return selection_error(ErrorCode::kPermissionDenied,
                               "PARTITION_SELECTION_PROFILE_FORBIDDEN");
      }
    }
    for (const auto& region : selection.effective_regions) {
      if (!region.effective_profile.has_value() ||
          !environment_profile_permitted(*region.effective_profile)) {
        return selection_error(ErrorCode::kPermissionDenied,
                               "PARTITION_SELECTION_PROFILE_FORBIDDEN");
      }
    }
    selected_algorithm = *environment_request;
    selection_source = PlanSelectionSource::kEnvironmentOverride;
  } else if (selection.compiler_service_request.has_value()) {
    selected_algorithm = *selection.compiler_service_request;
    selection_source = PlanSelectionSource::kCompilerService;
  } else if (selection.test_override_request.has_value()) {
    selected_algorithm = *selection.test_override_request;
    selection_source = PlanSelectionSource::kTestOverride;
  }

  auto planner = catalog_.make(selected_algorithm);
  if (!planner) {
    return forward_error(std::move(planner));
  }
  const PartitionPlannerDescriptor descriptor = planner.value()->descriptor();
  auto proposal = planner.value()->propose(problem);
  if (!proposal) {
    return forward_error(std::move(proposal));
  }
  auto validated = validate_partition_proposal(problem, descriptor, proposal.value());
  if (!validated) {
    return forward_error(std::move(validated));
  }

  HOLOSCAN_LOG_TRACE("Partition planner selected: algorithm={}/{} source={}",
                     descriptor.algorithm.id,
                     descriptor.algorithm.version,
                     static_cast<std::uint32_t>(selection_source));
  const bool experimental_warning_required =
      descriptor.maturity == AlgorithmMaturity::kExperimental;
  return PartitioningPassResult{descriptor.algorithm,
                                descriptor.maturity,
                                selection_source,
                                experimental_warning_required,
                                std::move(validated).value()};
}

}  // namespace holoscan::detail::partitioning
