/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/pool_capacity_plan_lowering.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../core/pool_capacity_reducer.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] Error pool_error(std::string_view token) {
  return Error{.code = ErrorCode::kInvalidArgument, .message = std::string(token)};
}

[[nodiscard]] bool canonical_reasons(std::span<const TensorSlotClaimReason> reasons) noexcept {
  if (reasons.empty()) {
    return false;
  }
  for (std::size_t index = 0U; index < reasons.size(); ++index) {
    if (static_cast<std::uint8_t>(reasons[index]) >
            static_cast<std::uint8_t>(TensorSlotClaimReason::kRetirementService) ||
        (index != 0U && static_cast<std::uint8_t>(reasons[index - 1U]) >=
                            static_cast<std::uint8_t>(reasons[index]))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] expected<std::pair<std::uint32_t, std::uint32_t>, Error> dependency_capacity(
    const PoolCapacityDependencyPlanRecord& dependency,
    std::span<const PoolCapacityContributorRecord> predecessors,
    std::span<const ResolvedQueueSizing> queue_sizings) {
  std::uint32_t active{};
  std::uint32_t ceiling{};
  if (dependency.source == PoolCapacityDependencySource::kQueueSizingRecipe) {
    if (!dependency.queue_sizing_recipe || !dependency.predecessor_contributor_id.empty() ||
        !canonical_reasons(dependency.reasons)) {
      return make_unexpected(pool_error("QUEUE_POOL_DEPENDENCY_INVALID"));
    }
    const auto sizing = std::ranges::find(
        queue_sizings, dependency.queue_sizing_recipe, &ResolvedQueueSizing::recipe);
    if (sizing == queue_sizings.end() || sizing->effective_logical_depth == 0U) {
      return make_unexpected(pool_error("QUEUE_POOL_DEPENDENCY_INVALID"));
    }
    active = sizing->effective_logical_depth;
    // An explicit queue depth is the exact logical and storage obligation. A nominal-rate
    // recommendation remains advisory evidence and must not silently inflate a provider pool.
    ceiling = sizing->explicit_depth ? active : sizing->admitted_depth_ceiling.value_or(active);
  } else if (dependency.source == PoolCapacityDependencySource::kPredecessorContributor) {
    if (dependency.queue_sizing_recipe || dependency.predecessor_contributor_id.empty() ||
        !dependency.reasons.empty()) {
      return make_unexpected(pool_error("QUEUE_POOL_DEPENDENCY_INVALID"));
    }
    const auto predecessor = std::ranges::find(predecessors,
                                               dependency.predecessor_contributor_id,
                                               &PoolCapacityContributorRecord::contributor_id);
    if (predecessor == predecessors.end() || predecessor->pool_id != dependency.pool_id ||
        predecessor->maximum_distinct_slots == 0U ||
        std::ranges::count_if(predecessors, [&](const auto& candidate) {
          return candidate.pool_id == dependency.pool_id &&
                 candidate.contributor_id == dependency.predecessor_contributor_id;
        }) != 1) {
      return make_unexpected(pool_error("QUEUE_POOL_DEPENDENCY_INVALID"));
    }
    active = predecessor->maximum_distinct_slots;
    ceiling = active;
  } else {
    return make_unexpected(pool_error("QUEUE_POOL_DEPENDENCY_INVALID"));
  }
  return std::pair{active, ceiling};
}

}  // namespace

expected<EvaluatedPoolCapacity, Error> evaluate_pool_capacity_plan(
    std::string_view pool_id, std::span<const PoolCapacityDependencyPlanRecord> dependencies,
    std::span<const PoolCapacityContributorRecord> predecessor_contributors,
    std::span<const ResolvedQueueSizing> queue_sizings) {
  if (pool_id.empty()) {
    return make_unexpected(pool_error("QUEUE_POOL_DEPENDENCY_INVALID"));
  }
  std::vector<PoolCapacityTerm> terms;
  for (const auto& dependency : dependencies) {
    if (dependency.pool_id != pool_id) {
      continue;
    }
    auto capacity = dependency_capacity(dependency, predecessor_contributors, queue_sizings);
    if (!capacity) {
      return make_unexpected(std::move(capacity.error()));
    }
    terms.push_back(PoolCapacityTerm{.scenario = dependency.scenario,
                                     .alternative_group = dependency.alternative_group,
                                     .active_entries = capacity->first,
                                     .admitted_entry_ceiling = capacity->second,
                                     .mapping_multiplicity = dependency.mapping_multiplicity});
  }
  if (terms.empty()) {
    return make_unexpected(pool_error("QUEUE_POOL_DEPENDENCY_INVALID"));
  }
  std::vector<MappedPoolCapacityTerm> scratch(terms.size());
  const auto capacity = reduce_pool_capacity(terms, scratch);
  if (!capacity || !capacity->admitted_entry_ceiling) {
    return make_unexpected(
        pool_error(!capacity && capacity.error() == PoolCapacityReductionError::kOverflow
                       ? "QUEUE_POOL_CAPACITY_UNBOUNDED"
                       : "QUEUE_POOL_DEPENDENCY_INVALID"));
  }
  return EvaluatedPoolCapacity{
      .required_entries = capacity->required_entries,
      .admitted_entry_ceiling =
          *capacity->admitted_entry_ceiling};  // NOLINT(bugprone-unchecked-optional-access)
}

}  // namespace holoscan::detail
