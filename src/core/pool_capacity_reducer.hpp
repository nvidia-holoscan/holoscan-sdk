/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

#include <holoscan/core/execution_plan_ids.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::detail {

enum class PoolCapacityReductionError : std::uint8_t {
  kEmpty = 0,
  kInvalidTerm = 1,
  kOverflow = 2,
  kScratchTooSmall = 3,
};

struct PoolCapacityTerm {
  PoolCapacityScenarioId scenario{};
  PoolCapacityAlternativeGroupId alternative_group{};
  std::uint32_t active_entries{};
  std::optional<std::uint32_t> admitted_entry_ceiling;
  std::uint32_t mapping_multiplicity{1U};
};

struct MappedPoolCapacityTerm {
  PoolCapacityScenarioId scenario{};
  PoolCapacityAlternativeGroupId alternative_group{};
  std::uint32_t active_entries{};
  std::optional<std::uint32_t> admitted_entry_ceiling;
};

struct ReducedPoolCapacity {
  std::uint32_t required_entries{};
  std::optional<std::uint32_t> admitted_entry_ceiling;

  friend constexpr bool operator==(const ReducedPoolCapacity&,
                                   const ReducedPoolCapacity&) noexcept = default;
};

// The reducer is a single constexpr proof over mapped terms and their overflow/ceiling rules.
// Keeping the checks adjacent makes compile-time and runtime evaluation identical.
[[nodiscard]] constexpr expected<ReducedPoolCapacity, PoolCapacityReductionError>
reduce_pool_capacity(  // NOLINT(readability-function-cognitive-complexity)
    std::span<const PoolCapacityTerm> terms, std::span<MappedPoolCapacityTerm> scratch) noexcept {
  if (terms.empty()) {
    return make_unexpected(PoolCapacityReductionError::kEmpty);
  }
  if (scratch.size() < terms.size()) {
    return make_unexpected(PoolCapacityReductionError::kScratchTooSmall);
  }
  auto mapped = scratch.first(terms.size());
  for (std::size_t index = 0U; index < terms.size(); ++index) {
    const auto& term = terms[index];
    if (term.scenario.value == 0U || term.alternative_group.value == 0U ||
        term.active_entries == 0U || term.mapping_multiplicity == 0U ||
        (term.admitted_entry_ceiling && (*term.admitted_entry_ceiling == 0U ||
                                         *term.admitted_entry_ceiling < term.active_entries))) {
      return make_unexpected(PoolCapacityReductionError::kInvalidTerm);
    }
    if (term.active_entries >
            std::numeric_limits<std::uint32_t>::max() / term.mapping_multiplicity ||
        (term.admitted_entry_ceiling &&
         *term.admitted_entry_ceiling >
             std::numeric_limits<std::uint32_t>::max() / term.mapping_multiplicity)) {
      return make_unexpected(PoolCapacityReductionError::kOverflow);
    }
    mapped[index] = MappedPoolCapacityTerm{
        .scenario = term.scenario,
        .alternative_group = term.alternative_group,
        .active_entries = term.active_entries * term.mapping_multiplicity,
        .admitted_entry_ceiling = term.admitted_entry_ceiling
                                      ? std::optional<std::uint32_t>{*term.admitted_entry_ceiling *
                                                                     term.mapping_multiplicity}
                                      : std::nullopt,
    };
  }

  ReducedPoolCapacity result{.required_entries = 1U, .admitted_entry_ceiling = 1U};
  for (std::size_t scenario_index = 0U; scenario_index < mapped.size(); ++scenario_index) {
    const auto scenario = mapped[scenario_index].scenario;
    if (std::ranges::any_of(mapped.first(scenario_index),
                            [&](const auto& prior) { return prior.scenario == scenario; })) {
      continue;
    }
    std::uint32_t scenario_active{};
    std::optional<std::uint32_t> scenario_ceiling{0U};
    for (std::size_t group_index = 0U; group_index < mapped.size(); ++group_index) {
      const auto& group_origin = mapped[group_index];
      if (group_origin.scenario != scenario ||
          std::ranges::any_of(mapped.first(group_index), [&](const auto& prior) {
            return prior.scenario == scenario &&
                   prior.alternative_group == group_origin.alternative_group;
          })) {
        continue;
      }
      std::uint32_t group_active{};
      std::optional<std::uint32_t> group_ceiling{0U};
      for (const auto& term : mapped) {
        if (term.scenario != scenario || term.alternative_group != group_origin.alternative_group) {
          continue;
        }
        group_active = std::max(group_active, term.active_entries);
        if (!term.admitted_entry_ceiling) {
          group_ceiling.reset();
        } else if (group_ceiling) {
          *group_ceiling = std::max(*group_ceiling, *term.admitted_entry_ceiling);
        }
      }
      if (group_active > std::numeric_limits<std::uint32_t>::max() - scenario_active) {
        return make_unexpected(PoolCapacityReductionError::kOverflow);
      }
      scenario_active += group_active;
      if (!group_ceiling) {
        scenario_ceiling.reset();
      } else if (scenario_ceiling) {
        if (*group_ceiling > std::numeric_limits<std::uint32_t>::max() - *scenario_ceiling) {
          return make_unexpected(PoolCapacityReductionError::kOverflow);
        }
        *scenario_ceiling += *group_ceiling;
      }
    }
    result.required_entries = std::max(result.required_entries, scenario_active);
    if (!scenario_ceiling) {
      result.admitted_entry_ceiling.reset();
    } else if (result.admitted_entry_ceiling) {
      *result.admitted_entry_ceiling = std::max(*result.admitted_entry_ceiling, *scenario_ceiling);
    }
  }
  return result;
}

}  // namespace holoscan::detail
