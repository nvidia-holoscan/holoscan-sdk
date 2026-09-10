/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdint>
#include <limits>
#include <optional>

#include <gtest/gtest.h>

#include "core/pool_capacity_reducer.hpp"

namespace holoscan::detail {
namespace {

constexpr PoolCapacityTerm term(std::uint32_t scenario, std::uint32_t group, std::uint32_t active,
                                std::optional<std::uint32_t> ceiling = std::nullopt,
                                std::uint32_t multiplicity = 1U) {
  return PoolCapacityTerm{.scenario = PoolCapacityScenarioId{scenario},
                          .alternative_group = PoolCapacityAlternativeGroupId{group},
                          .active_entries = active,
                          .admitted_entry_ceiling = ceiling,
                          .mapping_multiplicity = multiplicity};
}

TEST(PoolCapacityReducer, MaxesAlternativesSumsGroupsAndMaxesScenarios) {
  constexpr std::array terms{term(1U, 1U, 3U, 5U),
                             term(1U, 1U, 7U, 9U),
                             term(1U, 2U, 4U, 8U),
                             term(2U, 1U, 20U, 24U),
                             term(2U, 2U, 1U, 2U, 2U)};
  std::array<MappedPoolCapacityTerm, terms.size()> scratch{};
  const auto result = reduce_pool_capacity(terms, scratch);
  ASSERT_TRUE(result);
  EXPECT_EQ(result->required_entries, 22U);
  EXPECT_EQ(result->admitted_entry_ceiling, 28U);
  EXPECT_EQ(scratch[4].active_entries, 2U);
  EXPECT_EQ(scratch[4].admitted_entry_ceiling, 4U);
}

TEST(PoolCapacityReducer, IsOrderIndependentAndPropagatesAnUnknownCeiling) {
  constexpr std::array forward{term(1U, 1U, 5U, 8U), term(1U, 2U, 7U), term(2U, 1U, 4U, 6U)};
  constexpr std::array reverse{forward[2], forward[1], forward[0]};
  std::array<MappedPoolCapacityTerm, forward.size()> forward_scratch{};
  std::array<MappedPoolCapacityTerm, reverse.size()> reverse_scratch{};
  const auto first = reduce_pool_capacity(forward, forward_scratch);
  const auto second = reduce_pool_capacity(reverse, reverse_scratch);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first->required_entries, 12U);
  EXPECT_FALSE(first->admitted_entry_ceiling);
  EXPECT_EQ(*first, *second);
}

TEST(PoolCapacityReducer, AcceptsExactBoundaryAndRejectsOverflowOrInvalidTerms) {
  constexpr std::uint32_t kMaximum = std::numeric_limits<std::uint32_t>::max();
  {
    constexpr std::array terms{term(1U, 1U, kMaximum, kMaximum)};
    std::array<MappedPoolCapacityTerm, 1U> scratch{};
    const auto result = reduce_pool_capacity(terms, scratch);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->required_entries, kMaximum);
  }
  {
    constexpr std::array terms{term(1U, 1U, kMaximum, kMaximum, 2U)};
    std::array<MappedPoolCapacityTerm, 1U> scratch{};
    const auto result = reduce_pool_capacity(terms, scratch);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), PoolCapacityReductionError::kOverflow);
  }
  {
    constexpr std::array terms{term(1U, 1U, kMaximum, kMaximum), term(1U, 2U, 1U, 1U)};
    std::array<MappedPoolCapacityTerm, 2U> scratch{};
    const auto result = reduce_pool_capacity(terms, scratch);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), PoolCapacityReductionError::kOverflow);
  }
  {
    constexpr std::array terms{term(0U, 1U, 1U, 1U)};
    std::array<MappedPoolCapacityTerm, 1U> scratch{};
    const auto result = reduce_pool_capacity(terms, scratch);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), PoolCapacityReductionError::kInvalidTerm);
  }
  for (const auto invalid : std::array{
           term(1U, 0U, 1U, 1U),
           term(1U, 1U, 0U, 1U),
           term(1U, 1U, 2U, 1U),
           term(1U, 1U, 1U, 1U, 0U),
       }) {
    const std::array terms{invalid};
    std::array<MappedPoolCapacityTerm, 1U> scratch{};
    const auto result = reduce_pool_capacity(terms, scratch);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), PoolCapacityReductionError::kInvalidTerm);
  }
  {
    constexpr std::array terms{term(1U, 1U, 1U, 1U)};
    std::array<MappedPoolCapacityTerm, 0U> scratch{};
    const auto result = reduce_pool_capacity(terms, scratch);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), PoolCapacityReductionError::kScratchTooSmall);
  }
  {
    constexpr std::array<PoolCapacityTerm, 0U> terms{};
    std::array<MappedPoolCapacityTerm, 0U> scratch{};
    const auto result = reduce_pool_capacity(terms, scratch);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), PoolCapacityReductionError::kEmpty);
  }
}

}  // namespace
}  // namespace holoscan::detail
