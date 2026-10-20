/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace holoscan {

/// @brief Identify one compiler-produced execution partition within a plan.
struct ExecutionPartitionId {
  std::uint32_t value{};  ///< Dense nonzero canonical partition identity.

  /// @brief Return whether this identity names an accepted partition.
  /// @return \c true when \ref value is nonzero.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(ExecutionPartitionId, ExecutionPartitionId) noexcept = default;
};

/// @brief Identify one compile-admitted process-equivalence group.
struct ProcessGroupId {
  std::uint32_t value{};  ///< Dense nonzero identity in canonical group order.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(ProcessGroupId, ProcessGroupId) noexcept = default;
};

/// @brief Identify one immutable queue-sizing recipe.
struct QueueSizingRecipeId {
  std::uint32_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(QueueSizingRecipeId, QueueSizingRecipeId) noexcept = default;
};

/// @brief Identify one immutable accumulation-horizon record.
struct HorizonRecordId {
  std::uint32_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(HorizonRecordId, HorizonRecordId) noexcept = default;
};

/// @brief Identify one immutable hard-floor contribution.
struct HardFloorContributorId {
  std::uint32_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(HardFloorContributorId,
                                   HardFloorContributorId) noexcept = default;
};

/// @brief Identify one immutable hard-floor scenario.
struct HardFloorScenarioId {
  std::uint32_t value{};
  friend constexpr bool operator==(HardFloorScenarioId, HardFloorScenarioId) noexcept = default;
};

/// @brief Identify one group of mutually exclusive hard-floor alternatives.
struct HardFloorAlternativeGroupId {
  std::uint32_t value{};
  friend constexpr bool operator==(HardFloorAlternativeGroupId,
                                   HardFloorAlternativeGroupId) noexcept = default;
};

/// @brief Identify one immutable stable-pool capacity dependency.
struct PoolCapacityDependencyId {
  std::uint32_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(PoolCapacityDependencyId,
                                   PoolCapacityDependencyId) noexcept = default;
};

/// @brief Identify one immutable provider-allocation recipe.
struct ProviderAllocationPlanRecordId {
  std::uint32_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(ProviderAllocationPlanRecordId,
                                   ProviderAllocationPlanRecordId) noexcept = default;
};

/// @brief Identify one immutable stable-pool capacity scenario.
struct PoolCapacityScenarioId {
  std::uint32_t value{};
  friend constexpr bool operator==(PoolCapacityScenarioId,
                                   PoolCapacityScenarioId) noexcept = default;
};

/// @brief Identify one group of mutually exclusive stable-pool capacity alternatives.
struct PoolCapacityAlternativeGroupId {
  std::uint32_t value{};
  friend constexpr bool operator==(PoolCapacityAlternativeGroupId,
                                   PoolCapacityAlternativeGroupId) noexcept = default;
};

}  // namespace holoscan
