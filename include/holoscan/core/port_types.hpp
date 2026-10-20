/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace holoscan {

/// @brief Identify whether a logical port receives or publishes samples.
enum class PortDirection : std::uint8_t {
  kInput,   ///< Runtime selects samples for the Operator to inspect.
  kOutput,  ///< The Operator publishes samples to Runtime.
};

/// @brief Identify one logical port within a compiled execution plan.
///
/// The nonzero numeric representation is plan-local and carries no ordering, address, schema, or
/// transport meaning. Zero is invalid. Compilation assigns the final value after canonical port
/// ordering; an authoring port member does not manufacture one.
class PortId {
 public:
  /// @brief Construct an invalid port identity.
  constexpr PortId() noexcept = default;

  /// @brief Construct a plan-local port identity from its frozen representation.
  /// @param value Nonzero compiler-assigned value, or zero for an invalid value.
  explicit constexpr PortId(std::uint64_t value) noexcept : value_(value) {}

  /// @brief Return the fixed-width plan-local representation.
  /// @return Numeric representation, with zero denoting invalid.
  [[nodiscard]] constexpr std::uint64_t value() const noexcept { return value_; }

  /// @brief Return whether this identity is nonzero.
  /// @return \c true for a compiler-assigned value; otherwise \c false.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value_ != 0; }

  /// @brief Compare complete plan-local identities.
  /// @param lhs Left identity.
  /// @param rhs Right identity.
  /// @return \c true when both numeric representations match.
  friend constexpr bool operator==(PortId lhs, PortId rhs) noexcept = default;

 private:
  std::uint64_t value_{};
};

}  // namespace holoscan
