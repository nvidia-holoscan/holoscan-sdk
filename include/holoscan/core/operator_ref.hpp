/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace holoscan {

/// @brief Represent the nonzero identity of one compiled operator.
using OperatorInstanceId = std::uint32_t;

/// @brief Represent the nonzero compact provenance cookie of one immutable compiled plan.
///
/// The compiler derives this value deterministically from normalized Graph,
/// compile-policy, admission-evidence, and override semantics. It qualifies
/// plan-local numeric identities but is not a replacement for the full
/// cryptographic \ref ExecutionPlan artifact digest.
using PlanGenerationId = std::uint64_t;

/// @brief Represent the nonzero identity of one active run session.
using RunGenerationId = std::uint64_t;

/// @brief Represent one published immutable parameter-generation identity.
using ParamGeneration = std::uint64_t;

/// @brief Identify one compiled operator and, optionally, one active run.
///
/// A plan-bound reference has a zero run generation and carries no active
/// control authority. Runtime binds a separately authorized reference to a
/// nonzero run generation before run-scoped operations. Numeric identities are
/// fixed-size plan/run values, not object addresses or process-global names.
struct OperatorRef {
  OperatorInstanceId id{};             ///< Compiled operator identity; zero is invalid.
  PlanGenerationId plan_generation{};  ///< Immutable plan identity; zero is invalid.
  RunGenerationId run_generation{};    ///< Zero when plan-bound; nonzero when run-bound.

  /// @brief Return whether operator and plan identities are both nonzero.
  /// @return \c true for a plan-bound or run-bound reference with complete base identity.
  [[nodiscard]] constexpr bool valid() const noexcept { return id != 0 && plan_generation != 0; }

  /// @brief Return whether this value identifies only an immutable plan operator.
  /// @return \c true when \ref valid is true and \ref run_generation is zero.
  [[nodiscard]] constexpr bool plan_bound() const noexcept {
    return valid() && run_generation == 0;
  }

  /// @brief Return whether this value identifies an operator in one active run.
  /// @return \c true when \ref valid is true and \ref run_generation is nonzero.
  [[nodiscard]] constexpr bool run_bound() const noexcept { return valid() && run_generation != 0; }

  /// @brief Compare operator references by all plan and run identity fields.
  /// @param lhs Left operator reference.
  /// @param rhs Right operator reference.
  /// @return \c true when every plan/run identity field matches.
  friend constexpr bool operator==(OperatorRef lhs, OperatorRef rhs) noexcept = default;
};

static_assert(std::is_standard_layout_v<OperatorRef>);
static_assert(std::is_trivially_copyable_v<OperatorRef>);

}  // namespace holoscan
