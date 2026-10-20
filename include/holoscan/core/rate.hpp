/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <type_traits>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/operator_ref.hpp>

namespace holoscan {

/// @brief Represent an exact positive rational sample rate in samples per second.
///
/// Public construction and \c operator== preserve and compare the supplied representation. Use
/// \c normalized() or \c equivalent_to() when rational-value semantics are required. A zero
/// numerator or denominator is invalid and is rejected by semantic operations and cold authoring.
struct Rate {
  std::uint64_t numerator_hz{};     ///< Sample-rate numerator in hertz.
  std::uint64_t denominator_hz{1};  ///< Positive sample-rate denominator.

  /// @brief Construct one exact rational sample rate.
  /// @param numerator Numerator in samples per second.
  /// @param denominator Positive denominator.
  /// @return Unvalidated value normalized by the consuming cold authoring operation.
  [[nodiscard]] static constexpr Rate hz(std::uint64_t numerator,
                                         std::uint64_t denominator = 1U) noexcept {
    return Rate{numerator, denominator};
  }

  /// @brief Return this rational rate in lowest terms.
  /// @return Canonical positive rate, or \c std::nullopt when either component is zero.
  /// @note This operation is allocation-free.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT std::optional<Rate> normalized() const noexcept;

  /// @brief Compare this rate and another rate by exact rational value.
  /// @param other Rate to compare.
  /// @return \c true when both rates are valid and represent the same rational value.
  /// @note This operation is allocation-free. It does not change representation-level
  /// \c operator== semantics.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT bool equivalent_to(Rate other) const noexcept;

  friend constexpr bool operator==(const Rate&, const Rate&) noexcept = default;
};

/// @brief Represent an exact positive rational rate multiplier.
///
/// Public construction and \c operator== preserve and compare the supplied representation. Use
/// \c normalized() or \c equivalent_to() when rational-value semantics are required.
struct RateRatio {
  std::uint32_t numerator{1};    ///< Positive multiplier numerator.
  std::uint32_t denominator{1};  ///< Positive multiplier denominator.

  /// @brief Return this rational multiplier in lowest terms.
  /// @return Canonical positive ratio, or \c std::nullopt when either component is zero.
  /// @note This operation is allocation-free.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT std::optional<RateRatio> normalized() const noexcept;

  /// @brief Compare this ratio and another ratio by exact rational value.
  /// @param other Ratio to compare.
  /// @return \c true when both ratios are valid and represent the same rational value.
  /// @note This operation is allocation-free. It does not change representation-level
  /// \c operator== semantics.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT bool equivalent_to(RateRatio other) const noexcept;

  friend constexpr bool operator==(const RateRatio&, const RateRatio&) noexcept = default;
};

/// @brief Identify one private or atomically published rate/sizing candidate in a RunSession.
///
/// This identity changes when startup or restart evaluates a new candidate. It is deliberately
/// distinct from the stable RunGenerationId that identifies the containing active RunSession.
/// Runtime-generated values are nonzero and do not exceed 2^32 - 1 because generation-scoped
/// evidence record IDs reserve their upper 32 bits for this identity.
using RateGenerationId = std::uint64_t;

/// @brief Classify one authored port-rate declaration.
enum class PortRateKind : std::uint8_t {
  kUnspecified = 0,       ///< No rate declaration was authored.
  kFixedNominal = 1,      ///< Output has one fixed nominal rate.
  kExpectedInput = 2,     ///< Input carries a weaker expectation/fallback candidate.
  kFollowsInput = 3,      ///< Output derives from one owner-local input and exact ratio.
  kLifecycleNominal = 4,  ///< Output owner selects its value at one cold lifecycle barrier.
};

/// @brief Classify the root fact used to resolve a port rate for one run generation.
enum class ResolvedRateOrigin : std::uint8_t {
  kUnresolved = 0,
  kOutputNominal = 1,
  kConnectedOutput = 2,
  kFollowsExpression = 3,
  kInputExpectationFallback = 4,
  kLifecycleBinding = 5,
};

/// @brief Report whether a generation-scoped port-rate view is available.
enum class PortRateViewState : std::uint8_t {
  kPendingBarrier = 0,  ///< A selected lifecycle barrier has not closed globally.
  kResolved = 1,        ///< The generation contains a resolved value.
  kUnresolved = 2,      ///< Resolution completed without a usable value.
};

/// @brief Hold one small immutable generation-scoped rate result without graph traversal.
struct PortRateView {
  std::optional<Rate> resolved_nominal_rate;  ///< Resolved active nominal average.
  std::optional<Rate> sizing_rate;            ///< Exact rate used by sizing analysis.
  std::optional<Rate> max_nominal_rate;       ///< Propagated selectable-rate ceiling.
  ResolvedRateOrigin origin{ResolvedRateOrigin::kUnresolved};
  PortRateViewState state{PortRateViewState::kUnresolved};
  RunGenerationId run_generation{};    ///< Stable containing RunSession identity.
  RateGenerationId rate_generation{};  ///< Candidate or published rate-snapshot identity.

  friend bool operator==(const PortRateView&, const PortRateView&) noexcept = default;
};

static_assert(std::is_standard_layout_v<Rate>);
static_assert(std::is_trivially_copyable_v<Rate>);
static_assert(std::is_standard_layout_v<RateRatio>);
static_assert(std::is_trivially_copyable_v<RateRatio>);

}  // namespace holoscan
