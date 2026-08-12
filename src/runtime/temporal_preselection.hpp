/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/time/clock_registry.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan::detail {

enum class ReadinessWitnessDisposition : std::uint8_t {
  kNoWitness,
  kAccepted,
  kCoalesced,
};

struct ReadinessWitnessDecision {
  ReadinessWitnessDisposition disposition{ReadinessWitnessDisposition::kNoWitness};
  std::uint64_t source_sequence{};
  std::uint64_t skipped_doorbell_tokens{};
};

// This transport-neutral state owns only the finite source credit and source-specific acceptance
// state. Adapter binding, reference-time capture, Dispatcher ingress, and credit release after full
// activation retirement remain later runtime responsibilities.
class ReadinessPreselectionState {
 public:
  ReadinessPreselectionState(const ReadinessPreselectionState&) = delete;
  ReadinessPreselectionState& operator=(const ReadinessPreselectionState&) = delete;
  ReadinessPreselectionState(ReadinessPreselectionState&&) noexcept = default;
  ReadinessPreselectionState& operator=(ReadinessPreselectionState&&) noexcept = default;
  ~ReadinessPreselectionState() = default;

  [[nodiscard]] static expected<ReadinessPreselectionState, Error> create(
      ReadinessSourceKind kind, std::uint32_t capacity) noexcept {
    if (capacity == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    switch (kind) {
      case ReadinessSourceKind::kReadable:
      case ReadinessSourceKind::kNotification:
      case ReadinessSourceKind::kDoorbell:
      case ReadinessSourceKind::kSpin:
        break;
      default:
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if ((kind == ReadinessSourceKind::kReadable || kind == ReadinessSourceKind::kSpin) &&
        capacity != 1U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return ReadinessPreselectionState{kind, capacity};
  }

  [[nodiscard]] ReadinessSourceKind kind() const noexcept { return kind_; }
  [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::uint32_t outstanding() const noexcept { return outstanding_; }
  [[nodiscard]] std::uint64_t last_source_sequence() const noexcept {
    return last_source_sequence_;
  }
  [[nodiscard]] std::uint64_t accepted_doorbell_token() const noexcept {
    return accepted_doorbell_token_;
  }

  [[nodiscard]] expected<ReadinessWitnessDecision, Error> observe_readable(bool readable) noexcept {
    if (kind_ != ReadinessSourceKind::kReadable) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!readable || outstanding_ != 0U) {
      return no_witness();
    }
    return accept_one();
  }

  [[nodiscard]] expected<ReadinessWitnessDecision, Error> post_notification() noexcept {
    if (kind_ != ReadinessSourceKind::kNotification) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return accept_one();
  }

  [[nodiscard]] expected<ReadinessWitnessDecision, Error> observe_doorbell(
      std::uint64_t token) noexcept {
    if (kind_ != ReadinessSourceKind::kDoorbell || token == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (token < accepted_doorbell_token_) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (token == accepted_doorbell_token_) {
      return ReadinessWitnessDecision{
          .disposition = ReadinessWitnessDisposition::kCoalesced,
      };
    }

    const std::uint64_t skipped = token - accepted_doorbell_token_ - 1U;
    auto accepted = accept_one();
    if (!accepted) {
      return make_unexpected(accepted.error());
    }
    accepted_doorbell_token_ = token;
    accepted->skipped_doorbell_tokens = skipped;
    return accepted;
  }

  [[nodiscard]] expected<ReadinessWitnessDecision, Error> observe_spin(bool value) noexcept {
    if (kind_ != ReadinessSourceKind::kSpin) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!value) {
      spin_observed_true_ = false;
      return no_witness();
    }
    if (spin_observed_true_) {
      return no_witness();
    }

    // The edge is an observation even when its finite credit cannot be reserved. Remembering true
    // prevents repeated polling of one continuously true signal from manufacturing new edges.
    spin_observed_true_ = true;
    return accept_one();
  }

  [[nodiscard]] expected<void, Error> retire_one() noexcept {
    if (outstanding_ == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    --outstanding_;
    return {};
  }

 private:
  ReadinessPreselectionState(ReadinessSourceKind kind, std::uint32_t capacity) noexcept
      : kind_(kind), capacity_(capacity) {}

  [[nodiscard]] static constexpr ReadinessWitnessDecision no_witness() noexcept { return {}; }

  [[nodiscard]] expected<ReadinessWitnessDecision, Error> accept_one() noexcept {
    if (outstanding_ == capacity_) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (last_source_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    ++outstanding_;
    ++last_source_sequence_;
    return ReadinessWitnessDecision{
        .disposition = ReadinessWitnessDisposition::kAccepted,
        .source_sequence = last_source_sequence_,
    };
  }

  ReadinessSourceKind kind_;
  std::uint32_t capacity_{};
  std::uint32_t outstanding_{};
  std::uint64_t last_source_sequence_{};
  std::uint64_t accepted_doorbell_token_{};
  bool spin_observed_true_{};
};

// The compiler admits this edge only after the source, target, and every TimePoint supplied to the
// conversion seam have been rebound into one validated immutable plan/clock-catalog scope. This
// compact record cannot detect a foreign ClockId that happens to collide numerically.
struct DirectAffineClockEdge {
  ClockId source;
  ClockId target;
  AffineClockConversion conversion;
};

namespace temporal_preselection_internal {

struct UInt128 {
  std::uint64_t high{};
  std::uint64_t low{};
};

[[nodiscard]] constexpr UInt128 multiply_wide(std::uint64_t lhs, std::uint64_t rhs) noexcept {
  constexpr std::uint64_t kLowMask = 0xffffffffULL;
  const std::uint64_t lhs_high = lhs >> 32U;
  const std::uint64_t lhs_low = lhs & kLowMask;
  const std::uint64_t rhs_high = rhs >> 32U;
  const std::uint64_t rhs_low = rhs & kLowMask;

  std::uint64_t partial = lhs_low * rhs_low;
  const std::uint64_t low_word = partial & kLowMask;
  std::uint64_t carry = partial >> 32U;

  partial = lhs_high * rhs_low + carry;
  const std::uint64_t middle_low = partial & kLowMask;
  const std::uint64_t middle_high = partial >> 32U;

  partial = lhs_low * rhs_high + middle_low;
  return UInt128{
      .high = lhs_high * rhs_high + middle_high + (partial >> 32U),
      .low = (partial << 32U) + low_word,
  };
}

struct DivModResult {
  std::uint64_t quotient{};
  std::uint64_t remainder{};
};

[[nodiscard]] inline expected<DivModResult, Error> divide_wide(UInt128 dividend,
                                                               std::uint64_t divisor) noexcept {
  if (divisor == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (dividend.high >= divisor) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  std::uint64_t quotient = 0U;
  std::uint64_t remainder = dividend.high;
  for (std::uint32_t index = 64U; index-- > 0U;) {
    const bool carry = (remainder >> 63U) != 0U;
    remainder = (remainder << 1U) | ((dividend.low >> index) & 1U);
    if (carry || remainder >= divisor) {
      remainder -= divisor;
      quotient |= std::uint64_t{1} << index;
    }
  }
  return DivModResult{.quotient = quotient, .remainder = remainder};
}

[[nodiscard]] constexpr std::uint64_t magnitude(std::int64_t value) noexcept {
  return value >= 0 ? static_cast<std::uint64_t>(value)
                    : static_cast<std::uint64_t>(-(value + 1)) + 1U;
}

[[nodiscard]] constexpr bool less_than(UInt128 lhs, UInt128 rhs) noexcept {
  return lhs.high < rhs.high || (lhs.high == rhs.high && lhs.low < rhs.low);
}

[[nodiscard]] inline expected<UInt128, Error> add_wide(UInt128 lhs, UInt128 rhs) noexcept {
  const std::uint64_t low = lhs.low + rhs.low;
  const std::uint64_t carry = low < lhs.low ? 1U : 0U;
  const std::uint64_t partial_high = lhs.high + rhs.high;
  if (partial_high < lhs.high) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const std::uint64_t high = partial_high + carry;
  if (high < partial_high) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return UInt128{.high = high, .low = low};
}

[[nodiscard]] constexpr UInt128 subtract_wide(UInt128 lhs, UInt128 rhs) noexcept {
  const std::uint64_t borrow = lhs.low < rhs.low ? 1U : 0U;
  return UInt128{.high = lhs.high - rhs.high - borrow, .low = lhs.low - rhs.low};
}

struct SignedUInt128 {
  bool negative{};
  UInt128 magnitude{};
};

struct UInt192 {
  std::uint64_t high{};
  std::uint64_t middle{};
  std::uint64_t low{};
};

[[nodiscard]] constexpr UInt192 multiply_wider(UInt128 lhs, std::uint64_t rhs) noexcept {
  const UInt128 low_product = multiply_wide(lhs.low, rhs);
  const UInt128 high_product = multiply_wide(lhs.high, rhs);
  const std::uint64_t middle = low_product.high + high_product.low;
  const std::uint64_t carry = middle < low_product.high ? 1U : 0U;
  return UInt192{
      .high = high_product.high + carry,
      .middle = middle,
      .low = low_product.low,
  };
}

[[nodiscard]] constexpr int compare_wider(UInt192 lhs, UInt192 rhs) noexcept {
  if (lhs.high != rhs.high) {
    return lhs.high < rhs.high ? -1 : 1;
  }
  if (lhs.middle != rhs.middle) {
    return lhs.middle < rhs.middle ? -1 : 1;
  }
  if (lhs.low != rhs.low) {
    return lhs.low < rhs.low ? -1 : 1;
  }
  return 0;
}

[[nodiscard]] constexpr SignedUInt128 signed_product(std::int64_t value,
                                                     std::uint64_t factor) noexcept {
  const UInt128 product = multiply_wide(magnitude(value), factor);
  return SignedUInt128{
      .negative = value < 0 && (product.high != 0U || product.low != 0U),
      .magnitude = product,
  };
}

[[nodiscard]] inline expected<SignedUInt128, Error> add_signed(SignedUInt128 lhs,
                                                               SignedUInt128 rhs) noexcept {
  if (lhs.negative == rhs.negative) {
    auto sum = add_wide(lhs.magnitude, rhs.magnitude);
    if (!sum) {
      return make_unexpected(sum.error());
    }
    return SignedUInt128{.negative = lhs.negative, .magnitude = *sum};
  }
  if (lhs.magnitude.high == rhs.magnitude.high && lhs.magnitude.low == rhs.magnitude.low) {
    return SignedUInt128{};
  }
  if (less_than(lhs.magnitude, rhs.magnitude)) {
    return SignedUInt128{
        .negative = rhs.negative,
        .magnitude = subtract_wide(rhs.magnitude, lhs.magnitude),
    };
  }
  return SignedUInt128{
      .negative = lhs.negative,
      .magnitude = subtract_wide(lhs.magnitude, rhs.magnitude),
  };
}

[[nodiscard]] inline expected<SignedUInt128, Error> exact_affine_numerator(
    std::int64_t value, AffineClockConversion conversion) noexcept {
  if (conversion.scale.numerator <= 0 || conversion.scale.denominator == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  auto exact_numerator =
      add_signed(signed_product(value, static_cast<std::uint64_t>(conversion.scale.numerator)),
                 signed_product(conversion.offset_ns, conversion.scale.denominator));
  if (!exact_numerator) {
    return make_unexpected(exact_numerator.error());
  }
  return exact_numerator;
}

[[nodiscard]] inline expected<std::int64_t, Error> materialize_affine_coordinate(
    SignedUInt128 exact_numerator, std::uint64_t denominator) noexcept {
  auto divided = divide_wide(exact_numerator.magnitude, denominator);
  if (!divided) {
    return make_unexpected(divided.error());
  }

  const std::uint64_t half = denominator / 2U;
  const bool above_half = divided->remainder > half;
  const bool exact_half = denominator % 2U == 0U && divided->remainder == half;
  if (above_half || (exact_half && (divided->quotient & 1U) != 0U)) {
    if (divided->quotient == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    ++divided->quotient;
  }

  constexpr std::uint64_t kSignedMinimumMagnitude = std::uint64_t{1} << 63U;
  if (exact_numerator.negative) {
    if (divided->quotient > kSignedMinimumMagnitude) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (divided->quotient == kSignedMinimumMagnitude) {
      return std::numeric_limits<std::int64_t>::min();
    }
    return -static_cast<std::int64_t>(divided->quotient);
  }
  if (divided->quotient > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return static_cast<std::int64_t>(divided->quotient);
}

}  // namespace temporal_preselection_internal

struct ExactAffineCoordinate {
  ClockId target_clock;
  temporal_preselection_internal::SignedUInt128 numerator;
  std::uint64_t denominator{};
};

enum class ExactCoordinateOrder : std::int8_t {
  kLess = -1,
  kEqual = 0,
  kGreater = 1,
};

// This is deliberately a one-edge exact-coordinate hook. It never searches a registry, infers a
// reverse edge, composes a path, or rounds a coordinate before comparison.
[[nodiscard]] inline expected<ExactAffineCoordinate, Error> convert_identical_or_direct_exact(
    TimePoint point, ClockId target, const DirectAffineClockEdge* direct_edge) noexcept {
  using temporal_preselection_internal::signed_product;
  using temporal_preselection_internal::SignedUInt128;

  if (!point.valid() || !target.valid()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (point.clock_id == target) {
    return ExactAffineCoordinate{
        .target_clock = target,
        .numerator = signed_product(point.timestamp_ns, 1U),
        .denominator = 1U,
    };
  }
  if (direct_edge == nullptr || direct_edge->source != point.clock_id ||
      direct_edge->target != target) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  auto numerator = temporal_preselection_internal::exact_affine_numerator(point.timestamp_ns,
                                                                          direct_edge->conversion);
  if (!numerator) {
    return make_unexpected(numerator.error());
  }
  return ExactAffineCoordinate{
      .target_clock = target,
      .numerator = *numerator,
      .denominator = direct_edge->conversion.scale.denominator,
  };
}

[[nodiscard]] inline expected<ExactCoordinateOrder, Error> compare_exact_coordinates(
    const ExactAffineCoordinate& lhs, const ExactAffineCoordinate& rhs) noexcept {
  using temporal_preselection_internal::compare_wider;
  using temporal_preselection_internal::multiply_wider;

  if (!lhs.target_clock.valid() || lhs.target_clock != rhs.target_clock || lhs.denominator == 0U ||
      rhs.denominator == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (lhs.numerator.negative != rhs.numerator.negative) {
    return lhs.numerator.negative ? ExactCoordinateOrder::kLess : ExactCoordinateOrder::kGreater;
  }
  const int magnitude_order =
      compare_wider(multiply_wider(lhs.numerator.magnitude, rhs.denominator),
                    multiply_wider(rhs.numerator.magnitude, lhs.denominator));
  if (magnitude_order == 0) {
    return ExactCoordinateOrder::kEqual;
  }
  const bool lhs_magnitude_less = magnitude_order < 0;
  const bool lhs_less = lhs.numerator.negative ? !lhs_magnitude_less : lhs_magnitude_less;
  return lhs_less ? ExactCoordinateOrder::kLess : ExactCoordinateOrder::kGreater;
}

[[nodiscard]] inline expected<bool, Error> exact_interval_within_inclusive_window(
    const ExactAffineCoordinate& low, const ExactAffineCoordinate& high,
    std::chrono::nanoseconds window) noexcept {
  if (window.count() < 0) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto order = compare_exact_coordinates(low, high);
  if (!order || *order == ExactCoordinateOrder::kGreater) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto bound_numerator = temporal_preselection_internal::add_signed(
      low.numerator,
      temporal_preselection_internal::signed_product(window.count(), low.denominator));
  if (!bound_numerator) {
    return make_unexpected(bound_numerator.error());
  }
  const ExactAffineCoordinate bound{
      .target_clock = low.target_clock,
      .numerator = *bound_numerator,
      .denominator = low.denominator,
  };
  auto high_to_bound = compare_exact_coordinates(high, bound);
  if (!high_to_bound) {
    return make_unexpected(high_to_bound.error());
  }
  return *high_to_bound != ExactCoordinateOrder::kGreater;
}

[[nodiscard]] inline expected<TimePoint, Error> materialize_exact_coordinate(
    const ExactAffineCoordinate& coordinate) noexcept {
  if (!coordinate.target_clock.valid() || coordinate.denominator == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto timestamp = temporal_preselection_internal::materialize_affine_coordinate(
      coordinate.numerator, coordinate.denominator);
  if (!timestamp) {
    return make_unexpected(timestamp.error());
  }
  return TimePoint{.timestamp_ns = *timestamp, .clock_id = coordinate.target_clock};
}

// This convenience hook materializes an integer TimePoint only. Timing selectors must compare the
// exact coordinates above and round only after an ordering or inclusive-boundary decision.
[[nodiscard]] inline expected<TimePoint, Error> convert_identical_or_direct(
    TimePoint point, ClockId target, const DirectAffineClockEdge* direct_edge) noexcept {
  auto exact = convert_identical_or_direct_exact(point, target, direct_edge);
  if (!exact) {
    return make_unexpected(exact.error());
  }
  return materialize_exact_coordinate(*exact);
}

enum class EndpointHealthState : std::uint8_t {
  kNotRequired,
  kUnknown,
  kFreshHealthy,
  kFreshDegraded,
  kFreshUnhealthy,
  kStale,
};

enum class EndpointHealthAction : std::uint8_t {
  kAdmit,
  kDegrade,
  kReject,
};

struct EndpointHealthDisposition {
  EndpointHealthState state{EndpointHealthState::kNotRequired};
  EndpointHealthAction action{EndpointHealthAction::kAdmit};
};

enum class HealthPreselectionResult : std::uint8_t {
  kNotEligible,
  kAdmitted,
  kDegraded,
  kRejected,
};

[[nodiscard]] inline expected<HealthPreselectionResult, Error> apply_endpoint_health(
    bool already_eligible, EndpointHealthDisposition disposition) noexcept {
  switch (disposition.state) {
    case EndpointHealthState::kNotRequired:
    case EndpointHealthState::kUnknown:
    case EndpointHealthState::kFreshHealthy:
    case EndpointHealthState::kFreshDegraded:
    case EndpointHealthState::kFreshUnhealthy:
    case EndpointHealthState::kStale:
      break;
    default:
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  switch (disposition.action) {
    case EndpointHealthAction::kAdmit:
    case EndpointHealthAction::kDegrade:
    case EndpointHealthAction::kReject:
      break;
    default:
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  if (!already_eligible) {
    return HealthPreselectionResult::kNotEligible;
  }
  switch (disposition.action) {
    case EndpointHealthAction::kAdmit:
      return HealthPreselectionResult::kAdmitted;
    case EndpointHealthAction::kDegrade:
      return HealthPreselectionResult::kDegraded;
    case EndpointHealthAction::kReject:
      return HealthPreselectionResult::kRejected;
    default:
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
}

static_assert(std::is_nothrow_move_constructible_v<ReadinessPreselectionState>);
static_assert(std::is_trivially_destructible_v<ReadinessPreselectionState>);
static_assert(std::is_trivially_copyable_v<ReadinessWitnessDecision>);
static_assert(std::is_trivially_copyable_v<DirectAffineClockEdge>);
static_assert(std::is_trivially_copyable_v<EndpointHealthDisposition>);

}  // namespace holoscan::detail
