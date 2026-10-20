/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/rate.hpp>

namespace holoscan::detail::rate_math {
namespace arithmetic {

// A source-private, portable 128-bit unsigned value. Rate evaluation is cold-path work, so the
// bounded restoring division below favors exact cross-platform behavior over compiler extensions.
struct UInt128 {
  std::uint64_t high{};
  std::uint64_t low{};
};

[[nodiscard]] constexpr bool is_zero(UInt128 value) noexcept {
  return value.high == 0U && value.low == 0U;
}

[[nodiscard]] constexpr int compare(UInt128 left, UInt128 right) noexcept {
  if (left.high != right.high) {
    return left.high < right.high ? -1 : 1;
  }
  if (left.low != right.low) {
    return left.low < right.low ? -1 : 1;
  }
  return 0;
}

[[nodiscard]] constexpr UInt128 subtract(UInt128 left, UInt128 right) noexcept {
  const std::uint64_t borrow = left.low < right.low ? 1U : 0U;
  return UInt128{.high = left.high - right.high - borrow, .low = left.low - right.low};
}

[[nodiscard]] constexpr UInt128 multiply(std::uint64_t left, std::uint64_t right) noexcept {
  constexpr std::uint64_t kLow32 = 0xFFFF'FFFFULL;
  const std::uint64_t left_low = left & kLow32;
  const std::uint64_t left_high = left >> 32U;
  const std::uint64_t right_low = right & kLow32;
  const std::uint64_t right_high = right >> 32U;

  std::uint64_t work = left_low * right_low;
  const std::uint64_t result_low_low = work & kLow32;
  std::uint64_t carry = work >> 32U;

  work = left_high * right_low + carry;
  const std::uint64_t result_high_low = work & kLow32;
  const std::uint64_t result_high_high = work >> 32U;

  work = left_low * right_high + result_high_low;
  const std::uint64_t result_low_high = work & kLow32;
  carry = work >> 32U;

  return UInt128{.high = left_high * right_high + result_high_high + carry,
                 .low = (result_low_high << 32U) | result_low_low};
}

[[nodiscard]] constexpr std::optional<UInt128> multiply(UInt128 left,
                                                        std::uint64_t right) noexcept {
  const UInt128 low_product = multiply(left.low, right);
  const UInt128 high_product = multiply(left.high, right);
  if (high_product.high != 0U ||
      high_product.low > std::numeric_limits<std::uint64_t>::max() - low_product.high) {
    return std::nullopt;
  }
  return UInt128{.high = low_product.high + high_product.low, .low = low_product.low};
}

struct DivisionResult {
  UInt128 quotient;
  UInt128 remainder;
};

[[nodiscard]] constexpr bool bit(UInt128 value, unsigned index) noexcept {
  return index < 64U ? ((value.low >> index) & 1U) != 0U
                     : ((value.high >> (index - 64U)) & 1U) != 0U;
}

constexpr void set_bit(UInt128& value, unsigned index) noexcept {
  if (index < 64U) {
    value.low |= std::uint64_t{1U} << index;
  } else {
    value.high |= std::uint64_t{1U} << (index - 64U);
  }
}

[[nodiscard]] constexpr DivisionResult divide(UInt128 dividend, UInt128 divisor) noexcept {
  DivisionResult result{};
  if (is_zero(divisor)) {
    return result;
  }
  for (unsigned remaining = 128U; remaining != 0U; --remaining) {
    const unsigned index = remaining - 1U;
    const bool carry = (result.remainder.high >> 63U) != 0U;
    result.remainder.high = (result.remainder.high << 1U) | (result.remainder.low >> 63U);
    result.remainder.low = (result.remainder.low << 1U) | (bit(dividend, index) ? 1U : 0U);
    if (carry || compare(result.remainder, divisor) >= 0) {
      result.remainder = subtract(result.remainder, divisor);
      set_bit(result.quotient, index);
    }
  }
  return result;
}

}  // namespace arithmetic

[[nodiscard]] constexpr std::optional<Rate> normalize(Rate value) noexcept {
  if (value.numerator_hz == 0U || value.denominator_hz == 0U) {
    return std::nullopt;
  }
  const std::uint64_t divisor = std::gcd(value.numerator_hz, value.denominator_hz);
  return Rate{value.numerator_hz / divisor, value.denominator_hz / divisor};
}

[[nodiscard]] constexpr std::optional<RateRatio> normalize(RateRatio value) noexcept {
  if (value.numerator == 0U || value.denominator == 0U) {
    return std::nullopt;
  }
  const std::uint32_t divisor = std::gcd(value.numerator, value.denominator);
  return RateRatio{value.numerator / divisor, value.denominator / divisor};
}

[[nodiscard]] constexpr std::optional<Rate> multiply(Rate value, RateRatio ratio) noexcept {
  const auto normalized_rate = normalize(value);
  const auto normalized_ratio = normalize(ratio);
  if (!normalized_rate || !normalized_ratio) {
    return std::nullopt;
  }
  std::uint64_t rate_numerator = normalized_rate->numerator_hz;
  std::uint64_t rate_denominator = normalized_rate->denominator_hz;
  std::uint64_t ratio_numerator = normalized_ratio->numerator;
  std::uint64_t ratio_denominator = normalized_ratio->denominator;

  const std::uint64_t left = std::gcd(rate_numerator, ratio_denominator);
  rate_numerator /= left;
  ratio_denominator /= left;
  const std::uint64_t right = std::gcd(ratio_numerator, rate_denominator);
  ratio_numerator /= right;
  rate_denominator /= right;
  if (rate_numerator > std::numeric_limits<std::uint64_t>::max() / ratio_numerator ||
      rate_denominator > std::numeric_limits<std::uint64_t>::max() / ratio_denominator) {
    return std::nullopt;
  }
  return normalize(Rate{rate_numerator * ratio_numerator, rate_denominator * ratio_denominator});
}

[[nodiscard]] constexpr bool less_equal(Rate left, Rate right) noexcept {
  const auto normalized_left = normalize(left);
  const auto normalized_right = normalize(right);
  if (!normalized_left || !normalized_right) {
    return false;
  }
  return arithmetic::compare(
             arithmetic::multiply(normalized_left->numerator_hz, normalized_right->denominator_hz),
             arithmetic::multiply(normalized_right->numerator_hz,
                                  normalized_left->denominator_hz)) <= 0;
}

[[nodiscard]] constexpr std::optional<ExactDuration> inverse_horizon(Rate value) noexcept {
  const auto rate = normalize(value);
  if (!rate) {
    return std::nullopt;
  }
  std::uint64_t duration_numerator = rate->denominator_hz;
  std::uint64_t nanoseconds_per_second = 1'000'000'000U;
  std::uint64_t duration_denominator = rate->numerator_hz;
  const std::uint64_t scale_divisor = std::gcd(nanoseconds_per_second, duration_denominator);
  nanoseconds_per_second /= scale_divisor;
  duration_denominator /= scale_divisor;
  const std::uint64_t rate_divisor = std::gcd(duration_numerator, duration_denominator);
  duration_numerator /= rate_divisor;
  duration_denominator /= rate_divisor;
  if (duration_numerator > std::numeric_limits<std::uint64_t>::max() / nanoseconds_per_second) {
    return std::nullopt;
  }
  return ExactDuration{.numerator_ns = duration_numerator * nanoseconds_per_second,
                       .denominator = duration_denominator};
}

[[nodiscard]] constexpr std::optional<std::uint32_t> ceil_arrivals(Rate value,
                                                                   ExactDuration horizon) noexcept {
  const auto rate = normalize(value);
  if (!rate || horizon.numerator_ns == 0U || horizon.denominator == 0U) {
    return std::nullopt;
  }
  std::uint64_t rate_numerator = rate->numerator_hz;
  std::uint64_t rate_denominator = rate->denominator_hz;
  std::uint64_t duration_numerator = horizon.numerator_ns;
  std::uint64_t duration_denominator = horizon.denominator;
  std::uint64_t nanoseconds_per_second = 1'000'000'000U;
  const auto cancel = [](std::uint64_t& numerator, std::uint64_t& denominator) {
    const std::uint64_t divisor = std::gcd(numerator, denominator);
    numerator /= divisor;
    denominator /= divisor;
  };
  cancel(rate_numerator, rate_denominator);
  cancel(rate_numerator, duration_denominator);
  cancel(rate_numerator, nanoseconds_per_second);
  cancel(duration_numerator, rate_denominator);
  cancel(duration_numerator, duration_denominator);
  cancel(duration_numerator, nanoseconds_per_second);

  const arithmetic::UInt128 numerator = arithmetic::multiply(rate_numerator, duration_numerator);
  const arithmetic::UInt128 partial_denominator =
      arithmetic::multiply(rate_denominator, duration_denominator);
  const auto denominator = arithmetic::multiply(partial_denominator, nanoseconds_per_second);
  if (!denominator || arithmetic::is_zero(*denominator)) {
    return std::nullopt;
  }
  auto divided = arithmetic::divide(numerator, *denominator);
  if (!arithmetic::is_zero(divided.remainder)) {
    ++divided.quotient.low;
    if (divided.quotient.low == 0U) {
      ++divided.quotient.high;
    }
  }
  if (divided.quotient.high != 0U || divided.quotient.low == 0U ||
      divided.quotient.low > std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(divided.quotient.low);
}

[[nodiscard]] constexpr std::optional<std::uint32_t> apply_headroom(std::uint32_t arrivals,
                                                                    RateRatio value) noexcept {
  const auto headroom = normalize(value);
  if (!headroom) {
    return std::nullopt;
  }
  const std::uint64_t numerator = static_cast<std::uint64_t>(arrivals) * headroom->numerator;
  const std::uint64_t rounded =
      numerator / headroom->denominator + (numerator % headroom->denominator != 0U ? 1U : 0U);
  if (rounded == 0U || rounded > std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(rounded);
}

}  // namespace holoscan::detail::rate_math
