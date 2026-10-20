/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "core/rate_math_internal.hpp"

namespace holoscan::detail::rate_math {
namespace {

TEST(RateMathInternal, NormalizesPositiveRatesAndRatios) {
  EXPECT_EQ(normalize(Rate{120U, 2U}), (Rate{60U, 1U}));
  EXPECT_EQ(normalize(RateRatio{6U, 8U}), (RateRatio{3U, 4U}));
  EXPECT_FALSE(normalize(Rate{0U, 1U}));
  EXPECT_FALSE(normalize(Rate{1U, 0U}));
  EXPECT_FALSE(normalize(RateRatio{0U, 1U}));
  EXPECT_FALSE(normalize(RateRatio{1U, 0U}));
}

TEST(RateMathInternal, MultipliesAfterCrossCancellationAndRejectsOverflow) {
  constexpr std::uint64_t kMaximum = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(multiply(Rate{kMaximum, 2U}, RateRatio{2U, 1U}), (Rate{kMaximum, 1U}));
  EXPECT_EQ(multiply(Rate{120U, 1U}, RateRatio{1U, 4U}), (Rate{30U, 1U}));
  EXPECT_FALSE(multiply(Rate{kMaximum, 1U}, RateRatio{2U, 1U}));
  EXPECT_FALSE(multiply(Rate{60U, 1U}, RateRatio{0U, 1U}));
}

TEST(RateMathInternal, ComparesRatesWithoutOverflow) {
  constexpr std::uint64_t kMaximum = std::numeric_limits<std::uint64_t>::max();
  EXPECT_TRUE(less_equal(Rate{kMaximum - 1U, kMaximum}, Rate{kMaximum, kMaximum}));
  EXPECT_TRUE(less_equal(Rate{60U, 1U}, Rate{120U, 2U}));
  EXPECT_FALSE(less_equal(Rate{kMaximum, 1U}, Rate{kMaximum - 1U, 1U}));
}

TEST(RateMathInternal, EvaluatesHorizonsArrivalsAndHeadroomExactly) {
  EXPECT_EQ(inverse_horizon(Rate{60U, 1U}),
            (ExactDuration{.numerator_ns = 50'000'000U, .denominator = 3U}));
  EXPECT_EQ(ceil_arrivals(Rate{90U, 1U},
                          ExactDuration{.numerator_ns = 1'000'000'000U, .denominator = 60U}),
            2U);
  EXPECT_EQ(ceil_arrivals(Rate{1'000U, 1U},
                          ExactDuration{.numerator_ns = 16'666'667U, .denominator = 1U}),
            17U);
  EXPECT_EQ(apply_headroom(34U, RateRatio{2U, 1U}), 68U);
  EXPECT_FALSE(inverse_horizon(Rate{0U, 1U}));
  EXPECT_FALSE(ceil_arrivals(Rate{60U, 1U}, ExactDuration{}));
  EXPECT_FALSE(apply_headroom(1U, RateRatio{0U, 1U}));
  EXPECT_FALSE(apply_headroom(std::numeric_limits<std::uint32_t>::max(), RateRatio{2U, 1U}));
}

TEST(RateMathInternal, RejectsMaximumWidthArrivalProductsThatCannotFitThePublicDepth) {
  constexpr std::uint64_t kMaximum = std::numeric_limits<std::uint64_t>::max();
  EXPECT_FALSE(ceil_arrivals(Rate{kMaximum, 1U},
                             ExactDuration{.numerator_ns = kMaximum, .denominator = 1U}));
  EXPECT_EQ(ceil_arrivals(Rate{kMaximum, kMaximum},
                          ExactDuration{.numerator_ns = 1'000'000'000U, .denominator = 1U}),
            1U);
  EXPECT_EQ(apply_headroom(3U, RateRatio{3U, 2U}), 5U);
}

}  // namespace
}  // namespace holoscan::detail::rate_math
