/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/rate.hpp>

#include "rate_math_internal.hpp"

namespace holoscan {

std::optional<Rate> Rate::normalized() const noexcept {
  return detail::rate_math::normalize(*this);
}

bool Rate::equivalent_to(const Rate other) const noexcept {
  const auto left = detail::rate_math::normalize(*this);
  const auto right = detail::rate_math::normalize(other);
  return left && right && *left == *right;
}

std::optional<RateRatio> RateRatio::normalized() const noexcept {
  return detail::rate_math::normalize(*this);
}

bool RateRatio::equivalent_to(const RateRatio other) const noexcept {
  const auto left = detail::rate_math::normalize(*this);
  const auto right = detail::rate_math::normalize(other);
  return left && right && *left == *right;
}

}  // namespace holoscan
