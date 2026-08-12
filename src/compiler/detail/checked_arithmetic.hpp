/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace holoscan::detail {

template <std::unsigned_integral T, std::unsigned_integral L, std::unsigned_integral R>
[[nodiscard]] constexpr bool checked_add(L lhs, R rhs, T& result) noexcept {
  if (!std::in_range<T>(lhs) || !std::in_range<T>(rhs)) {
    return false;
  }
  const T checked_lhs = static_cast<T>(lhs);
  const T checked_rhs = static_cast<T>(rhs);
  if (checked_rhs > std::numeric_limits<T>::max() - checked_lhs) {
    return false;
  }
  result = checked_lhs + checked_rhs;
  return true;
}

template <std::unsigned_integral T, std::unsigned_integral L, std::unsigned_integral R>
[[nodiscard]] constexpr bool checked_multiply(L lhs, R rhs, T& result) noexcept {
  if (!std::in_range<T>(lhs) || !std::in_range<T>(rhs)) {
    return false;
  }
  const T checked_lhs = static_cast<T>(lhs);
  const T checked_rhs = static_cast<T>(rhs);
  if (checked_lhs != 0U && checked_rhs > std::numeric_limits<T>::max() / checked_lhs) {
    return false;
  }
  result = checked_lhs * checked_rhs;
  return true;
}

template <std::unsigned_integral T, std::unsigned_integral L, std::unsigned_integral R>
[[nodiscard]] constexpr bool checked_multiply_saturating(L lhs, R rhs, T& result) noexcept {
  if (!checked_multiply(lhs, rhs, result)) {
    result = std::numeric_limits<T>::max();
    return false;
  }
  return true;
}

[[nodiscard]] constexpr bool checked_size32(std::size_t value, std::uint32_t& result) noexcept {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  result = static_cast<std::uint32_t>(value);
  return true;
}

}  // namespace holoscan::detail
