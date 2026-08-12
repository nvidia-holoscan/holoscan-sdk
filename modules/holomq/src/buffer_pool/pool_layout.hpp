/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace holomq {

/// Returns \p value rounded up to a multiple of \p alignment. \p alignment must be greater than
/// zero.
[[nodiscard]] constexpr std::size_t round_up(const std::size_t value,
                                             const std::size_t alignment) noexcept {
  return (value + alignment - 1U) / alignment * alignment;
}

/// Returns \p lhs * \p rhs when the product fits in \c std::size_t; empty on overflow.
[[nodiscard]] constexpr std::optional<std::size_t> checked_product(const std::size_t lhs,
                                                                   const std::size_t rhs) noexcept {
  if (lhs != 0U && rhs > (std::numeric_limits<std::size_t>::max() / lhs)) {
    return std::nullopt;
  }
  return lhs * rhs;
}

}  // namespace holomq
