/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <optional>

namespace holoscan::channel_internal {

/// Maximum payload carried by one HoloMQ inline-payload frame.
inline constexpr std::size_t kMaximumInlinePayloadBytes = 32U * 1024U;

/// @brief Return the payload capacity of an inline-payload service for one payload bound.
///
/// Services are sized exactly: the capacity is the bound itself, so no size-class rounding
/// enlarges the preallocated sample pool.
/// @param payload_size Largest payload the service carries.
/// @return \p payload_size, or \c std::nullopt when it is zero or above
/// \ref kMaximumInlinePayloadBytes.
[[nodiscard]] constexpr std::optional<std::size_t> inline_payload_capacity(
    const std::size_t payload_size) noexcept {
  if (payload_size == 0U || payload_size > kMaximumInlinePayloadBytes) {
    return std::nullopt;
  }
  return payload_size;
}

}  // namespace holoscan::channel_internal
