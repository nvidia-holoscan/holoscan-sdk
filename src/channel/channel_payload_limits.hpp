/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <optional>

namespace holoscan::channel_internal {

/// Maximum payload carried by the fixed HoloMQ inline-payload endpoint family.
inline constexpr std::size_t kMaximumInlinePayloadBytes = 32U * 1024U;

/// Return the fixed HoloMQ inline-payload bucket selected for one logical payload size.
[[nodiscard]] constexpr std::optional<std::size_t> inline_payload_bucket_size(
    const std::size_t payload_size) noexcept {
  if (payload_size == 0U) {
    return std::nullopt;
  }
  if (payload_size <= 8U) {
    return 8U;
  }
  if (payload_size <= 16U) {
    return 16U;
  }
  if (payload_size <= 32U) {
    return 32U;
  }
  if (payload_size <= 64U) {
    return 64U;
  }
  if (payload_size <= 128U) {
    return 128U;
  }
  if (payload_size <= 256U) {
    return 256U;
  }
  if (payload_size <= 512U) {
    return 512U;
  }
  if (payload_size <= 1024U) {
    return 1024U;
  }
  if (payload_size <= 2048U) {
    return 2048U;
  }
  if (payload_size <= 4096U) {
    return 4096U;
  }
  if (payload_size <= 8192U) {
    return 8192U;
  }
  if (payload_size <= 16384U) {
    return 16384U;
  }
  if (payload_size <= kMaximumInlinePayloadBytes) {
    return kMaximumInlinePayloadBytes;
  }
  return std::nullopt;
}

}  // namespace holoscan::channel_internal
