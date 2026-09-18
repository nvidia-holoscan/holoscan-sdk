/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure swapchain-parameter helpers, split out of vk_swapchain.cpp so they are unit-testable
// without a device or a display.

#pragma once

#include <algorithm>
#include <cstdint>

namespace holoviz::vulkan {

/// Clamp a requested swapchain image count to a surface's reported bounds.
/// \p max_count follows the Vulkan convention: 0 = no upper limit.
inline uint32_t clamp_image_count(uint32_t requested, uint32_t min_count, uint32_t max_count) {
  uint32_t clamped = std::max(requested, min_count);
  if (max_count != 0) {
    clamped = std::min(clamped, max_count);
  }
  return clamped;
}

/// Clamp one extent dimension to a surface's [min, max] bounds.
inline uint32_t clamp_extent_dim(uint32_t requested, uint32_t min_dim, uint32_t max_dim) {
  return std::clamp(requested, min_dim, max_dim);
}

}  // namespace holoviz::vulkan
