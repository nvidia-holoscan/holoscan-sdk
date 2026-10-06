/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure swapchain-parameter helpers, split out of vk_swapchain.cpp so they are unit-testable
// without a device or a display.

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>

namespace holoviz::vulkan {

/// One surface-supported (format, color space) pairing, as raw Vulkan enum values so this
/// header stays Vulkan-free (pattern: vk_display_util.hpp).
struct SurfaceFormatPair {
  uint64_t format = 0;
  uint64_t color_space = 0;
};

/// The pairing select_surface_format() settled on, plus how far it had to fall.
struct SurfaceFormatChoice {
  uint64_t format = 0;
  uint64_t color_space = 0;
  /// False when the requested format was found but only with a substituted color space.
  bool exact_color_space = false;
  /// True when the requested format was unavailable and a fallback format was substituted.
  bool fallback_format = false;
};

/// Select a swapchain (format, color space) from what the surface offers:
///  1. the exact requested pairing;
///  2. the requested format with a substituted color space (first offered — the caller
///     warns with the substitute's raw value);
///  3. a preference-ordered fallback format from \p fallback_formats, keeping the
///     requested color space when that pairing exists, else the format's first offered
///     color space (the caller warns);
///  4. nullopt when nothing usable exists.
inline std::optional<SurfaceFormatChoice> select_surface_format(
    std::span<const SurfaceFormatPair> available, uint64_t desired_format,
    uint64_t desired_color_space, std::span<const uint64_t> fallback_formats) {
  const auto first_with_format =
      [&available](uint64_t format,
                   uint64_t preferred_color_space) -> std::optional<SurfaceFormatPair> {
    std::optional<SurfaceFormatPair> first;
    for (const SurfaceFormatPair& pair : available) {
      if (pair.format != format) {
        continue;
      }
      if (pair.color_space == preferred_color_space) {
        return pair;
      }
      if (!first) {
        first = pair;
      }
    }
    return first;
  };

  if (const auto pair = first_with_format(desired_format, desired_color_space)) {
    return SurfaceFormatChoice{.format = pair->format,
                               .color_space = pair->color_space,
                               .exact_color_space = pair->color_space == desired_color_space,
                               .fallback_format = false};
  }

  for (const uint64_t fallback : fallback_formats) {
    if (const auto pair = first_with_format(fallback, desired_color_space)) {
      return SurfaceFormatChoice{.format = pair->format,
                                 .color_space = pair->color_space,
                                 .exact_color_space = pair->color_space == desired_color_space,
                                 .fallback_format = true};
    }
  }

  return std::nullopt;
}

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
