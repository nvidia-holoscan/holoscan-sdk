/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure window-system selection policies, kept free of GLFW/Vulkan types so they are
// unit-testable without a window system (pattern: vk_display_util.hpp).

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include <holoviz/common.hpp>

namespace holoviz::window_systems {

/// Return the index of the monitor whose name exactly matches \p desired, or nullopt when
/// there is no match (the caller falls back to the primary monitor and logs \p names).
inline std::optional<size_t> find_monitor_by_name(std::span<const std::string_view> names,
                                                  std::string_view desired) {
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == desired) {
      return i;
    }
  }
  return std::nullopt;
}

/// Return the window size to create: a fullscreen window with a zero width or height takes
/// \p current_mode_size (the monitor's current video mode); any other request is used as-is.
constexpr Extent2D resolve_window_size(Extent2D requested, bool fullscreen,
                                       Extent2D current_mode_size) {
  if (fullscreen && (requested.width == 0 || requested.height == 0)) {
    return current_mode_size;
  }
  return requested;
}

/// Return the refresh rate in Hz to request for a fullscreen video mode: \p requested_mhz
/// rounded to the nearest Hz, or \p current_mode_hz when the request is zero (keep the
/// monitor's current rate; never "highest available").
constexpr int select_refresh_rate_hz(uint32_t requested_mhz, int current_mode_hz) {
  if (requested_mhz == 0) {
    return current_mode_hz;
  }
  return static_cast<int>((static_cast<uint64_t>(requested_mhz) + 500U) / 1000U);
}

}  // namespace holoviz::window_systems
