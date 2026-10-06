/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure display-selection helpers, split out of vk_display.cpp so they are
// unit-testable without an instance or a display (the vk_swapchain_util.hpp
// pattern). Deliberately free of Vulkan headers: handles ride as uint64_t and
// alpha modes as their VkDisplayPlaneAlphaFlagBitsKHR bit values (asserted
// equal in vk_display.cpp).

#pragma once

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>

#include <holoviz/display.hpp>

namespace holoviz::vulkan {

struct DisplayModeInfo {
  uint64_t handle = 0;  ///< VkDisplayModeKHR
  DisplayMode mode{};
};

/// Best-match mode selection, ported from the legacy ExclusiveWindow: score
/// each mode by (width diff, height diff, refresh diff), compared
/// lexicographically. A desired field of 0 means "don't care, prefer largest":
/// its diff is UINT32_MAX - value, so bigger values score better while still
/// participating in the tie-break order. Returns nullopt when modes is empty.
inline std::optional<size_t> select_display_mode(std::span<const DisplayModeInfo> modes,
                                                 const DisplayMode& desired) {
  const auto diff_for = [](uint32_t desired_value, uint32_t mode_value) -> uint32_t {
    if (desired_value > 0) {
      return std::abs(static_cast<int32_t>(mode_value) - static_cast<int32_t>(desired_value));
    }
    return std::numeric_limits<uint32_t>::max() - mode_value;
  };

  std::optional<size_t> best;
  uint32_t least_width_diff = std::numeric_limits<uint32_t>::max();
  uint32_t least_height_diff = std::numeric_limits<uint32_t>::max();
  uint32_t least_rate_diff = std::numeric_limits<uint32_t>::max();

  for (size_t i = 0; i < modes.size(); ++i) {
    const DisplayMode& m = modes[i].mode;
    const uint32_t width_diff = diff_for(desired.resolution.width, m.resolution.width);
    const uint32_t height_diff = diff_for(desired.resolution.height, m.resolution.height);
    const uint32_t rate_diff = diff_for(desired.refresh_rate_mhz, m.refresh_rate_mhz);

    if (!best.has_value() || (width_diff < least_width_diff) ||
        ((width_diff == least_width_diff) && (height_diff < least_height_diff)) ||
        ((width_diff == least_width_diff) && (height_diff == least_height_diff) &&
         (rate_diff < least_rate_diff))) {
      best = i;
      least_width_diff = width_diff;
      least_height_diff = height_diff;
      least_rate_diff = rate_diff;
    }
  }
  return best;
}

/// Whether `selected` satisfies every non-zero field of `desired` exactly
/// (drives the "using closest match" warning; a fully-zero desired always matches).
inline bool mode_matches_desired(const DisplayMode& selected, const DisplayMode& desired) {
  if (desired.resolution.width > 0 && selected.resolution.width != desired.resolution.width) {
    return false;
  }
  if (desired.resolution.height > 0 && selected.resolution.height != desired.resolution.height) {
    return false;
  }
  if (desired.refresh_rate_mhz > 0 && selected.refresh_rate_mhz != desired.refresh_rate_mhz) {
    return false;
  }
  return true;
}

struct DisplayPlaneInfo {
  uint64_t current_display = 0;  ///< VkDisplayKHR the plane is bound to, or 0.
  uint32_t current_stack_index = 0;
  bool supports_target = false;  ///< Target display in vkGetDisplayPlaneSupportedDisplaysKHR set.
};

/// First plane that is not bound to a different display and supports the
/// target (legacy behavior). `target_display` is the VkDisplayKHR being latched.
inline std::optional<uint32_t> select_display_plane(std::span<const DisplayPlaneInfo> planes,
                                                    uint64_t target_display) {
  for (uint32_t i = 0; i < planes.size(); ++i) {
    if (planes[i].current_display != 0 && planes[i].current_display != target_display) {
      continue;  // bound to a different display
    }
    if (planes[i].supports_target) {
      return i;
    }
  }
  return std::nullopt;
}

/// VkDisplayPlaneAlphaFlagBitsKHR values, mirrored so this header stays free of
/// Vulkan includes; vk_display.cpp static_asserts the equalities.
inline constexpr uint32_t kDisplayPlaneAlphaOpaque = 0x1;
inline constexpr uint32_t kDisplayPlaneAlphaGlobal = 0x2;
inline constexpr uint32_t kDisplayPlaneAlphaPerPixel = 0x4;
inline constexpr uint32_t kDisplayPlaneAlphaPerPixelPremultiplied = 0x8;

/// First supported alpha mode in legacy priority order; opaque as the fallback
/// when the mask advertises none of them.
inline uint32_t select_alpha_mode(uint32_t supported_alpha_mask) {
  for (uint32_t candidate : {kDisplayPlaneAlphaOpaque,
                             kDisplayPlaneAlphaGlobal,
                             kDisplayPlaneAlphaPerPixel,
                             kDisplayPlaneAlphaPerPixelPremultiplied}) {
    if ((supported_alpha_mask & candidate) != 0) {
      return candidate;
    }
  }
  return kDisplayPlaneAlphaOpaque;
}

/// Ownership truth table for a DRM connector. Any other DRM master (a display
/// server or compositor) holds the device as a whole, so while one runs every
/// connected connector belongs to it — including outputs the session has turned
/// off — and vkAcquireDrmDisplayEXT cannot succeed. With no other master
/// (a bare console) everything connected is available.
inline DisplayOwnership classify_drm_connector(bool connected,
                                               [[maybe_unused]] bool has_active_crtc,
                                               bool other_master_present) {
  if (!connected) {
    return DisplayOwnership::kUnknown;
  }
  if (other_master_present) {
    return DisplayOwnership::kCompositor;
  }
  return DisplayOwnership::kAvailable;
}

/// Minimum vertical refresh rate from an EDID's range-limits descriptor (tag 0xFD), in
/// Hz; nullopt when the EDID is too short or carries no such descriptor. Scans the four
/// 18-byte display descriptors of the base block (offsets 54/72/90/108; a descriptor
/// starts with a 0x0000 pixel clock, unlike a detailed timing) and honors the EDID 1.4
/// rate-offset flags (byte 4 bits 1:0 == 0b11 adds 255 Hz to the minimum rate too).
inline std::optional<uint32_t> edid_min_vertical_rate_hz(std::span<const uint8_t> edid) {
  constexpr size_t kBaseBlockSize = 128;
  if (edid.size() < kBaseBlockSize) {
    return std::nullopt;
  }
  for (const size_t offset : {size_t{54}, size_t{72}, size_t{90}, size_t{108}}) {
    if (edid[offset] != 0 || edid[offset + 1] != 0) {
      continue;  // a detailed timing (nonzero pixel clock), not a display descriptor
    }
    if (edid[offset + 3] != 0xFD) {
      continue;  // a descriptor, but not range limits
    }
    const uint8_t rate_offsets = edid[offset + 4];
    uint32_t min_vertical_hz = edid[offset + 5];
    if ((rate_offsets & 0x3) == 0x3) {
      min_vertical_hz += 255;
    }
    return min_vertical_hz;
  }
  return std::nullopt;
}

}  // namespace holoviz::vulkan
