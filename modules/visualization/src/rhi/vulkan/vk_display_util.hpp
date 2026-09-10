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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include <holoviz/rhi/display.hpp>

namespace holoviz::rhi::vulkan {

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

/// Ownership truth table for a DRM connector. A connector actively scanned out
/// (bound to an encoder/CRTC) while another DRM master runs belongs to that
/// compositor; a connected-but-inactive connector is free to latch (the
/// "xrandr --output X --off" flow), as is everything when no other master exists.
inline DisplayOwnership classify_drm_connector(bool connected, bool has_active_crtc,
                                               bool other_master_present) {
  if (!connected) {
    return DisplayOwnership::kUnknown;
  }
  if (other_master_present && has_active_crtc) {
    return DisplayOwnership::kCompositor;
  }
  return DisplayOwnership::kAvailable;
}

/// Ownership truth table for an X11 RandR output. An output driving a CRTC with
/// an active mode is scanned out by the X server (or a lease holder) —
/// kCompositor. A connected output with no CRTC is detached from the desktop
/// ("xrandr --output X --off" or the non-desktop property) and free to acquire
/// via VK_EXT_acquire_xlib_display. `non_desktop` does not change the verdict —
/// no-CRTC is already kAvailable, and a non-desktop output that somehow drives a
/// CRTC is being scanned out by someone — but rides along for logging.
inline DisplayOwnership classify_x11_output(bool connected, bool has_active_crtc,
                                            bool /*non_desktop*/) {
  if (!connected) {
    return DisplayOwnership::kUnknown;
  }
  return has_active_crtc ? DisplayOwnership::kCompositor : DisplayOwnership::kAvailable;
}

/// Whether the X11 classification/acquisition path should be attempted: DISPLAY
/// set and not a Wayland session. XWayland exposes fake RandR outputs that
/// vkGetRandROutputDisplayEXT cannot resolve and cannot lease — stay on DRM there.
inline bool prefer_x11_display_path(const char* display_env, const char* wayland_display_env,
                                    const char* xdg_session_type_env) {
  if (display_env == nullptr || display_env[0] == '\0') {
    return false;
  }
  if (wayland_display_env != nullptr && wayland_display_env[0] != '\0') {
    return false;
  }
  if (xdg_session_type_env != nullptr && std::strcmp(xdg_session_type_env, "wayland") == 0) {
    return false;
  }
  return true;
}

/// Merge a platform view of the displays into `base`, joined by the VkDisplayKHR
/// handle (the driver returns the same canonical handle per connector from
/// vkGetDrmDisplayEXT and vkGetRandROutputDisplayEXT, so handle equality is the
/// join key — no name/EDID matching). Overrides win ownership always, name and
/// current_mode only when they carry a value; unmatched overrides are appended;
/// base entries with no override keep their verdict. `Info` is any type with
/// vk_display/ownership/name/current_mode (display_systems::NativeDisplayInfo;
/// templated so this stays free of that header and unit-testable).
template <typename Info>
inline void merge_native_display_info(std::vector<Info>& base, std::span<const Info> overrides) {
  for (const Info& incoming : overrides) {
    if (incoming.vk_display == 0) {
      continue;
    }
    auto matched = std::find_if(base.begin(), base.end(), [&incoming](const Info& existing) {
      return existing.vk_display == incoming.vk_display;
    });
    if (matched == base.end()) {
      base.push_back(incoming);
      continue;
    }
    matched->ownership = incoming.ownership;
    if (incoming.name[0] != '\0') {
      matched->name = incoming.name;
    }
    if (incoming.current_mode.resolution.width > 0) {
      matched->current_mode = incoming.current_mode;
    }
  }
}

/// Ownership mapping for a Windows display target: only targets configured as
/// "Specialized display" (removed from the desktop) are acquirable via
/// VK_NV_acquire_winrt_display; everything else belongs to the DWM.
inline DisplayOwnership classify_winrt_target(bool is_specialized) {
  return is_specialized ? DisplayOwnership::kAvailable : DisplayOwnership::kCompositor;
}

}  // namespace holoviz::rhi::vulkan
