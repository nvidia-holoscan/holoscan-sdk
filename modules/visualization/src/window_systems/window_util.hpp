/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure window-system selection policies, kept free of GLFW/Vulkan types so they are
// unit-testable without a window system (pattern: vk_display_util.hpp).

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <holoviz/common.hpp>
#include <holoviz/display.hpp>

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

/// Which window-system connection routes the process environment advertises. Each flag means
/// "the variable is set", not "a server is reachable": libwayland-client connects through any of
/// \p wayland_display (a socket name or absolute path), \p wayland_socket (an inherited fd), or
/// the default \c wayland-0 socket under \p xdg_runtime_dir; X11 connects through
/// \p x11_display.
struct WindowSystemEnv {
  bool wayland_display = false;  ///< WAYLAND_DISPLAY
  bool wayland_socket = false;   ///< WAYLAND_SOCKET
  bool xdg_runtime_dir = false;  ///< XDG_RUNTIME_DIR
  bool x11_display = false;      ///< DISPLAY
};

/// GLFW platform to request before glfwInit().
enum class GLFWPlatformChoice : uint8_t {
  kNone,     ///< no route to any window system: skip glfwInit() entirely
  kX11Only,  ///< only X11 is reachable: pin GLFW_PLATFORM_X11
  kAny,      ///< a Wayland route exists: let GLFW probe (GLFW_ANY_PLATFORM)
};

/// Return the GLFW platform to request for \p env. A Wayland connection is possible through any
/// of its three routes, so any of them keeps GLFW's own probing (it also honors
/// XDG_SESSION_TYPE=x11 when DISPLAY is set). Without a Wayland route GLFW must not probe
/// Wayland: libwayland-client prints "XDG_RUNTIME_DIR is invalid or not set" to stderr in that
/// case, so X11 is pinned when DISPLAY is set, and with no route at all glfwInit() is skipped.
constexpr GLFWPlatformChoice choose_glfw_platform(WindowSystemEnv env) {
  const bool wayland_route = env.wayland_display || env.wayland_socket || env.xdg_runtime_dir;
  if (wayland_route) {
    return GLFWPlatformChoice::kAny;
  }
  if (env.x11_display) {
    return GLFWPlatformChoice::kX11Only;
  }
  return GLFWPlatformChoice::kNone;
}

/// A window system's view of the monitor a window occupies, for joining against the
/// backend's display enumeration (display_for_window()).
struct NativeMonitorInfo {
  std::array<char, 256> name{};   ///< NUL-terminated window-system monitor name.
  Extent2D physical_size_mm{};    ///< {0,0} when the window system does not report it.
  Extent2D current_resolution{};  ///< The monitor's active video-mode resolution in pixels.
  uint32_t refresh_rate_mhz = 0;  ///< Vulkan millihertz convention; 0 when unknown.
};

namespace detail_match {

inline std::string lowered(std::string_view text) {
  std::string result{text};
  std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return result;
}

/// Physical-size slack between the window system's and the driver's EDID reading.
inline constexpr uint32_t kPhysicalSizeToleranceMm = 10;

inline bool within_physical_tolerance(uint32_t a, uint32_t b) {
  return (a > b ? a - b : b - a) <= kPhysicalSizeToleranceMm;
}

constexpr Extent2D transposed(Extent2D extent) {
  return {extent.height, extent.width};
}

constexpr bool known(Extent2D extent) {
  return extent.width != 0 && extent.height != 0;
}

/// True when \p a and \p b agree read in the same orientation or with one of them
/// transposed; \p compare decides per-axis agreement (exact or within tolerance).
template <typename Compare>
constexpr bool agrees_in_either_orientation(Extent2D a, Extent2D b, Compare compare) {
  return (compare(a.width, b.width) && compare(a.height, b.height)) ||
         (compare(a.width, b.height) && compare(a.height, b.width));
}

/// True when every attribute *both* sides report agrees: active resolution and physical size
/// each in either orientation, refresh rate (millihertz on both sides) exactly. An attribute
/// either side leaves at 0 is unknown and never disagrees.
///
/// Resolution and physical size are compared independently of each other's orientation: the
/// window system may report them in different orientations for the same panel. GLFW 3.4 on X11
/// orients a monitor's physical size once, when it first sees the monitor, but reads the video
/// mode from the current CRTC, so after a runtime rotation the resolution is transposed while
/// the cached physical size is not. A different panel is still vetoed because its physical size
/// (or resolution) disagrees in *both* orientations.
inline bool geometry_agrees(const Display& display, const NativeMonitorInfo& monitor) {
  const Extent2D& display_res = display.current_mode.resolution;
  if (known(display_res) && known(monitor.current_resolution) &&
      !agrees_in_either_orientation(
          display_res, monitor.current_resolution, [](uint32_t a, uint32_t b) { return a == b; })) {
    return false;
  }
  if (display.current_mode.refresh_rate_mhz != 0 && monitor.refresh_rate_mhz != 0 &&
      display.current_mode.refresh_rate_mhz != monitor.refresh_rate_mhz) {
    return false;
  }
  const Extent2D& display_mm = display.physical_size_mm;
  if (known(display_mm) && known(monitor.physical_size_mm) &&
      !agrees_in_either_orientation(
          display_mm, monitor.physical_size_mm, within_physical_tolerance)) {
    return false;
  }
  return true;
}

/// True when the display's known geometry is a different panel's: resolution or physical size
/// disagrees in both orientations (see geometry_agrees()). The two sides describe the panel in
/// different coordinate systems — the driver (DRM mode, VkDisplayPropertiesKHR) reports the
/// panel's native orientation, the window system the desktop's — and neither exposes the
/// rotation itself. A name-only display record can still match.
inline bool contradicts_monitor(const Display& display, const NativeMonitorInfo& monitor) {
  return !geometry_agrees(display, monitor);
}

/// True when both sides report an active resolution and they agree in either orientation.
inline bool resolution_matches(const Display& display, const NativeMonitorInfo& monitor) {
  const Extent2D& display_res = display.current_mode.resolution;
  const Extent2D& monitor_res = monitor.current_resolution;
  if (!known(display_res) || !known(monitor_res)) {
    return false;
  }
  const auto same = [&display_res](Extent2D extent) {
    return display_res.width == extent.width && display_res.height == extent.height;
  };
  return same(monitor_res) || same(transposed(monitor_res));
}

}  // namespace detail_match

/// Join a window system's monitor record to the backend's display enumeration.
///
/// Names are not a shared identity: the window system reports its own output name (X11's
/// RandR name), the backend the driver's — on Linux the DRM connector name — and the same
/// panel can be "DP-0" on one side and "DP-2" on the other while a *different* panel takes
/// the string "DP-2" in the second namespace. So geometry is a veto in every tier, not just
/// a tie-breaker: a candidate whose known active resolution, refresh rate or physical size
/// contradicts the monitor's (detail_match::contradicts_monitor) is never selected, whatever
/// its name; attributes either side does not know (0) never reject. Geometry is compared
/// orientation-insensitively — the window system reports a rotated monitor's resolution and
/// physical size in desktop orientation, the driver in the panel's native one — with
/// resolution and physical size transposed together. (A shared native identity such as an
/// EDID hash would be a stronger tier 0; neither side exposes one yet.)
///
/// Tiered, most-specific first; a tier with exactly one hit wins, a tier with several hits is
/// ambiguous and aborts (nullopt — a looser tier cannot disambiguate), a tier with none
/// falls through:
///  1. exact name match (window-system names match the platform classifier's on X11,
///     where both are the connector names);
///  2. case-insensitive substring in either direction (the backend may report an EDID
///     model string where the window system reports a connector name, or vice versa);
///  3. active-mode resolution match, plus refresh rate and physical size within tolerance
///     when both sides report them (covers window systems whose names never match the driver's);
///  4. a single enumerated display matches trivially — unless its geometry contradicts the
///     monitor (the caller passed an adapter that does not drive this panel).
inline std::optional<size_t> match_display(std::span<const Display> displays,
                                           const NativeMonitorInfo& monitor) {
  const std::string_view monitor_name{monitor.name.data()};
  const auto compatible = [&monitor](const Display& display) {
    return !detail_match::contradicts_monitor(display, monitor);
  };

  const auto single_match = [&displays](auto&& predicate) -> std::optional<std::optional<size_t>> {
    std::optional<size_t> found;
    for (size_t i = 0; i < displays.size(); ++i) {
      if (!predicate(displays[i])) {
        continue;
      }
      if (found) {
        return std::optional<size_t>{};  // ambiguous: more than one display fits this tier
      }
      found = i;
    }
    if (!found) {
      return std::nullopt;  // no hit: fall through to the next tier
    }
    return found;
  };

  // Tier 1: exact name.
  if (!monitor_name.empty()) {
    if (const auto hit = single_match([&](const Display& display) {
          const std::string_view display_name{display.name.data()};
          return monitor_name == display_name && compatible(display);
        })) {
      return *hit;
    }
  }

  // Tier 2: case-insensitive substring, either direction.
  if (!monitor_name.empty()) {
    const std::string monitor_lower = detail_match::lowered(monitor_name);
    if (const auto hit = single_match([&](const Display& display) {
          const std::string display_lower =
              detail_match::lowered(std::string_view{display.name.data()});
          if (display_lower.empty()) {
            return false;
          }
          const bool name_hit = display_lower.find(monitor_lower) != std::string::npos ||
                                monitor_lower.find(display_lower) != std::string::npos;
          return name_hit && compatible(display);
        })) {
      return *hit;
    }
  }

  // Tier 3: active-mode resolution on both sides (either orientation), with the shared veto
  // covering refresh rate and physical size when both sides know them.
  if (detail_match::known(monitor.current_resolution)) {
    if (const auto hit = single_match([&](const Display& display) {
          return detail_match::resolution_matches(display, monitor) && compatible(display);
        })) {
      return *hit;
    }
  }

  // Tier 4: nothing to disambiguate — but a sole display that is demonstrably another panel
  // (wrong adapter) is a miss, not a match.
  if (displays.size() == 1 && compatible(displays[0])) {
    return 0;
  }
  return std::nullopt;
}

/// Return the index of the monitor whose rectangle (global desktop pixel coordinates)
/// overlaps the window's by the largest area, or nullopt when no monitor overlaps it — or
/// when \p window is nullopt because the window system cannot report global window
/// coordinates (Wayland): an unknown position must never be read as the origin, which would
/// silently select the output at (0,0).
inline std::optional<size_t> select_monitor_by_overlap(std::optional<Rect2D> window,
                                                       std::span<const Rect2D> monitors) {
  if (!window) {
    return std::nullopt;
  }
  const int64_t window_right = static_cast<int64_t>(window->x) + window->width;
  const int64_t window_bottom = static_cast<int64_t>(window->y) + window->height;
  std::optional<size_t> best;
  int64_t best_overlap = 0;  // a monitor must actually overlap the window to be chosen
  for (size_t i = 0; i < monitors.size(); ++i) {
    const Rect2D& monitor = monitors[i];
    const int64_t overlap_w =
        std::min<int64_t>(window_right, static_cast<int64_t>(monitor.x) + monitor.width) -
        std::max<int64_t>(window->x, monitor.x);
    const int64_t overlap_h =
        std::min<int64_t>(window_bottom, static_cast<int64_t>(monitor.y) + monitor.height) -
        std::max<int64_t>(window->y, monitor.y);
    const int64_t overlap = std::max<int64_t>(overlap_w, 0) * std::max<int64_t>(overlap_h, 0);
    if (overlap > best_overlap) {
      best_overlap = overlap;
      best = i;
    }
  }
  return best;
}

}  // namespace holoviz::window_systems
