/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Display: a physical display output connected to an Adapter, enumerable per
// adapter and convertible into a Surface for direct (compositor-bypass)
// scanout. Displays the desktop owns are still enumerated — with
// DisplayOwnership::kCompositor — so callers can list them and tell the user
// how to free them (stop the desktop session driving the GPU). Direct scanout
// is implemented on Linux through DRM; other platforms report
// CapabilityReport::exclusive_display == false and create_surface() fails.

#pragma once

#include <holoviz/holoviz_export.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "adapter.hpp"
#include "common.hpp"
#include "holoviz/expected.hpp"
#include "surface.hpp"

namespace holoviz {

/// Who currently drives a display's scanout.
enum class DisplayOwnership : uint8_t {
  kUnknown,     ///< Could not be classified; create_surface() is attempted best-effort.
  kCompositor,  ///< Driven by a running desktop / display server; direct acquisition fails
                ///< while any display server holds the GPU (Linux/DRM: run from a console).
  kAvailable,   ///< Free for direct latch-on via Display::create_surface().
};

/// One scanout timing a display supports. Refresh rate follows Vulkan's
/// convention: millihertz (60 Hz == 60000).
struct DisplayMode {
  Extent2D resolution{};
  uint32_t refresh_rate_mhz = 0;
};

struct DisplaySurfaceDesc {
  /// Desired mode; any field left 0 means "don't care, prefer largest/fastest".
  /// The closest supported mode wins (scored by width, then height, then
  /// refresh); an inexact match logs the available modes.
  DisplayMode mode{};
};

/// A display connected to an adapter, described as a plain value type like
/// Adapter so display lists can be copied and filtered freely. adapter is the owning
/// Adapter (copied in); display_id is the backend handle
/// (Vulkan: VkDisplayKHR), stable for the process lifetime because the backend
/// instance is a process singleton.
struct Display {
  std::array<char, 256> name{};  ///< NUL-terminated driver/EDID display name.
  Adapter adapter;  ///< The owning Adapter, copied in so a Display alone can create a device.
  Backend backend{};
  uint64_t display_id = 0;
  DisplayMode current_mode{};  ///< Zeroed when the display is not scanning out (or unknown).
  Extent2D physical_size_mm{};
  DisplayOwnership ownership = DisplayOwnership::kUnknown;
  /// The display and driver advertise variable refresh rate (G-Sync / Adaptive-Sync):
  /// under PresentMode::kImmediate on a direct-display surface the scanout may follow the
  /// present rate within the display's VRR range instead of tearing at fixed refresh.
  /// Capability only — what the EDID / DRM connector advertises, not a promise that the
  /// driver engages VRR (NVIDIA, for example, engages Adaptive-Sync on direct-display
  /// surfaces only for panels on its G-Sync Compatible list). False also means "unknown";
  /// only the Linux DRM classifier fills this in.
  bool vrr_capable = false;
  /// VRR range floor from the display's EDID range-limits descriptor (Vulkan millihertz
  /// convention); 0 = unknown. The ceiling is the chosen display mode's refresh rate.
  uint32_t vrr_min_refresh_mhz = 0;

  /// All modes the display supports. Queries the backend; setup-time work.
  [[nodiscard]] HOLOVIZ_EXPORT expected<std::vector<DisplayMode>> modes() const;

  /// Create a presentable Surface for exclusive use of the display. Ownership
  /// is acquired explicitly where supported, or by the driver at swapchain
  /// creation. The returned surface reports resizable() == false and extent()
  /// == the chosen mode's visible region. Destroy all swapchains using the
  /// surface before destroying it; driver-managed ownership ends with the last
  /// swapchain, while explicit acquisition ends with the surface. Fails
  /// with kInvalidState when ownership == kCompositor (ownership is what the
  /// enumeration observed; the acquisition itself re-checks against the
  /// platform or at swapchain creation). kUnknown is attempted and the driver
  /// decides whether the display is available.
  /// Surface::display_acquisition_mode() reports which acquisition path was taken.
  /// Fails with kNotImplemented on platforms without a direct-display path
  /// (non-Linux).
  [[nodiscard]] HOLOVIZ_EXPORT expected<Surface> create_surface(
      const DisplaySurfaceDesc& desc = {}) const;

  /// Vulkan: answers kVkDisplayKHR.
  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;
};

/// Enumerate the displays physically connected to `adapter`, including ones the
/// compositor owns. An empty vector means none are connected; kNotImplemented
/// means the platform/driver exposes no display enumeration path.
[[nodiscard]] HOLOVIZ_EXPORT expected<std::vector<Display>> enumerate_displays(
    const Adapter& adapter);

/// Return the first display (any adapter) whose name contains \p name; an empty name matches
/// the first display. Fails with ErrorCode::kNotFound when nothing matches.
[[nodiscard]] HOLOVIZ_EXPORT expected<Display> get_display(const std::string& name);

class Window;

/// Resolve the Display the window is (mostly) on, from `adapter`'s enumeration — the
/// bridge that makes display-scoped facilities (Device::wait_for_display_event(), vblank
/// counters) usable for windowed presentation. The window system's monitor (the
/// fullscreen monitor, otherwise the one with the largest overlap with the window
/// rectangle) is joined against enumerate_displays() by name (exact, then
/// case-insensitive substring in either direction), then by active-mode resolution and
/// physical size. Window-system monitor names and driver display names live in different
/// namespaces (the same panel can be "DP-0" to X11 and "DP-2" to DRM, while another panel
/// takes "DP-2" on the X11 side), so a name hit never overrides geometry: a display whose
/// known active resolution, refresh rate or physical size contradicts the monitor's is
/// rejected in every tier, including when it is the adapter's only display; attributes
/// either side does not report never reject, and a monitor the window system reports rotated
/// (its resolution and physical size swapped relative to the driver's native reading) still
/// matches. Fails with kNotFound when no unambiguous, geometry-consistent match exists and
/// with kNotImplemented on window systems that report no monitor geometry — including a
/// *windowed* window on Wayland, which exposes neither the window's global position nor the
/// output it is on; fullscreen windows resolve on every window system.
[[nodiscard]] HOLOVIZ_EXPORT expected<Display> display_for_window(const Window& window,
                                                                  const Adapter& adapter);

}  // namespace holoviz
