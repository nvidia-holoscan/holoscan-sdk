/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Display: a physical display output connected to an Adapter, enumerable per
// adapter and convertible into a Surface for direct (compositor-bypass)
// scanout. Displays the desktop owns are still enumerated — with
// DisplayOwnership::kCompositor — so callers can list them and tell the user
// how to free them (Windows: mark the monitor as a "Specialized display" in
// Settings; Linux: detach the output from the running session).

#pragma once

#include <holoviz/holoviz_export.h>

#include <array>
#include <cstdint>
#include <vector>

#include "common.hpp"
#include "surface.hpp"

namespace holoviz::rhi {

struct Adapter;

/// Who currently drives a display's scanout.
enum class DisplayOwnership : uint8_t {
  kUnknown,     ///< Could not be classified; create_surface() is attempted best-effort.
  kCompositor,  ///< Part of the desktop / window system; direct acquisition fails until
                ///< the display is detached from it.
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
/// Adapter so display lists can be copied and filtered freely. adapter_uuid is
/// the join key back to the owning Adapter; display_id is the backend handle
/// (Vulkan: VkDisplayKHR), stable for the process lifetime because the backend
/// instance is a process singleton.
struct Display {
  std::array<char, 256> name{};            ///< NUL-terminated driver/EDID display name.
  std::array<uint8_t, 16> adapter_uuid{};  ///< UUID of the Adapter the display is wired to.
  Backend backend{};
  uint64_t display_id = 0;
  DisplayMode current_mode{};  ///< Zeroed when the display is not scanning out (or unknown).
  Extent2D physical_size_mm{};
  DisplayOwnership ownership = DisplayOwnership::kUnknown;

  /// All modes the display supports. Queries the backend; setup-time work.
  [[nodiscard]] HOLOVIZ_EXPORT expected<std::vector<DisplayMode>> modes() const;

  /// Acquire the display for exclusive use and create a presentable Surface on
  /// it. The returned surface reports resizable() == false and extent() == the
  /// chosen mode's visible region; destroying it releases the display. Fails
  /// with kInvalidState when ownership == kCompositor (ownership is what the
  /// enumeration observed; the acquisition itself re-checks against the
  /// platform). kUnknown is attempted and the driver decides — a display freed
  /// by system configuration latches on without a platform acquire path.
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

}  // namespace holoviz::rhi
