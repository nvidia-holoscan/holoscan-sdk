/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// X11/RandR side of the Linux direct-display path, called from
// display_ownership_drm.cpp when the session is X11. Kept in its own TU because
// the Xlib headers leak macros (None, Bool, Status) that break everything else;
// this interface carries the X connection as void* so no X type escapes.
// Compiled only under HOLOVIZ_HAS_XLIB_XRANDR (libX11 + libXrandr found).

#pragma once

#include <cstdint>
#include <vector>

#include <holoviz/expected.hpp>

#include "display_ownership.hpp"

namespace holoviz::display_systems::x11 {

/// Enumerate and classify the RandR outputs that resolve to `physical_device`'s
/// Vulkan displays (vkGetRandROutputDisplayEXT is the authoritative
/// RROutput -> VkDisplayKHR join). Names are the RandR output names — the ones
/// the user sees in xrandr — and ownership comes from RandR CRTC state, which
/// stays correct where KMS is blind (the NVIDIA X driver modesets outside KMS).
/// Errors when no X server is reachable, RandR is too old, or zero outputs
/// resolve (XWayland / foreign GPU) — the caller then falls back to DRM.
[[nodiscard]] expected<std::vector<NativeDisplayInfo>> classify_outputs(
    VkPhysicalDevice physical_device);

/// A successful vkAcquireXlibDisplayEXT acquisition. The X connection must stay
/// open for as long as the display is held (the X server reclaims the display
/// when the connection dies) and be closed only after vkReleaseDisplayEXT.
struct Acquisition {
  void* x11_connection = nullptr;  ///< X Display*
};

/// Acquire `vk_display` from the X server via vkAcquireXlibDisplayEXT.
[[nodiscard]] expected<Acquisition> acquire(VkPhysicalDevice physical_device, uint64_t vk_display);

/// XCloseDisplay wrapper; no-op on nullptr.
void close_connection(void* x11_connection);

}  // namespace holoviz::display_systems::x11
