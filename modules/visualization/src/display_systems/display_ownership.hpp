/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Platform seam for direct-display support, the display_systems/ sibling of
// window_systems/: OS-API code (kernel DRM ioctls on Linux) lives behind this interface so
// vk_display.cpp stays pure Vulkan. display_ownership_drm.cpp implements it on
// Linux; display_ownership_unsupported.cpp is the stub everywhere else —
// exclusive display is Linux-only.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <holoviz/display.hpp>
#include <holoviz/expected.hpp>

#include "vulkan/vk_adapter.hpp"

namespace holoviz::display_systems {

/// Platform view of one display, joined to Vulkan by the VkDisplayKHR handle.
struct NativeDisplayInfo {
  uint64_t vk_display = 0;  ///< VkDisplayKHR
  DisplayOwnership ownership = DisplayOwnership::kUnknown;
  std::array<char, 256> name{};  ///< Better platform name; empty keeps the Vulkan one.
  DisplayMode current_mode{};    ///< Current scanout timing where known; zeroed otherwise.
  /// VRR (G-Sync / Adaptive-Sync) capability where the platform reports it (Linux DRM:
  /// the connector's vrr_capable property); false also means "unknown". Advertised
  /// capability only; whether the driver engages it is decided at modeset.
  bool vrr_capable = false;
  /// VRR range floor from the EDID range-limits descriptor, in millihertz; 0 = unknown.
  uint32_t vrr_min_refresh_mhz = 0;
};

/// Classify the displays of `physical_device`. May also surface displays the
/// plain Vulkan enumeration missed. An error means platform classification is
/// unavailable as a whole — enumeration then proceeds with every display kUnknown;
/// it is never an error for individual displays to stay unclassified. Platforms
/// without a direct-display path return an empty list (success).
[[nodiscard]] expected<std::vector<NativeDisplayInfo>> classify_displays(
    const vulkan::VulkanInstance& instance, VkPhysicalDevice physical_device,
    const vulkan::PhysicalDeviceInfo& device_info);

/// Result of a direct-display acquisition attempt.
struct DisplayAcquisition {
  /// True when the platform acquire path (DRM) succeeded and the OS resources
  /// below are held. False when no platform acquire path exists (extension
  /// missing) or the driver declined it — the legacy "display was freed by
  /// system configuration" flow proceeds without one and the driver has the
  /// final word at surface/swapchain creation, acquiring the display implicitly
  /// there. The display surface calls vkReleaseDisplayEXT on destroy in both
  /// cases; this flag only governs the OS resources.
  bool acquired = false;
  /// Linux: the DRM fd backing the acquisition. Owned by the display surface
  /// and closed when it releases the display. -1 when unused.
  int drm_fd = -1;
};

/// Acquire `vk_display` for direct scanout. Returns {acquired=false} when the
/// platform has no acquire path; returns an Error when the path exists but the
/// acquisition failed (with per-platform remediation in the message).
/// Platforms without a direct-display path return kNotImplemented.
[[nodiscard]] expected<DisplayAcquisition> acquire_display(const vulkan::VulkanInstance& instance,
                                                           VkPhysicalDevice physical_device,
                                                           uint64_t vk_display);

/// Close the OS resources backing an acquisition (the DRM fd) and reset them.
/// Call only AFTER vkReleaseDisplayEXT: the fd backs the acquisition, so closing
/// it first could release the display implicitly, invalidating the handle mid-use.
void close_acquisition_resources(DisplayAcquisition& acquisition);

}  // namespace holoviz::display_systems
