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

/// Result of a direct-display acquisition attempt: the ownership policy selected before creating
/// a display surface (never kNone). kImplicit retains no DRM fd but keeps the release at surface
/// teardown; kExplicit releases the vkAcquireDrmDisplayEXT acquisition before closing the retained
/// fd; kDriverManaged leaves acquisition and release to the Tegra iGPU driver's swapchains, so
/// surface creation alone does not acquire the display and vkReleaseDisplayEXT is never called.
struct DisplayAcquisition {
  DisplayAcquisitionMode mode = DisplayAcquisitionMode::kImplicit;
  /// Linux: the DRM fd backing the acquisition. Owned by the display surface
  /// and closed when it releases the display. -1 when unused.
  int drm_fd = -1;
};

/// Acquire `vk_display` for direct scanout, or select a driver-managed ownership
/// policy. Driver-managed ownership is checked by the driver at swapchain creation.
/// Returns an Error when the explicit acquisition path fails (with per-platform
/// remediation in the message), except for the existing adapter-without-KMS fallback.
/// Platforms without a direct-display path return kNotImplemented.
[[nodiscard]] expected<DisplayAcquisition> acquire_display(const vulkan::VulkanInstance& instance,
                                                           VkPhysicalDevice physical_device,
                                                           uint64_t vk_display);

/// Close the OS resources backing an acquisition (the DRM fd) and reset them.
/// For an explicit DRM acquisition, call only AFTER vkReleaseDisplayEXT: the fd
/// backs the acquisition, so closing it first could release the display implicitly,
/// invalidating the handle mid-use.
void close_acquisition_resources(DisplayAcquisition& acquisition);

}  // namespace holoviz::display_systems
