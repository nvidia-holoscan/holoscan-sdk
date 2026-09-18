/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Platform seam for direct-display support, the display_systems/ sibling of
// window_systems/: OS-API code (WinRT on Windows, libdrm on Linux) lives behind
// this interface so vk_display.cpp stays pure Vulkan. One TU per platform
// implements it (display_ownership_win32.cpp / display_ownership_drm.cpp).

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
};

/// Classify the displays of `physical_device`. May also surface displays the
/// plain Vulkan enumeration missed (Windows resolves WinRT targets through
/// vkGetWinrtDisplayNV). An error means platform classification is unavailable
/// as a whole — enumeration then proceeds with every display kUnknown; it is
/// never an error for individual displays to stay unclassified.
[[nodiscard]] expected<std::vector<NativeDisplayInfo>> classify_displays(
    const vulkan::VulkanInstance& instance, VkPhysicalDevice physical_device,
    const vulkan::PhysicalDeviceInfo& device_info);

/// Result of a direct-display acquisition attempt.
struct DisplayAcquisition {
  /// False when no platform acquire path exists (extension missing) — the
  /// legacy "display was freed by system configuration" flow proceeds without
  /// one and the driver has the final word at surface/swapchain creation.
  bool acquired = false;
  /// Linux: the DRM fd backing the acquisition. Owned by the display surface
  /// and closed when it releases the display. -1 when unused.
  int drm_fd = -1;
  /// Linux/X11: the X connection (a Display*) backing a vkAcquireXlibDisplayEXT
  /// acquisition, type-erased so vk_display.cpp stays free of X headers. The X
  /// server hands the display back when the connection dies, so it is owned by
  /// the display surface and closed only after vkReleaseDisplayEXT.
  void* x11_connection = nullptr;
};

/// Acquire `vk_display` for direct scanout. Returns {acquired=false} when the
/// platform has no acquire path; returns an Error when the path exists but the
/// acquisition failed (with per-platform remediation in the message).
[[nodiscard]] expected<DisplayAcquisition> acquire_display(const vulkan::VulkanInstance& instance,
                                                           VkPhysicalDevice physical_device,
                                                           uint64_t vk_display);

/// Close the OS resources backing an acquisition (DRM fd / X connection) and
/// reset them. Call only AFTER vkReleaseDisplayEXT: closing the X connection
/// first would implicitly release the display, invalidating the handle mid-use.
void close_acquisition_resources(DisplayAcquisition& acquisition);

}  // namespace holoviz::display_systems
