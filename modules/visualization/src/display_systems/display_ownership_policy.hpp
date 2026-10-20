/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <vulkan/vulkan_core.h>

namespace holoviz::display_systems {

inline constexpr uint32_t kNVIDIAVendorID = 0x10de;

/// NVIDIA's Tegra integrated GPUs (Orin, Thor) drive their display through a
/// separate platform DRM device, not the GPU's own node, and the proprietary
/// driver manages display ownership at swapchain creation/destruction;
/// VK_EXT_acquire_drm_display cannot hand off the display through the GPU's node.
/// Match the selected Vulkan device by vendor, device type and driver rather than
/// by device ID, so every Tegra iGPU takes the same path. Discrete GPUs and other
/// drivers keep explicit DRM acquisition.
[[nodiscard]] constexpr bool uses_driver_managed_display(
    const VkPhysicalDeviceProperties& properties, VkDriverId driver_id) noexcept {
  return properties.vendorID == kNVIDIAVendorID &&
         properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU &&
         driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY;
}

/// High-refresh and high-bandwidth direct-display modes may be unstable on these same
/// Tegra iGPUs (a failure can hang the display engine until a power cycle), so their
/// modes are limited to kIgpuMaxRefreshRateHz and kIgpuMaxPixelRate (vk_display_util.hpp)
/// unless HOLOVIZ_DISABLE_IGPU_HZ_CAP=1.
[[nodiscard]] constexpr bool igpu_mode_cap_applies(const VkPhysicalDeviceProperties& properties,
                                                   VkDriverId driver_id) noexcept {
  return uses_driver_managed_display(properties, driver_id);
}

}  // namespace holoviz::display_systems
