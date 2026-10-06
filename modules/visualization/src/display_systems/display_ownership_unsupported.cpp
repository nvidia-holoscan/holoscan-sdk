/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Display ownership + acquisition on platforms without a direct-display path.
// Exclusive display is implemented on Linux only (display_ownership_drm.cpp);
// everywhere else this TU keeps the display_systems seam linkable so the window
// path builds unchanged: classification is empty (every display stays kUnknown,
// no per-enumeration warning for a permanent platform condition) and
// Display::create_surface() fails with kNotImplemented before any Vulkan
// surface is created.

#include "display_ownership.hpp"

#include <cstdint>
#include <vector>

namespace holoviz::display_systems {

expected<std::vector<NativeDisplayInfo>> classify_displays(
    const vulkan::VulkanInstance& /*instance*/, VkPhysicalDevice /*physical_device*/,
    const vulkan::PhysicalDeviceInfo& /*device_info*/) {
  // No platform view: enumeration keeps the plain Vulkan list, all kUnknown.
  return std::vector<NativeDisplayInfo>{};
}

expected<DisplayAcquisition> acquire_display(const vulkan::VulkanInstance& /*instance*/,
                                             VkPhysicalDevice /*physical_device*/,
                                             uint64_t /*vk_display*/) {
  return make_unexpected(
      Error{.code = ErrorCode::kNotImplemented,
            .message = "[holoviz] Exclusive display is only implemented on Linux (DRM); this "
                       "platform cannot acquire a display for direct scanout."});
}

void close_acquisition_resources(DisplayAcquisition& acquisition) {
  // No OS resources exist on this platform; keep the record in its released state.
  acquisition.drm_fd = -1;
}

}  // namespace holoviz::display_systems
