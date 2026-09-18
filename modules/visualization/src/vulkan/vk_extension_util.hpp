/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>

namespace holoviz::vulkan {

/// Select the device extension whose instance dependency was enabled. The EXT and KHR
/// commands alias each other, but their extension dependencies require matching names.
/// Return nullptr if the device does not offer a compatible extension.
template <typename ExtensionSupported>
inline const char* select_swapchain_maintenance1_extension(
    std::string_view surface_extension, const ExtensionSupported& extension_supported) {
  // Literal extension names keep this helper independent of Vulkan headers for host testing.
  const char* device_extension = nullptr;
  if (surface_extension == "VK_KHR_surface_maintenance1") {
    device_extension = "VK_KHR_swapchain_maintenance1";
  } else if (surface_extension == "VK_EXT_surface_maintenance1") {
    device_extension = "VK_EXT_swapchain_maintenance1";
  }
  return device_extension != nullptr && extension_supported(device_extension) ? device_extension
                                                                              : nullptr;
}

}  // namespace holoviz::vulkan
