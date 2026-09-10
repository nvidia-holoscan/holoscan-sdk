/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <array>
#include <memory>
#include <span>
#include <vector>

#include <holoviz/errors.hpp>
#include <holoviz/expected.hpp>

#include <holoviz/rhi/adapter.hpp>

#include "vk_impls.hpp"  // IWYU pragma: keep

namespace holoviz::rhi::vulkan {

/// HOLOVIZ_RHI_VALIDATION=1|true|on enables the Khronos validation layer (when the
/// loader finds it), the debug-utils messenger, and the module's extra debug checks
/// (e.g. the device live-child count); unset/0/false/off disables them all. Default is
/// off in every build type. Read once — the value is latched before the lazily created
/// instance exists (the first read happens inside the once-guarded instance init), so
/// changing the variable later in the process has no effect.
bool validation_enabled();

struct PhysicalDeviceInfo {
  VkPhysicalDevice physical_device_;

  VkPhysicalDeviceProperties2 physical_device_properties_;
  VkPhysicalDeviceIDProperties physical_device_id_properties_;

  VkPhysicalDeviceFeatures2 physical_device_features_;
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore_features_;

  std::vector<VkQueueFamilyProperties2> queue_family_properties_;

  int cuda_device_index_;
  CUdevice cuda_device_;
  CUuuid cuda_device_uuid_;
};

struct VulkanInstance {
  VkInstance vk_instance_;
  VkAllocationCallbacks* vk_allocator_;
  VkDebugUtilsMessengerEXT vk_debug_messenger_;
  std::span<PhysicalDeviceInfo> physical_device_infos_;

  /// The window system was unavailable at instance creation (headless console);
  /// window creation fails, only direct-display surfaces can present.
  bool headless_ = false;
  /// Optional direct-display instance extensions, enabled when the driver has them.
  bool display_supported_ = false;              ///< VK_KHR_display
  bool direct_mode_display_supported_ = false;  ///< VK_EXT_direct_mode_display
  bool acquire_drm_supported_ = false;          ///< VK_EXT_acquire_drm_display (Linux)
  bool acquire_xlib_supported_ = false;         ///< VK_EXT_acquire_xlib_display (Linux/X11)
};

[[nodiscard]] expected<VulkanInstance> GetVulkanInstance();

/// Resolve an adapter UUID to the instance's physical-device record.
[[nodiscard]] expected<const PhysicalDeviceInfo*> find_physical_device_info(
    const std::array<uint8_t, 16>& uuid);

expected<std::vector<Adapter>> enumerate_adapters();

/// Return the first queue family with graphics support that, when \p surface is not
/// VK_NULL_HANDLE, can also present to it. The RHI creates a single queue used for both
/// rendering and presentation, so surface support anywhere else on the device does not help.
[[nodiscard]] expected<uint32_t> find_graphics_queue_family(VkPhysicalDevice physical_device,
                                                            VkSurfaceKHR surface);

expected<bool> adapter_supports_surface(const Adapter& adapter, const Surface& surface);
expected<NativeObject> adapter_native_object(const Adapter& adapter, NativeObjectType type);

/// When \p surface is non-null the device's queue family is chosen to both render and
/// present to it; creation fails if no such family exists.
expected<std::unique_ptr<detail::DeviceImpl>> adapter_create_device(
    const Adapter& adapter, const DeviceDesc& desc, const Surface* surface = nullptr);

}  // namespace holoviz::rhi::vulkan
