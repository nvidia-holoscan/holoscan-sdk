/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vk_display_surface.hpp"

#include <string>

#include <holoviz/logger.hpp>

#include "detail/destroy_timeout.hpp"
#include "vk_errors.hpp"

namespace holoviz::vulkan {

VkResult release_display_then_close(VkPhysicalDevice physical_device, VkDisplayKHR display,
                                    display_systems::DisplayAcquisition& acquisition) {
  VkResult result = VK_SUCCESS;
  if (acquisition.mode != DisplayAcquisitionMode::kDriverManaged &&
      vkReleaseDisplayEXT != nullptr && physical_device != VK_NULL_HANDLE &&
      display != VK_NULL_HANDLE) {
    result = vkReleaseDisplayEXT(physical_device, display);
  }
  // OS resources go after the Vulkan release (the DRM fd backs the acquisition), and go
  // regardless of its outcome: ownership ends here either way.
  display_systems::close_acquisition_resources(acquisition);
  return result;
}

VulkanDisplaySurface::~VulkanDisplaySurface() {
  if (expected<void> result = destroy(detail::kDestructorDestroyTimeout); !result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] VulkanDisplaySurface: destroy failed in destructor: {}",
                        result.error().message);
      // Logging can throw during teardown; cleanup must continue without escaping.
      // NOLINTNEXTLINE(bugprone-empty-catch)
    } catch (...) {
    }
  }
}

expected<void> VulkanDisplaySurface::destroy([[maybe_unused]] std::chrono::nanoseconds timeout) {
  if (surface_ != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(instance_, surface_, allocator_);
    surface_ = VK_NULL_HANDLE;
  }
  if (display_ == VK_NULL_HANDLE) {
    return {};  // already torn down (destructor re-run, or a repeated destroy())
  }
  // Thor's driver-managed ownership ends when its swapchains are destroyed.
  // The explicit DRM and existing non-Thor implicit paths retain their release
  // behavior. Teardown is terminal even when that release fails: the public
  // Surface handle empties, so retaining the fd here would leave it owner-less.
  const VkResult result = release_display_then_close(physical_device_, display_, acquisition_);
  display_ = VK_NULL_HANDLE;
  if (result != VK_SUCCESS) {
    const std::string message =
        "[holoviz] vkReleaseDisplayEXT rejected the display release; the driver may hold the "
        "display until process exit. The surface's display handle and OS resources were "
        "relinquished: " +
        vk_error_string(result, __FILE__, __LINE__);
    HOLOVIZ_LOG_ERROR("{}", message);
    return make_unexpected(Error{.code = ErrorCode::kVulkanError, .message = message});
  }
  return {};
}

expected<void> VulkanDisplaySurface::init(VkInstance instance, VkAllocationCallbacks* allocator,
                                          VkSurfaceKHR surface, Extent2D extent,
                                          VkPhysicalDevice physical_device, VkDisplayKHR display,
                                          const display_systems::DisplayAcquisition& acquisition) {
  instance_ = instance;
  allocator_ = allocator;
  surface_ = surface;
  extent_ = extent;
  physical_device_ = physical_device;
  display_ = display;
  acquisition_ = acquisition;
  return {};
}

expected<NativeObject> VulkanDisplaySurface::native_object(NativeObjectType type) const {
  switch (type) {
    case NativeObjectType::kVkSurfaceKHR:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(surface_),
      };
    case NativeObjectType::kVkDisplayKHR:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(display_),
      };
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] Unsupported native object type for VulkanDisplaySurface."});
  }
}

}  // namespace holoviz::vulkan
