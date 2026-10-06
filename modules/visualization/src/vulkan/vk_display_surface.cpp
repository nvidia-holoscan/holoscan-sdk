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
  if (vkReleaseDisplayEXT != nullptr && physical_device != VK_NULL_HANDLE &&
      display != VK_NULL_HANDLE) {
    result = vkReleaseDisplayEXT(physical_device, display);
  }
  // OS resources go after the Vulkan release (the DRM fd backs the acquisition), and go
  // regardless of its outcome: ownership ends here either way.
  display_systems::close_acquisition_resources(acquisition);
  acquisition.acquired = false;
  return result;
}

VulkanDisplaySurface::~VulkanDisplaySurface() {
  if (expected<void> result = destroy(detail::kDestructorDestroyTimeout); !result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] VulkanDisplaySurface: destroy failed in destructor: {}",
                        result.error().message);
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
  // Release the display unconditionally, not only when the platform acquire (DRM) succeeded: on
  // the "proceed unacquired" paths of acquire_display the driver acquires the display implicitly
  // at surface / swapchain creation, and without an explicit vkReleaseDisplayEXT that hold lives
  // until process exit (the VkInstance is process-lifetime). The driver then tears the head down
  // on its own, which stalls the exit and logs a flip timeout to the console.
  // VK_EXT_direct_mode_display places no acquired-state precondition on vkReleaseDisplayEXT.
  //
  // Teardown is terminal. VK_EXT_direct_mode_display documents no failure codes for
  // vkReleaseDisplayEXT, so a rejection is a driver anomaly with nothing meaningful to retry, and
  // the public Surface::destroy() empties the handle on every error except kBackpressured — an
  // object that kept display_ and the DRM fd would lose its only owner and leak the fd until
  // process exit. So display_ is nulled and the OS resources are closed whatever the driver
  // said (the same policy as the creation unwind in display_create_surface), and the rejection
  // is reported so the caller is not told the display is free when the driver may still hold it.
  HOLOVIZ_LOG_DEBUG("[holoviz] Releasing direct-mode display ({} acquisition).",
                    acquisition_.acquired ? "platform" : "implicit driver");
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
