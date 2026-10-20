/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vk_surface.hpp"
#include "../detail/destroy_timeout.hpp"

#include "holoviz/common.hpp"

namespace holoviz::vulkan {

VulkanSurface::~VulkanSurface() {
  if (expected<void> result = destroy(detail::kDestructorDestroyTimeout); !result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] VulkanSurface: destroy failed in destructor: {}",
                        result.error().message);
      // Logging can throw during teardown; cleanup must continue without escaping.
      // NOLINTNEXTLINE(bugprone-empty-catch)
    } catch (...) {
    }
  }
}

expected<void> VulkanSurface::destroy([[maybe_unused]] std::chrono::nanoseconds timeout) {
  if (surface_ != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(instance_, surface_, allocator_);
    surface_ = VK_NULL_HANDLE;
  }
  return {};
}

expected<void> VulkanSurface::init(VkInstance instance, VkAllocationCallbacks* allocator,
                                   VkSurfaceKHR surface, Extent2D extent, bool resizable) {
  instance_ = instance;
  allocator_ = allocator;
  surface_ = surface;
  extent_ = extent;
  resizable_ = resizable;
  return {};
}

expected<NativeObject> VulkanSurface::native_object(NativeObjectType type) const {
  switch (type) {
    case NativeObjectType::kVkSurfaceKHR:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(surface_),
      };
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] Unsupported native object type for VulkanSurface."});
  }
}

}  // namespace holoviz::vulkan
