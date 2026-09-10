/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "rhi/detail/surface_impl.hpp"
#include "vk_impls.hpp"  // IWYU pragma: keep

namespace holoviz::rhi::vulkan {

/// Vulkan backend for Surface. Owns the VkSurfaceKHR and destroys it on destruction.
class VulkanSurface final : public detail::SurfaceImpl {
 public:
  VulkanSurface() = default;
  ~VulkanSurface() override;

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  expected<void> init(VkInstance instance, VkAllocationCallbacks* allocator, VkSurfaceKHR surface,
                      Extent2D extent, bool resizable);

  Extent2D extent() const override { return extent_; }
  bool resizable() const override { return resizable_; }
  expected<NativeObject> native_object(NativeObjectType type) const override;

 private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkAllocationCallbacks* allocator_ = nullptr;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  Extent2D extent_{};
  bool resizable_ = false;
};

}  // namespace holoviz::rhi::vulkan
