/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <vector>

#include <holoviz/rhi/display.hpp>

#include "rhi/detail/surface_impl.hpp"
#include "rhi/display_systems/display_ownership.hpp"
#include "vk_impls.hpp"  // IWYU pragma: keep

namespace holoviz::rhi::vulkan {

expected<std::vector<Display>> enumerate_displays(const Adapter& adapter);

expected<std::vector<DisplayMode>> display_modes(const Display& display);
expected<NativeObject> display_native_object(const Display& display, NativeObjectType type);

expected<std::unique_ptr<detail::SurfaceImpl>> display_create_surface(
    const Display& display, const DisplaySurfaceDesc& desc);

/// Vulkan backend for a Surface created from a Display. Owns the VkSurfaceKHR
/// and, when the display was acquired for direct mode, releases the display
/// (and closes the backing OS resources — DRM fd or X connection on Linux) on
/// destruction. Fixed-extent by construction: resizable() stays false, so
/// swapchains skip all per-frame resize detection.
class VulkanDisplaySurface final : public detail::SurfaceImpl {
 public:
  VulkanDisplaySurface() = default;
  ~VulkanDisplaySurface() override;

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  expected<void> init(VkInstance instance, VkAllocationCallbacks* allocator, VkSurfaceKHR surface,
                      Extent2D extent, VkPhysicalDevice physical_device, VkDisplayKHR display,
                      const display_systems::DisplayAcquisition& acquisition);

  Extent2D extent() const override { return extent_; }
  /// Answers kVkSurfaceKHR and kVkDisplayKHR.
  expected<NativeObject> native_object(NativeObjectType type) const override;

 private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkAllocationCallbacks* allocator_ = nullptr;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  Extent2D extent_{};
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDisplayKHR display_ = VK_NULL_HANDLE;
  display_systems::DisplayAcquisition acquisition_{};
};

}  // namespace holoviz::rhi::vulkan
