/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The Vulkan surface object for direct-to-display output, kept in its own translation unit so
// its ownership/teardown policy is unit-testable without a GPU: tests compile this TU with fake
// volk entry points and a fake display_systems::close_acquisition_resources
// (tests/unit/test_display_surface_teardown.cpp). Enumeration and surface creation stay in
// vk_display.cpp.

#pragma once

#include <chrono>

#include <holoviz/display.hpp>

#include "detail/surface_impl.hpp"
#include "display_systems/display_ownership.hpp"
#include "vk_impls.hpp"  // IWYU pragma: keep

namespace holoviz::vulkan {

/// Release \p display through vkReleaseDisplayEXT for explicit DRM and non-Thor implicit
/// acquisitions, then close the OS resources backing \p acquisition. Driver-managed
/// ownership ends with swapchain destruction and skips the Vulkan release. The order is
/// load-bearing: the DRM fd backs the acquisition, so closing it first could release the display
/// implicitly and invalidate the handle mid-use. The OS resources are closed even when the driver
/// rejects the release: the caller is giving up ownership either way, and an owner-less fd would
/// otherwise live until process exit. Returns the vkReleaseDisplayEXT result (VK_SUCCESS when the
/// call was skipped).
VkResult release_display_then_close(VkPhysicalDevice physical_device, VkDisplayKHR display,
                                    display_systems::DisplayAcquisition& acquisition);

/// Vulkan backend for a Surface created from a Display. Owns the VkSurfaceKHR and the
/// acquisition policy, releasing an explicit acquisition before closing its DRM fd.
/// Driver-managed ownership belongs to the swapchains, which must be destroyed first.
/// Teardown is terminal: a release the driver rejects is logged
/// and reported as kVulkanError, but the display handle and OS resources are still relinquished,
/// so the public handle empties (Surface::destroy contract) and cleanup runs exactly once.
/// Fixed-extent by construction: resizable() stays false, so swapchains skip all per-frame
/// resize detection.
class VulkanDisplaySurface final : public detail::SurfaceImpl {
 public:
  VulkanDisplaySurface() = default;
  ~VulkanDisplaySurface() override;

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  expected<void> init(VkInstance instance, VkAllocationCallbacks* allocator, VkSurfaceKHR surface,
                      Extent2D extent, VkPhysicalDevice physical_device, VkDisplayKHR display,
                      const display_systems::DisplayAcquisition& acquisition);

  Extent2D extent() const override { return extent_; }
  DisplayAcquisitionMode display_acquisition_mode() const override { return acquisition_.mode; }
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

}  // namespace holoviz::vulkan
