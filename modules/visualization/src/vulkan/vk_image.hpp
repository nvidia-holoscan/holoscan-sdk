/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/image_impl.hpp"
#include "vk_device.hpp"
#include "vk_impls.hpp"

namespace holoviz::vulkan {

class VulkanSampler;

class VulkanImage final : public detail::ImageImpl {
 public:
  VulkanImage() = default;
  ~VulkanImage() override;

  expected<void> init_raw(const VulkanDevice* device, VkImage image, VkImageView image_view,
                          VkFormat format, VkExtent2D extent);

  expected<void> init(const VulkanDevice& device, const ImageDesc& desc);

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  Extent2D extent() const override;
  expected<CUmipmappedArray> cuda_mipmapped_array() const override;
  expected<CUarray> cuda_mipmapped_array_get_level(uint32_t level) const override;

  expected<Format> format() const override;
  expected<NativeObject> native_object(NativeObjectType type) const override;

  expected<VmaAllocationInfo> allocation_info() const;

  VkAccessFlags pinned_access_flags() const { return pinned_access_flags_; }
  VkImageLayout pinned_layout() const { return pinned_layout_; }

  inline VkImage vk_image() const { return image_; }
  inline VkImageView vk_image_view() const { return image_view_; }
  inline VkFormat vk_format() const { return format_; }
  inline bool owns_image() const { return owns_image_; }

 private:
  const VulkanDevice* device_ = nullptr;
  bool registered_with_device_ = false;  // see VulkanDevice::register_child()
  VkImage image_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;

  VmaAllocation allocation_ = VK_NULL_HANDLE;
  VmaAllocationInfo allocation_info_ = {};

  VkAccessFlags pinned_access_flags_{};
  VkImageLayout pinned_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

  VkImageView image_view_ = VK_NULL_HANDLE;
  VkFormat format_ = VK_FORMAT_UNDEFINED;
  Extent2D extent_ = {0, 0};

  bool owns_image_ = false;

  bool is_shared_ = false;
  CUexternalMemory cuda_external_memory_ = nullptr;
  CUmipmappedArray cuda_mipmapped_array_ = nullptr;
};

}  // namespace holoviz::vulkan
