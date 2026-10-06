/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "detail/render_pass_impl.hpp"

#include "vk_device.hpp"

namespace holoviz::vulkan {

class VulkanImage;

class VulkanRenderPass final : public detail::RenderPassImpl {
 public:
  VulkanRenderPass() = default;
  ~VulkanRenderPass() override;

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  expected<void> init(const VulkanDevice& device, const RenderPassDesc& desc);

  expected<void> set_render_image(const detail::ImageImpl& image, uint32_t attachment_index,
                                  Color clear_value) override;
  expected<void> set_depth_image(const detail::ImageImpl& image, float clear_value) override;
  expected<void> prepare_render_target(const detail::ImageImpl& image) override;

  expected<VkFramebuffer> get_framebuffer(uint32_t width, uint32_t height);

  /// Destroy and evict every cached framebuffer that references \p view.
  /// Called (via VulkanDevice::notify_image_view_destroyed) when an image view
  /// is about to be destroyed; at that point the caller has already drained all
  /// submissions referencing the view, which is exactly the precondition
  /// vkDestroyFramebuffer needs. Cached framebuffers whose views are all still
  /// alive are never touched — they may be referenced by in-flight work.
  void invalidate_framebuffers(VkImageView view);

  const std::vector<VkClearValue>& clear_values() const { return set_clear_values_; }

  /// The images currently set as color attachments (may contain nulls for
  /// unset slots). end_render uses them to restore each offscreen render
  /// target to its pinned home layout after the pass's PRESENT_SRC_KHR
  /// final layout.
  const std::vector<const VulkanImage*>& color_images() const { return set_color_images_; }

  /// Per-attachment "preserving pass" flags (kLoad / preserve / shared_present);
  /// begin_render validates that such an attachment is bound to a presentable swapchain
  /// image (the baked initial layout is the presentation home layout, which an
  /// offscreen target never holds).
  const std::vector<bool>& color_requires_home_layout() const {
    return color_requires_home_layout_;
  }

  /// Per-attachment presentation home layout the pass was created for (PRESENT_SRC_KHR, or
  /// SHARED_PRESENT_KHR when RenderTargetDesc::shared_present); begin_render validates it
  /// against the bound swapchain image's pinned layout.
  const std::vector<VkImageLayout>& color_home_layouts() const { return color_home_layouts_; }

  uint64_t render_pass_id() const { return render_pass_id_; }

  expected<NativeObject> native_object(NativeObjectType type) const override;

 private:
  /// One cached framebuffer with an owned copy of the attachment views it was
  /// created from. Entries are evicted by invalidate_framebuffers() when one of
  /// their views dies, so the cache holds only framebuffers over live views and
  /// its size is bounded by the attachment sets actually rendered to.
  struct FramebufferEntry {
    std::vector<VkImageView> color_attachments;
    VkImageView depth_attachment = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;

    inline bool matches(const std::vector<VkImageView>& colors, VkImageView depth, uint32_t w,
                        uint32_t h) const {
      return width == w && height == h && depth_attachment == depth && color_attachments == colors;
    }
  };

  const VulkanDevice* device_ = nullptr;
  bool registered_with_device_ = false;  // see VulkanDevice::register_child()

  std::vector<FramebufferEntry> framebuffers_;

  std::vector<VkClearValue> set_clear_values_;
  std::vector<VkImageView> set_color_attachments_;
  std::vector<const VulkanImage*> set_color_images_;
  std::vector<bool> color_requires_home_layout_;
  std::vector<VkImageLayout> color_home_layouts_;
  VkImageView set_depth_attachment_ = VK_NULL_HANDLE;

  VkRenderPass render_pass_ = VK_NULL_HANDLE;

  uint64_t render_pass_id_ = 0;  // incremented on every init() to invalidate cached framebuffers
};

}  // namespace holoviz::vulkan
