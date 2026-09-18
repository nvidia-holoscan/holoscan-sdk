/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../detail/destroy_timeout.hpp"
#include "detail/image_impl.hpp"

#include "vk_render_pass.hpp"

#include "vk_enum_conversions.hpp"
#include "vk_errors.hpp"
#include "vk_image.hpp"

#include <algorithm>
#include <atomic>
#include <vector>

namespace holoviz::vulkan {

static std::atomic<uint64_t> next_render_pass_id{1};

expected<void> VulkanRenderPass::init(const VulkanDevice& device, const RenderPassDesc& desc) {
  if (desc.render_targets.empty()) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] VulkanRenderPass: at least one render target is required."});
  }

  if (desc.render_targets.size() > 1) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] VulkanRenderPass: multiple render targets are not supported yet."});
  }

  device_ = &device;

  device_->register_child();
  registered_with_device_ = true;
  device_->register_render_pass(this);

  set_clear_values_.resize(desc.render_targets.size(), VkClearValue{});
  set_color_attachments_.resize(desc.render_targets.size(), VK_NULL_HANDLE);
  set_color_images_.resize(desc.render_targets.size(), nullptr);

  std::vector<VkAttachmentDescription> attachments;
  for (size_t i = 0; i < desc.render_targets.size(); ++i) {
    VIZ_UNWRAP(VkFormat, vk_format, from_holoviz_format(desc.render_targets[i].format));

    VkAttachmentDescription color_attachment_desc{
        .format = vk_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};
    attachments.push_back(color_attachment_desc);
  }

  std::vector<VkAttachmentReference> color_attachment_refs;
  for (size_t i = 0; i < desc.render_targets.size(); ++i) {
    VkAttachmentReference color_attachment_ref{.attachment = static_cast<uint32_t>(i),
                                               .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    color_attachment_refs.push_back(color_attachment_ref);
  }

  VkSubpassDescription subpass_desc{
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = static_cast<uint32_t>(color_attachment_refs.size()),
      .pColorAttachments = color_attachment_refs.data()};

  VkRenderPassCreateInfo render_pass_info{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = static_cast<uint32_t>(attachments.size()),
      .pAttachments = attachments.data(),
      .subpassCount = 1,
      .pSubpasses = &subpass_desc};

  VIZ_VK_DEVICE_FUNC(
      vkCreateRenderPass, device_, &render_pass_info, device_->allocator(), &render_pass_);

  render_pass_id_ = next_render_pass_id.fetch_add(1, std::memory_order_relaxed);
  return {};
}

expected<void> VulkanRenderPass::set_render_image(const detail::ImageImpl& image,
                                                  uint32_t attachment_index, Color clear_value) {
  VIZ_UNWRAP_NATIVE(VkImageView, image_view, image);

  if (attachment_index >= set_color_attachments_.size()) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Attachment index out of bounds."});
  }

  set_color_attachments_[attachment_index] = image_view;
  set_color_images_[attachment_index] = dynamic_cast<const VulkanImage*>(&image);
  set_clear_values_[attachment_index].color = {
      clear_value.x, clear_value.y, clear_value.z, clear_value.w};

  return {};
}

expected<void> VulkanRenderPass::set_depth_image(const detail::ImageImpl& image,
                                                 float clear_value) {
  return make_unexpected(
      Error{ErrorCode::kNotImplemented,
            "[holoviz] VulkanRenderPass::set_depth_image is not implemented yet."});
}

expected<void> VulkanRenderPass::prepare_render_target(const detail::ImageImpl& image) {
  // Stage the target exactly as begin_render will see it (attachment 0, the image's full
  // extent) and resolve the framebuffer now, so the frame path's get_framebuffer() is a
  // pure cache hit. The staged clear value is a placeholder — every frame re-points the
  // attachment with its own clear color before rendering.
  VIZ_FORWARD_ERROR(set_render_image(image, 0, Color{}));
  const Extent2D extent = image.extent();
  VIZ_FORWARD_ERROR(get_framebuffer(extent.width, extent.height));
  return {};
}

expected<VkFramebuffer> VulkanRenderPass::get_framebuffer(uint32_t width, uint32_t height) {
  for (const FramebufferEntry& entry : framebuffers_) {
    if (entry.matches(set_color_attachments_, set_depth_attachment_, width, height)) {
      return entry.framebuffer;
    }
  }

  VkFramebufferCreateInfo framebuffer_info{
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass_,
      .attachmentCount = static_cast<uint32_t>(set_color_attachments_.size()),
      .pAttachments = set_color_attachments_.data(),
      .width = width,
      .height = height,
      .layers = 1};

  VkFramebuffer framebuffer;
  VIZ_VK_DEVICE_FUNC(
      vkCreateFramebuffer, device_, &framebuffer_info, device_->allocator(), &framebuffer);
  render_targets_created_++;

  framebuffers_.push_back(FramebufferEntry{.color_attachments = set_color_attachments_,
                                           .depth_attachment = set_depth_attachment_,
                                           .width = width,
                                           .height = height,
                                           .framebuffer = framebuffer});

  return framebuffer;
}

void VulkanRenderPass::invalidate_framebuffers(VkImageView view) {
  for (auto it = framebuffers_.begin(); it != framebuffers_.end();) {
    const bool references_view =
        it->depth_attachment == view ||
        std::find(it->color_attachments.begin(), it->color_attachments.end(), view) !=
            it->color_attachments.end();
    if (references_view) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(
          vkDestroyFramebuffer, device_, it->framebuffer, device_->allocator());
      it = framebuffers_.erase(it);
    } else {
      ++it;
    }
  }
}

expected<NativeObject> VulkanRenderPass::native_object(NativeObjectType type) const {
  switch (type) {
    case NativeObjectType::kVkRenderPass:
      return NativeObject{.type = NativeObjectType::kVkRenderPass,
                          .value = reinterpret_cast<uint64_t>(render_pass_)};
    default:
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                   .message = "[holoviz] Invalid native object type requested."});
  }
}

VulkanRenderPass::~VulkanRenderPass() {
  if (expected<void> result = destroy(detail::kDestructorDestroyTimeout); !result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] VulkanRenderPass: destroy failed in destructor: {}",
                        result.error().message);
    } catch (...) {
    }
  }
}

expected<void> VulkanRenderPass::destroy([[maybe_unused]] std::chrono::nanoseconds timeout) {
  if (device_ == nullptr) {
    return {};  // init() never ran (or destroy() already completed).
  }

  for (const FramebufferEntry& entry : framebuffers_) {
    VIZ_VK_DEVICE_FUNC_NO_ERROR(
        vkDestroyFramebuffer, device_, entry.framebuffer, device_->allocator());
  }
  framebuffers_.clear();

  if (render_pass_ != VK_NULL_HANDLE) {
    VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyRenderPass, device_, render_pass_, device_->allocator());
    render_pass_ = VK_NULL_HANDLE;
  }

  // Idempotent: destroy() runs again from the destructor after an explicit
  // destroy(); nulling device_ makes the second pass a no-op (and stops the
  // device notifying a dead pass).
  if (registered_with_device_) {
    device_->unregister_child();
    registered_with_device_ = false;
  }
  device_->unregister_render_pass(this);
  device_ = nullptr;

  return {};
}

}  // namespace holoviz::vulkan
