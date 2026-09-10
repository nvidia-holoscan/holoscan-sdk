/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../detail/destroy_timeout.hpp"
#include "holoviz/rhi/common.hpp"
#include "rhi/detail/buffer_impl.hpp"  // IWYU pragma: keep
#include "rhi/detail/descriptor_impl.hpp"

#include "vk_command_list.hpp"
#include "vk_errors.hpp"

#include "vk_enum_conversions.hpp"
#include "vk_image.hpp"
#include "vk_pipeline.hpp"
#include "vk_query.hpp"
#include "vk_render_pass.hpp"
#include "vulkan/vulkan_core.h"

namespace holoviz::rhi::vulkan {

expected<void> VulkanCommandList::init(const VulkanDevice& device, const CommandListDesc& desc) {
  device_ = &device;

  device_->register_child();
  registered_with_device_ = true;

  VIZ_RHI_UNWRAP_NATIVE(VkQueue, queue, device);
  queue_ = queue;
  type_ = desc.type;

  VkCommandPoolCreateInfo pool_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = desc.type == CommandListType::kTransient ? VK_COMMAND_POOL_CREATE_TRANSIENT_BIT : 0u,
      .queueFamilyIndex = device.vulkan_queue_family_index(),
  };

  VIZ_VK_DEVICE_FUNC(vkCreateCommandPool, device_, &pool_info, device.allocator(), &command_pool_);

  VkCommandBufferAllocateInfo alloc_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool_,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };

  VIZ_VK_DEVICE_FUNC(vkAllocateCommandBuffers, device_, &alloc_info, &command_buffer_);

  return {};
}

expected<void> VulkanCommandList::begin() {
  VkCommandBufferUsageFlags usage_flags = 0;

  if (type_ == CommandListType::kTransient) {
    usage_flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  }

  if (type_ == CommandListType::kReusable) {
    usage_flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  }

  if (usage_flags == 0) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure, .message = "[holoviz] Unsupported command list type."});
  }

  VkCommandBufferBeginInfo begin_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = usage_flags,
  };

  VIZ_VK_FORWARD_ERROR(device_->fn()->vkBeginCommandBuffer(command_buffer_, &begin_info));

  return {};
}

expected<void> VulkanCommandList::end() {
  VIZ_VK_FORWARD_ERROR(device_->fn()->vkEndCommandBuffer(command_buffer_));

  return {};
}

expected<void> VulkanCommandList::begin_render(detail::RenderPassImpl& pass, Rect2D render_area) {
  if (current_render_pass_object_ != nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Cannot begin a new render pass while another is active."});
  }

  VIZ_RHI_UNWRAP_NATIVE(VkRenderPass, render_pass, pass);

  VulkanRenderPass& vulkan_pass = dynamic_cast<VulkanRenderPass&>(pass);

  VIZ_UNWRAP(VkFramebuffer,
             framebuffer,
             vulkan_pass.get_framebuffer(render_area.width, render_area.height));

  VkRenderPassBeginInfo render_pass_info{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {.offset = {render_area.x, render_area.y},
                     .extent = {render_area.width, render_area.height}},
      .clearValueCount = static_cast<uint32_t>(vulkan_pass.clear_values().size()),
      .pClearValues = vulkan_pass.clear_values().data(),
  };

  current_render_pass_object_ = &vulkan_pass;

  device_->fn()->vkCmdBeginRenderPass(
      command_buffer_, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

  return {};
}

expected<void> VulkanCommandList::end_render() {
  device_->fn()->vkCmdEndRenderPass(command_buffer_);

  // The pass's fixed finalLayout is PRESENT_SRC_KHR, which is where swapchain images must
  // end up; offscreen render targets, however, are pinned to their home layout (see
  // image.hpp), so transition those back to keep the home-state contract observable to
  // readbacks and CUDA consumers.
  if (current_render_pass_object_ != nullptr) {
    for (const VulkanImage* image : current_render_pass_object_->color_images()) {
      if (image == nullptr || image->pinned_layout() == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
        continue;
      }
      if (image->pinned_layout() != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        continue;  // swapchain wrappers and other non-render-target pins stay as presented
      }
      VIZ_RHI_UNWRAP_NATIVE(VkImage, vk_image, *image);
      const VkImageMemoryBarrier barrier{
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
          .dstAccessMask = image->pinned_access_flags(),
          .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
          .newLayout = image->pinned_layout(),
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = vk_image,
          .subresourceRange =
              // NOLINTNEXTLINE(whitespace/braces)
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .baseMipLevel = 0,
              .levelCount = 1,
              .baseArrayLayer = 0,
              .layerCount = 1,
          },
      };
      device_->fn()->vkCmdPipelineBarrier(command_buffer_,
                                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                          0,
                                          0,
                                          nullptr,
                                          0,
                                          nullptr,
                                          1,
                                          &barrier);
    }
  }

  current_render_pass_object_ = nullptr;

  return {};
}

expected<void> VulkanCommandList::bind_vertex_buffer(uint32_t slot, BufferSlice slice) {
  if (current_render_pass_object_ == nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Cannot bind a vertex buffer outside of a render pass."});
  }

  VIZ_RHI_UNWRAP_NATIVE(VkBuffer, vertex_buffers, *slice.buffer);
  VkDeviceSize offsets = slice.offset_bytes;

  device_->fn()->vkCmdBindVertexBuffers(command_buffer_, slot, 1, &vertex_buffers, &offsets);

  return {};
}

expected<void> VulkanCommandList::bind_index_buffer(BufferSlice slice, IndexType type) {
  if (current_render_pass_object_ == nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Cannot bind an index buffer outside of a render pass."});
  }

  VIZ_RHI_UNWRAP_NATIVE(VkBuffer, index_buffer, *slice.buffer);

  VIZ_UNWRAP(VkIndexType, vk_index_type, from_holoviz_index_type(type));

  device_->fn()->vkCmdBindIndexBuffer(
      command_buffer_, index_buffer, slice.offset_bytes, vk_index_type);

  return {};
}

expected<void> VulkanCommandList::bind_pipeline(detail::PipelineImpl& pipeline) {
  if (current_render_pass_object_ == nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Cannot bind a pipeline outside of a render pass."});
  }

  auto* vulkan_pipeline = dynamic_cast<VulkanPipeline*>(&pipeline);
  if (vulkan_pipeline == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Pipeline is not a Vulkan pipeline."});
  }

  VIZ_UNWRAP(VkPipeline, vk_pipeline, vulkan_pipeline->get_pipeline(current_render_pass_object_));

  device_->fn()->vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

  return {};
}

expected<void> VulkanCommandList::bind_descriptor(const detail::DescriptorImpl& descriptor) {
  VIZ_RHI_UNWRAP_NATIVE(VkDescriptorSet, descriptor_set, descriptor);
  VIZ_RHI_UNWRAP_NATIVE(VkPipelineLayout, pipeline_layout, descriptor);

  device_->fn()->vkCmdBindDescriptorSets(command_buffer_,
                                         VK_PIPELINE_BIND_POINT_GRAPHICS,
                                         pipeline_layout,
                                         0,
                                         1,
                                         &descriptor_set,
                                         0,
                                         nullptr);

  return {};
}

expected<void> VulkanCommandList::draw(uint32_t vertex_count, uint32_t instance_count,
                                       uint32_t first_vertex, uint32_t first_instance) {
  if (current_render_pass_object_ == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kFailure,
                                 .message = "[holoviz] Cannot draw outside of a render pass."});
  }

  device_->fn()->vkCmdDraw(
      command_buffer_, vertex_count, instance_count, first_vertex, first_instance);

  return {};
}

expected<void> VulkanCommandList::draw_indexed(uint32_t index_count, uint32_t instance_count,
                                               uint32_t first_index, int32_t vertex_offset,
                                               uint32_t first_instance) {
  if (current_render_pass_object_ == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kFailure,
                                 .message = "[holoviz] Cannot draw outside of a render pass."});
  }
  device_->fn()->vkCmdDrawIndexed(
      command_buffer_, index_count, instance_count, first_index, vertex_offset, first_instance);

  return {};
}

expected<void> VulkanCommandList::set_push_data(const detail::PipelineImpl& pipeline,
                                                std::span<const std::byte> bytes) {
  VIZ_RHI_UNWRAP_NATIVE(VkPipelineLayout, pipeline_layout, pipeline);

  const auto* vulkan_pipeline = dynamic_cast<const VulkanPipeline*>(&pipeline);
  if (vulkan_pipeline == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Pipeline is not a Vulkan pipeline."});
  }

  if (bytes.size() > vulkan_pipeline->push_constant_size()) {
    return make_unexpected(Error{
        .code = ErrorCode::kPushConstantOverflow,
        .message = "[holoviz] Push constant data size exceeds pipeline's push constant size."});
  }

  device_->fn()->vkCmdPushConstants(command_buffer_,
                                    pipeline_layout,
                                    VK_SHADER_STAGE_ALL_GRAPHICS,
                                    0,
                                    bytes.size(),
                                    bytes.data());

  return {};
}

expected<void> VulkanCommandList::set_viewport(Rect2D viewport) {
  VkViewport vk_viewport{.x = static_cast<float>(viewport.x),
                         .y = static_cast<float>(viewport.y),
                         .width = static_cast<float>(viewport.width),
                         .height = static_cast<float>(viewport.height),
                         .minDepth = 0.0f,
                         .maxDepth = 1.0f};

  device_->fn()->vkCmdSetViewport(command_buffer_, 0, 1, &vk_viewport);

  return {};
}

expected<void> VulkanCommandList::set_scissor(Rect2D scissor) {
  VkRect2D vk_scissor{
      .offset = {static_cast<int32_t>(scissor.x), static_cast<int32_t>(scissor.y)},
      .extent = {static_cast<uint32_t>(scissor.width), static_cast<uint32_t>(scissor.height)}};

  device_->fn()->vkCmdSetScissor(command_buffer_, 0, 1, &vk_scissor);

  return {};
}

expected<void> VulkanCommandList::reset_queries(detail::QueryPoolImpl& pool, uint32_t first,
                                                uint32_t count) {
  if (current_render_pass_object_ != nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Cannot reset queries inside a render pass."});
  }

  auto* vulkan_pool = dynamic_cast<VulkanQueryPool*>(&pool);
  if (vulkan_pool == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Query pool is not a Vulkan query pool."});
  }

  if (first > vulkan_pool->timestamp_count() || count > vulkan_pool->timestamp_count() - first) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Query reset range is out of bounds."});
  }

  device_->fn()->vkCmdResetQueryPool(command_buffer_, vulkan_pool->vk_query_pool(), first, count);

  return {};
}

expected<void> VulkanCommandList::write_timestamp(detail::QueryPoolImpl& pool, uint32_t index,
                                                  TimestampLocation location) {
  auto* vulkan_pool = dynamic_cast<VulkanQueryPool*>(&pool);
  if (vulkan_pool == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Query pool is not a Vulkan query pool."});
  }

  if (index >= vulkan_pool->timestamp_count()) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Timestamp query index is out of bounds."});
  }

  // kStart samples as soon as prior commands start draining (top of pipe);
  // kEnd samples once everything before it has fully completed.
  const VkPipelineStageFlagBits stage = location == TimestampLocation::kStart
                                            ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                            : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

  device_->fn()->vkCmdWriteTimestamp(command_buffer_, stage, vulkan_pool->vk_query_pool(), index);

  return {};
}

expected<void> VulkanCommandList::resolve_queries(detail::QueryPoolImpl& pool, uint32_t first,
                                                  uint32_t count) {
  if (current_render_pass_object_ != nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Cannot resolve queries inside a render pass."});
  }

  auto* vulkan_pool = dynamic_cast<VulkanQueryPool*>(&pool);
  if (vulkan_pool == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Query pool is not a Vulkan query pool."});
  }

  if (first > vulkan_pool->timestamp_count() || count > vulkan_pool->timestamp_count() - first) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Query resolve range is out of bounds."});
  }

  // Recording no-op: QueryPool::results_ns reads straight from the pool via
  // vkGetQueryPoolResults once the submit completes. A GPU-side
  // vkCmdCopyQueryPoolResults with VK_QUERY_RESULT_WAIT_BIT would stall the
  // queue on the bottom-of-pipe end timestamp — a full pipeline drain per
  // frame that serializes otherwise-overlapping frames. The validation above
  // still runs so misuse fails identically on every backend.
  return {};
}

expected<void> VulkanCommandList::reset(std::chrono::nanoseconds timeout) {
  if (type_ == CommandListType::kReusable) {
    return make_unexpected(Error{.code = ErrorCode::kFailure,
                                 .message = "[holoviz] Cannot reset a reusable command list."});
  }

  uint64_t last_completed = 0;
  VIZ_VK_DEVICE_FUNC(
      vkGetSemaphoreCounterValue, device_, device_->timeline_semaphore(), &last_completed);
  if (last_completed < submit_timestamp_) {
    VkSemaphore timeline_semaphore = device_->timeline_semaphore();
    VkSemaphoreWaitInfo wait_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &timeline_semaphore,
        .pValues = &submit_timestamp_,
    };
    VkResult result =
        device_->fn()->vkWaitSemaphores(device_->vk_device(), &wait_info, timeout.count());

    if (result == VK_TIMEOUT) {
      return make_unexpected(
          Error{.code = ErrorCode::kBackpressured,
                .message = "[holoviz] Timeout while waiting for Vulkan timeline semaphore."});
    }

    VIZ_VK_FORWARD_ERROR(result);
  }

  VIZ_VK_DEVICE_FUNC(vkResetCommandPool, device_, command_pool_, 0);

  // A partially recorded, never-submitted list may be reset mid-render-scope
  // (the frame-abort path); clear the recording state so the next begin_render()
  // doesn't see a phantom active pass. vkResetCommandPool only requires that no
  // buffer in the pool is pending, which the timeline wait above guarantees.
  current_render_pass_object_ = nullptr;

  return {};
}

expected<NativeObject> VulkanCommandList::native_object(NativeObjectType type) const {
  switch (type) {
    case NativeObjectType::kVkDevice:
      return NativeObject{.type = type, .value = reinterpret_cast<uint64_t>(device_->vk_device())};
    case NativeObjectType::kVkQueue:
      return NativeObject{.type = type, .value = reinterpret_cast<uint64_t>(queue_)};
    case NativeObjectType::kVkCommandBuffer:
      return NativeObject{.type = type, .value = reinterpret_cast<uint64_t>(command_buffer_)};
    case NativeObjectType::kVkCommandPool:
      return NativeObject{.type = type, .value = reinterpret_cast<uint64_t>(command_pool_)};
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported native object type requested."});
  }
}

VulkanCommandList::~VulkanCommandList() {
  if (expected<void> result = destroy(detail::kDestructorDestroyTimeout); !result) {
    try {
      HOLOVIZ_LOG_ERROR(
          "[holoviz] VulkanCommandList: destroy failed in destructor, leaking the command pool: "
          "{}",
          result.error().message);
    } catch (...) {
    }
  }
}

expected<void> VulkanCommandList::destroy(std::chrono::nanoseconds timeout) {
  if (command_pool_ == VK_NULL_HANDLE) {
    return {};  // already destroyed or never initialized
  }

  // Refuse to destroy a command list whose last submit has not completed on the
  // GPU; freeing the pool would be a device-loss hazard. The pool is kept so a
  // later destroy() (e.g. after wait_idle()) can succeed.
  uint64_t last_completed = 0;
  VIZ_VK_DEVICE_FUNC(
      vkGetSemaphoreCounterValue, device_, device_->timeline_semaphore(), &last_completed);
  if (last_completed < submit_timestamp_) {
    if (timeout == std::chrono::nanoseconds::zero()) {
      return make_unexpected(
          Error{.code = ErrorCode::kBackpressured,
                .message = "[holoviz] Command list is still in use; destroy aborted."});
    }

    VkSemaphore timeline_semaphore = device_->timeline_semaphore();
    VkSemaphoreWaitInfo wait_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &timeline_semaphore,
        .pValues = &submit_timestamp_,
    };
    VkResult result =
        device_->fn()->vkWaitSemaphores(device_->vk_device(), &wait_info, timeout.count());

    if (result == VK_TIMEOUT) {
      return make_unexpected(Error{
          .code = ErrorCode::kBackpressured,
          .message = "[holoviz] Timed out waiting for the command list to stop being in use."});
    }

    VIZ_VK_FORWARD_ERROR(result);
  }

  VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyCommandPool, device_, command_pool_, device_->allocator());
  command_pool_ = VK_NULL_HANDLE;

  if (registered_with_device_) {
    device_->unregister_child();
    registered_with_device_ = false;
  }

  return {};
}

}  // namespace holoviz::rhi::vulkan
