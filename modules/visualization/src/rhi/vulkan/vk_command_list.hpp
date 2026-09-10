/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "rhi/detail/command_list_impl.hpp"

#include "vk_device.hpp"

namespace holoviz::rhi::vulkan {

class VulkanRenderPass;

class VulkanCommandList final : public detail::CommandListImpl {
 public:
  VulkanCommandList() = default;
  ~VulkanCommandList() override;

  expected<void> init(const VulkanDevice& device, const CommandListDesc& desc);

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  expected<void> begin() override;
  expected<void> end() override;

  expected<void> begin_render(detail::RenderPassImpl& pass, Rect2D render_area) override;
  expected<void> end_render() override;

  expected<void> bind_vertex_buffer(uint32_t slot, BufferSlice slice) override;
  expected<void> bind_index_buffer(BufferSlice slice, IndexType type) override;
  expected<void> bind_pipeline(detail::PipelineImpl& pipeline) override;

  expected<void> bind_descriptor(const detail::DescriptorImpl& descriptor) override;

  expected<void> draw(uint32_t vertex_count, uint32_t instance_count = 1, uint32_t first_vertex = 0,
                      uint32_t first_instance = 0) override;
  expected<void> draw_indexed(uint32_t index_count, uint32_t instance_count = 1,
                              uint32_t first_index = 0, int32_t vertex_offset = 0,
                              uint32_t first_instance = 0) override;

  expected<void> set_push_data(const detail::PipelineImpl& pipeline,
                               std::span<const std::byte> bytes) override;

  expected<void> set_viewport(Rect2D viewport) override;
  expected<void> set_scissor(Rect2D scissor) override;

  expected<void> reset_queries(detail::QueryPoolImpl& pool, uint32_t first,
                               uint32_t count) override;
  expected<void> write_timestamp(detail::QueryPoolImpl& pool, uint32_t index,
                                 TimestampLocation location) override;
  expected<void> resolve_queries(detail::QueryPoolImpl& pool, uint32_t first,
                                 uint32_t count) override;

  expected<void> reset(std::chrono::nanoseconds timeout) override;

  inline void set_submit_timeline(uint64_t timeline) { submit_timestamp_ = timeline; }

  expected<NativeObject> native_object(NativeObjectType type) const override;

 private:
  const VulkanDevice* device_ = nullptr;
  bool registered_with_device_ = false;  // see VulkanDevice::register_child()

  CommandListType type_;

  uint64_t submit_timestamp_ = 0;

  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;

  // The pass object for the active render scope; end_render reads its bound
  // attachments to restore offscreen render targets to their home layout.
  const VulkanRenderPass* current_render_pass_object_ = nullptr;
};

}  // namespace holoviz::rhi::vulkan
