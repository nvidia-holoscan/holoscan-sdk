/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "rhi/detail/descriptor_impl.hpp"

#include "vk_pipeline.hpp"

namespace holoviz::rhi::vulkan {

class VulkanDescriptor final : public detail::DescriptorImpl {
 public:
  VulkanDescriptor() = default;
  ~VulkanDescriptor() override;

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  expected<void> init(const VulkanPipeline& pipeline);

  expected<void> bind_buffer(const BufferSlice& slice, uint32_t binding_index) override;
  expected<void> bind_buffer(const BufferSlice& slice, const std::string& binding_name) override;

  expected<void> bind_image(const detail::ImageImpl& image, uint32_t binding_index) override;
  expected<void> bind_image(const detail::ImageImpl& image,
                            const std::string& binding_name) override;

  expected<void> bind_sampler(detail::DeviceImpl& device, const SamplerDesc& sampler,
                              uint32_t binding_index) override;
  expected<void> bind_sampler(detail::DeviceImpl& device, const SamplerDesc& sampler,
                              const std::string& binding_name) override;

  expected<NativeObject> native_object(NativeObjectType type) const override;

 private:
  // Native descriptor writes hardcode the type and a single array element, so every bind
  // must match the pipeline's reflected binding (existence, kind, and non-array count)
  // before anything reaches vkUpdateDescriptorSets.
  [[nodiscard]] expected<void> validate_binding(uint32_t binding_index,
                                                BindingKind expected_kind) const;

  const VulkanDevice* device_ = nullptr;
  bool registered_with_device_ = false;  // see VulkanDevice::register_child()
  const VulkanPipeline* pipeline_ = nullptr;

  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
};

}  // namespace holoviz::rhi::vulkan
