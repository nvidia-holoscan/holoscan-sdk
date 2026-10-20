/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/pipeline_impl.hpp"
#include "vk_device.hpp"
#include "vk_impls.hpp"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace holoviz::vulkan {

class VulkanRenderPass;

class VulkanPipeline final : public detail::PipelineImpl {
 public:
  VulkanPipeline() = default;
  ~VulkanPipeline() override;

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  struct PipelineCreateState {
    std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages{};
    std::vector<VkVertexInputAttributeDescription> vertex_attribute_descriptions;
    VkVertexInputBindingDescription buffer_binding;
    VkPipelineVertexInputStateCreateInfo vertex_input_info;
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info;
    VkPipelineViewportStateCreateInfo viewport_state_info;
    VkPipelineRasterizationStateCreateInfo rasterization_info;
    VkPipelineMultisampleStateCreateInfo multisample_info;
    VkPipelineDepthStencilStateCreateInfo depth_stencil_info;
    std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments;
    VkPipelineColorBlendStateCreateInfo color_blend_info;
    std::array<VkDynamicState, 2> dynamic_states{};
    VkPipelineDynamicStateCreateInfo dynamic_state_info;
    VkGraphicsPipelineCreateInfo pipeline_info;
  };

  expected<void> init(const VulkanDevice& device, const PipelineDesc& desc);

  expected<VkPipeline> get_pipeline(const VulkanRenderPass* render_pass);

  expected<std::unique_ptr<detail::DescriptorImpl>> create_descriptor() override;

  expected<uint32_t> binding_name_to_index(const std::string& name) const override;

  expected<NativeObject> native_object(NativeObjectType type) const override;

  const std::vector<AttributeDesc>& attributes() const { return attributes_; }
  const std::vector<BindingDesc>& bindings() const { return bindings_; }
  uint32_t push_constant_size() const { return push_constant_size_; }
  const VulkanDevice* device() const { return device_; }
  const VolkDeviceTable* fn() const { return device_->fn(); }
  VkDevice vk_device() const { return device_->vk_device(); }

 private:
  const VulkanDevice* device_ = nullptr;
  bool registered_with_device_ = false;  // see VulkanDevice::register_child()

  std::vector<AttributeDesc> attributes_;
  std::vector<BindingDesc> bindings_;
  uint32_t push_constant_size_ = 0;

  VkShaderModule vertex_shader_module_ = VK_NULL_HANDLE;
  VkShaderModule fragment_shader_module_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  PipelineCreateState create_state_{};

  std::map<uint64_t, VkPipeline> pipelines_;
};

}  // namespace holoviz::vulkan
