/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vk_pipeline.hpp"
#include "../detail/destroy_timeout.hpp"
#include "holoviz/rhi/common.hpp"
#include "rhi/vulkan/vk_render_pass.hpp"
#include "vk_descriptor.hpp"
#include "vk_errors.hpp"

#include "vk_enum_conversions.hpp"

#include <slang.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace holoviz::rhi::vulkan {

expected<void> VulkanPipeline::init(const VulkanDevice& device, const PipelineDesc& desc) {
  if (desc.shader_blob.ir != ShaderIR::kSpirv) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Only SPIR-V shaders are supported in VulkanPipeline."});
  }

  device_ = &device;

  device_->register_child();
  registered_with_device_ = true;
  attributes_ = desc.shader_blob.attributes;
  bindings_ = desc.shader_blob.bindings;
  push_constant_size_ = desc.shader_blob.push_constant_size;

  VkShaderModuleCreateInfo vertex_shader_module_info{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = desc.shader_blob.vertex_bytes.size(),
      .pCode = reinterpret_cast<const uint32_t*>(desc.shader_blob.vertex_bytes.data()),
  };

  VIZ_VK_DEVICE_FUNC(vkCreateShaderModule,
                     device_,
                     &vertex_shader_module_info,
                     device.allocator(),
                     &vertex_shader_module_);

  VkShaderModuleCreateInfo fragment_shader_module_info{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = desc.shader_blob.fragment_bytes.size(),
      .pCode = reinterpret_cast<const uint32_t*>(desc.shader_blob.fragment_bytes.data()),
  };
  VIZ_VK_DEVICE_FUNC(vkCreateShaderModule,
                     device_,
                     &fragment_shader_module_info,
                     device.allocator(),
                     &fragment_shader_module_);

  create_state_.shader_stages[0] = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vertex_shader_module_,
      .pName = "main",  // Slang always emits SPIR-V entry points as "main"
  };

  create_state_.shader_stages[1] = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = fragment_shader_module_,
      .pName = "main",  // Slang always emits SPIR-V entry points as "main"
  };

  create_state_.vertex_attribute_descriptions = {};

  uint32_t stride = 0;
  auto sorted_attributes = desc.shader_blob.attributes;  // pack in location order
  std::sort(sorted_attributes.begin(), sorted_attributes.end(), [](auto& a, auto& b) {
    return a.location < b.location;
  });
  for (const auto& a : sorted_attributes) {
    auto format_result = from_holoviz_format(a.format);
    if (!format_result) {
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Failed to convert Slang format to Format: " +
                           format_result.error().message});
    }

    create_state_.vertex_attribute_descriptions.push_back(VkVertexInputAttributeDescription{
        .location = a.location, .binding = 0, .format = format_result.value(), .offset = stride});
    stride += a.byte_size;
  }

  auto vertex_step_mode_result = from_holoviz_vertex_step_mode(desc.step);
  if (!vertex_step_mode_result) {
    return make_unexpected(Error{
        .code = ErrorCode::kFailure,
        .message =
            "[holoviz] Failed to convert Slang vertex step mode to Vulkan vertex input rate: " +
            vertex_step_mode_result.error().message});
  }

  create_state_.buffer_binding = {
      .binding = 0, .stride = stride, .inputRate = vertex_step_mode_result.value()};

  create_state_.vertex_input_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount =
          create_state_.vertex_attribute_descriptions.empty() ? 0u : 1u,
      .pVertexBindingDescriptions = &create_state_.buffer_binding,
      .vertexAttributeDescriptionCount =
          static_cast<uint32_t>(create_state_.vertex_attribute_descriptions.size()),
      .pVertexAttributeDescriptions = create_state_.vertex_attribute_descriptions.data(),
  };

  auto topology_result = from_holoviz_primitive_topology(desc.topology);
  if (!topology_result) {
    return make_unexpected(Error{
        .code = ErrorCode::kFailure,
        .message =
            "[holoviz] Failed to convert Slang primitive topology to Vulkan primitive topology: " +
            topology_result.error().message});
  }

  create_state_.input_assembly_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = topology_result.value(),
      .primitiveRestartEnable = VK_FALSE,
  };

  create_state_.viewport_state_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = nullptr,  // set dynamically at draw time
      .scissorCount = 1,
      .pScissors = nullptr,  // set dynamically at draw time
  };

  create_state_.dynamic_states[0] = VK_DYNAMIC_STATE_VIEWPORT;
  create_state_.dynamic_states[1] = VK_DYNAMIC_STATE_SCISSOR;
  create_state_.dynamic_state_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = create_state_.dynamic_states,
  };

  auto cull_mode_result = from_holoviz_cull_mode(desc.raster.cull);
  if (!cull_mode_result) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Failed to convert Slang cull mode to Vulkan cull mode: " +
                         cull_mode_result.error().message});
  }

  if (desc.raster.polygon_mode == PolygonMode::kFillRectangle &&
      !device_->capabilities().fill_rectangle) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] Device does not support PolygonMode::kFillRectangle "
                   "(VK_NV_fill_rectangle); check CapabilityReport::fill_rectangle.",
    });
  }
  VIZ_UNWRAP(VkPolygonMode, vk_polygon_mode, from_holoviz_polygon_mode(desc.raster.polygon_mode));

  create_state_.rasterization_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = vk_polygon_mode,
      .cullMode = cull_mode_result.value(),
      .frontFace = VK_FRONT_FACE_CLOCKWISE,
      .depthBiasEnable = VK_FALSE,
      .lineWidth = 1.0f,
  };

  create_state_.multisample_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable = VK_FALSE,
  };

  auto depth_test_result = from_holoviz_compare_op(desc.depth.compare);
  if (!depth_test_result) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Failed to convert Slang compare op to Vulkan compare op: " +
                         depth_test_result.error().message});
  }

  create_state_.depth_stencil_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = desc.depth.test ? VK_TRUE : VK_FALSE,
      .depthWriteEnable = desc.depth.write ? VK_TRUE : VK_FALSE,
      .depthCompareOp = depth_test_result.value(),
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable = VK_FALSE,
  };

  auto blend_enable = desc.blend.enable;
  auto src_factor_result = from_holoviz_blend_factor(desc.blend.src_color);
  if (!src_factor_result) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Failed to convert Slang blend factor to Vulkan blend factor: " +
                         src_factor_result.error().message});
  }
  auto dst_factor_result = from_holoviz_blend_factor(desc.blend.dst_color);
  if (!dst_factor_result) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Failed to convert Slang blend factor to Vulkan blend factor: " +
                         dst_factor_result.error().message});
  }

  auto color_op_result = from_holoviz_blend_op(desc.blend.color_op);
  if (!color_op_result) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Failed to convert Slang blend op to Vulkan blend op: " +
                         color_op_result.error().message});
  }

  auto alpha_src_factor_result = from_holoviz_blend_factor(desc.blend.src_alpha);
  if (!alpha_src_factor_result) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Failed to convert Slang blend factor to Vulkan blend factor: " +
                         alpha_src_factor_result.error().message});
  }
  auto alpha_dst_factor_result = from_holoviz_blend_factor(desc.blend.dst_alpha);
  if (!alpha_dst_factor_result) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Failed to convert Slang blend factor to Vulkan blend factor: " +
                         alpha_dst_factor_result.error().message});
  }

  auto alpha_op_result = from_holoviz_blend_op(desc.blend.alpha_op);
  if (!alpha_op_result) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Failed to convert Slang blend op to Vulkan blend op: " +
                         alpha_op_result.error().message});
  }

  // TODO: fix this to allow for more than one color attachment, and to allow for different blend
  // states per attachment
  create_state_.color_blend_attachments = {{
      .blendEnable = blend_enable ? VK_TRUE : VK_FALSE,
      .srcColorBlendFactor = src_factor_result.value(),
      .dstColorBlendFactor = dst_factor_result.value(),
      .colorBlendOp = color_op_result.value(),
      .srcAlphaBlendFactor = alpha_src_factor_result.value(),
      .dstAlphaBlendFactor = alpha_dst_factor_result.value(),
      .alphaBlendOp = alpha_op_result.value(),
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
  }};

  create_state_.color_blend_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_COPY,
      .attachmentCount = static_cast<uint32_t>(create_state_.color_blend_attachments.size()),
      .pAttachments = create_state_.color_blend_attachments.data(),
      .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
  };

  std::vector<VkDescriptorSetLayoutBinding> desc_set_layouts;
  for (const auto& b : desc.shader_blob.bindings) {
    auto type_result = from_holoviz_binding_kind(b.kind);
    if (!type_result) {
      return make_unexpected(Error{
          .code = ErrorCode::kFailure,
          .message = "[holoviz] Failed to convert Slang binding kind to Vulkan descriptor type: " +
                     type_result.error().message});
    }

    desc_set_layouts.push_back({.binding = b.index,
                                .descriptorType = type_result.value(),
                                .descriptorCount = b.count,
                                .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS});
  }

  VkDescriptorSetLayoutCreateInfo create_info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = static_cast<uint32_t>(desc_set_layouts.size()),
      .pBindings = desc_set_layouts.data(),
  };
  VIZ_VK_DEVICE_FUNC(vkCreateDescriptorSetLayout,
                     device_,
                     &create_info,
                     device_->allocator(),
                     &descriptor_set_layout_);

  VkPushConstantRange push_range{.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
                                 .offset = 0,
                                 .size = desc.shader_blob.push_constant_size};

  VkPipelineLayoutCreateInfo pipeline_layout_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &descriptor_set_layout_,
      .pushConstantRangeCount = desc.shader_blob.push_constant_size ? 1u : 0u,
      .pPushConstantRanges = &push_range,
  };

  VIZ_VK_DEVICE_FUNC(vkCreatePipelineLayout,
                     device_,
                     &pipeline_layout_info,
                     device.allocator(),
                     &pipeline_layout_);

  create_state_.pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = create_state_.shader_stages,
      .pVertexInputState = &create_state_.vertex_input_info,
      .pInputAssemblyState = &create_state_.input_assembly_info,
      .pViewportState = &create_state_.viewport_state_info,
      .pRasterizationState = &create_state_.rasterization_info,
      .pMultisampleState = &create_state_.multisample_info,
      .pDepthStencilState = &create_state_.depth_stencil_info,
      .pColorBlendState = &create_state_.color_blend_info,
      .pDynamicState = &create_state_.dynamic_state_info,
      .layout = pipeline_layout_,
  };

  return {};
}

expected<VkPipeline> VulkanPipeline::get_pipeline(const VulkanRenderPass* render_pass) {
  uint64_t render_pass_id = render_pass->render_pass_id();

  auto it = pipelines_.find(render_pass_id);
  if (it != pipelines_.end()) {
    return it->second;
  }

  VIZ_RHI_UNWRAP_NATIVE(VkRenderPass, vk_render_pass, *render_pass);

  create_state_.pipeline_info.renderPass = vk_render_pass;
  VkPipeline pipeline;
  VIZ_VK_DEVICE_FUNC(vkCreateGraphicsPipelines,
                     device_,
                     VK_NULL_HANDLE,
                     1,
                     &create_state_.pipeline_info,
                     device_->allocator(),
                     &pipeline);

  pipelines_[render_pass_id] = pipeline;
  return pipeline;
}

expected<std::unique_ptr<detail::DescriptorImpl>> VulkanPipeline::create_descriptor() {
  auto descriptor = std::make_unique<VulkanDescriptor>();

  VIZ_FORWARD_ERROR(descriptor->init(*this));

  return std::move(descriptor);
}

expected<uint32_t> VulkanPipeline::binding_name_to_index(const std::string& name) const {
  for (const auto& binding : bindings_) {
    if (binding.name == name) {
      return binding.index;
    }
  }
  return make_unexpected(
      Error{.code = ErrorCode::kFailure, .message = "[holoviz] Binding name not found: " + name});
}

expected<NativeObject> VulkanPipeline::native_object(NativeObjectType type) const {
  switch (type) {
    case NativeObjectType::kVkDevice:
      return NativeObject{
          .type = NativeObjectType::kVkDevice,
          .value = reinterpret_cast<uint64_t>(device_->vk_device()),
      };
    case NativeObjectType::kVkPipelineLayout:
      return NativeObject{
          .type = NativeObjectType::kVkPipelineLayout,
          .value = reinterpret_cast<uint64_t>(pipeline_layout_),
      };
    case NativeObjectType::kVkDescriptorSetLayout:
      return NativeObject{
          .type = NativeObjectType::kVkDescriptorSetLayout,
          .value = reinterpret_cast<uint64_t>(descriptor_set_layout_),
      };
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported native object type for VulkanPipeline."});
  }
}

VulkanPipeline::~VulkanPipeline() {
  if (expected<void> result = destroy(detail::kDestructorDestroyTimeout); !result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] VulkanPipeline: destroy failed in destructor: {}",
                        result.error().message);
    } catch (...) {
    }
  }
}

expected<void> VulkanPipeline::destroy([[maybe_unused]] std::chrono::nanoseconds timeout) {
  for (const auto& [render_pass, pipeline] : pipelines_) {
    VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyPipeline, device_, pipeline, device_->allocator());
  }
  pipelines_.clear();

  if (pipeline_layout_ != VK_NULL_HANDLE) {
    VIZ_VK_DEVICE_FUNC_NO_ERROR(
        vkDestroyPipelineLayout, device_, pipeline_layout_, device_->allocator());
    pipeline_layout_ = VK_NULL_HANDLE;
  }

  if (descriptor_set_layout_ != VK_NULL_HANDLE) {
    VIZ_VK_DEVICE_FUNC_NO_ERROR(
        vkDestroyDescriptorSetLayout, device_, descriptor_set_layout_, device_->allocator());
    descriptor_set_layout_ = VK_NULL_HANDLE;
  }

  if (vertex_shader_module_ != VK_NULL_HANDLE) {
    VIZ_VK_DEVICE_FUNC_NO_ERROR(
        vkDestroyShaderModule, device_, vertex_shader_module_, device_->allocator());
    vertex_shader_module_ = VK_NULL_HANDLE;
  }

  if (fragment_shader_module_ != VK_NULL_HANDLE) {
    VIZ_VK_DEVICE_FUNC_NO_ERROR(
        vkDestroyShaderModule, device_, fragment_shader_module_, device_->allocator());
    fragment_shader_module_ = VK_NULL_HANDLE;
  }

  if (registered_with_device_) {
    device_->unregister_child();
    registered_with_device_ = false;
  }

  return {};
}

}  // namespace holoviz::rhi::vulkan
