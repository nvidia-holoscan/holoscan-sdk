/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../detail/destroy_timeout.hpp"
#include "detail/buffer_impl.hpp"  // IWYU pragma: keep
#include "detail/image_impl.hpp"

#include "vk_descriptor.hpp"
#include "vk_enum_conversions.hpp"
#include "vk_errors.hpp"
#include "vk_image.hpp"

#include <string>
#include <vector>

namespace holoviz::vulkan {

expected<void> VulkanDescriptor::init(const VulkanPipeline& pipeline) {
  device_ = pipeline.device();
  pipeline_ = &pipeline;

  device_->register_child();
  registered_with_device_ = true;

  VIZ_UNWRAP_NATIVE(VkDescriptorSetLayout, descriptor_set_layout, *pipeline_);

  std::vector<VkDescriptorPoolSize> pool_sizes;

  for (const auto& binding : pipeline_->bindings()) {
    VIZ_UNWRAP(VkDescriptorType, vk_descriptor_type, from_holoviz_binding_kind(binding.kind));

    VkDescriptorPoolSize pool_size{.type = vk_descriptor_type, .descriptorCount = binding.count};
    pool_sizes.push_back(pool_size);
  }

  VkDescriptorPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = 1,
                                       .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
                                       .pPoolSizes = pool_sizes.data()};

  VIZ_VK_DEVICE_FUNC(
      vkCreateDescriptorPool, device_, &pool_info, device_->allocator(), &descriptor_pool_);

  VkDescriptorSetAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                         .descriptorPool = descriptor_pool_,
                                         .descriptorSetCount = 1,
                                         .pSetLayouts = &descriptor_set_layout};

  VIZ_VK_DEVICE_FUNC(vkAllocateDescriptorSets, device_, &alloc_info, &descriptor_set_);

  return {};
}

expected<void> VulkanDescriptor::validate_binding(uint32_t binding_index,
                                                  BindingKind expected_kind) const {
  const BindingDesc* binding = nullptr;
  for (const auto& candidate : pipeline_->bindings()) {
    if (candidate.index == binding_index) {
      binding = &candidate;
      break;
    }
  }

  if (binding == nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Binding index " + std::to_string(binding_index) +
                         " does not exist in the pipeline's reflected bindings."});
  }

  if (binding->kind != expected_kind) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Bind method does not match the reflected kind of binding " +
                         std::to_string(binding_index) + " ('" + binding->name + "')."});
  }

  // Descriptor writes update a single array element; reflected array bindings have no
  // bind operation yet (compile_slang_shader() rejects them, but manually constructed
  // ShaderBlobs can still carry one).
  if (binding->count != 1) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Binding " + std::to_string(binding_index) +
                                            " ('" + binding->name +
                                            "') is an array binding, which is not supported."});
  }

  return {};
}

expected<void> VulkanDescriptor::bind_buffer(const BufferSlice& slice, uint32_t binding_index) {
  VIZ_FORWARD_ERROR(validate_binding(binding_index, BindingKind::kStorageBuffer));

  VIZ_UNWRAP_NATIVE(VkDevice, buffer_device, *slice.buffer);
  if (buffer_device != device_->vk_device()) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Buffer device does not match descriptor device."});
  }

  VIZ_UNWRAP_NATIVE(VkBuffer, vk_buffer, *slice.buffer);

  VkDescriptorBufferInfo buffer_info{
      .buffer = vk_buffer, .offset = slice.offset_bytes, .range = slice.size_bytes};

  VkWriteDescriptorSet write_descriptor_set{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_set_,
      .dstBinding = binding_index,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &buffer_info,
  };

  VIZ_VK_DEVICE_FUNC_NO_ERROR(
      vkUpdateDescriptorSets, device_, 1, &write_descriptor_set, 0, nullptr);

  return {};
}

expected<void> VulkanDescriptor::bind_buffer(const BufferSlice& slice,
                                             const std::string& binding_name) {
  VIZ_UNWRAP(uint32_t, binding_index, pipeline_->binding_name_to_index(binding_name));

  return bind_buffer(slice, binding_index);
}

expected<void> VulkanDescriptor::bind_image(const detail::ImageImpl& image,
                                            uint32_t binding_index) {
  VIZ_FORWARD_ERROR(validate_binding(binding_index, BindingKind::kSampledImage));

  VIZ_UNWRAP_NATIVE(VkDevice, image_device, image);
  if (image_device != device_->vk_device()) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Image device does not match descriptor device."});
  }

  const auto* vulkan_image = dynamic_cast<const VulkanImage*>(&image);
  if (vulkan_image == nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Image bind operation requires a valid Vulkan image."});
  }

  // The descriptor write declares SHADER_READ_ONLY_OPTIMAL below; an image pinned in any
  // other home layout (render targets, borrowed swapchain images) would be sampled in a
  // layout it is never in.
  if (vulkan_image->pinned_layout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] bind_image requires an image created with ImageUsage::kSampled."});
  }

  VIZ_UNWRAP_NATIVE(VkImageView, vk_image_view, image);

  VkDescriptorImageInfo image_info{.sampler = VK_NULL_HANDLE,
                                   .imageView = vk_image_view,
                                   .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

  VkWriteDescriptorSet write_descriptor_set{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_set_,
      .dstBinding = binding_index,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .pImageInfo = &image_info,
  };

  VIZ_VK_DEVICE_FUNC_NO_ERROR(
      vkUpdateDescriptorSets, device_, 1, &write_descriptor_set, 0, nullptr);

  return {};
}

expected<void> VulkanDescriptor::bind_image(const detail::ImageImpl& image,
                                            const std::string& binding_name) {
  VIZ_UNWRAP(uint32_t, binding_index, pipeline_->binding_name_to_index(binding_name));

  return bind_image(image, binding_index);
}

expected<void> VulkanDescriptor::bind_sampler(detail::DeviceImpl& device,
                                              const SamplerDesc& sampler, uint32_t binding_index) {
  VIZ_FORWARD_ERROR(validate_binding(binding_index, BindingKind::kSampler));

  VIZ_UNWRAP_NATIVE(VkDevice, vk_device, device);

  if (vk_device != device_->vk_device()) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Sampler device does not match descriptor device."});
  }

  VulkanDevice* vulkan_device = dynamic_cast<VulkanDevice*>(&device);

  VIZ_UNWRAP(VkSampler, vk_sampler, vulkan_device->get_sampler(sampler));

  VkDescriptorImageInfo sampler_info{
      .sampler = vk_sampler, .imageView = VK_NULL_HANDLE, .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED};

  VkWriteDescriptorSet write_descriptor_set{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_set_,
      .dstBinding = binding_index,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
      .pImageInfo = &sampler_info,
  };

  VIZ_VK_DEVICE_FUNC_NO_ERROR(
      vkUpdateDescriptorSets, device_, 1, &write_descriptor_set, 0, nullptr);

  return {};
}

expected<void> VulkanDescriptor::bind_sampler(detail::DeviceImpl& device,
                                              const SamplerDesc& sampler,
                                              const std::string& binding_name) {
  VIZ_UNWRAP(uint32_t, binding_index, pipeline_->binding_name_to_index(binding_name));

  return bind_sampler(device, sampler, binding_index);
}

expected<NativeObject> VulkanDescriptor::native_object(NativeObjectType type) const {
  switch (type) {
    case NativeObjectType::kVkDevice:
      return NativeObject{.type = type, .value = reinterpret_cast<uint64_t>(device_->vk_device())};
    case NativeObjectType::kVkDescriptorSetLayout:
      return NativeObject{
          .type = type,
          .value =
              pipeline_->native_object(NativeObjectType::kVkDescriptorSetLayout).value().value};
    case NativeObjectType::kVkPipelineLayout:
      return NativeObject{
          .type = type,
          .value = pipeline_->native_object(NativeObjectType::kVkPipelineLayout).value().value};
    case NativeObjectType::kVkDescriptorPool:
      return NativeObject{.type = NativeObjectType::kVkDescriptorPool,
                          .value = reinterpret_cast<uint64_t>(descriptor_pool_)};
    case NativeObjectType::kVkDescriptorSet:
      return NativeObject{.type = NativeObjectType::kVkDescriptorSet,
                          .value = reinterpret_cast<uint64_t>(descriptor_set_)};
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported native object type requested."});
  }
}

VulkanDescriptor::~VulkanDescriptor() {
  if (expected<void> result = destroy(detail::kDestructorDestroyTimeout); !result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] VulkanDescriptor: destroy failed in destructor: {}",
                        result.error().message);
    } catch (...) {
    }
  }
}

expected<void> VulkanDescriptor::destroy([[maybe_unused]] std::chrono::nanoseconds timeout) {
  if (descriptor_pool_ != VK_NULL_HANDLE) {
    VIZ_VK_DEVICE_FUNC_NO_ERROR(
        vkDestroyDescriptorPool, device_, descriptor_pool_, device_->allocator());
    descriptor_pool_ = VK_NULL_HANDLE;
  }
  if (registered_with_device_) {
    device_->unregister_child();
    registered_with_device_ = false;
  }
  return {};
}

}  // namespace holoviz::vulkan
