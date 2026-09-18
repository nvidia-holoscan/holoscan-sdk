/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vk_image.hpp"
#include "../detail/destroy_timeout.hpp"
#include "vk_enum_conversions.hpp"
#include "vk_errors.hpp"

#include "../cuda/cu_enum_conversions.hpp"
#include "../cuda/cu_errors.hpp"
#include "../cuda/cu_scoped_context.hpp"
#include "vulkan/vulkan_core.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace holoviz::vulkan {

expected<void> VulkanImage::init_raw(const VulkanDevice* device, VkImage image,
                                     VkImageView image_view, VkFormat format, VkExtent2D extent) {
  // Only for swapchain images, which are created by the swapchain and not owned by this class
  device_ = device;

  // A resize rebinds surviving wrappers through here; register only on the first bind so
  // the device's child count stays balanced against the single unregister in destroy().
  if (!registered_with_device_) {
    device_->register_child();
    registered_with_device_ = true;
  }
  image_ = image;
  image_view_ = image_view;
  format_ = format;
  extent_ = Extent2D{extent.width, extent.height};
  owns_image_ = false;

  // Swapchain images really live in PRESENT_SRC_KHR: render passes end with
  // finalLayout = PRESENT_SRC_KHR and end_render() skips the restore for this pin.
  // Pinning anything else would feed a wrong oldLayout to any future transfer barrier.
  pinned_access_flags_ = VK_ACCESS_MEMORY_READ_BIT;
  pinned_layout_ = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  return {};
}

expected<void> VulkanImage::init(const VulkanDevice& device, const ImageDesc& desc) {
  if (desc.array_layers > 1) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] VulkanImage: array layers > 1 is not supported yet."});
  }

  if (desc.mip_levels != 1) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] VulkanImage: mip levels != 1 is not supported yet."});
  }

  if (desc.usage == ImageUsage::kNone) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] VulkanImage: usage must be explicitly set by the user."});
  }

  device_ = &device;

  device_->register_child();
  registered_with_device_ = true;

  VIZ_UNWRAP(VkFormat, vk_format, from_holoviz_format(desc.format));
  format_ = vk_format;
  extent_ = desc.extent;
  owns_image_ = true;

  bool render_target = desc.usage == ImageUsage::kRenderTarget;

  VkImageUsageFlags usage_flags = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

  pinned_access_flags_ = VK_ACCESS_SHADER_READ_BIT;
  pinned_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  if (render_target) {
    usage_flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    pinned_access_flags_ = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    pinned_layout_ = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  }

  VkImageCreateInfo image_info{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format_,
      .extent =
          VkExtent3D{
              .width = static_cast<uint32_t>(extent_.width),
              .height = static_cast<uint32_t>(extent_.height),
              .depth = 1,
          },
      .mipLevels = desc.mip_levels,
      .arrayLayers = desc.array_layers,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage_flags,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  if (desc.share == ShareMode::kExported) {
    is_shared_ = true;

    VkExternalMemoryImageCreateInfo external_mem_image_create_info{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR,
        .pNext = nullptr,
#if defined(_WIN32)
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
#else
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
#endif
    };

    image_info.pNext = &external_mem_image_create_info;

    VIZ_VK_DEVICE_FUNC(vkCreateImage, device_, &image_info, device_->allocator(), &image_);

    VkMemoryRequirements memoryRequirements;
    VIZ_VK_DEVICE_FUNC_NO_ERROR(vkGetImageMemoryRequirements, device_, image_, &memoryRequirements);

    VkMemoryDedicatedAllocateInfoKHR dedicated_alloc_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR,
        .pNext = nullptr,
        .image = image_,
        .buffer = VK_NULL_HANDLE,
    };

    VkExportMemoryAllocateInfoKHR export_memory_allocate_info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
        .pNext = &dedicated_alloc_info,
#if defined(_WIN32)
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
#else
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
#endif
    };

    VIZ_UNWRAP(
        VkDeviceMemory,
        allocated_memory,
        device.allocate_device_memory(
            memoryRequirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &export_memory_allocate_info));
    memory_ = allocated_memory;

    VIZ_VK_DEVICE_FUNC(vkBindImageMemory, device_, image_, memory_, 0);
#if defined(_WIN32)
    VkMemoryGetWin32HandleInfoKHR get_win32_handle_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
        .memory = memory_,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
    };
    HANDLE handle = nullptr;
    VIZ_VK_DEVICE_FUNC(vkGetMemoryWin32HandleKHR, device_, &get_win32_handle_info, &handle);

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC external_memory_handle_desc{
        .type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32,
        .handle = {.win32 = {.handle = handle}},
        .size = memoryRequirements.size,
        .flags = CUDA_EXTERNAL_MEMORY_DEDICATED,
    };
#else

    VkMemoryGetFdInfoKHR memory_get_fd_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = memory_,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
    };

    int fd = -1;
    VIZ_VK_DEVICE_FUNC(vkGetMemoryFdKHR, device_, &memory_get_fd_info, &fd);

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC external_memory_handle_desc{
        .type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
        .handle = {.fd = fd},
        .size = memoryRequirements.size,
        .flags = CUDA_EXTERNAL_MEMORY_DEDICATED,
    };
#endif

    const bool import_failed =
        VIZ_CU_CHECK(cuImportExternalMemory(&cuda_external_memory_, &external_memory_handle_desc));
#if defined(_WIN32)
    // CUDA never takes ownership of opaque Win32 handles — close ours regardless of outcome.
    CloseHandle(handle);
#else
    // Ownership of the fd transfers to CUDA only on successful import.
    if (import_failed) {
      close(fd);
    }
#endif
    if (import_failed) {
      return make_unexpected(Error{ErrorCode::kFailure, "cuImportExternalMemory failed"});
    }

    VIZ_UNWRAP(cuda::CuArrayFormat, cu_format, cuda::from_holoviz_format(desc.format));

    CUDA_ARRAY3D_DESCRIPTOR array3d_desc{
        .Width = extent_.width,
        .Height = extent_.height,
        .Depth = 0,
        .Format = cu_format.format,
        .NumChannels = cu_format.num_channels,
        .Flags = (render_target ? CUDA_ARRAY3D_COLOR_ATTACHMENT
                                : 0u),  // |
                                        // (storage_load_store ? CUDA_ARRAY3D_SURFACE_LDST : 0u),
    };

    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC external_mipmapped_desc{
        .offset = 0,
        .arrayDesc = array3d_desc,
        .numLevels = desc.mip_levels,
    };

    VIZ_CU_FORWARD_ERROR(cuExternalMemoryGetMappedMipmappedArray(
        &cuda_mipmapped_array_, cuda_external_memory_, &external_mipmapped_desc));
  } else {
    is_shared_ = false;

    VmaAllocationCreateInfo allocation_create_info = {
        .flags = render_target ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : 0u,
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };

    VIZ_VK_FORWARD_ERROR(vmaCreateImage(device.vma_allocator(),
                                        &image_info,
                                        &allocation_create_info,
                                        &image_,
                                        &allocation_,
                                        &allocation_info_));
  }

  VkImageViewCreateInfo view_info{
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image_,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format_,
      .subresourceRange =
          // NOLINTNEXTLINE(whitespace/braces)
      {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,  // TODO: support depth/stencil images
          .baseMipLevel = 0,
          .levelCount = desc.mip_levels,
          .baseArrayLayer = 0,
          .layerCount = desc.array_layers,
      },
  };

  VIZ_VK_DEVICE_FUNC(vkCreateImageView, device_, &view_info, device.allocator(), &image_view_);

  VIZ_UNWRAP(VkCommandBuffer, cmd_buffer, device.start_cmd_buffer_record());

  VkImageMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = 0,
      .dstAccessMask = pinned_access_flags_,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = pinned_layout_,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image_,
      .subresourceRange =
          // NOLINTNEXTLINE(whitespace/braces)
      {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel = 0,
          .levelCount = desc.mip_levels,
          .baseArrayLayer = 0,
          .layerCount = desc.array_layers,
      },
  };

  // Nothing has touched a freshly created image, so nothing needs to complete
  // before the transition; only later use must wait for it.
  device_->fn()->vkCmdPipelineBarrier(cmd_buffer,
                                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                      0,
                                      0,
                                      nullptr,
                                      0,
                                      nullptr,
                                      1,
                                      &barrier);

  VIZ_FORWARD_ERROR(device.submit_cmd_buffer(cmd_buffer));

  return {};
}

VulkanImage::~VulkanImage() {
  if (expected<void> result = destroy(detail::kDestructorDestroyTimeout); !result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] VulkanImage: destroy failed in destructor: {}",
                        result.error().message);
    } catch (...) {
    }
  }
}

expected<void> VulkanImage::destroy([[maybe_unused]] std::chrono::nanoseconds timeout) {
  if (!owns_image_) {
    if (registered_with_device_) {
      device_->unregister_child();
      registered_with_device_ = false;
    }
    device_ = nullptr;
    image_ = VK_NULL_HANDLE;
    image_view_ = VK_NULL_HANDLE;
    return {};
  }

  expected<void> result;

  if (image_view_ != VK_NULL_HANDLE) {
    // Evict cached framebuffers over this view before it dies: a later view
    // with a recycled handle value must not match a stale cache entry.
    device_->notify_image_view_destroyed(image_view_);
    VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyImageView, device_, image_view_, device_->allocator());
    image_view_ = VK_NULL_HANDLE;
  }

  if (is_shared_) {
    // Teardown may run with no CUDA context current; free the imports under the
    // device's context.
    const ScopedCudaContext cuda_scope(device_->cuda_context().value_or(nullptr));
    if (cuda_mipmapped_array_ != nullptr) {
      VIZ_CU_ACCUMULATE_ERROR(result, cuMipmappedArrayDestroy(cuda_mipmapped_array_));
      cuda_mipmapped_array_ = nullptr;
    }

    if (cuda_external_memory_ != nullptr) {
      VIZ_CU_ACCUMULATE_ERROR(result, cuDestroyExternalMemory(cuda_external_memory_));
      cuda_external_memory_ = nullptr;
    }

    if (image_ != VK_NULL_HANDLE) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyImage, device_, image_, device_->allocator());
      image_ = VK_NULL_HANDLE;
    }
    if (memory_ != VK_NULL_HANDLE) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkFreeMemory, device_, memory_, device_->allocator());
      memory_ = VK_NULL_HANDLE;
    }
  } else {
    if (allocation_ != VK_NULL_HANDLE) {
      vmaDestroyImage(device_->vma_allocator(), image_, allocation_);
      image_ = VK_NULL_HANDLE;
      allocation_ = VK_NULL_HANDLE;
    }
  }

  if (registered_with_device_) {
    device_->unregister_child();
    registered_with_device_ = false;
  }

  return result;
}

Extent2D VulkanImage::extent() const {
  return extent_;
}

expected<Format> VulkanImage::format() const {
  return to_holoviz_format(format_);
}

expected<CUmipmappedArray> VulkanImage::cuda_mipmapped_array() const {
  if (!is_shared_) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] VulkanImage is not shared, CUDA mipmapped array is not available."});
  }

  return cuda_mipmapped_array_;
}

expected<CUarray> VulkanImage::cuda_mipmapped_array_get_level(uint32_t level) const {
  if (!is_shared_) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] VulkanImage is not shared, CUDA array level is not available."});
  }

  CUarray array{};
  VIZ_CU_FORWARD_ERROR(cuMipmappedArrayGetLevel(&array, cuda_mipmapped_array_, level));
  return array;
}

expected<NativeObject> VulkanImage::native_object(NativeObjectType type) const {
  switch (type) {
    case NativeObjectType::kVkDevice:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(device_->vk_device()),
      };
    case NativeObjectType::kVkImage:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(image_),
      };
    case NativeObjectType::kVkImageView:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(image_view_),
      };
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] Unsupported native object type for VulkanImage."});
  }
}

expected<VmaAllocationInfo> VulkanImage::allocation_info() const {
  if (!owns_image_) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message =
            "[holoviz] VulkanImage does not own the image, allocation info is not available."});
  }

  return allocation_info_;
}

}  // namespace holoviz::vulkan
