/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vk_buffer.hpp"
#include "../cuda/cu_errors.hpp"
#include "../cuda/cu_scoped_context.hpp"
#include "../detail/destroy_timeout.hpp"
#include "holoviz/expected.hpp"
#include "vk_errors.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace holoviz::vulkan {

expected<void> VulkanBuffer::init(const VulkanDevice& device, const BufferDesc& desc) {
  VkBufferCreateInfo buffer_info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = desc.size_bytes,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
  };

  device_ = &device;
  size_bytes_ = desc.size_bytes;

  device_->register_child();
  registered_with_device_ = true;

  if (desc.share == ShareMode::kExported) {
    is_shared_ = true;

    VkExternalMemoryBufferCreateInfo external_mem_buffer_create_info{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR,
        .pNext = nullptr,
#if defined(_WIN32)
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
#else
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
#endif
    };

    buffer_info.pNext = &external_mem_buffer_create_info;

    VIZ_VK_DEVICE_FUNC(vkCreateBuffer, device_, &buffer_info, device_->allocator(), &buffer_);

    VkMemoryRequirements memoryRequirements;
    VIZ_VK_DEVICE_FUNC_NO_ERROR(
        vkGetBufferMemoryRequirements, device_, buffer_, &memoryRequirements);

    VkMemoryDedicatedAllocateInfoKHR dedicated_alloc_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR,
        .pNext = nullptr,
        .image = VK_NULL_HANDLE,
        .buffer = buffer_,
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

    VIZ_VK_DEVICE_FUNC(vkBindBufferMemory, device_, buffer_, memory_, 0);

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
        VIZ_CU_CHECK(cuImportExternalMemory(&external_memory_, &external_memory_handle_desc));
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

    CUDA_EXTERNAL_MEMORY_BUFFER_DESC external_memory_buffer_desc{
        .offset = 0, .size = size_bytes_, .flags = 0u};

    VIZ_CU_FORWARD_ERROR(cuExternalMemoryGetMappedBuffer(
        &device_ptr_, external_memory_, &external_memory_buffer_desc));
  } else {
    is_shared_ = false;

    VmaAllocationCreateInfo allocation_create_info = {
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };

    VIZ_VK_FORWARD_ERROR(vmaCreateBuffer(device.vma_allocator(),
                                         &buffer_info,
                                         &allocation_create_info,
                                         &buffer_,
                                         &allocation_,
                                         &allocation_info_));
  }

  return {};
}

BufferSlice VulkanBuffer::whole() const {
  return BufferSlice(*this, 0, size_bytes_);
}

expected<BufferSlice> VulkanBuffer::slice(size_t offset_bytes, size_t size_bytes) const {
  if (offset_bytes > size_bytes_ || size_bytes > size_bytes_ - offset_bytes) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] VulkanBuffer: slice out of range"});
  }

  return BufferSlice(*this, offset_bytes, size_bytes);
}

expected<CUdeviceptr> VulkanBuffer::cuda_device_ptr() const {
  if (!is_shared_) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] VulkanBuffer: cuda_device_ptr() called on non-shared buffer"});
  }

  return device_ptr_;
}

expected<NativeObject> VulkanBuffer::native_object(NativeObjectType type) const {
  switch (type) {
    case NativeObjectType::kVkDevice:
      return NativeObject{.type = NativeObjectType::kVkDevice,
                          .value = reinterpret_cast<uint64_t>(device_->vk_device())};
    case NativeObjectType::kVkBuffer:
      return NativeObject{.type = NativeObjectType::kVkBuffer,
                          .value = reinterpret_cast<uint64_t>(buffer_)};
    case NativeObjectType::kVmaAllocation:
      return NativeObject{.type = NativeObjectType::kVmaAllocation,
                          .value = reinterpret_cast<uint64_t>(allocation_)};
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] VulkanBuffer: unsupported native object type requested"});
  }
}

VulkanBuffer::~VulkanBuffer() {
  if (expected<void> result = destroy(detail::kDestructorDestroyTimeout); !result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] VulkanBuffer: destroy failed in destructor: {}",
                        result.error().message);
    } catch (...) {
    }
  }
}

expected<void> VulkanBuffer::destroy([[maybe_unused]] std::chrono::nanoseconds timeout) {
  expected<void> result;

  if (is_shared_) {
    // Teardown may run with no CUDA context current (e.g. from a destructor on a
    // foreign thread); free the imports under the device's context.
    const ScopedCudaContext cuda_scope(device_->cuda_context().value_or(nullptr));
    if (device_ptr_ != 0) {
      VIZ_CU_ACCUMULATE_ERROR(result, cuMemFree(device_ptr_));
      device_ptr_ = 0;
    }

    if (external_memory_ != nullptr) {
      VIZ_CU_ACCUMULATE_ERROR(result, cuDestroyExternalMemory(external_memory_));
      external_memory_ = nullptr;
    }

    if (buffer_ != VK_NULL_HANDLE) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyBuffer, device_, buffer_, device_->allocator());
      buffer_ = VK_NULL_HANDLE;
    }
    if (memory_ != VK_NULL_HANDLE) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkFreeMemory, device_, memory_, device_->allocator());
      memory_ = VK_NULL_HANDLE;
    }
  } else {
    if (buffer_ != VK_NULL_HANDLE && allocation_ != VK_NULL_HANDLE) {
      vmaDestroyBuffer(device_->vma_allocator(), buffer_, allocation_);
      buffer_ = VK_NULL_HANDLE;
      allocation_ = VK_NULL_HANDLE;
    }
  }

  if (registered_with_device_) {
    device_->unregister_child();
    registered_with_device_ = false;
  }

  return result;
}

}  // namespace holoviz::vulkan
