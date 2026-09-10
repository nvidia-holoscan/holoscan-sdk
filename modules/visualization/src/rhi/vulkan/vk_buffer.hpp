/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "vk_impls.hpp"

#include "rhi/detail/buffer_impl.hpp"
#include "vk_device.hpp"

#include <cuda.h>

namespace holoviz::rhi::vulkan {

class VulkanBuffer final : public detail::BufferImpl {
 public:
  VulkanBuffer() = default;
  ~VulkanBuffer() override;

  expected<void> init(const VulkanDevice& device, const BufferDesc& desc);

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  BufferSlice whole() const override;
  expected<BufferSlice> slice(size_t offset_bytes, size_t size_bytes) const override;
  expected<CUdeviceptr> cuda_device_ptr() const override;
  expected<NativeObject> native_object(NativeObjectType type) const override;

 private:
  const VulkanDevice* device_ = nullptr;
  bool registered_with_device_ = false;  // see VulkanDevice::register_child()
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = VK_NULL_HANDLE;
  VmaAllocationInfo allocation_info_{};
  size_t size_bytes_ = 0;

  bool is_shared_ = false;
  CUexternalMemory external_memory_ = nullptr;  // CUDA external memory handle
  CUdeviceptr device_ptr_ = 0;                  // CUDA device pointer for mapped memory
};

}  // namespace holoviz::rhi::vulkan
