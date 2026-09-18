/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/external_semaphore_impl.hpp"

#include "vk_device.hpp"

namespace holoviz::vulkan {

[[nodiscard]] expected<std::pair<VkSemaphore, CUexternalSemaphore>> make_exported_semaphore_pair(
    const VulkanDevice* device);

class VulkanExternalSemaphore final : public detail::ExternalSemaphoreImpl {
 public:
  VulkanExternalSemaphore() = default;
  ~VulkanExternalSemaphore() override;

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  expected<void> init(const VulkanDevice& device, CUstream stream);

  expected<void> wait(const SubmitSyncToken& sync_token) override;
  expected<ExternalSyncToken> signal() override;
  expected<CUstream> stream() const override;

  const VulkanDevice* device() const { return device_; }
  VkSemaphore timeline_semaphore() const { return timeline_semaphore_; }
  CUexternalSemaphore external_semaphore() const { return external_semaphore_; }

  /// Called by VulkanDevice::submit() when a batch waits on one of our tokens, so
  /// destroy() can drain those submits before releasing timeline_semaphore_.
  void record_submit_timeline(uint64_t value) { last_waiter_submit_timeline_ = value; }

 private:
  const VulkanDevice* device_ = nullptr;
  bool registered_with_device_ = false;  // see VulkanDevice::register_child()
  CUstream stream_ = nullptr;

  VkSemaphore timeline_semaphore_ = VK_NULL_HANDLE;
  CUexternalSemaphore external_semaphore_ = nullptr;
  uint64_t timeline_value_ = 0;
  // Device timeline value of the latest submit() that waited on one of our tokens.
  uint64_t last_waiter_submit_timeline_ = 0;
};

}  // namespace holoviz::vulkan
