/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/query_impl.hpp"

#include "vk_device.hpp"

namespace holoviz::vulkan {

class VulkanQueryPool final : public detail::QueryPoolImpl {
 public:
  VulkanQueryPool() = default;
  ~VulkanQueryPool() override;

  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  expected<void> init(const VulkanDevice& device, const QueryPoolDesc& desc);

  uint32_t timestamp_count() const override { return timestamp_count_; }
  expected<void> results_ns(uint32_t first, uint32_t count, std::span<uint64_t> out_ns) override;
  expected<NativeObject> native_object(NativeObjectType type) const override;

  VkQueryPool vk_query_pool() const { return query_pool_; }

 private:
  const VulkanDevice* device_ = nullptr;
  bool registered_with_device_ = false;  // see VulkanDevice::register_child()
  uint32_t timestamp_count_ = 0;
  VkQueryPool query_pool_ = VK_NULL_HANDLE;
};

}  // namespace holoviz::vulkan
