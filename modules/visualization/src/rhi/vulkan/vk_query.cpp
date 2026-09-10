/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <vector>
#include "../detail/destroy_timeout.hpp"

#include "vk_errors.hpp"
#include "vk_query.hpp"

namespace holoviz::rhi::vulkan {

expected<void> VulkanQueryPool::init(const VulkanDevice& device, const QueryPoolDesc& desc) {
  device_ = &device;

  device_->register_child();
  registered_with_device_ = true;

  if (desc.timestamp_count == 0) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Query pool timestamp count must not be 0."});
  }
  timestamp_count_ = desc.timestamp_count;

  VIZ_RHI_UNWRAP_NATIVE(VkPhysicalDevice, physical_device, device);

  // timestampValidBits is a per-queue-family capability; 0 means the graphics
  // queue cannot timestamp at all.
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &queue_family_count, queue_families.data());

  const uint32_t queue_family_index = device.vulkan_queue_family_index();
  if (queue_family_index >= queue_family_count ||
      queue_families[queue_family_index].timestampValidBits == 0) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Vulkan graphics queue does not support timestamp queries."});
  }
  const uint32_t valid_bits = queue_families[queue_family_index].timestampValidBits;
  valid_bits_mask_ = valid_bits >= 64 ? ~0ULL : (1ULL << valid_bits) - 1;

  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical_device, &properties);
  timestamp_period_ns_ = properties.limits.timestampPeriod;

  VkQueryPoolCreateInfo pool_info{
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = timestamp_count_,
  };

  VIZ_VK_DEVICE_FUNC(vkCreateQueryPool, device_, &pool_info, device_->allocator(), &query_pool_);

  return {};
}

expected<void> VulkanQueryPool::results_ns(uint32_t first, uint32_t count,
                                           std::span<uint64_t> out_ns) {
  if (first > timestamp_count_ || count > timestamp_count_ - first) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Query result range is out of bounds."});
  }
  if (out_ns.size() < count) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Query result span is smaller than the requested range."});
  }

  if (count == 0) {
    return {};
  }

  // Read straight from the pool. Deliberately no VK_QUERY_RESULT_WAIT_BIT:
  // the contract says the resolving submit has completed, so the results are
  // available and this returns immediately; a violated contract surfaces as
  // kBackpressured instead of an open-ended stall.
  const VkResult result =
      device_->fn()->vkGetQueryPoolResults(device_->vk_device(),
                                           query_pool_,
                                           first,
                                           count,
                                           static_cast<size_t>(count) * sizeof(uint64_t),
                                           out_ns.data(),
                                           sizeof(uint64_t),
                                           VK_QUERY_RESULT_64_BIT);
  if (result == VK_NOT_READY) {
    return make_unexpected(
        Error{.code = ErrorCode::kBackpressured,
              .message = "[holoviz] Query results are not available yet; the submit carrying the "
                         "resolve has not completed."});
  }
  VIZ_VK_FORWARD_ERROR(result);

  for (uint32_t i = 0; i < count; ++i) {
    out_ns[i] = static_cast<uint64_t>(static_cast<double>(out_ns[i] & valid_bits_mask_) *
                                      static_cast<double>(timestamp_period_ns_));
  }

  return {};
}

expected<NativeObject> VulkanQueryPool::native_object(NativeObjectType type) const {
  switch (type) {
    case NativeObjectType::kVkQueryPool:
      return NativeObject{.type = type, .value = reinterpret_cast<uint64_t>(query_pool_)};
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Unsupported native object type requested."});
  }
}

VulkanQueryPool::~VulkanQueryPool() {
  if (expected<void> result = destroy(detail::kDestructorDestroyTimeout); !result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] VulkanQueryPool: destroy failed in destructor: {}",
                        result.error().message);
    } catch (...) {
    }
  }
}

expected<void> VulkanQueryPool::destroy([[maybe_unused]] std::chrono::nanoseconds timeout) {
  if (query_pool_ != VK_NULL_HANDLE) {
    VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyQueryPool, device_, query_pool_, device_->allocator());
    query_pool_ = VK_NULL_HANDLE;
  }
  if (registered_with_device_) {
    device_->unregister_child();
    registered_with_device_ = false;
  }
  return {};
}

}  // namespace holoviz::rhi::vulkan
