/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vk_external_semaphore.hpp"
#include "../detail/destroy_timeout.hpp"
#include "vk_errors.hpp"

#include "../cuda/cu_errors.hpp"
#include "../cuda/cu_scoped_context.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <utility>

namespace holoviz::rhi::vulkan {

[[nodiscard]] expected<std::pair<VkSemaphore, CUexternalSemaphore>> make_exported_semaphore_pair(
    const VulkanDevice* device) {
  VkExportSemaphoreCreateInfoKHR export_info{
      .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR,
#if defined(_WIN32)
      .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
#else
      .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
#endif
  };
  VkSemaphoreTypeCreateInfo type_info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .pNext = &export_info,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0,
  };
  VkSemaphoreCreateInfo sem_info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &type_info,
  };
  VkSemaphore timeline_semaphore = VK_NULL_HANDLE;
  VIZ_VK_DEVICE_FUNC(
      vkCreateSemaphore, device, &sem_info, device->allocator(), &timeline_semaphore);
#if defined(_WIN32)
  VkSemaphoreGetWin32HandleInfoKHR get_win32_handle_info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
      .semaphore = timeline_semaphore,
      .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
  };
  HANDLE handle = nullptr;
  VIZ_VK_DEVICE_FUNC(vkGetSemaphoreWin32HandleKHR, device, &get_win32_handle_info, &handle);

  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC cu_sem_desc{
      .type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32,
      .handle = {.win32 = {.handle = handle}},
      .flags = 0,
  };
#else
  VkSemaphoreGetFdInfoKHR get_fd_info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
      .semaphore = timeline_semaphore,
      .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
  };
  int fd = -1;
  VIZ_VK_DEVICE_FUNC(vkGetSemaphoreFdKHR, device, &get_fd_info, &fd);

  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC cu_sem_desc{
      .type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD,
      .handle = {.fd = fd},
      .flags = 0,
  };
#endif
  CUexternalSemaphore external_semaphore = nullptr;
  const bool import_failed =
      VIZ_CU_CHECK(cuImportExternalSemaphore(&external_semaphore, &cu_sem_desc));
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
    VIZ_VK_DEVICE_FUNC_NO_ERROR(
        vkDestroySemaphore, device, timeline_semaphore, device->allocator());
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Failed to import Vulkan timeline semaphore into CUDA."});
  }

  return std::make_pair(timeline_semaphore, external_semaphore);
}

expected<void> VulkanExternalSemaphore::init(const VulkanDevice& device, CUstream stream) {
  device_ = &device;

  device_->register_child();
  registered_with_device_ = true;
  stream_ = stream;

  if (stream_ == nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure, .message = "[holoviz] CUDA stream is null."});
  }

  auto semaphore_pair = make_exported_semaphore_pair(device_);
  if (!semaphore_pair) {
    return make_unexpected(semaphore_pair.error());
  }

  timeline_semaphore_ = semaphore_pair->first;
  external_semaphore_ = semaphore_pair->second;

  return {};
}

expected<void> VulkanExternalSemaphore::wait(const SubmitSyncToken& sync_token) {
  CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait_params{
      .params = {.fence = {.value = sync_token.timeline_value}},
      .flags = 0,
  };

  auto device_external_semaphore = device_->timeline_external_semaphore();

  VIZ_CU_FORWARD_ERROR(
      cuWaitExternalSemaphoresAsync(&device_external_semaphore, &wait_params, 1, stream_));

  return {};
}

expected<ExternalSyncToken> VulkanExternalSemaphore::signal() {
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal_params{
      .params = {.fence = {.value = timeline_value_ + 1}},
      .flags = 0,
  };

  VIZ_CU_FORWARD_ERROR(
      cuSignalExternalSemaphoresAsync(&external_semaphore_, &signal_params, 1, stream_));

  timeline_value_++;

  return ExternalSyncToken{
      .timeline_value = timeline_value_,
      .backend = Backend::kVulkan,
      .impl = this,
  };
}

expected<CUstream> VulkanExternalSemaphore::stream() const {
  return stream_;
}

VulkanExternalSemaphore::~VulkanExternalSemaphore() {
  if (expected<void> result = destroy(detail::kDestructorDestroyTimeout); !result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] VulkanExternalSemaphore: destroy failed in destructor: {}",
                        result.error().message);
    } catch (...) {
    }
  }
}

expected<void> VulkanExternalSemaphore::destroy(std::chrono::nanoseconds timeout) {
  expected<void> result;

  // Refuse to destroy while the semaphore may still be referenced: CUDA requires
  // every enqueued signal to complete before cuDestroyExternalSemaphore, and Vulkan
  // requires every submitted batch waiting on the semaphore to complete before
  // vkDestroySemaphore. Both are observable as timeline counters — this semaphore
  // reaching timeline_value_ covers the signals enqueued by signal(), and the
  // device timeline reaching last_waiter_submit_timeline_ covers the submits that
  // waited on our tokens. (Waits enqueued by wait() reference the device's import,
  // not ours, so they do not gate this teardown.) On kBackpressured nothing has
  // been destroyed, so a later destroy() can retry.
  if (timeline_semaphore_ != VK_NULL_HANDLE) {
    VkSemaphore pending_semaphores[2];
    uint64_t pending_values[2];
    uint32_t pending_count = 0;

    uint64_t last_signaled = 0;
    VIZ_VK_DEVICE_FUNC(vkGetSemaphoreCounterValue, device_, timeline_semaphore_, &last_signaled);
    if (last_signaled < timeline_value_) {
      pending_semaphores[pending_count] = timeline_semaphore_;
      pending_values[pending_count] = timeline_value_;
      pending_count++;
    }

    uint64_t device_completed = 0;
    VIZ_VK_DEVICE_FUNC(
        vkGetSemaphoreCounterValue, device_, device_->timeline_semaphore(), &device_completed);
    if (device_completed < last_waiter_submit_timeline_) {
      pending_semaphores[pending_count] = device_->timeline_semaphore();
      pending_values[pending_count] = last_waiter_submit_timeline_;
      pending_count++;
    }

    if (pending_count > 0) {
      if (timeout == std::chrono::nanoseconds::zero()) {
        return make_unexpected(
            Error{.code = ErrorCode::kBackpressured,
                  .message = "[holoviz] External semaphore is still in use; destroy aborted."});
      }

      VkSemaphoreWaitInfo wait_info{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
          .semaphoreCount = pending_count,
          .pSemaphores = pending_semaphores,
          .pValues = pending_values,
      };
      const VkResult wait_result =
          device_->fn()->vkWaitSemaphores(device_->vk_device(), &wait_info, timeout.count());
      if (wait_result == VK_TIMEOUT) {
        return make_unexpected(Error{.code = ErrorCode::kBackpressured,
                                     .message = "[holoviz] Timed out waiting for the external "
                                                "semaphore to stop being in use."});
      }
      VIZ_VK_FORWARD_ERROR(wait_result);
    }
  }

  if (external_semaphore_ != nullptr) {
    // Teardown may run with no CUDA context current; destroy the import under the
    // device's context.
    const ScopedCudaContext cuda_scope(device_->cuda_context().value_or(nullptr));
    VIZ_CU_ACCUMULATE_ERROR(result, cuDestroyExternalSemaphore(external_semaphore_));
    external_semaphore_ = nullptr;
  }

  if (timeline_semaphore_ != VK_NULL_HANDLE) {
    VIZ_VK_DEVICE_FUNC_NO_ERROR(
        vkDestroySemaphore, device_, timeline_semaphore_, device_->allocator());
    timeline_semaphore_ = VK_NULL_HANDLE;
  }

  if (registered_with_device_) {
    device_->unregister_child();
    registered_with_device_ = false;
  }

  return result;
}

}  // namespace holoviz::rhi::vulkan
