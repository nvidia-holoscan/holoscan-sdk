/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vk_swapchain.hpp"
#include "../detail/destroy_timeout.hpp"
#include "holoviz/common.hpp"
#include "holoviz/expected.hpp"
#include "vk_enum_conversions.hpp"
#include "vk_errors.hpp"
#include "vk_image.hpp"
#include "vk_swapchain_util.hpp"

#include <magic_enum/magic_enum.hpp>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace holoviz::vulkan {

VulkanAcquireSignalGate::~VulkanAcquireSignalGate() {}

VulkanAcquireSignalGate::VulkanAcquireSignalGate(VulkanAcquireSignalGate&& other) noexcept
    : AcquireSignalGateImpl(),
      device_(other.device_),
      queue_(other.queue_),
      semaphore_(std::exchange(other.semaphore_, VK_NULL_HANDLE)),
      consumed_ptr_(std::exchange(other.consumed_ptr_, nullptr)),
      slot_token_ptr_(std::exchange(other.slot_token_ptr_, nullptr)),
      drain_fence_(std::exchange(other.drain_fence_, VK_NULL_HANDLE)),
      drain_pending_(std::exchange(other.drain_pending_, false)) {
  wait_stage = other.wait_stage;
}

expected<void> VulkanAcquireSignalGate::wait(std::chrono::nanoseconds timeout) {
  if (!drain_pending_) {
    // A pending drain skips both the consumed check and the submit: the queue-side wait on
    // the binary semaphore was irrevocably enqueued by the earlier timed-out wait(), and
    // only the CPU-side fence wait remains.
    if (*consumed_ptr_ == VK_TRUE) {
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] Vulkan acquire signal gate has already been consumed."});
    }

    if (drain_fence_ == VK_NULL_HANDLE) {
      // Persistent and reused across calls; destroying a fence while its drain submit can
      // still be pending would violate Vulkan lifetime, so it lives until the swapchain
      // knows the drain completed (destroy_drain_fence()).
      VkFenceCreateInfo fence_create_info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      VIZ_VK_DEVICE_FUNC(
          vkCreateFence, device_, &fence_create_info, device_->allocator(), &drain_fence_);
    }

    VkPipelineStageFlags wait_stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &semaphore_,
        .pWaitDstStageMask = &wait_stage_flags,
        .commandBufferCount = 0,
        .pCommandBuffers = nullptr,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    VIZ_VK_FORWARD_ERROR(device_->fn()->vkQueueSubmit(queue_, 1, &submit_info, drain_fence_));
    // The queue-side wait on the binary acquire semaphore is irrevocably enqueued the
    // moment the submit succeeds — the gate is consumed even if the CPU-side fence wait
    // below times out (re-waiting the same binary semaphore would be invalid Vulkan).
    *consumed_ptr_ = VK_TRUE;
    drain_pending_ = true;
  }

  const VkResult wait_result = device_->fn()->vkWaitForFences(
      device_->vk_device(), 1, &drain_fence_, VK_TRUE, static_cast<uint64_t>(timeout.count()));
  if (wait_result == VK_TIMEOUT) {
    // drain_pending_ stays set: the fence is retained for the still-pending submit and the
    // next wait() picks up the CPU-side wait without touching the semaphore again.
    return make_unexpected(
        Error{.code = ErrorCode::kBackpressured,
              .message = "[holoviz] Timed out waiting for the acquire gate; the gate was "
                         "consumed queue-side — retry wait() to finish draining."});
  }
  VIZ_VK_FORWARD_ERROR(wait_result);
  drain_pending_ = false;
  VIZ_VK_DEVICE_FUNC(vkResetFences, device_, 1, &drain_fence_);

  return {};
}

expected<void> VulkanAcquireSignalGate::drain_pending_wait(std::chrono::nanoseconds timeout) {
  if (!drain_pending_) {
    return {};
  }
  const VkResult wait_result = device_->fn()->vkWaitForFences(
      device_->vk_device(), 1, &drain_fence_, VK_TRUE, static_cast<uint64_t>(timeout.count()));
  if (wait_result == VK_TIMEOUT) {
    return make_unexpected(
        Error{.code = ErrorCode::kBackpressured,
              .message = "[holoviz] Timed out waiting for a pending acquire-gate drain."});
  }
  VIZ_VK_FORWARD_ERROR(wait_result);
  drain_pending_ = false;
  VIZ_VK_DEVICE_FUNC(vkResetFences, device_, 1, &drain_fence_);
  return {};
}

void VulkanAcquireSignalGate::destroy_drain_fence() {
  if (drain_fence_ != VK_NULL_HANDLE) {
    VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyFence, device_, drain_fence_, device_->allocator());
    drain_fence_ = VK_NULL_HANDLE;
    drain_pending_ = false;
  }
}

expected<bool> VulkanAcquireSignalGate::consumed() const {
  return *consumed_ptr_ == VK_TRUE;
}

expected<void> VulkanAcquireSignalGate::consume() {
  *consumed_ptr_ = VK_TRUE;
  return {};
}

expected<NativeObject> VulkanAcquireSignalGate::native_object(NativeObjectType type) const {
  switch (type) {
    case NativeObjectType::kVkDevice:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(device_->vk_device()),
      };
    case NativeObjectType::kVkQueue:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(queue_),
      };
    case NativeObjectType::kVkSemaphore:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(semaphore_),
      };
    default:
      return make_unexpected(Error{
          .code = ErrorCode::kInvalidArgument,
          .message = "[holoviz] Unsupported native object type for VulkanAcquireSignalGate."});
  }
}

VulkanPresentWaitGate::~VulkanPresentWaitGate() {}

VulkanPresentWaitGate::VulkanPresentWaitGate(VulkanPresentWaitGate&& other) noexcept
    : PresentWaitGateImpl(),
      device_(other.device_),
      queue_(other.queue_),
      semaphore_(std::exchange(other.semaphore_, VK_NULL_HANDLE)),
      consumed_ptr_(std::exchange(other.consumed_ptr_, nullptr)),
      slot_token_ptr_(std::exchange(other.slot_token_ptr_, nullptr)) {}

expected<void> VulkanPresentWaitGate::signal() {
  if (*consumed_ptr_ == VK_TRUE) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Vulkan present wait gate has already been consumed."});
  }

  // Route through the device so the submit also signals the device timeline —
  // a raw fenceless vkQueueSubmit here would be unobservable CPU-side and force
  // VulkanDevice::destroy() into an unbounded vkDeviceWaitIdle.
  VIZ_UNWRAP(uint64_t, timeline_value, device_->submit_semaphore_signal(semaphore_));

  // The arm submit references the present semaphore; record it on the frame slot so a
  // resize()/destroy() that arrives before present() waits it out.
  record_submit_timeline(timeline_value);

  *consumed_ptr_ = VK_TRUE;

  return {};
}

expected<bool> VulkanPresentWaitGate::consumed() const {
  return *consumed_ptr_ == VK_TRUE;
}

expected<void> VulkanPresentWaitGate::consume() {
  *consumed_ptr_ = VK_TRUE;
  return {};
}

expected<NativeObject> VulkanPresentWaitGate::native_object(NativeObjectType type) const {
  switch (type) {
    case NativeObjectType::kVkDevice:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(device_->vk_device()),
      };
    case NativeObjectType::kVkQueue:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(queue_),
      };
    case NativeObjectType::kVkSemaphore:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(semaphore_),
      };
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] Unsupported native object type for VulkanPresentWaitGate."});
  }
}

VulkanSwapchain::~VulkanSwapchain() {
  expected<void> result = destroy(detail::kDestructorDestroyTimeout);
  if (!result && result.error().code == ErrorCode::kBackpressured) {
    // Terminal escalation — the destructor cannot retry later. Presentation completes in
    // bounded time on a healthy driver, so one generous re-wait normally succeeds.
    try {
      HOLOVIZ_LOG_WARN(
          "[holoviz] VulkanSwapchain: presentation still in flight at destruction; waiting "
          "up to 5 s.");
    } catch (...) {
    }
    result = destroy(std::chrono::seconds(5));
    if (!result && result.error().code == ErrorCode::kBackpressured && device_ != nullptr) {
      // The presentation engine is wedged. Destroying objects it still owns is undefined
      // behavior, so deliberately leak them and make this object inert instead. The leaked
      // objects still reference the VkDevice (with presentation work possibly pending on
      // its queue), so poison the device — its destroy() then leaks the native device too
      // instead of destroying the parent of live objects — and keep the child registration:
      // the leak is a real live dependency and the diagnostic count should say so.
      try {
        HOLOVIZ_LOG_ERROR(
            "[holoviz] VulkanSwapchain: presentation engine never released this swapchain's "
            "presents; leaking its Vulkan objects (the owning device will be leaked too).");
      } catch (...) {
      }
      device_->report_leaked_resources();
      device_ = nullptr;
      return;
    }
  }
  if (!result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] VulkanSwapchain: destroy failed in destructor: {}",
                        result.error().message);
    } catch (...) {
    }
  }
}

expected<void> VulkanSwapchain::destroy(std::chrono::nanoseconds timeout) {
  expected<void> result;

  if (device_ && device_->fn()) {
    // Drain this swapchain's in-flight frame slots before touching anything (same recipe
    // as drain_and_recreate): the semaphores and image views destroyed below must not be
    // referenced by a pending render submit. No 1-second floor here — the destroy()
    // contract (zero timeout => immediate kBackpressured when in use) applies to GPU work;
    // the floor below is only for the presentation engine, which finishes in bounded time.
    // On kBackpressured nothing has been destroyed, so the caller can retry.
    uint64_t wait_value = 0;
    for (const auto& token : slot_submit_sync_tokens_) {
      wait_value = std::max(wait_value, token.timeline_value);
    }
    if (wait_value > 0) {
      VkSemaphore timeline = device_->timeline_semaphore();
      VkSemaphoreWaitInfo wait_info{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
          .semaphoreCount = 1,
          .pSemaphores = &timeline,
          .pValues = &wait_value,
      };
      const VkResult wait_result = device_->fn()->vkWaitSemaphores(
          device_->vk_device(), &wait_info, static_cast<uint64_t>(timeout.count()));
      if (wait_result == VK_TIMEOUT) {
        return make_unexpected(Error{.code = ErrorCode::kBackpressured,
                                     .message =
                                         "[holoviz] Timed out waiting for in-flight frames before "
                                         "destroying the swapchain."});
      }
      VIZ_VK_FORWARD_ERROR(wait_result);
    }

    // A gate wait() that timed out left a drain submit in flight referencing its fence and
    // an acquire semaphore destroyed below; finish those waits first. On kBackpressured
    // nothing has been destroyed and destroy() stays retryable.
    for (auto& gate : acquire_gates_) {
      VIZ_FORWARD_ERROR(gate.drain_pending_wait(timeout));
    }

    // Present fences tell exactly when the presentation engine is done with every present
    // this swapchain (and its retired generations) issued; wait for them instead of
    // trusting the caller's quiesce. The caller's timeout is the bound (zero = immediate
    // poll); expiry returns kBackpressured with nothing destroyed — the presentation
    // engine still owns these objects and tearing them down would violate Vulkan lifetime.
    // Only a lost device proceeds: its pending operations are complete by definition.
    // Ordered BEFORE release_unpresented_acquire(): this is the last wait that can
    // backpressure, and giving the acquired image back first would be irreversible — a
    // kBackpressured return after it would leave the caller's outstanding token able to
    // present an image the presentation engine already owns, breaking the "fully intact
    // and retryable" destroy() contract.
    std::vector<VkFence> outstanding_fences = pending_present_fences_;
    for (const auto& retired : retired_) {
      outstanding_fences.insert(
          outstanding_fences.end(), retired.fences.begin(), retired.fences.end());
    }
    if (!outstanding_fences.empty()) {
      const VkResult wait_result =
          device_->fn()->vkWaitForFences(device_->vk_device(),
                                         static_cast<uint32_t>(outstanding_fences.size()),
                                         outstanding_fences.data(),
                                         VK_TRUE,
                                         static_cast<uint64_t>(timeout.count()));
      if (wait_result == VK_TIMEOUT) {
        return make_unexpected(Error{.code = ErrorCode::kBackpressured,
                                     .message =
                                         "[holoviz] Timed out waiting for the presentation engine "
                                         "to release this swapchain's presents; retry destroy()."});
      }
      if (wait_result != VK_ERROR_DEVICE_LOST) {
        VIZ_VK_FORWARD_ERROR(wait_result);
      }
    }

    // Give an acquired-but-unpresented image back and drain the pending binary signal off
    // its acquire semaphore — that semaphore is destroyed below and must not carry a
    // pending signal operation. Transactional too: its only backpressure exit (the
    // unconsumed-acquire drain) retains the persistent drain fence with swapchain state
    // untouched, and once it releases the image nothing below returns kBackpressured.
    VIZ_FORWARD_ERROR(release_unpresented_acquire(timeout));

    // Every present fence has signaled (or the device is lost), so forced reaping cannot
    // destroy objects the presentation engine still owns.
    reap_retired(/*force=*/true);
    for (auto& gate : acquire_gates_) {
      gate.destroy_drain_fence();
    }
    if (acquire_drain_fence_ != VK_NULL_HANDLE) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(
          vkDestroyFence, device_, acquire_drain_fence_, device_->allocator());
      acquire_drain_fence_ = VK_NULL_HANDLE;
      acquire_drain_pending_ = false;
    }
    for (const auto& fence : pending_present_fences_) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyFence, device_, fence, device_->allocator());
    }
    pending_present_fences_.clear();
    for (const auto& fence : free_present_fences_) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyFence, device_, fence, device_->allocator());
    }
    free_present_fences_.clear();

    for (const auto& image : images_) {
      // Evict cached framebuffers over the dying view (recycled-handle aliasing).
      device_->notify_image_view_destroyed(image->vk_image_view());
      VIZ_VK_DEVICE_FUNC_NO_ERROR(
          vkDestroyImageView, device_, image->vk_image_view(), device_->allocator());
    }

    for (const auto& semaphore : present_finished_semaphores_) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroySemaphore, device_, semaphore, device_->allocator());
    }
    present_finished_semaphores_.clear();

    for (const auto& semaphore : image_available_semaphores_) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroySemaphore, device_, semaphore, device_->allocator());
    }
    image_available_semaphores_.clear();

    // Gates are non-owning views recycled by this swapchain; drop them before
    // the image handles they reference.
    acquire_gates_.clear();
    present_gates_.clear();

    // The swapchain-backed Images borrow their VkImage; their destroy() frees
    // nothing, but propagate any error for uniformity.
    for (auto& image_handle : image_handles_) {
      if (expected<void> image_result = image_handle.destroy(timeout); !image_result && result) {
        result = image_result;
      }
    }
    images_.clear();
    image_handles_.clear();

    if (swapchain_ != VK_NULL_HANDLE) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroySwapchainKHR, device_, swapchain_, device_->allocator());
      swapchain_ = VK_NULL_HANDLE;
    }
    // surface_ is borrowed from the caller's Surface object; do not destroy it here.

    // Idempotence: a second destroy() (the destructor re-runs it) drains nothing and
    // skips teardown entirely — acquire()/resize() already treat a null device_ as
    // destroyed.
    slot_submit_sync_tokens_.clear();
    has_unpresented_acquire_ = false;
    if (registered_with_device_) {
      device_->unregister_child();
      registered_with_device_ = false;
    }
    device_ = nullptr;
  }

  return result;
}

expected<void> VulkanSwapchain::init(const VulkanDevice& device, const SwapchainDesc& desc) {
  if (!desc.surface.has_value()) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] VulkanSwapchain requires a valid surface."});
  }

  const auto& surface = desc.surface.value().get();

  VIZ_UNWRAP_NATIVE(VkPhysicalDevice, vk_physical_device, device);
  VIZ_UNWRAP_NATIVE(VkQueue, vk_queue, device);

  device_ = &device;

  device_->register_child();
  registered_with_device_ = true;
  physical_device_ = vk_physical_device;
  present_queue_ = vk_queue;

  VIZ_UNWRAP(Extent2D, surface_extent, surface.extent());
  VIZ_UNWRAP(bool, surface_resizable, surface.resizable());
  resizable_ = surface_resizable;

  VIZ_UNWRAP_NATIVE(VkSurfaceKHR, vk_surface, surface);
  surface_ = vk_surface;

  // present() runs on the device's single queue; a device created without a surface
  // (Adapter::create_device) may have picked a family that cannot present here. Fail with a
  // precise error now rather than a valid-usage violation at vkQueuePresentKHR.
  VkBool32 queue_can_present = VK_FALSE;
  VIZ_VK_FORWARD_ERROR(vkGetPhysicalDeviceSurfaceSupportKHR(
      vk_physical_device, device.vulkan_queue_family_index(), surface_, &queue_can_present));
  if (queue_can_present != VK_TRUE) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] The device's queue family cannot present to this surface; "
                         "create the device with create_device_for_surface()."});
  }

  // Validate the request against what the (device, surface) pair actually supports: reject
  // an unsupported present mode outright; recreate() clamps the image count and extent to
  // the surface's bounds and validates the scaling request against the surface. The HoloViz
  // stays policy-free — mode fallback (e.g. kFifoLatestReady -> kFifo) is the caller's
  // decision, made against Device::surface_capabilities().
  requested_image_count_ = desc.image_count;
  scaling_ = desc.scaling;

  uint32_t supported_mode_count = 0;
  VIZ_VK_FORWARD_ERROR(vkGetPhysicalDeviceSurfacePresentModesKHR(
      vk_physical_device, surface_, &supported_mode_count, nullptr));
  std::vector<VkPresentModeKHR> supported_modes(supported_mode_count);
  VIZ_VK_FORWARD_ERROR(vkGetPhysicalDeviceSurfacePresentModesKHR(
      vk_physical_device, surface_, &supported_mode_count, supported_modes.data()));

  VIZ_UNWRAP(
      VkPresentModeKHR, requested_present_mode, from_holoviz_present_mode(desc.present_mode));

  const bool mode_reported =
      std::find(supported_modes.begin(), supported_modes.end(), requested_present_mode) !=
      supported_modes.end();
  // A surface may report FIFO_LATEST_READY even when the device extension/feature could not
  // be enabled at device creation; using it would then be invalid.
  const bool mode_usable = mode_reported && (desc.present_mode != PresentMode::kFIFOLatestReady ||
                                             device.fifo_latest_ready_enabled());
  if (!mode_usable) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = std::string("[holoviz] Present mode ") +
                         std::string(magic_enum::enum_name(desc.present_mode)) +
                         " is not supported by this surface/device; query "
                         "Device::surface_capabilities() and pick a supported mode."});
  }
  present_mode_ = requested_present_mode;

  uint32_t surface_format_count = 0;
  VIZ_VK_FORWARD_ERROR(vkGetPhysicalDeviceSurfaceFormatsKHR(
      vk_physical_device, surface_, &surface_format_count, nullptr));
  std::vector<VkSurfaceFormatKHR> surface_formats(surface_format_count);
  VIZ_VK_FORWARD_ERROR(vkGetPhysicalDeviceSurfaceFormatsKHR(
      vk_physical_device, surface_, &surface_format_count, surface_formats.data()));

  VIZ_UNWRAP(VkFormat, desired_format, from_holoviz_format(desc.format));
  VIZ_UNWRAP(VkColorSpaceKHR, desired_color_space, from_holoviz_color_space(desc.color_space));

  VkFormat best_format = VK_FORMAT_UNDEFINED;
  VkColorSpaceKHR best_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

  bool colorspace_found = false;

  for (const auto& surface_format : surface_formats) {
    if (surface_format.format == desired_format &&
        surface_format.colorSpace == desired_color_space) {
      best_format = surface_format.format;
      best_color_space = surface_format.colorSpace;
      colorspace_found = true;
      break;
    }

    if (surface_format.format == desired_format) {
      best_format = surface_format.format;
      best_color_space = surface_format.colorSpace;
      colorspace_found = false;
    }
  }

  // The documented fallback: the format matched but the surface does not offer it with the
  // requested color space, so a different one was substituted. Report the Vulkan value for the
  // substitute rather than mapping it back — the surface may well offer a color space this
  // module does not model, which is exactly when the diagnostic matters most.
  if (!colorspace_found && best_format == desired_format) {
    HOLOVIZ_LOG_WARN(
        "[holoviz] Requested color space {} not supported, using Vulkan color space {} instead.",
        magic_enum::enum_name(desc.color_space),
        static_cast<int32_t>(best_color_space));
  }

  if (best_format == VK_FORMAT_UNDEFINED) {
    return make_unexpected(Error{
        .code = ErrorCode::kFailure,
        .message = "[holoviz] Failed to find a suitable surface format for Vulkan swapchain."});
  }

  swapchain_format_ = best_format;
  swapchain_color_space_ = best_color_space;

  // TODO: Improve robustness of swapchain format selection

  VIZ_FORWARD_ERROR(recreate(surface_extent));
  caller_visible_image_count_ = image_count_;
  return {};
}

expected<void> VulkanSwapchain::recreate(Extent2D desired_extent) {
  // Per-present-mode capability query (VK_EXT_surface_maintenance1): the image-count/extent
  // bounds specific to present_mode_, the modes it can switch to per present, and the
  // scaling behaviors the surface supports. Two calls: the first sizes the mode list.
  VkSurfacePresentModeEXT surface_present_mode{
      .sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_EXT,
      .presentMode = present_mode_,
  };
  const VkPhysicalDeviceSurfaceInfo2KHR surface_info{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
      .pNext = &surface_present_mode,
      .surface = surface_,
  };
  VkSurfacePresentScalingCapabilitiesEXT scaling_caps{
      .sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_EXT,
  };
  VkSurfacePresentModeCompatibilityEXT mode_compatibility{
      .sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_COMPATIBILITY_EXT,
      .pNext = &scaling_caps,
  };
  VkSurfaceCapabilities2KHR surface_caps2{
      .sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR,
      .pNext = &mode_compatibility,
  };
  VIZ_VK_FORWARD_ERROR(
      vkGetPhysicalDeviceSurfaceCapabilities2KHR(physical_device_, &surface_info, &surface_caps2));
  compatible_present_modes_.resize(mode_compatibility.presentModeCount);
  mode_compatibility.pPresentModes = compatible_present_modes_.data();
  // Re-stamp the output chain before the sizing call's second pass: for display
  // surfaces the NVIDIA driver zeroes the chained output structs (sType included),
  // which the next call would then reject as an invalid pNext chain.
  scaling_caps.sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_EXT;
  scaling_caps.pNext = nullptr;
  mode_compatibility.sType = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_COMPATIBILITY_EXT;
  mode_compatibility.pNext = &scaling_caps;
  VIZ_VK_FORWARD_ERROR(
      vkGetPhysicalDeviceSurfaceCapabilities2KHR(physical_device_, &surface_info, &surface_caps2));
  if (compatible_present_modes_.empty()) {
    // The base mode is always compatible with itself; be defensive about drivers that
    // report an empty list.
    compatible_present_modes_.push_back(present_mode_);
  }
  const VkSurfaceCapabilitiesKHR& vk_surface_caps = surface_caps2.surfaceCapabilities;

  uint32_t clamped_image_count = clamp_image_count(
      requested_image_count_, vk_surface_caps.minImageCount, vk_surface_caps.maxImageCount);
  if (fault_injection_.bump_next_recreate_image_count != 0) {
    // Test-only: request a different image count for this one recreation (re-clamped, so
    // the request stays valid) to make an image-count-changing recreation deterministic.
    clamped_image_count =
        clamp_image_count(clamped_image_count + fault_injection_.bump_next_recreate_image_count,
                          vk_surface_caps.minImageCount,
                          vk_surface_caps.maxImageCount);
    fault_injection_.bump_next_recreate_image_count = 0;
  }
  if (clamped_image_count != requested_image_count_) {
    HOLOVIZ_LOG_INFO(
        "[holoviz] Requested swapchain image count {} is outside the surface's supported range "
        "[{}, {}]; clamped to {}.",
        requested_image_count_,
        vk_surface_caps.minImageCount,
        vk_surface_caps.maxImageCount == 0 ? clamped_image_count : vk_surface_caps.maxImageCount,
        clamped_image_count);
  }

  Extent2D extent = desired_extent;
  if (extent.width == 0 || extent.height == 0) {
    // 0xFFFFFFFF is the Vulkan sentinel for "the surface sizes itself to the swapchain"
    // (e.g. Wayland); the caller must then say which size it wants.
    if (vk_surface_caps.currentExtent.width == 0xFFFFFFFFU) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] This surface does not dictate a size; pass the window's "
                           "framebuffer extent to resize()."});
    }
    extent = {vk_surface_caps.currentExtent.width, vk_surface_caps.currentExtent.height};
  }
  extent.width = clamp_extent_dim(
      extent.width, vk_surface_caps.minImageExtent.width, vk_surface_caps.maxImageExtent.width);
  extent.height = clamp_extent_dim(
      extent.height, vk_surface_caps.minImageExtent.height, vk_surface_caps.maxImageExtent.height);
  if (extent.width == 0 || extent.height == 0) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Cannot create a zero-sized swapchain (is the window "
                         "minimized?); skip rendering until the surface has an extent."});
  }

  // Declare every compatible present mode at creation so present() can switch between
  // them per frame (set_present_mode()) without recreating the swapchain.
  VkSwapchainPresentModesCreateInfoEXT present_modes_create_info{
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT,
      .presentModeCount = static_cast<uint32_t>(compatible_present_modes_.size()),
      .pPresentModes = compatible_present_modes_.data(),
  };

  VkSwapchainPresentScalingCreateInfoEXT scaling_create_info{
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT,
  };
  if (scaling_ != SwapchainScaling::kNone) {
    VkPresentScalingFlagsEXT scaling_behavior = 0;
    switch (scaling_) {
      case SwapchainScaling::kOneToOne:
        scaling_behavior = VK_PRESENT_SCALING_ONE_TO_ONE_BIT_EXT;
        break;
      case SwapchainScaling::kStretch:
        scaling_behavior = VK_PRESENT_SCALING_STRETCH_BIT_EXT;
        break;
      case SwapchainScaling::kAspectRatioStretch:
        scaling_behavior = VK_PRESENT_SCALING_ASPECT_RATIO_STRETCH_BIT_EXT;
        break;
      case SwapchainScaling::kNone:
        break;
    }
    if ((scaling_caps.supportedPresentScaling & scaling_behavior) == 0) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = std::string("[holoviz] SwapchainDesc::scaling mode ") +
                           std::string(magic_enum::enum_name(scaling_)) +
                           " is not supported by this surface."});
    }
    // Gravity places the image when it does not fill the surface (one-to-one and
    // aspect-ratio letterboxing). Prefer centered; both fields must be zero or nonzero
    // together, and zero is only valid when the surface reports no gravity support.
    const auto pick_gravity = [](VkPresentGravityFlagsEXT supported) -> VkPresentGravityFlagsEXT {
      if (supported & VK_PRESENT_GRAVITY_CENTERED_BIT_EXT) {
        return VK_PRESENT_GRAVITY_CENTERED_BIT_EXT;
      }
      if (supported & VK_PRESENT_GRAVITY_MIN_BIT_EXT) {
        return VK_PRESENT_GRAVITY_MIN_BIT_EXT;
      }
      if (supported & VK_PRESENT_GRAVITY_MAX_BIT_EXT) {
        return VK_PRESENT_GRAVITY_MAX_BIT_EXT;
      }
      return 0;
    };
    if (scaling_caps.supportedPresentGravityX != 0 && scaling_caps.supportedPresentGravityY != 0) {
      scaling_create_info.presentGravityX = pick_gravity(scaling_caps.supportedPresentGravityX);
      scaling_create_info.presentGravityY = pick_gravity(scaling_caps.supportedPresentGravityY);
    }
    scaling_create_info.scalingBehavior = scaling_behavior;
    present_modes_create_info.pNext = &scaling_create_info;
  }

  // imageUsage/preTransform/compositeAlpha are valid-usage constrained to what the surface
  // reports; derive them from the capabilities instead of assuming the common case.
  if ((vk_surface_caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Surface does not support color-attachment swapchain "
                         "images."});
  }
  VkImageUsageFlags image_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  // Transfer-src is opportunistic (readback/screenshots); rendering does not depend on it.
  image_usage |= vk_surface_caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  VkSurfaceTransformFlagBitsKHR pre_transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  if ((vk_surface_caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) == 0) {
    // currentTransform is always a supported value. The module does no pre-rotation in
    // rendering, so the presentation engine will apply the transform itself.
    pre_transform = vk_surface_caps.currentTransform;
    HOLOVIZ_LOG_WARN(
        "[holoviz] Surface does not support the identity transform; using surface transform "
        "{:#x}. Output is rotated/mirrored by the presentation engine.",
        static_cast<uint32_t>(pre_transform));
  }

  // The module renders opaque output: prefer opaque, then leaving alpha policy to the
  // platform, then the multiplied modes (opaque content is unaffected by either).
  constexpr std::array<VkCompositeAlphaFlagBitsKHR, 4> composite_alpha_preference{
      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
      VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
      VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
  };
  VkCompositeAlphaFlagBitsKHR composite_alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  bool composite_alpha_found = false;
  for (const VkCompositeAlphaFlagBitsKHR candidate : composite_alpha_preference) {
    if (vk_surface_caps.supportedCompositeAlpha & candidate) {
      composite_alpha = candidate;
      composite_alpha_found = true;
      break;
    }
  }
  if (!composite_alpha_found) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Surface reports no supported composite alpha mode."});
  }

  VkSwapchainCreateInfoKHR swapchain_create_info{
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .pNext = &present_modes_create_info,
      .surface = surface_,
      .minImageCount = clamped_image_count,
      .imageFormat = swapchain_format_,
      .imageColorSpace = swapchain_color_space_,
      .imageExtent = {extent.width, extent.height},
      .imageArrayLayers = 1,
      .imageUsage = image_usage,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = pre_transform,
      .compositeAlpha = composite_alpha,
      .presentMode = present_mode_,
      .clipped = VK_TRUE,
      // Smooth handover on resize: the presentation engine keeps showing the retired
      // swapchain's frames until the first present on the new one, so the surface is never
      // swapchain-less.
      .oldSwapchain = swapchain_,
  };

  VkSwapchainKHR new_swapchain = VK_NULL_HANDLE;
  VkResult create_result;
  if (fault_injection_.fail_next_swapchain_create > 0) {
    // Test-only: fail the successor creation while phase 1 below still runs — the exact
    // failed-recreation state (old swapchain retired, generation bumped, no images).
    fault_injection_.fail_next_swapchain_create--;
    create_result = VK_ERROR_INITIALIZATION_FAILED;
  } else {
    create_result = device_->fn()->vkCreateSwapchainKHR(
        device_->vk_device(), &swapchain_create_info, device_->allocator(), &new_swapchain);
  }

  // --- Phase 1: invalidate the old generation, unconditionally. The old swapchain is
  // retired at this point whether or not creation succeeded (a Vulkan rule), and a retired
  // swapchain must never be passed as oldSwapchain again. Everything keyed to it — views,
  // gate storage, trackers, the generation counter — is invalidated here, before any
  // fallible successor work, so no failure below can leave a mixed new/stale generation.
  if (swapchain_ != VK_NULL_HANDLE) {
    // The old image views are referenced only by this module's own submissions, which
    // resize() waited out; the swapchain itself and the semaphores its queued presents may
    // still touch have no such guarantee, so they live on in retired_ (RetiredSwapchain).
    for (const auto& image : images_) {
      // Evict cached framebuffers over the dying view (recycled-handle aliasing).
      device_->notify_image_view_destroyed(image->vk_image_view());
      VIZ_VK_DEVICE_FUNC_NO_ERROR(
          vkDestroyImageView, device_, image->vk_image_view(), device_->allocator());
    }
    RetiredSwapchain retired{.swapchain = swapchain_,
                             .semaphores = std::move(present_finished_semaphores_),
                             .fences = std::move(pending_present_fences_)};
    retired.semaphores.insert(retired.semaphores.end(),
                              image_available_semaphores_.begin(),
                              image_available_semaphores_.end());
    retired_.push_back(std::move(retired));
    swapchain_ = VK_NULL_HANDLE;
  }
  present_finished_semaphores_.clear();
  image_available_semaphores_.clear();
  pending_present_fences_.clear();
  has_unpresented_acquire_ = false;

  // The gates' persistent drain fences are idle here (drain_and_recreate() waited out any
  // pending drain before calling recreate(); init() starts with no gates).
  for (auto& gate : acquire_gates_) {
    gate.destroy_drain_fence();
  }
  acquire_gates_.clear();
  present_gates_.clear();
  image_gate_consumed_.clear();
  slot_submit_sync_tokens_.clear();
  frame_tracker_.clear();
  present_gate_consumed_.clear();
  image_count_ = 0;
  frame_count_ = 0;

  // Bump the generation the moment the old gate storage is invalidated (not at the end of
  // recreate()): outstanding AcquireSyncTokens, gates, and Image references now belong to a
  // dead generation, and holders detect that through generation() instead of dereferencing
  // them. If a later step in this function fails, staleness errs toward "stale too early"
  // (a spurious kInvalidState) rather than "stale too late" (a use-after-free).
  generation_ += 1;

  if (create_result != VK_SUCCESS) {
    // Leave the object image-less but consistent: acquire()/present() answer kOutOfDate,
    // destroy() cleans up, and a later resize() may still succeed (recreating from scratch,
    // losing only presentation continuity).
    images_.clear();
    image_handles_.clear();
  }
  VIZ_VK_FORWARD_ERROR(create_result);

  // --- Phase 2: stage every fallible successor resource in locals. No member state is
  // touched until all of it exists, so a failure destroys the staged objects and the new
  // swapchain and returns with the object exactly in phase 1's lost-but-consistent state
  // (no stale native handles anywhere, retrying resize() recreates from scratch). The
  // early-return macros cannot be used here — they would skip that cleanup.
  uint32_t image_count = 0;
  VkResult stage_result = device_->fn()->vkGetSwapchainImagesKHR(
      device_->vk_device(), new_swapchain, &image_count, nullptr);
  std::vector<VkImage> vk_images(image_count);
  if (stage_result == VK_SUCCESS) {
    stage_result = device_->fn()->vkGetSwapchainImagesKHR(
        device_->vk_device(), new_swapchain, &image_count, vk_images.data());
  }

  std::vector<VkImageView> staged_views;
  staged_views.reserve(image_count);
  for (uint32_t i = 0; stage_result == VK_SUCCESS && i < image_count; ++i) {
    VkImageViewCreateInfo image_view_create_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = vk_images[i],
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = swapchain_format_,
        .components =
            // NOLINTNEXTLINE(whitespace/braces)
        {
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange =
            // NOLINTNEXTLINE(whitespace/braces)
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    VkImageView image_view = VK_NULL_HANDLE;
    stage_result = device_->fn()->vkCreateImageView(
        device_->vk_device(), &image_view_create_info, device_->allocator(), &image_view);
    if (stage_result == VK_SUCCESS) {
      staged_views.push_back(image_view);
    }
  }

  VkSemaphoreCreateInfo semaphore_create_info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
  };
  // Frame-indexed (one slot per frame-in-flight, image_count + 1) acquire semaphores, and
  // image-indexed present-signal semaphores: the latter must be indexed by swapchain image
  // so the presentation engine never sees the same semaphore reused before it re-acquires
  // the corresponding image.
  std::vector<VkSemaphore> staged_image_available;
  staged_image_available.reserve(image_count + 1);
  for (uint32_t i = 0; stage_result == VK_SUCCESS && i < image_count + 1; ++i) {
    VkSemaphore semaphore = VK_NULL_HANDLE;
    stage_result = device_->fn()->vkCreateSemaphore(
        device_->vk_device(), &semaphore_create_info, device_->allocator(), &semaphore);
    if (stage_result == VK_SUCCESS) {
      staged_image_available.push_back(semaphore);
    }
  }
  std::vector<VkSemaphore> staged_present_finished;
  staged_present_finished.reserve(image_count);
  for (uint32_t i = 0; stage_result == VK_SUCCESS && i < image_count; ++i) {
    VkSemaphore semaphore = VK_NULL_HANDLE;
    stage_result = device_->fn()->vkCreateSemaphore(
        device_->vk_device(), &semaphore_create_info, device_->allocator(), &semaphore);
    if (stage_result == VK_SUCCESS) {
      staged_present_finished.push_back(semaphore);
    }
  }

  if (stage_result != VK_SUCCESS) {
    for (const auto& view : staged_views) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyImageView, device_, view, device_->allocator());
    }
    for (const auto& semaphore : staged_image_available) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroySemaphore, device_, semaphore, device_->allocator());
    }
    for (const auto& semaphore : staged_present_finished) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroySemaphore, device_, semaphore, device_->allocator());
    }
    // Nothing was ever acquired from or presented to the successor, so it can be destroyed
    // outright — no retirement needed.
    VIZ_VK_DEVICE_FUNC_NO_ERROR(
        vkDestroySwapchainKHR, device_, new_swapchain, device_->allocator());
    // The wrappers still hold the destroyed old views; drop them so no stale handle
    // survives (identity stability is moot — the generation is already dead).
    images_.clear();
    image_handles_.clear();
    VIZ_VK_FORWARD_ERROR(stage_result);
  }

  // --- Phase 3: commit. Nothing below can fail against the device (init_raw() only stores
  // the raw handles — see VulkanImage::init_raw), so the successor becomes visible whole.
  swapchain_ = new_swapchain;
  extent_ = extent;

  // Image wrappers are identity-stable per index across a resize: surviving indices are
  // rebound in place, so callers' index-keyed resources stay valid.
  if (image_handles_.size() > image_count) {
    images_.resize(image_count);
    image_handles_.resize(image_count);
  }
  for (uint32_t i = 0; i < image_count; ++i) {
    if (i < image_handles_.size()) {
      VIZ_FORWARD_ERROR(images_[i]->init_raw(device_,
                                             vk_images[i],
                                             staged_views[i],
                                             swapchain_format_,
                                             {extent_.width, extent_.height}));
    } else {
      auto vk_img = std::make_unique<VulkanImage>();
      VIZ_FORWARD_ERROR(vk_img->init_raw(device_,
                                         vk_images[i],
                                         staged_views[i],
                                         swapchain_format_,
                                         {extent_.width, extent_.height}));
      images_.push_back(vk_img.get());
      image_handles_.emplace_back(std::move(vk_img));
    }
  }

  image_available_semaphores_ = std::move(staged_image_available);
  present_finished_semaphores_ = std::move(staged_present_finished);
  image_gate_consumed_.assign(image_count + 1, VK_FALSE);
  slot_submit_sync_tokens_.assign(image_count + 1, SubmitSyncToken{});
  frame_tracker_.assign(image_count + 1, static_cast<uint32_t>(-1));
  present_gate_consumed_.assign(image_count, VK_FALSE);

  image_count_ = image_count;
  frame_count_ = image_count + 1;
  current_frame_ = 0;
  current_image_ = 0;

  acquire_gates_.reserve(frame_count_);
  present_gates_.reserve(frame_count_);
  for (uint32_t i = 0; i < frame_count_; ++i) {
    acquire_gates_.emplace_back(device_, present_queue_);
    present_gates_.emplace_back(device_, present_queue_);
  }

  return {};
}

expected<void> VulkanSwapchain::resize(Extent2D new_extent, std::chrono::nanoseconds timeout) {
  // swapchain_ may legitimately be null here: a failed resize() retires the old swapchain
  // without a successor (see recreate()), and retrying is the documented way out.
  if (device_ == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] resize() on a destroyed swapchain."});
  }
  VIZ_FORWARD_ERROR(drain_and_recreate(new_extent, timeout));
  // An explicit resize is the caller's own visibility point for the image count (the docs
  // say to re-query it here), so the automatic paths do not re-report this change.
  caller_visible_image_count_ = image_count_;
  return {};
}

expected<void> VulkanSwapchain::drain_and_recreate(Extent2D new_extent,
                                                   std::chrono::nanoseconds timeout) {
  // Wait out every in-flight submission that renders into the current images. This is
  // narrower than a Device::wait_idle(): only this swapchain's frame slots are drained, and
  // the presentation side needs no drain at all because the old swapchain is handed to the
  // new one via oldSwapchain (recreate()).
  uint64_t wait_value = 0;
  for (const auto& token : slot_submit_sync_tokens_) {
    wait_value = std::max(wait_value, token.timeline_value);
  }
  if (wait_value > 0) {
    VkSemaphore timeline = device_->timeline_semaphore();
    VkSemaphoreWaitInfo wait_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &timeline,
        .pValues = &wait_value,
    };
    const VkResult wait_result = device_->fn()->vkWaitSemaphores(
        device_->vk_device(), &wait_info, static_cast<uint64_t>(timeout.count()));
    if (wait_result == VK_TIMEOUT) {
      return make_unexpected(
          Error{.code = ErrorCode::kBackpressured,
                .message = "[holoviz] Timed out waiting for in-flight frames before resizing "
                           "the swapchain."});
    }
    VIZ_VK_FORWARD_ERROR(wait_result);
  }

  // A gate wait() that timed out left a drain submit in flight referencing its fence and an
  // acquire semaphore this recreation retires; finish those waits first (kBackpressured
  // leaves the swapchain untouched, retrying is fine).
  for (auto& gate : acquire_gates_) {
    VIZ_FORWARD_ERROR(gate.drain_pending_wait(timeout));
  }

  // An acquired-but-unpresented image goes back to the presentation engine; the caller's
  // token for it is invalidated by the recreation either way.
  VIZ_FORWARD_ERROR(release_unpresented_acquire(timeout));

  // Destroy every retired generation whose presents the presentation engine has finished
  // with (its present fences all signaled).
  reap_retired(/*force=*/false);

  return recreate(new_extent);
}

expected<void> VulkanSwapchain::auto_resize(std::chrono::nanoseconds timeout) {
  expected<void> resized = drain_and_recreate({0, 0}, timeout);
  if (!resized) {
    if (resized.error().code == ErrorCode::kBackpressured) {
      return resized;  // the drain timed out; the swapchain is unchanged, retrying is fine
    }
    // Everything else (typically a surface that reports no fixed extent — Wayland — where
    // the automatic path cannot know the window's size) needs the caller's help.
    return make_unexpected(Error{.code = ErrorCode::kOutOfDate,
                                 .message = "[holoviz] Automatic swapchain recreation failed (" +
                                            resized.error().message +
                                            "); call resize() with an explicit extent."});
  }
  return {};
}

void VulkanSwapchain::reap_retired(bool force) {
  // Precondition for force: every retired present fence has signaled or the device is lost
  // (destroy() waits them out first) — otherwise destroying these objects while the
  // presentation engine still owns them would violate Vulkan lifetime.
  while (!retired_.empty()) {
    RetiredSwapchain& retired = retired_.front();
    // Present fences signal in submission order, and generations retire in order, so a
    // front-to-back sweep that stops at the first unsignaled fence visits exactly the
    // destroyable prefix.
    const bool all_signaled =
        std::all_of(retired.fences.begin(), retired.fences.end(), [this](VkFence fence) {
          return device_->fn()->vkGetFenceStatus(device_->vk_device(), fence) == VK_SUCCESS;
        });
    if (!all_signaled && !force) {
      break;
    }
    for (const auto& semaphore : retired.semaphores) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroySemaphore, device_, semaphore, device_->allocator());
    }
    if (all_signaled && !retired.fences.empty()) {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkResetFences,
                                  device_,
                                  static_cast<uint32_t>(retired.fences.size()),
                                  retired.fences.data());
      free_present_fences_.insert(
          free_present_fences_.end(), retired.fences.begin(), retired.fences.end());
    } else {
      // Forced teardown with unsignaled fences (present-fence wait expired in destroy()):
      // the fences' state is unknowable, so destroy rather than recycle them.
      for (const auto& fence : retired.fences) {
        VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyFence, device_, fence, device_->allocator());
      }
    }
    VIZ_VK_DEVICE_FUNC_NO_ERROR(
        vkDestroySwapchainKHR, device_, retired.swapchain, device_->allocator());
    retired_.erase(retired_.begin());
  }
}

expected<VkFence> VulkanSwapchain::take_present_fence() {
  if (fault_injection_.fail_next_take_present_fence > 0) {
    // Test-only: a deterministic pre-queue present failure (nothing consumed).
    fault_injection_.fail_next_take_present_fence--;
    return make_unexpected(Error{.code = ErrorCode::kVulkanError,
                                 .message = "[holoviz] Injected present-fence failure."});
  }
  recycle_signaled_present_fences();
  if (!free_present_fences_.empty()) {
    VkFence fence = free_present_fences_.back();
    free_present_fences_.pop_back();
    return fence;
  }
  VkFenceCreateInfo fence_create_info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  VIZ_VK_DEVICE_FUNC(vkCreateFence, device_, &fence_create_info, device_->allocator(), &fence);
  return fence;
}

void VulkanSwapchain::recycle_signaled_present_fences() {
  // Present fences signal in order per swapchain, so only the front of the pending list
  // needs checking.
  while (!pending_present_fences_.empty() &&
         device_->fn()->vkGetFenceStatus(device_->vk_device(), pending_present_fences_.front()) ==
             VK_SUCCESS) {
    VkFence fence = pending_present_fences_.front();
    pending_present_fences_.erase(pending_present_fences_.begin());
    if (device_->fn()->vkResetFences(device_->vk_device(), 1, &fence) == VK_SUCCESS) {
      free_present_fences_.push_back(fence);
    } else {
      VIZ_VK_DEVICE_FUNC_NO_ERROR(vkDestroyFence, device_, fence, device_->allocator());
    }
  }
}

expected<void> VulkanSwapchain::release_unpresented_acquire(std::chrono::nanoseconds timeout) {
  if (!has_unpresented_acquire_) {
    return {};
  }
  const uint32_t slot = current_frame_ % frame_count_;
  // When a submit consumed the acquire gate, that submit recorded its timeline value on
  // this frame slot (VulkanDevice::submit() via the gate), and both callers waited the
  // slot tokens out before calling here — the rendering is retired and the present
  // semaphore's signal (if the present gate was consumed too) has executed, so skipping
  // straight to the release below is safe.
  if (image_gate_consumed_[slot] != VK_TRUE || acquire_drain_pending_) {
    if (!acquire_drain_pending_) {
      // No submit consumed the acquire gate, so the acquire semaphore still carries the
      // presentation engine's pending signal; drain it CPU-side so both the release below
      // and the semaphore's later retirement are race-free.
      if (acquire_drain_fence_ == VK_NULL_HANDLE) {
        // Persistent and reused: the fence must outlive its submit even across a timed-out
        // wait, so it is only destroyed in destroy()'s teardown once all drains completed.
        VkFenceCreateInfo fence_create_info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VIZ_VK_DEVICE_FUNC(vkCreateFence,
                           device_,
                           &fence_create_info,
                           device_->allocator(),
                           &acquire_drain_fence_);
      }
      VkPipelineStageFlags wait_stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      VkSubmitInfo submit_info{
          .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
          .waitSemaphoreCount = 1,
          .pWaitSemaphores = &image_available_semaphores_[slot],
          .pWaitDstStageMask = &wait_stage_flags,
      };
      VIZ_VK_FORWARD_ERROR(
          device_->fn()->vkQueueSubmit(present_queue_, 1, &submit_info, acquire_drain_fence_));
      // The queue-side wait on the one-shot binary semaphore is irrevocably enqueued the
      // moment the submit succeeds; commit consumption now so a retry after a timed-out
      // fence wait below never enqueues a second wait on the same semaphore.
      image_gate_consumed_[slot] = VK_TRUE;
      acquire_drain_pending_ = true;
    }
    const VkResult wait_result =
        device_->fn()->vkWaitForFences(device_->vk_device(),
                                       1,
                                       &acquire_drain_fence_,
                                       VK_TRUE,
                                       static_cast<uint64_t>(timeout.count()));
    if (wait_result == VK_TIMEOUT) {
      // acquire_drain_pending_ stays set; the fence is retained for the pending submit and
      // a retry resumes the CPU-side wait here.
      return make_unexpected(Error{.code = ErrorCode::kBackpressured,
                                   .message =
                                       "[holoviz] Timed out draining an unpresented acquire before "
                                       "resizing the swapchain."});
    }
    VIZ_VK_FORWARD_ERROR(wait_result);
    acquire_drain_pending_ = false;
    VIZ_VK_DEVICE_FUNC(vkResetFences, device_, 1, &acquire_drain_fence_);
  }

  const VkReleaseSwapchainImagesInfoEXT release_info{
      .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_EXT,
      .swapchain = swapchain_,
      .imageIndexCount = 1,
      .pImageIndices = &current_image_,
  };
  // The KHR and EXT commands are aliases; volk loads whichever name the driver advertised.
  const auto release_images = device_->fn()->vkReleaseSwapchainImagesKHR != nullptr
                                  ? device_->fn()->vkReleaseSwapchainImagesKHR
                                  : device_->fn()->vkReleaseSwapchainImagesEXT;
  VIZ_VK_FORWARD_ERROR(release_images(device_->vk_device(), &release_info));
  has_unpresented_acquire_ = false;
  return {};
}

Format VulkanSwapchain::format() const {
  // Swapchain formats will always be valid, no need to check for errors here.
  return to_holoviz_format(swapchain_format_).value();
}

ColorSpace VulkanSwapchain::color_space() const {
  // Swapchain color spaces will always be valid, no need to check for errors here.
  return to_holoviz_color_space(swapchain_color_space_).value();
}

PresentMode VulkanSwapchain::present_mode() const {
  // Only modes this module itself mapped can be current, so the conversion cannot fail.
  return to_holoviz_present_mode(present_mode_).value();
}

expected<void> VulkanSwapchain::set_present_mode(PresentMode mode) {
  VIZ_UNWRAP(VkPresentModeKHR, vk_mode, from_holoviz_present_mode(mode));
  if (mode == PresentMode::kFIFOLatestReady && !device_->fifo_latest_ready_enabled()) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message =
                                     "[holoviz] PresentMode::kFIFOLatestReady requires a device "
                                     "extension this device did not enable."});
  }
  if (std::find(compatible_present_modes_.begin(), compatible_present_modes_.end(), vk_mode) ==
      compatible_present_modes_.end()) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = std::string("[holoviz] Present mode ") +
                         std::string(magic_enum::enum_name(mode)) +
                         " is not switchable on this swapchain (the surface does not report "
                         "it as compatible); create a swapchain with that mode instead."});
  }
  present_mode_ = vk_mode;
  return {};
}

expected<std::reference_wrapper<const Image>> VulkanSwapchain::image(uint32_t index) const {
  if (index >= images_.size()) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Vulkan swapchain image index out of bounds."});
  }

  return std::cref(image_handles_[index]);
}

uint32_t VulkanSwapchain::image_count() const {
  return image_count_;
}

expected<AcquireSyncToken> VulkanSwapchain::acquire(std::chrono::nanoseconds timeout) {
  if (device_ == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] acquire() on a destroyed swapchain."});
  }
  // Checked before any native work: a second acquire would pass this slot's one-shot
  // binary acquire semaphore to vkAcquireNextImageKHR again (it may still be signaled or
  // carry a pending signal) and overwrite current_image_, silently losing the first image.
  if (has_unpresented_acquire_) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidState,
              .message = "[holoviz] acquire() while a previous acquire is outstanding; "
                         "present() it, or resize()/destroy() the swapchain to drop it."});
  }

  // --- Self-healing preamble: recreate here whenever the swapchain no longer matches the
  // surface, so resizes need no caller code. Gated on Surface::resizable(): a fixed-size
  // window or a direct-to-display output never runs any of this. One lightweight
  // surface-capabilities query per frame is the price of catching platforms that keep
  // presenting (stretched) at the old size without ever reporting out-of-date.
  if (resizable_) {
    VkSurfaceCapabilitiesKHR surface_caps{};
    VIZ_VK_FORWARD_ERROR(
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &surface_caps));
    if (surface_caps.currentExtent.width != 0xFFFFFFFFU) {
      if (surface_caps.currentExtent.width == 0 || surface_caps.currentExtent.height == 0) {
        return make_unexpected(Error{.code = ErrorCode::kBackpressured,
                                     .message =
                                         "[holoviz] The surface has no presentable area (window "
                                         "minimized?); keep polling events and retry."});
      }
      if (swapchain_ != VK_NULL_HANDLE && (surface_caps.currentExtent.width != extent_.width ||
                                           surface_caps.currentExtent.height != extent_.height)) {
        VIZ_FORWARD_ERROR(auto_resize(timeout));
      }
    }
    // A previous recreation failed and left the swapchain lost; heal it here. (On surfaces
    // with no fixed extent this fails again with the explicit-resize hint, as it must.)
    if (swapchain_ == VK_NULL_HANDLE) {
      VIZ_FORWARD_ERROR(auto_resize(timeout));
    }
    if (image_count_ != caller_visible_image_count_) {
      caller_visible_image_count_ = image_count_;
      return make_unexpected(
          Error{.code = ErrorCode::kImageCountChanged,
                .message = "[holoviz] A swapchain recreation changed image_count(); rebuild "
                           "per-image resources, re-query it, then re-acquire."});
    }
  } else if (swapchain_ == VK_NULL_HANDLE) {
    return make_unexpected(
        Error{.code = ErrorCode::kOutOfDate,
              .message = "[holoviz] The swapchain was lost by a failed resize(); resize() "
                         "again or recreate it."});
  }

  const SubmitSyncToken& wait_token = slot_submit_sync_tokens_[current_frame_ % frame_count_];
  if (wait_token.timeline_value > 0) {
    VkSemaphore timeline = device_->timeline_semaphore();
    VkSemaphoreWaitInfo wait_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &timeline,
        .pValues = &wait_token.timeline_value,
    };
    const VkResult wait_result = device_->fn()->vkWaitSemaphores(
        device_->vk_device(), &wait_info, static_cast<uint64_t>(timeout.count()));
    if (wait_result == VK_TIMEOUT) {
      return make_unexpected(
          Error{.code = ErrorCode::kBackpressured,
                .message = "[holoviz] Timed out waiting for prior frame slot to complete."});
    }
    VIZ_VK_FORWARD_ERROR(wait_result);
  }

  // The acquire attempt, self-healing on out-of-date: recreate and retry once (bounded so a
  // surface that invalidates every recreation cannot livelock the frame loop). A recreation
  // clears the frame slots, so no slot re-wait is needed on the retry.
  for (int attempt = 0;; ++attempt) {
    const VkResult result = device_->fn()->vkAcquireNextImageKHR(
        device_->vk_device(),
        swapchain_,
        static_cast<uint64_t>(timeout.count()),
        image_available_semaphores_[current_frame_ % frame_count_],
        VK_NULL_HANDLE,
        &current_image_);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_ERROR_SURFACE_LOST_KHR) {
      if (!resizable_) {
        // A non-resizable surface opted out of automatic recovery; this is exceptional
        // there (display mode change, surface loss) and the caller decides.
        return make_unexpected(Error{
            .code = ErrorCode::kOutOfDate,
            .message = result == VK_ERROR_OUT_OF_DATE_KHR
                           ? "[holoviz] Swapchain is out of date with its surface; recreate it."
                           : "[holoviz] Surface was lost; recreate the surface and swapchain."});
      }
      if (attempt >= 1) {
        return make_unexpected(
            Error{.code = ErrorCode::kOutOfDate,
                  .message = "[holoviz] The surface kept invalidating the swapchain across "
                             "recreations; retry or resize() explicitly."});
      }
      VIZ_FORWARD_ERROR(auto_resize(timeout));
      if (image_count_ != caller_visible_image_count_) {
        caller_visible_image_count_ = image_count_;
        return make_unexpected(
            Error{.code = ErrorCode::kImageCountChanged,
                  .message = "[holoviz] A swapchain recreation changed image_count(); rebuild "
                             "per-image resources, re-query it, then re-acquire."});
      }
      continue;
    }
    if (result == VK_TIMEOUT || result == VK_NOT_READY) {
      return make_unexpected(
          Error{.code = ErrorCode::kBackpressured,
                .message = "[holoviz] Timed out acquiring a Vulkan swapchain image."});
    }
    if (result != VK_SUBOPTIMAL_KHR) {
      VIZ_VK_FORWARD_ERROR(result);
    }
    break;
  }

  auto acquire_signal_gate = &acquire_gates_[current_frame_ % frame_count_];
  auto present_wait_gate = &present_gates_[current_frame_ % frame_count_];

  // Both gates point at the current frame slot's submit token: whichever submit consumes
  // either gate records its timeline value there (VulkanDevice::submit(), gate signal()),
  // so the slot-drain waits in resize()/destroy() cover submits that never present.
  acquire_signal_gate->set_semaphore(image_available_semaphores_[current_frame_ % frame_count_],
                                     &image_gate_consumed_[current_frame_ % frame_count_],
                                     &slot_submit_sync_tokens_[current_frame_ % frame_count_]);

  present_wait_gate->set_semaphore(present_finished_semaphores_[current_image_],
                                   &present_gate_consumed_[current_image_],
                                   &slot_submit_sync_tokens_[current_frame_ % frame_count_]);

  frame_tracker_[current_image_ % image_count_] = current_frame_;
  has_unpresented_acquire_ = true;

  return AcquireSyncToken{current_image_,
                          current_frame_,
                          generation_,
                          AcquireSignalGate{acquire_signal_gate, this, generation_},
                          PresentWaitGate{present_wait_gate, this, generation_}};
}

expected<void> VulkanSwapchain::present(AcquireSyncToken&& acquired,
                                        SubmitSyncToken submit_sync_token, uint64_t present_id,
                                        PresentConsumed* consumed) {
  if (consumed != nullptr) {
    *consumed = {};
  }
  if (swapchain_ == VK_NULL_HANDLE) {
    // The swapchain was recreated (or lost) after this token was acquired; its gates point
    // into freed storage and must not be touched. The frame is gone — drop the token and
    // re-acquire (which self-heals a lost swapchain). The token is dead either way, so the
    // frame reports consumed: presenting it again can never succeed.
    if (consumed != nullptr) {
      consumed->image = true;
    }
    return make_unexpected(Error{.code = ErrorCode::kFrameDropped,
                                 .message =
                                     "[holoviz] The swapchain was recreated after this frame was "
                                     "acquired; drop its token and re-acquire."});
  }
  if (acquired.generation != generation_) {
    // Checked BEFORE the frame_tracker_ lookup below: a stale token's frame_index may
    // collide with a fresh tracker entry after an in-flight recreation reset the counters,
    // and its gates point into recycled storage — never touch either. Dead token: consumed.
    if (consumed != nullptr) {
      consumed->image = true;
    }
    return make_unexpected(Error{.code = ErrorCode::kFrameDropped,
                                 .message =
                                     "[holoviz] The swapchain was recreated after this frame was "
                                     "acquired; drop its token and re-acquire."});
  }
  // Token fields are caller-mutable and therefore untrusted: bound image_index before any
  // container lookup or native use (a modulo would let an out-of-range index alias a valid
  // one and still reach vkQueuePresentKHR and the gate vectors unbounded).
  if (acquired.image_index >= image_count_) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] AcquireSyncToken::image_index is out of range for this swapchain."});
  }
  // The tracker stores the exact current_frame_ value recorded at acquire, so this equality
  // also bounds every later frame_index % frame_count_ use.
  if (acquired.frame_index != frame_tracker_[acquired.image_index]) {
    return make_unexpected(Error{
        .code = ErrorCode::kFailure,
        .message =
            "[holoviz] Attempted to present a Vulkan swapchain image that was not acquired."});
  }

  VIZ_UNWRAP(bool, acquire_gate_consumed, acquired.acquire_gate.consumed());
  VIZ_UNWRAP(bool, present_gate_consumed, acquired.present_gate.consumed());

  if (acquire_gate_consumed != present_gate_consumed) {
    return make_unexpected(Error{
        .code = ErrorCode::kFailure,
        .message = "[holoviz] If a Vulkan swapchain acquire gate has been consumed, the present "
                   "gate must also be consumed before presenting."});
  }

  VkSemaphore wait_semaphore;
  if (present_gate_consumed) {
    wait_semaphore = reinterpret_cast<VkSemaphore>(
        acquired.present_gate.native_object(NativeObjectType::kVkSemaphore).value().value);
  } else {
    wait_semaphore = reinterpret_cast<VkSemaphore>(
        acquired.acquire_gate.native_object(NativeObjectType::kVkSemaphore).value().value);
  }

#if defined(VK_KHR_present_id)
  const bool tag_present_id = present_id != 0 && device_->capabilities().present_completion;
  if (tag_present_id && present_id <= last_present_id_) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Nonzero present ids must be strictly increasing per "
                         "swapchain."});
  }
#endif

  // Every present carries a fence that signals once the presentation engine is done with
  // the present's resources — the signal destroy()/reap_retired() key teardown off — and
  // the current present mode, which set_present_mode() may have switched since creation.
  VIZ_UNWRAP(VkFence, present_fence, take_present_fence());
  VkSwapchainPresentFenceInfoEXT present_fence_info{
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT,
      .swapchainCount = 1,
      .pFences = &present_fence,
  };
  VkSwapchainPresentModeInfoEXT present_mode_info{
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_EXT,
      .pNext = &present_fence_info,
      .swapchainCount = 1,
      .pPresentModes = &present_mode_,
  };

  VkPresentInfoKHR present_info{
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .pNext = &present_mode_info,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &wait_semaphore,
      .swapchainCount = 1,
      .pSwapchains = &swapchain_,
      .pImageIndices = &acquired.image_index,
      .pResults = nullptr,
  };

  // Tag the frame for wait_for_present() when the device enabled the present-id extension.
  // Ids must be strictly increasing per swapchain (a Vulkan validity rule, validated
  // above); 0 = untagged.
#if defined(VK_KHR_present_id)
  VkPresentIdKHR present_id_info{
      .sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
      .swapchainCount = 1,
      .pPresentIds = &present_id,
  };
  if (tag_present_id) {
    present_id_info.pNext = present_info.pNext;
    present_info.pNext = &present_id_info;
  }
#endif

  VkResult result;
  if (fault_injection_.skip_next_present_with_result != 0) {
    // Test-only: an honest non-enqueued failure — the queue call never happens, so the
    // wait semaphore, the acquired image, and the fence really are untouched, exactly the
    // state the spec guarantees for the non-enqueued results classified below.
    result = static_cast<VkResult>(fault_injection_.skip_next_present_with_result);
    fault_injection_.skip_next_present_with_result = 0;
  } else {
    result = device_->fn()->vkQueuePresentKHR(present_queue_, &present_info);
    if (fault_injection_.override_next_present_result != 0) {
      // Test-only: the present genuinely executed (device healthy, fence will signal);
      // only the result interpretation below changes.
      result = static_cast<VkResult>(fault_injection_.override_next_present_result);
      fault_injection_.override_next_present_result = 0;
    }
  }

  if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
    // Not enqueued: for these two results the spec guarantees every referenced resource
    // and synchronization primitive is unaffected (an implementation that cannot promise
    // that must return VK_ERROR_DEVICE_LOST instead). The wait semaphore keeps its
    // pending signal, the image stays acquired, the id sequence is unadvanced, and
    // *consumed stays cleared, so the caller may retry this same present or release the
    // image. The fence was never registered — return it to the pool, still reset.
    free_present_fences_.push_back(present_fence);
    VIZ_VK_FORWARD_ERROR(result);
  }

  // Every other result is an enqueued present operation — even the rejection errors
  // consume the image, and the fence will signal — so commit all frame bookkeeping
  // before classifying the result: the fence, the tagged id, the consumed report, the
  // slot timeline, the gate resets, and the frame advance. A skipped commit would hand
  // the next acquire consumed binary-semaphore state or wedge teardown on the fence.
  // DEVICE_LOST/UNKNOWN get the same best-effort commit; teardown tolerates a lost
  // device. Max-merge the slot timeline: submit() already recorded its value on this
  // slot through the gates, and the caller-passed (untrusted) token must not regress it.
  pending_present_fences_.push_back(present_fence);
  has_unpresented_acquire_ = false;
#if defined(VK_KHR_present_id)
  if (tag_present_id) {
    last_present_id_ = present_id;
  }
#endif
  if (consumed != nullptr) {
    consumed->image = true;
#if defined(VK_KHR_present_id)
    consumed->id = tag_present_id;
#endif
  }
  SubmitSyncToken& slot_token = slot_submit_sync_tokens_[acquired.frame_index % frame_count_];
  slot_token.timeline_value = std::max(slot_token.timeline_value, submit_sync_token.timeline_value);
  image_gate_consumed_[acquired.frame_index % frame_count_] = VK_FALSE;
  present_gate_consumed_[acquired.image_index] = VK_FALSE;
  current_frame_++;

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_ERROR_SURFACE_LOST_KHR) {
    // The presentation request was dropped (frame state already committed above). Self-
    // heal: recreate the swapchain here so the next acquire hands out a fresh-size image,
    // and report the unavoidable drop distinctly.
    if (!resizable_) {
      // A non-resizable surface opted out of automatic recovery; the caller decides.
      return make_unexpected(Error{
          .code = ErrorCode::kOutOfDate,
          .message = result == VK_ERROR_OUT_OF_DATE_KHR
                         ? "[holoviz] Swapchain is out of date with its surface; recreate it."
                         : "[holoviz] Surface was lost; recreate the surface and swapchain."});
    }

    // present() has no timeout parameter; bound the drain like destroy() bounds its
    // present-fence wait. On failure the next acquire() retries the recovery.
    VIZ_FORWARD_ERROR(auto_resize(std::chrono::seconds(1)));
    if (image_count_ != caller_visible_image_count_) {
      caller_visible_image_count_ = image_count_;
      return make_unexpected(
          Error{.code = ErrorCode::kImageCountChanged,
                .message = "[holoviz] A swapchain recreation changed image_count() (this "
                           "frame was also dropped); rebuild per-image resources and "
                           "re-query it."});
    }
    return make_unexpected(
        Error{.code = ErrorCode::kFrameDropped,
              .message = "[holoviz] The presentation engine dropped this frame while the "
                         "swapchain was recreated; continue with the next frame."});
  }
  if (result != VK_SUBOPTIMAL_KHR) {
    VIZ_VK_FORWARD_ERROR(result);
  }

  return {};
}

expected<void> VulkanSwapchain::wait_for_present(uint64_t present_id,
                                                 std::chrono::nanoseconds timeout) {
#if defined(VK_KHR_present_wait)
  if (!device_->capabilities().present_completion) {
    return make_unexpected(Error{.code = ErrorCode::kNotImplemented,
                                 .message =
                                     "[holoviz] wait_for_present requires present-id/present-wait "
                                     "support; branch on CapabilityReport::present_completion."});
  }
  if (present_id == 0 || present_id > last_present_id_) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] wait_for_present takes a nonzero id already passed to "
                         "present() on this swapchain."});
  }
  if (swapchain_ == VK_NULL_HANDLE) {
    return make_unexpected(Error{.code = ErrorCode::kOutOfDate,
                                 .message = "[holoviz] The swapchain was lost by a failed "
                                            "resize() while a present was outstanding."});
  }
  const VkResult result = device_->fn()->vkWaitForPresentKHR(
      device_->vk_device(), swapchain_, present_id, static_cast<uint64_t>(timeout.count()));
  if (result == VK_TIMEOUT) {
    return make_unexpected(
        Error{.code = ErrorCode::kBackpressured,
              .message = "[holoviz] Timed out waiting for the frame to be presented."});
  }
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_ERROR_SURFACE_LOST_KHR) {
    return make_unexpected(Error{.code = ErrorCode::kOutOfDate,
                                 .message = "[holoviz] Swapchain went out of date while "
                                            "waiting for a present."});
  }
  VIZ_VK_FORWARD_ERROR(result);
  return {};
#else
  (void)present_id;
  (void)timeout;
  return make_unexpected(Error{.code = ErrorCode::kNotImplemented,
                               .message = "[holoviz] Built without VK_KHR_present_wait "
                                          "headers."});
#endif
}

expected<NativeObject> VulkanSwapchain::native_object(NativeObjectType type) const {
  switch (type) {
    case NativeObjectType::kVkDevice:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(device_->vk_device()),
      };
    case NativeObjectType::kVkQueue:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(present_queue_),
      };
    case NativeObjectType::kVkSurfaceKHR:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(surface_),
      };
    case NativeObjectType::kVkSwapchainKHR:
      return NativeObject{
          .type = type,
          .value = reinterpret_cast<uint64_t>(swapchain_),
      };
    default:
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] Unsupported native object type for VulkanSwapchain."});
  }
}

SwapchainFaultInjection& swapchain_fault_injection(Swapchain& swapchain) {
  // Sanctioned via the Swapchain::impl() backend-internal hatch; the Vulkan backend is the
  // only SwapchainImpl implementer.
  return static_cast<VulkanSwapchain&>(swapchain.impl()).fault_injection_;
}

bool device_resources_leaked(const Device& device) {
  return device.impl().resources_leaked();
}

}  // namespace holoviz::vulkan
