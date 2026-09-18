// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Sanctioned fault-injection seam for swapchain tests (the imgui backend's host_alloc_fn_
// pattern): deterministic failure knobs for the present/recreate paths that no real driver
// will produce on cue. Production code never sets these; tests reach them through
// swapchain_fault_injection(). Single-threaded, like every HoloViz handle. This header is
// deliberately Vulkan-header-free so test targets can include it directly.

#pragma once

#include <cstdint>

#include <holoviz/holoviz_export.h>

namespace holoviz {
class Device;
class Swapchain;
}  // namespace holoviz

namespace holoviz::vulkan {

struct SwapchainFaultInjection {
  /// Fail the next N take_present_fence() calls with kVulkanError — a deterministic
  /// pre-queue present failure (nothing consumed).
  uint32_t fail_next_take_present_fence = 0;
  /// After the next real vkQueuePresentKHR, replace its VkResult with this value (0 =
  /// off). The present genuinely executed, so the device stays healthy; only the result
  /// interpretation — and therefore the recovery/bookkeeping path — changes.
  int32_t override_next_present_result = 0;
  /// Skip the next N vkCreateSwapchainKHR calls inside recreate() and fail them: phase 1
  /// still runs (old swapchain retired, generation bumped), leaving the object in the
  /// image-less-but-consistent failed-recreation state.
  uint32_t fail_next_swapchain_create = 0;
  /// Skip the next vkQueuePresentKHR call entirely and use this VkResult instead (0 =
  /// off). Unlike override_next_present_result, nothing was enqueued: the wait semaphore,
  /// the acquired image, and the present fence are genuinely untouched — an honest
  /// simulation of a non-enqueued failure (VK_ERROR_OUT_OF_HOST_MEMORY et al.).
  int32_t skip_next_present_with_result = 0;
  /// Add this many images to the next recreate()'s requested minImageCount (still clamped
  /// to the surface's supported range; 0 = off). Combined with an out-of-date injection it
  /// makes an image-count-changing recreation deterministic — the driver may still return
  /// the same total count, so tests must read back image_count() rather than assume.
  uint32_t bump_next_recreate_image_count = 0;
};

/// Numeric VkResult twins so tests need no Vulkan headers.
inline constexpr int32_t kVkErrorOutOfDate = -1000001004;  // VK_ERROR_OUT_OF_DATE_KHR
inline constexpr int32_t kVkErrorUnknown = -13;            // VK_ERROR_UNKNOWN
inline constexpr int32_t kVkErrorOutOfHostMemory = -1;     // VK_ERROR_OUT_OF_HOST_MEMORY
// VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: an enqueued presentation rejection
// (unreachable without the Win32-only extension, but honest for override-based tests —
// the real present executed and the error only changes the interpretation).
inline constexpr int32_t kVkErrorFullScreenExclusiveModeLost = -1000255000;

/// The injection state of a Vulkan swapchain (via the Swapchain::impl() hatch).
/// Exported for fault-injection test targets only; this header lives in src/ and is never
/// installed. Precondition: the handle is non-empty and Vulkan-backed.
[[nodiscard]] HOLOVIZ_EXPORT SwapchainFaultInjection& swapchain_fault_injection(
    Swapchain& swapchain);

/// Observe the terminal device poison (a child teardown quarantined/leaked resources that
/// still reference the device — see DeviceImpl::report_leaked_resources()). Test-only
/// observer for the same seam; the poison deliberately has no public spelling.
[[nodiscard]] HOLOVIZ_EXPORT bool device_resources_leaked(const Device& device);

}  // namespace holoviz::vulkan
