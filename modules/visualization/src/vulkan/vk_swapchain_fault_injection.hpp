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
  /// After the next real vkGetSwapchainTimingPropertiesEXT (whether it succeeded or reported
  /// VK_NOT_READY), report success with refreshInterval = 0 (mode undetermined) and a known
  /// refreshDuration — the engine's own, or a nominal 60 Hz period when it reported none — to
  /// exercise the legal "known period, unknown mode" pairing. One-shot.
  bool override_next_timing_unknown_refresh_mode = false;
  /// Rewrite the VkTimeDomainKHR of every native report returned by the next
  /// past_present_timings() call that returns any (-1 = off), before decoding. With a domain
  /// this API cannot represent the reports are dropped while their completed slots are still
  /// freed — the documented drop contract, made deterministic.
  int32_t override_next_timing_report_domain = -1;
  /// Observer: completed native reports past_present_timings() dropped because their clock is
  /// not representable (injected or genuine). Lets tests distinguish "no sample because the
  /// report was dropped" from "no report arrived".
  uint32_t dropped_unrepresentable_reports = 0;
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
  /// After the next real vkGetSwapchainTimingPropertiesEXT, replace its VkResult with this
  /// value (0 = off). The query genuinely ran; only the result interpretation changes — the
  /// honest model for VK_NOT_READY, which a driver returns on its own schedule.
  int32_t override_next_timing_properties_result = 0;
};

/// Numeric VkResult twins so tests need no Vulkan headers.
inline constexpr int32_t kVkErrorOutOfDate = -1000001004;  // VK_ERROR_OUT_OF_DATE_KHR
inline constexpr int32_t kVkErrorUnknown = -13;            // VK_ERROR_UNKNOWN
inline constexpr int32_t kVkErrorOutOfHostMemory = -1;     // VK_ERROR_OUT_OF_HOST_MEMORY
// VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: an enqueued presentation rejection
// (unreachable without the Win32-only extension, but honest for override-based tests —
// the real present executed and the error only changes the interpretation).
inline constexpr int32_t kVkErrorFullScreenExclusiveModeLost = -1000255000;
// VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT: the presentation engine rejected the present's
// timing request because its result queue was full. An *enqueued* result — the present's
// wait, image release and fence all happened; only the frame's display and its timing
// sample are lost — so override-based injection is the honest model.
inline constexpr int32_t kVkErrorPresentTimingQueueFull = -1000208000;
// VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR: a valid Vulkan clock this API deliberately does not
// map to a TimeDomain (see from_vk_time_domain()); reports on it are dropped.
inline constexpr int32_t kVkTimeDomainClockMonotonicRaw = 2;
// VK_NOT_READY: a success code — vkGetSwapchainTimingPropertiesEXT could not determine the
// refresh cycle yet (nothing presented, or the properties just changed).
inline constexpr int32_t kVkNotReady = 1;

/// The injection state of a Vulkan swapchain (via the Swapchain::impl() hatch).
/// Exported for fault-injection test targets only; this header lives in src/ and is never
/// installed. Precondition: the handle is non-empty and Vulkan-backed.
[[nodiscard]] HOLOVIZ_EXPORT SwapchainFaultInjection& swapchain_fault_injection(
    Swapchain& swapchain);

/// Where a Vulkan swapchain's present fences currently live: in flight on the current
/// generation, or carried by the generations resize() retired and not yet reaped. Test-only
/// observer letting a test prove a present's fence stayed in the retirement proof.
struct SwapchainFenceCounts {
  uint32_t pending = 0;                  ///< In flight on the current swapchain.
  uint32_t pending_timing_rejected = 0;  ///< Of \c pending, from timing-rejected presents.
  uint32_t retired_generations = 0;      ///< Retired swapchains not yet reaped.
  uint32_t retired_fences = 0;           ///< Fences those retired generations still wait on.
  uint32_t retired_timing_rejected = 0;  ///< Of \c retired_fences, from timing-rejected presents.
};

/// Observe the swapchain's present-fence ownership (via the Swapchain::impl() hatch).
/// Precondition: the handle is non-empty and Vulkan-backed.
[[nodiscard]] HOLOVIZ_EXPORT SwapchainFenceCounts
swapchain_fence_counts(const Swapchain& swapchain);

/// Observe the terminal device poison (a child teardown quarantined/leaked resources that
/// still reference the device — see DeviceImpl::report_leaked_resources()). Test-only
/// observer for the same seam; the poison deliberately has no public spelling.
[[nodiscard]] HOLOVIZ_EXPORT bool device_resources_leaked(const Device& device);

}  // namespace holoviz::vulkan
