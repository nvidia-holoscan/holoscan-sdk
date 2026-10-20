/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include <holoviz/swapchain.hpp>

#include "vk_impls.hpp"  // IWYU pragma: keep

namespace holoviz::vulkan {

#if defined(VK_EXT_present_timing)
/// Translate one vkGetPastPresentationTimingEXT result into the public sample. Pure and
/// GPU-free so the contract is unit-testable:
///  - stage times are copied as-is: VkPresentStageTimeEXT::time is already nanoseconds in
///    every domain (unlike raw timestamp-query ticks, which VulkanDevice::device_ticks_to_ns
///    scales), so no period/frequency conversion applies here;
///  - a stage reported with time 0 is "timing unavailable" (legal even when
///    \c reportComplete is set) and leaves its optional disengaged, exactly like a stage
///    that was not reported at all — consumers must not read 0 ns as a timestamp;
///  - \c time_domain is the domain the presentation engine actually reported for this
///    present (it may legally fall back from the swapchain's requested domain), and
///    \c time_domain_id its identity for the opaque local domains;
///  - nullopt when the reported domain has no TimeDomain equivalent (the caller drops the
///    sample rather than mislabel it).
std::optional<PresentTimingSample> decode_past_presentation_timing(
    const VkPastPresentationTimingEXT& timing);
#endif

}  // namespace holoviz::vulkan
