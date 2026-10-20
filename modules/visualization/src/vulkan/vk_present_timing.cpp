/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vk_present_timing.hpp"

#include "vk_enum_conversions.hpp"

namespace holoviz::vulkan {

#if defined(VK_EXT_present_timing)
std::optional<PresentTimingSample> decode_past_presentation_timing(
    const VkPastPresentationTimingEXT& timing) {
  const std::optional<TimeDomain> domain = from_vk_time_domain(timing.timeDomain);
  if (!domain) {
    return std::nullopt;
  }
  PresentTimingSample sample{
      .present_id = timing.presentId,
      .complete = timing.reportComplete == VK_TRUE,
      .time_domain = *domain,
      .time_domain_id = timing.timeDomainId,
  };
  for (uint32_t s = 0; s < timing.presentStageCount; ++s) {
    const VkPresentStageTimeEXT& stage = timing.pPresentStages[s];
    // A zero time means the engine could not time this stage (legal even in a complete
    // report); leave the optional disengaged rather than hand out 0 ns as a timestamp.
    if (stage.time == 0) {
      continue;
    }
    switch (stage.stage) {
      case VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT:
        sample.queue_operations_end_ns = stage.time;
        break;
      case VK_PRESENT_STAGE_REQUEST_DEQUEUED_BIT_EXT:
        sample.request_dequeued_ns = stage.time;
        break;
      case VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT:
        sample.first_pixel_out_ns = stage.time;
        break;
      case VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT:
        sample.first_pixel_visible_ns = stage.time;
        break;
      default:
        break;
    }
  }
  return sample;
}
#endif

}  // namespace holoviz::vulkan
