/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pins the VK_EXT_present_timing result decoder (vulkan/vk_present_timing.hpp) without a
// GPU: the module-private TU is compiled into this test. Present-stage times are nanoseconds
// in every domain and must reach the public sample unscaled; the sample's domain is the one
// the presentation engine reported for that present, not the one the swapchain requested.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>

#include <holoviz/common.hpp>
#include <holoviz/swapchain.hpp>

#include "vulkan/vk_enum_conversions.hpp"
#include "vulkan/vk_present_timing.hpp"

namespace holoviz::vulkan {
namespace {

#if defined(VK_EXT_present_timing)

constexpr uint64_t kOneSecondNs = 1'000'000'000ULL;

VkPastPresentationTimingEXT make_timing(VkTimeDomainKHR domain, uint64_t domain_id,
                                        std::array<VkPresentStageTimeEXT, 2>& stages,
                                        VkBool32 complete = VK_TRUE) {
  stages = {{
      {.stage = VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT, .time = kOneSecondNs},
      {.stage = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT, .time = 2 * kOneSecondNs},
  }};
  return VkPastPresentationTimingEXT{
      .sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT,
      .presentId = 7,
      .presentStageCount = static_cast<uint32_t>(stages.size()),
      .pPresentStages = stages.data(),
      .timeDomain = domain,
      .timeDomainId = domain_id,
      .reportComplete = complete,
  };
}

TEST(PresentTimingDecode, DeviceDomainStageTimesAreCopiedUnscaled) {
  // Regression: stage times used to go through the raw-timestamp tick conversion, which
  // rescales (and masks) them on devices whose timestampPeriod is not 1 ns.
  std::array<VkPresentStageTimeEXT, 2> stages{};
  const auto sample =
      decode_past_presentation_timing(make_timing(VK_TIME_DOMAIN_DEVICE_KHR, 0, stages));
  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->present_id, 7U);
  EXPECT_TRUE(sample->complete);
  EXPECT_EQ(sample->time_domain, TimeDomain::kDevice);
  EXPECT_EQ(sample->time_domain_id, 0U);
  ASSERT_TRUE(sample->queue_operations_end_ns.has_value());
  EXPECT_EQ(*sample->queue_operations_end_ns, kOneSecondNs);
  ASSERT_TRUE(sample->first_pixel_out_ns.has_value());
  EXPECT_EQ(*sample->first_pixel_out_ns, 2 * kOneSecondNs);
  EXPECT_FALSE(sample->request_dequeued_ns.has_value());
  EXPECT_FALSE(sample->first_pixel_visible_ns.has_value());
}

TEST(PresentTimingDecode, HostDomainStageTimesAreCopiedUnscaled) {
  // Regression: on Windows host-domain times were multiplied by 1e9 / QPC frequency.
#ifdef _WIN32
  constexpr VkTimeDomainKHR kHost = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR;
  constexpr VkTimeDomainKHR kOtherPlatformHost = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
#else
  constexpr VkTimeDomainKHR kHost = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
  constexpr VkTimeDomainKHR kOtherPlatformHost = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR;
#endif
  std::array<VkPresentStageTimeEXT, 2> stages{};
  const auto sample = decode_past_presentation_timing(make_timing(kHost, 0, stages));
  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->time_domain, TimeDomain::kHostMonotonic);
  EXPECT_EQ(*sample->queue_operations_end_ns, kOneSecondNs);
  EXPECT_EQ(*sample->first_pixel_out_ns, 2 * kOneSecondNs);

  // The other platform's host clock is not this platform's kHostMonotonic.
  EXPECT_FALSE(
      decode_past_presentation_timing(make_timing(kOtherPlatformHost, 0, stages)).has_value());
}

TEST(PresentTimingDecode, ReturnedDomainWinsOverTheRequestedOne) {
  // Regression: samples were labelled with the swapchain's requested domain even when the
  // engine fell back to an opaque local clock for the present.
  std::array<VkPresentStageTimeEXT, 2> stages{};
  const auto local =
      decode_past_presentation_timing(make_timing(VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT, 42, stages));
  ASSERT_TRUE(local.has_value());
  EXPECT_EQ(local->time_domain, TimeDomain::kSwapchainLocal);
  EXPECT_EQ(local->time_domain_id, 42U);
  EXPECT_EQ(*local->first_pixel_out_ns, 2 * kOneSecondNs);

  const auto stage_local = decode_past_presentation_timing(
      make_timing(VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT, 43, stages));
  ASSERT_TRUE(stage_local.has_value());
  EXPECT_EQ(stage_local->time_domain, TimeDomain::kPresentStageLocal);
  EXPECT_EQ(stage_local->time_domain_id, 43U);
}

TEST(PresentTimingDecode, UnrepresentableDomainIsRejectedNotMislabelled) {
  std::array<VkPresentStageTimeEXT, 2> stages{};
  EXPECT_FALSE(decode_past_presentation_timing(
                   make_timing(VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR, 0, stages))
                   .has_value());
  EXPECT_FALSE(decode_past_presentation_timing(
                   make_timing(static_cast<VkTimeDomainKHR>(0x7ABCDEF), 0, stages))
                   .has_value());
}

TEST(PresentTimingDecode, IncompleteSampleKeepsReportedStagesOnly) {
  std::array<VkPresentStageTimeEXT, 2> stages{};
  auto timing = make_timing(VK_TIME_DOMAIN_DEVICE_KHR, 0, stages, VK_FALSE);
  timing.presentStageCount = 1;  // only QUEUE_OPERATIONS_END has happened
  const auto sample = decode_past_presentation_timing(timing);
  ASSERT_TRUE(sample.has_value());
  EXPECT_FALSE(sample->complete);
  EXPECT_TRUE(sample->queue_operations_end_ns.has_value());
  EXPECT_FALSE(sample->first_pixel_out_ns.has_value());
}

/// All four mapped stages, with \p zero_stage reported as time 0 and the others nonzero.
VkPastPresentationTimingEXT make_timing_with_zero_stage(
    VkPresentStageFlagBitsEXT zero_stage, std::array<VkPresentStageTimeEXT, 4>& stages) {
  stages = {{
      {.stage = VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT, .time = kOneSecondNs},
      {.stage = VK_PRESENT_STAGE_REQUEST_DEQUEUED_BIT_EXT, .time = 2 * kOneSecondNs},
      {.stage = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT, .time = 3 * kOneSecondNs},
      {.stage = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT, .time = 4 * kOneSecondNs},
  }};
  for (auto& stage : stages) {
    if (stage.stage == zero_stage) {
      stage.time = 0;
    }
  }
  return VkPastPresentationTimingEXT{
      .sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT,
      .presentId = 9,
      .presentStageCount = static_cast<uint32_t>(stages.size()),
      .pPresentStages = stages.data(),
      .timeDomain = VK_TIME_DOMAIN_DEVICE_KHR,
      .timeDomainId = 0,
      .reportComplete = VK_TRUE,
  };
}

const std::optional<uint64_t>& stage_value(const PresentTimingSample& sample,
                                           VkPresentStageFlagBitsEXT stage) {
  switch (stage) {
    case VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT:
      return sample.queue_operations_end_ns;
    case VK_PRESENT_STAGE_REQUEST_DEQUEUED_BIT_EXT:
      return sample.request_dequeued_ns;
    case VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT:
      return sample.first_pixel_out_ns;
    default:
      return sample.first_pixel_visible_ns;
  }
}

TEST(PresentTimingDecode, ZeroStageTimeInCompleteReportStaysDisengaged) {
  // Regression: a complete report may still carry time 0 for a stage the engine could not
  // time. Engaging the optional with 0 made consumers compute wrapped-around intervals.
  constexpr std::array<VkPresentStageFlagBitsEXT, 4> kStages{
      VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT,
      VK_PRESENT_STAGE_REQUEST_DEQUEUED_BIT_EXT,
      VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT,
      VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT,
  };
  for (const VkPresentStageFlagBitsEXT zero_stage : kStages) {
    std::array<VkPresentStageTimeEXT, 4> stages{};
    const auto sample =
        decode_past_presentation_timing(make_timing_with_zero_stage(zero_stage, stages));
    ASSERT_TRUE(sample.has_value());
    EXPECT_TRUE(sample->complete) << "stage " << zero_stage;
    EXPECT_EQ(sample->present_id, 9U);
    EXPECT_EQ(sample->time_domain, TimeDomain::kDevice);
    for (const VkPresentStageFlagBitsEXT stage : kStages) {
      const auto& value = stage_value(*sample, stage);
      if (stage == zero_stage) {
        EXPECT_FALSE(value.has_value()) << "zero stage " << stage << " must stay disengaged";
      } else {
        ASSERT_TRUE(value.has_value()) << "nonzero stage " << stage << " must be preserved";
        EXPECT_NE(*value, 0U);
      }
    }
  }
}

TEST(PresentTimingDecode, DomainMappingRoundTrips) {
  for (const TimeDomain domain : {TimeDomain::kDevice,
                                  TimeDomain::kHostMonotonic,
                                  TimeDomain::kSwapchainLocal,
                                  TimeDomain::kPresentStageLocal}) {
    const auto back = from_vk_time_domain(to_vk_time_domain(domain));
    ASSERT_TRUE(back.has_value());
    EXPECT_EQ(*back, domain);
  }
}

#else

TEST(PresentTimingDecode, SkippedWithoutPresentTimingHeaders) {
  GTEST_SKIP() << "built without VK_EXT_present_timing headers";
}

#endif

}  // namespace
}  // namespace holoviz::vulkan
