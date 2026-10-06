/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pins the pure display-selection contracts in vulkan/vk_display_util.hpp:
// mode best-match (including the legacy 0=don't-care "prefer largest"
// arithmetic), plane selection, alpha priority, and the DRM ownership truth table.
// Everything here runs headless — no instance, no display.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "vulkan/vk_display_util.hpp"

using holoviz::vulkan::DisplayModeInfo;
using holoviz::vulkan::DisplayPlaneInfo;

#if defined(__linux__)
#include "display_systems/display_ownership_drm.hpp"
#endif

namespace {

DisplayModeInfo make_mode(uint32_t width, uint32_t height, uint32_t refresh_mhz) {
  return DisplayModeInfo{.handle = 0, .mode = {{width, height}, refresh_mhz}};
}

}  // namespace

TEST(DisplayModeSelect, EmptyReturnsNullopt) {
  EXPECT_FALSE(holoviz::vulkan::select_display_mode({}, {}).has_value());
}

TEST(DisplayModeSelect, ExactMatchWins) {
  const std::vector<DisplayModeInfo> modes = {
      make_mode(1920, 1080, 60000),
      make_mode(2560, 1440, 60000),
      make_mode(3840, 2160, 30000),
  };
  const auto index = holoviz::vulkan::select_display_mode(modes, {{2560, 1440}, 60000});
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 1u);
}

TEST(DisplayModeSelect, AllZeroPrefersLargestThenFastest) {
  // 0 = don't care scores as UINT32_MAX - value, so the largest width wins,
  // then height, then the fastest refresh.
  const std::vector<DisplayModeInfo> modes = {
      make_mode(1920, 1080, 240000),
      make_mode(3840, 2160, 30000),
      make_mode(3840, 2160, 120000),
      make_mode(2560, 1440, 165000),
  };
  const auto index = holoviz::vulkan::select_display_mode(modes, {});
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 2u);  // 3840x2160, and 120 Hz beats 30 Hz
}

TEST(DisplayModeSelect, ClosestWidthDominatesHeightAndRefresh) {
  // Lexicographic order: a closer width beats any height/refresh combination.
  const std::vector<DisplayModeInfo> modes = {
      make_mode(1920, 1080, 60000),
      make_mode(2000, 9999, 1),
  };
  const auto index = holoviz::vulkan::select_display_mode(modes, {{2010, 1080}, 60000});
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 1u);
}

TEST(DisplayModeSelect, RefreshBreaksResolutionTies) {
  const std::vector<DisplayModeInfo> modes = {
      make_mode(1920, 1080, 60000),
      make_mode(1920, 1080, 120000),
      make_mode(1920, 1080, 144000),
  };
  const auto index = holoviz::vulkan::select_display_mode(modes, {{1920, 1080}, 120000});
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 1u);
}

TEST(DisplayModeSelect, ZeroRefreshPrefersFastestAtDesiredResolution) {
  const std::vector<DisplayModeInfo> modes = {
      make_mode(1920, 1080, 60000),
      make_mode(1920, 1080, 240000),
      make_mode(1920, 1080, 120000),
  };
  const auto index = holoviz::vulkan::select_display_mode(modes, {{1920, 1080}, 0});
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 1u);
}

TEST(DisplayModeMatch, ZeroFieldsAlwaysMatch) {
  EXPECT_TRUE(holoviz::vulkan::mode_matches_desired({{3840, 2160}, 120000}, {}));
  EXPECT_TRUE(holoviz::vulkan::mode_matches_desired({{3840, 2160}, 120000}, {{3840, 0}, 0}));
}

TEST(DisplayModeMatch, NonZeroFieldMustMatchExactly) {
  EXPECT_FALSE(holoviz::vulkan::mode_matches_desired({{3840, 2160}, 120000}, {{1920, 0}, 0}));
  EXPECT_FALSE(
      holoviz::vulkan::mode_matches_desired({{3840, 2160}, 120000}, {{3840, 2160}, 60000}));
  EXPECT_TRUE(
      holoviz::vulkan::mode_matches_desired({{3840, 2160}, 120000}, {{3840, 2160}, 120000}));
}

TEST(DisplayPlaneSelect, SkipsPlanesBoundToOtherDisplays) {
  constexpr uint64_t kTarget = 0x1000;
  constexpr uint64_t kOther = 0x2000;
  const std::vector<DisplayPlaneInfo> planes = {
      {.current_display = kOther, .current_stack_index = 0, .supports_target = true},
      {.current_display = 0, .current_stack_index = 1, .supports_target = true},
  };
  const auto index = holoviz::vulkan::select_display_plane(planes, kTarget);
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 1u);
}

TEST(DisplayPlaneSelect, AcceptsPlaneAlreadyBoundToTarget) {
  constexpr uint64_t kTarget = 0x1000;
  const std::vector<DisplayPlaneInfo> planes = {
      {.current_display = kTarget, .current_stack_index = 0, .supports_target = true},
  };
  const auto index = holoviz::vulkan::select_display_plane(planes, kTarget);
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 0u);
}

TEST(DisplayPlaneSelect, RequiresSupportedSetMembership) {
  constexpr uint64_t kTarget = 0x1000;
  const std::vector<DisplayPlaneInfo> planes = {
      {.current_display = 0, .current_stack_index = 0, .supports_target = false},
      {.current_display = 0, .current_stack_index = 1, .supports_target = false},
  };
  EXPECT_FALSE(holoviz::vulkan::select_display_plane(planes, kTarget).has_value());
  EXPECT_FALSE(holoviz::vulkan::select_display_plane({}, kTarget).has_value());
}

TEST(DisplayAlphaSelect, LegacyPriorityOrder) {
  using namespace holoviz::vulkan;  // NOLINT(build/namespaces)
  EXPECT_EQ(select_alpha_mode(kDisplayPlaneAlphaOpaque | kDisplayPlaneAlphaGlobal),
            kDisplayPlaneAlphaOpaque);
  EXPECT_EQ(select_alpha_mode(kDisplayPlaneAlphaGlobal | kDisplayPlaneAlphaPerPixel),
            kDisplayPlaneAlphaGlobal);
  EXPECT_EQ(select_alpha_mode(kDisplayPlaneAlphaPerPixelPremultiplied | kDisplayPlaneAlphaPerPixel),
            kDisplayPlaneAlphaPerPixel);
  EXPECT_EQ(select_alpha_mode(kDisplayPlaneAlphaPerPixelPremultiplied),
            kDisplayPlaneAlphaPerPixelPremultiplied);
  // Fallback when the mask advertises nothing usable.
  EXPECT_EQ(select_alpha_mode(0), kDisplayPlaneAlphaOpaque);
}

TEST(DisplayOwnershipTables, DrmConnector) {
  using holoviz::DisplayOwnership;
  using holoviz::vulkan::classify_drm_connector;
  // Another DRM master (a display server) holds the device: every connected
  // connector belongs to it, scanned out or not (an output turned off in the
  // session cannot be acquired either — the X server keeps DRM master).
  EXPECT_EQ(classify_drm_connector(true, true, true), DisplayOwnership::kCompositor);
  EXPECT_EQ(classify_drm_connector(true, false, true), DisplayOwnership::kCompositor);
  // Bare console: everything connected is available.
  EXPECT_EQ(classify_drm_connector(true, true, false), DisplayOwnership::kAvailable);
  EXPECT_EQ(classify_drm_connector(true, false, false), DisplayOwnership::kAvailable);
  EXPECT_EQ(classify_drm_connector(false, false, false), DisplayOwnership::kUnknown);
}

#if defined(__linux__)
TEST(DrmConnectorTypeName, KnownAndUnknownTypes) {
  using holoviz::display_systems::drm::connector_type_name;

  EXPECT_STREQ(connector_type_name(DRM_MODE_CONNECTOR_DisplayPort), "DP");
  EXPECT_STREQ(connector_type_name(DRM_MODE_CONNECTOR_HDMIA), "HDMI-A");
  EXPECT_STREQ(connector_type_name(DRM_MODE_CONNECTOR_eDP), "eDP");
  EXPECT_STREQ(connector_type_name(DRM_MODE_CONNECTOR_Unknown), "Unknown");
  EXPECT_EQ(connector_type_name(UINT32_MAX), nullptr);
}
#endif

TEST(EdidMinVerticalRate, RangeLimitsDescriptor) {
  // Minimal 128-byte base block with one range-limits descriptor (tag 0xFD) at the
  // second descriptor slot: min vertical 60 Hz, max 144 Hz, no rate offsets.
  std::vector<uint8_t> edid(128, 0);
  edid[54] = 0x01;  // slot 1: a detailed timing (nonzero pixel clock) — must be skipped
  edid[72 + 3] = 0xFD;
  edid[72 + 4] = 0x00;  // rate offsets: none
  edid[72 + 5] = 60;    // min vertical rate
  edid[72 + 6] = 144;   // max vertical rate
  EXPECT_EQ(holoviz::vulkan::edid_min_vertical_rate_hz(edid), 60U);
}

TEST(EdidMinVerticalRate, RateOffsetFlagAdds255ToMin) {
  std::vector<uint8_t> edid(128, 0);
  edid[54 + 3] = 0xFD;
  edid[54 + 4] = 0x03;  // vertical rate offsets 0b11: +255 to max AND min
  edid[54 + 5] = 45;    // 45 + 255 = 300 Hz floor
  EXPECT_EQ(holoviz::vulkan::edid_min_vertical_rate_hz(edid), 300U);

  edid[54 + 4] = 0x02;  // 0b10: +255 to max only
  EXPECT_EQ(holoviz::vulkan::edid_min_vertical_rate_hz(edid), 45U);
}

TEST(EdidMinVerticalRate, MissingDescriptorAndShortBlobReturnNothing) {
  const std::vector<uint8_t> no_descriptor(128, 0);
  EXPECT_FALSE(holoviz::vulkan::edid_min_vertical_rate_hz(no_descriptor).has_value());
  const std::vector<uint8_t> truncated(64, 0);
  EXPECT_FALSE(holoviz::vulkan::edid_min_vertical_rate_hz(truncated).has_value());
}
