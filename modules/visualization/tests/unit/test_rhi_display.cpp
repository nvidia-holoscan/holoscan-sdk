/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pins the pure display-selection contracts in rhi/vulkan/vk_display_util.hpp:
// mode best-match (including the legacy 0=don't-care "prefer largest"
// arithmetic), plane selection, alpha priority, and the ownership truth tables.
// Everything here runs headless — no instance, no display.

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <span>
#include <vector>

#include "rhi/vulkan/vk_display_util.hpp"

namespace rhi = holoviz::rhi;
using holoviz::rhi::vulkan::DisplayModeInfo;
using holoviz::rhi::vulkan::DisplayPlaneInfo;

namespace {

DisplayModeInfo make_mode(uint32_t width, uint32_t height, uint32_t refresh_mhz) {
  return DisplayModeInfo{.handle = 0, .mode = {{width, height}, refresh_mhz}};
}

}  // namespace

TEST(DisplayModeSelect, EmptyReturnsNullopt) {
  EXPECT_FALSE(rhi::vulkan::select_display_mode({}, {}).has_value());
}

TEST(DisplayModeSelect, ExactMatchWins) {
  const std::vector<DisplayModeInfo> modes = {
      make_mode(1920, 1080, 60000),
      make_mode(2560, 1440, 60000),
      make_mode(3840, 2160, 30000),
  };
  const auto index = rhi::vulkan::select_display_mode(modes, {{2560, 1440}, 60000});
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
  const auto index = rhi::vulkan::select_display_mode(modes, {});
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 2u);  // 3840x2160, and 120 Hz beats 30 Hz
}

TEST(DisplayModeSelect, ClosestWidthDominatesHeightAndRefresh) {
  // Lexicographic order: a closer width beats any height/refresh combination.
  const std::vector<DisplayModeInfo> modes = {
      make_mode(1920, 1080, 60000),
      make_mode(2000, 9999, 1),
  };
  const auto index = rhi::vulkan::select_display_mode(modes, {{2010, 1080}, 60000});
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 1u);
}

TEST(DisplayModeSelect, RefreshBreaksResolutionTies) {
  const std::vector<DisplayModeInfo> modes = {
      make_mode(1920, 1080, 60000),
      make_mode(1920, 1080, 120000),
      make_mode(1920, 1080, 144000),
  };
  const auto index = rhi::vulkan::select_display_mode(modes, {{1920, 1080}, 120000});
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 1u);
}

TEST(DisplayModeSelect, ZeroRefreshPrefersFastestAtDesiredResolution) {
  const std::vector<DisplayModeInfo> modes = {
      make_mode(1920, 1080, 60000),
      make_mode(1920, 1080, 240000),
      make_mode(1920, 1080, 120000),
  };
  const auto index = rhi::vulkan::select_display_mode(modes, {{1920, 1080}, 0});
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 1u);
}

TEST(DisplayModeMatch, ZeroFieldsAlwaysMatch) {
  EXPECT_TRUE(rhi::vulkan::mode_matches_desired({{3840, 2160}, 120000}, {}));
  EXPECT_TRUE(rhi::vulkan::mode_matches_desired({{3840, 2160}, 120000}, {{3840, 0}, 0}));
}

TEST(DisplayModeMatch, NonZeroFieldMustMatchExactly) {
  EXPECT_FALSE(rhi::vulkan::mode_matches_desired({{3840, 2160}, 120000}, {{1920, 0}, 0}));
  EXPECT_FALSE(rhi::vulkan::mode_matches_desired({{3840, 2160}, 120000}, {{3840, 2160}, 60000}));
  EXPECT_TRUE(rhi::vulkan::mode_matches_desired({{3840, 2160}, 120000}, {{3840, 2160}, 120000}));
}

TEST(DisplayPlaneSelect, SkipsPlanesBoundToOtherDisplays) {
  constexpr uint64_t kTarget = 0x1000;
  constexpr uint64_t kOther = 0x2000;
  const std::vector<DisplayPlaneInfo> planes = {
      {.current_display = kOther, .current_stack_index = 0, .supports_target = true},
      {.current_display = 0, .current_stack_index = 1, .supports_target = true},
  };
  const auto index = rhi::vulkan::select_display_plane(planes, kTarget);
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 1u);
}

TEST(DisplayPlaneSelect, AcceptsPlaneAlreadyBoundToTarget) {
  constexpr uint64_t kTarget = 0x1000;
  const std::vector<DisplayPlaneInfo> planes = {
      {.current_display = kTarget, .current_stack_index = 0, .supports_target = true},
  };
  const auto index = rhi::vulkan::select_display_plane(planes, kTarget);
  ASSERT_TRUE(index.has_value());
  EXPECT_EQ(*index, 0u);
}

TEST(DisplayPlaneSelect, RequiresSupportedSetMembership) {
  constexpr uint64_t kTarget = 0x1000;
  const std::vector<DisplayPlaneInfo> planes = {
      {.current_display = 0, .current_stack_index = 0, .supports_target = false},
      {.current_display = 0, .current_stack_index = 1, .supports_target = false},
  };
  EXPECT_FALSE(rhi::vulkan::select_display_plane(planes, kTarget).has_value());
  EXPECT_FALSE(rhi::vulkan::select_display_plane({}, kTarget).has_value());
}

TEST(DisplayAlphaSelect, LegacyPriorityOrder) {
  using namespace holoviz::rhi::vulkan;  // NOLINT(build/namespaces)
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
  using rhi::DisplayOwnership;
  using rhi::vulkan::classify_drm_connector;
  // Actively scanned out while another DRM master (compositor) runs.
  EXPECT_EQ(classify_drm_connector(true, true, true), DisplayOwnership::kCompositor);
  // Connected but inactive: free to latch even under a running session
  // (the "xrandr --output X --off" flow).
  EXPECT_EQ(classify_drm_connector(true, false, true), DisplayOwnership::kAvailable);
  // Bare console: everything connected is available.
  EXPECT_EQ(classify_drm_connector(true, true, false), DisplayOwnership::kAvailable);
  EXPECT_EQ(classify_drm_connector(true, false, false), DisplayOwnership::kAvailable);
  EXPECT_EQ(classify_drm_connector(false, false, false), DisplayOwnership::kUnknown);
}

TEST(DisplayOwnershipTables, WinrtTarget) {
  using rhi::DisplayOwnership;
  using rhi::vulkan::classify_winrt_target;
  EXPECT_EQ(classify_winrt_target(true), DisplayOwnership::kAvailable);
  EXPECT_EQ(classify_winrt_target(false), DisplayOwnership::kCompositor);
}

TEST(DisplayOwnershipTables, X11Output) {
  using rhi::DisplayOwnership;
  using rhi::vulkan::classify_x11_output;
  // An active CRTC means the X server (or a lease holder) scans it out —
  // regardless of the non-desktop property.
  EXPECT_EQ(classify_x11_output(true, true, false), DisplayOwnership::kCompositor);
  EXPECT_EQ(classify_x11_output(true, true, true), DisplayOwnership::kCompositor);
  // Connected but off ("xrandr --output X --off" or non-desktop): acquirable.
  EXPECT_EQ(classify_x11_output(true, false, false), DisplayOwnership::kAvailable);
  EXPECT_EQ(classify_x11_output(true, false, true), DisplayOwnership::kAvailable);
  EXPECT_EQ(classify_x11_output(false, false, false), DisplayOwnership::kUnknown);
}

TEST(DisplayOwnershipTables, PreferX11Path) {
  using rhi::vulkan::prefer_x11_display_path;
  EXPECT_TRUE(prefer_x11_display_path(":1", nullptr, "x11"));
  EXPECT_TRUE(prefer_x11_display_path(":1", nullptr, nullptr));  // ssh -X: no XDG var
  EXPECT_TRUE(prefer_x11_display_path(":1", "", "x11"));
  // No X server reachable.
  EXPECT_FALSE(prefer_x11_display_path(nullptr, nullptr, nullptr));
  EXPECT_FALSE(prefer_x11_display_path("", nullptr, "x11"));
  // XWayland: fake RandR outputs, stay on the DRM path.
  EXPECT_FALSE(prefer_x11_display_path(":0", "wayland-0", "wayland"));
  EXPECT_FALSE(prefer_x11_display_path(":0", "wayland-0", nullptr));
  EXPECT_FALSE(prefer_x11_display_path(":0", nullptr, "wayland"));
}

namespace {

/// Mirror of display_systems::NativeDisplayInfo's merge-relevant fields (that
/// header pulls Vulkan/CUDA types; merge_native_display_info is templated on
/// exactly this shape to stay testable headless).
struct MergeInfo {
  uint64_t vk_display = 0;
  rhi::DisplayOwnership ownership = rhi::DisplayOwnership::kUnknown;
  std::array<char, 256> name{};
  rhi::DisplayMode current_mode{};
};

MergeInfo make_merge_info(uint64_t handle, rhi::DisplayOwnership ownership, const char* name,
                          uint32_t width = 0) {
  MergeInfo info{.vk_display = handle, .ownership = ownership};
  std::snprintf(info.name.data(), info.name.size(), "%s", name);
  info.current_mode.resolution.width = width;
  return info;
}

}  // namespace

TEST(MergeNativeDisplayInfo, OverridesByHandleAppendsUnmatchedKeepsRest) {
  using rhi::DisplayOwnership;
  // DRM view: two connectors, both wrongly available (NVIDIA X modesets outside
  // KMS), kernel connector names.
  std::vector<MergeInfo> base = {
      make_merge_info(0x4, DisplayOwnership::kAvailable, "DP-4"),
      make_merge_info(0x5, DisplayOwnership::kAvailable, "DP-5"),
      make_merge_info(0x6, DisplayOwnership::kAvailable, "DP-6"),
  };
  // X11 view: 0x4 is scanned out (xrandr name DP-0), 0x5 detached (DP-2), 0x6
  // unresolved by RandR (absent), 0x7 known only to RandR.
  const std::vector<MergeInfo> overrides = {
      make_merge_info(0x4, DisplayOwnership::kCompositor, "DP-0", 3440),
      make_merge_info(0x5, DisplayOwnership::kAvailable, "DP-2"),
      make_merge_info(0x7, DisplayOwnership::kCompositor, "DP-3", 1920),
      make_merge_info(0, DisplayOwnership::kCompositor, "ignored"),  // no handle: dropped
  };

  rhi::vulkan::merge_native_display_info(base, std::span<const MergeInfo>(overrides));

  ASSERT_EQ(base.size(), 4U);
  EXPECT_EQ(base[0].ownership, DisplayOwnership::kCompositor);
  EXPECT_STREQ(base[0].name.data(), "DP-0");
  EXPECT_EQ(base[0].current_mode.resolution.width, 3440U);
  EXPECT_EQ(base[1].ownership, DisplayOwnership::kAvailable);
  EXPECT_STREQ(base[1].name.data(), "DP-2");
  // Unresolved by X11: keeps the DRM verdict and name.
  EXPECT_EQ(base[2].ownership, DisplayOwnership::kAvailable);
  EXPECT_STREQ(base[2].name.data(), "DP-6");
  // Known only to X11: appended.
  EXPECT_EQ(base[3].vk_display, 0x7U);
  EXPECT_STREQ(base[3].name.data(), "DP-3");
}

TEST(MergeNativeDisplayInfo, EmptyNameAndModeDoNotOverride) {
  using rhi::DisplayOwnership;
  std::vector<MergeInfo> base = {
      make_merge_info(0x4, DisplayOwnership::kAvailable, "DP-4", 1920),
  };
  const std::vector<MergeInfo> overrides = {
      make_merge_info(0x4, DisplayOwnership::kCompositor, "", 0),
  };
  rhi::vulkan::merge_native_display_info(base, std::span<const MergeInfo>(overrides));
  ASSERT_EQ(base.size(), 1U);
  EXPECT_EQ(base[0].ownership, DisplayOwnership::kCompositor);
  EXPECT_STREQ(base[0].name.data(), "DP-4");
  EXPECT_EQ(base[0].current_mode.resolution.width, 1920U);
}
