/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure window-system policy tests: monitor-name selection, size, and refresh-rate selection
// for fullscreen placement. No window system or device is needed.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include "window_systems/window_util.hpp"

namespace holoviz::window_systems {
namespace {

TEST(FindMonitorByName, ExactMatchWins) {
  const std::vector<std::string_view> names{"DP-2", "HDMI-1", "DP-4"};
  EXPECT_EQ(find_monitor_by_name(names, "HDMI-1"), 1U);
  EXPECT_EQ(find_monitor_by_name(names, "DP-4"), 2U);
}

TEST(FindMonitorByName, MissAndEmptyDesiredReturnNothing) {
  const std::vector<std::string_view> names{"DP-2", "HDMI-1"};
  EXPECT_FALSE(find_monitor_by_name(names, "DP-3").has_value());
  // Substrings are not matches: fullscreen selection is exact by contract.
  EXPECT_FALSE(find_monitor_by_name(names, "DP").has_value());
  EXPECT_FALSE(find_monitor_by_name(names, "").has_value());
  EXPECT_FALSE(find_monitor_by_name({}, "DP-2").has_value());
}

// Fullscreen: a zero width OR height takes the monitor's current mode; an explicit size is
// requested as-is (GLFW then picks the closest mode). Decorated windows never touch the mode.
TEST(ResolveWindowSize, FullscreenZeroAxisTakesCurrentMode) {
  const Extent2D mode{2560, 1440};
  auto eq = [](Extent2D a, Extent2D b) { return a.width == b.width && a.height == b.height; };
  EXPECT_TRUE(eq(resolve_window_size({0, 0}, true, mode), mode));
  EXPECT_TRUE(eq(resolve_window_size({1920, 0}, true, mode), mode));
  EXPECT_TRUE(eq(resolve_window_size({0, 1080}, true, mode), mode));
  EXPECT_TRUE(eq(resolve_window_size({1920, 1080}, true, mode), Extent2D{1920, 1080}));

  // Decorated windows pass their request through untouched (the caller owns the defaults).
  EXPECT_TRUE(eq(resolve_window_size({0, 0}, false, mode), Extent2D{0, 0}));
  EXPECT_TRUE(eq(resolve_window_size({1920, 0}, false, mode), Extent2D{1920, 0}));
  EXPECT_TRUE(eq(resolve_window_size({1280, 720}, false, mode), Extent2D{1280, 720}));
}

// Zero keeps the monitor's current rate (never "highest available"); an explicit request
// rounds to the nearest Hz without overflowing on large millihertz values.
TEST(SelectRefreshRateHz, ZeroKeepsCurrentRate) {
  EXPECT_EQ(select_refresh_rate_hz(0, 60), 60);
  EXPECT_EQ(select_refresh_rate_hz(0, 144), 144);
}

TEST(SelectRefreshRateHz, ExplicitRateRoundsToNearestHz) {
  EXPECT_EQ(select_refresh_rate_hz(144'000, 60), 144);
  EXPECT_EQ(select_refresh_rate_hz(59'940, 60), 60);
  EXPECT_EQ(select_refresh_rate_hz(120'499, 60), 120);
  EXPECT_EQ(select_refresh_rate_hz(120'500, 60), 121);
  EXPECT_EQ(select_refresh_rate_hz(std::numeric_limits<uint32_t>::max(), 60), 4'294'967);
}

}  // namespace
}  // namespace holoviz::window_systems
