/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure window-system policy tests: monitor-name selection, size and refresh-rate selection
// for fullscreen placement, and the window-monitor -> holoviz::Display join behind
// display_for_window(). No window system or device is needed.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
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

// Any of libwayland's three connection routes (named socket, inherited fd, default socket in the
// runtime dir) keeps GLFW's own probing; only a route-less environment is headless, and X11 is
// pinned only when no Wayland route exists at all.
TEST(ChooseGLFWPlatform, AnyWaylandRouteLetsGLFWProbe) {
  EXPECT_EQ(choose_glfw_platform({.wayland_display = true}), GLFWPlatformChoice::kAny);
  EXPECT_EQ(choose_glfw_platform({.wayland_socket = true}), GLFWPlatformChoice::kAny);
  EXPECT_EQ(choose_glfw_platform({.xdg_runtime_dir = true}), GLFWPlatformChoice::kAny);
  // Default Wayland socket plus DISPLAY: GLFW decides (XDG_SESSION_TYPE), never forced X11.
  EXPECT_EQ(choose_glfw_platform({.xdg_runtime_dir = true, .x11_display = true}),
            GLFWPlatformChoice::kAny);
  EXPECT_EQ(choose_glfw_platform({.wayland_display = true, .x11_display = true}),
            GLFWPlatformChoice::kAny);
}

TEST(ChooseGLFWPlatform, X11OnlyPinsX11) {
  EXPECT_EQ(choose_glfw_platform({.x11_display = true}), GLFWPlatformChoice::kX11Only);
}

TEST(ChooseGLFWPlatform, NoRouteIsHeadless) {
  EXPECT_EQ(choose_glfw_platform({}), GLFWPlatformChoice::kNone);
}

Display make_display(std::string_view name, uint32_t width = 0, uint32_t height = 0,
                     uint32_t physical_w = 0, uint32_t physical_h = 0, uint32_t refresh_mhz = 0) {
  Display display{};
  std::memcpy(display.name.data(), name.data(), std::min(name.size(), display.name.size() - 1));
  display.current_mode.resolution = {width, height};
  display.current_mode.refresh_rate_mhz = refresh_mhz;
  display.physical_size_mm = {physical_w, physical_h};
  return display;
}

NativeMonitorInfo make_monitor(std::string_view name, uint32_t width = 0, uint32_t height = 0,
                               uint32_t physical_w = 0, uint32_t physical_h = 0,
                               uint32_t refresh_mhz = 0) {
  NativeMonitorInfo monitor{};
  std::memcpy(monitor.name.data(), name.data(), std::min(name.size(), monitor.name.size() - 1));
  monitor.current_resolution = {width, height};
  monitor.refresh_rate_mhz = refresh_mhz;
  monitor.physical_size_mm = {physical_w, physical_h};
  return monitor;
}

TEST(MatchDisplay, ExactNameWins) {
  const std::vector<Display> displays{make_display("DP-2"), make_display("HDMI-1")};
  EXPECT_EQ(match_display(displays, make_monitor("HDMI-1")), 1U);
}

TEST(MatchDisplay, CaseInsensitiveSubstringEitherDirection) {
  // Backend reports an EDID model string; the window system reports the connector name —
  // or the reverse. Both directions must join.
  const std::vector<Display> displays{make_display("DELL U2723QE (DP-2)"),
                                      make_display("ACME Panel")};
  EXPECT_EQ(match_display(displays, make_monitor("dp-2")), 0U);

  const std::vector<Display> connector_names{make_display("DP-2"), make_display("HDMI-1")};
  EXPECT_EQ(match_display(connector_names, make_monitor("Something dp-2 Something")), 0U);
}

TEST(MatchDisplay, AmbiguousSubstringAborts) {
  // The monitor name is a substring of both display names: ambiguous at tier 2, and with
  // no geometry to break the tie the join must refuse to guess.
  const std::vector<Display> displays{make_display("left DP-2 panel"),
                                      make_display("right DP-2 panel")};
  EXPECT_FALSE(match_display(displays, make_monitor("DP-2")).has_value());
}

TEST(MatchDisplay, ResolutionAndPhysicalSizeTierBreaksNameMismatch) {
  const std::vector<Display> displays{make_display("display 0", 1920, 1080, 600, 340),
                                      make_display("display 1", 3840, 2160, 700, 400)};
  // Names share the "display" prefix on both entries -> ambiguous tier 2; tier 3 decides.
  EXPECT_EQ(match_display(displays, make_monitor("DP-4", 3840, 2160, 698, 398)), 1U);
  // Physical size outside tolerance rejects the resolution match.
  EXPECT_FALSE(match_display(displays, make_monitor("DP-4", 3840, 2160, 200, 100)).has_value());
}

TEST(MatchDisplay, RefreshRateBreaksIdenticalGeometry) {
  // Two panels of the same model: same resolution and physical size, different active refresh
  // rate. The window system's rate (millihertz, like DisplayMode) is the only tier-3 tie-breaker.
  const std::vector<Display> displays{make_display("display 0", 1920, 1080, 600, 340, 60'000),
                                      make_display("display 1", 1920, 1080, 600, 340, 144'000)};
  EXPECT_EQ(match_display(displays, make_monitor("DP-4", 1920, 1080, 600, 340, 144'000)), 1U);
  // A side that does not know its rate (0) must not reject the match on it.
  EXPECT_FALSE(match_display(displays, make_monitor("DP-4", 1920, 1080, 600, 340)).has_value());
  const std::vector<Display> unknown_rate{make_display("display 0", 1920, 1080, 600, 340),
                                          make_display("other", 3840, 2160)};
  EXPECT_EQ(match_display(unknown_rate, make_monitor("DP-4", 1920, 1080, 600, 340, 60'000)), 0U);
}

TEST(MatchDisplay, SingleCandidateFallsThrough) {
  const std::vector<Display> displays{make_display("weird driver name")};
  EXPECT_EQ(match_display(displays, make_monitor("DP-1", 1920, 1080)), 0U);
}

TEST(MatchDisplay, SameNameDifferentPanelIsRejectedByGeometry) {
  // X11 calls the 1080p/144 Hz panel DP-2; DRM calls that panel DP-3 and gives "DP-2" to a
  // different 1440p/180 Hz panel. The exact-name hit is contradicted by geometry and must
  // not win; tier 3 then finds the real panel.
  const std::vector<Display> displays{make_display("DP-2", 2560, 1440, 600, 340, 180'000),
                                      make_display("DP-3", 1920, 1080, 530, 300, 144'000)};
  EXPECT_EQ(match_display(displays, make_monitor("DP-2", 1920, 1080, 530, 300, 144'000)), 1U);
  // Resolution alone (rate/size unknown on the monitor side) is enough of a contradiction.
  EXPECT_EQ(match_display(displays, make_monitor("DP-2", 1920, 1080)), 1U);
  // Same resolution, different refresh rate: still another panel.
  const std::vector<Display> same_res{make_display("DP-2", 1920, 1080, 0, 0, 60'000),
                                      make_display("DP-3", 1920, 1080, 0, 0, 144'000)};
  EXPECT_EQ(match_display(same_res, make_monitor("DP-2", 1920, 1080, 0, 0, 144'000)), 1U);
}

TEST(MatchDisplay, ExactNameWithUnknownGeometryStillWins) {
  // A name-only display record (no mode/size known) cannot contradict anything.
  const std::vector<Display> displays{make_display("DP-2"), make_display("HDMI-1")};
  EXPECT_EQ(match_display(displays, make_monitor("DP-2", 1920, 1080, 530, 300, 144'000)), 0U);
}

TEST(MatchDisplay, SubstringCollisionIsRejectedByGeometry) {
  // The EDID-string display contains the monitor's connector name but is a 4K panel; the
  // window is on the 1080p one. The substring hit is vetoed and tier 3 decides.
  const std::vector<Display> displays{make_display("DELL U2723QE (DP-2)", 3840, 2160),
                                      make_display("ACME Panel", 1920, 1080)};
  EXPECT_EQ(match_display(displays, make_monitor("dp-2", 1920, 1080)), 1U);
}

TEST(MatchDisplay, SoleDisplayFromWrongAdapterIsRejected) {
  // One display on this adapter, but it is demonstrably another panel: no match rather
  // than pacing on a foreign display.
  const std::vector<Display> wrong_panel{make_display("DP-2", 2560, 1440)};
  EXPECT_FALSE(match_display(wrong_panel, make_monitor("DP-2", 1920, 1080)).has_value());
  EXPECT_FALSE(match_display(wrong_panel, make_monitor("HDMI-1", 1920, 1080)).has_value());
  // Physical size beyond tolerance is a contradiction too.
  const std::vector<Display> wrong_size{make_display("DP-2", 1920, 1080, 700, 400)};
  EXPECT_FALSE(match_display(wrong_size, make_monitor("DP-2", 1920, 1080, 530, 300)).has_value());
  // Unknown geometry on the display side is not a contradiction: the sole candidate still
  // falls through (SingleCandidateFallsThrough's contract).
  const std::vector<Display> unknown{make_display("weird driver name")};
  EXPECT_EQ(match_display(unknown, make_monitor("DP-2", 1920, 1080, 530, 300, 144'000)), 0U);
}

TEST(MatchDisplay, NoUnambiguousMatchReturnsNothing) {
  const std::vector<Display> displays{make_display("DP-2", 1920, 1080),
                                      make_display("DP-3", 1920, 1080)};
  // Name misses both; resolution matches both (ambiguous); two candidates remain.
  EXPECT_FALSE(match_display(displays, make_monitor("HDMI-9", 1920, 1080)).has_value());
}

// A monitor rotated 90/270 degrees: GLFW (X11) reports the desktop-oriented 1080x1920 mode
// and 300x530 mm, the driver the panel's native 1920x1080 and 530x300 mm. Same panel.
TEST(MatchDisplay, RotatedMonitorStillMatchesByName) {
  const std::vector<Display> displays{make_display("DP-2", 1920, 1080, 530, 300, 144'000),
                                      make_display("DP-3", 2560, 1440, 600, 340, 60'000)};
  EXPECT_EQ(match_display(displays, make_monitor("DP-2", 1080, 1920, 300, 530, 144'000)), 0U);
  // Rotated resolution alone (size unknown on the monitor side) is not a contradiction.
  EXPECT_EQ(match_display(displays, make_monitor("DP-2", 1080, 1920)), 0U);
  // The unrotated control still matches.
  EXPECT_EQ(match_display(displays, make_monitor("DP-2", 1920, 1080, 530, 300, 144'000)), 0U);
}

TEST(MatchDisplay, RotatedSoleCandidateMatches) {
  const std::vector<Display> sole{make_display("DP-2", 1920, 1080, 530, 300)};
  EXPECT_EQ(match_display(sole, make_monitor("HDMI-1", 1080, 1920, 300, 530)), 0U);
}

TEST(MatchDisplay, RotatedResolutionTierMatches) {
  // Names never match (driver reports EDID strings); the rotated active mode joins the panel
  // through tier 3, and the veto still rejects the other panel.
  const std::vector<Display> displays{make_display("DELL U2723QE", 3840, 2160, 600, 340),
                                      make_display("ACME Panel", 1920, 1080, 530, 300)};
  EXPECT_EQ(match_display(displays, make_monitor("DP-4", 1080, 1920, 300, 530)), 1U);
}

TEST(MatchDisplay, RuntimeRotationWithCachedPhysicalSizeMatches) {
  // GLFW 3.4 on X11 caches a monitor's physical size in the orientation it first saw and reads
  // the video mode from the current CRTC: after a landscape panel is rotated to portrait at
  // runtime the monitor record is 1080x1920 px with a still-landscape 530x300 mm. Both
  // directions of runtime rotation must join the panel.
  const std::vector<Display> displays{make_display("DP-2", 1920, 1080, 530, 300),
                                      make_display("DP-3", 2560, 1440, 600, 340)};
  EXPECT_EQ(match_display(displays, make_monitor("DP-2", 1080, 1920, 530, 300)), 0U);
  const std::vector<Display> portrait_native{make_display("DP-2", 1080, 1920, 300, 530),
                                             make_display("DP-3", 2560, 1440, 600, 340)};
  EXPECT_EQ(match_display(portrait_native, make_monitor("DP-2", 1920, 1080, 300, 530)), 0U);
}

TEST(MatchDisplay, DifferentPanelIsRejectedInBothOrientations) {
  // Same pixel count, but a physical size that matches in neither orientation: another panel.
  const std::vector<Display> portrait{make_display("DP-2", 1080, 1920, 300, 530)};
  EXPECT_FALSE(match_display(portrait, make_monitor("DP-2", 1080, 1920, 600, 340)).has_value());
  // Same physical size in some orientation, but a resolution that matches in neither.
  const std::vector<Display> displays{make_display("DP-2", 1920, 1080, 530, 300)};
  EXPECT_FALSE(match_display(displays, make_monitor("DP-2", 2560, 1440, 300, 530)).has_value());
}

TEST(SelectMonitorByOverlap, UnknownWindowPositionSelectsNothing) {
  // Wayland: GLFW cannot report the window position. The fallback value (0,0) would overlap
  // the origin output; an unknown position must select nothing instead.
  const std::vector<Rect2D> monitors{{.x = 0, .y = 0, .width = 1920, .height = 1080},
                                     {.x = 1920, .y = 0, .width = 2560, .height = 1440}};
  EXPECT_FALSE(select_monitor_by_overlap(std::nullopt, monitors).has_value());
  // The same rectangle with a known origin position does select the origin output.
  const Rect2D at_origin{.x = 0, .y = 0, .width = 320, .height = 240};
  EXPECT_EQ(select_monitor_by_overlap(at_origin, monitors), 0U);
}

TEST(SelectMonitorByOverlap, LargestOverlapWins) {
  const std::vector<Rect2D> monitors{{.x = 0, .y = 0, .width = 1920, .height = 1080},
                                     {.x = 1920, .y = 0, .width = 2560, .height = 1440}};
  // Straddling the seam, mostly on the second monitor.
  EXPECT_EQ(
      select_monitor_by_overlap(Rect2D{.x = 1800, .y = 100, .width = 640, .height = 480}, monitors),
      1U);
  // Mostly on the first.
  EXPECT_EQ(
      select_monitor_by_overlap(Rect2D{.x = 1500, .y = 100, .width = 640, .height = 480}, monitors),
      0U);
  // Negative desktop coordinates (a monitor left of the primary) are ordinary positions.
  const std::vector<Rect2D> left_of_origin{{.x = -1920, .y = 0, .width = 1920, .height = 1080},
                                           {.x = 0, .y = 0, .width = 1920, .height = 1080}};
  EXPECT_EQ(select_monitor_by_overlap(Rect2D{.x = -1000, .y = 10, .width = 320, .height = 240},
                                      left_of_origin),
            0U);
}

TEST(SelectMonitorByOverlap, NoOverlapSelectsNothing) {
  const std::vector<Rect2D> monitors{{.x = 0, .y = 0, .width = 1920, .height = 1080}};
  EXPECT_FALSE(
      select_monitor_by_overlap(Rect2D{.x = 5000, .y = 5000, .width = 320, .height = 240}, monitors)
          .has_value());
  // A monitor without a video mode has zero area and never overlaps.
  const std::vector<Rect2D> modeless{{.x = 0, .y = 0, .width = 0, .height = 0}};
  EXPECT_FALSE(
      select_monitor_by_overlap(Rect2D{.x = 0, .y = 0, .width = 320, .height = 240}, modeless)
          .has_value());
  EXPECT_FALSE(select_monitor_by_overlap(std::nullopt, {}).has_value());
}

}  // namespace
}  // namespace holoviz::window_systems
