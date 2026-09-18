// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <variant>

#include <gtest/gtest.h>

#include <holoscan/visualization/holoviz_op.hpp>

// HolovizOp's private WindowSinkDesc -> holoviz::WindowDesc mapping (operators/src on the include
// path via the backend object library): pure, so the operator's window policy is testable
// without a GPU.
#include "holoviz_op/holoviz_op_window_desc.hpp"

namespace holoscan::visualization {

// The caller-facing present declaration surface: a SinkDesc holds exactly one sink kind, and
// the EA2-in-scope kinds (Window, ExclusiveDisplay) declare a present mode.
TEST(SinkDescTest, HoldsExactlyOneSinkKind) {
  SinkDesc window = WindowSinkDesc{"viz", 1920, 1080, PresentMode::kFIFOLatestReady};
  ASSERT_TRUE(std::holds_alternative<WindowSinkDesc>(window));
  EXPECT_EQ(std::get<WindowSinkDesc>(window).present_mode, PresentMode::kFIFOLatestReady);

  SinkDesc exclusive = ExclusiveDisplaySinkDesc{/*display_index=*/0};
  ASSERT_TRUE(std::holds_alternative<ExclusiveDisplaySinkDesc>(exclusive));
  EXPECT_FALSE(std::holds_alternative<WindowSinkDesc>(exclusive));
  EXPECT_EQ(std::get<ExclusiveDisplaySinkDesc>(exclusive).present_mode, PresentMode::kAuto);
}

// Fullscreen is an additive refinement of the window sink: off by default, with the monitor and
// refresh selectors defaulting to "primary" / "current mode".
TEST(SinkDescTest, WindowSinkFullscreenDefaultsOff) {
  const WindowSinkDesc windowed{.title = "viz"};
  EXPECT_FALSE(windowed.fullscreen);
  EXPECT_TRUE(windowed.display_name.empty());
  EXPECT_EQ(windowed.refresh_hz, 0U);
  EXPECT_FALSE(windowed.close_on_escape);

  const WindowSinkDesc fullscreen{.title = "viz",
                                  .fullscreen = true,
                                  .display_name = "DP-0",
                                  .refresh_hz = 180,
                                  .close_on_escape = true};
  EXPECT_TRUE(fullscreen.fullscreen);
  EXPECT_EQ(fullscreen.width, 0U);  // 0 = the monitor's current mode when fullscreen
  EXPECT_EQ(fullscreen.display_name, "DP-0");
  EXPECT_EQ(fullscreen.refresh_hz, 180U);
  EXPECT_TRUE(fullscreen.close_on_escape);
}

// HolovizOp maps the sink onto holoviz::WindowDesc. A decorated window defaults each zero
// dimension independently (the pre-fullscreen behavior: width-only 1920 stays 1920x720), while
// a fullscreen window with a zero axis hands {0, 0} to HoloViz to take the monitor's mode.
TEST(SinkDescTest, WindowSinkMapsToWindowDescWithIndependentDefaults) {
  using holoscan::visualization::detail::window_desc_from_sink;

  const auto both_zero = window_desc_from_sink(WindowSinkDesc{.title = "viz"});
  EXPECT_EQ(both_zero.size.width, 1280U);
  EXPECT_EQ(both_zero.size.height, 720U);
  EXPECT_TRUE(both_zero.resizable);
  EXPECT_FALSE(both_zero.fullscreen);

  const auto width_only = window_desc_from_sink(WindowSinkDesc{.title = "viz", .width = 1920});
  EXPECT_EQ(width_only.size.width, 1920U);
  EXPECT_EQ(width_only.size.height, 720U);

  const auto height_only = window_desc_from_sink(WindowSinkDesc{.title = "viz", .height = 1080});
  EXPECT_EQ(height_only.size.width, 1280U);
  EXPECT_EQ(height_only.size.height, 1080U);

  const auto sized =
      window_desc_from_sink(WindowSinkDesc{.title = "viz", .width = 640, .height = 480});
  EXPECT_EQ(sized.size.width, 640U);
  EXPECT_EQ(sized.size.height, 480U);
}

TEST(SinkDescTest, FullscreenWindowSinkMapsToWindowDesc) {
  using holoscan::visualization::detail::window_desc_from_sink;

  const auto monitor_mode = window_desc_from_sink(WindowSinkDesc{.title = "viz",
                                                                 .width = 1920,
                                                                 .fullscreen = true,
                                                                 .display_name = "DP-0",
                                                                 .refresh_hz = 144,
                                                                 .close_on_escape = true});
  EXPECT_EQ(monitor_mode.title, "viz");
  EXPECT_EQ(monitor_mode.size.width, 0U);  // one zero axis: the monitor's current mode
  EXPECT_EQ(monitor_mode.size.height, 0U);
  EXPECT_FALSE(monitor_mode.resizable);
  EXPECT_TRUE(monitor_mode.fullscreen);
  EXPECT_EQ(monitor_mode.display_name, "DP-0");
  EXPECT_EQ(monitor_mode.refresh_rate_mhz, 144'000U);
  EXPECT_TRUE(monitor_mode.close_on_escape);

  const auto explicit_mode = window_desc_from_sink(
      WindowSinkDesc{.title = "viz", .width = 1920, .height = 1080, .fullscreen = true});
  EXPECT_EQ(explicit_mode.size.width, 1920U);
  EXPECT_EQ(explicit_mode.size.height, 1080U);
  EXPECT_EQ(explicit_mode.refresh_rate_mhz, 0U);  // keep the current refresh rate
}

// The CUDA sinks are part of the stable declaration surface even though EA2 does not wire
// their present path — callers can construct code against them today.
TEST(SinkDescTest, CudaSinksAreDeclarable) {
  SinkDesc pitch = CudaPitchLinearSinkDesc{1920, 1080, ImageFormat::kR8G8B8A8Unorm};
  SinkDesc array = CudaArraySinkDesc{1920, 1080, ImageFormat::kR8G8B8A8Unorm};
  EXPECT_TRUE(std::holds_alternative<CudaPitchLinearSinkDesc>(pitch));
  EXPECT_TRUE(std::holds_alternative<CudaArraySinkDesc>(array));
}

}  // namespace holoscan::visualization
