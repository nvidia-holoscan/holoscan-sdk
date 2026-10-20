// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/visualization/holoviz_op.hpp>

// HolovizOp's private WindowSinkDesc -> holoviz::WindowDesc mapping (operators/src on the include
// path via the backend object library): pure, so the operator's window policy is testable
// without a GPU.
#include "common/holoviz_conversions.hpp"
#include "holoviz_op/backends/srgb_surface_format.hpp"
#include "holoviz_op/holoviz_op_window_desc.hpp"

namespace holoscan::visualization {

TEST(SinkDescTest, SRGBFramebufferIsExplicitOptIn) {
  EXPECT_FALSE(WindowSinkDesc{}.framebuffer_srgb);
  EXPECT_FALSE(ExclusiveDisplaySinkDesc{}.framebuffer_srgb);
  EXPECT_TRUE((WindowSinkDesc{.framebuffer_srgb = true}).framebuffer_srgb);
  EXPECT_TRUE((ExclusiveDisplaySinkDesc{.framebuffer_srgb = true}).framebuffer_srgb);
}

TEST(SinkDescTest, SRGBSurfaceSelectionRequiresTheExactColorContract) {
  using backends::resolve_srgb_surface_format;
  using holoviz::ColorSpace;
  using holoviz::Format;
  using holoviz::SurfaceFormat;
  const std::vector<SurfaceFormat> both{{Format::kR8G8B8A8Srgb, ColorSpace::kSrgbNonlinear},
                                        {Format::kB8G8R8A8Srgb, ColorSpace::kSrgbNonlinear}};
  EXPECT_EQ(resolve_srgb_surface_format(both).value(), Format::kB8G8R8A8Srgb);
  EXPECT_EQ(resolve_srgb_surface_format(std::span(both).first(1)).value(), Format::kR8G8B8A8Srgb);
  const std::vector<SurfaceFormat> wrong{{Format::kB8G8R8A8Unorm, ColorSpace::kSrgbNonlinear},
                                         {Format::kR8G8B8A8Srgb, ColorSpace::kHdr10St2084}};
  EXPECT_EQ(resolve_srgb_surface_format(wrong).error().code, holoviz::ErrorCode::kNotImplemented);
  EXPECT_FALSE(resolve_srgb_surface_format({}).has_value());
  EXPECT_EQ(resolve_srgb_surface_format(wrong, true).value(), Format::kB8G8R8A8Unorm);
  EXPECT_FALSE(resolve_srgb_surface_format(std::span(wrong).last(1), true).has_value());
  EXPECT_EQ(resolve_srgb_surface_format(both, true).value(), Format::kB8G8R8A8Srgb);
  for (const auto& pair : both) {
    const auto mapped = common::to_image_format(pair.format);
    ASSERT_TRUE(mapped.has_value());
    EXPECT_EQ(common::to_holoviz_format(*mapped).value(), pair.format);
  }
}

TEST(SinkDescTest, UnormSurfaceSelectionPrefersBgraThenRgba) {
  using backends::resolve_unorm_surface_format;
  using holoviz::ColorSpace;
  using holoviz::Format;
  using holoviz::SurfaceFormat;
  const std::vector<SurfaceFormat> both{{Format::kR8G8B8A8Unorm, ColorSpace::kSrgbNonlinear},
                                        {Format::kB8G8R8A8Unorm, ColorSpace::kSrgbNonlinear}};
  EXPECT_EQ(resolve_unorm_surface_format(both).value(), Format::kB8G8R8A8Unorm);
  EXPECT_EQ(resolve_unorm_surface_format(std::span(both).first(1)).value(), Format::kR8G8B8A8Unorm);
  // sRGB storage and a non-sRGB color space are both rejected: the default framebuffer writes
  // UNORM and presents sRGB-nonlinear, and neither half of that contract is substitutable.
  const std::vector<SurfaceFormat> wrong{{Format::kB8G8R8A8Srgb, ColorSpace::kSrgbNonlinear},
                                         {Format::kB8G8R8A8Unorm, ColorSpace::kHdr10St2084}};
  EXPECT_EQ(resolve_unorm_surface_format(wrong).error().code, holoviz::ErrorCode::kNotImplemented);
  EXPECT_FALSE(resolve_unorm_surface_format({}).has_value());
}

// The caller-facing present declaration surface: a SinkDesc holds exactly one sink kind, and
// the EA2-in-scope kinds (Window, ExclusiveDisplay) declare a present mode.
TEST(SinkDescTest, HoldsExactlyOneSinkKind) {
  SinkDesc window = WindowSinkDesc{"viz", 1920, 1080, PresentMode::kFIFOLatestReady};
  ASSERT_TRUE(std::holds_alternative<WindowSinkDesc>(window));
  EXPECT_EQ(std::get<WindowSinkDesc>(window).present_mode, PresentMode::kFIFOLatestReady);

  SinkDesc exclusive = ExclusiveDisplaySinkDesc{.display_name = "DP-2", .refresh_hz = 144};
  ASSERT_TRUE(std::holds_alternative<ExclusiveDisplaySinkDesc>(exclusive));
  EXPECT_FALSE(std::holds_alternative<WindowSinkDesc>(exclusive));
  EXPECT_EQ(std::get<ExclusiveDisplaySinkDesc>(exclusive).display_name, "DP-2");
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

// The CUDA sink is declarable positionally, defaults to RGBA8, and is the third and last kind.
TEST(SinkDescTest, CudaSinkIsDeclarable) {
  SinkDesc cuda = CudaSinkDesc{1920, 1080, ImageFormat::kR8G8B8A8Unorm};
  ASSERT_TRUE(std::holds_alternative<CudaSinkDesc>(cuda));
  EXPECT_EQ(std::get<CudaSinkDesc>(cuda).width, 1920U);
  EXPECT_EQ(CudaSinkDesc{}.format, ImageFormat::kR8G8B8A8Unorm);
  EXPECT_EQ(std::variant_size_v<SinkDesc>, 3U);
}

}  // namespace holoscan::visualization
