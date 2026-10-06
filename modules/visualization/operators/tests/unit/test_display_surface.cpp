// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// DisplaySurface tests. The device-free part checks the factory's argument validation
// and the surface-kind policy; the acquire/present protocol runs only on a machine with a
// display that is free for exclusive scanout (skipped otherwise — a desktop session owns its
// displays, so CI normally exercises just the policy checks).

#include <chrono>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoviz/holoviz.hpp>
#include "holoviz_op/backends/display_surface.hpp"
#include "holoviz_op/backends/holoviz_render_backend.hpp"
#include "holoviz_op/backends/present_backend.hpp"
#include "holoviz_op/backends/srgb_surface_format.hpp"

#include "holoviz_op/backends/backend_tokens.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoscan::visualization::backends {
namespace {

TEST(DisplaySurface, FactoryRejectsAnEmptyDevice) {
  holoviz::Device empty;
  const auto backend = create_display_surface(empty);
  ASSERT_FALSE(backend.has_value());
  EXPECT_EQ(backend.error().code, ErrorCode::kInvalidArgument);
}

TEST(DisplaySurface, ProtocolCallsBeforeCreateAreInvalidState) {
  auto adapters = holoviz::enumerate_adapters();
  if (!adapters || adapters->empty()) {
    HOLOVIZ_ENV_SKIP() << "no adapter";
  }
  auto device = adapters->front().create_device({});
  if (!device) {
    HOLOVIZ_ENV_SKIP() << "no device: " << device.error().message;
  }
  auto backend = create_display_surface(*device);
  ASSERT_TRUE(backend.has_value()) << backend.error().message;

  EXPECT_EQ((*backend)->info().name, "display_surface");
  EXPECT_EQ((*backend)->acquire(std::chrono::milliseconds{1}).error().code,
            ErrorCode::kInvalidState);
  EXPECT_EQ((*backend)->swapchain_info().error().code, ErrorCode::kInvalidState);
  // Display surfaces are fixed-size: resize() is rejected as an argument error even before a
  // swapchain exists (there is no extent the caller could legitimately pick).
  EXPECT_EQ((*backend)->resize({1, 1}).error().code, ErrorCode::kInvalidArgument);
  // A foreign surface handle never reaches HoloViz.
  EXPECT_EQ((*backend)->create_swapchain(SwapchainDesc{}).error().code,
            ErrorCode::kInvalidArgument);
  EXPECT_TRUE((*backend)->destroy(std::chrono::seconds{1}).has_value());
}

/// The first display free for exclusive scanout, or nullopt.
std::optional<holoviz::Display> find_available_display() {
  auto adapters = holoviz::enumerate_adapters();
  if (!adapters) {
    return std::nullopt;
  }
  for (const holoviz::Adapter& adapter : *adapters) {
    auto displays = holoviz::enumerate_displays(adapter);
    if (!displays) {
      continue;
    }
    for (const holoviz::Display& display : *displays) {
      if (display.ownership == holoviz::DisplayOwnership::kAvailable) {
        return display;
      }
    }
  }
  return std::nullopt;
}

/// The mode a zeroed DisplaySurfaceDesc resolves to: largest width, then largest height, then
/// fastest refresh (the documented "don't care" default). Independent re-statement of the
/// selection rule so the test does not borrow the implementation's scorer.
holoviz::DisplayMode largest_fastest_mode(const std::vector<holoviz::DisplayMode>& modes) {
  holoviz::DisplayMode best{};
  for (const auto& mode : modes) {
    const bool better = mode.resolution.width > best.resolution.width ||
                        (mode.resolution.width == best.resolution.width &&
                         (mode.resolution.height > best.resolution.height ||
                          (mode.resolution.height == best.resolution.height &&
                           mode.refresh_rate_mhz > best.refresh_rate_mhz)));
    if (better) {
      best = mode;
    }
  }
  return best;
}

class DisplaySurfaceHardwareTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto display = find_available_display();
    if (!display) {
      GTEST_SKIP() << "no display is free for exclusive scanout (detach one from the desktop "
                      "session to run this test)";
    }
    display_ = *display;
    auto modes = display_.modes();
    ASSERT_TRUE(modes.has_value()) << modes.error().message;
    ASSERT_FALSE(modes->empty());
    modes_ = std::move(*modes);

    auto surface_handle = make_display_surface_handle(display_);
    if (!surface_handle) {
      GTEST_SKIP() << "the display could not be acquired: " << surface_handle.error().message;
    }
    surface_handle_ = std::move(*surface_handle);

    auto device = holoviz::create_device_for_surface({}, *surface_from_handle(surface_handle_));
    if (!device) {
      HOLOVIZ_ENV_SKIP() << "no presentable adapter: " << device.error().message;
    }
    device_ = std::move(*device);

    auto backend = create_display_surface(device_);
    ASSERT_TRUE(backend.has_value()) << backend.error().message;
    backend_ = std::move(*backend);
  }

  void TearDown() override {
    if (backend_ != nullptr) {
      EXPECT_TRUE(backend_->destroy(std::chrono::seconds(5)).has_value());
      backend_.reset();
    }
    surface_handle_.impl.reset();
  }

  /// Release the fixture's surface (and backend) so a test can acquire the display again with a
  /// different mode request; a display is acquired by at most one surface at a time.
  void release_fixture_surface() {
    if (backend_ != nullptr) {
      ASSERT_TRUE(backend_->destroy(std::chrono::seconds(5)).has_value());
      backend_.reset();
    }
    surface_handle_.impl.reset();
  }

  holoviz::Display display_;
  std::vector<holoviz::DisplayMode> modes_;
  SurfaceHandle surface_handle_;
  holoviz::Device device_;
  std::unique_ptr<PresentBackend> backend_;
};

TEST_F(DisplaySurfaceHardwareTest, NativeSRGBSwapchainUsesSRGBImages) {
  const auto caps = device_.surface_capabilities(*surface_from_handle(surface_handle_));
  ASSERT_TRUE(caps.has_value());
  const auto format =
      resolve_srgb_surface_format(caps->formats, device_.capabilities()->mutable_swapchain_format);
  const auto created = backend_->create_swapchain(
      SwapchainDesc{.surface = surface_handle_, .framebuffer_srgb = true});
  if (!format) {
    ASSERT_FALSE(created.has_value());
    EXPECT_EQ(created.error().code, ErrorCode::kNotImplemented);
    return;
  }
  ASSERT_TRUE(created.has_value()) << created.error().message;
  const auto info = backend_->swapchain_info();
  ASSERT_TRUE(info.has_value());
  EXPECT_TRUE(info->format == ImageFormat::kB8G8R8A8Srgb ||
              info->format == ImageFormat::kR8G8B8A8Srgb);
  auto acquired = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(acquired.has_value());
  auto renderer = create_holoviz_render_backend(device_, {});
  ASSERT_TRUE(renderer.has_value());
  RenderConfig config{.max_frames_in_flight = created->image_count};
  config.clear_color[0] = config.clear_color[1] = config.clear_color[2] = 0.18F;
  ASSERT_TRUE((*renderer)->allocate(LayerStack{}, config).has_value());
  ASSERT_TRUE((*renderer)->build(info->format).has_value());
  auto commands = (*renderer)->begin_frame(*acquired, 0);
  ASSERT_TRUE(commands.has_value());
  ASSERT_TRUE((*renderer)->record(*commands, RenderList{}, CompositeSpec{}).has_value());
  auto submitted = (*renderer)->submit(*commands, SubmitContext{0});
  ASSERT_TRUE(submitted.has_value());
  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = acquired->image_index,
                                        .acquire_issuance = acquired->issuance,
                                        .render_done = *submitted})
                  .has_value());
  ASSERT_TRUE((*renderer)->destroy(std::chrono::seconds(5)).has_value());
}

TEST_F(DisplaySurfaceHardwareTest, MutableSRGBViewsSurviveSwapchainRecreation) {
  const auto caps = device_.capabilities();
  ASSERT_TRUE(caps.has_value());
  if (!caps->mutable_swapchain_format) {
    GTEST_SKIP() << "mutable swapchain views unavailable";
  }
  const auto surface_caps = device_.surface_capabilities(*surface_from_handle(surface_handle_));
  ASSERT_TRUE(surface_caps.has_value());
  bool has_unorm = false;
  for (const auto& pair : surface_caps->formats) {
    has_unorm |= pair.format == holoviz::Format::kB8G8R8A8Unorm &&
                 pair.color_space == holoviz::ColorSpace::kSrgbNonlinear;
  }
  if (!has_unorm) {
    GTEST_SKIP() << "BGRA8 UNORM scanout unavailable";
  }
  auto swapchain = device_.create_swapchain(
      holoviz::SwapchainDesc{.surface = std::cref(*surface_from_handle(surface_handle_)),
                             .format = holoviz::Format::kB8G8R8A8Unorm,
                             .srgb_attachment = true});
  ASSERT_TRUE(swapchain.has_value()) << swapchain.error().message;
  const auto generation = swapchain->generation().value();
  for (int iteration = 0; iteration < 2; ++iteration) {
    EXPECT_EQ(swapchain->format().value(), holoviz::Format::kB8G8R8A8Srgb);
    auto acquired = swapchain->acquire(std::chrono::seconds(5));
    ASSERT_TRUE(acquired.has_value());
    EXPECT_EQ(swapchain->image(acquired->image_index)->get().format().value(),
              holoviz::Format::kB8G8R8A8Srgb);
    // Render before presenting a fresh image; an empty present cannot initialize its layout.
    auto pass = device_.create_render_pass(
        holoviz::RenderPassDesc{.render_targets = {{.format = holoviz::Format::kB8G8R8A8Srgb}}});
    ASSERT_TRUE(pass.has_value());
    ASSERT_TRUE(pass->set_render_image(swapchain->image(acquired->image_index)->get(),
                                       0,
                                       {0.18F, 0.18F, 0.18F, 1.F})
                    .has_value());
    auto commands = device_.create_command_list({});
    ASSERT_TRUE(commands.has_value());
    const auto extent = swapchain->extent().value();
    ASSERT_TRUE(commands->begin().has_value());
    ASSERT_TRUE(commands->begin_render(*pass, {0, 0, extent.width, extent.height}).has_value());
    ASSERT_TRUE(commands->end_render().has_value());
    ASSERT_TRUE(commands->end().has_value());
    auto submitted =
        device_.submit(holoviz::SubmitDesc{.command_lists = {&*commands, 1},
                                           .acquire_gates = {&acquired->acquire_gate, 1},
                                           .present_gates = {&acquired->present_gate, 1}});
    ASSERT_TRUE(submitted.has_value());
    ASSERT_TRUE(device_.wait_on_submit(*submitted, std::chrono::seconds(5)).has_value());
    ASSERT_TRUE(swapchain->present(std::move(*acquired), *submitted).has_value());
    ASSERT_TRUE(swapchain->resize(swapchain->extent().value()).has_value());
    EXPECT_GT(swapchain->generation().value(), generation);
  }
  ASSERT_TRUE(swapchain->destroy(std::chrono::seconds(5)).has_value());
}

TEST_F(DisplaySurfaceHardwareTest, CreatesAFixedSizeTwoImageSwapchainAndPresents) {
  auto created = backend_->create_swapchain(SwapchainDesc{
      .surface = surface_handle_,
      .preferred_extent = {0, 0},  // ignored: the display mode sizes the surface
      .preferred_mode = holoviz::PresentMode::kAuto,
      .image_count = std::nullopt,
  });
  ASSERT_TRUE(created.has_value()) << created.error().message;
  EXPECT_GT(created->extent.width, 0U);
  EXPECT_GT(created->extent.height, 0U);
  EXPECT_GE(created->image_count, 2U);
  EXPECT_NE(created->present_mode, holoviz::PresentMode::kAuto);

  const auto surface_extent = surface_from_handle(surface_handle_)->extent();
  ASSERT_TRUE(surface_extent.has_value());
  EXPECT_EQ(created->extent.width, surface_extent->width);
  EXPECT_EQ(created->extent.height, surface_extent->height);

  // The fixture asked for a zeroed mode; the documented default is the display's largest
  // resolution (then its fastest refresh), so the surface must be sized to exactly that mode.
  const holoviz::DisplayMode default_mode = largest_fastest_mode(modes_);
  EXPECT_EQ(surface_extent->width, default_mode.resolution.width);
  EXPECT_EQ(surface_extent->height, default_mode.resolution.height);

  EXPECT_EQ(backend_->resize(created->extent).error().code, ErrorCode::kInvalidArgument);

  // A few cleared frames drive the whole acquire/render/present/wait protocol on the display.
  // Every acquired image is rendered before it is presented: a fresh swapchain image has no
  // layout yet, and presenting it untouched violates VUID-VkPresentInfoKHR-pImageIndices-01430
  // (the validation layer runs under HOLOVIZ_VALIDATION=1).
  const auto info = backend_->swapchain_info();
  ASSERT_TRUE(info.has_value()) << info.error().message;
  auto renderer = create_holoviz_render_backend(device_, {});
  ASSERT_TRUE(renderer.has_value()) << renderer.error().message;
  RenderConfig config{.max_frames_in_flight = created->image_count};
  config.clear_color[0] = config.clear_color[1] = config.clear_color[2] = 0.18F;
  ASSERT_TRUE((*renderer)->allocate(LayerStack{}, config).has_value());
  ASSERT_TRUE((*renderer)->build(info->format).has_value());
  for (PresentId id = 1; id <= 3; ++id) {
    auto acquired = backend_->acquire(std::chrono::seconds{2});
    ASSERT_TRUE(acquired.has_value()) << acquired.error().message;
    const auto frame_index = static_cast<uint32_t>((id - 1) % created->image_count);
    auto commands = (*renderer)->begin_frame(*acquired, frame_index);
    ASSERT_TRUE(commands.has_value()) << commands.error().message;
    ASSERT_TRUE((*renderer)->record(*commands, RenderList{}, CompositeSpec{}).has_value());
    auto submitted = (*renderer)->submit(*commands, SubmitContext{frame_index});
    ASSERT_TRUE(submitted.has_value()) << submitted.error().message;
    const auto presented = backend_->present(PresentArgs{.image_index = acquired->image_index,
                                                         .acquire_issuance = acquired->issuance,
                                                         .render_done = *submitted,
                                                         .present_id = id});
    ASSERT_TRUE(presented.has_value()) << "present " << id << ": " << presented.error().message;
    const auto waited = backend_->wait_for_present(id, std::chrono::seconds{2});
    if (backend_->info().has_present_wait) {
      ASSERT_TRUE(waited.has_value()) << waited.error().message;
      EXPECT_NE(*waited, WaitResult::kDropped);
    } else {
      EXPECT_EQ(waited.error().code, ErrorCode::kNotImplemented);
    }
  }
  // Children before the device: the renderer goes before the fixture tears down the backend.
  ASSERT_TRUE((*renderer)->destroy(std::chrono::seconds(5)).has_value());
}

TEST_F(DisplaySurfaceHardwareTest, ExplicitModeRequestSelectsThatMode) {
  // Pick a supported resolution other than the default so the test can tell the explicit
  // request apart from the "largest" fallback.
  const holoviz::DisplayMode default_mode = largest_fastest_mode(modes_);
  std::optional<holoviz::DisplayMode> other;
  for (const auto& mode : modes_) {
    if (mode.resolution.width != default_mode.resolution.width ||
        mode.resolution.height != default_mode.resolution.height) {
      other = mode;
      break;
    }
  }
  if (!other) {
    GTEST_SKIP() << "the display offers a single resolution";
  }

  release_fixture_surface();
  auto surface_handle = make_display_surface_handle(display_, {.mode = *other});
  ASSERT_TRUE(surface_handle.has_value()) << surface_handle.error().message;
  const auto extent = surface_from_handle(*surface_handle)->extent();
  ASSERT_TRUE(extent.has_value());
  EXPECT_EQ(extent->width, other->resolution.width);
  EXPECT_EQ(extent->height, other->resolution.height);
  surface_handle->impl.reset();
}

}  // namespace
}  // namespace holoscan::visualization::backends
