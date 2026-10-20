// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// DisplaySurface tests. The device-free part checks the factory's argument validation
// and the surface-kind policy; the acquire/present protocol runs only on a machine with a
// display that is free for exclusive scanout (skipped otherwise — a desktop session owns its
// displays, so CI normally exercises just the policy checks). On a driver-managed display
// (Tegra iGPUs) a busy display is only detected at swapchain creation, so the hardware tests
// also skip when that fails, as it does while an Xorg server is running.

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
#include "vulkan/vk_swapchain_fault_injection.hpp"

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

/// Prefer an available display, then attempt one whose ownership is unknown,
/// matching the operator's exclusive-display selection policy.
std::optional<holoviz::Display> find_available_display() {
  auto adapters = holoviz::enumerate_adapters();
  if (!adapters) {
    return std::nullopt;
  }
  std::optional<holoviz::Display> unknown;
  for (const holoviz::Adapter& adapter : *adapters) {
    auto displays = holoviz::enumerate_displays(adapter);
    if (!displays) {
      continue;
    }
    for (const holoviz::Display& display : *displays) {
      if (display.ownership == holoviz::DisplayOwnership::kAvailable) {
        return display;
      }
      if (!unknown && display.ownership == holoviz::DisplayOwnership::kUnknown) {
        unknown = display;
      }
    }
  }
  return unknown;
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

// On a driver-managed display (DisplayAcquisitionMode::kDriverManaged: Tegra iGPUs) HoloViz skips
// the libDRM ownership check and the driver acquires the display only when a swapchain is
// created. While a display server such as Xorg is active the display therefore still looks free
// and the surface is created, but swapchain creation fails. That failure is the expected behavior
// in that environment, not a defect, so the test skips instead of failing. Use it only to
// qualify the environment on a test's first swapchain creation, never on a repeated creation
// whose success is the behavior under test.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage) — must expand GTEST_SKIP in the test body.
#define SKIP_IF_DRIVER_MANAGED_DISPLAY_TAKEN(result)                                       \
  if (driver_managed_ && !(result).has_value()) {                                          \
    GTEST_SKIP() << "the driver could not acquire the driver-managed display (is an Xorg " \
                    "server active?): "                                                    \
                 << (result).error().message;                                              \
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
    refresh_driver_managed();

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

  /// Record whether the current surface's display is acquired by the driver at swapchain
  /// creation (see SKIP_IF_DRIVER_MANAGED_DISPLAY_TAKEN).
  void refresh_driver_managed() {
    const auto mode = surface_from_handle(surface_handle_)->display_acquisition_mode();
    ASSERT_TRUE(mode.has_value()) << mode.error().message;
    driver_managed_ = *mode == holoviz::DisplayAcquisitionMode::kDriverManaged;
  }

  /// Release the fixture's surface (and backend) so a test can acquire the display again with a
  /// different mode request. Destroy the swapchain before the surface so both explicit
  /// acquisition and driver-managed ownership have ended.
  void release_fixture_surface() {
    if (backend_ != nullptr) {
      ASSERT_TRUE(backend_->destroy(std::chrono::seconds(5)).has_value());
      backend_.reset();
    }
    surface_handle_.impl.reset();
  }

  /// Clear the acquired image and submit it, waiting for the render to finish so a later
  /// zero-timeout call meets no render backpressure. Renders before presenting: a fresh
  /// swapchain image has no layout until it is rendered.
  holoviz::expected<holoviz::SubmitSyncToken> render_cleared(holoviz::Swapchain& swapchain,
                                                             holoviz::AcquireSyncToken& acquired) {
    const auto format = swapchain.format();
    if (!format) {
      return holoviz::make_unexpected(format.error());
    }
    auto pass = device_.create_render_pass(
        holoviz::RenderPassDesc{.render_targets = {{.format = *format}}});
    if (!pass) {
      return holoviz::make_unexpected(pass.error());
    }
    if (auto set = pass->set_render_image(
            swapchain.image(acquired.image_index)->get(), 0, {0.18F, 0.18F, 0.18F, 1.F});
        !set) {
      return holoviz::make_unexpected(set.error());
    }
    auto commands = device_.create_command_list({});
    if (!commands) {
      return holoviz::make_unexpected(commands.error());
    }
    const auto extent = swapchain.extent().value();
    if (!commands->begin() || !commands->begin_render(*pass, {0, 0, extent.width, extent.height}) ||
        !commands->end_render() || !commands->end()) {
      return holoviz::make_unexpected(
          holoviz::Error{.code = ErrorCode::kVulkanError, .message = "recording failed"});
    }
    auto submitted =
        device_.submit(holoviz::SubmitDesc{.command_lists = {&*commands, 1},
                                           .acquire_gates = {&acquired.acquire_gate, 1},
                                           .present_gates = {&acquired.present_gate, 1}});
    if (!submitted) {
      return submitted;
    }
    if (auto waited = device_.wait_on_submit(*submitted, std::chrono::seconds(5)); !waited) {
      return holoviz::make_unexpected(waited.error());
    }
    return submitted;
  }

  /// Acquire, clear and present one frame.
  void present_one_frame(holoviz::Swapchain& swapchain) {
    auto acquired = swapchain.acquire(std::chrono::seconds(5));
    ASSERT_TRUE(acquired.has_value()) << acquired.error().message;
    auto submitted = render_cleared(swapchain, *acquired);
    ASSERT_TRUE(submitted.has_value()) << submitted.error().message;
    ASSERT_TRUE(swapchain.present(std::move(*acquired), *submitted).has_value());
  }

  holoviz::Display display_;
  std::vector<holoviz::DisplayMode> modes_;
  SurfaceHandle surface_handle_;
  holoviz::Device device_;
  std::unique_ptr<PresentBackend> backend_;
  bool driver_managed_ = false;
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
  SKIP_IF_DRIVER_MANAGED_DISPLAY_TAKEN(created);
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
  SKIP_IF_DRIVER_MANAGED_DISPLAY_TAKEN(swapchain);
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

TEST_F(DisplaySurfaceHardwareTest, BackpressuredResizeLeavesOutstandingAcquirePresentable) {
  // On a driver-managed direct display (Tegra) resize() destroys the current swapchain
  // before creating its successor, so it waits for the present fences first. That wait is
  // the last one that can return kBackpressured, so it must come before the outstanding
  // acquired image is given back to the presentation engine: otherwise the caller keeps a
  // token for an image it no longer owns. Elsewhere resize() hands over through
  // oldSwapchain and this zero-timeout probe simply succeeds.
  auto swapchain = device_.create_swapchain(
      holoviz::SwapchainDesc{.surface = std::cref(*surface_from_handle(surface_handle_))});
  SKIP_IF_DRIVER_MANAGED_DISPLAY_TAKEN(swapchain);
  ASSERT_TRUE(swapchain.has_value()) << swapchain.error().message;
  present_one_frame(*swapchain);  // present A; its fence signals no earlier than scanout

  auto token = swapchain->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  const auto submitted = render_cleared(*swapchain, *token);
  ASSERT_TRUE(submitted.has_value()) << submitted.error().message;
  const auto generation = swapchain->generation().value();

  const auto extent = swapchain->extent().value();
  const auto probed = swapchain->resize(extent, std::chrono::nanoseconds::zero());
  if (!probed.has_value()) {
    ASSERT_EQ(probed.error().code, ErrorCode::kBackpressured) << probed.error().message;
    EXPECT_EQ(swapchain->generation().value(), generation);
    // The image is still acquired: a second acquire is rejected, and the token presents.
    const auto second = swapchain->acquire(std::chrono::nanoseconds::zero());
    ASSERT_FALSE(second.has_value()) << "the outstanding image was released";
    EXPECT_EQ(second.error().code, ErrorCode::kInvalidState);
    ASSERT_TRUE(swapchain->present(std::move(*token), *submitted).has_value());
    ASSERT_TRUE(swapchain->resize(extent, std::chrono::seconds(5)).has_value());
  }
  // Either way the swapchain recreated once and keeps presenting.
  EXPECT_GT(swapchain->generation().value(), generation);
  present_one_frame(*swapchain);
  ASSERT_TRUE(swapchain->destroy(std::chrono::seconds(5)).has_value());
}

TEST_F(DisplaySurfaceHardwareTest, ResizeFailingBeforeCreationLeavesAConsistentSwapchain) {
  // A recreation can fail before vkCreateSwapchainKHR (a surface lost between the drain and
  // the capability query). On a driver-managed direct display the old generation is
  // already destroyed at that point: it must be fully invalidated, with no image or
  // semaphore left for a later destroy() to destroy a second time. Elsewhere the failure
  // happens before anything changes. Both must recover with a later resize().
  auto swapchain = device_.create_swapchain(
      holoviz::SwapchainDesc{.surface = std::cref(*surface_from_handle(surface_handle_))});
  SKIP_IF_DRIVER_MANAGED_DISPLAY_TAKEN(swapchain);
  ASSERT_TRUE(swapchain.has_value()) << swapchain.error().message;
  present_one_frame(*swapchain);
  const auto generation = swapchain->generation().value();
  const auto extent = swapchain->extent().value();

  holoviz::vulkan::swapchain_fault_injection(*swapchain).fail_next_recreate_surface_query =
      holoviz::vulkan::kVkErrorSurfaceLost;
  const auto failed = swapchain->resize(extent, std::chrono::seconds(5));
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().code, ErrorCode::kVulkanError);
  if (swapchain->generation().value() != generation) {
    // The old generation was destroyed, and that is published: none of its images may
    // still be reachable, and acquire() asks for a resize.
    EXPECT_EQ(swapchain->image_count().value(), 0U);
    EXPECT_FALSE(swapchain->image(0).has_value());
    const auto acquired = swapchain->acquire(std::chrono::nanoseconds::zero());
    ASSERT_FALSE(acquired.has_value());
    EXPECT_EQ(acquired.error().code, ErrorCode::kOutOfDate);
  } else {
    // An unchanged generation promises the swapchain is untouched: it still presents.
    present_one_frame(*swapchain);
    EXPECT_EQ(swapchain->generation().value(), generation);
  }

  ASSERT_TRUE(swapchain->resize(extent, std::chrono::seconds(5)).has_value());
  EXPECT_GT(swapchain->generation().value(), generation);
  present_one_frame(*swapchain);
  ASSERT_TRUE(swapchain->destroy(std::chrono::seconds(5)).has_value());
}

TEST_F(DisplaySurfaceHardwareTest, CreatesAFixedSizeTwoImageSwapchainAndPresents) {
  auto created = backend_->create_swapchain(SwapchainDesc{
      .surface = surface_handle_,
      .preferred_extent = {0, 0},  // ignored: the display mode sizes the surface
      .preferred_mode = holoviz::PresentMode::kAuto,
      .image_count = std::nullopt,
  });
  SKIP_IF_DRIVER_MANAGED_DISPLAY_TAKEN(created);
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

TEST_F(DisplaySurfaceHardwareTest, PresentsAgainAfterDestroyingSwapchainAndSurface) {
  for (int iteration = 0; iteration < 2; ++iteration) {
    if (iteration != 0) {
      auto surface = make_display_surface_handle(display_);
      ASSERT_TRUE(surface.has_value()) << surface.error().message;
      surface_handle_ = std::move(*surface);
      refresh_driver_managed();
      auto backend = create_display_surface(device_);
      ASSERT_TRUE(backend.has_value()) << backend.error().message;
      backend_ = std::move(*backend);
    }
    const auto created = backend_->create_swapchain(SwapchainDesc{.surface = surface_handle_});
    if (iteration == 0) {
      // Only the first acquisition qualifies the environment. Once a frame was presented the
      // display is known to be usable, so a later failure is a release/reacquire regression.
      SKIP_IF_DRIVER_MANAGED_DISPLAY_TAKEN(created);
    }
    ASSERT_TRUE(created.has_value()) << created.error().message;
    const auto info = backend_->swapchain_info();
    ASSERT_TRUE(info.has_value()) << info.error().message;
    auto renderer = create_holoviz_render_backend(device_, {});
    ASSERT_TRUE(renderer.has_value()) << renderer.error().message;
    const RenderConfig config{.max_frames_in_flight = created->image_count};
    ASSERT_TRUE((*renderer)->allocate(LayerStack{}, config).has_value());
    ASSERT_TRUE((*renderer)->build(info->format).has_value());
    auto acquired = backend_->acquire(std::chrono::seconds{5});
    ASSERT_TRUE(acquired.has_value()) << acquired.error().message;
    auto commands = (*renderer)->begin_frame(*acquired, 0);
    ASSERT_TRUE(commands.has_value()) << commands.error().message;
    ASSERT_TRUE((*renderer)->record(*commands, RenderList{}, CompositeSpec{}).has_value());
    auto submitted = (*renderer)->submit(*commands, SubmitContext{0});
    ASSERT_TRUE(submitted.has_value()) << submitted.error().message;
    const auto presented = backend_->present(PresentArgs{.image_index = acquired->image_index,
                                                         .acquire_issuance = acquired->issuance,
                                                         .render_done = *submitted});
    ASSERT_TRUE(presented.has_value()) << presented.error().message;
    ASSERT_TRUE((*renderer)->destroy(std::chrono::seconds{5}).has_value());
    release_fixture_surface();
  }
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
