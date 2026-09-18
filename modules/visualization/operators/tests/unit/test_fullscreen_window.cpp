// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Fullscreen window runtime behavior: monitor-sized creation and swapchain recovery on an
// out-of-date surface. These tests take over a monitor (and may switch its video mode), so
// they are opt-in: set HOLOVIZ_FULLSCREEN_TESTS=1 to run them; without it every test skips.
// The no-monitor rejection path (GlfwWindow::initialize) cannot be reproduced on cue with a
// real window system and is covered by the pure policy tests in test_window_util.cpp plus
// review. Refresh-rate preservation with a non-maximal current rate is a manual check
// (xrandr before/during/after gray_scale_example --fullscreen); this file deliberately uses
// no GLFW API, because in shared builds libholoviz's statically absorbed GLFW is a
// different instance from one the test could link.

#include <chrono>
#include <cstdlib>
#include <memory>
#include <utility>

#include <holoviz/holoviz.hpp>

// Operator-private backend headers and the module-private swapchain fault-injection seam; both
// include roots come from the backend object library, like the present fault-injection target.
#include "holoviz_op/backends/backend_tokens.hpp"
#include "holoviz_op/backends/present_backend.hpp"
#include "holoviz_op/backends/wsi_present_backend.hpp"
#include "vulkan/vk_swapchain_fault_injection.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoscan::visualization::backends {
namespace {

bool fullscreen_tests_enabled() {
  const char* value = std::getenv("HOLOVIZ_FULLSCREEN_TESTS");
  return value != nullptr && value[0] == '1';
}

class FullscreenWindowTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!fullscreen_tests_enabled()) {
      GTEST_SKIP() << "fullscreen tests take over a monitor; set HOLOVIZ_FULLSCREEN_TESTS=1";
    }
    // Size {0, 0} and refresh 0: the monitor's current mode, on the primary monitor.
    auto window = holoviz::create_window({.title = "fullscreen window test",
                                          .size = {0, 0},
                                          .fullscreen = true,
                                          .close_on_escape = true});
    if (!window) {
      HOLOVIZ_ENV_SKIP() << "no window system available: " << window.error().message;
    }
    window_ = std::move(*window);
  }

  void TearDown() override {
    if (backend_ != nullptr) {
      EXPECT_TRUE(backend_->destroy(std::chrono::seconds(5)).has_value());
      backend_.reset();
    }
    surface_handle_.impl.reset();
    if (window_) {
      EXPECT_TRUE(window_.destroy().has_value());
    }
  }

  // Surface + device + WSI present backend on the fullscreen window (skips without a
  // presentable adapter).
  void create_present_backend() {
    auto surface_handle = make_surface_handle(window_);
    ASSERT_TRUE(surface_handle.has_value()) << surface_handle.error().message;
    surface_handle_ = std::move(*surface_handle);

    auto device = holoviz::create_device_for_surface({}, *surface_from_handle(surface_handle_));
    if (!device) {
      HOLOVIZ_ENV_SKIP() << "no presentable adapter: " << device.error().message;
    }
    device_ = std::move(*device);

    auto backend = create_wsi_present_backend(device_);
    ASSERT_TRUE(backend.has_value()) << backend.error().message;
    backend_ = std::move(*backend);

    ASSERT_TRUE(backend_
                    ->create_swapchain(SwapchainDesc{.surface = surface_handle_,
                                                     .preferred_mode = holoviz::PresentMode::kAuto,
                                                     .image_count = 2})
                    .has_value());
  }

  holoviz::vulkan::SwapchainFaultInjection& fault_injection() {
    return holoviz::vulkan::swapchain_fault_injection(
        static_cast<WsiPresentBackend&>(*backend_).swapchain_for_test());
  }

  holoviz::Window window_;
  SurfaceHandle surface_handle_;
  holoviz::Device device_;
  std::unique_ptr<PresentBackend> backend_;
};

// A {0, 0} fullscreen window is sized by the window system to the monitor's current mode:
// the framebuffer is non-empty and the surface reports that same fixed extent (which is
// also what the automatic swapchain recreation sizes from).
TEST_F(FullscreenWindowTest, TakesMonitorModeAndSurfaceReportsFixedExtent) {
  const auto extent = window_.extent();
  ASSERT_TRUE(extent.has_value()) << extent.error().message;
  EXPECT_GT(extent->width, 0U);
  EXPECT_GT(extent->height, 0U);

  create_present_backend();
  if (IsSkipped()) {
    return;
  }
  const auto caps = device_.surface_capabilities(*surface_from_handle(surface_handle_));
  ASSERT_TRUE(caps.has_value()) << caps.error().message;
  EXPECT_EQ(caps->current_extent.width, extent->width);
  EXPECT_EQ(caps->current_extent.height, extent->height);
}

// The P2-3 regression: a fullscreen surface is not user-resizable, but it must keep the
// swapchain's automatic recovery. An out-of-date present is reported as a consumed frame
// (kFrameDropped / kImageCountChanged) — never the caller-owned kOutOfDate — and the frame
// loop continues on the recreated swapchain.
TEST_F(FullscreenWindowTest, SwapchainRecoversFromOutOfDatePresent) {
  create_present_backend();
  if (IsSkipped()) {
    return;
  }

  auto token = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;

  fault_injection().override_next_present_result = holoviz::vulkan::kVkErrorOutOfDate;
  const auto dropped = backend_->present(PresentArgs{
      .image_index = token->image_index, .acquire_issuance = token->issuance, .present_id = 1});
  ASSERT_FALSE(dropped.has_value());
  EXPECT_TRUE(dropped.error().code == ErrorCode::kFrameDropped ||
              dropped.error().code == ErrorCode::kImageCountChanged)
      << dropped.error().message;

  // The loop continues without any explicit resize() (a kImageCountChanged outcome may need
  // one reconciling acquire).
  for (int i = 0; i < 3; ++i) {
    auto next = backend_->acquire(std::chrono::seconds(5));
    if (!next.has_value()) {
      EXPECT_EQ(next.error().code, ErrorCode::kImageCountChanged) << next.error().message;
      continue;
    }
    ASSERT_TRUE(backend_
                    ->present(PresentArgs{.image_index = next->image_index,
                                          .acquire_issuance = next->issuance,
                                          .present_id = 2})
                    .has_value());
    return;
  }
  FAIL() << "acquire never recovered after the injected out-of-date present";
}

}  // namespace
}  // namespace holoscan::visualization::backends
