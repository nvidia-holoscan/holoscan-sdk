/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// RenderTargetDesc::load tests: kLoad renders across frames on a real swapchain (the
// sub-frame pattern — preserve prior contents, render only a fragment), and kLoad bound
// to an offscreen render target is rejected at begin_render (its between-pass layout is
// not the presentation home layout the pass loads from). Skipped without a display.

#include <chrono>
#include <utility>

#include <holoviz/holoviz.hpp>

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz {
namespace {

class RhiRenderPassLoadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto window = create_window({.title = "render pass load test", .size = {320, 240}});
    if (!window) {
      HOLOVIZ_ENV_SKIP() << "no window system available: " << window.error().message;
    }
    window_ = std::move(*window);

    auto surface = window_.create_surface();
    ASSERT_TRUE(surface.has_value()) << surface.error().message;
    surface_ = std::move(*surface);

    auto device = create_device_for_surface({}, surface_);
    if (!device) {
      HOLOVIZ_ENV_SKIP() << "no presentable adapter: " << device.error().message;
    }
    device_ = std::move(*device);

    auto swapchain = device_.create_swapchain(
        {.surface = surface_, .present_mode = PresentMode::kFIFO, .image_count = 3});
    ASSERT_TRUE(swapchain.has_value()) << swapchain.error().message;
    swapchain_ = std::move(*swapchain);
  }

  void TearDown() override {
    if (device_) {
      EXPECT_TRUE(device_.wait_idle().has_value());
    }
    if (swapchain_) {
      EXPECT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
    }
  }

  // Acquire, record one pass with `pass`, submit consuming both gates, present.
  void render_one_frame(RenderPass& pass) {
    auto token = swapchain_.acquire(std::chrono::seconds(5));
    ASSERT_TRUE(token.has_value()) << token.error().message;
    auto acquired = std::move(*token);

    auto image = swapchain_.image(acquired.image_index);
    ASSERT_TRUE(image.has_value()) << image.error().message;
    ASSERT_TRUE(pass.set_render_image(*image, 0, Color{0.5F, 0.25F, 0.125F, 1.0F}).has_value());

    auto command_list = device_.create_command_list({.type = CommandListType::kTransient});
    ASSERT_TRUE(command_list.has_value()) << command_list.error().message;
    const auto extent = swapchain_.extent().value();
    ASSERT_TRUE(command_list->begin().has_value());
    // Render only a fragment of the image: with kLoad the rest must stay untouched.
    const Rect2D fragment{0, 0, extent.width / 2, extent.height / 2};
    ASSERT_TRUE(command_list->begin_render(pass, fragment).has_value());
    ASSERT_TRUE(command_list->end_render().has_value());
    ASSERT_TRUE(command_list->end().has_value());

    holoviz::CommandList* command_list_ptr = &*command_list;
    const SubmitDesc submit_desc{
        .command_lists = {command_list_ptr, 1},
        .acquire_gates = {&acquired.acquire_gate, 1},
        .present_gates = {&acquired.present_gate, 1},
    };
    auto submitted = device_.submit(submit_desc);
    ASSERT_TRUE(submitted.has_value()) << submitted.error().message;
    ASSERT_TRUE(swapchain_.present(std::move(acquired), *submitted).has_value());
    ASSERT_TRUE(device_.wait_idle().has_value());
  }

  Window window_;
  Surface surface_;
  Device device_;
  Swapchain swapchain_;
};

TEST_F(RhiRenderPassLoadTest, LoadPreservingPassRendersOnSwapchain) {
  // Frame 1 establishes contents with a clearing pass; frame 2 loads them (kLoad) and
  // renders a fragment. Correctness of the preserved pixels needs readback (not exposed
  // yet); this pins that the kLoad flavor records, submits, and presents cleanly —
  // under HOLOVIZ_VALIDATION=1 a wrong baked layout would fail loudly.
  auto clear_pass = device_.create_render_pass(
      {.render_targets = {{.load = LoadOp::kClear, .format = swapchain_.format().value()}}});
  ASSERT_TRUE(clear_pass.has_value()) << clear_pass.error().message;
  auto load_pass = device_.create_render_pass(
      {.render_targets = {{.load = LoadOp::kLoad, .format = swapchain_.format().value()}}});
  ASSERT_TRUE(load_pass.has_value()) << load_pass.error().message;

  render_one_frame(*clear_pass);
  render_one_frame(*load_pass);
  render_one_frame(*load_pass);
}

TEST_F(RhiRenderPassLoadTest, LoadRejectsOffscreenRenderTarget) {
  auto load_pass = device_.create_render_pass(
      {.render_targets = {{.load = LoadOp::kLoad, .format = Format::kR8G8B8A8Unorm}}});
  ASSERT_TRUE(load_pass.has_value()) << load_pass.error().message;

  auto offscreen = device_.create_image({.extent = {64, 64},
                                         .format = Format::kR8G8B8A8Unorm,
                                         .usage = ImageUsage::kRenderTarget,
                                         .debug_name = "kLoad offscreen target"});
  ASSERT_TRUE(offscreen.has_value()) << offscreen.error().message;
  ASSERT_TRUE(load_pass->set_render_image(*offscreen, 0, Color{0, 0, 0, 1}).has_value());

  auto command_list = device_.create_command_list({.type = CommandListType::kTransient});
  ASSERT_TRUE(command_list.has_value()) << command_list.error().message;
  ASSERT_TRUE(command_list->begin().has_value());
  const auto begun = command_list->begin_render(*load_pass, Rect2D{0, 0, 64, 64});
  ASSERT_FALSE(begun.has_value());
  EXPECT_EQ(begun.error().code, ErrorCode::kInvalidArgument);
  ASSERT_TRUE(command_list->end().has_value());
}

}  // namespace
}  // namespace holoviz
