// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>

#include <holoviz/holoviz.hpp>

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz {
namespace {

// destroy() on a default-constructed (empty) handle is a no-op success for
// every handle type.
TEST(Destroy, EmptyHandlesDestroySuccessfully) {
  EXPECT_TRUE(Device{}.destroy().has_value());
  EXPECT_TRUE(Buffer{}.destroy().has_value());
  EXPECT_TRUE(Image{}.destroy().has_value());
  EXPECT_TRUE(Surface{}.destroy().has_value());
  EXPECT_TRUE(Swapchain{}.destroy().has_value());
  EXPECT_TRUE(Window{}.destroy().has_value());
  EXPECT_TRUE(CommandList{}.destroy().has_value());
  EXPECT_TRUE(Pipeline{}.destroy().has_value());
  EXPECT_TRUE(Descriptor{}.destroy().has_value());
  EXPECT_TRUE(RenderPass{}.destroy().has_value());
  EXPECT_TRUE(QueryPool{}.destroy().has_value());
  EXPECT_TRUE(ExternalSemaphore{}.destroy().has_value());
}

// Concurrent first calls must serialize on the once-guarded backend init (instance +
// cuInit) and all see the same result. Deterministic either way; the data race itself is
// caught when the suite runs under TSAN in dev (not CI-wired).
TEST(Init, ConcurrentEnumerateAdaptersIsThreadSafe) {
  constexpr int kThreads = 4;
  std::array<expected<std::vector<Adapter>>, kThreads> results;
  {
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
      threads.emplace_back([&results, i] { results[i] = enumerate_adapters(); });
    }
    for (std::thread& t : threads) {
      t.join();
    }
  }
  for (int i = 1; i < kThreads; ++i) {
    ASSERT_EQ(results[i].has_value(), results[0].has_value());
    if (results[0].has_value()) {
      EXPECT_EQ(results[i]->size(), results[0]->size());
    } else {
      EXPECT_EQ(results[i].error().code, results[0].error().code);
    }
  }
}

// GPU-backed coverage; skipped when no adapter/device is available.
class DestroyDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    expected<std::vector<Adapter>> adapters = enumerate_adapters();
    if (!adapters || adapters->empty()) {
      HOLOVIZ_ENV_SKIP() << "No HoloViz adapter available";
    }
    expected<Device> device = (*adapters)[0].create_device({});
    if (!device) {
      HOLOVIZ_ENV_SKIP() << "Device creation failed: " << device.error().message;
    }
    device_ = std::move(*device);
  }

  void TearDown() override {
    if (device_) {
      EXPECT_TRUE(device_.destroy().has_value());
    }
  }

  Device device_;
};

TEST_F(DestroyDeviceTest, BufferDestroyIsIdempotent) {
  expected<Buffer> buffer =
      device_.create_buffer({.size_bytes = 1024, .share = ShareMode::kPrivate});
  ASSERT_TRUE(buffer.has_value()) << buffer.error().message;

  EXPECT_TRUE(buffer->destroy().has_value());
  EXPECT_FALSE(static_cast<bool>(*buffer));
  EXPECT_TRUE(buffer->destroy().has_value());  // second destroy: no-op success
}

TEST_F(DestroyDeviceTest, ExportedBufferDestroyReleasesCudaResources) {
  expected<Buffer> buffer =
      device_.create_buffer({.size_bytes = 1024, .share = ShareMode::kExported});
  ASSERT_TRUE(buffer.has_value()) << buffer.error().message;

  EXPECT_TRUE(buffer->destroy().has_value());
  EXPECT_FALSE(static_cast<bool>(*buffer));
}

TEST_F(DestroyDeviceTest, CommandListDestroyAfterSubmitCompletes) {
  expected<CommandList> command_list =
      device_.create_command_list({.type = CommandListType::kTransient});
  ASSERT_TRUE(command_list.has_value()) << command_list.error().message;

  ASSERT_TRUE(command_list->begin().has_value());
  ASSERT_TRUE(command_list->end().has_value());

  SubmitDesc submit_desc{.command_lists = {&*command_list, 1}};
  const expected<SubmitSyncToken> token = device_.submit(submit_desc);
  ASSERT_TRUE(token.has_value()) << token.error().message;

  // With a zero timeout the destroy either succeeds (submit already retired)
  // or reports backpressure and leaves the handle usable for a retry.
  const expected<void> immediate = command_list->destroy();
  if (!immediate) {
    EXPECT_EQ(immediate.error().code, ErrorCode::kBackpressured);
    EXPECT_TRUE(static_cast<bool>(*command_list));
    ASSERT_TRUE(device_.wait_idle().has_value());
    EXPECT_TRUE(command_list->destroy(std::chrono::seconds(1)).has_value());
  }
  EXPECT_FALSE(static_cast<bool>(*command_list));
}

TEST_F(DestroyDeviceTest, RecreatedImageViewGetsFreshFramebuffer) {
  // Regression test for the framebuffer cache's recycled-handle aliasing: render into
  // image A, destroy it (which must evict the cached framebuffer over its view), then
  // create identically-described image B — drivers frequently recycle the freed handle
  // values — and render into it. Pre-fix, a recycled view handle matched the stale cache
  // entry and returned a framebuffer built over the destroyed view (validation errors
  // under HOLOVIZ_VALIDATION=1; leaked key arrays regardless).
  constexpr uint32_t kSize = 64;
  const ImageDesc image_desc{.extent = {kSize, kSize},
                             .format = Format::kR8G8B8A8Unorm,
                             .usage = ImageUsage::kRenderTarget,
                             .share = ShareMode::kPrivate,
                             .debug_name = "framebuffer recycle test"};

  expected<RenderPass> render_pass =
      device_.create_render_pass({.render_targets = {{.format = Format::kR8G8B8A8Unorm}}});
  ASSERT_TRUE(render_pass.has_value()) << render_pass.error().message;
  expected<CommandList> cmd = device_.create_command_list({.type = CommandListType::kTransient});
  ASSERT_TRUE(cmd.has_value()) << cmd.error().message;

  const auto render_once = [&](Image& target) {
    ASSERT_TRUE(
        render_pass->set_render_image(target, 0, Color{0.0F, 0.0F, 0.0F, 1.0F}).has_value());
    ASSERT_TRUE(cmd->reset(std::chrono::seconds(5)).has_value());
    ASSERT_TRUE(cmd->begin().has_value());
    ASSERT_TRUE(cmd->begin_render(*render_pass, Rect2D{0, 0, kSize, kSize}).has_value());
    ASSERT_TRUE(cmd->end_render().has_value());
    ASSERT_TRUE(cmd->end().has_value());
    SubmitDesc submit_desc{.command_lists = {&*cmd, 1}};
    const expected<SubmitSyncToken> token = device_.submit(submit_desc);
    ASSERT_TRUE(token.has_value()) << token.error().message;
    ASSERT_TRUE(device_.wait_on_submit(*token, std::chrono::seconds(5)).has_value());
  };

  expected<Image> image_a = device_.create_image(image_desc);
  ASSERT_TRUE(image_a.has_value()) << image_a.error().message;
  ASSERT_NO_FATAL_FAILURE(render_once(*image_a));
  ASSERT_TRUE(image_a->destroy(std::chrono::seconds(5)).has_value());

  expected<Image> image_b = device_.create_image(image_desc);
  ASSERT_TRUE(image_b.has_value()) << image_b.error().message;
  ASSERT_NO_FATAL_FAILURE(render_once(*image_b));
  ASSERT_TRUE(image_b->destroy(std::chrono::seconds(5)).has_value());

  ASSERT_TRUE(cmd->destroy(std::chrono::seconds(5)).has_value());
  ASSERT_TRUE(render_pass->destroy(std::chrono::seconds(5)).has_value());
}

}  // namespace
}  // namespace holoviz
