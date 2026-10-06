// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Headless smoke tests for the push-constant monospace text renderer. GPU
// tests create a real device (no window, no CUDA interop beyond device
// bring-up) and skip cleanly on machines without a usable adapter.

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

#include <holoviz/holoviz.hpp>
#include <holoviz/text.hpp>

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz::text {
namespace {

/// Create a device on the first adapter, or an empty handle if this machine
/// cannot (no Vulkan adapter, no matching CUDA device, ...).
Device try_create_device() {
  auto adapters = enumerate_adapters();
  if (!adapters || adapters->empty()) {
    return {};
  }
  auto device = adapters->front().create_device({});
  if (!device) {
    return {};
  }
  return std::move(*device);
}

#define SKIP_WITHOUT_DEVICE(device)                                \
  if (!(device)) {                                                 \
    HOLOVIZ_ENV_SKIP() << "no usable GPU/adapter on this machine"; \
  }

TEST(MonospaceText, FactoryRejectsInvalidAtlasHeight) {
  Device device = try_create_device();
  SKIP_WITHOUT_DEVICE(device);

  const auto renderer = create_monospace_text_renderer(device, {.atlas_glyph_height_px = 0.0F});
  ASSERT_FALSE(renderer);
  EXPECT_EQ(renderer.error().code, ErrorCode::kInvalidArgument);

  EXPECT_TRUE(device.wait_idle().has_value());
  EXPECT_TRUE(device.destroy().has_value());
}

TEST(MonospaceText, DeviceReportsPushConstantLimit) {
  Device device = try_create_device();
  SKIP_WITHOUT_DEVICE(device);

  const auto capabilities = device.capabilities();
  ASSERT_TRUE(capabilities.has_value());
  // The Vulkan minimum; the monospace renderer itself needs 256 and checks at creation.
  EXPECT_GE(capabilities->max_push_data_bytes, 128u);

  EXPECT_TRUE(device.destroy().has_value());
}

TEST(MonospaceText, FrameStateMachine) {
  Device device = try_create_device();
  SKIP_WITHOUT_DEVICE(device);

  auto renderer_result = create_monospace_text_renderer(device, {});
  ASSERT_TRUE(renderer_result.has_value()) << renderer_result.error().message;
  TextRenderer renderer = std::move(*renderer_result);

  const Extent2D extent{256, 128};

  // add_text/end_frame outside a frame fail.
  const auto orphan_add = renderer.add_text({}, "x");
  ASSERT_FALSE(orphan_add);
  EXPECT_EQ(orphan_add.error().code, ErrorCode::kInvalidState);
  ASSERT_FALSE(renderer.end_frame());

  EXPECT_TRUE(renderer.begin_frame(extent, 0, {}).has_value());
  EXPECT_TRUE(renderer.add_text({.size_px = 16.0F}, "").has_value());  // empty: no-op success
  ASSERT_FALSE(renderer.begin_frame(extent, 0, {}));                   // double begin
  EXPECT_TRUE(renderer.end_frame().has_value());

  EXPECT_TRUE(device.wait_idle().has_value());
  EXPECT_TRUE(renderer.destroy().has_value());
  EXPECT_TRUE(device.destroy().has_value());
}

TEST(MonospaceText, CancelFrameIsIdempotentAndUnbricks) {
  Device device = try_create_device();
  SKIP_WITHOUT_DEVICE(device);

  auto renderer_result = create_monospace_text_renderer(device, {});
  ASSERT_TRUE(renderer_result.has_value()) << renderer_result.error().message;
  TextRenderer renderer = std::move(*renderer_result);
  const Extent2D extent{256, 128};

  // Open a frame and hit the mid-frame rejection that used to leave it stuck open.
  ASSERT_TRUE(renderer.begin_frame(extent, 0, {}).has_value());
  const auto bad = renderer.add_text({.size_px = 0.0F}, "x");
  ASSERT_FALSE(bad);
  EXPECT_EQ(bad.error().code, ErrorCode::kInvalidArgument);

  // cancel_frame closes it (idempotently — a second cancel is a no-op success)...
  EXPECT_TRUE(renderer.cancel_frame().has_value());
  EXPECT_TRUE(renderer.cancel_frame().has_value());

  // ...and the renderer runs a full valid frame afterwards.
  ASSERT_TRUE(renderer.begin_frame(extent, 1, {}).has_value());
  ASSERT_TRUE(renderer.add_text({.size_px = 16.0F}, "ok").has_value());
  EXPECT_TRUE(renderer.end_frame().has_value());

  EXPECT_TRUE(device.wait_idle().has_value());
  EXPECT_TRUE(renderer.destroy().has_value());
  EXPECT_TRUE(device.destroy().has_value());
}

// End-to-end helper: render "Hi" into an offscreen target with the given
// renderer desc and check pixels changed. Passing end_frame()'s no-op token
// through wait_external_sync_tokens doubles as the regression test for
// submit() skipping default-constructed tokens.
void expect_renders_visible_pixels(Device& device, const MonospaceTextRendererDesc& desc) {
  const Extent2D extent{256, 128};
  auto target = device.create_image({
      .extent = extent,
      .format = Format::kR8G8B8A8Unorm,
      .usage = ImageUsage::kRenderTarget,
      .debug_name = "monospace test target",
  });
  ASSERT_TRUE(target.has_value()) << target.error().message;

  auto render_pass =
      device.create_render_pass({.render_targets = {{.format = Format::kR8G8B8A8Unorm}}});
  ASSERT_TRUE(render_pass.has_value()) << render_pass.error().message;
  ASSERT_TRUE(render_pass->set_render_image(*target, 0, {0.0F, 0.0F, 0.0F, 1.0F}).has_value());

  auto renderer_result = create_monospace_text_renderer(device, desc);
  ASSERT_TRUE(renderer_result.has_value()) << renderer_result.error().message;
  TextRenderer renderer = std::move(*renderer_result);

  ASSERT_TRUE(renderer.begin_frame(extent, 0, {}).has_value());
  ASSERT_TRUE(
      renderer.add_text({.position = {20.0F, 20.0F}, .size_px = 64.0F, .color = 0xFFFFFFFFu}, "Hi")
          .has_value());
  const auto token = renderer.end_frame();
  ASSERT_TRUE(token.has_value());
  EXPECT_EQ(token->impl, nullptr);  // the no-op token

  auto command_list = device.create_command_list({.type = CommandListType::kTransient});
  ASSERT_TRUE(command_list.has_value()) << command_list.error().message;
  ASSERT_TRUE(command_list->begin().has_value());
  ASSERT_TRUE(
      command_list->begin_render(*render_pass, {0, 0, extent.width, extent.height}).has_value());
  ASSERT_TRUE(renderer.record(*command_list, extent).has_value());
  ASSERT_TRUE(command_list->end_render().has_value());
  ASSERT_TRUE(command_list->end().has_value());

  const SubmitDesc submit_desc{
      .wait_external_sync_tokens = {&token.value(), 1},
      .command_lists = {&command_list.value(), 1},
  };
  const auto submit_token = device.submit(submit_desc);
  ASSERT_TRUE(submit_token.has_value()) << submit_token.error().message;
  ASSERT_TRUE(device.wait_on_submit(*submit_token, std::chrono::seconds(5)).has_value());

  const size_t row_pitch = static_cast<size_t>(extent.width) * 4;
  std::vector<std::byte> pixels(row_pitch * extent.height);
  ASSERT_TRUE(device.download_from_image(*target, 0, pixels, row_pitch).has_value());

  bool any_text_pixel = false;
  for (size_t i = 0; i < pixels.size(); i += 4) {
    // Any non-black RGB byte means glyphs landed on the cleared target.
    if (std::to_integer<uint8_t>(pixels[i]) != 0 || std::to_integer<uint8_t>(pixels[i + 1]) != 0 ||
        std::to_integer<uint8_t>(pixels[i + 2]) != 0) {
      any_text_pixel = true;
      break;
    }
  }
  EXPECT_TRUE(any_text_pixel) << "rendered image is entirely clear color";

  EXPECT_TRUE(device.wait_idle().has_value());
  EXPECT_TRUE(renderer.destroy().has_value());
  EXPECT_TRUE(command_list->destroy().has_value());
  EXPECT_TRUE(render_pass->destroy().has_value());
  EXPECT_TRUE(target->destroy().has_value());
}

TEST(MonospaceText, RendersVisiblePixels) {
  Device device = try_create_device();
  SKIP_WITHOUT_DEVICE(device);

  expect_renders_visible_pixels(device, {.fill_mode = MonospaceFillMode::kTriangles});
  EXPECT_TRUE(device.destroy().has_value());
}

TEST(MonospaceText, FillRectRendersVisiblePixels) {
  Device device = try_create_device();
  SKIP_WITHOUT_DEVICE(device);

  const auto capabilities = device.capabilities();
  ASSERT_TRUE(capabilities.has_value());
  if (!capabilities->fill_rectangle) {
    // Selecting the mode anyway must fail cleanly on devices without the extension.
    const auto renderer =
        create_monospace_text_renderer(device, {.fill_mode = MonospaceFillMode::kFillRectangle});
    ASSERT_FALSE(renderer);
    EXPECT_TRUE(device.destroy().has_value());
    GTEST_SKIP() << "VK_NV_fill_rectangle unavailable on this device";
  }

  expect_renders_visible_pixels(device, {.fill_mode = MonospaceFillMode::kFillRectangle});
  EXPECT_TRUE(device.destroy().has_value());
}

}  // namespace
}  // namespace holoviz::text
