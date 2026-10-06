// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Text layer tests for HolovizRenderBackend. The load-bearing test is differential: the same
// text rendered through the backend seam (headless) and through a directly-driven monospace
// TextRenderer on the same device must agree almost exactly — the backend adds coordinate
// mapping and frame plumbing, not rasterization. Analytic ink-box and glyph-budget tests
// cover what the differential cannot.

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/holoviz.hpp>
#include <holoviz/text.hpp>
#include "holoviz_op/backends/frame_data.hpp"
#include "holoviz_op/backends/holoviz_render_backend.hpp"
#include "holoviz_op/backends/render_backend.hpp"

#include "image_compare_test_util.hpp"

// Module-private headers (src/ is on this target's include path).
#include "holoviz_op/backends/backend_tokens.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoscan::visualization::backends {

namespace test_util = holoviz::test_util;
namespace {

using test_util::compare_images;
using test_util::ImageDiff;
using test_util::kInkThreshold;
using test_util::log_diff;
using test_util::maybe_dump;
using test_util::try_create_device;

constexpr uint32_t kWidth = 256;
constexpr uint32_t kHeight = 128;

class HolovizBackendTextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = try_create_device();
    if (!device_) {
      HOLOVIZ_ENV_SKIP() << "no usable rendering device/adapter.";
    }
  }

  void TearDown() override {
    backend_.reset();
    if (device_) {
      ASSERT_TRUE(device_.destroy(std::chrono::seconds(5)).has_value());
    }
  }

  /// Bring a backend with one declared text layer up to the built state. Positions are in
  /// framebuffer pixels unless \p normalized_coordinates selects fractions of the framebuffer.
  void make_backend(uint32_t max_glyphs = 256, bool normalized_coordinates = false) {
    auto backend = create_holoviz_render_backend(device_, {});
    ASSERT_TRUE(backend.has_value());
    backend_ = std::move(*backend);

    LayerStack bundle;
    bundle.add_text_layer(
        "labels",
        TextLayerDesc{.max_glyphs = max_glyphs, .normalized_coordinates = normalized_coordinates});

    RenderConfig config;
    config.headless = true;
    config.headless_width = kWidth;
    config.headless_height = kHeight;
    config.max_frames_in_flight = 2;
    ASSERT_TRUE(backend_->allocate(bundle, config).has_value());
    ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());
  }

  /// One full frame through the seam: begin -> record(runs) -> submit -> wait -> readback.
  void render_backend_frame(std::span<const TextRun> runs, std::vector<uint8_t>& out_pixels,
                            uint32_t frame_index = 0) {
    auto cmds = backend_->begin_frame(AcquireToken{}, frame_index);
    ASSERT_TRUE(cmds.has_value()) << cmds.error().message;

    const TextFrameData text{runs};
    RenderList list;
    list.layers.push_back(LayerFrame{0, &text});
    ASSERT_TRUE(backend_->record(*cmds, list, CompositeSpec{}).has_value());

    auto token = backend_->submit(*cmds, SubmitContext{frame_index});
    ASSERT_TRUE(token.has_value()) << token.error().message;
    const SubmitPayload* payload = submit_payload(*token, device_identity(device_)).payload;
    ASSERT_NE(payload, nullptr);
    ASSERT_TRUE(device_.wait_on_submit(payload->token, std::chrono::seconds(5)).has_value());

    auto* concrete = static_cast<HolovizRenderBackend*>(backend_.get());
    const auto target = concrete->headless_target(frame_index);
    ASSERT_TRUE(target.has_value());
    out_pixels.resize(static_cast<size_t>(kWidth) * kHeight * 4);
    ASSERT_TRUE(device_
                    .download_from_image(target->get(),
                                         0,
                                         std::as_writable_bytes(std::span(out_pixels)),
                                         static_cast<size_t>(kWidth) * 4)
                    .has_value());
  }

  void expect_backend_matches_direct(std::span<const TextRun> runs, const char* dump_name);

  holoviz::Device device_;
  std::unique_ptr<RenderBackend> backend_;
};

// Render the same two runs through the backend seam (in the layer's declared unit) and straight
// through a monospace TextRenderer at the equivalent pixel positions; the images must match.
void HolovizBackendTextTest::expect_backend_matches_direct(std::span<const TextRun> runs,
                                                           const char* dump_name) {
  std::vector<uint8_t> backend_pixels;
  ASSERT_NO_FATAL_FAILURE(render_backend_frame(runs, backend_pixels));

  // The same content straight through a monospace TextRenderer (same device, same built-in
  // font, same pixel positions). 13 px draws match the built-in font's bake size (no bitmap
  // scaling on either side).
  auto direct_result = text::create_monospace_text_renderer(device_, {});
  ASSERT_TRUE(direct_result.has_value()) << direct_result.error().message;
  text::TextRenderer direct = std::move(*direct_result);

  const test_util::Draw draws[] = {
      {{.position = {32.0F, 32.0F}, .size_px = 13.0F, .color = 0xFFFFFFFFU}, "Hello seam 42!"},
      {{.position = {32.0F, 70.4F}, .size_px = 13.0F, .color = 0xFF40C0FFU}, "layer {text}"},
  };
  std::vector<uint8_t> direct_pixels;
  ASSERT_NO_FATAL_FAILURE(test_util::render_with(
      device_, direct, holoviz::Extent2D{kWidth, kHeight}, draws, direct_pixels));

  const ImageDiff diff = compare_images(backend_pixels, direct_pixels, /*channel_tol=*/2);
  log_diff("backend(a) vs direct(b), channel_tol=2", diff);
  EXPECT_GT(diff.ink_a, 200u) << "backend image is (nearly) blank";
  EXPECT_GT(diff.ink_b, 200u) << "direct image is (nearly) blank";
  EXPECT_LT(diff.differing_fraction, 0.001);

  maybe_dump(
      dump_name, backend_pixels, direct_pixels, holoviz::Extent2D{kWidth, kHeight}, HasFailure());

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(direct.destroy().has_value());
}

// Default unit: TextRun positions are framebuffer pixels and reach the renderer unscaled.
TEST_F(HolovizBackendTextTest, DifferentialBackendVsDirectRendererInPixels) {
  make_backend();
  const TextRun runs[] = {
      {32.0F, 32.0F, 13.0F, 0xFFFFFFFFU, "Hello seam 42!"},
      {32.0F, 70.4F, 13.0F, 0xFF40C0FFU, "layer {text}"},
  };
  expect_backend_matches_direct(runs, "backend_vs_direct_pixels");
}

// Opt-in unit: normalized positions are scaled by the framebuffer extent on the CPU. Chosen to
// land on the same pixels as the pixel case (0.125 * 256 = 32, 0.25 * 128 = 32, 0.55 * 128 = 70.4).
TEST_F(HolovizBackendTextTest, DifferentialBackendVsDirectRendererNormalized) {
  make_backend(/*max_glyphs=*/256, /*normalized_coordinates=*/true);
  const TextRun runs[] = {
      {0.125F, 0.25F, 13.0F, 0xFFFFFFFFU, "Hello seam 42!"},
      {0.125F, 0.55F, 13.0F, 0xFF40C0FFU, "layer {text}"},
  };
  expect_backend_matches_direct(runs, "backend_vs_direct_normalized");
}

TEST_F(HolovizBackendTextTest, InkStaysInsideTheGlyphBox) {
  make_backend();

  // One 'H' at pixel (32, 32), drawn at 32 px.
  const TextRun runs[] = {{32.0F, 32.0F, 32.0F, 0xFFFFFFFFU, "H"}};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render_backend_frame(runs, pixels));

  const uint32_t box_min = 30;
  const uint32_t box_max = 32 + 32 + 2;
  size_t ink_inside = 0;
  size_t ink_outside = 0;
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      const size_t i = (static_cast<size_t>(y) * kWidth + x) * 4;
      const bool ink = pixels[i] > kInkThreshold || pixels[i + 1] > kInkThreshold ||
                       pixels[i + 2] > kInkThreshold;
      if (!ink) {
        continue;
      }
      const bool inside = x >= box_min && x <= box_max && y >= box_min && y <= box_max;
      (inside ? ink_inside : ink_outside) += 1;
    }
  }
  EXPECT_GE(ink_inside, 40u) << "glyph 'H' left too little ink in its cell";
  EXPECT_EQ(ink_outside, 0u) << "glyph 'H' leaked ink outside its cell";

  maybe_dump("backend_ink_box", pixels, pixels, holoviz::Extent2D{kWidth, kHeight}, HasFailure());
}

TEST_F(HolovizBackendTextTest, GlyphBudgetOverflowIsAMisdeclaration) {
  make_backend(/*max_glyphs=*/8);

  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());

  const TextRun runs[] = {{10.0F, 10.0F, 13.0F, 0xFFFFFFFFU, "this run is far too long"}};
  const TextFrameData text{runs};
  RenderList list;
  list.layers.push_back(LayerFrame{0, &text});
  const auto recorded = backend_->record(*cmds, list, CompositeSpec{});
  ASSERT_FALSE(recorded.has_value());
  EXPECT_EQ(recorded.error().code, ErrorCode::kInvalidArgument);

  // The frame is still submittable after the rejected layer (nothing was recorded for it).
  ASSERT_TRUE(backend_->submit(*cmds, SubmitContext{0}).has_value());
}

TEST_F(HolovizBackendTextTest, BadSizePxFailsWithoutOpeningTheFrame) {
  make_backend();

  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());

  // Invalid runs must be rejected BEFORE the layer opens its renderer frame; a rejection
  // after opening used to leave the renderer permanently refusing every later frame.
  const TextRun zero_size[] = {{10.0F, 10.0F, 0.0F, 0xFFFFFFFFU, "zero"}};
  const TextFrameData zero_frame{zero_size};
  RenderList zero_list;
  zero_list.layers.push_back(LayerFrame{0, &zero_frame});
  const auto zero_recorded = backend_->record(*cmds, zero_list, CompositeSpec{});
  ASSERT_FALSE(zero_recorded.has_value());
  EXPECT_EQ(zero_recorded.error().code, ErrorCode::kInvalidArgument);

  const TextRun nan_size[] = {
      {10.0F, 10.0F, std::numeric_limits<float>::quiet_NaN(), 0xFFFFFFFFU, "nan"}};
  const TextFrameData nan_frame{nan_size};
  RenderList nan_list;
  nan_list.layers.push_back(LayerFrame{0, &nan_frame});
  const auto nan_recorded = backend_->record(*cmds, nan_list, CompositeSpec{});
  ASSERT_FALSE(nan_recorded.has_value());
  EXPECT_EQ(nan_recorded.error().code, ErrorCode::kInvalidArgument);

  // The mandated recovery (abort) and a visibly successful next frame.
  ASSERT_TRUE(backend_->abort_frame(*cmds).has_value());
  const TextRun valid[] = {{32.0F, 32.0F, 32.0F, 0xFFFFFFFFU, "OK"}};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render_backend_frame(valid, pixels));
  size_t ink = 0;
  for (size_t i = 0; i < pixels.size(); i += 4) {
    ink += pixels[i] > kInkThreshold ? 1 : 0;
  }
  EXPECT_GT(ink, 20u) << "text layer stayed bricked after a rejected record + abort";
}

TEST_F(HolovizBackendTextTest, RecordRetryAfterFailedRecordSucceeds) {
  make_backend();

  // Fail a record, then record valid content in the SAME open frame (no abort): the fix
  // must live in the record path itself, not only in abort_frame.
  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());
  const TextRun bad[] = {{10.0F, 10.0F, -1.0F, 0xFFFFFFFFU, "bad"}};
  const TextFrameData bad_frame{bad};
  RenderList bad_list;
  bad_list.layers.push_back(LayerFrame{0, &bad_frame});
  ASSERT_FALSE(backend_->record(*cmds, bad_list, CompositeSpec{}).has_value());

  const TextRun good[] = {{32.0F, 32.0F, 32.0F, 0xFFFFFFFFU, "OK"}};
  const TextFrameData good_frame{good};
  RenderList good_list;
  good_list.layers.push_back(LayerFrame{0, &good_frame});
  ASSERT_TRUE(backend_->record(*cmds, good_list, CompositeSpec{}).has_value());
  auto token = backend_->submit(*cmds, SubmitContext{0});
  ASSERT_TRUE(token.has_value()) << token.error().message;
  ASSERT_TRUE(device_
                  .wait_on_submit(submit_payload(*token, device_identity(device_)).payload->token,
                                  std::chrono::seconds(5))
                  .has_value());
}

TEST_F(HolovizBackendTextTest, WorstCaseOneCharLinesFitTheDeclaredBudget) {
  // Worst case draws-per-glyph: N single-character runs at max_glyphs = N. The renderer's
  // draw list is preallocated from the declared budget, so both frames (first and steady
  // state) must succeed.
  constexpr uint32_t kMaxGlyphs = 64;
  make_backend(kMaxGlyphs);

  std::vector<TextRun> runs;
  runs.reserve(kMaxGlyphs);
  for (uint32_t i = 0; i < kMaxGlyphs; ++i) {
    runs.push_back(TextRun{
        5.0F + 3.0F * static_cast<float>(i % 32), i < 32 ? 32.0F : 77.0F, 13.0F, 0xFFFFFFFFU, "a"});
  }
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render_backend_frame(runs, pixels, 0));
  ASSERT_NO_FATAL_FAILURE(render_backend_frame(runs, pixels, 0));
  size_t ink = 0;
  for (size_t i = 0; i < pixels.size(); i += 4) {
    ink += pixels[i] > kInkThreshold ? 1 : 0;
  }
  EXPECT_GT(ink, 100u) << "worst-case fragmented frame rendered nothing";
}

TEST_F(HolovizBackendTextTest, SecondFrameReplacesTheFirst) {
  make_backend();

  // Two frames through the same slot with different content; the readback must show only
  // the second frame's text (slot recycling + clear-on-load).
  const TextRun first[] = {{32.0F, 32.0F, 32.0F, 0xFFFFFFFFU, "AAAA"}};
  const TextRun second[] = {{128.0F, 64.0F, 32.0F, 0xFFFFFFFFU, "B"}};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render_backend_frame(first, pixels, 0));
  ASSERT_NO_FATAL_FAILURE(render_backend_frame(second, pixels, 0));

  // "AAAA" started at x=32; "B" starts at x=128. Ink left of x=100 means frame 1 leaked.
  size_t ink_left_half = 0;
  size_t ink_total = 0;
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      const size_t i = (static_cast<size_t>(y) * kWidth + x) * 4;
      const bool ink = pixels[i] > kInkThreshold || pixels[i + 1] > kInkThreshold ||
                       pixels[i + 2] > kInkThreshold;
      ink_total += ink ? 1 : 0;
      ink_left_half += (ink && x < 100) ? 1 : 0;
    }
  }
  EXPECT_GT(ink_total, 20u) << "second frame rendered nothing";
  EXPECT_EQ(ink_left_half, 0u) << "first frame's text leaked into the second frame";
}

}  // namespace
}  // namespace holoscan::visualization::backends
