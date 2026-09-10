// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Differential tests between the two text renderer implementations: the
// ImGui-backed renderer (create_text_renderer) and the push-constant
// monospace renderer (create_monospace_text_renderer). Both render the same
// text into identical offscreen targets on the same device in the same run,
// so the comparison needs no golden images and is immune to driver/GPU
// rasterization variance between machines.
//
// Content is chosen to dodge the known, intentional layout divergences
// between the implementations (multi-line pitch, anchor math, non-ASCII
// handling): single lines, Anchor::kTopLeft, printable ASCII, draw size equal
// to the atlas bake size, and a genuinely monospace font (imgui's bundled
// Cousine-Regular.ttf, path injected as HOLOVIZ_TEST_MONO_TTF) so ImGui's
// per-glyph advances equal the monospace renderer's forced max advance.
//
// On comparison failure the tests dump the rendered images plus a per-pixel
// diff as BMPs into the working directory (ctest runs in the build dir) for
// inspection; set HOLOVIZ_TEXT_COMPARE_DUMP=1 to get the dumps on success
// too. The diff is amplified 8x so near-threshold differences are visible.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoviz/rhi.hpp>
#include <holoviz/text.hpp>

#include "image_compare_test_util.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz::text {
namespace {

namespace rhi = holoviz::rhi;

using test_util::compare_images;
using test_util::Draw;
using test_util::ImageDiff;
using test_util::kInkThreshold;
using test_util::log_diff;
using test_util::maybe_dump;
using test_util::render_with;
using test_util::try_create_device;

// The differential acceptance predicate, shared by the GPU tests below and the CPU oracles
// that prove it rejects equal-ink displacement. Whole-image differing_fraction alone is too
// forgiving here: the 512x128 target is mostly clear background, so a 5% gate would allow
// thousands of differing pixels while each renderer needs only ~500 ink pixels.
constexpr double kDifferentialInkIouGate = 0.5;
constexpr double kDifferentialDiffOverInkGate = 0.75;

// CPU-only self-test: the comparator must actually be able to report
// differences, or the differential tests above it prove nothing.
TEST(TextCompareMetric, DetectsDifferencesAndInk) {
  // 4 pixels RGBA: black, dim (below ink), bright red, bright white.
  const std::vector<uint8_t> a = {
      0, 0, 0, 255, 10, 10, 10, 255, 200, 0, 0, 255, 255, 255, 255, 255};
  std::vector<uint8_t> b = a;

  const ImageDiff same = compare_images(a, b, 0);
  EXPECT_EQ(same.differing_fraction, 0.0);
  EXPECT_EQ(same.max_channel_delta, 0);
  EXPECT_EQ(same.ink_a, 2u);  // red + white
  EXPECT_EQ(same.ink_b, 2u);

  b[8] = 100;  // dim the red pixel's R channel by 100
  const ImageDiff changed = compare_images(a, b, 32);
  EXPECT_EQ(changed.max_channel_delta, 100);
  EXPECT_DOUBLE_EQ(changed.differing_fraction, 0.25);
  const ImageDiff tolerated = compare_images(a, b, 100);
  EXPECT_EQ(tolerated.differing_fraction, 0.0);
}

/// Build a width*height RGBA8 image, black except a white w*h block at (x, y).
std::vector<uint8_t> block_image(uint32_t width, uint32_t height, uint32_t x, uint32_t y,
                                 uint32_t w, uint32_t h) {
  std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4, 0);
  for (size_t i = 3; i < pixels.size(); i += 4) {
    pixels[i] = 255;  // opaque alpha everywhere
  }
  for (uint32_t row = y; row < y + h; ++row) {
    for (uint32_t col = x; col < x + w; ++col) {
      const size_t i = (static_cast<size_t>(row) * width + col) * 4;
      pixels[i] = pixels[i + 1] = pixels[i + 2] = 255;
    }
  }
  return pixels;
}

// CPU oracle for the foreground gate: images with EQUAL ink that is displaced or disjoint
// must fail the differential predicate, and identical images must pass it. This is what the
// whole-image differing_fraction gate could not guarantee.
TEST(TextCompareMetric, RejectsEqualInkDisplacement) {
  constexpr uint32_t kSide = 64;
  const auto base = block_image(kSide, kSide, 8, 8, 16, 16);

  // Identical foregrounds pass.
  const ImageDiff same = compare_images(base, base, 32);
  EXPECT_DOUBLE_EQ(same.ink_iou, 1.0);
  EXPECT_DOUBLE_EQ(same.differing_over_ink_union, 0.0);
  EXPECT_GT(same.ink_iou, kDifferentialInkIouGate);
  EXPECT_LT(same.differing_over_ink_union, kDifferentialDiffOverInkGate);

  // Equal ink translated by half a block: heavy overlap loss must trip the gate even though
  // ink_a == ink_b and the whole-image differing fraction stays small.
  const auto translated = block_image(kSide, kSide, 16, 16, 16, 16);
  const ImageDiff shifted = compare_images(base, translated, 32);
  EXPECT_EQ(shifted.ink_a, shifted.ink_b);
  EXPECT_LT(shifted.differing_fraction, 0.15) << "oracle premise: whole-image gate is weak";
  EXPECT_LE(shifted.ink_iou, kDifferentialInkIouGate) << "translated equal ink must fail";
  EXPECT_GE(shifted.differing_over_ink_union, kDifferentialDiffOverInkGate);

  // Equal but fully disjoint ink: zero overlap, every inked pixel differs.
  const auto disjoint = block_image(kSide, kSide, 40, 40, 16, 16);
  const ImageDiff apart = compare_images(base, disjoint, 32);
  EXPECT_EQ(apart.ink_a, apart.ink_b);
  EXPECT_DOUBLE_EQ(apart.ink_iou, 0.0);
  EXPECT_GE(apart.differing_over_ink_union, 1.0);
}

/// Device plus a CUDA stream on its context, as the ImGui-backed renderer
/// requires. Skips (never fails) on machines without a usable adapter.
class TextCompareTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = try_create_device();
    if (!device_) {
      HOLOVIZ_ENV_SKIP() << "no usable GPU/adapter on this machine";
    }
    const auto context = device_.cuda_context();
    ASSERT_TRUE(context.has_value()) << context.error().message;
    if (cuCtxSetCurrent(*context) != CUDA_SUCCESS ||
        cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
      HOLOVIZ_ENV_SKIP() << "cannot create a CUDA stream on the device context";
    }
  }

  void TearDown() override {
    if (device_) {
      EXPECT_TRUE(device_.wait_idle().has_value());
    }
    if (stream_ != nullptr) {
      cuStreamSynchronize(stream_);
      cuStreamDestroy(stream_);
    }
    if (device_) {
      EXPECT_TRUE(device_.destroy().has_value());
    }
  }

  /// The bundled monospace TTF, or skips the current test when this build
  /// resolved imgui without its misc/fonts directory (SDK-integrated builds).
  static const char* require_mono_ttf() {
    if (!std::filesystem::exists(HOLOVIZ_TEST_MONO_TTF)) {
      return nullptr;
    }
    return HOLOVIZ_TEST_MONO_TTF;
  }

  rhi::Device device_;
  CUstream stream_ = nullptr;
};

#define SKIP_WITHOUT_MONO_TTF(ttf)                                              \
  if ((ttf) == nullptr) {                                                       \
    GTEST_SKIP() << "monospace test font not found: " << HOLOVIZ_TEST_MONO_TTF; \
  }

// Public factory contract: a default-constructed TextRendererDesc carries a null stream,
// which the ImGui-backed renderer rejects up front with kInvalidArgument (the descriptor
// documents the non-null requirement) instead of failing deep inside resource creation.
TEST_F(TextCompareTest, FactoryRejectsNullStream) {
  const auto renderer = create_text_renderer(device_, {});
  ASSERT_FALSE(renderer.has_value());
  EXPECT_EQ(renderer.error().code, ErrorCode::kInvalidArgument);
  EXPECT_NE(renderer.error().message.find("stream"), std::string::npos);
  // TearDown proves the device still tears down cleanly after the rejected create.
}

// Both renderers bake Cousine at 32 px and draw at 32 px (no bitmap scaling),
// single-line kTopLeft runs, so the images should agree closely. The gate is
// deliberately tolerant of subpixel placement drift between the two layout
// paths but fails on missing/garbled/misplaced text.
constexpr float kBakeSizePx = 32.0F;
constexpr rhi::Extent2D kCompareExtent{512, 128};
constexpr Draw kCompareDraws[] = {
    {{.position = {16.0F, 16.0F}, .size_px = kBakeSizePx, .color = 0xFFFFFFFFu},
     "Hello RHI 0123 !?"},
    {{.position = {16.0F, 72.0F}, .size_px = kBakeSizePx, .color = 0xFF4080FFu},
     "differential {test}"},
};

TEST_F(TextCompareTest, DifferentialImGuiVsMonospace) {
  const char* ttf = require_mono_ttf();
  SKIP_WITHOUT_MONO_TTF(ttf);

  auto imgui_result = create_text_renderer(device_,
                                           {.stream = stream_,
                                            .frames_in_flight = 2,
                                            .ttf_path = ttf,
                                            .base_font_size_px = kBakeSizePx});
  ASSERT_TRUE(imgui_result.has_value()) << imgui_result.error().message;
  TextRenderer imgui_renderer = std::move(*imgui_result);

  auto mono_result = create_monospace_text_renderer(
      device_, {.ttf_path = ttf, .atlas_glyph_height_px = kBakeSizePx});
  ASSERT_TRUE(mono_result.has_value()) << mono_result.error().message;
  TextRenderer mono_renderer = std::move(*mono_result);

  std::vector<uint8_t> imgui_pixels;
  std::vector<uint8_t> mono_pixels;
  ASSERT_NO_FATAL_FAILURE(
      render_with(device_, imgui_renderer, kCompareExtent, kCompareDraws, imgui_pixels));
  ASSERT_NO_FATAL_FAILURE(
      render_with(device_, mono_renderer, kCompareExtent, kCompareDraws, mono_pixels));

  const ImageDiff diff = compare_images(imgui_pixels, mono_pixels, /*channel_tol=*/32);
  log_diff("imgui(a) vs monospace(b), channel_tol=32", diff);
  RecordProperty("differing_fraction", std::to_string(diff.differing_fraction));
  RecordProperty("ink_imgui", static_cast<int>(diff.ink_a));
  RecordProperty("ink_mono", static_cast<int>(diff.ink_b));

  // Both images must actually contain text (roughly comparable amounts), and
  // the per-pixel disagreement must stay small.
  EXPECT_GT(diff.ink_a, 500u) << "ImGui image is (nearly) blank";
  EXPECT_GT(diff.ink_b, 500u) << "monospace image is (nearly) blank";
  const double ink_ratio =
      static_cast<double>(std::min(diff.ink_a, diff.ink_b)) / std::max(diff.ink_a, diff.ink_b);
  EXPECT_GT(ink_ratio, 0.75) << "ink " << diff.ink_a << " vs " << diff.ink_b;
  EXPECT_LT(diff.differing_fraction, 0.05);
  // Foreground gate: the ink must actually overlap (see TextCompareMetric's
  // RejectsEqualInkDisplacement oracle) — equal-ink but shifted/garbled glyph regions fail
  // here even when the whole-image fraction above stays green.
  EXPECT_GT(diff.ink_iou, kDifferentialInkIouGate);
  EXPECT_LT(diff.differing_over_ink_union, kDifferentialDiffOverInkGate);

  maybe_dump("imgui_vs_mono", imgui_pixels, mono_pixels, kCompareExtent, HasFailure());

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(imgui_renderer.destroy().has_value());
  EXPECT_TRUE(mono_renderer.destroy().has_value());
}

// Same renderer, same atlas, same shader math — only the rasterization path
// (two triangles vs VK_NV_fill_rectangle) differs — so the images must match
// almost exactly.
TEST_F(TextCompareTest, DifferentialTrianglesVsFillRect) {
  const char* ttf = require_mono_ttf();
  SKIP_WITHOUT_MONO_TTF(ttf);

  const auto capabilities = device_.capabilities();
  ASSERT_TRUE(capabilities.has_value());
  if (!capabilities->fill_rectangle) {
    GTEST_SKIP() << "VK_NV_fill_rectangle unavailable on this device";
  }

  auto make_renderer = [&](MonospaceFillMode fill_mode) {
    return create_monospace_text_renderer(
        device_, {.fill_mode = fill_mode, .ttf_path = ttf, .atlas_glyph_height_px = kBakeSizePx});
  };
  auto triangles_result = make_renderer(MonospaceFillMode::kTriangles);
  ASSERT_TRUE(triangles_result.has_value()) << triangles_result.error().message;
  TextRenderer triangles_renderer = std::move(*triangles_result);
  auto fillrect_result = make_renderer(MonospaceFillMode::kFillRectangle);
  ASSERT_TRUE(fillrect_result.has_value()) << fillrect_result.error().message;
  TextRenderer fillrect_renderer = std::move(*fillrect_result);

  std::vector<uint8_t> triangles_pixels;
  std::vector<uint8_t> fillrect_pixels;
  ASSERT_NO_FATAL_FAILURE(
      render_with(device_, triangles_renderer, kCompareExtent, kCompareDraws, triangles_pixels));
  ASSERT_NO_FATAL_FAILURE(
      render_with(device_, fillrect_renderer, kCompareExtent, kCompareDraws, fillrect_pixels));

  const ImageDiff diff = compare_images(triangles_pixels, fillrect_pixels, /*channel_tol=*/2);
  log_diff("triangles(a) vs fillrect(b), channel_tol=2", diff);
  RecordProperty("differing_fraction", std::to_string(diff.differing_fraction));

  EXPECT_GT(diff.ink_a, 500u) << "triangles image is (nearly) blank";
  EXPECT_LT(diff.differing_fraction, 0.001);
  // Identical geometry through two rasterization paths: the foregrounds must coincide
  // almost exactly, not merely balance in total ink.
  EXPECT_GT(diff.ink_iou, 0.95);
  EXPECT_LT(diff.differing_over_ink_union, 0.05);

  maybe_dump("tri_vs_fillrect", triangles_pixels, fillrect_pixels, kCompareExtent, HasFailure());

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(triangles_renderer.destroy().has_value());
  EXPECT_TRUE(fillrect_renderer.destroy().has_value());
}

// Analytic oracle (needs no reference image): a single glyph's ink must land
// inside its expected cell and nowhere else on the target.
TEST_F(TextCompareTest, MonospaceInkStaysInsideExpectedBox) {
  const char* ttf = require_mono_ttf();
  SKIP_WITHOUT_MONO_TTF(ttf);

  auto renderer_result = create_monospace_text_renderer(
      device_, {.ttf_path = ttf, .atlas_glyph_height_px = kBakeSizePx});
  ASSERT_TRUE(renderer_result.has_value()) << renderer_result.error().message;
  TextRenderer renderer = std::move(*renderer_result);

  const rhi::Extent2D extent{128, 128};
  const Draw draws[] = {
      {{.position = {32.0F, 32.0F}, .size_px = kBakeSizePx, .color = 0xFFFFFFFFu}, "H"},
  };
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render_with(device_, renderer, extent, draws, pixels));

  // A 32 px glyph placed with a top-left pen at (32,32) must keep all ink in
  // a box of one glyph-height square, padded by 2 px for rounding.
  const uint32_t box_min = 30;
  const uint32_t box_max = 32 + 32 + 2;
  size_t ink_inside = 0;
  size_t ink_outside = 0;
  for (uint32_t y = 0; y < extent.height; ++y) {
    for (uint32_t x = 0; x < extent.width; ++x) {
      const size_t i = (static_cast<size_t>(y) * extent.width + x) * 4;
      const bool ink = pixels[i] > kInkThreshold || pixels[i + 1] > kInkThreshold ||
                       pixels[i + 2] > kInkThreshold;
      if (!ink) {
        continue;
      }
      const bool inside = x >= box_min && x <= box_max && y >= box_min && y <= box_max;
      (inside ? ink_inside : ink_outside) += 1;
    }
  }
  std::cout << "[compare] single 'H' glyph: ink_inside=" << ink_inside
            << " ink_outside=" << ink_outside << "\n";
  EXPECT_GE(ink_inside, 40u) << "glyph 'H' left too little ink in its cell";
  EXPECT_EQ(ink_outside, 0u) << "glyph 'H' leaked ink outside its cell";

  maybe_dump("ink_box", pixels, pixels, extent, HasFailure());

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(renderer.destroy().has_value());
}

// Smoke test for the ImGui-backed renderer with the default font: no other
// test exercises this path (or a real CUDA end_frame token through submit).
TEST_F(TextCompareTest, ImGuiRendererRendersVisiblePixels) {
  auto renderer_result = create_text_renderer(device_, {.stream = stream_, .frames_in_flight = 2});
  ASSERT_TRUE(renderer_result.has_value()) << renderer_result.error().message;
  TextRenderer renderer = std::move(*renderer_result);

  const rhi::Extent2D extent{256, 128};
  const Draw draws[] = {
      {{.position = {20.0F, 20.0F}, .size_px = 64.0F, .color = 0xFFFFFFFFu}, "Hi"},
  };
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render_with(device_, renderer, extent, draws, pixels));

  bool any_text_pixel = false;
  for (size_t i = 0; i < pixels.size() && !any_text_pixel; i += 4) {
    any_text_pixel = pixels[i] != 0 || pixels[i + 1] != 0 || pixels[i + 2] != 0;
  }
  EXPECT_TRUE(any_text_pixel) << "rendered image is entirely clear color";

  maybe_dump("imgui_smoke", pixels, pixels, extent, HasFailure());

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(renderer.destroy().has_value());
}

}  // namespace
}  // namespace holoviz::text
