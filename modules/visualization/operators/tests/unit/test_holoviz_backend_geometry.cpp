// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Geometry layer tests for HolovizRenderBackend, all analytic (no reference images): each
// primitive's ink must land where its parameters say and nowhere else. Rendered headless
// at 256x256 so normalized coordinates map to exact pixels.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/holoviz.hpp>
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

using test_util::kInkThreshold;
using test_util::maybe_dump;
using test_util::try_create_device;

constexpr uint32_t kSize = 256;  // square target: normalized units map 1:1 per axis

bool is_ink(const std::vector<uint8_t>& pixels, uint32_t x, uint32_t y) {
  const size_t i = (static_cast<size_t>(y) * kSize + x) * 4;
  return pixels[i] > kInkThreshold || pixels[i + 1] > kInkThreshold ||
         pixels[i + 2] > kInkThreshold;
}

class HolovizBackendGeometryTest : public ::testing::Test {
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

  void make_backend(std::vector<GeometryTopology> topologies, uint32_t max_primitives = 64) {
    auto backend = create_holoviz_render_backend(device_, {});
    ASSERT_TRUE(backend.has_value());
    backend_ = std::move(*backend);

    LayerStack bundle;
    bundle.add_geometry_layer("overlay",
                              GeometryLayerDesc{std::move(topologies), max_primitives, 4.0F});

    RenderConfig config;
    config.headless = true;
    config.headless_width = kSize;
    config.headless_height = kSize;
    config.max_frames_in_flight = 2;
    ASSERT_TRUE(backend_->allocate(bundle, config).has_value());
    ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());
  }

  void render(std::span<const GeometryBatch> batches, std::vector<uint8_t>& out_pixels,
              const CompositeSpec& composite = {}) {
    auto cmds = backend_->begin_frame(AcquireToken{}, 0);
    ASSERT_TRUE(cmds.has_value()) << cmds.error().message;

    const GeometryFrameData geometry{batches};
    RenderList list;
    list.layers.push_back(LayerFrame{0, &geometry});
    ASSERT_TRUE(backend_->record(*cmds, list, composite).has_value());

    auto token = backend_->submit(*cmds, SubmitContext{0});
    ASSERT_TRUE(token.has_value()) << token.error().message;
    const SubmitPayload* payload = submit_payload(*token, device_identity(device_)).payload;
    ASSERT_NE(payload, nullptr);
    ASSERT_TRUE(device_.wait_on_submit(payload->token, std::chrono::seconds(5)).has_value());

    auto* concrete = static_cast<HolovizRenderBackend*>(backend_.get());
    const auto target = concrete->headless_target(0);
    ASSERT_TRUE(target.has_value());
    out_pixels.resize(static_cast<size_t>(kSize) * kSize * 4);
    ASSERT_TRUE(device_
                    .download_from_image(target->get(),
                                         0,
                                         std::as_writable_bytes(std::span(out_pixels)),
                                         static_cast<size_t>(kSize) * 4)
                    .has_value());
  }

  holoviz::Device device_;
  std::unique_ptr<RenderBackend> backend_;
};

TEST_F(HolovizBackendGeometryTest, RectangleOutlineInkStaysInTheBand) {
  make_backend({GeometryTopology::kRectangleList});

  // Rect (64,64)-(192,192) in pixels, 8 px outline: ink must lie within 4+1 px of the rect
  // perimeter and nowhere else.
  const float coords[] = {0.25F, 0.25F, 0.75F, 0.75F};
  const GeometryBatch batch{
      GeometryTopology::kRectangleList, coords, 1, {1.0F, 1.0F, 1.0F, 1.0F}, 8.0F};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render({&batch, 1}, pixels));

  const float half_band = 4.0F + 1.0F;  // half line width + rasterization tolerance
  size_t ink_in_band = 0;
  size_t ink_outside = 0;
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      if (!is_ink(pixels, x, y)) {
        continue;
      }
      // Distance from the pixel center to the rectangle's perimeter.
      const float px = x + 0.5F;
      const float py = y + 0.5F;
      const float dx = std::max({64.0F - px, px - 192.0F, 0.0F});
      const float dy = std::max({64.0F - py, py - 192.0F, 0.0F});
      const float outside_dist = std::sqrt(dx * dx + dy * dy);
      const float inside_dist = (dx == 0.0F && dy == 0.0F)
                                    ? std::min({px - 64.0F, 192.0F - px, py - 64.0F, 192.0F - py})
                                    : 0.0F;
      const float perimeter_dist = std::max(outside_dist, inside_dist);
      (perimeter_dist <= half_band ? ink_in_band : ink_outside) += 1;
    }
  }
  EXPECT_GT(ink_in_band, 500u) << "outline rendered too little ink";
  EXPECT_EQ(ink_outside, 0u) << "outline leaked outside its band";

  maybe_dump("geom_rect", pixels, pixels, holoviz::Extent2D{kSize, kSize}, HasFailure());
}

TEST_F(HolovizBackendGeometryTest, CrossInkOnlyOnItsArms) {
  make_backend({GeometryTopology::kCrossList});

  // Cross at (128,128), size 64 px: arms span 96..160 on each axis.
  const float coords[] = {0.5F, 0.5F, 0.25F};
  const GeometryBatch batch{GeometryTopology::kCrossList, coords, 1, {1.0F, 1.0F, 1.0F, 1.0F}};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render({&batch, 1}, pixels));

  size_t ink_on_arms = 0;
  size_t ink_elsewhere = 0;
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      if (!is_ink(pixels, x, y)) {
        continue;
      }
      const bool on_horizontal = y >= 126 && y <= 130 && x >= 94 && x <= 162;
      const bool on_vertical = x >= 126 && x <= 130 && y >= 94 && y <= 162;
      ((on_horizontal || on_vertical) ? ink_on_arms : ink_elsewhere) += 1;
    }
  }
  EXPECT_GT(ink_on_arms, 60u) << "cross rendered too little ink";
  EXPECT_EQ(ink_elsewhere, 0u) << "cross leaked ink off its arms";

  maybe_dump("geom_cross", pixels, pixels, holoviz::Extent2D{kSize, kSize}, HasFailure());
}

TEST_F(HolovizBackendGeometryTest, OvalRingInkAtTheRadii) {
  make_backend({GeometryTopology::kOvalList});

  // Circle at (128,128), radius 64 px, 8 px ring.
  const float coords[] = {0.5F, 0.5F, 0.25F, 0.25F};
  const GeometryBatch batch{GeometryTopology::kOvalList, coords, 1, {1.0F, 1.0F, 1.0F, 1.0F}, 8.0F};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render({&batch, 1}, pixels));

  size_t ink_in_ring = 0;
  size_t ink_outside = 0;
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      if (!is_ink(pixels, x, y)) {
        continue;
      }
      const float dx = (x + 0.5F) - 128.0F;
      const float dy = (y + 0.5F) - 128.0F;
      const float r = std::sqrt(dx * dx + dy * dy);
      // 4 px half-width + 1 px rasterization tolerance + polygon-vs-circle sagitta error
      // (32 segments over r=68: (1 - cos(pi/32)) * 68 < 0.4 px).
      const bool in_ring = r >= 64.0F - 5.5F && r <= 64.0F + 5.5F;
      (in_ring ? ink_in_ring : ink_outside) += 1;
    }
  }
  EXPECT_GT(ink_in_ring, 800u) << "ring rendered too little ink";
  EXPECT_EQ(ink_outside, 0u) << "ring leaked ink off the circle";

  const size_t center_i = (static_cast<size_t>(128) * kSize + 128) * 4;
  EXPECT_LE(pixels[center_i], kInkThreshold) << "circle center must stay empty";

  maybe_dump("geom_oval", pixels, pixels, holoviz::Extent2D{kSize, kSize}, HasFailure());
}

TEST_F(HolovizBackendGeometryTest, LineStripIsContinuous) {
  make_backend({GeometryTopology::kLineStrip});

  // Horizontal-ish strip across the middle: (32,128) -> (128,144) -> (224,128).
  const float coords[] = {0.125F, 0.5F, 0.5F, 0.5625F, 0.875F, 0.5F};
  const GeometryBatch batch{GeometryTopology::kLineStrip, coords, 2, {1.0F, 1.0F, 1.0F, 1.0F}};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render({&batch, 1}, pixels));

  // Every column between the endpoints must contain ink (continuity), and nothing outside
  // the strip's bounding band may.
  size_t missing_columns = 0;
  size_t ink_outside = 0;
  for (uint32_t x = 33; x < 224; ++x) {
    bool column_has_ink = false;
    for (uint32_t y = 0; y < kSize; ++y) {
      if (!is_ink(pixels, x, y)) {
        continue;
      }
      column_has_ink = true;
      if (y < 126 || y > 146) {
        ++ink_outside;
      }
    }
    missing_columns += column_has_ink ? 0 : 1;
  }
  EXPECT_EQ(missing_columns, 0u) << "the strip has gaps";
  EXPECT_EQ(ink_outside, 0u) << "strip ink outside its band";

  maybe_dump("geom_strip", pixels, pixels, holoviz::Extent2D{kSize, kSize}, HasFailure());
}

TEST_F(HolovizBackendGeometryTest, LineListHonorsLineWidth) {
  make_backend({GeometryTopology::kLineList});

  // Horizontal segment (64,128)-(192,128). Width 8: an ~8 px tall band centered on y=128.
  const float coords[] = {0.25F, 0.5F, 0.75F, 0.5F};
  const auto render_with_width = [&](float width, std::vector<uint8_t>& pixels) {
    const GeometryBatch batch{
        GeometryTopology::kLineList, coords, 1, {1.0F, 1.0F, 1.0F, 1.0F}, width};
    ASSERT_NO_FATAL_FAILURE(render({&batch, 1}, pixels));
  };

  std::vector<uint8_t> wide;
  render_with_width(8.0F, wide);
  const auto column_thickness = [&](const std::vector<uint8_t>& pixels, uint32_t x) {
    uint32_t rows = 0;
    for (uint32_t y = 0; y < kSize; ++y) {
      rows += is_ink(pixels, x, y) ? 1 : 0;
    }
    return rows;
  };
  // Mid-segment column: ~8 rows of ink (1 px rasterization tolerance each side).
  const uint32_t wide_rows = column_thickness(wide, 128);
  EXPECT_GE(wide_rows, 6u) << "8 px line rendered too thin";
  EXPECT_LE(wide_rows, 10u) << "8 px line rendered too thick";
  // Nothing outside the band (+ square caps extending 4 px past the endpoints).
  size_t ink_outside = 0;
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      if (is_ink(wide, x, y) && (y < 123 || y > 133 || x < 59 || x > 197)) {
        ++ink_outside;
      }
    }
  }
  EXPECT_EQ(ink_outside, 0u) << "line ink leaked outside its band";

  // A narrower width renders strictly thinner.
  std::vector<uint8_t> narrow;
  render_with_width(2.0F, narrow);
  EXPECT_LT(column_thickness(narrow, 128), wide_rows);

  maybe_dump("geom_line_width", wide, wide, holoviz::Extent2D{kSize, kSize}, HasFailure());
}

TEST_F(HolovizBackendGeometryTest, LineStripHonorsLineWidthAndCoversJoints) {
  make_backend({GeometryTopology::kLineStrip});

  // L-shaped strip (64,64) -> (192,64) -> (192,192), width 8.
  const float coords[] = {0.25F, 0.25F, 0.75F, 0.25F, 0.75F, 0.75F};
  const GeometryBatch batch{
      GeometryTopology::kLineStrip, coords, 2, {1.0F, 1.0F, 1.0F, 1.0F}, 8.0F};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render({&batch, 1}, pixels));

  // Arm thickness: ~8 rows on the horizontal arm, ~8 columns on the vertical arm.
  uint32_t horizontal_rows = 0;
  for (uint32_t y = 0; y < kSize; ++y) {
    horizontal_rows += is_ink(pixels, 128, y) ? 1 : 0;
  }
  EXPECT_GE(horizontal_rows, 6u);
  EXPECT_LE(horizontal_rows, 10u);
  uint32_t vertical_cols = 0;
  for (uint32_t x = 0; x < kSize; ++x) {
    vertical_cols += is_ink(pixels, x, 128) ? 1 : 0;
  }
  EXPECT_GE(vertical_cols, 6u);
  EXPECT_LE(vertical_cols, 10u);

  // The corner pixel itself is covered (square caps overlap at the joint).
  EXPECT_TRUE(is_ink(pixels, 192, 64)) << "strip joint left a gap at the corner";

  // No ink outside the union of the two arm bands (+ cap slack).
  size_t ink_outside = 0;
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      if (!is_ink(pixels, x, y)) {
        continue;
      }
      const bool on_horizontal = y >= 59 && y <= 69 && x >= 59 && x <= 197;
      const bool on_vertical = x >= 187 && x <= 197 && y >= 59 && y <= 197;
      if (!on_horizontal && !on_vertical) {
        ++ink_outside;
      }
    }
  }
  EXPECT_EQ(ink_outside, 0u) << "strip ink outside its arms";

  maybe_dump("geom_strip_width", pixels, pixels, holoviz::Extent2D{kSize, kSize}, HasFailure());
}

TEST_F(HolovizBackendGeometryTest, CrossHonorsLineWidth) {
  make_backend({GeometryTopology::kCrossList});

  // Cross at (128,128), size 64 px (arms 96..160), width 6.
  const float coords[] = {0.5F, 0.5F, 0.25F};
  const GeometryBatch batch{
      GeometryTopology::kCrossList, coords, 1, {1.0F, 1.0F, 1.0F, 1.0F}, 6.0F};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render({&batch, 1}, pixels));

  // Thickness probe on the horizontal arm away from the center: ~6 rows of ink.
  uint32_t rows = 0;
  for (uint32_t y = 0; y < kSize; ++y) {
    rows += is_ink(pixels, 104, y) ? 1 : 0;
  }
  EXPECT_GE(rows, 4u) << "6 px cross arm rendered too thin";
  EXPECT_LE(rows, 8u) << "6 px cross arm rendered too thick";

  // Only on the arms (bands widened for the 3 px half-width + tolerance + caps).
  size_t ink_elsewhere = 0;
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      if (!is_ink(pixels, x, y)) {
        continue;
      }
      const bool on_horizontal = y >= 124 && y <= 132 && x >= 92 && x <= 164;
      const bool on_vertical = x >= 124 && x <= 132 && y >= 92 && y <= 164;
      if (!on_horizontal && !on_vertical) {
        ++ink_elsewhere;
      }
    }
  }
  EXPECT_EQ(ink_elsewhere, 0u) << "cross leaked ink off its arms";

  maybe_dump("geom_cross_width", pixels, pixels, holoviz::Extent2D{kSize, kSize}, HasFailure());
}

TEST_F(HolovizBackendGeometryTest, MultipleStripsAtExactPrimitiveBudgetSucceed) {
  // Two one-segment strips summing to exactly max_primitives: each batch stages its own
  // trailing point, and allocation must budget for that worst case (4 floats/primitive).
  make_backend({GeometryTopology::kLineStrip}, /*max_primitives=*/2);

  const float coords_a[] = {0.1F, 0.25F, 0.9F, 0.25F};  // y = 64
  const float coords_b[] = {0.1F, 0.75F, 0.9F, 0.75F};  // y = 192
  const GeometryBatch batches[] = {
      GeometryBatch{GeometryTopology::kLineStrip, coords_a, 1, {1.0F, 1.0F, 1.0F, 1.0F}},
      GeometryBatch{GeometryTopology::kLineStrip, coords_b, 1, {1.0F, 1.0F, 1.0F, 1.0F}},
  };
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render({batches, 2}, pixels));
  EXPECT_TRUE(is_ink(pixels, 128, 64)) << "first strip did not render";
  EXPECT_TRUE(is_ink(pixels, 128, 192)) << "second strip did not render";
}

TEST_F(HolovizBackendGeometryTest, MixedTopologyAtExactBudgetSucceeds) {
  make_backend({GeometryTopology::kLineStrip, GeometryTopology::kLineList},
               /*max_primitives=*/3);

  // One 1-segment strip + two line-list segments = 3 primitives, the exact budget.
  const float strip_coords[] = {0.1F, 0.25F, 0.9F, 0.25F};
  const float line_coords[] = {0.1F, 0.5F, 0.9F, 0.5F, 0.1F, 0.75F, 0.9F, 0.75F};
  const GeometryBatch batches[] = {
      GeometryBatch{GeometryTopology::kLineStrip, strip_coords, 1, {1.0F, 1.0F, 1.0F, 1.0F}},
      GeometryBatch{GeometryTopology::kLineList, line_coords, 2, {1.0F, 1.0F, 1.0F, 1.0F}},
  };
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render({batches, 2}, pixels));
  EXPECT_TRUE(is_ink(pixels, 128, 64));
  EXPECT_TRUE(is_ink(pixels, 128, 128));
  EXPECT_TRUE(is_ink(pixels, 128, 192));
}

TEST_F(HolovizBackendGeometryTest, PrimitiveBudgetOverflowIsAMisdeclaration) {
  make_backend({GeometryTopology::kLineList}, /*max_primitives=*/2);

  const float coords[] = {0.1F, 0.1F, 0.2F, 0.2F, 0.3F, 0.3F, 0.4F, 0.4F, 0.5F, 0.5F, 0.6F, 0.6F};
  const GeometryBatch batch{GeometryTopology::kLineList, coords, 3, {1.0F, 1.0F, 1.0F, 1.0F}};

  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());
  const GeometryFrameData geometry{{&batch, 1}};
  RenderList list;
  list.layers.push_back(LayerFrame{0, &geometry});
  const auto recorded = backend_->record(*cmds, list, CompositeSpec{});
  ASSERT_FALSE(recorded.has_value());
  EXPECT_EQ(recorded.error().code, ErrorCode::kInvalidArgument);
  // A prevalidation rejection leaves the frame open (record() contract); this test takes
  // the always-sound abort instead and proves it works here too.
  ASSERT_TRUE(backend_->abort_frame(*cmds).has_value());
}

TEST_F(HolovizBackendGeometryTest, UndeclaredTopologyIsRejected) {
  make_backend({GeometryTopology::kLineList});

  const float coords[] = {0.25F, 0.25F, 0.75F, 0.75F};
  const GeometryBatch batch{GeometryTopology::kRectangleList, coords, 1};

  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());
  const GeometryFrameData geometry{{&batch, 1}};
  RenderList list;
  list.layers.push_back(LayerFrame{0, &geometry});
  const auto recorded = backend_->record(*cmds, list, CompositeSpec{});
  ASSERT_FALSE(recorded.has_value());
  EXPECT_EQ(recorded.error().code, ErrorCode::kInvalidArgument);
  ASSERT_TRUE(backend_->abort_frame(*cmds).has_value());
}

TEST_F(HolovizBackendGeometryTest, AbortRecoversTheSlotAfterFailedRecord) {
  make_backend({GeometryTopology::kRectangleList}, /*max_primitives=*/2);

  // Fail a record() on slot 0 (undeclared max_primitives overflow) ...
  const float bad_coords[] = {
      0.1F, 0.1F, 0.2F, 0.2F, 0.3F, 0.3F, 0.4F, 0.4F, 0.5F, 0.5F, 0.6F, 0.6F};
  const GeometryBatch bad_batch{
      GeometryTopology::kRectangleList, bad_coords, 3, {1.0F, 1.0F, 1.0F, 1.0F}};
  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());
  const GeometryFrameData bad_geometry{{&bad_batch, 1}};
  RenderList bad_list;
  bad_list.layers.push_back(LayerFrame{0, &bad_geometry});
  ASSERT_FALSE(backend_->record(*cmds, bad_list, CompositeSpec{}).has_value());

  // ... abort it (idempotently) ...
  ASSERT_TRUE(backend_->abort_frame(*cmds).has_value());
  ASSERT_TRUE(backend_->abort_frame(*cmds).has_value()) << "double abort must be a no-op success";

  // ... and the slot must open again and render a valid frame correctly.
  const float coords[] = {0.25F, 0.25F, 0.75F, 0.75F};
  const GeometryBatch batch{
      GeometryTopology::kRectangleList, coords, 1, {1.0F, 1.0F, 1.0F, 1.0F}, 8.0F};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render({&batch, 1}, pixels));
  const size_t top_edge_i = (static_cast<size_t>(64) * kSize + 128) * 4;
  EXPECT_GT(pixels[top_edge_i], kInkThreshold) << "recovered frame did not render";

  // The slot was since re-issued by render()'s begin_frame, so the retained handle is
  // stale: abort on it is the contract's no-op success (its frame is long gone) and must
  // never touch the newer issuance — the slot still renders correctly afterwards. The
  // handle is released first: retaining it through one more issuance would (correctly)
  // backpressure begin_frame under the two-handles retention bound.
  ASSERT_TRUE(backend_->abort_frame(*cmds).has_value());
  cmds->impl.reset();
  std::vector<uint8_t> pixels_after;
  ASSERT_NO_FATAL_FAILURE(render({&batch, 1}, pixels_after));
  EXPECT_GT(pixels_after[top_edge_i], kInkThreshold) << "the stale abort disturbed the live slot";
}

TEST_F(HolovizBackendGeometryTest, SemiTransparentOutlineComposesStraightAlpha) {
  make_backend({GeometryTopology::kRectangleList});

  // 50%-alpha white outline over the opaque black clear: straight-alpha blending must
  // produce ~50% gray. The premultiplied blend factors this test guards against produce
  // full white (255) instead.
  const float coords[] = {0.25F, 0.25F, 0.75F, 0.75F};
  const GeometryBatch batch{
      GeometryTopology::kRectangleList, coords, 1, {1.0F, 1.0F, 1.0F, 0.5F}, 8.0F};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render({&batch, 1}, pixels));

  const size_t top_edge_i = (static_cast<size_t>(64) * kSize + 128) * 4;
  for (int channel = 0; channel < 3; ++channel) {
    EXPECT_GE(pixels[top_edge_i + channel], 119) << "channel " << channel;
    EXPECT_LE(pixels[top_edge_i + channel], 137) << "channel " << channel;
  }

  maybe_dump("geom_alpha", pixels, pixels, holoviz::Extent2D{kSize, kSize}, HasFailure());
}

TEST_F(HolovizBackendGeometryTest, AlternatingSlotsKeepIndependentGeometry) {
  make_backend({GeometryTopology::kRectangleList});

  // Submit two frames back-to-back on both slots before waiting on either, then verify
  // each slot's target independently — a cross-slot sanity check for the per-slot
  // staging/pool indexing (slot 1's staging must not clobber slot 0's pending upload).
  const float coords_a[] = {0.25F, 0.25F, 0.75F, 0.75F};      // top edge at y=64
  const float coords_b[] = {0.125F, 0.125F, 0.875F, 0.875F};  // top edge at y=32
  const GeometryBatch batch_a{
      GeometryTopology::kRectangleList, coords_a, 1, {1.0F, 1.0F, 1.0F, 1.0F}, 8.0F};
  const GeometryBatch batch_b{
      GeometryTopology::kRectangleList, coords_b, 1, {1.0F, 1.0F, 1.0F, 1.0F}, 8.0F};

  const GeometryBatch* batches[2] = {&batch_a, &batch_b};
  SyncToken tokens[2];
  for (uint32_t slot = 0; slot < 2; ++slot) {
    auto cmds = backend_->begin_frame(AcquireToken{}, slot);
    ASSERT_TRUE(cmds.has_value()) << cmds.error().message;
    const GeometryFrameData geometry{{batches[slot], 1}};
    RenderList list;
    list.layers.push_back(LayerFrame{0, &geometry});
    ASSERT_TRUE(backend_->record(*cmds, list, CompositeSpec{}).has_value());
    auto token = backend_->submit(*cmds, SubmitContext{slot});
    ASSERT_TRUE(token.has_value()) << token.error().message;
    tokens[slot] = *token;
  }

  auto* concrete = static_cast<HolovizRenderBackend*>(backend_.get());
  for (uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_TRUE(
        device_
            .wait_on_submit(submit_payload(tokens[slot], device_identity(device_)).payload->token,
                            std::chrono::seconds(5))
            .has_value());
    std::vector<uint8_t> pixels(static_cast<size_t>(kSize) * kSize * 4);
    ASSERT_TRUE(device_
                    .download_from_image(concrete->headless_target(slot)->get(),
                                         0,
                                         std::as_writable_bytes(std::span(pixels)),
                                         static_cast<size_t>(kSize) * 4)
                    .has_value());
    // Each slot's rect has its top edge at a different row; both probes must see only
    // their own frame's geometry.
    const uint32_t own_edge_y = slot == 0 ? 64 : 32;
    const uint32_t other_edge_y = slot == 0 ? 32 : 64;
    EXPECT_TRUE(is_ink(pixels, 128, own_edge_y)) << "slot " << slot << " lost its geometry";
    EXPECT_FALSE(is_ink(pixels, 128, other_edge_y))
        << "slot " << slot << " shows the other slot's geometry";
  }
}

// Two overlapping opaque rects in one layer recorded twice through CompositeSpec is not
// expressible (order indexes layers, not batches) — instead verify per-frame reorder across
// two layers: the layer drawn last must win at the overlap.
class HolovizBackendGeometryCompositeTest : public HolovizBackendGeometryTest {};

TEST_F(HolovizBackendGeometryCompositeTest, CompositeOrderDecidesTheTopLayer) {
  auto backend = create_holoviz_render_backend(device_, {});
  ASSERT_TRUE(backend.has_value());
  backend_ = std::move(*backend);

  LayerStack bundle;
  bundle.add_geometry_layer("red", GeometryLayerDesc{{GeometryTopology::kRectangleList}, 4, 16.0F});
  bundle.add_geometry_layer("green",
                            GeometryLayerDesc{{GeometryTopology::kRectangleList}, 4, 16.0F});

  RenderConfig config;
  config.headless = true;
  config.headless_width = kSize;
  config.headless_height = kSize;
  config.max_frames_in_flight = 2;
  ASSERT_TRUE(backend_->allocate(bundle, config).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  // The same rect in both layers, fat outline; only the color tells them apart.
  const float coords[] = {0.25F, 0.25F, 0.75F, 0.75F};
  const GeometryBatch red{
      GeometryTopology::kRectangleList, coords, 1, {1.0F, 0.0F, 0.0F, 1.0F}, 16.0F};
  const GeometryBatch green{
      GeometryTopology::kRectangleList, coords, 1, {0.0F, 1.0F, 0.0F, 1.0F}, 16.0F};
  const GeometryFrameData red_data{{&red, 1}};
  const GeometryFrameData green_data{{&green, 1}};

  const auto render_ordered = [&](std::vector<uint32_t> order, std::vector<uint8_t>& pixels) {
    auto cmds = backend_->begin_frame(AcquireToken{}, 0);
    ASSERT_TRUE(cmds.has_value());
    RenderList list;
    list.layers.push_back(LayerFrame{0, &red_data});
    list.layers.push_back(LayerFrame{1, &green_data});
    ASSERT_TRUE(backend_->record(*cmds, list, CompositeSpec{std::move(order)}).has_value());
    auto token = backend_->submit(*cmds, SubmitContext{0});
    ASSERT_TRUE(token.has_value());
    ASSERT_TRUE(device_
                    .wait_on_submit(submit_payload(*token, device_identity(device_)).payload->token,
                                    std::chrono::seconds(5))
                    .has_value());
    auto* concrete = static_cast<HolovizRenderBackend*>(backend_.get());
    pixels.resize(static_cast<size_t>(kSize) * kSize * 4);
    ASSERT_TRUE(device_
                    .download_from_image(concrete->headless_target(0)->get(),
                                         0,
                                         std::as_writable_bytes(std::span(pixels)),
                                         static_cast<size_t>(kSize) * 4)
                    .has_value());
  };

  // On-outline probe pixel: the rect's top edge center.
  const auto probe = [&](const std::vector<uint8_t>& pixels) {
    const size_t i = (static_cast<size_t>(64) * kSize + 128) * 4;
    return std::pair<uint8_t, uint8_t>(pixels[i], pixels[i + 1]);  // (red, green)
  };

  std::vector<uint8_t> green_on_top;
  ASSERT_NO_FATAL_FAILURE(render_ordered({0, 1}, green_on_top));
  EXPECT_GT(probe(green_on_top).second, 200) << "layer 1 (green) should be on top";

  std::vector<uint8_t> red_on_top;
  ASSERT_NO_FATAL_FAILURE(render_ordered({1, 0}, red_on_top));
  EXPECT_GT(probe(red_on_top).first, 200) << "layer 0 (red) should be on top after reorder";
}

}  // namespace
}  // namespace holoscan::visualization::backends
