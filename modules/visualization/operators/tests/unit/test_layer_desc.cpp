// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstring>
#include <variant>

#include <gtest/gtest.h>

#include <holoscan/visualization/holoviz_op.hpp>

namespace holoscan::visualization {

// Walks the declaration model against a reference app shape (endoscopy: one video layer +
// one geometry overlay layer).
TEST(LayerStackTest, DeclaresEndoscopyReferenceShape) {
  LayerStack stack;

  stack.add_image_layer("video_in",
                        ImageLayerDesc{ImageFormat::kNV12,
                                       1920,
                                       1080,
                                       ColorSpace::kPassThrough,
                                       YUVModelConversion::kBT709,
                                       YUVRange::kLimited,
                                       std::nullopt});
  stack.add_geometry_layer("detections",
                           GeometryLayerDesc{{GeometryTopology::kRectangleList}, 256, 1.0f});

  ASSERT_EQ(stack.layers().size(), 2u);

  const auto& video = stack.layers()[0];
  EXPECT_EQ(video.port_name, "video_in");
  ASSERT_TRUE(std::holds_alternative<ImageLayerDesc>(video.desc));
  EXPECT_EQ(std::get<ImageLayerDesc>(video.desc).max_width, 1920u);
  EXPECT_EQ(std::get<ImageLayerDesc>(video.desc).max_height, 1080u);

  const auto& detections = stack.layers()[1];
  ASSERT_TRUE(std::holds_alternative<GeometryLayerDesc>(detections.desc));
  EXPECT_EQ(std::get<GeometryLayerDesc>(detections.desc).max_primitives, 256u);
}

TEST(LayerStackTest, PreservesDeclarationOrderAsDefaultCompositeOrder) {
  LayerStack stack;
  stack.add_text_layer("label", TextLayerDesc{64});
  stack.add_lut_layer("mask_lut", LUTLayerDesc{256});

  ASSERT_EQ(stack.layers().size(), 2u);
  EXPECT_TRUE(std::holds_alternative<TextLayerDesc>(stack.layers()[0].desc));
  EXPECT_TRUE(std::holds_alternative<LUTLayerDesc>(stack.layers()[1].desc));
}

// The text layer's row header is a wire format: its size and field offsets are the contract a
// producer packs against, independent of the compiler laying it out.
TEST(TextRunHeaderTest, IsSixteenPackedBytesInWireOrder) {
  static_assert(kTextRunHeaderBytes == 16U);
  EXPECT_EQ(sizeof(TextRunHeader), 16U);
  EXPECT_EQ(offsetof(TextRunHeader, x), 0U);
  EXPECT_EQ(offsetof(TextRunHeader, y), 4U);
  EXPECT_EQ(offsetof(TextRunHeader, size_px), 8U);
  EXPECT_EQ(offsetof(TextRunHeader, color), 12U);

  const TextRunHeader header{.x = 0.25f, .y = 0.5f, .size_px = 24.0f, .color = 0xFF00D4FFu};
  unsigned char bytes[16];
  std::memcpy(bytes, &header, sizeof(bytes));
  TextRunHeader round_trip{};
  std::memcpy(&round_trip, bytes, sizeof(round_trip));
  EXPECT_EQ(round_trip.x, 0.25f);
  EXPECT_EQ(round_trip.y, 0.5f);
  EXPECT_EQ(round_trip.size_px, 24.0f);
  EXPECT_EQ(round_trip.color, 0xFF00D4FFu);
  // Defaults: white, 16 px, top-left origin.
  const TextRunHeader defaults{};
  EXPECT_EQ(defaults.color, 0xFFFFFFFFu);
  EXPECT_EQ(defaults.size_px, 16.0f);
}

// The position unit is a layer-level, CPU-side choice: the record bytes never change. Pixels
// are the default; the normalized [0,1] interpretation is an opt-in on the descriptor, and the
// positional form `TextLayerDesc{n}` keeps meaning "n glyphs, pixel positions".
TEST(TextLayerDescTest, PositionsArePixelsUnlessNormalizedIsRequested) {
  EXPECT_FALSE(TextLayerDesc{}.normalized_coordinates);
  const TextLayerDesc positional{64};
  EXPECT_EQ(positional.max_glyphs, 64u);
  EXPECT_FALSE(positional.normalized_coordinates);
  const TextLayerDesc normalized{.max_glyphs = 64, .normalized_coordinates = true};
  EXPECT_TRUE(normalized.normalized_coordinates);
}

}  // namespace holoscan::visualization
