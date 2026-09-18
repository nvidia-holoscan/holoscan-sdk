// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

}  // namespace holoscan::visualization
