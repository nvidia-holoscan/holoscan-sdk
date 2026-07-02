/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <vulkan/format_util.hpp>

using namespace holoscan::viz;

TEST(FormatUtil, ReportsRepresentativeBitDepths) {
  EXPECT_EQ(format_bit_depth(ImageFormat::R8G8B8A8_UNORM), 8);
  EXPECT_EQ(format_bit_depth(ImageFormat::B8G8R8A8_UNORM), 8);
  EXPECT_EQ(format_bit_depth(ImageFormat::A8B8G8R8_UNORM_PACK32), 8);

  EXPECT_EQ(format_bit_depth(ImageFormat::R8G8B8A8_SRGB), 8);
  EXPECT_EQ(format_bit_depth(ImageFormat::B8G8R8A8_SRGB), 8);
  EXPECT_EQ(format_bit_depth(ImageFormat::A8B8G8R8_SRGB_PACK32), 8);

  EXPECT_EQ(format_bit_depth(ImageFormat::A2B10G10R10_UNORM_PACK32), 10);
  EXPECT_EQ(format_bit_depth(ImageFormat::A2R10G10B10_UNORM_PACK32), 10);

  EXPECT_EQ(format_bit_depth(ImageFormat::R16G16B16A16_UNORM), 16);
  EXPECT_EQ(format_bit_depth(ImageFormat::R16G16B16A16_SFLOAT), 16);

  EXPECT_EQ(format_bit_depth(ImageFormat::R32G32B32A32_SFLOAT), 32);
}

TEST(FormatUtil, IdentifiesWhenOutputHasMoreBitDepthThanFramebuffer) {
  EXPECT_GT(format_bit_depth(ImageFormat::A2B10G10R10_UNORM_PACK32),
            format_bit_depth(ImageFormat::R8G8B8A8_UNORM));

  EXPECT_GT(format_bit_depth(ImageFormat::A2B10G10R10_UNORM_PACK32),
            format_bit_depth(ImageFormat::A8B8G8R8_UNORM_PACK32));

  EXPECT_GT(format_bit_depth(ImageFormat::A2R10G10B10_UNORM_PACK32),
            format_bit_depth(ImageFormat::A8B8G8R8_SRGB_PACK32));

  EXPECT_GT(format_bit_depth(ImageFormat::A2R10G10B10_UNORM_PACK32),
            format_bit_depth(ImageFormat::B8G8R8A8_SRGB));

  EXPECT_GT(format_bit_depth(ImageFormat::A2B10G10R10_UNORM_PACK32),
            format_bit_depth(ImageFormat::R8G8B8A8_SRGB));

  EXPECT_GT(format_bit_depth(ImageFormat::R16G16B16A16_UNORM),
            format_bit_depth(ImageFormat::B8G8R8A8_UNORM));

  EXPECT_GT(format_bit_depth(ImageFormat::R32G32B32A32_SFLOAT),
            format_bit_depth(ImageFormat::A2B10G10R10_UNORM_PACK32));
}

TEST(FormatUtil, IdentifiesWhenOutputDoesNotHaveMoreBitDepthThanFramebuffer) {
  EXPECT_LE(format_bit_depth(ImageFormat::R8G8B8A8_UNORM),
            format_bit_depth(ImageFormat::A2B10G10R10_UNORM_PACK32));

  EXPECT_LE(format_bit_depth(ImageFormat::B8G8R8A8_UNORM),
            format_bit_depth(ImageFormat::R8G8B8A8_UNORM));

  EXPECT_LE(format_bit_depth(ImageFormat::A2R10G10B10_UNORM_PACK32),
            format_bit_depth(ImageFormat::R16G16B16A16_UNORM));
}
