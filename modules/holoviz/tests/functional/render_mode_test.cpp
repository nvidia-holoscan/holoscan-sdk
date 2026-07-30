/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) test code

#include <gtest/gtest.h>

#include <vector>

#include <holoviz/holoviz.hpp>
#include "test_fixture.hpp"

namespace viz = holoscan::viz;

class RenderMode : public TestHeadless {};

TEST_F(RenderMode, DontClearColor) {
  const uint8_t red[] = {255, 0, 0, 255};
  const uint8_t black[] = {0, 0, 0, 0};

  // fill with red
  EXPECT_NO_THROW(viz::Begin());
  EXPECT_NO_THROW(viz::BeginImageLayer());
  EXPECT_NO_THROW(viz::ImageHost(1, 1, viz::ImageFormat::R8G8B8A8_UNORM, red));
  EXPECT_NO_THROW(viz::EndLayer());
  EXPECT_NO_THROW(viz::End());

  // don't clear color and don't swap the framebuffer
  EXPECT_NO_THROW(
      viz::Begin(viz::RenderFlags::DONT_CLEAR_COLOR | viz::RenderFlags::DONT_SWAP_BUFFERS));
  EXPECT_NO_THROW(viz::End());

  std::vector<uint8_t> read_color_data;

  // image should be still red (not cleared)
  ReadColorData(read_color_data);
  EXPECT_EQ(read_color_data[0], red[0]);
  EXPECT_EQ(read_color_data[1], red[1]);
  EXPECT_EQ(read_color_data[2], red[2]);
  EXPECT_EQ(read_color_data[3], red[3]);

  // clear color
  EXPECT_NO_THROW(viz::Begin(viz::RenderFlags::DONT_SWAP_BUFFERS));
  EXPECT_NO_THROW(viz::End());

  // image should be black (cleared)
  ReadColorData(read_color_data);
  EXPECT_EQ(read_color_data[0], black[0]);
  EXPECT_EQ(read_color_data[1], black[1]);
  EXPECT_EQ(read_color_data[2], black[2]);
  EXPECT_EQ(read_color_data[3], black[3]);
}

TEST_F(RenderMode, DontClearDepth) {
  const uint8_t red[] = {255, 0, 0, 255};
  const uint8_t black[] = {0, 0, 0, 0};
  const float depth[] = {.5f};

  // fill with red
  EXPECT_NO_THROW(viz::Begin());
  EXPECT_NO_THROW(viz::BeginImageLayer());
  EXPECT_NO_THROW(viz::ImageHost(1, 1, viz::ImageFormat::R8G8B8A8_UNORM, red));
  EXPECT_NO_THROW(viz::ImageHost(1, 1, viz::ImageFormat::D32_SFLOAT, depth));
  EXPECT_NO_THROW(viz::EndLayer());
  EXPECT_NO_THROW(viz::End());

  // don't clear depth and don't swap the framebuffer
  EXPECT_NO_THROW(
      viz::Begin(viz::RenderFlags::DONT_CLEAR_DEPTH | viz::RenderFlags::DONT_SWAP_BUFFERS));
  EXPECT_NO_THROW(viz::End());

  std::vector<float> read_depth_data;
  std::vector<uint8_t> read_color_data;
  // depth should be still the same (not cleared)
  ReadDepthData(read_depth_data);
  EXPECT_EQ(read_depth_data[0], depth[0]);
  // color should be cleared
  ReadColorData(read_color_data);
  EXPECT_EQ(read_color_data[0], black[0]);
  EXPECT_EQ(read_color_data[1], black[1]);
  EXPECT_EQ(read_color_data[2], black[2]);
  EXPECT_EQ(read_color_data[3], black[3]);

  // clear depth
  EXPECT_NO_THROW(viz::Begin(viz::RenderFlags::DONT_SWAP_BUFFERS));
  EXPECT_NO_THROW(viz::End());

  // image should be 1.0f (cleared to default depth clear value)
  ReadDepthData(read_depth_data);
  EXPECT_EQ(read_depth_data[0], 1.f);
}

TEST_F(RenderMode, DontPresent) {
  const uint8_t red[] = {255, 0, 0, 255};
  const uint8_t black[] = {0, 0, 0, 0};
  const float depth[] = {.5f};

  // fill with red
  EXPECT_NO_THROW(viz::Begin());
  EXPECT_NO_THROW(viz::BeginImageLayer());
  EXPECT_NO_THROW(viz::ImageHost(1, 1, viz::ImageFormat::R8G8B8A8_UNORM, red));
  EXPECT_NO_THROW(viz::ImageHost(1, 1, viz::ImageFormat::D32_SFLOAT, depth));
  EXPECT_NO_THROW(viz::EndLayer());
  EXPECT_NO_THROW(viz::End());

  std::vector<float> read_depth_data;
  std::vector<uint8_t> read_color_data;

  ReadDepthData(read_depth_data);
  EXPECT_EQ(read_depth_data[0], depth[0]);
  ReadColorData(read_color_data);
  EXPECT_EQ(read_color_data[0], red[0]);
  EXPECT_EQ(read_color_data[1], red[1]);
  EXPECT_EQ(read_color_data[2], red[2]);
  EXPECT_EQ(read_color_data[3], red[3]);

  // don't present (to avoid a swap of the framebuffer)
  EXPECT_NO_THROW(viz::Begin(viz::RenderFlags::DONT_CLEAR_DEPTH |
                             viz::RenderFlags::DONT_CLEAR_COLOR |
                             viz::RenderFlags::DONT_SWAP_BUFFERS));
  EXPECT_NO_THROW(viz::End());

  // depth should be still the same (not cleared and not swapped)
  ReadDepthData(read_depth_data);
  EXPECT_EQ(read_depth_data[0], depth[0]);
  // color should be still the same (not cleared and not swapped)
  ReadColorData(read_color_data);
  EXPECT_EQ(read_color_data[0], red[0]);
  EXPECT_EQ(read_color_data[1], red[1]);
  EXPECT_EQ(read_color_data[2], red[2]);
  EXPECT_EQ(read_color_data[3], red[3]);

  // swap buffers but don't clear depth or color
  EXPECT_NO_THROW(
      viz::Begin(viz::RenderFlags::DONT_CLEAR_DEPTH | viz::RenderFlags::DONT_CLEAR_COLOR));
  EXPECT_NO_THROW(viz::End());

  // depth should be default-initialized
  ReadDepthData(read_depth_data);
  EXPECT_EQ(read_depth_data[0], 0.f);
  // color should be default-initialized
  ReadColorData(read_color_data);
  EXPECT_EQ(read_color_data[0], black[0]);
  EXPECT_EQ(read_color_data[1], black[1]);
  EXPECT_EQ(read_color_data[2], black[2]);
  EXPECT_EQ(read_color_data[3], black[3]);
}

// NOLINTEND(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
