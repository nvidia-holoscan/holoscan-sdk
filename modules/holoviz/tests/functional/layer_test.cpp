/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include <holoviz/holoviz.hpp>
#include "test_fixture.hpp"

namespace viz = holoscan::viz;

class Layer : public TestHeadless {};

TEST_F(Layer, Opacity) {
  const viz::ImageFormat kFormat = viz::ImageFormat::R8G8B8A8_UNORM;
  const float opacity = 0.4f;

  SetupData(kFormat);

  std::vector<uint8_t> data_with_opacity;

  data_with_opacity.resize(width_ * height_ * sizeof(uint32_t));
  for (size_t index = 0; index < width_ * height_ * 4; ++index) {
    data_with_opacity.data()[index] =
        uint8_t((static_cast<float>(color_data_[index]) / 255.f * opacity) * 255.f + 0.5f);
  }

  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginImageLayer());

  EXPECT_NO_THROW(viz::LayerOpacity(opacity));

  EXPECT_NO_THROW(viz::ImageHost(width_,
                                 height_,
                                 viz::ImageFormat::R8G8B8A8_UNORM,
                                 reinterpret_cast<void*>(color_data_.data())));

  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());

  std::swap(color_data_, data_with_opacity);
  CompareColorResult();
  std::swap(data_with_opacity, color_data_);
}

TEST_F(Layer, Priority) {
  const viz::ImageFormat kFormat = viz::ImageFormat::R8G8B8A8_UNORM;

  const uint32_t red = 0xFF0000FF;
  const uint32_t green = 0xFF00FF00;

  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_NO_THROW(viz::Begin());

    EXPECT_NO_THROW(viz::BeginImageLayer());
    EXPECT_NO_THROW(viz::LayerPriority(i));
    EXPECT_NO_THROW(viz::ImageHost(1, 1, kFormat, reinterpret_cast<const void*>(&red)));
    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::BeginImageLayer());
    EXPECT_NO_THROW(viz::LayerPriority(1 - i));
    EXPECT_NO_THROW(viz::ImageHost(1, 1, kFormat, reinterpret_cast<const void*>(&green)));
    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::End());

    std::vector<uint8_t> color_data;
    ReadColorData(color_data);
    EXPECT_EQ(reinterpret_cast<uint32_t*>(color_data.data())[0], (i == 0) ? green : red);

    std::vector<float> depth_data;
    ReadDepthData(depth_data);
    EXPECT_EQ(depth_data.data()[0], 0.f);
  }
}

TEST_F(Layer, View) {
  const viz::ImageFormat kFormat = viz::ImageFormat::R8G8B8A8_UNORM;

  const uint32_t red = 0xFF0000FF;
  const uint32_t green = 0xFF00FF00;
  const uint32_t blue = 0xFFFF0000;

  EXPECT_NO_THROW(viz::Begin());

  // top left - red image
  EXPECT_NO_THROW(viz::BeginImageLayer());
  EXPECT_NO_THROW(viz::LayerAddView(0.f, 0.f, 0.5f, 0.5f));
  EXPECT_NO_THROW(viz::ImageHost(1, 1, kFormat, reinterpret_cast<const void*>(&red)));
  EXPECT_NO_THROW(viz::EndLayer());

  // top right - green image
  EXPECT_NO_THROW(viz::BeginImageLayer());
  EXPECT_NO_THROW(viz::LayerAddView(0.5f, 0.0f, 0.5f, 0.5f));
  EXPECT_NO_THROW(viz::ImageHost(1, 1, kFormat, reinterpret_cast<const void*>(&green)));
  EXPECT_NO_THROW(viz::EndLayer());

  // two views
  // - bottom left - blue triangles
  // - bottom right, half size - blue triangles
  constexpr uint32_t triangles = 2;
  std::array<float, triangles * 3 * 2> data{
      0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f};
  EXPECT_NO_THROW(viz::BeginGeometryLayer());
  EXPECT_NO_THROW(viz::LayerAddView(0.f, 0.5f, 0.5f, .5f));
  EXPECT_NO_THROW(viz::LayerAddView(0.625f, 0.625f, 0.25f, .25f));
  EXPECT_NO_THROW(viz::Color(0.f, 0.f, 1.f, 1.f));
  EXPECT_NO_THROW(
      viz::Primitive(viz::PrimitiveTopology::TRIANGLE_LIST, triangles, data.size(), data.data()));
  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());

  color_data_.resize(width_ * height_ * sizeof(uint32_t));
  uint32_t* cur_color_data = reinterpret_cast<uint32_t*>(color_data_.data());
  for (uint32_t y = 0; y < height_; ++y) {
    for (uint32_t x = 0; x < width_; ++x) {
      uint32_t color = 0;
      if ((x < width_ / 2) && (y < height_ / 2)) { color = red; }
      if ((x >= width_ / 2) && (y < height_ / 2)) { color = green; }
      if ((x < width_ / 2) && (y >= height_ / 2)) { color = blue; }
      if ((x >= width_ / 2 + width_ / 8) && (y >= height_ / 2 + height_ / 8) &&
          (x < width_ - width_ / 8) && (y < height_ - height_ / 8)) {
        color = blue;
      }
      *cur_color_data = color;
      ++cur_color_data;
    }
  }

  CompareColorResult();
}

TEST_F(Layer, Errors) {
  // it's an error to call EndLayer without an active layer
  EXPECT_THROW(viz::EndLayer(), std::runtime_error);

  // it's an error to call layer functions without an active layer
  EXPECT_THROW(viz::LayerOpacity(1.f), std::runtime_error);
  EXPECT_THROW(viz::LayerPriority(0), std::runtime_error);
  EXPECT_THROW(viz::LayerAddView(0.f, 0.f, 1.f, 1.f), std::runtime_error);

  EXPECT_NO_THROW(viz::Begin());
  EXPECT_NO_THROW(viz::BeginImageLayer());

  // passing case
  EXPECT_NO_THROW(viz::LayerOpacity(1.0f));
  // it's an error to set negative opacity
  EXPECT_THROW(viz::LayerOpacity(-0.1f), std::invalid_argument);
  // it's an error to set opacity higher than 1.0
  EXPECT_THROW(viz::LayerOpacity(1.1f), std::invalid_argument);

  // passing case
  EXPECT_NO_THROW(viz::LayerAddView(0.f, 0.f, 1.f, 1.f));
  // it's an error to add a layer view with zero width
  EXPECT_THROW(viz::LayerAddView(0.f, 0.f, 0.f, 1.f), std::invalid_argument);
  // it's an error to add a layer view with zero height
  EXPECT_THROW(viz::LayerAddView(0.f, 0.f, 1.f, 0.f), std::invalid_argument);

  EXPECT_NO_THROW(viz::EndLayer());
  EXPECT_NO_THROW(viz::End());
}
