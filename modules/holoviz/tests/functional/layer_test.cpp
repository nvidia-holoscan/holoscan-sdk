/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoviz/holoviz.hpp>
#include "headless_fixture.hpp"

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
        uint8_t((static_cast<float>(data_[index]) / 255.f * opacity) * 255.f + 0.5f);
  }

  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginImageLayer());

  EXPECT_NO_THROW(viz::LayerOpacity(opacity));

  EXPECT_NO_THROW(viz::ImageHost(
      width_, height_, viz::ImageFormat::R8G8B8A8_UNORM, reinterpret_cast<void*>(data_.data())));

  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());

  std::swap(data_, data_with_opacity);
  CompareResult();
  std::swap(data_with_opacity, data_);
}

TEST_F(Layer, Priority) {
  const viz::ImageFormat kFormat = viz::ImageFormat::R8G8B8A8_UNORM;

  const uint32_t red = 0xFF0000FF;
  const uint32_t green = 0xFF00FF00;

  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_NO_THROW(viz::Begin());

    EXPECT_NO_THROW(viz::BeginImageLayer());
    EXPECT_NO_THROW(viz::LayerPriority(i));
    EXPECT_NO_THROW(viz::ImageHost(
        1, 1, viz::ImageFormat::R8G8B8A8_UNORM, reinterpret_cast<const void*>(&red)));
    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::BeginImageLayer());
    EXPECT_NO_THROW(viz::LayerPriority(1 - i));
    EXPECT_NO_THROW(viz::ImageHost(
        1, 1, viz::ImageFormat::R8G8B8A8_UNORM, reinterpret_cast<const void*>(&green)));
    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::End());

    std::vector<uint8_t> read_data;
    ReadData(read_data);
    EXPECT_EQ(reinterpret_cast<uint32_t*>(read_data.data())[0], (i == 0) ? green : red);
  }
}

TEST_F(Layer, Errors) {
  // it's an error to call EndLayer without an active layer
  EXPECT_THROW(viz::EndLayer(), std::runtime_error);

  // it's an error to call layer functions without an active layer
  EXPECT_THROW(viz::LayerOpacity(1.f), std::runtime_error);
  EXPECT_THROW(viz::LayerPriority(0), std::runtime_error);
}
