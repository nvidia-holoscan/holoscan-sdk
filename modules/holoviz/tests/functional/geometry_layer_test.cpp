/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda/cuda_service.hpp>
#include <holoviz/holoviz.hpp>
#include "headless_fixture.hpp"

namespace viz = holoscan::viz;

namespace holoscan::viz {

// define the '<<' operator to get a nice parameter string
std::ostream& operator<<(std::ostream& os, const PrimitiveTopology& topology) {
#define CASE(VALUE)            \
  case VALUE:                  \
    os << std::string(#VALUE); \
    break;

  switch (topology) {
    CASE(viz::PrimitiveTopology::POINT_LIST);
    CASE(viz::PrimitiveTopology::LINE_LIST);
    CASE(viz::PrimitiveTopology::LINE_STRIP);
    CASE(viz::PrimitiveTopology::TRIANGLE_LIST);
    CASE(viz::PrimitiveTopology::CROSS_LIST);
    CASE(viz::PrimitiveTopology::RECTANGLE_LIST);
    CASE(viz::PrimitiveTopology::OVAL_LIST);
    default:
      os.setstate(std::ios_base::failbit);
  }
  return os;

#undef CASE
}

}  // namespace holoscan::viz

// Fixture that initializes Holoviz
class PrimitiveTopology : public TestHeadless,
                          public testing::WithParamInterface<viz::PrimitiveTopology> {};

TEST_P(PrimitiveTopology, Primitive) {
  const viz::PrimitiveTopology topology = GetParam();

  uint32_t crc;
  uint32_t primitive_count;
  std::vector<float> data;
  switch (topology) {
    case viz::PrimitiveTopology::POINT_LIST:
      primitive_count = 1;
      data.push_back(0.5f);
      data.push_back(0.5f);
      crc = 0xE81FD1BB;
      break;
    case viz::PrimitiveTopology::LINE_LIST:
      primitive_count = 2;
      data.push_back(0.1f);
      data.push_back(0.1f);
      data.push_back(0.9f);
      data.push_back(0.9f);

      data.push_back(0.7f);
      data.push_back(0.3f);
      data.push_back(0.2f);
      data.push_back(0.4f);
      crc = 0xF7E63B21;
      break;
    case viz::PrimitiveTopology::LINE_STRIP:
      primitive_count = 2;
      data.push_back(0.1f);
      data.push_back(0.1f);
      data.push_back(0.7f);
      data.push_back(0.9f);

      data.push_back(0.3f);
      data.push_back(0.2f);
      crc = 0x392E35D8;
      break;
    case viz::PrimitiveTopology::TRIANGLE_LIST:
      primitive_count = 2;
      data.push_back(0.1f);
      data.push_back(0.1f);
      data.push_back(0.5f);
      data.push_back(0.9f);
      data.push_back(0.9f);
      data.push_back(0.1f);

      data.push_back(0.05f);
      data.push_back(0.7f);
      data.push_back(0.15f);
      data.push_back(0.8f);
      data.push_back(0.25f);
      data.push_back(0.6f);
      crc = 0xB29BAA37;
      break;
    case viz::PrimitiveTopology::CROSS_LIST:
      primitive_count = 2;
      data.push_back(0.5f);
      data.push_back(0.5f);
      data.push_back(0.1f);

      data.push_back(0.1f);
      data.push_back(0.3f);
      data.push_back(0.01f);
      crc = 0xa32f4dcb;
      break;
    case viz::PrimitiveTopology::RECTANGLE_LIST:
      primitive_count = 2;
      data.push_back(0.1f);
      data.push_back(0.1f);
      data.push_back(0.9f);
      data.push_back(0.9f);

      data.push_back(0.3f);
      data.push_back(0.2f);
      data.push_back(0.5f);
      data.push_back(0.3f);
      crc = 0x355A2C00;
      break;
    case viz::PrimitiveTopology::OVAL_LIST:
      primitive_count = 2;
      data.push_back(0.5f);
      data.push_back(0.5f);
      data.push_back(0.2f);
      data.push_back(0.1f);

      data.push_back(0.6f);
      data.push_back(0.4f);
      data.push_back(0.05f);
      data.push_back(0.07f);
      crc = 0xA907614F;
      break;
    default:
      EXPECT_TRUE(false) << "Unhandled primitive topoplogy";
  }

  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginGeometryLayer());

  for (uint32_t i = 0; i < 3; ++i) {
    if (i == 1) {
      EXPECT_NO_THROW(viz::Color(1.f, 0.5f, 0.25f, 0.75f));
    } else if (i == 2) {
      EXPECT_NO_THROW(viz::PointSize(4.f));
      EXPECT_NO_THROW(viz::LineWidth(3.f));
    }

    EXPECT_NO_THROW(viz::Primitive(topology, primitive_count, data.size(), data.data()));

    for (auto&& item : data) { item += 0.1f; }
  }
  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());

  CompareResultCRC32({crc});
}

INSTANTIATE_TEST_SUITE_P(
    GeometryLayer, PrimitiveTopology,
    testing::Values(viz::PrimitiveTopology::POINT_LIST, viz::PrimitiveTopology::LINE_LIST,
                    viz::PrimitiveTopology::LINE_STRIP, viz::PrimitiveTopology::TRIANGLE_LIST,
                    viz::PrimitiveTopology::CROSS_LIST, viz::PrimitiveTopology::RECTANGLE_LIST,
                    viz::PrimitiveTopology::OVAL_LIST));

// Fixture that initializes Holoviz
class GeometryLayer : public TestHeadless {};

TEST_F(GeometryLayer, Text) {
  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginGeometryLayer());
  EXPECT_NO_THROW(viz::Text(0.4f, 0.4f, 0.4f, "Text"));
  EXPECT_NO_THROW(viz::Color(0.5f, 0.9f, 0.7f, 0.9f));
  EXPECT_NO_THROW(viz::Text(0.1f, 0.1f, 0.2f, "Colored"));
  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());

  CompareResultCRC32({0xc68d7716});
}

TEST_F(GeometryLayer, TextClipped) {
  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginGeometryLayer());
  EXPECT_NO_THROW(viz::Text(1.1f, 0.4f, 0.4f, "Text"));

  EXPECT_NO_THROW(viz::End());

  CompareResultCRC32({0x8a9c008});
}

class GeometryLayerWithFont : public TestHeadless {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(viz::SetFont("../modules/holoviz/src/fonts/Roboto-Bold.ttf", 12.f));

    // call base class
    ::TestHeadless::SetUp();
  }

  void TearDown() override {
    // call base class
    ::TestHeadless::TearDown();

    ASSERT_NO_THROW(viz::SetFont("", 0.f));
  }
};

TEST_F(GeometryLayerWithFont, Text) {
  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginGeometryLayer());
  EXPECT_NO_THROW(viz::Text(0.1f, 0.1f, 0.7f, "Font"));
  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());

  CompareResultCRC32({0xbccffe56});
}

// Fixture that initializes Holoviz
class DepthMapRenderMode : public TestHeadless,
                           public testing::WithParamInterface<viz::DepthMapRenderMode> {};

TEST_P(DepthMapRenderMode, DepthMap) {
  const viz::DepthMapRenderMode depth_map_render_mode = GetParam();
  const uint32_t map_width = 8;
  const uint32_t map_height = 8;

  // allocate device memory
  viz::CudaService::ScopedPush cuda_context = viz::CudaService::get().PushContext();

  viz::UniqueCUdeviceptr depth_ptr;
  depth_ptr.reset([this] {
    CUdeviceptr device_ptr;
    EXPECT_EQ(cuMemAlloc(&device_ptr, map_width * map_height * sizeof(uint8_t)), CUDA_SUCCESS);
    return device_ptr;
  }());
  std::vector<uint8_t> depth_data(map_width * map_height);
  for (size_t index = 0; index < depth_data.size(); ++index) { depth_data[index] = index * 4; }
  EXPECT_EQ(cuMemcpyHtoD(depth_ptr.get(), depth_data.data(), depth_data.size()), CUDA_SUCCESS);

  viz::UniqueCUdeviceptr color_ptr;
  color_ptr.reset([this] {
    CUdeviceptr device_ptr;
    EXPECT_EQ(cuMemAlloc(&device_ptr, map_width * map_height * sizeof(uint32_t)), CUDA_SUCCESS);
    return device_ptr;
  }());
  std::vector<uint32_t> color_data(map_width * map_height);
  for (uint32_t index = 0; index < color_data.size(); ++index) {
    color_data[index] = (index << 18) | (index << 9) | index | 0xFF000000;
  }
  EXPECT_EQ(cuMemcpyHtoD(color_ptr.get(), color_data.data(), color_data.size() * sizeof(uint32_t)),
            CUDA_SUCCESS);

  uint32_t crc;
  switch (depth_map_render_mode) {
    case viz::DepthMapRenderMode::POINTS:
      crc = 0x1eb98bfa;
      break;
    case viz::DepthMapRenderMode::LINES:
      crc = 0xbf3be45a;
      break;
    case viz::DepthMapRenderMode::TRIANGLES:
      crc = 0x5ac3bd4b;
      break;
  }
  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginGeometryLayer());
  EXPECT_NO_THROW(viz::DepthMap(depth_map_render_mode,
                                map_width,
                                map_height,
                                viz::ImageFormat::R8_UNORM,
                                depth_ptr.get(),
                                viz::ImageFormat::R8G8B8A8_UNORM,
                                color_ptr.get()));
  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());

  CompareResultCRC32({crc});
}

INSTANTIATE_TEST_SUITE_P(GeometryLayer, DepthMapRenderMode,
                         testing::Values(viz::DepthMapRenderMode::POINTS,
                                         viz::DepthMapRenderMode::LINES,
                                         viz::DepthMapRenderMode::TRIANGLES));

TEST_F(GeometryLayer, Reuse) {
  std::vector<float> data{0.5f, 0.5f};

  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_NO_THROW(viz::Begin());

    EXPECT_NO_THROW(viz::BeginGeometryLayer());
    EXPECT_NO_THROW(viz::Color(0.1f, 0.2f, 0.3f, 0.4f));
    EXPECT_NO_THROW(viz::LineWidth(2.f));
    EXPECT_NO_THROW(viz::PointSize(3.f));
    EXPECT_NO_THROW(
        viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size(), data.data()));
    EXPECT_NO_THROW(viz::Text(0.4f, 0.4f, 0.1f, "Text"));
    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::End());
  }
}

TEST_F(GeometryLayer, Errors) {
  std::vector<float> data{0.5f, 0.5f};

  EXPECT_NO_THROW(viz::Begin());

  // it's an error to call geometry functions without calling BeginGeometryLayer first
  EXPECT_THROW(viz::Color(0.f, 0.f, 0.f, 1.f), std::runtime_error);
  EXPECT_THROW(viz::LineWidth(1.0f), std::runtime_error);
  EXPECT_THROW(viz::PointSize(1.0f), std::runtime_error);
  EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size(), data.data()),
               std::runtime_error);
  EXPECT_THROW(viz::Text(0.5f, 0.5f, 0.1f, "Text"), std::runtime_error);

  // it's an error to call BeginGeometryLayer again without calling EndLayer
  EXPECT_NO_THROW(viz::BeginGeometryLayer());
  EXPECT_THROW(viz::BeginGeometryLayer(), std::runtime_error);
  EXPECT_NO_THROW(viz::EndLayer());

  // it's an error to call geometry functions when a different layer is active
  EXPECT_NO_THROW(viz::BeginImageLayer());
  EXPECT_THROW(viz::Color(0.f, 0.f, 0.f, 1.f), std::runtime_error);
  EXPECT_THROW(viz::LineWidth(1.0f), std::runtime_error);
  EXPECT_THROW(viz::PointSize(1.0f), std::runtime_error);
  EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size(), data.data()),
               std::runtime_error);
  EXPECT_THROW(viz::Text(0.5f, 0.5f, 0.1f, "Text"), std::runtime_error);
  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::BeginGeometryLayer());

  // Primitive function errors, first call the passing function
  EXPECT_NO_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size(), data.data()));
  // it's an error to call Primitive with a primitive count of zero
  EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 0, data.size(), data.data()),
               std::invalid_argument);
  // it's an error to call Primitive with a data size of zero
  EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, 0, data.data()),
               std::invalid_argument);
  // it's an error to call Primitive with a null data pointer
  EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size(), nullptr),
               std::invalid_argument);
  // it's an error to call Primitive with a data size which is too small for the primitive count
  EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size() - 1, data.data()),
               std::runtime_error);

  // Text function errors, first call the passing function
  EXPECT_NO_THROW(viz::Text(0.5f, 0.5f, 0.1f, "Text"));
  // it's an error to call Text with a size of zero
  EXPECT_THROW(viz::Text(0.5f, 0.5f, 0.0f, "Text"), std::invalid_argument);
  // it's an error to call Text with null text pointer
  EXPECT_THROW(viz::Text(0.5f, 0.5f, 0.1f, nullptr), std::invalid_argument);

  // Depth map function errors, first call the passing function
  const uint32_t map_width = 8;
  const uint32_t map_height = 8;

  // allocate device memory
  viz::CudaService::ScopedPush cuda_context = viz::CudaService::get().PushContext();
  viz::UniqueCUdeviceptr depth_ptr;
  depth_ptr.reset([this] {
    CUdeviceptr device_ptr;
    EXPECT_EQ(cuMemAlloc(&device_ptr, map_width * map_height * sizeof(uint8_t)), CUDA_SUCCESS);
    return device_ptr;
  }());
  viz::UniqueCUdeviceptr color_ptr;
  color_ptr.reset([this] {
    CUdeviceptr device_ptr;
    EXPECT_EQ(cuMemAlloc(&device_ptr, map_width * map_height * sizeof(uint32_t)), CUDA_SUCCESS);
    return device_ptr;
  }());

  // First call the passing function
  EXPECT_NO_THROW(viz::DepthMap(viz::DepthMapRenderMode::POINTS,
                                map_width,
                                map_height,
                                viz::ImageFormat::R8_UNORM,
                                depth_ptr.get(),
                                viz::ImageFormat::R8G8B8A8_UNORM,
                                color_ptr.get()));
  // it's an error to call DepthMap with a width of zero
  EXPECT_THROW(viz::DepthMap(viz::DepthMapRenderMode::POINTS,
                             0,
                             map_height,
                             viz::ImageFormat::R8_UNORM,
                             depth_ptr.get(),
                             viz::ImageFormat::R8G8B8A8_UNORM,
                             color_ptr.get()),
               std::invalid_argument);
  // it's an error to call DepthMap with a width of zero
  EXPECT_THROW(viz::DepthMap(viz::DepthMapRenderMode::POINTS,
                             map_width,
                             0,
                             viz::ImageFormat::R8_UNORM,
                             depth_ptr.get(),
                             viz::ImageFormat::R8G8B8A8_UNORM,
                             color_ptr.get()),
               std::invalid_argument);
  // it's an error to call DepthMap with a depth format other than viz::ImageFormat::R8_UINT
  EXPECT_THROW(viz::DepthMap(viz::DepthMapRenderMode::POINTS,
                             map_width,
                             map_height,
                             viz::ImageFormat::R16_UNORM,
                             depth_ptr.get(),
                             viz::ImageFormat::R8G8B8A8_UNORM,
                             color_ptr.get()),
               std::invalid_argument);
  // it's an error to call DepthMap with no depth map pointer
  EXPECT_THROW(viz::DepthMap(viz::DepthMapRenderMode::POINTS,
                             map_width,
                             map_height,
                             viz::ImageFormat::R8_UNORM,
                             0,
                             viz::ImageFormat::R8G8B8A8_UNORM,
                             color_ptr.get()),
               std::invalid_argument);

  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());
}
