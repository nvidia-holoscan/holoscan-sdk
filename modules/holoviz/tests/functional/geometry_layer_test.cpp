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

#include <string>
#include <tuple>
#include <vector>

#include <cuda/cuda_service.hpp>
#include <holoviz/holoviz.hpp>
#include "test_fixture.hpp"

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
    CASE(viz::PrimitiveTopology::POINT_LIST_3D);
    CASE(viz::PrimitiveTopology::LINE_LIST_3D);
    CASE(viz::PrimitiveTopology::LINE_STRIP_3D);
    CASE(viz::PrimitiveTopology::TRIANGLE_LIST_3D);
    default:
      os.setstate(std::ios_base::failbit);
  }
  return os;

#undef CASE
}

}  // namespace holoscan::viz

enum class Source { HOST, CUDA_DEVICE };

// Fixture that initializes Holoviz
class PrimitiveTopology
    : public TestHeadless,
      public testing::WithParamInterface<std::tuple<viz::PrimitiveTopology, Source>> {};

TEST_P(PrimitiveTopology, Primitive) {
  const viz::PrimitiveTopology topology = std::get<0>(GetParam());
  const Source source = std::get<1>(GetParam());

  std::vector<uint32_t> color_crc, depth_crc;
  uint32_t primitive_count;
  std::vector<float> data;
  switch (topology) {
    case viz::PrimitiveTopology::POINT_LIST:
      primitive_count = 1;
      data.push_back(0.5F);
      data.push_back(0.5F);
      color_crc = {0x3088e839};
      depth_crc = {0x748e4c96};
      break;
    case viz::PrimitiveTopology::LINE_LIST:
      primitive_count = 2;
      data.push_back(0.1F);
      data.push_back(0.1F);
      data.push_back(0.9F);
      data.push_back(0.9F);

      data.push_back(0.7F);
      data.push_back(0.3F);
      data.push_back(0.2F);
      data.push_back(0.4F);
      color_crc = {
          0xe96c7246,  // Quadro
          0x5f7bf4d3   // non-Quadro
      };
      depth_crc = {
          0x802dbbb0,  // Quadro
          0xbd6bedea   // non-Quadro
      };
      break;
    case viz::PrimitiveTopology::LINE_STRIP:
      primitive_count = 2;
      data.push_back(0.1F);
      data.push_back(0.1F);
      data.push_back(0.7F);
      data.push_back(0.9F);

      data.push_back(0.3F);
      data.push_back(0.2F);
      color_crc = {
          0x162496c0,  // Quadro
          0x9118f5cb   // non-Quadro
      };
      depth_crc = {
          0xfae233b9,  // Quadro
          0x92c04b5    // non-Quadro
      };
      break;
    case viz::PrimitiveTopology::TRIANGLE_LIST:
      primitive_count = 2;
      data.push_back(0.1F);
      data.push_back(0.1F);
      data.push_back(0.5F);
      data.push_back(0.9F);
      data.push_back(0.9F);
      data.push_back(0.1F);

      data.push_back(0.05F);
      data.push_back(0.7F);
      data.push_back(0.15F);
      data.push_back(0.8F);
      data.push_back(0.25F);
      data.push_back(0.6F);
      color_crc = {0x9de9f5f3};
      depth_crc = {0x101577b};
      break;
    case viz::PrimitiveTopology::CROSS_LIST:
      primitive_count = 2;
      data.push_back(0.5F);
      data.push_back(0.5F);
      data.push_back(0.1F);

      data.push_back(0.1F);
      data.push_back(0.3F);
      data.push_back(0.01F);
      color_crc = {
          0xb507fa88,  // Quadro
          0xf298654    // non-Quadro
      };
      depth_crc = {
          0x44098c3f,  // Quadro
          0x6fe44aee   // non-Quadro
      };
      break;
    case viz::PrimitiveTopology::RECTANGLE_LIST:
      primitive_count = 2;
      data.push_back(0.1F);
      data.push_back(0.1F);
      data.push_back(0.9F);
      data.push_back(0.9F);

      data.push_back(0.3F);
      data.push_back(0.2F);
      data.push_back(0.5F);
      data.push_back(0.3F);
      color_crc = {
          0x19a05481,  // Quadro
          0xf1f8f1b3   // non-Quadro
      };
      depth_crc = {
          0xf67bacdc,  // Quadro
          0x41396ef5   // non-Quadro
      };
      break;
    case viz::PrimitiveTopology::OVAL_LIST:
      primitive_count = 2;
      data.push_back(0.5F);
      data.push_back(0.5F);
      data.push_back(0.2F);
      data.push_back(0.1F);

      data.push_back(0.6F);
      data.push_back(0.4F);
      data.push_back(0.05F);
      data.push_back(0.07F);
      color_crc = {
          0x2341eef6,  // Quadro
          0xae3f0636   // non-Quadro
      };
      depth_crc = {
          0x41d7da93,  // Quadro
          0x7e44520d   // non-Quadro
      };
      break;
    case viz::PrimitiveTopology::POINT_LIST_3D:
      primitive_count = 1;
      data.push_back(-0.5F);
      data.push_back(0.5F);
      data.push_back(0.8F);
      color_crc = {0xd8f49994};
      depth_crc = {0x4e371ba0};
      break;
    case viz::PrimitiveTopology::LINE_LIST_3D:
      primitive_count = 2;
      data.push_back(-0.1F);
      data.push_back(-0.1F);
      data.push_back(0.1F);
      data.push_back(0.9F);
      data.push_back(0.9F);
      data.push_back(0.3F);

      data.push_back(-0.7F);
      data.push_back(-0.3F);
      data.push_back(0.2F);
      data.push_back(0.2F);
      data.push_back(0.4F);
      data.push_back(0.5F);
      color_crc = {
          0xc7762cc5,  // Quadro
          0xe9f3dbc3   // non-Quadro
      };
      depth_crc = {
          0x782f15cf,  // Quadro
          0xed2056f8   // non-Quadro
      };
      break;
    case viz::PrimitiveTopology::LINE_STRIP_3D:
      primitive_count = 2;
      data.push_back(-0.1F);
      data.push_back(-0.1F);
      data.push_back(0.1F);
      data.push_back(0.7F);
      data.push_back(0.9F);
      data.push_back(0.3F);

      data.push_back(-0.3F);
      data.push_back(-0.2F);
      data.push_back(0.2F);
      color_crc = {
          0x135ba8af,  // Quadro
          0x322d3fdd   // non-Quadro
      };
      depth_crc = {
          0x38dcc175,  // Quadro
          0xa2292265   // non-Quadro
      };
      break;
    case viz::PrimitiveTopology::TRIANGLE_LIST_3D:
      primitive_count = 2;
      data.push_back(-0.1F);
      data.push_back(-0.1F);
      data.push_back(0.F);
      data.push_back(0.5F);
      data.push_back(0.9F);
      data.push_back(0.1F);
      data.push_back(0.9F);
      data.push_back(0.1F);
      data.push_back(0.2F);

      data.push_back(-0.05F);
      data.push_back(-0.7F);
      data.push_back(0.3F);
      data.push_back(0.15F);
      data.push_back(0.8F);
      data.push_back(0.2F);
      data.push_back(0.25F);
      data.push_back(0.6F);
      data.push_back(0.5F);
      color_crc = {0xf372dff7};
      depth_crc = {0x90e4e07d};
      break;
    default:
      EXPECT_TRUE(false) << "Unhandled primitive topology";
  }

  EXPECT_NO_THROW(viz::Begin());

  viz::CudaService::ScopedPush cuda_context;
  viz::UniqueCUdeviceptr device_ptr;

  viz::CudaService cuda_service(0);

  if (source == Source::CUDA_DEVICE) {
    cuda_context = cuda_service.PushContext();
    device_ptr.reset([size = data.size() * sizeof(float)] {
      CUdeviceptr device_ptr;
      EXPECT_EQ(cuMemAlloc(&device_ptr, size), CUDA_SUCCESS);
      return device_ptr;
    }());

    EXPECT_EQ(cuMemcpyHtoD(device_ptr.get(), data.data(), data.size() * sizeof(float)),
              CUDA_SUCCESS);
  }

  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_NO_THROW(viz::BeginGeometryLayer());

    if (i != 0) { EXPECT_NO_THROW(viz::Color(1.F, 0.5F, 0.25F, 0.75F)); }
    if (i == 2) {
      EXPECT_NO_THROW(viz::PointSize(4.F));
      EXPECT_NO_THROW(viz::LineWidth(3.F));
    }

    if (source == Source::CUDA_DEVICE) {
      EXPECT_NO_THROW(
          viz::PrimitiveCudaDevice(topology, primitive_count, data.size(), device_ptr.get()));
    } else {
      EXPECT_NO_THROW(viz::Primitive(topology, primitive_count, data.size(), data.data()));
    }

    EXPECT_NO_THROW(viz::EndLayer());

    for (auto&& item : data) { item += 0.1F; }
    if (source == Source::CUDA_DEVICE) {
      EXPECT_EQ(cuMemcpyHtoD(device_ptr.get(), data.data(), data.size() * sizeof(float)),
                CUDA_SUCCESS);
    }
  }

  EXPECT_NO_THROW(viz::End());

  CompareColorResultCRC32(color_crc);
  CompareDepthResultCRC32(depth_crc);
}

INSTANTIATE_TEST_SUITE_P(
    GeometryLayer, PrimitiveTopology,
    testing::Combine(
        testing::Values(viz::PrimitiveTopology::POINT_LIST, viz::PrimitiveTopology::LINE_LIST,
                        viz::PrimitiveTopology::LINE_STRIP, viz::PrimitiveTopology::TRIANGLE_LIST,
                        viz::PrimitiveTopology::CROSS_LIST, viz::PrimitiveTopology::RECTANGLE_LIST,
                        viz::PrimitiveTopology::OVAL_LIST, viz::PrimitiveTopology::POINT_LIST_3D,
                        viz::PrimitiveTopology::LINE_LIST_3D, viz::PrimitiveTopology::LINE_STRIP_3D,
                        viz::PrimitiveTopology::TRIANGLE_LIST_3D),
        testing::Values(Source::HOST, Source::CUDA_DEVICE)));

// Fixture that initializes Holoviz
class GeometryLayer : public TestHeadless {};

TEST_F(GeometryLayer, Text) {
  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginGeometryLayer());
  EXPECT_NO_THROW(viz::Text(0.4F, 0.4F, 0.4F, "Text"));
  EXPECT_NO_THROW(viz::Color(0.5F, 0.9F, 0.7F, 0.9F));
  EXPECT_NO_THROW(viz::Text(0.1F, 0.1F, 0.2F, "Colored"));
  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());

  CompareColorResultCRC32({0xcb23d3cf});
}

TEST_F(GeometryLayer, TextClipped) {
  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginGeometryLayer());
  EXPECT_NO_THROW(viz::Text(1.1F, 0.4F, 0.4F, "Text"));

  EXPECT_NO_THROW(viz::End());

  CompareColorResultCRC32({0xd8f49994});
}

class GeometryLayerWithFont : public TestHeadless {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(viz::SetFont("../modules/holoviz/src/fonts/Roboto-Bold.ttf", 12.F));

    // call base class
    ::TestHeadless::SetUp();
  }

  void TearDown() override {
    // call base class
    ::TestHeadless::TearDown();

    ASSERT_NO_THROW(viz::SetFont("", 0.F));
  }
};

TEST_F(GeometryLayerWithFont, Text) {
  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginGeometryLayer());
  EXPECT_NO_THROW(viz::Text(0.1F, 0.1F, 0.7F, "Font"));
  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());

  CompareColorResultCRC32({0xb149eac7});
}

// Fixture that initializes Holoviz
class DepthMapRenderMode
    : public TestHeadless,
      public testing::WithParamInterface<std::tuple<viz::DepthMapRenderMode, viz::ImageFormat>> {};

TEST_P(DepthMapRenderMode, DepthMap) {
  const viz::DepthMapRenderMode depth_map_render_mode = std::get<0>(GetParam());
  const viz::ImageFormat depth_fmt = std::get<1>(GetParam());
  const uint32_t map_width = 8;
  const uint32_t map_height = 8;

  // allocate device memory
  viz::CudaService cuda_service(0);
  viz::CudaService::ScopedPush cuda_context = cuda_service.PushContext();

  uint32_t depth_component_size = 0;
  switch (depth_fmt) {
    case viz::ImageFormat::R8_UNORM:
      depth_component_size = sizeof(uint8_t);
      break;
    case viz::ImageFormat::D32_SFLOAT:
      depth_component_size = sizeof(uint32_t);
      break;
    default:
      FAIL();
  }

  viz::UniqueCUdeviceptr depth_ptr;
  depth_ptr.reset([this, depth_component_size] {
    CUdeviceptr device_ptr;
    EXPECT_EQ(cuMemAlloc(&device_ptr, map_width * map_height * depth_component_size), CUDA_SUCCESS);
    return device_ptr;
  }());

  switch (depth_fmt) {
    case viz::ImageFormat::R8_UNORM: {
      std::vector<uint8_t> depth_data(map_width * map_height);
      for (size_t index = 0; index < depth_data.size(); ++index) { depth_data[index] = index * 4; }
      EXPECT_EQ(cuMemcpyHtoD(
                    depth_ptr.get(), depth_data.data(), depth_data.size() * depth_component_size),
                CUDA_SUCCESS);
      break;
    }
    case viz::ImageFormat::D32_SFLOAT: {
      std::vector<float> depth_data(map_width * map_height);
      for (size_t index = 0; index < depth_data.size(); ++index) {
        depth_data[index] = index * 4 / 255.F;
      }
      EXPECT_EQ(cuMemcpyHtoD(
                    depth_ptr.get(), depth_data.data(), depth_data.size() * depth_component_size),
                CUDA_SUCCESS);
      break;
    }
    default:
      FAIL();
  }

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

  std::vector<uint32_t> crc;
  switch (depth_map_render_mode) {
    case viz::DepthMapRenderMode::POINTS:
      crc = {0x46e021cb};
      break;
    case viz::DepthMapRenderMode::LINES:
      crc = {
          0x6b63061e,  // Quadro
          0x69207440   // non-Quadro
      };
      break;
    case viz::DepthMapRenderMode::TRIANGLES:
      crc = {0x9cb8d951};
      break;
  }
  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginGeometryLayer());
  EXPECT_NO_THROW(viz::DepthMap(depth_map_render_mode,
                                map_width,
                                map_height,
                                depth_fmt,
                                depth_ptr.get(),
                                viz::ImageFormat::R8G8B8A8_UNORM,
                                color_ptr.get()));
  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());

  CompareColorResultCRC32(crc);
}

INSTANTIATE_TEST_SUITE_P(GeometryLayer, DepthMapRenderMode,
                         testing::Combine(testing::Values(viz::DepthMapRenderMode::POINTS,
                                                          viz::DepthMapRenderMode::LINES,
                                                          viz::DepthMapRenderMode::TRIANGLES),
                                          testing::Values(viz::ImageFormat::R8_UNORM,
                                                          viz::ImageFormat::D32_SFLOAT)));

TEST_F(GeometryLayer, Reuse) {
  std::vector<float> data{0.5F, 0.5F};

  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_NO_THROW(viz::Begin());

    EXPECT_NO_THROW(viz::BeginGeometryLayer());
    EXPECT_NO_THROW(viz::Color(0.1F, 0.2F, 0.3F, 0.4F));
    EXPECT_NO_THROW(viz::LineWidth(2.F));
    EXPECT_NO_THROW(viz::PointSize(3.F));
    EXPECT_NO_THROW(
        viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size(), data.data()));
    EXPECT_NO_THROW(viz::Text(0.4F, 0.4F, 0.1F, "Text"));
    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::End());
  }
}

TEST_F(GeometryLayer, Errors) {
  std::vector<float> data{0.5F, 0.5F};

  EXPECT_NO_THROW(viz::Begin());

  // it's an error to call geometry functions without calling BeginGeometryLayer first
  EXPECT_THROW(viz::Color(0.F, 0.F, 0.F, 1.F), std::runtime_error);
  EXPECT_THROW(viz::LineWidth(1.0F), std::runtime_error);
  EXPECT_THROW(viz::PointSize(1.0F), std::runtime_error);
  EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size(), data.data()),
               std::runtime_error);
  EXPECT_THROW(viz::Text(0.5F, 0.5F, 0.1F, "Text"), std::runtime_error);

  // it's an error to call BeginGeometryLayer again without calling EndLayer
  EXPECT_NO_THROW(viz::BeginGeometryLayer());
  EXPECT_THROW(viz::BeginGeometryLayer(), std::runtime_error);
  EXPECT_NO_THROW(viz::EndLayer());

  // it's an error to call geometry functions when a different layer is active
  EXPECT_NO_THROW(viz::BeginImageLayer());
  EXPECT_THROW(viz::Color(0.F, 0.F, 0.F, 1.F), std::runtime_error);
  EXPECT_THROW(viz::LineWidth(1.0F), std::runtime_error);
  EXPECT_THROW(viz::PointSize(1.0F), std::runtime_error);
  EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size(), data.data()),
               std::runtime_error);
  EXPECT_THROW(viz::Text(0.5F, 0.5F, 0.1F, "Text"), std::runtime_error);
  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::BeginGeometryLayer());

  struct {
    viz::PrimitiveTopology topology;
    uint32_t values;
  } required[] = {
      {viz::PrimitiveTopology::POINT_LIST, 2},
      {viz::PrimitiveTopology::LINE_LIST, 4},
      {viz::PrimitiveTopology::LINE_STRIP, 4},
      {viz::PrimitiveTopology::TRIANGLE_LIST, 6},
      {viz::PrimitiveTopology::CROSS_LIST, 3},
      {viz::PrimitiveTopology::RECTANGLE_LIST, 4},
      {viz::PrimitiveTopology::OVAL_LIST, 4},
      {viz::PrimitiveTopology::POINT_LIST_3D, 3},
      {viz::PrimitiveTopology::LINE_LIST_3D, 6},
      {viz::PrimitiveTopology::LINE_STRIP_3D, 6},
      {viz::PrimitiveTopology::TRIANGLE_LIST_3D, 9},
  };

  for (auto&& cur : required) {
    std::vector<float> data(cur.values, 0.F);
    // Primitive function errors, first call the passing function
    EXPECT_NO_THROW(viz::Primitive(cur.topology, 1, data.size(), data.data()));
    // it's an error to call Primitive with a data size which is too small for the primitive count
    EXPECT_THROW(viz::Primitive(cur.topology, 1, data.size() - 1, data.data()), std::runtime_error);
  }

  // it's an error to call Primitive with a primitive count of zero
  EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 0, data.size(), data.data()),
               std::invalid_argument);
  // it's an error to call Primitive with a data size of zero
  EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, 0, data.data()),
               std::invalid_argument);
  // it's an error to call Primitive with a null data pointer
  EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size(), nullptr),
               std::invalid_argument);

  // Text function errors, first call the passing function
  EXPECT_NO_THROW(viz::Text(0.5F, 0.5F, 0.1F, "Text"));
  // it's an error to call Text with a size of zero
  EXPECT_THROW(viz::Text(0.5F, 0.5F, 0.0F, "Text"), std::invalid_argument);
  // it's an error to call Text with null text pointer
  EXPECT_THROW(viz::Text(0.5F, 0.5F, 0.1F, nullptr), std::invalid_argument);

  // Depth map function errors, first call the passing function
  const uint32_t map_width = 8;
  const uint32_t map_height = 8;

  // allocate device memory
  viz::CudaService cuda_service(0);
  viz::CudaService::ScopedPush cuda_context = cuda_service.PushContext();
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
