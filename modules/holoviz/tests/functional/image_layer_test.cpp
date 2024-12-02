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

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda/cuda_service.hpp>
#include <holoviz/holoviz.hpp>
#include "test_fixture.hpp"

namespace viz = holoscan::viz;

enum class Source { HOST, CUDA_DEVICE };

enum class Reuse { DISABLE, ENABLE };

enum class UseLut { DISABLE, ENABLE, ENABLE_WITH_NORMALIZE };

enum class UseStream { DISABLE, ENABLE };

// define the '<<' operators to get a nice parameter string

#define CASE(VALUE)            \
  case VALUE:                  \
    os << std::string(#VALUE); \
    break;

std::ostream& operator<<(std::ostream& os, const Source& source) {
  switch (source) {
    CASE(Source::HOST)
    CASE(Source::CUDA_DEVICE)
    default:
      os.setstate(std::ios_base::failbit);
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Reuse& reuse) {
  switch (reuse) {
    CASE(Reuse::DISABLE)
    CASE(Reuse::ENABLE)
    default:
      os.setstate(std::ios_base::failbit);
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const UseLut& use_lut) {
  switch (use_lut) {
    CASE(UseLut::DISABLE)
    CASE(UseLut::ENABLE)
    CASE(UseLut::ENABLE_WITH_NORMALIZE)
    default:
      os.setstate(std::ios_base::failbit);
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const UseStream& use_stream) {
  switch (use_stream) {
    CASE(UseStream::DISABLE)
    CASE(UseStream::ENABLE)
    default:
      os.setstate(std::ios_base::failbit);
  }
  return os;
}

namespace holoscan {
namespace viz {

std::ostream& operator<<(std::ostream& os, const ImageFormat& format) {
  switch (format) {
    CASE(ImageFormat::R8_UINT)
    CASE(ImageFormat::R8_SINT)
    CASE(ImageFormat::R8_UNORM)
    CASE(ImageFormat::R8_SNORM)
    CASE(ImageFormat::R8_SRGB)
    CASE(ImageFormat::R16_UINT)
    CASE(ImageFormat::R16_SINT)
    CASE(ImageFormat::R16_UNORM)
    CASE(ImageFormat::R16_SNORM)
    CASE(ImageFormat::R16_SFLOAT)
    CASE(ImageFormat::R32_UINT)
    CASE(ImageFormat::R32_SINT)
    CASE(ImageFormat::R32_SFLOAT)
    CASE(ImageFormat::R8G8B8_UNORM)
    CASE(ImageFormat::R8G8B8_SNORM)
    CASE(ImageFormat::R8G8B8_SRGB)
    CASE(ImageFormat::R8G8B8A8_UNORM)
    CASE(ImageFormat::R8G8B8A8_SNORM)
    CASE(ImageFormat::R8G8B8A8_SRGB)
    CASE(ImageFormat::R16G16B16A16_UNORM)
    CASE(ImageFormat::R16G16B16A16_SNORM)
    CASE(ImageFormat::R16G16B16A16_SFLOAT)
    CASE(ImageFormat::R32G32B32A32_SFLOAT)
    CASE(ImageFormat::D16_UNORM)
    CASE(ImageFormat::X8_D24_UNORM)
    CASE(ImageFormat::D32_SFLOAT)
    CASE(ImageFormat::A2B10G10R10_UNORM_PACK32)
    CASE(ImageFormat::A2R10G10B10_UNORM_PACK32)
    CASE(ImageFormat::B8G8R8A8_UNORM)
    CASE(ImageFormat::B8G8R8A8_SRGB)
    CASE(ImageFormat::A8B8G8R8_UNORM_PACK32)
    CASE(ImageFormat::A8B8G8R8_SRGB_PACK32)
    CASE(ImageFormat::Y8U8Y8V8_422_UNORM)
    CASE(ImageFormat::U8Y8V8Y8_422_UNORM)
    CASE(ImageFormat::Y8_U8V8_2PLANE_420_UNORM)
    CASE(ImageFormat::Y8_U8V8_2PLANE_422_UNORM)
    CASE(ImageFormat::Y8_U8_V8_3PLANE_420_UNORM)
    CASE(ImageFormat::Y8_U8_V8_3PLANE_422_UNORM)
    CASE(ImageFormat::Y16_U16V16_2PLANE_420_UNORM)
    CASE(ImageFormat::Y16_U16V16_2PLANE_422_UNORM)
    CASE(ImageFormat::Y16_U16_V16_3PLANE_420_UNORM)
    CASE(ImageFormat::Y16_U16_V16_3PLANE_422_UNORM)
    default:
      os.setstate(std::ios_base::failbit);
  }
  return os;
}
}  // namespace viz
}  // namespace holoscan

#undef CASE

class ImageLayer
    : public TestHeadless,
      public testing::WithParamInterface<std::tuple<Source, Reuse, UseLut, viz::ImageFormat,
                                                    viz::YuvModelConversion, viz::YuvRange>> {};

TEST_P(ImageLayer, Image) {
  const Source source = std::get<0>(GetParam());
  const bool reuse = std::get<1>(GetParam()) == Reuse::ENABLE;
  const UseLut use_lut = std::get<2>(GetParam());
  const viz::ImageFormat image_format = std::get<3>(GetParam());
  const viz::YuvModelConversion yuv_model_conversion = std::get<4>(GetParam());
  const viz::YuvRange yuv_range = std::get<5>(GetParam());

  if (use_lut == UseLut::ENABLE_WITH_NORMALIZE) {
    GTEST_SKIP() << "LUT with normalize tests not working yet, reference image generation needs to "
                    "be fixed.";
  }

  bool use_depth = false;
  bool convert_color = false;
  bool is_yuv = false;
  uintptr_t offset_plane_1 = 0;
  uintptr_t offset_plane_2 = 0;
  std::vector<uint8_t> converted_data;

  switch (image_format) {
    case viz::ImageFormat::R8_UINT:
    case viz::ImageFormat::R8_SINT:
    case viz::ImageFormat::R8_UNORM:
    case viz::ImageFormat::R16_UINT:
    case viz::ImageFormat::R16_SINT:
    case viz::ImageFormat::R32_UINT:
    case viz::ImageFormat::R32_SINT:
    case viz::ImageFormat::R8G8B8_UNORM:
    case viz::ImageFormat::R8G8B8A8_UNORM:
      break;
    case viz::ImageFormat::R8_SNORM:
    case viz::ImageFormat::R8_SRGB:
    case viz::ImageFormat::R16_UNORM:
    case viz::ImageFormat::R16_SNORM:
    case viz::ImageFormat::R32_SFLOAT:
    case viz::ImageFormat::R8G8B8_SNORM:
    case viz::ImageFormat::R8G8B8_SRGB:
    case viz::ImageFormat::R8G8B8A8_SNORM:
    case viz::ImageFormat::R8G8B8A8_SRGB:
    case viz::ImageFormat::R16G16B16A16_UNORM:
    case viz::ImageFormat::R16G16B16A16_SNORM:
    case viz::ImageFormat::R32G32B32A32_SFLOAT:
    case viz::ImageFormat::A2B10G10R10_UNORM_PACK32:
    case viz::ImageFormat::A2R10G10B10_UNORM_PACK32:
    case viz::ImageFormat::B8G8R8A8_UNORM:
    case viz::ImageFormat::B8G8R8A8_SRGB:
    case viz::ImageFormat::A8B8G8R8_UNORM_PACK32:
    case viz::ImageFormat::A8B8G8R8_SRGB_PACK32:
      convert_color = true;
      break;
    case viz::ImageFormat::Y8U8Y8V8_422_UNORM:
    case viz::ImageFormat::U8Y8V8Y8_422_UNORM:
      is_yuv = true;
      converted_data.resize((width_ * height_ * 1 + (width_ / 2) * height_ * 2) * sizeof(uint8_t));
      break;
    case viz::ImageFormat::Y8_U8V8_2PLANE_420_UNORM:
      is_yuv = true;
      offset_plane_1 = width_ * height_ * sizeof(uint8_t);
      converted_data.resize(offset_plane_1 + ((width_ / 2) * (height_ / 2) * 2) * sizeof(uint8_t));
      break;
    case viz::ImageFormat::Y8_U8V8_2PLANE_422_UNORM:
      is_yuv = true;
      offset_plane_1 = width_ * height_ * sizeof(uint8_t);
      converted_data.resize(offset_plane_1 + ((width_ / 2) * height_ * 2) * sizeof(uint8_t));
      break;
    case viz::ImageFormat::Y8_U8_V8_3PLANE_420_UNORM:
      is_yuv = true;
      offset_plane_1 = width_ * height_ * sizeof(uint8_t);
      offset_plane_2 = offset_plane_1 + (width_ / 2) * (height_ / 2) * sizeof(uint8_t);
      converted_data.resize(offset_plane_2 + ((width_ / 2) * (height_ / 2)) * sizeof(uint8_t));
      break;
    case viz::ImageFormat::Y8_U8_V8_3PLANE_422_UNORM:
      is_yuv = true;
      offset_plane_1 = width_ * height_ * sizeof(uint8_t);
      offset_plane_2 = offset_plane_1 + (width_ / 2) * height_ * sizeof(uint8_t);
      converted_data.resize(offset_plane_2 + ((width_ / 2) * height_) * sizeof(uint8_t));
      break;
    case viz::ImageFormat::Y16_U16V16_2PLANE_420_UNORM:
      is_yuv = true;
      offset_plane_1 = width_ * height_ * sizeof(uint16_t);
      converted_data.resize(offset_plane_1 + ((width_ / 2) * (height_ / 2) * 2) * sizeof(uint16_t));
      break;
    case viz::ImageFormat::Y16_U16V16_2PLANE_422_UNORM:
      is_yuv = true;
      offset_plane_1 = width_ * height_ * sizeof(uint16_t);
      converted_data.resize(offset_plane_1 + ((width_ / 2) * height_ * 2) * sizeof(uint16_t));
      break;
    case viz::ImageFormat::Y16_U16_V16_3PLANE_420_UNORM:
      is_yuv = true;
      offset_plane_1 = width_ * height_ * sizeof(uint16_t);
      offset_plane_2 = offset_plane_1 + (width_ / 2) * (height_ / 2) * sizeof(uint16_t);
      converted_data.resize(offset_plane_2 + ((width_ / 2) * (height_ / 2)) * sizeof(uint16_t));
      break;
    case viz::ImageFormat::Y16_U16_V16_3PLANE_422_UNORM:
      is_yuv = true;
      offset_plane_1 = width_ * height_ * sizeof(uint16_t);
      offset_plane_2 = offset_plane_1 + (width_ / 2) * height_ * sizeof(uint16_t);
      converted_data.resize(offset_plane_2 + ((width_ / 2) * height_) * sizeof(uint16_t));
      break;
    case viz::ImageFormat::D16_UNORM:
    case viz::ImageFormat::X8_D24_UNORM:
    case viz::ImageFormat::D32_SFLOAT:
      use_depth = true;
      break;
    case viz::ImageFormat::R16_SFLOAT:
    case viz::ImageFormat::R16G16B16A16_SFLOAT:
    default:
      ASSERT_TRUE(false) << "Unhandled image format";
      break;
  }

  viz::ImageFormat color_format, depth_format;

  if (use_depth) {
    color_format = viz::ImageFormat::R8G8B8A8_UNORM;
    depth_format = image_format;
    SetupData(color_format);
    SetupData(depth_format);
  } else if (is_yuv) {
    color_format = image_format;

    // create a smooth RGB pattern so we don't need to deal with linear chroma filtering when
    // calculating the expected result
    color_data_.resize(width_ * height_ * 3);
    for (uint32_t y = 0; y < height_; ++y) {
      for (uint32_t x = 0; x < width_; ++x) {
        color_data_[y * (width_ * 3) + x * 3 + 0] = x;
        color_data_[y * (width_ * 3) + x * 3 + 1] = y;
        color_data_[y * (width_ * 3) + x * 3 + 2] = 255 - x;
      }
    }
    // convert to YUV
    for (uint32_t y = 0; y < height_; ++y) {
      for (uint32_t x = 0; x < width_; ++x) {
        // RGB -> YUV conversion
        const float r = color_data_[y * (width_ * 3) + x * 3 + 0] / 255.F;
        const float g = color_data_[y * (width_ * 3) + x * 3 + 1] / 255.F;
        const float b = color_data_[y * (width_ * 3) + x * 3 + 2] / 255.F;
        float Kr, Kg, Kb;
        switch (yuv_model_conversion) {
          case viz::YuvModelConversion::YUV_601:
            Kr = 0.299F;
            Kb = 0.114F;
            break;
          case viz::YuvModelConversion::YUV_709:
            Kb = 0.0722F;
            Kr = 0.2126F;
            break;
          case viz::YuvModelConversion::YUV_2020:
            Kb = 0.0593F;
            Kr = 0.2627F;
            break;
          default:
            ASSERT_TRUE(false) << "Unhandled yuv model conversion";
            break;
        }
        // since Kr + Kg + Kb = 1.F, calculate Kg
        Kg = 1.F - Kb - Kr;

        float luma = Kr * r + Kg * g + Kb * b;  // 0 ... 1
        float u = (b - luma) / (1.F - Kb);      // -1 ... 1
        float v = (r - luma) / (1.F - Kr);      // -1 ... 1

        switch (yuv_range) {
          case viz::YuvRange::ITU_FULL:
            u = u * 0.5F + 0.5F;
            v = v * 0.5F + 0.5F;
            break;
          case viz::YuvRange::ITU_NARROW:
            luma = 16.F / 255.F + luma * (219.F / 255.F);
            u = 128.F / 255.F + u * 0.5F * (224.F / 255.F);
            v = 128.F / 255.F + v * 0.5F * (224.F / 255.F);
            break;
          default:
            ASSERT_TRUE(false) << "Unhandled yuv range";
            break;
        }

        switch (image_format) {
          case viz::ImageFormat::Y8U8Y8V8_422_UNORM:
            converted_data[y * (width_ * 2) + x * 2] = uint8_t(luma * 255.F + 0.5F);
            if ((x & 1) == 0) {
              converted_data[y * (width_ * 2) + (x * 2) + 1] = uint8_t(u * 255.F + 0.5F);
              converted_data[y * (width_ * 2) + (x * 2) + 3] = uint8_t(v * 255.F + 0.5F);
            }
            break;
          case viz::ImageFormat::U8Y8V8Y8_422_UNORM:
            converted_data[y * (width_ * 2) + (x * 2) + 1] = uint8_t(luma * 255.F + 0.5F);
            if ((x & 1) == 0) {
              converted_data[y * (width_ * 2) + (x * 2) + 0] = uint8_t(u * 255.F + 0.5F);
              converted_data[y * (width_ * 2) + (x * 2) + 2] = uint8_t(v * 255.F + 0.5F);
            }
            break;
          case viz::ImageFormat::Y8_U8V8_2PLANE_420_UNORM:
            converted_data[y * width_ + x] = uint8_t(luma * 255.F + 0.5F);
            if (((x & 1) == 0) && ((y & 1) == 0)) {
              converted_data[offset_plane_1 + ((y / 2) * (width_ / 2) + (x / 2)) * 2 + 0] =
                  uint8_t(u * 255.F + 0.5F);
              converted_data[offset_plane_1 + ((y / 2) * (width_ / 2) + (x / 2)) * 2 + 1] =
                  uint8_t(v * 255.F + 0.5F);
            }
            break;
          case viz::ImageFormat::Y8_U8V8_2PLANE_422_UNORM:
            converted_data[y * width_ + x] = uint8_t(luma * 255.F + 0.5F);
            if ((x & 1) == 0) {
              converted_data[offset_plane_1 + (y * (width_ / 2) + (x / 2)) * 2 + 0] =
                  uint8_t(u * 255.F + 0.5F);
              converted_data[offset_plane_1 + (y * (width_ / 2) + (x / 2)) * 2 + 1] =
                  uint8_t(v * 255.F + 0.5F);
            }
            break;
          case viz::ImageFormat::Y8_U8_V8_3PLANE_420_UNORM:
            converted_data[y * width_ + x] = uint8_t(luma * 255.F + 0.5F);
            if (((x & 1) == 0) && ((y & 1) == 0)) {
              converted_data[offset_plane_1 + (y / 2) * (width_ / 2) + (x / 2)] =
                  uint8_t(u * 255.F + 0.5F);
              converted_data[offset_plane_2 + (y / 2) * (width_ / 2) + (x / 2)] =
                  uint8_t(v * 255.F + 0.5F);
            }
            break;
          case viz::ImageFormat::Y8_U8_V8_3PLANE_422_UNORM:
            converted_data[y * width_ + x] = uint8_t(luma * 255.F + 0.5F);
            if ((x & 1) == 0) {
              converted_data[offset_plane_1 + y * (width_ / 2) + (x / 2)] =
                  uint8_t(u * 255.F + 0.5F);
              converted_data[offset_plane_2 + y * (width_ / 2) + (x / 2)] =
                  uint8_t(v * 255.F + 0.5F);
            }
            break;
          case viz::ImageFormat::Y16_U16V16_2PLANE_420_UNORM:
            reinterpret_cast<uint16_t*>(converted_data.data())[y * width_ + x] =
                uint16_t(luma * 65535.F + 0.5F);
            if (((x & 1) == 0) && ((y & 1) == 0)) {
              reinterpret_cast<uint16_t*>(
                  converted_data.data())[offset_plane_1 / sizeof(uint16_t) +
                                         ((y / 2) * (width_ / 2) + (x / 2)) * 2 + 0] =
                  uint16_t(u * 65535.F + 0.5F);
              reinterpret_cast<uint16_t*>(
                  converted_data.data())[offset_plane_1 / sizeof(uint16_t) +
                                         ((y / 2) * (width_ / 2) + (x / 2)) * 2 + 1] =
                  uint16_t(v * 65535.F + 0.5F);
            }
            break;
          case viz::ImageFormat::Y16_U16V16_2PLANE_422_UNORM:
            reinterpret_cast<uint16_t*>(converted_data.data())[y * width_ + x] =
                uint16_t(luma * 65535.F + 0.5F);
            if ((x & 1) == 0) {
              reinterpret_cast<uint16_t*>(
                  converted_data.data())[offset_plane_1 / sizeof(uint16_t) +
                                         (y * (width_ / 2) + (x / 2)) * 2 + 0] =
                  uint16_t(u * 65535.F + 0.5F);
              reinterpret_cast<uint16_t*>(
                  converted_data.data())[offset_plane_1 / sizeof(uint16_t) +
                                         (y * (width_ / 2) + (x / 2)) * 2 + 1] =
                  uint16_t(v * 65535.F + 0.5F);
            }
            break;
          case viz::ImageFormat::Y16_U16_V16_3PLANE_420_UNORM:
            reinterpret_cast<uint16_t*>(converted_data.data())[y * width_ + x] =
                uint16_t(luma * 65535.F + 0.5F);
            if (((x & 1) == 0) && ((y & 1) == 0)) {
              reinterpret_cast<uint16_t*>(converted_data.data())[offset_plane_1 / sizeof(uint16_t) +
                                                                 (y / 2) * (width_ / 2) + (x / 2)] =
                  uint16_t(u * 65535.F + 0.5F);
              reinterpret_cast<uint16_t*>(converted_data.data())[offset_plane_2 / sizeof(uint16_t) +
                                                                 (y / 2) * (width_ / 2) + (x / 2)] =
                  uint16_t(v * 65535.F + 0.5F);
            }
            break;
          case viz::ImageFormat::Y16_U16_V16_3PLANE_422_UNORM:
            reinterpret_cast<uint16_t*>(converted_data.data())[y * width_ + x] =
                uint16_t(luma * 65535.F + 0.5F);
            if ((x & 1) == 0) {
              reinterpret_cast<uint16_t*>(converted_data.data())[offset_plane_1 / sizeof(uint16_t) +
                                                                 y * (width_ / 2) + (x / 2)] =
                  uint16_t(u * 65535.F + 0.5F);
              reinterpret_cast<uint16_t*>(converted_data.data())[offset_plane_2 / sizeof(uint16_t) +
                                                                 y * (width_ / 2) + (x / 2)] =
                  uint16_t(v * 65535.F + 0.5F);
            }
            break;
          default:
            ASSERT_TRUE(false) << "Unhandled image format";
            break;
        }
      }
    }
    // use YUV as source and RGB as reference
    std::swap(color_data_, converted_data);

    depth_format = viz::ImageFormat::D32_SFLOAT;
    depth_data_ = std::vector<float>(width_ * height_ * 1 * sizeof(float), 0.F);
  } else {
    color_format = image_format;
    depth_format = viz::ImageFormat::D32_SFLOAT;
    SetupData(image_format);
    depth_data_ = std::vector<float>(width_ * height_ * 1 * sizeof(float), 0.F);
  }

  std::vector<uint32_t> lut;

  viz::CudaService::ScopedPush cuda_context;
  viz::UniqueCUdeviceptr color_device_ptr;
  viz::UniqueCUdeviceptr depth_device_ptr;

  viz::CudaService cuda_service(0);

  if (source == Source::CUDA_DEVICE) {
    cuda_context = cuda_service.PushContext();
    color_device_ptr.reset([this] {
      CUdeviceptr device_ptr;
      EXPECT_EQ(cuMemAlloc(&device_ptr, color_data_.size()), CUDA_SUCCESS);
      return device_ptr;
    }());

    EXPECT_EQ(cuMemcpyHtoD(color_device_ptr.get(), color_data_.data(), color_data_.size()),
              CUDA_SUCCESS);

    if (use_depth) {
      depth_device_ptr.reset([this] {
        CUdeviceptr device_ptr;
        EXPECT_EQ(cuMemAlloc(&device_ptr, depth_data_.size()), CUDA_SUCCESS);
        return device_ptr;
      }());

      EXPECT_EQ(cuMemcpyHtoD(depth_device_ptr.get(), depth_data_.data(), depth_data_.size()),
                CUDA_SUCCESS);
    }
  }

  if (use_lut != UseLut::DISABLE) {
    std::srand(1);
    lut.resize(lut_size_);
    for (uint32_t index = 0; index < lut_size_; ++index) {
      lut[index] = static_cast<uint32_t>(std::rand()) | 0xFF000000;
    }

    // lookup color to produce result
    converted_data.resize(width_ * height_ * sizeof(uint32_t));
    if (use_lut == UseLut::ENABLE) {
      for (size_t index = 0; index < width_ * height_; ++index) {
        uint32_t lut_index;
        switch (color_format) {
          case viz::ImageFormat::R8_UINT:
          case viz::ImageFormat::R8_SINT:
            lut_index = color_data_[index];
            break;
          case viz::ImageFormat::R16_UINT:
            lut_index = reinterpret_cast<uint16_t*>(color_data_.data())[index];
            break;
          case viz::ImageFormat::R16_SINT:
            lut_index = reinterpret_cast<int16_t*>(color_data_.data())[index];
            break;
          case viz::ImageFormat::R32_UINT:
            lut_index = reinterpret_cast<uint32_t*>(color_data_.data())[index];
            break;
          case viz::ImageFormat::R32_SINT:
            lut_index = reinterpret_cast<int32_t*>(color_data_.data())[index];
            break;
          case viz::ImageFormat::R32_SFLOAT:
            lut_index =
                static_cast<uint32_t>(reinterpret_cast<float*>(color_data_.data())[index] + 0.5F);
            break;
          default:
            ASSERT_TRUE(false) << "Unhandled LUT image format";
            break;
        }
        reinterpret_cast<uint32_t*>(converted_data.data())[index] = lut[lut_index];
      }
    } else if (use_lut == UseLut::ENABLE_WITH_NORMALIZE) {
      for (size_t index = 0; index < width_ * height_; ++index) {
        float offset;
        switch (color_format) {
          case viz::ImageFormat::R8_UINT:
          case viz::ImageFormat::R8_SINT:
            offset = color_data_[index];
            break;
          case viz::ImageFormat::R8_UNORM:
            offset = (static_cast<float>(color_data_[index]) / 255.F) * (lut_size_ - 1);
            break;
          case viz::ImageFormat::R8_SNORM:
            offset = (static_cast<float>(color_data_[index]) / 127.F) * (lut_size_ - 1);
            break;
          case viz::ImageFormat::R16_UNORM:
            offset = (static_cast<float>(reinterpret_cast<uint16_t*>(color_data_.data())[index]) /
                      65535.F) *
                     (lut_size_ - 1);
            break;
          case viz::ImageFormat::R16_SNORM:
            offset = (static_cast<float>(reinterpret_cast<int16_t*>(color_data_.data())[index]) /
                      32767.F) *
                     (lut_size_ - 1);
            break;
          case viz::ImageFormat::R32_SFLOAT:
            offset = reinterpret_cast<float*>(color_data_.data())[index] * (lut_size_ - 1);
            break;
          default:
            ASSERT_TRUE(false) << "Unhandled LUT with normalize color format";
            break;
        }

        const uint32_t val0 =
            lut[std::max(0, std::min(int32_t(lut_size_) - 1, int32_t(offset * (lut_size_ - 1))))];
        const uint32_t val1 = lut[std::max(
            0,
            std::min(int32_t(lut_size_) - 1,
                     int32_t((offset + (1.0F / float(lut_size_))) * (lut_size_ - 1))))];
        float dummy;
        const float frac = std::modf(offset, &dummy);

        const float r0 = float(val0 & 0xFF);
        const float g0 = float((val0 & 0xFF00) >> 8);
        const float b0 = float((val0 & 0xFF0000) >> 16);
        const float a0 = float((val0 & 0xFF000000) >> 24);

        const float r1 = float(val1 & 0xFF);
        const float g1 = float((val1 & 0xFF00) >> 8);
        const float b1 = float((val1 & 0xFF0000) >> 16);
        const float a1 = float((val1 & 0xFF000000) >> 24);

        const float r = r0 + frac * (r1 - r0);
        const float g = g0 + frac * (g1 - g0);
        const float b = b0 + frac * (b1 - b0);
        const float a = a0 + frac * (a1 - a0);

        reinterpret_cast<uint32_t*>(converted_data.data())[index] =
            uint32_t(r + 0.5F) | (uint32_t(g + 0.5F) << 8) | (uint32_t(b + 0.5F) << 16) |
            (uint32_t(a + 0.5F) << 24);
      }
    }
  } else if (convert_color) {
    uint32_t components;
    switch (color_format) {
      case viz::ImageFormat::R8_SRGB:
      case viz::ImageFormat::R8_SNORM:
      case viz::ImageFormat::R16_UNORM:
      case viz::ImageFormat::R16_SNORM:
      case viz::ImageFormat::R32_SFLOAT:
        components = 1;
        break;
      case viz::ImageFormat::R8G8B8_SNORM:
      case viz::ImageFormat::R8G8B8_SRGB:
        components = 3;
        break;
      case viz::ImageFormat::R8G8B8A8_SNORM:
      case viz::ImageFormat::R8G8B8A8_SRGB:
      case viz::ImageFormat::R16G16B16A16_SNORM:
      case viz::ImageFormat::R16G16B16A16_UNORM:
      case viz::ImageFormat::A2B10G10R10_UNORM_PACK32:
      case viz::ImageFormat::A2R10G10B10_UNORM_PACK32:
      case viz::ImageFormat::B8G8R8A8_UNORM:
      case viz::ImageFormat::B8G8R8A8_SRGB:
      case viz::ImageFormat::A8B8G8R8_UNORM_PACK32:
      case viz::ImageFormat::A8B8G8R8_SRGB_PACK32:
        components = 4;
        break;
      default:
        ASSERT_TRUE(false) << "Unhandled color format in conversion";
        break;
    }

    const size_t elements = width_ * height_ * components;
    converted_data.resize(elements);
    for (size_t index = 0; index < elements; ++index) {
      switch (color_format) {
        case viz::ImageFormat::R8_SNORM:
        case viz::ImageFormat::R8G8B8_SNORM:
        case viz::ImageFormat::R8G8B8A8_SNORM:
          converted_data[index] = uint8_t(
              (float(reinterpret_cast<int8_t*>(color_data_.data())[index]) / 127.F) * 255.F + 0.5F);
          break;
        case viz::ImageFormat::R16_UNORM:
        case viz::ImageFormat::R16G16B16A16_UNORM:
          converted_data[index] = uint8_t(
              (float(reinterpret_cast<uint16_t*>(color_data_.data())[index]) / 65535.F) * 255.F +
              0.5F);
          break;
        case viz::ImageFormat::R16_SNORM:
        case viz::ImageFormat::R16G16B16A16_SNORM:
          converted_data[index] = uint8_t(
              (float(reinterpret_cast<int16_t*>(color_data_.data())[index]) / 32767.F) * 255.F +
              0.5F);
          break;
        case viz::ImageFormat::R32_SFLOAT:
          converted_data[index] =
              uint8_t(reinterpret_cast<float*>(color_data_.data())[index] * 255.F + 0.5F);
          break;
        case viz::ImageFormat::R8_SRGB:
        case viz::ImageFormat::R8G8B8_SRGB:
        case viz::ImageFormat::R8G8B8A8_SRGB:
        case viz::ImageFormat::A8B8G8R8_UNORM_PACK32:
        case viz::ImageFormat::A8B8G8R8_SRGB_PACK32:
          converted_data[index] = color_data_[index];
          break;
        case viz::ImageFormat::B8G8R8A8_UNORM:
        case viz::ImageFormat::B8G8R8A8_SRGB: {
          const uint32_t component = index % 4;
          converted_data[index] =
              color_data_[index - component + ((component < 3) ? (2 - component) : component)];
        } break;
        case viz::ImageFormat::A2B10G10R10_UNORM_PACK32: {
          const uint32_t value = reinterpret_cast<uint32_t*>(color_data_.data())[index / 4];
          const uint32_t component = index % 4;
          if (component == 3) {
            converted_data[index] = uint8_t((float(value >> 30) / 3.F) * 255.F + 0.5F);
          } else {
            converted_data[index] =
                uint8_t((float((value >> (component * 10)) & 0x3FF) / 1023.F) * 255.F + 0.5F);
          }
        } break;
        case viz::ImageFormat::A2R10G10B10_UNORM_PACK32: {
          const uint32_t value = reinterpret_cast<uint32_t*>(color_data_.data())[index / 4];
          const uint32_t component = index % 4;
          if (component == 3) {
            converted_data[index] = uint8_t((float(value >> 30) / 3.F) * 255.F + 0.5F);
          } else {
            converted_data[index] =
                uint8_t((float((value >> ((2 - component) * 10)) & 0x3FF) / 1023.F) * 255.F + 0.5F);
          }
        } break;
        default:
          ASSERT_TRUE(false) << "Unhandled color format in conversion";
          break;
      }
    }
    // sRGB EOTF conversion
    // https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html
    switch (color_format) {
      case viz::ImageFormat::R8_SRGB:
      case viz::ImageFormat::R8G8B8_SRGB:
      case viz::ImageFormat::R8G8B8A8_SRGB:
      case viz::ImageFormat::B8G8R8A8_SRGB:
      case viz::ImageFormat::A8B8G8R8_SRGB_PACK32:
        for (size_t index = 0; index < elements; index += components) {
          for (size_t component = 0; component < components; ++component) {
            float value = float(converted_data[index + component]) / 255.F;
            if (value < 0.04045F) {
              value /= 12.92F;
            } else {
              value = std::pow(((value + 0.055F) / 1.055F), 2.4F);
            }
            converted_data[index + component] = uint8_t((value * 255.F) + 0.5F);
          }
        }
        break;
      default:
        // non sRGB
        break;
    }
  }

  for (uint32_t i = 0; i < (reuse ? 2 : 1); ++i) {
    EXPECT_NO_THROW(viz::Begin());

    EXPECT_NO_THROW(viz::BeginImageLayer());

    if (is_yuv) {
      EXPECT_NO_THROW(viz::ImageYuvModelConversion(yuv_model_conversion));
      EXPECT_NO_THROW(viz::ImageYuvRange(yuv_range));
    }

    if (use_lut != UseLut::DISABLE) {
      EXPECT_NO_THROW(viz::LUT(lut_size_,
                               viz::ImageFormat::R8G8B8A8_UNORM,
                               lut.size() * sizeof(uint32_t),
                               lut.data(),
                               use_lut == UseLut::ENABLE_WITH_NORMALIZE));
    }

    switch (source) {
      case Source::HOST:
        EXPECT_NO_THROW(
            viz::ImageHost(width_,
                           height_,
                           color_format,
                           reinterpret_cast<void*>(color_data_.data()),
                           0,
                           offset_plane_1 ? color_data_.data() + offset_plane_1 : nullptr,
                           0,
                           offset_plane_2 ? color_data_.data() + offset_plane_2 : nullptr,
                           0));
        break;
      case Source::CUDA_DEVICE:
        EXPECT_NO_THROW(
            viz::ImageCudaDevice(width_,
                                 height_,
                                 color_format,
                                 color_device_ptr.get(),
                                 0,
                                 offset_plane_1 ? color_device_ptr.get() + offset_plane_1 : 0,
                                 0,
                                 offset_plane_2 ? color_device_ptr.get() + offset_plane_2 : 0,
                                 0));
        break;
      default:
        EXPECT_TRUE(false) << "Unhandled source type";
    }

    if (use_depth) {
      switch (source) {
        case Source::HOST:
          EXPECT_NO_THROW(viz::ImageHost(
              width_, height_, depth_format, reinterpret_cast<void*>(depth_data_.data())));
          break;
        case Source::CUDA_DEVICE:
          EXPECT_NO_THROW(
              viz::ImageCudaDevice(width_, height_, depth_format, depth_device_ptr.get()));
          break;
        default:
          EXPECT_TRUE(false) << "Unhandled source type";
      }
    }

    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::End());
  }

  if (converted_data.size() != 0) {
    std::swap(color_data_, converted_data);
    // YUV data requires a higher absolute error because of conversion between color spaces
    CompareColorResult(is_yuv ? 4 : 1);
    std::swap(converted_data, color_data_);
  } else {
    CompareColorResult();
  }

  CompareDepthResult();
}

// source host or device with reuse test
INSTANTIATE_TEST_SUITE_P(ImageLayerSource, ImageLayer,
                         testing::Combine(testing::Values(Source::HOST, Source::CUDA_DEVICE),
                                          testing::Values(Reuse::DISABLE, Reuse::ENABLE),
                                          testing::Values(UseLut::DISABLE),
                                          testing::Values(viz::ImageFormat::R8G8B8A8_UNORM),
                                          testing::Values(viz::YuvModelConversion::YUV_601),
                                          testing::Values(viz::YuvRange::ITU_FULL)));

// native RGB color formats
INSTANTIATE_TEST_SUITE_P(
    ImageLayerFormat, ImageLayer,
    testing::Combine(
        testing::Values(Source::CUDA_DEVICE, Source::HOST), testing::Values(Reuse::DISABLE),
        testing::Values(UseLut::DISABLE),
        testing::Values(
            viz::ImageFormat::R8_UNORM, viz::ImageFormat::R8_SNORM, viz::ImageFormat::R8_SRGB,
            viz::ImageFormat::R16_UNORM, viz::ImageFormat::R16_SNORM, viz::ImageFormat::R32_SFLOAT,
            viz::ImageFormat::R8G8B8A8_UNORM, viz::ImageFormat::R8G8B8A8_SNORM,
            viz::ImageFormat::R8G8B8A8_SRGB, viz::ImageFormat::R16G16B16A16_SNORM,
            viz::ImageFormat::R16G16B16A16_UNORM, viz::ImageFormat::A2B10G10R10_UNORM_PACK32,
            viz::ImageFormat::A2R10G10B10_UNORM_PACK32, viz::ImageFormat::B8G8R8A8_UNORM,
            viz::ImageFormat::B8G8R8A8_SRGB, viz::ImageFormat::A8B8G8R8_UNORM_PACK32,
            viz::ImageFormat::A8B8G8R8_SRGB_PACK32),
        testing::Values(viz::YuvModelConversion::YUV_601),
        testing::Values(viz::YuvRange::ITU_FULL)));

// native YUV color formats
INSTANTIATE_TEST_SUITE_P(
    ImageLayerFormatYUV, ImageLayer,
    testing::Combine(testing::Values(Source::CUDA_DEVICE, Source::HOST),
                     testing::Values(Reuse::DISABLE), testing::Values(UseLut::DISABLE),
                     testing::Values(viz::ImageFormat::Y8U8Y8V8_422_UNORM,
                                     viz::ImageFormat::U8Y8V8Y8_422_UNORM,
                                     viz::ImageFormat::Y8_U8V8_2PLANE_420_UNORM,
                                     viz::ImageFormat::Y8_U8V8_2PLANE_422_UNORM,
                                     viz::ImageFormat::Y8_U8_V8_3PLANE_420_UNORM,
                                     viz::ImageFormat::Y8_U8_V8_3PLANE_422_UNORM,
                                     viz::ImageFormat::Y16_U16V16_2PLANE_420_UNORM,
                                     viz::ImageFormat::Y16_U16V16_2PLANE_422_UNORM,
                                     viz::ImageFormat::Y16_U16_V16_3PLANE_420_UNORM,
                                     viz::ImageFormat::Y16_U16_V16_3PLANE_422_UNORM),
                     testing::Values(viz::YuvModelConversion::YUV_601,
                                     viz::YuvModelConversion::YUV_709,
                                     viz::YuvModelConversion::YUV_2020),
                     testing::Values(viz::YuvRange::ITU_FULL, viz::YuvRange::ITU_NARROW)));

// LUT tests
INSTANTIATE_TEST_SUITE_P(
    ImageLayerLUT, ImageLayer,
    testing::Combine(testing::Values(Source::CUDA_DEVICE), testing::Values(Reuse::DISABLE),
                     testing::Values(UseLut::ENABLE),
                     testing::Values(viz::ImageFormat::R8_UINT, viz::ImageFormat::R8_SINT,
                                     viz::ImageFormat::R16_UINT, viz::ImageFormat::R16_SINT,
                                     viz::ImageFormat::R32_UINT, viz::ImageFormat::R32_SINT),
                     testing::Values(viz::YuvModelConversion::YUV_601),
                     testing::Values(viz::YuvRange::ITU_FULL)));

// LUT with normalize tests
INSTANTIATE_TEST_SUITE_P(
    ImageLayerLUTNorm, ImageLayer,
    testing::Combine(testing::Values(Source::CUDA_DEVICE), testing::Values(Reuse::DISABLE),
                     testing::Values(UseLut::ENABLE_WITH_NORMALIZE),
                     testing::Values(viz::ImageFormat::R8_UINT, viz::ImageFormat::R8_SINT,
                                     viz::ImageFormat::R8_UNORM, viz::ImageFormat::R8_SNORM,
                                     viz::ImageFormat::R16_UINT, viz::ImageFormat::R16_SINT,
                                     viz::ImageFormat::R16_UNORM, viz::ImageFormat::R16_SNORM,
                                     viz::ImageFormat::R32_UINT, viz::ImageFormat::R32_SINT,
                                     viz::ImageFormat::R16_SFLOAT, viz::ImageFormat::R32_SFLOAT),
                     testing::Values(viz::YuvModelConversion::YUV_601),
                     testing::Values(viz::YuvRange::ITU_FULL)));

// RGB is non-native, converted by CUDA kernel or host code
INSTANTIATE_TEST_SUITE_P(ImageLayerConvert, ImageLayer,
                         testing::Combine(testing::Values(Source::HOST, Source::CUDA_DEVICE),
                                          testing::Values(Reuse::DISABLE),
                                          testing::Values(UseLut::DISABLE),
                                          testing::Values(viz::ImageFormat::R8G8B8_UNORM,
                                                          viz::ImageFormat::R8G8B8_SNORM,
                                                          viz::ImageFormat::R8G8B8_SRGB),
                                          testing::Values(viz::YuvModelConversion::YUV_601),
                                          testing::Values(viz::YuvRange::ITU_FULL)));

// depth format tests
INSTANTIATE_TEST_SUITE_P(ImageLayerDepth, ImageLayer,
                         testing::Combine(testing::Values(Source::HOST, Source::CUDA_DEVICE),
                                          testing::Values(Reuse::DISABLE, Reuse::ENABLE),
                                          testing::Values(UseLut::DISABLE),
                                          testing::Values(viz::ImageFormat::D32_SFLOAT),
                                          testing::Values(viz::YuvModelConversion::YUV_601),
                                          testing::Values(viz::YuvRange::ITU_FULL)));

TEST_F(ImageLayer, ImageCudaArray) {
  constexpr viz::ImageFormat kFormat = viz::ImageFormat::R8G8B8A8_UNORM;

  CUarray array;

  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginImageLayer());
  // Cuda array support is not yet implemented, the test below should throw. Add the cuda array
  // test to the test case above when cuda array support is implemented.
  EXPECT_THROW(viz::ImageCudaArray(kFormat, array), std::runtime_error);
  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());
}

/**
 * Switch between host data and device data each frame to make sure that layers are not reused.
 *
 * Background: Holoviz is maintaining a layer cache to avoid re-creating resources each frame
 * to improve performance. The function `Layer::can_be_reused()` is used to check if a layer can
 * be reused. For the image layer there it's special case when switching from host to device memory,
 * That special case is tested here.
 */
TEST_F(ImageLayer, SwitchHostDevice) {
  const viz::ImageFormat kFormat = viz::ImageFormat::R8G8B8A8_UNORM;

  viz::CudaService cuda_service(0);
  const viz::CudaService::ScopedPush cuda_context = cuda_service.PushContext();

  for (uint32_t i = 0; i < 3; ++i) {
    SetupData(kFormat, i);

    viz::UniqueCUdeviceptr device_ptr;
    device_ptr.reset([this] {
      CUdeviceptr device_ptr;
      EXPECT_EQ(cuMemAlloc(&device_ptr, color_data_.size()), CUDA_SUCCESS);
      return device_ptr;
    }());

    EXPECT_EQ(cuMemcpyHtoD(device_ptr.get(), color_data_.data(), color_data_.size()), CUDA_SUCCESS);

    EXPECT_NO_THROW(viz::Begin());

    EXPECT_NO_THROW(viz::BeginImageLayer());

    if (i & 1) {
      EXPECT_NO_THROW(
          viz::ImageHost(width_, height_, kFormat, reinterpret_cast<void*>(color_data_.data())));
    } else {
      EXPECT_NO_THROW(viz::ImageCudaDevice(width_, height_, kFormat, device_ptr.get()));
    }

    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::End());

    if (!CompareColorResult()) { break; }
  }
}

TEST_F(ImageLayer, Errors) {
  constexpr viz::ImageFormat kFormat = viz::ImageFormat::R8G8B8A8_UNORM;
  constexpr viz::ImageFormat kLutFormat = viz::ImageFormat::R8G8B8A8_UNORM;

  std::vector<uint8_t> host_data(width_ * height_ * 4);
  std::vector<uint32_t> lut(lut_size_);

  viz::CudaService cuda_service(0);
  const viz::CudaService::ScopedPush cuda_context = cuda_service.PushContext();

  viz::UniqueCUdeviceptr device_ptr;
  device_ptr.reset([this] {
    CUdeviceptr device_ptr;
    EXPECT_EQ(cuMemAlloc(&device_ptr, width_ * height_ * 4), CUDA_SUCCESS);
    return device_ptr;
  }());

  CUarray array;

  EXPECT_NO_THROW(viz::Begin());

  // it's an error to specify both a host and a device image for a layer.
  for (auto&& format : {viz::ImageFormat::R8G8B8A8_UNORM, viz::ImageFormat::D32_SFLOAT}) {
    EXPECT_NO_THROW(viz::BeginImageLayer());
    EXPECT_NO_THROW(
        viz::ImageHost(width_, height_, format, reinterpret_cast<void*>(host_data.data())));
    EXPECT_THROW(viz::ImageCudaDevice(width_, height_, format, device_ptr.get()),
                 std::runtime_error);
    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::BeginImageLayer());
    EXPECT_NO_THROW(viz::ImageCudaDevice(width_, height_, format, device_ptr.get()));
    EXPECT_THROW(viz::ImageHost(width_, height_, format, reinterpret_cast<void*>(host_data.data())),
                 std::runtime_error);
    EXPECT_NO_THROW(viz::EndLayer());
  }

  // it's an error to call image functions without calling BeginImageLayer first
  EXPECT_THROW(viz::ImageCudaDevice(width_, height_, kFormat, device_ptr.get()),
               std::runtime_error);
  EXPECT_THROW(viz::ImageHost(width_, height_, kFormat, reinterpret_cast<void*>(host_data.data())),
               std::runtime_error);
  EXPECT_THROW(viz::ImageCudaArray(kFormat, array), std::runtime_error);
  EXPECT_THROW(viz::LUT(lut_size_, kLutFormat, lut.size() * sizeof(uint32_t), lut.data()),
               std::runtime_error);
  EXPECT_THROW(viz::ImageComponentMapping(viz::ComponentSwizzle::IDENTITY,
                                          viz::ComponentSwizzle::IDENTITY,
                                          viz::ComponentSwizzle::IDENTITY,
                                          viz::ComponentSwizzle::IDENTITY),
               std::runtime_error);
  EXPECT_THROW(viz::ImageYuvModelConversion(viz::YuvModelConversion::YUV_601), std::runtime_error);
  EXPECT_THROW(viz::ImageYuvRange(viz::YuvRange::ITU_FULL), std::runtime_error);
  EXPECT_THROW(viz::ImageChromaLocation(viz::ChromaLocation::COSITED_EVEN,
                                        viz::ChromaLocation::COSITED_EVEN),
               std::runtime_error);

  // it's an error to call BeginImageLayer again without calling EndLayer
  EXPECT_NO_THROW(viz::BeginImageLayer());
  EXPECT_THROW(viz::BeginImageLayer(), std::runtime_error);
  EXPECT_NO_THROW(viz::EndLayer());

  // it's an error to call image functions when a different layer is active
  EXPECT_NO_THROW(viz::BeginGeometryLayer());
  EXPECT_THROW(viz::ImageCudaDevice(width_, height_, kFormat, device_ptr.get()),
               std::runtime_error);
  EXPECT_THROW(viz::ImageHost(width_, height_, kFormat, reinterpret_cast<void*>(host_data.data())),
               std::runtime_error);
  EXPECT_THROW(viz::ImageCudaArray(kFormat, array), std::runtime_error);
  EXPECT_THROW(viz::LUT(lut_size_, kLutFormat, lut.size() * sizeof(uint32_t), lut.data()),
               std::runtime_error);
  EXPECT_THROW(viz::ImageComponentMapping(viz::ComponentSwizzle::IDENTITY,
                                          viz::ComponentSwizzle::IDENTITY,
                                          viz::ComponentSwizzle::IDENTITY,
                                          viz::ComponentSwizzle::IDENTITY),
               std::runtime_error);
  EXPECT_THROW(viz::ImageYuvModelConversion(viz::YuvModelConversion::YUV_601), std::runtime_error);
  EXPECT_THROW(viz::ImageYuvRange(viz::YuvRange::ITU_FULL), std::runtime_error);
  EXPECT_THROW(viz::ImageChromaLocation(viz::ChromaLocation::COSITED_EVEN,
                                        viz::ChromaLocation::COSITED_EVEN),
               std::runtime_error);
  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());
}

class ImageLayerFormat : public TestHeadless,
                         public testing::WithParamInterface<std::tuple<Source, viz::ImageFormat>> {
};

TEST_P(ImageLayerFormat, RowPitch) {
  const Source source = std::get<0>(GetParam());
  const viz::ImageFormat format = std::get<1>(GetParam());

  viz::CudaService cuda_service(0);
  const viz::CudaService::ScopedPush cuda_context = cuda_service.PushContext();

  SetupData(format);

  const size_t row_size = color_data_.size() / height_;
  // add some extra data for each row
  const size_t row_pitch = row_size + 7;

  // copy color data to array with row pitch
  std::unique_ptr<uint8_t> host_ptr;
  host_ptr.reset(new uint8_t[height_ * row_pitch]);
  const uint8_t* src = color_data_.data();
  uint8_t* dst = host_ptr.get();
  for (uint32_t y = 0; y < height_; ++y) {
    memcpy(dst, src, row_size);
    src += row_size;
    dst += row_pitch;
  }

  viz::UniqueCUdeviceptr device_ptr;
  if (source == Source::CUDA_DEVICE) {
    device_ptr.reset([this, row_pitch] {
      CUdeviceptr device_ptr;
      EXPECT_EQ(cuMemAlloc(&device_ptr, height_ * row_pitch), CUDA_SUCCESS);
      return device_ptr;
    }());

    EXPECT_EQ(cuMemcpyHtoD(device_ptr.get(), host_ptr.get(), height_ * row_pitch), CUDA_SUCCESS);
  }

  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginImageLayer());

  if (source == Source::HOST) {
    EXPECT_NO_THROW(viz::ImageHost(
        width_, height_, format, reinterpret_cast<void*>(host_ptr.get()), row_pitch));
  } else {
    EXPECT_NO_THROW(viz::ImageCudaDevice(width_, height_, format, device_ptr.get(), row_pitch));
  }

  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());

  CompareColorResult();
}

INSTANTIATE_TEST_SUITE_P(ImageLayer, ImageLayerFormat,
                         testing::Combine(testing::Values(Source::HOST, Source::CUDA_DEVICE),
                                          testing::Values(viz::ImageFormat::R8G8B8A8_UNORM,
                                                          viz::ImageFormat::R8G8B8_UNORM)));

class MultiGPU : public TestHeadless,
                 public testing::WithParamInterface<std::tuple<UseStream, viz::ImageFormat>> {};

TEST_P(MultiGPU, Peer) {
  const bool use_stream = (std::get<0>(GetParam()) == UseStream::ENABLE);
  const viz::ImageFormat color_format = std::get<1>(GetParam());

  int device_count = 0;
  CudaCheck(cuDeviceGetCount(&device_count));
  if (device_count < 2) { GTEST_SKIP() << "Single GPU system, skipping test."; }

  SetupData(color_format);

  // iterate over all available devices, allocate CUDA memory on that device, draw it to an
  // image layer, read it back
  // this will exercise different paths depending on the CUDA memory being on the display device
  // or not
  for (int source = 0; source < device_count; ++source) {
    CUdevice device;
    CudaCheck(cuDeviceGet(&device, source));
    CUcontext context;
    CudaCheck(cuDevicePrimaryCtxRetain(&context, device));

    SetCUDADevice(source);

    CudaCheck(cuCtxPushCurrent(context));

    {
      CUstream stream = 0;
      if (use_stream) { CudaCheck(cuStreamCreate(&stream, CU_STREAM_DEFAULT)); }
      EXPECT_NO_THROW(viz::SetCudaStream(stream));

      viz::UniqueCUdeviceptr color_device_ptr;

      color_device_ptr.reset([this] {
        CUdeviceptr device_ptr;
        EXPECT_EQ(cuMemAlloc(&device_ptr, color_data_.size()), CUDA_SUCCESS);
        return device_ptr;
      }());

      EXPECT_EQ(
          cuMemcpyHtoDAsync(color_device_ptr.get(), color_data_.data(), color_data_.size(), stream),
          CUDA_SUCCESS);

      EXPECT_NO_THROW(viz::Begin());
      EXPECT_NO_THROW(viz::BeginImageLayer());
      EXPECT_NO_THROW(viz::ImageCudaDevice(width_, height_, color_format, color_device_ptr.get()));
      EXPECT_NO_THROW(viz::EndLayer());
      EXPECT_NO_THROW(viz::End());

      CompareColorResult();
    }

    CudaCheck(cuCtxPopCurrent(nullptr));

    CudaCheck(cuDevicePrimaryCtxRelease(device));
  }
}

INSTANTIATE_TEST_SUITE_P(ImageLayer, MultiGPU,
                         testing::Combine(testing::Values(UseStream::DISABLE, UseStream::ENABLE),
                                          testing::Values(viz::ImageFormat::R8G8B8A8_UNORM,
                                                          viz::ImageFormat::R8G8B8_UNORM)));

class ImageLayerSwizzle : public TestHeadless,
                          public testing::WithParamInterface<std::array<viz::ComponentSwizzle, 4>> {
};

TEST_P(ImageLayerSwizzle, Swizzle) {
  const std::array<viz::ComponentSwizzle, 4> mapping = GetParam();

  viz::CudaService cuda_service(0);
  const viz::CudaService::ScopedPush cuda_context = cuda_service.PushContext();

  const viz::ImageFormat image_format = viz::ImageFormat::R8G8B8A8_UNORM;
  SetupData(image_format);

  viz::UniqueCUdeviceptr device_ptr;
  device_ptr.reset([this] {
    CUdeviceptr device_ptr;
    EXPECT_EQ(cuMemAlloc(&device_ptr, color_data_.size()), CUDA_SUCCESS);
    return device_ptr;
  }());

  EXPECT_EQ(cuMemcpyHtoD(device_ptr.get(), color_data_.data(), color_data_.size()), CUDA_SUCCESS);

  for (uint32_t index = 0; index < width_ * height_; ++index) {
    uint8_t pixel[4];
    for (uint32_t component = 0; component < 4; ++component) {
      switch (mapping[component]) {
        case viz::ComponentSwizzle::IDENTITY:
          pixel[component] = color_data_[index * 4 + component];
          break;
        case viz::ComponentSwizzle::ZERO:
          pixel[component] = 0x00;
          break;
        case viz::ComponentSwizzle::ONE:
          pixel[component] = 0xFF;
          break;
        case viz::ComponentSwizzle::R:
          pixel[component] = color_data_[index * 4 + 0];
          break;
        case viz::ComponentSwizzle::G:
          pixel[component] = color_data_[index * 4 + 1];
          break;
        case viz::ComponentSwizzle::B:
          pixel[component] = color_data_[index * 4 + 2];
          break;
        case viz::ComponentSwizzle::A:
          pixel[component] = color_data_[index * 4 + 3];
          break;
        default:
          EXPECT_TRUE(false) << "Unhandled component swizzle";
      }
    }
    if (pixel[3] != 0xFF) {
      float alpha = float(pixel[3]) / 255.F;
      color_data_[index * 4 + 0] = uint8_t(float(pixel[0]) * alpha + 0.5F);
      color_data_[index * 4 + 1] = uint8_t(float(pixel[1]) * alpha + 0.5F);
      color_data_[index * 4 + 2] = uint8_t(float(pixel[2]) * alpha + 0.5F);
      color_data_[index * 4 + 3] = pixel[3];
    } else {
      color_data_[index * 4 + 0] = pixel[0];
      color_data_[index * 4 + 1] = pixel[1];
      color_data_[index * 4 + 2] = pixel[2];
      color_data_[index * 4 + 3] = pixel[3];
    }
  }

  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginImageLayer());

  EXPECT_NO_THROW(viz::ImageComponentMapping(mapping[0], mapping[1], mapping[2], mapping[3]));

  EXPECT_NO_THROW(viz::ImageCudaDevice(width_, height_, image_format, device_ptr.get()));

  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());

  CompareColorResult();
}

INSTANTIATE_TEST_SUITE_P(
    ImageLayer, ImageLayerSwizzle,
    testing::Values(std::array<viz::ComponentSwizzle, 4>{viz::ComponentSwizzle::IDENTITY,
                                                         viz::ComponentSwizzle::IDENTITY,
                                                         viz::ComponentSwizzle::IDENTITY,
                                                         viz::ComponentSwizzle::IDENTITY},
                    std::array<viz::ComponentSwizzle, 4>{viz::ComponentSwizzle::ZERO,
                                                         viz::ComponentSwizzle::ZERO,
                                                         viz::ComponentSwizzle::ZERO,
                                                         viz::ComponentSwizzle::ZERO},
                    std::array<viz::ComponentSwizzle, 4>{viz::ComponentSwizzle::ONE,
                                                         viz::ComponentSwizzle::ONE,
                                                         viz::ComponentSwizzle::ONE,
                                                         viz::ComponentSwizzle::ONE},
                    std::array<viz::ComponentSwizzle, 4>{viz::ComponentSwizzle::R,
                                                         viz::ComponentSwizzle::R,
                                                         viz::ComponentSwizzle::R,
                                                         viz::ComponentSwizzle::R},
                    std::array<viz::ComponentSwizzle, 4>{viz::ComponentSwizzle::G,
                                                         viz::ComponentSwizzle::G,
                                                         viz::ComponentSwizzle::G,
                                                         viz::ComponentSwizzle::G},
                    std::array<viz::ComponentSwizzle, 4>{viz::ComponentSwizzle::B,
                                                         viz::ComponentSwizzle::B,
                                                         viz::ComponentSwizzle::B,
                                                         viz::ComponentSwizzle::B},
                    std::array<viz::ComponentSwizzle, 4>{viz::ComponentSwizzle::A,
                                                         viz::ComponentSwizzle::A,
                                                         viz::ComponentSwizzle::A,
                                                         viz::ComponentSwizzle::A},
                    std::array<viz::ComponentSwizzle, 4>{viz::ComponentSwizzle::A,
                                                         viz::ComponentSwizzle::B,
                                                         viz::ComponentSwizzle::G,
                                                         viz::ComponentSwizzle::R}));
