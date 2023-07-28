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

#include <cmath>
#include <string>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda/cuda_service.hpp>
#include <holoviz/holoviz.hpp>
#include "headless_fixture.hpp"

namespace viz = holoscan::viz;

enum class Source { HOST, CUDA_DEVICE };

enum class Reuse { DISABLE, ENABLE };

enum class UseLut { DISABLE, ENABLE, ENABLE_WITH_NORMALIZE };

enum class UseDepth { DISABLE, ENABLE };

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

std::ostream& operator<<(std::ostream& os, const UseDepth& use_depth) {
  switch (use_depth) {
    CASE(UseDepth::DISABLE)
    CASE(UseDepth::ENABLE)
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
    CASE(ImageFormat::R16_UINT)
    CASE(ImageFormat::R16_SFLOAT)
    CASE(ImageFormat::R32_UINT)
    CASE(ImageFormat::R32_SFLOAT)
    CASE(ImageFormat::R8G8B8_UNORM)
    CASE(ImageFormat::B8G8R8_UNORM)
    CASE(ImageFormat::R8G8B8A8_UNORM)
    CASE(ImageFormat::B8G8R8A8_UNORM)
    CASE(ImageFormat::R16G16B16A16_UNORM)
    CASE(ImageFormat::R16G16B16A16_SFLOAT)
    CASE(ImageFormat::R32G32B32A32_SFLOAT)
    CASE(ImageFormat::R8_UNORM)
    CASE(ImageFormat::R16_UNORM)
    CASE(ImageFormat::D32_SFLOAT)
    default:
      os.setstate(std::ios_base::failbit);
  }
  return os;
}
}  // namespace viz
}  // namespace holoscan

#undef CASE

class ImageLayer : public TestHeadless,
                   public testing::WithParamInterface<std::tuple<Source, Reuse, UseLut, UseDepth>> {
};

TEST_P(ImageLayer, Image) {
  const Source source = std::get<0>(GetParam());
  const bool reuse = std::get<1>(GetParam()) == Reuse::ENABLE;
  const UseLut use_lut = std::get<2>(GetParam());
  const bool use_depth = std::get<3>(GetParam()) == UseDepth::ENABLE;

  const viz::ImageFormat kLutFormat = viz::ImageFormat::R8G8B8A8_UNORM;

  viz::ImageFormat color_format = viz::ImageFormat::R8G8B8A8_UNORM;
  if (use_lut == UseLut::ENABLE) {
    color_format = viz::ImageFormat::R8_UINT;
  } else if (use_lut == UseLut::ENABLE_WITH_NORMALIZE) {
    color_format = viz::ImageFormat::R8_UNORM;
  }

  SetupData(color_format);

  viz::ImageFormat depth_format = viz::ImageFormat::D32_SFLOAT;
  if (use_depth) {
    SetupData(depth_format);
  } else {
    depth_data_ = std::vector<float>(width_ * height_ * 1 * sizeof(float), 0.f);
  }

  std::vector<uint32_t> lut;
  std::vector<uint8_t> data_with_lut;

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
    data_with_lut.resize(width_ * height_ * sizeof(uint32_t));
    if (use_lut == UseLut::ENABLE) {
      for (size_t index = 0; index < width_ * height_; ++index) {
        const uint8_t src_value = color_data_[index];
        reinterpret_cast<uint32_t*>(data_with_lut.data())[index] = lut[src_value];
      }
    } else if (use_lut == UseLut::ENABLE_WITH_NORMALIZE) {
      for (size_t index = 0; index < width_ * height_; ++index) {
        const float offset = (float(color_data_[index]) / 255.f) * lut_size_;
        uint32_t val0 = lut[uint32_t(offset - 0.5f)];
        uint32_t val1 = lut[uint32_t(offset + 0.5f)];
        float dummy;
        float frac = std::modf(offset, &dummy);

        float r = float(val1 & 0xFF) + frac * (float(val0 & 0xFF) - float(val1 & 0xFF));
        float g = float((val1 & 0xFF00) >> 8) +
                  frac * (float((val0 & 0xFF00) >> 8) - float((val1 & 0xFF00) >> 8));
        float b = float((val1 & 0xFF0000) >> 16) +
                  frac * (float((val0 & 0xFF0000) >> 16) - float((val1 & 0xFF0000) >> 16));
        float a = float((val1 & 0xFF000000) >> 24) +
                  frac * (float((val0 & 0xFF000000) >> 24) - float((val1 & 0xFF000000) >> 24));

        reinterpret_cast<uint32_t*>(data_with_lut.data())[index] =
            uint32_t(r + 0.5f) | (uint32_t(g + 0.5f) << 8) | (uint32_t(b + 0.5f) << 16) |
            (uint32_t(a + 0.5f) << 24);
      }
    }
  }

  for (uint32_t i = 0; i < (reuse ? 2 : 1); ++i) {
    EXPECT_NO_THROW(viz::Begin());

    EXPECT_NO_THROW(viz::BeginImageLayer());

    if (use_lut != UseLut::DISABLE) {
      EXPECT_NO_THROW(viz::LUT(lut_size_,
                               kLutFormat,
                               lut.size() * sizeof(uint32_t),
                               lut.data(),
                               use_lut == UseLut::ENABLE_WITH_NORMALIZE));
    }

    switch (source) {
      case Source::HOST:
        EXPECT_NO_THROW(viz::ImageHost(
            width_, height_, color_format, reinterpret_cast<void*>(color_data_.data())));
        break;
      case Source::CUDA_DEVICE:
        EXPECT_NO_THROW(
            viz::ImageCudaDevice(width_, height_, color_format, color_device_ptr.get()));
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

  if (use_lut != UseLut::DISABLE) {
    std::swap(color_data_, data_with_lut);
    CompareColorResult();
    std::swap(data_with_lut, color_data_);
  } else {
    CompareColorResult();
  }

  CompareDepthResult();
}

INSTANTIATE_TEST_SUITE_P(ImageLayer, ImageLayer,
                         testing::Combine(testing::Values(Source::HOST, Source::CUDA_DEVICE),
                                          testing::Values(Reuse::DISABLE, Reuse::ENABLE),
                                          testing::Values(UseLut::DISABLE, UseLut::ENABLE),
                                          testing::Values(UseDepth::DISABLE, UseDepth::ENABLE)));

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
