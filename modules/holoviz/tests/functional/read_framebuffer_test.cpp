/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>

#include <cuda/cuda_service.hpp>
#include <holoviz/holoviz.hpp>

#include "test_fixture.hpp"

namespace viz = holoscan::viz;

class ReadFramebuffer : public TestHeadless {};

/**
 * Test the `row_pitch` parameter
 */
TEST_F(ReadFramebuffer, RowPitch) {
  // setup source color data
  const viz::ImageFormat color_format = viz::ImageFormat::R8G8B8A8_UNORM;
  SetupData(color_format);

  viz::CudaService cuda_service(0);
  viz::CudaService::ScopedPush cuda_context = cuda_service.PushContext();

  viz::UniqueCUdeviceptr device_ptr;
  device_ptr.reset([this] {
    CUdeviceptr device_ptr;
    EXPECT_EQ(cuMemAlloc(&device_ptr, color_data_.size()), CUDA_SUCCESS);
    return device_ptr;
  }());

  EXPECT_EQ(cuMemcpyHtoD(device_ptr.get(), color_data_.data(), color_data_.size()), CUDA_SUCCESS);

  // draw to window using a image layer
  EXPECT_NO_THROW(viz::Begin());

  EXPECT_NO_THROW(viz::BeginImageLayer());

  EXPECT_NO_THROW(viz::ImageCudaDevice(width_, height_, color_format, device_ptr.get()));

  EXPECT_NO_THROW(viz::EndLayer());

  EXPECT_NO_THROW(viz::End());

  // read contiguous color_data_ into a framebuffer with a larger row pitch (non-contiguous)
  const size_t row_pitch = width_ * 4 * sizeof(uint8_t) + 13;
  viz::UniqueCUdeviceptr read_device_ptr;
  read_device_ptr.reset([this, row_pitch] {
    CUdeviceptr device_ptr;
    EXPECT_EQ(cuMemAlloc(&device_ptr, row_pitch * height_), CUDA_SUCCESS);
    return device_ptr;
  }());

  EXPECT_NO_THROW(viz::ReadFramebuffer(
      color_format, width_, height_, row_pitch * height_, read_device_ptr.get(), row_pitch));

  std::vector<uint8_t> read_host_data;
  read_host_data.resize(row_pitch * height_);
  EXPECT_EQ(cuMemcpyDtoH(read_host_data.data(), read_device_ptr.get(), read_host_data.size()),
            CUDA_SUCCESS);

  // compare
  bool different = false;
  for (uint32_t y = 0; y < height_ && !different; ++y) {
    for (uint32_t x = 0; x < width_ && !different; ++x) {
      const uint8_t* ref = color_data_.data() + (y * width_ * 4) + x * 4;
      const uint8_t* read = read_host_data.data() + (y * row_pitch) + x * 4;
      if ((ref[0] != read[0]) || (ref[1] != read[1]) || (ref[2] != read[2]) ||
          (ref[3] != read[3])) {
        EXPECT_TRUE(false) << "Data mismatch at column " << x << ", row " << y;
        different = true;
      }
    }
  }
}

TEST_F(ReadFramebuffer, Errors) {
  viz::CudaService::ScopedPush cuda_context;
  viz::UniqueCUdeviceptr color_device_ptr;

  viz::CudaService cuda_service(0);

  cuda_context = cuda_service.PushContext();
  color_device_ptr.reset([this] {
    CUdeviceptr device_ptr;
    EXPECT_EQ(cuMemAlloc(&device_ptr, sizeof(uint32_t)), CUDA_SUCCESS);
    return device_ptr;
  }());

  // passing case
  EXPECT_NO_THROW(viz::ReadFramebuffer(
      viz::ImageFormat::R8G8B8A8_UNORM, 1, 1, 1 * 1 * 4 * sizeof(uint8_t), color_device_ptr.get()));

  // it's an error to call ReadFrameBuffer with unsupported image formats
  EXPECT_THROW(
      viz::ReadFramebuffer(
          viz::ImageFormat::D16_UNORM, 1, 1, 1 * 1 * sizeof(uint16_t), color_device_ptr.get()),
      std::runtime_error);

  // it's an error to if the data size is too small
  EXPECT_THROW(
      viz::ReadFramebuffer(viz::ImageFormat::R8G8B8A8_UNORM, 1, 1, 1, color_device_ptr.get()),
      std::runtime_error);
}
