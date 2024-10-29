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

#include "test_fixture.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda/cuda_service.hpp>

namespace viz = holoscan::viz;

template <typename T>
void Fill(void* data, size_t elements, T or_mask = T(),
          T and_mask = std::numeric_limits<T>::max()) {
  // fill volume with random data
  for (size_t index = 0; index < elements; ++index) {
    reinterpret_cast<T*>(data)[index] = (static_cast<T>(std::rand()) | or_mask) & and_mask;
  }
}

template <>
void Fill(void* data, size_t elements, float min, float max) {
  // fill volume with random data
  for (size_t index = 0; index < elements; ++index) {
    reinterpret_cast<float*>(data)[index] =
        std::max(min, std::min(max, static_cast<float>(std::rand()) / RAND_MAX));
  }
}

void TestBase::SetUp() {
  if (!(init_flags_ & viz::InitFlags::HEADLESS)) {
    if (glfwInit() == GLFW_FALSE) {
      const char* description;
      int code = glfwGetError(&description);
      ASSERT_EQ(code, GLFW_PLATFORM_UNAVAILABLE) << "Expected `GLFW_PLATFORM_UNAVAILABLE` but got `"
                                                 << code << "`: `" << description << "`";
      GTEST_SKIP() << "No display server available, skipping test." << description;
    }
  }

  ASSERT_NO_THROW(viz::Init(width_, height_, "Holoviz test", init_flags_));
  initialized_ = true;
}

void TestBase::TearDown() {
  if (initialized_) {
    ASSERT_NO_THROW(viz::Shutdown());
    initialized_ = false;
  }
}

void TestBase::SetCUDADevice(uint32_t device_ordinal) {
  device_ordinal_ = device_ordinal;
}

void TestBase::SetupData(viz::ImageFormat format, uint32_t rand_seed) {
  std::srand(rand_seed);

  uint32_t channels;
  uint32_t component_size;
  uint32_t elements;
  switch (format) {
    case viz::ImageFormat::R8_UINT:
    case viz::ImageFormat::R8_SINT:
      channels = 1;
      component_size = sizeof(uint8_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<uint8_t>(color_data_.data(), width_ * height_, 0x00, lut_size_ - 1);
      break;
    case viz::ImageFormat::R8_UNORM:
    case viz::ImageFormat::R8_SRGB:
      channels = 1;
      component_size = sizeof(uint8_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<uint8_t>(color_data_.data(), width_ * height_);
      break;
    case viz::ImageFormat::R8_SNORM:
      channels = 1;
      component_size = sizeof(int8_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<int8_t>(color_data_.data(), width_ * height_);
      break;
    case viz::ImageFormat::R16_UINT:
    case viz::ImageFormat::R16_SINT:
      channels = 1;
      component_size = sizeof(uint16_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<uint16_t>(color_data_.data(), width_ * height_, 0x00, lut_size_ - 1);
      break;
    case viz::ImageFormat::R16_UNORM:
      channels = 1;
      component_size = sizeof(uint16_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<uint16_t>(color_data_.data(), width_ * height_);
      break;
    case viz::ImageFormat::R16_SNORM:
      channels = 1;
      component_size = sizeof(int16_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<int16_t>(color_data_.data(), width_ * height_);
      break;
    case viz::ImageFormat::R32_UINT:
    case viz::ImageFormat::R32_SINT:
      channels = 1;
      component_size = sizeof(uint32_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<uint32_t>(color_data_.data(), width_ * height_, 0x00, lut_size_ - 1);
      break;
    case viz::ImageFormat::R32_SFLOAT:
      channels = 1;
      component_size = sizeof(float);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<float>(color_data_.data(), width_ * height_, 0.F, float(lut_size_ - 1));
      break;
    case viz::ImageFormat::R8G8B8_UNORM:
    case viz::ImageFormat::R8G8B8_SRGB:
      channels = 3;
      component_size = sizeof(uint8_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<uint8_t>(color_data_.data(), width_ * height_ * channels);
      break;
    case viz::ImageFormat::R8G8B8_SNORM:
      channels = 3;
      component_size = sizeof(int8_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<int8_t>(color_data_.data(), width_ * height_ * channels, 0x0, 0x7F);
      break;
    case viz::ImageFormat::R8G8B8A8_UNORM:
    case viz::ImageFormat::R8G8B8A8_SRGB:
    case viz::ImageFormat::B8G8R8A8_UNORM:
    case viz::ImageFormat::B8G8R8A8_SRGB:
    case viz::ImageFormat::A8B8G8R8_UNORM_PACK32:
    case viz::ImageFormat::A8B8G8R8_SRGB_PACK32:
      channels = 4;
      component_size = sizeof(uint8_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<uint32_t>(color_data_.data(), width_ * height_, 0xFF000000);
      break;
    case viz::ImageFormat::R8G8B8A8_SNORM:
      channels = 4;
      component_size = sizeof(int8_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<uint32_t>(color_data_.data(), width_ * height_, 0x7F000000, 0x7F7F7F7F);
      break;
    case viz::ImageFormat::R16G16B16A16_UNORM:
      channels = 4;
      component_size = sizeof(uint16_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<uint64_t>(color_data_.data(), width_ * height_, 0xFFFF000000000000UL);
      break;
    case viz::ImageFormat::R16G16B16A16_SNORM:
      channels = 4;
      component_size = sizeof(int16_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<uint64_t>(
          color_data_.data(), width_ * height_, 0x7FFF000000000000UL, 0x7FFF7FFF7FFF7FFFUL);
      break;
    case viz::ImageFormat::D16_UNORM:
      channels = 1;
      component_size = sizeof(uint16_t);
      depth_data_.resize(width_ * height_ * channels * component_size);
      Fill<uint16_t>(depth_data_.data(), width_ * height_);
      break;
    case viz::ImageFormat::X8_D24_UNORM:
      channels = 1;
      component_size = sizeof(uint32_t);
      depth_data_.resize(width_ * height_ * channels * component_size);
      Fill<uint32_t>(depth_data_.data(), width_ * height_, 0, 0x00FFFFFF);
      break;
    case viz::ImageFormat::D32_SFLOAT:
      channels = 1;
      component_size = sizeof(float);
      depth_data_.resize(width_ * height_ * channels * component_size);
      Fill<float>(depth_data_.data(), width_ * height_, 0.F, std::numeric_limits<float>::max());
      break;
    case viz::ImageFormat::A2B10G10R10_UNORM_PACK32:
    case viz::ImageFormat::A2R10G10B10_UNORM_PACK32:
      channels = 1;
      component_size = sizeof(uint32_t);
      color_data_.resize(width_ * height_ * channels * component_size);
      Fill<uint32_t>(
          color_data_.data(), width_ * height_, 0b1100'0000'0000'0000'0000'0000'0000'0000);
      break;
    default:
      ASSERT_TRUE(false) << "Unsupported image format " << static_cast<int>(format);
  }
}

void TestBase::ReadColorData(std::vector<uint8_t>& color_data) {
  const size_t data_size = width_ * height_ * sizeof(uint8_t) * 4;

  viz::CudaService cuda_service(device_ordinal_);
  const viz::CudaService::ScopedPush cuda_context = cuda_service.PushContext();

  viz::UniqueCUdeviceptr device_ptr;
  device_ptr.reset([this](size_t data_size) {
    CUdeviceptr device_ptr;
    EXPECT_EQ(cuMemAlloc(&device_ptr, data_size), CUDA_SUCCESS);
    return device_ptr;
  }(data_size));

  ASSERT_TRUE(device_ptr);

  ASSERT_NO_THROW(viz::ReadFramebuffer(
      viz::ImageFormat::R8G8B8A8_UNORM, width_, height_, data_size, device_ptr.get()));

  color_data.resize(data_size);

  ASSERT_EQ(cuMemcpyDtoH(color_data.data(), device_ptr.get(), color_data.size()), CUDA_SUCCESS);
}

void TestBase::ReadDepthData(std::vector<float>& depth_data) {
  const size_t data_size = width_ * height_ * sizeof(float);

  viz::CudaService cuda_service(device_ordinal_);
  const viz::CudaService::ScopedPush cuda_context = cuda_service.PushContext();

  viz::UniqueCUdeviceptr device_ptr;
  device_ptr.reset([this](size_t data_size) {
    CUdeviceptr device_ptr;
    EXPECT_EQ(cuMemAlloc(&device_ptr, data_size), CUDA_SUCCESS);
    return device_ptr;
  }(data_size));

  ASSERT_TRUE(device_ptr);

  ASSERT_NO_THROW(viz::ReadFramebuffer(
      viz::ImageFormat::D32_SFLOAT, width_, height_, data_size, device_ptr.get()));

  depth_data.resize(data_size);

  ASSERT_EQ(cuMemcpyDtoH(depth_data.data(), device_ptr.get(), depth_data.size()), CUDA_SUCCESS);
}

static std::string BuildFileName(const std::string& end) {
  const testing::TestInfo* test_info = testing::UnitTest::GetInstance()->current_test_info();

  std::string file_name;
  file_name += test_info->test_suite_name();
  file_name += "_";
  file_name += test_info->name();
  file_name += "_";
  file_name += end;
  file_name += ".png";

  // parameterized tests have '/', replace with '_'
  std::replace(file_name.begin(), file_name.end(), '/', '_');

  return file_name;
}

bool TestBase::CompareColorResult(uint8_t absolute_error) {
  const uint32_t components = color_data_.size() / (width_ * height_);
  if ((components != 1) && (components != 3) && (components != 4)) {
    EXPECT_TRUE(false) << "Can compare R8_UNORM, R8G8B8_UNORM or R8G8B8A8_UNORM data only";
    return false;
  }

  std::vector<uint8_t> color_data;
  ReadColorData(color_data);

  for (size_t index = 0; index < width_ * height_; ++index) {
    bool different = false;
    for (uint32_t component = 0; component < components; ++component) {
      different |= std::abs(color_data_[index * components + component] -
                            color_data[index * 4 + component]) > absolute_error;
    }
    if (different) {
      const std::string ref_file_name = BuildFileName("color_ref");
      const std::string fail_file_name = BuildFileName("color_fail");

      stbi_write_png(ref_file_name.c_str(), width_, height_, components, color_data_.data(), 0);
      stbi_write_png(fail_file_name.c_str(), width_, height_, 4, color_data.data(), 0);

      EXPECT_TRUE(false) << "Data mismatch, wrote images to " << ref_file_name << " and "
                         << fail_file_name;
      return false;
    }
  }
  return true;
}

bool TestBase::CompareDepthResult() {
  if (depth_data_.size() != width_ * height_ * sizeof(float)) {
    EXPECT_TRUE(false) << "Can compare D32_SFLOAT data only";
    return false;
  }

  std::vector<float> depth_data;
  ReadDepthData(depth_data);

  for (size_t index = 0; index < depth_data_.size(); ++index) {
    if (depth_data_[index] != depth_data[index]) {
      const std::string ref_file_name = BuildFileName("depth_ref");
      const std::string fail_file_name = BuildFileName("depth_fail");

      // convert to single channel uint8_t assuming depth is between 0...1
      std::vector<uint8_t> image_data(depth_data_.size());
      for (size_t index = 0; index < depth_data_.size(); ++index) {
        image_data[index] = static_cast<uint8_t>(depth_data_[index] * 255.F + 0.5F);
      }
      stbi_write_png(ref_file_name.c_str(), width_, height_, 1, image_data.data(), 0);
      for (size_t index = 0; index < depth_data.size(); ++index) {
        image_data[index] = static_cast<uint8_t>(depth_data[index] * 255.F + 0.5F);
      }
      stbi_write_png(fail_file_name.c_str(), width_, height_, 1, image_data.data(), 0);

      EXPECT_TRUE(false) << "Data mismatch, wrote images to " << ref_file_name << " and "
                         << fail_file_name;
      return false;
    }
  }
  return true;
}

static const char* const crc_instructions = R"(
To check images, set the HOLOVIZ_TEST_GEN_IMAGES environment variable and run the test on
a configuration where the test passes. This will generate images with the `_ref` suffix.
Compare these images with the `_fail` images. Update or add the CRC values of the test
accordingly.
)";

bool TestBase::CompareColorResultCRC32(const std::vector<uint32_t> crc32) {
  std::vector<uint8_t> read_data;
  ReadColorData(read_data);

  const uint32_t read_crc32 = stbiw__crc32(read_data.data(), read_data.size());

  std::string image_file_name;
  bool passed;
  if (std::find(crc32.begin(), crc32.end(), read_crc32) == crc32.end()) {
    image_file_name = BuildFileName("color_fail");

    std::ostringstream str;
    str << "CRC mismatch, expected {" << std::hex << std::setw(8);
    for (auto&& value : crc32) { str << "0x" << value << ", "; }
    str << "} but calculated 0x" << read_crc32 << ", wrote failing image to " << image_file_name
        << ". " << crc_instructions;
    EXPECT_FALSE(true) << str.str();

    passed = false;
  } else {
    if (std::getenv("HOLOVIZ_TEST_GEN_IMAGES")) {
      image_file_name = BuildFileName("color_ref");
      std::cout << "Test passed and HOLOVIZ_TEST_GEN_IMAGES is set, writing image to "
                << image_file_name << ". " << std::endl;
    }
    passed = true;
  }

  if (!image_file_name.empty()) {
    stbi_write_png(image_file_name.c_str(), width_, height_, 4, read_data.data(), 0);
  }

  return passed;
}

bool TestBase::CompareDepthResultCRC32(const std::vector<uint32_t> crc32) {
  std::vector<float> read_data;
  ReadDepthData(read_data);

  const uint32_t read_crc32 = stbiw__crc32(reinterpret_cast<unsigned char*>(read_data.data()),
                                           read_data.size() * sizeof(float));

  std::string image_file_name;
  bool passed;
  if (std::find(crc32.begin(), crc32.end(), read_crc32) == crc32.end()) {
    image_file_name = BuildFileName("depth_fail");

    std::ostringstream str;
    str << "CRC mismatch, expected {" << std::hex << std::setw(8);
    for (auto&& value : crc32) { str << "0x" << value << ", "; }
    str << "} but calculated 0x" << read_crc32 << ", wrote failing image to " << image_file_name
        << ". " << crc_instructions;
    EXPECT_FALSE(true) << str.str();

    passed = false;
  } else {
    if (std::getenv("HOLOVIZ_TEST_GEN_IMAGES")) {
      image_file_name = BuildFileName("depth_ref");
      std::cout << "Test passed and HOLOVIZ_TEST_GEN_IMAGES is set, writing image to "
                << image_file_name << ". " << std::endl;
    }
  }

  if (!image_file_name.empty()) {
    // convert to single channel uint8_t assuming depth is between 0...1
    std::vector<uint8_t> image_data(read_data.size());
    for (size_t index = 0; index < read_data.size(); ++index) {
      image_data[index] = static_cast<uint8_t>(read_data[index] * 255.F + 0.5F);
    }

    stbi_write_png(image_file_name.c_str(), width_, height_, 1, image_data.data(), 0);
  }

  return true;
}
