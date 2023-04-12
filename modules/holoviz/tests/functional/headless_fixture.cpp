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

#include "headless_fixture.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <iomanip>
#include <sstream>

#include <cuda/cuda_service.hpp>

namespace viz = holoscan::viz;

template <typename T>
void Fill(std::vector<uint8_t>& data, size_t elements, T or_mask = T(),
          T and_mask = std::numeric_limits<T>::max()) {
  // fill volume with random data and build histogram
  for (size_t index = 0; index < elements; ++index) {
    if (std::is_same<T, float>::value) {
      reinterpret_cast<T*>(data.data())[index] = static_cast<T>(std::rand()) / RAND_MAX;
    } else {
      reinterpret_cast<T*>(data.data())[index] = (static_cast<T>(std::rand()) | or_mask) & and_mask;
    }
  }
}

void TestHeadless::SetUp() {
  ASSERT_NO_THROW(viz::Init(width_, height_, "Holoviz test", viz::InitFlags::HEADLESS));
}

void TestHeadless::TearDown() {
  ASSERT_NO_THROW(viz::Shutdown());
}

void TestHeadless::SetupData(viz::ImageFormat format, uint32_t rand_seed) {
  std::srand(rand_seed);

  uint32_t channels;
  uint32_t component_size;
  switch (format) {
    case viz::ImageFormat::R8G8B8A8_UNORM:
      channels = 4;
      component_size = sizeof(uint8_t);
      data_.resize(width_ * height_ * channels * component_size);
      Fill<uint32_t>(data_, width_ * height_, 0xFF000000);
      break;
    case viz::ImageFormat::R8_UINT:
      channels = 1;
      component_size = sizeof(uint8_t);
      data_.resize(width_ * height_ * channels * component_size);
      Fill<uint8_t>(data_, width_ * height_, 0x00, lut_size_ - 1);
      break;
    case viz::ImageFormat::R8_UNORM:
      channels = 1;
      component_size = sizeof(uint8_t);
      data_.resize(width_ * height_ * channels * component_size);
      Fill<uint8_t>(data_, width_ * height_, 0x00, 0xFF);
      break;
    default:
      ASSERT_TRUE(false) << "Unsupported image format " << static_cast<int>(format);
  }
}

void TestHeadless::ReadData(std::vector<uint8_t>& read_data) {
  const size_t data_size = width_ * height_ * sizeof(uint8_t) * 4;

  const viz::CudaService::ScopedPush cuda_context = viz::CudaService::get().PushContext();

  viz::UniqueCUdeviceptr device_ptr;
  device_ptr.reset([this](size_t data_size) {
    CUdeviceptr device_ptr;
    EXPECT_EQ(cuMemAlloc(&device_ptr, data_size), CUDA_SUCCESS);
    return device_ptr;
  }(data_size));

  ASSERT_TRUE(device_ptr);

  ASSERT_NO_THROW(viz::ReadFramebuffer(
      viz::ImageFormat::R8G8B8A8_UNORM, width_, height_, data_size, device_ptr.get()));

  read_data.resize(data_size);

  ASSERT_EQ(cuMemcpyDtoH(read_data.data(), device_ptr.get(), read_data.size()), CUDA_SUCCESS);
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

bool TestHeadless::CompareResult() {
  if (data_.size() != width_ * height_ * sizeof(uint8_t) * 4) {
    EXPECT_TRUE(false) << "Can compare R8G8B8A8_UNORM data only";
    return false;
  }

  std::vector<uint8_t> read_data;
  ReadData(read_data);

  for (size_t index = 0; index < data_.size(); ++index) {
    if (data_[index] != read_data[index]) {
      const std::string ref_file_name = BuildFileName("ref");
      const std::string fail_file_name = BuildFileName("fail");

      stbi_write_png(ref_file_name.c_str(), width_, height_, 4, data_.data(), 0);
      stbi_write_png(fail_file_name.c_str(), width_, height_, 4, read_data.data(), 0);

      EXPECT_TRUE(false) << "Data mismatch, wrote images to " << ref_file_name << " and "
                         << fail_file_name;
      return false;
    }
  }
  return true;
}

bool TestHeadless::CompareResultCRC32(const std::vector<uint32_t> crc32) {
  std::vector<uint8_t> read_data;
  ReadData(read_data);

  const uint32_t read_crc32 = stbiw__crc32(read_data.data(), read_data.size());

  if (std::find(crc32.begin(), crc32.end(), read_crc32) == crc32.end()) {
    const std::string fail_file_name = BuildFileName("fail");

    stbi_write_png(fail_file_name.c_str(), width_, height_, 4, read_data.data(), 0);

    std::ostringstream str;
    str << "CRC mismatch, expected {" << std::hex << std::setw(8);
    for (auto&& value : crc32) { str << "0x" << value << ", "; }
    str << "} but calculated 0x" << read_crc32 << ", wrote failing image to " << fail_file_name;
    EXPECT_FALSE(true) << str.str();
    return false;
  }

  return true;
}
