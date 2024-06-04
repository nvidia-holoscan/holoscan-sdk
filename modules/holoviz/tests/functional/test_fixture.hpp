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

#ifndef MODULES_HOLOVIZ_TESTS_FUNCTIONAL_TEST_FIXTURE_HPP
#define MODULES_HOLOVIZ_TESTS_FUNCTIONAL_TEST_FIXTURE_HPP

#include <gtest/gtest.h>

#include <vector>

#include <holoviz/holoviz.hpp>

/**
 * Fixture that initializes Holoviz and has support functions to setup, read back
 * and compare data.
 */
class TestBase : public ::testing::Test {
 protected:
  /**
   * Construct a new TestBase object
   *
   * @param init_flags init flags
   */
  explicit TestBase(holoscan::viz::InitFlags init_flags) : init_flags_(init_flags) {}
  TestBase() = delete;

  /**
   * Construct a new TestBase object with a given window size
   *
   * @param width window width
   * @param height window height
   * @param init_flags init flags
   */
  TestBase(uint32_t width, uint32_t height, holoscan::viz::InitFlags init_flags)
      : width_(width), height_(height), init_flags_(init_flags) {}

  /// ::testing::Test virtual members
  ///@{
  void SetUp() override;
  void TearDown() override;
  ///@}

  /**
   * Set the CUDA device ordinal to be used for CUDA operations.
   *
   * @param device_ordinal CUDA device ordinal
   */
  void SetCUDADevice(uint32_t device_ordinal);

  /**
   * Setup random pixel data.
   *
   * @param format pixel format
   * @param rand_seed random seed
   */
  void SetupData(holoscan::viz::ImageFormat format, uint32_t rand_seed = 1);

  /**
   * Read back color data from the Holoviz window.
   *
   * @param color_data vector to hold the color data
   */
  void ReadColorData(std::vector<uint8_t>& color_data);

  /**
   * Read back depth data from the Holoviz window.
   *
   * @param depth_data vector to hold the depth data
   */
  void ReadDepthData(std::vector<float>& depth_data);

  /**
   * Read back color data and compare with the data generated with SetupData().
   *
   * @return false if read back and generated data do not match
   */
  bool CompareColorResult();

  /**
   * Read back depth data and compare with the data generated with SetupData().
   *
   * @return false if read back and generated data do not match
   */
  bool CompareDepthResult();

  /**
   * Read back color data, generate a CRC32 and compare with the provided CRC32's.
   *
   * @param crc32 vector of expected CRC32's
   *
   * @return false if CRC32 of read back does not match provided CRC32
   */
  bool CompareColorResultCRC32(const std::vector<uint32_t> crc32);

  /**
   * Read back depth data, generate a CRC32 and compare with the provided CRC32's.
   *
   * @param crc32 vector of expected CRC32's
   *
   * @return false if CRC32 of read back does not match provided CRC32
   */
  bool CompareDepthResultCRC32(const std::vector<uint32_t> crc32);

  uint32_t device_ordinal_ = 0;

  const uint32_t lut_size_ = 8;

  const uint32_t width_ = 64;
  const uint32_t height_ = 32;

  std::vector<uint8_t> color_data_;
  std::vector<float> depth_data_;

 protected:
  holoscan::viz::InitFlags init_flags_ = holoscan::viz::InitFlags::NONE;

 private:
  bool initialized_ = false;
};

/**
 * Fixture that initializes Holoviz in headless mode.
 */
class TestHeadless : public TestBase {
 public:
  TestHeadless() : TestBase(holoscan::viz::InitFlags::HEADLESS) {}

  /**
   * Construct a new TestHeadless object with a given window size
   *
   * @param width window width
   * @param height window height
   */
  TestHeadless(uint32_t width, uint32_t height)
      : TestBase(width, height, holoscan::viz::InitFlags::HEADLESS) {}
};

/**
 * Fixture that initializes Holoviz in headless mode.
 */
class TestWindow : public TestBase {
 public:
  TestWindow() : TestBase(holoscan::viz::InitFlags::NONE) {}
};

#endif /* MODULES_HOLOVIZ_TESTS_FUNCTIONAL_TEST_FIXTURE_HPP */
