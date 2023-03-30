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

#ifndef HOLOVIZ_TESTS_FUNCTIONAL_HEADLESS_FIXTURE_HPP
#define HOLOVIZ_TESTS_FUNCTIONAL_HEADLESS_FIXTURE_HPP

#include <gtest/gtest.h>

#include <vector>

#include <holoviz/holoviz.hpp>

/**
 * Fixture that initializes Holoviz in headless mode and support functions to setup, read back
 * and compare data.
 */
class TestHeadless : public ::testing::Test {
 protected:
  /**
   * Construct a new TestHeadless object
   */
  TestHeadless() = default;

  /**
   * Construct a new TestHeadless object with a given window size
   *
   * @param width window width
   * @param height window height
   */
  TestHeadless(uint32_t width, uint32_t height) : width_(width), height_(height) {}

  /// ::testing::Test virtual members
  ///@{
  void SetUp() override;
  void TearDown() override;
  ///@}

  /**
   * Setup random pixel data.
   *
   * @param format pixel format
   * @param rand_seed random seed
   */
  void SetupData(holoscan::viz::ImageFormat format, uint32_t rand_seed = 1);

  /**
   * Read back data from the Holoviz window.
   *
   * @param read_data vector to hold the read back data
   */
  void ReadData(std::vector<uint8_t>& read_data);

  /**
   * Read back data and compare with the data generated with SetupData().
   *
   * @returns false if read back and generated data do not match
   */
  bool CompareResult();

  /**
   * Read back data, generate a CRC32 and compare with the provided CRC32's.
   *
   * @param crc32 vector of expected CRC32's
   *
   * @returns false if read back and generated data do not match
   */
  bool CompareResultCRC32(const std::vector<uint32_t> crc32);

  const uint32_t lut_size_ = 8;

  const uint32_t width_ = 64;
  const uint32_t height_ = 32;

  std::vector<uint8_t> data_;
};

#endif /* HOLOVIZ_TESTS_FUNCTIONAL_HEADLESS_FIXTURE_HPP */
