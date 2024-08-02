/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdlib.h>

#include <vector>

#include <holoviz/holoviz.hpp>
#include "test_fixture.hpp"

namespace viz = holoscan::viz;

class PresentMode : public TestWindow {};

TEST_F(PresentMode, Get) {
  uint32_t present_mode_count = 0;
  EXPECT_NO_THROW(viz::GetPresentModes(&present_mode_count, nullptr));
  EXPECT_GT(present_mode_count, 0);
  std::vector<viz::PresentMode> present_modes(present_mode_count);
  EXPECT_NO_THROW(viz::GetPresentModes(&present_mode_count, present_modes.data()));
  EXPECT_GE(present_mode_count, present_modes.size());

  // FIFO is always supported
  bool found_fifo = false;
  for (auto&& present_mode : present_modes) {
    if (present_mode == viz::PresentMode::FIFO) {
      found_fifo = true;
      break;
    }
  }
  EXPECT_TRUE(found_fifo);
}

TEST_F(PresentMode, Errors) {
  // it's an error to call GetPresentModes with an invalid present_mode_count
  EXPECT_THROW(viz::GetPresentModes(nullptr, nullptr), std::invalid_argument);

  // it's an error to call GetPresentModes with a present_mode_count != 0 and an invalid
  // present_modes
  uint32_t present_mode_count = 1;
  EXPECT_THROW(viz::GetPresentModes(&present_mode_count, nullptr), std::invalid_argument);
}

TEST(PresentModeNoInit, Errors) {
  // it's an error to call GetPresentModes without calling `viz::Init()` before
  uint32_t present_mode_count = 0;
  EXPECT_THROW(viz::GetPresentModes(&present_mode_count, nullptr), std::runtime_error);
}

class PresentModeBeforeInit : public TestWindow {
  void SetUp() override {
    EXPECT_NO_THROW(viz::SetPresentMode(viz::PresentMode::FIFO));
    // call base class
    TestWindow::SetUp();
  }
};

TEST_F(PresentModeBeforeInit, Pass) {}
