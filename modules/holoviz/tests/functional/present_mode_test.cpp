/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
