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
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <stdlib.h>

#include <vector>

#include <holoviz/holoviz.hpp>

namespace viz = holoscan::viz;

class VSync : public testing::TestWithParam<viz::PresentMode> {};

TEST_P(VSync, Modes) {
  if (glfwInit() == GLFW_FALSE) {
    const char* description;
    int code = glfwGetError(&description);
    ASSERT_EQ(code, GLFW_PLATFORM_UNAVAILABLE)
        << "Expected `GLFW_PLATFORM_UNAVAILABLE` but got `" << code << "`: `" << description << "`";
    GTEST_SKIP() << "No display server available, skipping test." << description;
  }

  GLFWmonitor* const monitor = glfwGetPrimaryMonitor();
  if (monitor == nullptr) { GTEST_SKIP() << "No monitor connected, skipping test."; }
  const GLFWvidmode* const video_mode = glfwGetVideoMode(monitor);
  ASSERT_TRUE(video_mode != nullptr);
  // when running with Xvfb the refresh rate is zero, skip the test
  if (video_mode->refreshRate == 0) { GTEST_SKIP() << "Refresh rate is zero, skipping test."; }

  viz::PresentMode present_mode = GetParam();

  EXPECT_NO_THROW(viz::Init(128, 64, "Holoviz test"));

  // check if the present mode is supported
  uint32_t present_mode_count = 0;
  EXPECT_NO_THROW(viz::GetPresentModes(&present_mode_count, nullptr));
  EXPECT_GT(present_mode_count, 0);
  std::vector<viz::PresentMode> present_modes(present_mode_count);
  EXPECT_NO_THROW(viz::GetPresentModes(&present_mode_count, present_modes.data()));
  bool supported = false;
  for (auto&& supported_present_mode : present_modes) {
    if (supported_present_mode == present_mode) {
      supported = true;
      break;
    }
  }
  if (!supported) {
    EXPECT_NO_THROW(viz::Shutdown());
    GTEST_SKIP() << "Present mode " << int(present_mode) << " is not supported, skipping test.";
  }

  EXPECT_NO_THROW(viz::SetPresentMode(present_mode));

  // warm up
  for (int frame = 0; frame < video_mode->refreshRate; ++frame) {
    EXPECT_NO_THROW(viz::Begin());
    EXPECT_NO_THROW(viz::End());
  }

  // measure for a while
  const auto runtime = std::chrono::milliseconds(1000);
  const auto start = std::chrono::steady_clock::now();
  int frames = 0;
  do {
    EXPECT_NO_THROW(viz::Begin());
    EXPECT_NO_THROW(viz::End());
    ++frames;
  } while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                 start) < runtime);
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);

  const float displayed_frames = float(video_mode->refreshRate) * elapsed.count() / 1000.F;

  switch (present_mode) {
    case viz::PresentMode::FIFO:
      // rendered frames should be within 10% of displayed frames
      EXPECT_LE(std::abs((float(frames) / displayed_frames) - 1.F), 0.1F);
      break;
    case viz::PresentMode::AUTO:
    case viz::PresentMode::IMMEDIATE:
    case viz::PresentMode::MAILBOX:
      // no vsync, should render at least two times the refresh rate
      EXPECT_GT(frames, displayed_frames * 2.F);
      break;
  }

  EXPECT_NO_THROW(viz::Shutdown());
}

INSTANTIATE_TEST_SUITE_P(VSync, VSync,
                         testing::Values(viz::PresentMode::AUTO, viz::PresentMode::FIFO,
                                         viz::PresentMode::IMMEDIATE, viz::PresentMode::MAILBOX));
