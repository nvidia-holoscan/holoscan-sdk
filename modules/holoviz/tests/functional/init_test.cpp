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
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <stdlib.h>

#include <future>
#include <vector>

#include <holoviz/holoviz.hpp>

namespace viz = holoscan::viz;

TEST(Init, GLFWWindow) {
  if (glfwInit() == GLFW_FALSE) {
    const char* description;
    int code = glfwGetError(&description);
    ASSERT_EQ(code, GLFW_PLATFORM_UNAVAILABLE)
        << "Expected `GLFW_PLATFORM_UNAVAILABLE` but got `" << code << "`: `" << description << "`";
    GTEST_SKIP() << "No display server available, skipping test." << description;
  }

  if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) {
    GTEST_SKIP() << "With Wayland and statically linked GLFW creating the GLFW window outside of "
                    "Holoviz is not supported.";
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* const window = glfwCreateWindow(128, 64, "Holoviz test", NULL, NULL);

  ASSERT_NO_THROW(viz::Init(window));
  EXPECT_FALSE(viz::WindowShouldClose());
  EXPECT_FALSE(viz::WindowIsMinimized());
  EXPECT_NO_THROW(viz::Shutdown());
}

TEST(Init, CreateWindow) {
  if (glfwInit() == GLFW_FALSE) {
    const char* description;
    int code = glfwGetError(&description);
    ASSERT_EQ(code, GLFW_PLATFORM_UNAVAILABLE)
        << "Expected `GLFW_PLATFORM_UNAVAILABLE` but got `" << code << "`: `" << description << "`";
    GTEST_SKIP() << "No display server available, skipping test." << description;
  }

  EXPECT_NO_THROW(viz::Init(128, 64, "Holoviz test"));
  EXPECT_FALSE(viz::WindowShouldClose());
  EXPECT_FALSE(viz::WindowIsMinimized());
  EXPECT_NO_THROW(viz::Shutdown());
}

TEST(Init, Fullscreen) {
  if (glfwInit() == GLFW_FALSE) {
    const char* description;
    int code = glfwGetError(&description);
    ASSERT_EQ(code, GLFW_PLATFORM_UNAVAILABLE)
        << "Expected `GLFW_PLATFORM_UNAVAILABLE` but got `" << code << "`: `" << description << "`";
    GTEST_SKIP() << "No display server available, skipping test." << description;
  }

  // There is an issue when setting a mode with lower resolution than the current mode, in
  // this case the nvidia driver switches to panning mode. This could be avoided when calling
  // XRRSetScreenSize() additionally to setting the mode:
  //   https://www.winehq.org/pipermail/wine-patches/2016-July/152357.html
  // For now just use a really big resolution to avoid the issue.

  EXPECT_NO_THROW(viz::Init(4096, 4096, "Holoviz test", viz::InitFlags::FULLSCREEN));
  EXPECT_FALSE(viz::WindowShouldClose());
  EXPECT_FALSE(viz::WindowIsMinimized());
  EXPECT_NO_THROW(viz::Shutdown());
}

TEST(Init, Headless) {
  EXPECT_NO_THROW(viz::Init(128, 64, "Holoviz test", viz::InitFlags::HEADLESS));
  EXPECT_FALSE(viz::WindowShouldClose());
  EXPECT_FALSE(viz::WindowIsMinimized());
  EXPECT_NO_THROW(viz::Shutdown());
}

/**
 * Check that viz::Init() is returning an error (and not crashing) when the Vulkan loader can't
 * find a ICD.
 */
TEST(Init, VulkanLoaderFail) {
  // VK_LOADER_DRIVERS_SELECT selects manifest files which matches the provided glob
  const char* vk_drivers_files_str = "VK_DRIVER_FILES";
  // VK_ICD_FILENAMES force the loader to use the specific driver JSON files. This is deprecated
  // and replaced by VK_LOADER_DRIVERS_SELECT above, but we need to set both variables to support
  // both the old loader and the new one.
  const char* vk_icd_filenames_str = "VK_ICD_FILENAMES";

  // get the old value so we can restore it later
  char* old_vk_drivers_files_value = getenv(vk_drivers_files_str);
  char* old_vk_icd_filenames_value = getenv(vk_icd_filenames_str);

  // set to 'none' so now manifest files ar found
  EXPECT_EQ(setenv(vk_drivers_files_str, "none", 1), 0);
  EXPECT_EQ(setenv(vk_icd_filenames_str, "none", 1), 0);

  // should throw because no driver is found
  EXPECT_THROW(viz::Init(128, 64, "Holoviz test", viz::InitFlags::HEADLESS), std::runtime_error);

  // reset env to old value or remove
  if (old_vk_drivers_files_value) {
    EXPECT_EQ(setenv(vk_drivers_files_str, old_vk_drivers_files_value, 1), 0);
  } else {
    EXPECT_EQ(unsetenv(vk_drivers_files_str), 0);
  }
  if (old_vk_icd_filenames_value) {
    EXPECT_EQ(setenv(vk_icd_filenames_str, old_vk_icd_filenames_value, 1), 0);
  } else {
    EXPECT_EQ(unsetenv(vk_icd_filenames_str), 0);
  }

  EXPECT_NO_THROW(viz::Shutdown());
}

TEST(Init, Errors) {
  // should thrown when specifying an invalid font file
  EXPECT_NO_THROW(viz::SetFont("NonExistingFile.ttf", 12.F));
  EXPECT_THROW(viz::Init(128, 64, "Holoviz test", viz::InitFlags::HEADLESS), std::runtime_error);
  EXPECT_NO_THROW(viz::Shutdown());

  // should throw when there is no window
  EXPECT_THROW(viz::Begin(), std::runtime_error);
  EXPECT_THROW(viz::End(), std::runtime_error);
  EXPECT_THROW(
      viz::ReadFramebuffer(viz::ImageFormat::R8G8B8A8_UNORM, 1, 1, 1 * 1 * 4 * sizeof(uint8_t), 0),
      std::runtime_error);
  EXPECT_NO_THROW(viz::Shutdown());
}

TEST(Init, MultiThreaded) {
  if (glfwInit() == GLFW_FALSE) {
    const char* description;
    int code = glfwGetError(&description);
    ASSERT_EQ(code, GLFW_PLATFORM_UNAVAILABLE)
        << "Expected `GLFW_PLATFORM_UNAVAILABLE` but got `" << code << "`: `" << description << "`";
    GTEST_SKIP() << "No display server available, skipping test." << description;
  }

  // create multiple windows from threads
  std::vector<std::future<void>> futures;
  for (int i = 0; i < 8; ++i) {
    futures.push_back(std::async(std::launch::async, [] { viz::Init(640, 480, "Holoviz test"); }));
  }
  for (auto&& future : futures) { future.wait(); }
}
