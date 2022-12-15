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

#include <gtest/gtest.h>
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <holoviz/holoviz.hpp>

namespace viz = holoscan::viz;

TEST(Init, GLFWWindow) {
    Display *display = XOpenDisplay(NULL);
    if (!display) {
        GTEST_SKIP() << "X11 server is not running or DISPLAY variable is not set, skipping test.";
    }

    EXPECT_EQ(glfwInit(), GLFW_TRUE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *const window = glfwCreateWindow(128, 64, "Holoviz test", NULL, NULL);

    EXPECT_NO_THROW(viz::Init(window));
    EXPECT_FALSE(viz::WindowShouldClose());
    EXPECT_FALSE(viz::WindowIsMinimized());
    EXPECT_NO_THROW(viz::Shutdown());
}

TEST(Init, CreateWindow) {
    Display *display = XOpenDisplay(NULL);
    if (!display) {
        GTEST_SKIP() << "X11 server is not running or DISPLAY variable is not set, skipping test.";
    }

    EXPECT_NO_THROW(viz::Init(128, 64, "Holoviz test"));
    EXPECT_FALSE(viz::WindowShouldClose());
    EXPECT_FALSE(viz::WindowIsMinimized());
    EXPECT_NO_THROW(viz::Shutdown());
}

TEST(Init, Fullscreen) {
    Display *display = XOpenDisplay(NULL);
    if (!display) {
        GTEST_SKIP() << "X11 server is not running or DISPLAY variable is not set, skipping test.";
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
