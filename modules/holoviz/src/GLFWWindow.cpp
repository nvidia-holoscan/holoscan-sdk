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

#include "GLFWWindow.h"

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <backends/imgui_impl_glfw.h>

#include <nvvk/error_vk.hpp>

#include <iostream>
#include <unistd.h>

namespace clara::holoviz
{

static void glfw_error_callback(int error, const char *description)
{
    std::cerr << "Glfw Error " << error << ": " << description << std::endl;
}

struct GLFWWindow::Impl
{
public:
    Impl()
    {
        glfwSetErrorCallback(glfw_error_callback);

        if (glfwInit() == GLFW_FALSE)
        {
            throw std::runtime_error("Failed to initialize glfw");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        if (!glfwVulkanSupported())
        {
            throw std::runtime_error("Vulkan is not supported");
        }
    }

    ~Impl()
    {
        if (intern_window_ && window_)
        {
            glfwDestroyWindow(window_);
        }

        glfwTerminate();
    }

    static void KeyCB(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void FrameBufferSizeCB(GLFWwindow *window, int width, int height);

    GLFWwindow *window_ = nullptr;
    bool intern_window_ = false;

    std::function<void(int width, int height)> frame_buffer_size_cb_;
};

GLFWWindow::GLFWWindow(GLFWwindow *window)
    : impl_(new Impl)
{
    impl_->window_ = window;
}

GLFWWindow::GLFWWindow(uint32_t width, uint32_t height, const char *title, InitFlags flags)
    : impl_(new Impl)
{
    impl_->window_ =
        glfwCreateWindow(width, height, title, (flags & InitFlags::FULLSCREEN) ? glfwGetPrimaryMonitor() : NULL, NULL);
    if (!impl_->window_)
    {
        throw std::runtime_error("Failed to create glfw window");
    }

    impl_->intern_window_ = true;
}

GLFWWindow::~GLFWWindow() {}

void GLFWWindow::InitImGui()
{
    ImGui_ImplGlfw_InitForVulkan(impl_->window_, true);
}

void GLFWWindow::Impl::KeyCB(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if ((action != GLFW_RELEASE) && (key == GLFW_KEY_ESCAPE))
    {
        glfwSetWindowShouldClose(window, 1);
    }
}

void GLFWWindow::Impl::FrameBufferSizeCB(GLFWwindow *window, int width, int height)
{
    static_cast<GLFWWindow::Impl *>(glfwGetWindowUserPointer(window))->frame_buffer_size_cb_(width, height);
}

void GLFWWindow::SetupCallbacks(std::function<void(int width, int height)> frame_buffer_size_cb_)
{
    impl_->frame_buffer_size_cb_ = frame_buffer_size_cb_;

    glfwSetWindowUserPointer(impl_->window_, impl_.get());
    glfwSetFramebufferSizeCallback(impl_->window_, &GLFWWindow::Impl::FrameBufferSizeCB);
    glfwSetKeyCallback(impl_->window_, &GLFWWindow::Impl::KeyCB);
}

const char **GLFWWindow::GetRequiredInstanceExtensions(uint32_t *count)
{
    return glfwGetRequiredInstanceExtensions(count);
}

void GLFWWindow::GetFramebufferSize(uint32_t *width, uint32_t *height)
{
    glfwGetFramebufferSize(impl_->window_, reinterpret_cast<int *>(width), reinterpret_cast<int *>(height));
}

VkSurfaceKHR GLFWWindow::CreateSurface(VkPhysicalDevice pysical_device, VkInstance instance)
{
    VkSurfaceKHR surface;
    NVVK_CHECK(glfwCreateWindowSurface(instance, impl_->window_, nullptr, &surface));
    return surface;
}

bool GLFWWindow::ShouldClose()
{
    return (glfwWindowShouldClose(impl_->window_) != 0);
}

bool GLFWWindow::IsMinimized()
{
    int w, h;
    glfwGetWindowSize(impl_->window_, &w, &h);
    bool minimized(w == 0 || h == 0);
    if (minimized)
    {
        usleep(50);
    }
    return minimized;
}

void GLFWWindow::ImGuiNewFrame()
{
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void GLFWWindow::Begin()
{
    glfwPollEvents();
}

void GLFWWindow::End() {}

} // namespace clara::holoviz