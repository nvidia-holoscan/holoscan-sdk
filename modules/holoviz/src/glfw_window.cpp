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

#include "glfw_window.hpp"

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <unistd.h>
#include <backends/imgui_impl_glfw.h>

#include <iostream>
#include <stdexcept>

#include <nvvk/error_vk.hpp>

namespace holoscan::viz {

static void glfw_error_callback(int error, const char *description) {
    std::cerr << "Glfw Error " << error << ": " << description << std::endl;
}

struct GLFWWindow::Impl {
 public:
    Impl() {
        glfwSetErrorCallback(glfw_error_callback);

        if (glfwInit() == GLFW_FALSE) {
            throw std::runtime_error("Failed to initialize glfw");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        if (!glfwVulkanSupported()) {
            throw std::runtime_error("Vulkan is not supported");
        }
    }

    ~Impl() {
        if (intern_window_ && window_) {
            // GLFW is not switching back to the original code when just destroying the window,
            // have to set the window monitor explicitly to NULL before destroy to switch back
            // to the original mode.
            glfwSetWindowMonitor(window_, NULL, 0, 0, 640, 480, GLFW_DONT_CARE);
            glfwDestroyWindow(window_);
        }

        glfwTerminate();
    }

    static void key_cb(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void frame_buffer_size_cb(GLFWwindow *window, int width, int height);

    GLFWwindow *window_ = nullptr;
    bool intern_window_ = false;

    std::function<void(int width, int height)> frame_buffer_size_cb_;
};

GLFWWindow::GLFWWindow(GLFWwindow *window)
    : impl_(new Impl) {
    impl_->window_ = window;
}

GLFWWindow::GLFWWindow(uint32_t width, uint32_t height, const char *title, InitFlags flags)
    : impl_(new Impl) {
    impl_->window_ =
        glfwCreateWindow(width, height, title, (flags & InitFlags::FULLSCREEN) ?
                                          glfwGetPrimaryMonitor() : NULL, NULL);
    if (!impl_->window_) {
        throw std::runtime_error("Failed to create glfw window");
    }

    impl_->intern_window_ = true;
}

GLFWWindow::~GLFWWindow() {}

void GLFWWindow::init_im_gui() {
    ImGui_ImplGlfw_InitForVulkan(impl_->window_, true);
}

void GLFWWindow::Impl::key_cb(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if ((action != GLFW_RELEASE) && (key == GLFW_KEY_ESCAPE)) {
        glfwSetWindowShouldClose(window, 1);
    }
}

void GLFWWindow::Impl::frame_buffer_size_cb(GLFWwindow *window, int width, int height) {
    static_cast<GLFWWindow::Impl *>(glfwGetWindowUserPointer(window))->
                                                    frame_buffer_size_cb_(width, height);
}

void GLFWWindow::setup_callbacks(std::function<void(int width, int height)> frame_buffer_size_cb_) {
    impl_->frame_buffer_size_cb_ = frame_buffer_size_cb_;

    glfwSetWindowUserPointer(impl_->window_, impl_.get());
    glfwSetFramebufferSizeCallback(impl_->window_, &GLFWWindow::Impl::frame_buffer_size_cb);
    glfwSetKeyCallback(impl_->window_, &GLFWWindow::Impl::key_cb);
}

const char **GLFWWindow::get_required_instance_extensions(uint32_t *count) {
    return glfwGetRequiredInstanceExtensions(count);
}

const char **GLFWWindow::get_required_device_extensions(uint32_t *count) {
    static char const *extensions[]{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    *count = sizeof(extensions) / sizeof(extensions[0]);
    return extensions;
}

void GLFWWindow::get_framebuffer_size(uint32_t *width, uint32_t *height) {
    glfwGetFramebufferSize(impl_->window_, reinterpret_cast<int *>(width),
                                         reinterpret_cast<int *>(height));
}

VkSurfaceKHR GLFWWindow::create_surface(VkPhysicalDevice physical_device, VkInstance instance) {
    VkSurfaceKHR surface;
    NVVK_CHECK(glfwCreateWindowSurface(instance, impl_->window_, nullptr, &surface));
    return surface;
}

bool GLFWWindow::should_close() {
    return (glfwWindowShouldClose(impl_->window_) != 0);
}

bool GLFWWindow::is_minimized() {
    int w, h;
    glfwGetWindowSize(impl_->window_, &w, &h);
    bool minimized(w == 0 || h == 0);
    if (minimized) {
        usleep(50);
    }
    return minimized;
}

void GLFWWindow::im_gui_new_frame() {
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void GLFWWindow::begin() {
    glfwPollEvents();
}

void GLFWWindow::end() {}

}  // namespace holoscan::viz
