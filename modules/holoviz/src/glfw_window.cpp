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

#include "glfw_window.hpp"

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <unistd.h>

#include <iostream>
#include <set>
#include <stdexcept>

#include <nvh/cameramanipulator.hpp>
#include <nvh/timesampler.hpp>
#include <nvvk/error_vk.hpp>

namespace holoscan::viz {

static void glfw_error_callback(int error, const char* description) {
  std::cerr << "Glfw Error " << error << ": " << description << std::endl;
}

struct GLFWWindow::Impl {
 public:
  Impl() {
    glfwSetErrorCallback(glfw_error_callback);

    if (glfwInit() == GLFW_FALSE) { throw std::runtime_error("Failed to initialize glfw"); }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    if (!glfwVulkanSupported()) { throw std::runtime_error("Vulkan is not supported"); }
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

  static void frame_buffer_size_cb(GLFWwindow* window, int width, int height);
  static void key_cb(GLFWwindow* window, int key, int scancode, int action, int mods);
  static void cursor_pos_cb(GLFWwindow* window, double x, double y);
  static void mouse_button_cb(GLFWwindow* window, int button, int action, int mods);
  static void scroll_cb(GLFWwindow* window, double x, double y);

  GLFWwindow* window_ = nullptr;
  bool intern_window_ = false;

  std::function<void(int width, int height)> frame_buffer_size_cb_;
  GLFWframebuffersizefun prev_frame_buffer_size_cb_ = nullptr;

  GLFWkeyfun prev_key_cb_ = nullptr;
  GLFWcursorposfun prev_cursor_pos_cb_ = nullptr;
  GLFWmousebuttonfun prev_mouse_button_cb_ = nullptr;
  GLFWscrollfun prev_scroll_cb_ = nullptr;

  nvh::CameraManipulator::Inputs inputs_;  ///< Mouse button pressed
  nvh::Stopwatch timer_;  ///< measure time from frame to frame to base camera movement on

  uint32_t width_;
  uint32_t height_;
};

GLFWWindow::GLFWWindow(GLFWwindow* window) : impl_(new Impl) {
  impl_->window_ = window;

  // set the user pointer to the implementation class to be used in callbacks, fail if the provided
  // window already has the user pointer set.
  if (glfwGetWindowUserPointer(impl_->window_) != nullptr) {
    throw std::runtime_error("GLFW window user pointer already set");
  }
  glfwSetWindowUserPointer(impl_->window_, impl_.get());

  // set framebuffer size with initial window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);
  impl_->frame_buffer_size_cb(window, width, height);

  // setup camera
  CameraManip.setLookat(
      nvmath::vec3f(0.f, 0.f, 1.f), nvmath::vec3f(0.f, 0.f, 0.f), nvmath::vec3f(0.f, 1.f, 0.f));
}

GLFWWindow::GLFWWindow(uint32_t width, uint32_t height, const char* title, InitFlags flags)
    : impl_(new Impl) {
  impl_->window_ = glfwCreateWindow(
      width, height, title, (flags & InitFlags::FULLSCREEN) ? glfwGetPrimaryMonitor() : NULL, NULL);
  if (!impl_->window_) { throw std::runtime_error("Failed to create glfw window"); }

  impl_->intern_window_ = true;

  // set the user pointer to the implementation class to be used in callbacks
  glfwSetWindowUserPointer(impl_->window_, impl_.get());

  // set framebuffer size with initial window size
  impl_->frame_buffer_size_cb(impl_->window_, width, height);

  // setup camera
  CameraManip.setLookat(
      nvmath::vec3f(0.f, 0.f, 1.f), nvmath::vec3f(0.f, 0.f, 0.f), nvmath::vec3f(0.f, 1.f, 0.f));
}

GLFWWindow::~GLFWWindow() {}

void GLFWWindow::init_im_gui() {
  ImGui_ImplGlfw_InitForVulkan(impl_->window_, true);
}

void GLFWWindow::Impl::frame_buffer_size_cb(GLFWwindow* window, int width, int height) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_frame_buffer_size_cb_) { impl->prev_frame_buffer_size_cb_(window, width, height); }

  if (impl->frame_buffer_size_cb_) { impl->frame_buffer_size_cb_(width, height); }

  impl->width_ = width;
  impl->height_ = height;
  CameraManip.setWindowSize(width, height);
}

void GLFWWindow::Impl::key_cb(GLFWwindow* window, int key, int scancode, int action, int mods) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_key_cb_) { impl->prev_key_cb_(window, key, scancode, action, mods); }

  const bool pressed = action != GLFW_RELEASE;

  if (pressed && (key == GLFW_KEY_ESCAPE)) { glfwSetWindowShouldClose(window, 1); }

  // Keeping track of the modifiers
  impl->inputs_.ctrl =
      pressed & ((key == GLFW_KEY_LEFT_CONTROL) || (key == GLFW_KEY_RIGHT_CONTROL));
  impl->inputs_.shift = pressed & ((key == GLFW_KEY_LEFT_SHIFT) || (key == GLFW_KEY_RIGHT_SHIFT));
  impl->inputs_.alt = pressed & ((key == GLFW_KEY_LEFT_ALT) || (key == GLFW_KEY_RIGHT_ALT));
}

void GLFWWindow::Impl::mouse_button_cb(GLFWwindow* window, int button, int action, int mods) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_mouse_button_cb_) { impl->prev_mouse_button_cb_(window, button, action, mods); }

  double x, y;
  glfwGetCursorPos(impl->window_, &x, &y);
  CameraManip.setMousePosition(static_cast<int>(x), static_cast<int>(y));

  impl->inputs_.lmb = (button == GLFW_MOUSE_BUTTON_LEFT) && (action == GLFW_PRESS);
  impl->inputs_.mmb = (button == GLFW_MOUSE_BUTTON_MIDDLE) && (action == GLFW_PRESS);
  impl->inputs_.rmb = (button == GLFW_MOUSE_BUTTON_RIGHT) && (action == GLFW_PRESS);
}

void GLFWWindow::Impl::cursor_pos_cb(GLFWwindow* window, double x, double y) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_cursor_pos_cb_) { impl->prev_cursor_pos_cb_(window, x, y); }

  // Allow camera movement only when not editing
  if ((ImGui::GetCurrentContext() != nullptr) && ImGui::GetIO().WantCaptureMouse) { return; }

  if (impl->inputs_.lmb || impl->inputs_.rmb || impl->inputs_.mmb) {
    CameraManip.mouseMove(static_cast<int>(x), static_cast<int>(y), impl->inputs_);
  }
}

void GLFWWindow::Impl::scroll_cb(GLFWwindow* window, double x, double y) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_scroll_cb_) { impl->prev_scroll_cb_(window, x, y); }

  // Allow camera movement only when not editing
  if ((ImGui::GetCurrentContext() != nullptr) && ImGui::GetIO().WantCaptureMouse) { return; }

  CameraManip.wheel(y > 0.0 ? 1 : -1, impl->inputs_);
}

void GLFWWindow::setup_callbacks(std::function<void(int width, int height)> frame_buffer_size_cb_) {
  impl_->frame_buffer_size_cb_ = frame_buffer_size_cb_;

  impl_->prev_frame_buffer_size_cb_ =
      glfwSetFramebufferSizeCallback(impl_->window_, &GLFWWindow::Impl::frame_buffer_size_cb);
  impl_->prev_mouse_button_cb_ =
      glfwSetMouseButtonCallback(impl_->window_, &GLFWWindow::Impl::mouse_button_cb);
  impl_->prev_scroll_cb_ = glfwSetScrollCallback(impl_->window_, &GLFWWindow::Impl::scroll_cb);
  impl_->prev_cursor_pos_cb_ =
      glfwSetCursorPosCallback(impl_->window_, &GLFWWindow::Impl::cursor_pos_cb);
  impl_->prev_key_cb_ = glfwSetKeyCallback(impl_->window_, &GLFWWindow::Impl::key_cb);
}

const char** GLFWWindow::get_required_instance_extensions(uint32_t* count) {
  return glfwGetRequiredInstanceExtensions(count);
}

const char** GLFWWindow::get_required_device_extensions(uint32_t* count) {
  static char const* extensions[]{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  *count = sizeof(extensions) / sizeof(extensions[0]);
  return extensions;
}

uint32_t GLFWWindow::select_device(vk::Instance instance,
                                   const std::vector<vk::PhysicalDevice>& physical_devices) {
  // select the first device which has presentation support
  for (uint32_t index = 0; index < physical_devices.size(); ++index) {
    if (glfwGetPhysicalDevicePresentationSupport(instance, physical_devices[index], 0) ==
        GLFW_TRUE) {
      return index;
    }
  }
  throw std::runtime_error("No device with presentation support found");
}

void GLFWWindow::get_framebuffer_size(uint32_t* width, uint32_t* height) {
  glfwGetFramebufferSize(
      impl_->window_, reinterpret_cast<int*>(width), reinterpret_cast<int*>(height));
}

vk::SurfaceKHR GLFWWindow::create_surface(vk::PhysicalDevice physical_device,
                                          vk::Instance instance) {
  VkSurfaceKHR surface;
  const vk::Result result =
      vk::Result(glfwCreateWindowSurface(instance, impl_->window_, nullptr, &surface));
  vk::resultCheck(result, "Failed to create glfw window surface");
  return surface;
}

bool GLFWWindow::should_close() {
  return (glfwWindowShouldClose(impl_->window_) != 0);
}

bool GLFWWindow::is_minimized() {
  int w, h;
  glfwGetWindowSize(impl_->window_, &w, &h);
  bool minimized(w == 0 || h == 0);
  if (minimized) { usleep(50); }
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

void GLFWWindow::get_view_matrix(nvmath::mat4f* view_matrix) {
  *view_matrix = nvmath::perspectiveVK(CameraManip.getFov(), 1.f /*aspectRatio*/, 0.1f, 1000.0f) *
                 CameraManip.getMatrix();
}

float GLFWWindow::get_aspect_ratio() {
  return float(impl_->width_) / float(impl_->height_);
}

}  // namespace holoscan::viz
