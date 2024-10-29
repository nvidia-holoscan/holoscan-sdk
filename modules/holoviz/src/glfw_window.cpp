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

#include "glfw_window.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <unistd.h>

#include <iostream>
#include <list>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include <nvh/cameramanipulator.hpp>
#include <nvh/timesampler.hpp>
#include <nvvk/error_vk.hpp>

namespace holoscan::viz {

static void glfw_error_callback(int error, const char* description) {
  std::cerr << "Glfw Error " << error << ": " << description << std::endl;
}

struct GLFWWindow::Impl {
 public:
  explicit Impl(InitFlags init_flags) : init_flags_(init_flags) {
    std::lock_guard<std::mutex> guard(mutex_);

    if (glfw_init_count_ == 0) {
      glfwSetErrorCallback(glfw_error_callback);

      if (glfwInit() == GLFW_FALSE) { throw std::runtime_error("Failed to initialize glfw"); }
    }
    ++glfw_init_count_;

    if (!glfwVulkanSupported()) { throw std::runtime_error("Vulkan is not supported"); }
  }
  Impl() = delete;

  ~Impl() {
    if (intern_window_ && window_) {
      if (init_flags_ & InitFlags::FULLSCREEN) {
        // GLFW is not switching back to the original mode when just destroying the window,
        // have to set the window monitor explicitly to NULL before destroy to switch back
        // to the original mode.
        glfwSetWindowMonitor(
            window_, NULL, 0, 0, framebuffer_width_, framebuffer_height_, GLFW_DONT_CARE);
      }
      glfwDestroyWindow(window_);
    }

    std::lock_guard<std::mutex> guard(mutex_);
    --glfw_init_count_;
    if (glfw_init_count_ == 0) { glfwTerminate(); }
  }

  static void key_cb(GLFWwindow* window, int key, int scancode, int action, int mods);
  static void unicode_char_cb(GLFWwindow* window, unsigned int code_point);
  static void cursor_pos_cb(GLFWwindow* window, double x, double y);
  static void mouse_button_cb(GLFWwindow* window, int button, int action, int mods);
  static void scroll_cb(GLFWwindow* window, double x, double y);
  static void framebuffer_size_cb(GLFWwindow* window, int width, int height);
  static void window_size_cb(GLFWwindow* window, int width, int height);

  void setup_callbacks() {
    prev_key_cb_ = glfwSetKeyCallback(window_, &GLFWWindow::Impl::key_cb);
    prev_unicode_char_cb_ = glfwSetCharCallback(window_, &GLFWWindow::Impl::unicode_char_cb);
    prev_cursor_pos_cb_ = glfwSetCursorPosCallback(window_, &GLFWWindow::Impl::cursor_pos_cb);
    prev_mouse_button_cb_ = glfwSetMouseButtonCallback(window_, &GLFWWindow::Impl::mouse_button_cb);
    prev_scroll_cb_ = glfwSetScrollCallback(window_, &GLFWWindow::Impl::scroll_cb);
    prev_framebuffer_size_cb_ =
        glfwSetFramebufferSizeCallback(window_, &GLFWWindow::Impl::framebuffer_size_cb);
    prev_window_size_cb_ = glfwSetWindowSizeCallback(window_, &GLFWWindow::Impl::window_size_cb);
  }

  KeyModifiers to_modifiers(int mods);

  const InitFlags init_flags_;

  static std::mutex mutex_;          ///< mutex to protect glfw init counter
  static uint32_t glfw_init_count_;  ///< glfw init counter

  GLFWwindow* window_ = nullptr;
  bool intern_window_ = false;

  std::list<KeyCallbackFunction> key_callbacks_;
  GLFWkeyfun prev_key_cb_ = nullptr;
  std::list<UnicodeCharCallbackFunction> unicode_char_callbacks_;
  GLFWcharfun prev_unicode_char_cb_ = nullptr;
  std::list<MouseButtonCallbackFunction> mouse_button_callbacks_;
  GLFWmousebuttonfun prev_mouse_button_cb_ = nullptr;
  std::list<ScrollCallbackFunction> scroll_callbacks_;
  GLFWscrollfun prev_scroll_cb_ = nullptr;
  std::list<CursorPosCallbackFunction> cursor_pos_callbacks_;
  GLFWcursorposfun prev_cursor_pos_cb_ = nullptr;
  std::list<FramebufferSizeCallbackFunction> framebuffer_size_callbacks_;
  GLFWframebuffersizefun prev_framebuffer_size_cb_ = nullptr;
  std::list<WindowSizeCallbackFunction> window_size_callbacks_;
  GLFWwindowsizefun prev_window_size_cb_ = nullptr;

  bool caps_lock_ = false;
  bool num_lock_ = false;
  nvh::CameraManipulator::Inputs inputs_;  ///< Mouse button pressed
  nvh::Stopwatch timer_;  ///< measure time from frame to frame to base camera movement on

  uint32_t framebuffer_width_ = 0;
  uint32_t framebuffer_height_ = 0;
  uint32_t window_width_ = 0;
  uint32_t window_height_ = 0;
};

// Initialize static members
std::mutex GLFWWindow::Impl::mutex_;
uint32_t GLFWWindow::Impl::glfw_init_count_ = 0;

GLFWWindow::GLFWWindow(GLFWwindow* window) : impl_(new Impl(InitFlags::NONE)) {
  impl_->window_ = window;

  // set the user pointer to the implementation class to be used in callbacks, fail if the provided
  // window already has the user pointer set.
  if (glfwGetWindowUserPointer(impl_->window_) != nullptr) {
    throw std::runtime_error("GLFW window user pointer already set");
  }
  glfwSetWindowUserPointer(impl_->window_, impl_.get());
  impl_->setup_callbacks();

  // set framebuffer and window size with initial window size
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  impl_->framebuffer_size_cb(impl_->window_, width, height);
  glfwGetWindowSize(window, &width, &height);
  impl_->window_size_cb(impl_->window_, width, height);
}

GLFWWindow::GLFWWindow(uint32_t width, uint32_t height, const char* title, InitFlags flags,
                       const char* display_name)
    : impl_(new Impl(flags)) {
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  GLFWmonitor* monitor = nullptr;
  if (flags & InitFlags::FULLSCREEN) {
    if (display_name) {
      int monitor_count = 0;
      GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);
      for (int index = 0; index < monitor_count; ++index) {
        const char* monitor_name = glfwGetMonitorName(monitors[index]);
        if (std::strcmp(monitor_name, display_name) == 0) {
          monitor = monitors[index];
          break;
        }
      }
      if (!monitor) {
        HOLOSCAN_LOG_WARN("Display \"{}\" not found, using the primary display instead",
                          display_name);
        HOLOSCAN_LOG_INFO("____________________");
        HOLOSCAN_LOG_INFO("Available displays :");
        for (int index = 0; index < monitor_count; ++index) {
          HOLOSCAN_LOG_INFO("{}", glfwGetMonitorName(monitors[index]));
        }
      }
    }
    if (!monitor) { monitor = glfwGetPrimaryMonitor(); }
  }

  impl_->window_ = glfwCreateWindow(width, height, title, monitor, NULL);
  if (!impl_->window_) { throw std::runtime_error("Failed to create glfw window"); }

  impl_->intern_window_ = true;

  // set the user pointer to the implementation class to be used in callbacks
  glfwSetWindowUserPointer(impl_->window_, impl_.get());
  impl_->setup_callbacks();

  // set framebuffer and window size with initial window size
  impl_->framebuffer_size_cb(impl_->window_, width, height);
  impl_->window_size_cb(impl_->window_, width, height);
}

GLFWWindow::~GLFWWindow() {}

void GLFWWindow::init_im_gui() {
  ImGui_ImplGlfw_InitForVulkan(impl_->window_, true);
}

static KeyAndButtonAction to_key_and_button_action(int action) {
  switch (action) {
    case GLFW_PRESS:
      return KeyAndButtonAction::PRESS;
    case GLFW_RELEASE:
      return KeyAndButtonAction::RELEASE;
    case GLFW_REPEAT:
      return KeyAndButtonAction::REPEAT;
    default:
      throw std::runtime_error(fmt::format("Unhandled GLFW key action {}", action));
  }
}

KeyModifiers GLFWWindow::Impl::to_modifiers(int mods) {
  KeyModifiers modifiers{};
  if (mods & GLFW_MOD_SHIFT) { modifiers.shift = 1; }
  if (mods & GLFW_MOD_CONTROL) { modifiers.control = 1; }
  if (mods & GLFW_MOD_ALT) { modifiers.alt = 1; }
  if (caps_lock_) { modifiers.caps_lock = 1; }
  if (num_lock_) { modifiers.num_lock = 1; }
  return modifiers;
}

void GLFWWindow::Impl::key_cb(GLFWwindow* window, int key, int scancode, int action, int mods) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_key_cb_) { impl->prev_key_cb_(window, key, scancode, action, mods); }

  if (!impl->key_callbacks_.empty()) {
    // The Holoviz key enum values are identical to the GLFW key values, so we can just cast
    const Key ext_key = (Key)key;
    const KeyAndButtonAction ext_action = to_key_and_button_action(action);
    const KeyModifiers ext_modifiers = impl->to_modifiers(mods);
    for (auto&& key_callback : impl->key_callbacks_) {
      key_callback(ext_key, ext_action, ext_modifiers);
    }
  }

  const bool pressed = action != GLFW_RELEASE;

  if (pressed && (key == GLFW_KEY_ESCAPE)) { glfwSetWindowShouldClose(window, 1); }

  // Keeping track of the modifiers
  if ((key == GLFW_KEY_LEFT_CONTROL) || (key == GLFW_KEY_RIGHT_CONTROL)) {
    impl->inputs_.ctrl = pressed;
  }
  if ((key == GLFW_KEY_LEFT_SHIFT) || (key == GLFW_KEY_RIGHT_SHIFT)) {
    impl->inputs_.shift = pressed;
  }
  if ((key == GLFW_KEY_LEFT_ALT) || (key == GLFW_KEY_RIGHT_ALT)) { impl->inputs_.alt = pressed; }
  if (key == GLFW_KEY_CAPS_LOCK) { impl->caps_lock_ = pressed; }
  if (key == GLFW_KEY_NUM_LOCK) { impl->num_lock_ = pressed; }
}

void GLFWWindow::Impl::unicode_char_cb(GLFWwindow* window, unsigned int code_point) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_unicode_char_cb_) { impl->prev_unicode_char_cb_(window, code_point); }

  for (auto&& unicode_char_callbacks : impl->unicode_char_callbacks_) {
    unicode_char_callbacks(code_point);
  }
}

void GLFWWindow::Impl::mouse_button_cb(GLFWwindow* window, int button, int action, int mods) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_mouse_button_cb_) { impl->prev_mouse_button_cb_(window, button, action, mods); }

  if (!impl->mouse_button_callbacks_.empty()) {
    MouseButton ext_mouse_button;
    switch (button) {
      case GLFW_MOUSE_BUTTON_LEFT:
        ext_mouse_button = MouseButton::LEFT;
        break;
      case GLFW_MOUSE_BUTTON_MIDDLE:
        ext_mouse_button = MouseButton::MIDDLE;
        break;
      case GLFW_MOUSE_BUTTON_RIGHT:
        ext_mouse_button = MouseButton::RIGHT;
        break;
      default:
        throw std::runtime_error(fmt::format("Unhandled GLFW mouse button {}", button));
    }
    const KeyAndButtonAction ext_action = to_key_and_button_action(action);
    const KeyModifiers ext_modifiers = impl->to_modifiers(mods);
    for (auto&& mouse_button_callback : impl->mouse_button_callbacks_) {
      mouse_button_callback(ext_mouse_button, ext_action, ext_modifiers);
    }
  }

  double x, y;
  glfwGetCursorPos(impl->window_, &x, &y);
  CameraManip.setMousePosition(static_cast<int>(x), static_cast<int>(y));

  impl->inputs_.lmb = (button == GLFW_MOUSE_BUTTON_LEFT) && (action == GLFW_PRESS);
  impl->inputs_.mmb = (button == GLFW_MOUSE_BUTTON_MIDDLE) && (action == GLFW_PRESS);
  impl->inputs_.rmb = (button == GLFW_MOUSE_BUTTON_RIGHT) && (action == GLFW_PRESS);
}

void GLFWWindow::Impl::scroll_cb(GLFWwindow* window, double x, double y) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_scroll_cb_) { impl->prev_scroll_cb_(window, x, y); }

  if (!impl->scroll_callbacks_.empty()) {
    for (auto&& scroll_callback : impl->scroll_callbacks_) { scroll_callback(x, y); }
  }

  // Allow camera movement only when not editing
  if ((ImGui::GetCurrentContext() != nullptr) && ImGui::GetIO().WantCaptureMouse) { return; }

  CameraManip.wheel(y > 0.0 ? 1 : -1, impl->inputs_);
}

void GLFWWindow::Impl::cursor_pos_cb(GLFWwindow* window, double x, double y) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_cursor_pos_cb_) { impl->prev_cursor_pos_cb_(window, x, y); }

  if (!impl->cursor_pos_callbacks_.empty()) {
    for (auto&& cursor_pos_callback : impl->cursor_pos_callbacks_) { cursor_pos_callback(x, y); }
  }

  // Allow camera movement only when not editing
  if ((ImGui::GetCurrentContext() != nullptr) && ImGui::GetIO().WantCaptureMouse) { return; }

  if (impl->inputs_.lmb || impl->inputs_.rmb || impl->inputs_.mmb) {
    CameraManip.mouseMove(static_cast<int>(x), static_cast<int>(y), impl->inputs_);
  }
}

void GLFWWindow::Impl::framebuffer_size_cb(GLFWwindow* window, int width, int height) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_framebuffer_size_cb_) { impl->prev_framebuffer_size_cb_(window, width, height); }

  for (auto&& framebuffer_size_callback : impl->framebuffer_size_callbacks_) {
    framebuffer_size_callback(width, height);
  }

  impl->framebuffer_width_ = width;
  impl->framebuffer_height_ = height;
  CameraManip.setWindowSize(width, height);
}

void GLFWWindow::Impl::window_size_cb(GLFWwindow* window, int width, int height) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_window_size_cb_) { impl->prev_window_size_cb_(window, width, height); }

  for (auto&& window_size_callback : impl->window_size_callbacks_) {
    window_size_callback(width, height);
  }

  impl->window_width_ = width;
  impl->window_height_ = height;
}

Window::CallbackHandle GLFWWindow::add_key_callback(KeyCallbackFunction callback) {
  impl_->key_callbacks_.push_back(std::move(callback));
  // create a handle which automatically removes the callback from the list when the handle is
  // destroyed
  return CallbackHandle(
      reinterpret_cast<void*>(
          impl_->key_callbacks_.back().target<std::remove_pointer_t<KeyCallbackType>>()),
      [this](void* callback) {
        impl_->key_callbacks_.remove_if([callback](KeyCallbackFunction f) {
          return reinterpret_cast<void*>(f.target<std::remove_pointer_t<KeyCallbackType>>()) ==
                 callback;
        });
      });
}

Window::CallbackHandle GLFWWindow::add_unicode_char_callback(UnicodeCharCallbackFunction callback) {
  impl_->unicode_char_callbacks_.push_back(std::move(callback));
  // create a handle which automatically removes the callback from the list when the handle is
  // destroyed
  return CallbackHandle(
      reinterpret_cast<void*>(impl_->unicode_char_callbacks_.back()
                                  .target<std::remove_pointer_t<UnicodeCharCallbackType>>()),
      [this](void* callback) {
        impl_->unicode_char_callbacks_.remove_if([callback](UnicodeCharCallbackFunction f) {
          return reinterpret_cast<void*>(
                     f.target<std::remove_pointer_t<UnicodeCharCallbackType>>()) == callback;
        });
      });
}

Window::CallbackHandle GLFWWindow::add_mouse_button_callback(MouseButtonCallbackFunction callback) {
  impl_->mouse_button_callbacks_.push_back(std::move(callback));
  // create a handle which automatically removes the callback from the list when the handle is
  // destroyed
  return CallbackHandle(
      reinterpret_cast<void*>(impl_->mouse_button_callbacks_.back()
                                  .target<std::remove_pointer_t<MouseButtonCallbackType>>()),
      [this](void* callback) {
        impl_->mouse_button_callbacks_.remove_if([callback](MouseButtonCallbackFunction f) {
          return reinterpret_cast<void*>(
                     f.target<std::remove_pointer_t<MouseButtonCallbackType>>()) == callback;
        });
      });
}

Window::CallbackHandle GLFWWindow::add_scroll_callback(ScrollCallbackFunction callback) {
  impl_->scroll_callbacks_.push_back(std::move(callback));
  // create a handle which automatically removes the callback from the list when the handle is
  // destroyed
  return CallbackHandle(
      reinterpret_cast<void*>(
          impl_->scroll_callbacks_.back().target<std::remove_pointer_t<ScrollCallbackType>>()),
      [this](void* callback) {
        impl_->scroll_callbacks_.remove_if([callback](ScrollCallbackFunction f) {
          return reinterpret_cast<void*>(f.target<std::remove_pointer_t<ScrollCallbackType>>()) ==
                 callback;
        });
      });
}

Window::CallbackHandle GLFWWindow::add_cursor_pos_callback(CursorPosCallbackFunction callback) {
  impl_->cursor_pos_callbacks_.push_back(std::move(callback));
  // create a handle which automatically removes the callback from the list when the handle is
  // destroyed
  return CallbackHandle(
      reinterpret_cast<void*>(impl_->cursor_pos_callbacks_.back()
                                  .target<std::remove_pointer_t<CursorPosCallbackType>>()),
      [this](void* callback) {
        impl_->cursor_pos_callbacks_.remove_if([callback](CursorPosCallbackFunction f) {
          return reinterpret_cast<void*>(
                     f.target<std::remove_pointer_t<CursorPosCallbackType>>()) == callback;
        });
      });
}

Window::CallbackHandle GLFWWindow::add_framebuffer_size_callback(
    FramebufferSizeCallbackFunction callback) {
  impl_->framebuffer_size_callbacks_.push_back(std::move(callback));
  // create a handle which automatically removes the callback from the list when the handle is
  // destroyed
  return CallbackHandle(
      reinterpret_cast<void*>(impl_->framebuffer_size_callbacks_.back()
                                  .target<std::remove_pointer_t<FramebufferSizeCallbackType>>()),
      [this](void* callback) {
        impl_->framebuffer_size_callbacks_.remove_if([callback](FramebufferSizeCallbackFunction f) {
          return reinterpret_cast<void*>(
                     f.target<std::remove_pointer_t<FramebufferSizeCallbackType>>()) == callback;
        });
      });
}

Window::CallbackHandle GLFWWindow::add_window_size_callback(WindowSizeCallbackFunction callback) {
  impl_->window_size_callbacks_.push_back(std::move(callback));
  // create a handle which automatically removes the callback from the list when the handle is
  // destroyed
  return CallbackHandle(
      reinterpret_cast<void*>(impl_->window_size_callbacks_.back()
                                  .target<std::remove_pointer_t<WindowSizeCallbackType>>()),
      [this](void* callback) {
        impl_->window_size_callbacks_.remove_if([callback](WindowSizeCallbackFunction f) {
          return reinterpret_cast<void*>(
                     f.target<std::remove_pointer_t<WindowSizeCallbackType>>()) == callback;
        });
      });
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
  *width = impl_->framebuffer_width_;
  *height = impl_->framebuffer_height_;
}

void GLFWWindow::get_window_size(uint32_t* width, uint32_t* height) {
  *width = impl_->window_width_;
  *height = impl_->window_height_;
}

vk::SurfaceKHR GLFWWindow::create_surface(vk::PhysicalDevice physical_device,
                                          vk::Instance instance) {
  VkSurfaceKHR surface;
  const vk::Result result =
      vk::Result(glfwCreateWindowSurface(instance, impl_->window_, nullptr, &surface));
  if (result != vk::Result::eSuccess) {
    vk::throwResultException(result, "Failed to create glfw window surface");
  }
  return surface;
}

bool GLFWWindow::should_close() {
  return (glfwWindowShouldClose(impl_->window_) != 0);
}

bool GLFWWindow::is_minimized() {
  const bool minimized = glfwGetWindowAttrib(impl_->window_, GLFW_ICONIFIED);
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

void GLFWWindow::end() {
  // call the base class
  Window::end();
}

float GLFWWindow::get_aspect_ratio() {
  if (impl_->framebuffer_height_) {
    return float(impl_->framebuffer_width_) / float(impl_->framebuffer_height_);
  } else {
    return 1.F;
  }
}

}  // namespace holoscan::viz
