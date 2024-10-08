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

#ifndef MODULES_HOLOVIZ_SRC_GLFW_WINDOW_HPP
#define MODULES_HOLOVIZ_SRC_GLFW_WINDOW_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "holoviz/init_flags.hpp"
#include "window.hpp"

typedef struct GLFWwindow GLFWwindow;

namespace holoscan::viz {

/**
 * Specialization of the Window class handling a GLFWwindow.
 */
class GLFWWindow : public Window {
 public:
  /**
   * Construct a new GLFWWindow object with an existing GLFWwindow object.
   *
   * @param window    existing GLFWwindow window
   */
  explicit GLFWWindow(GLFWwindow* window);

  /**
   * Construct a new GLFWWindow object of a given size.
   *
   * @param width, height     window size
   * @param title             window tile
   * @param flags             init flags
   * @param display_name  name of the display, this can either be the EDID name as displayed
   *                      in the NVIDIA Settings, or the output name provided by `xrandr` or
   *                      `hwinfo --monitor`.
   *                      if nullptr then the primary display is selected.
   */
  GLFWWindow(uint32_t width, uint32_t height, const char* title, InitFlags flags,
             const char* display_name);

  /**
   * Delete the standard constructor, always need parameters to construct.
   */
  GLFWWindow() = delete;

  /**
   * Destroy the GLFWWindow object.
   */
  virtual ~GLFWWindow();

  /// holoscan::viz::Window virtual members
  ///@{
  void init_im_gui() override;
  CallbackHandle add_key_callback(KeyCallbackFunction callback) override;
  CallbackHandle add_unicode_char_callback(UnicodeCharCallbackFunction callback) override;
  CallbackHandle add_mouse_button_callback(MouseButtonCallbackFunction callback) override;
  CallbackHandle add_scroll_callback(ScrollCallbackFunction callback) override;
  CallbackHandle add_cursor_pos_callback(CursorPosCallbackFunction callback) override;
  CallbackHandle add_framebuffer_size_callback(FramebufferSizeCallbackFunction callback) override;
  CallbackHandle add_window_size_callback(WindowSizeCallbackFunction callback) override;

  const char** get_required_instance_extensions(uint32_t* count) override;
  const char** get_required_device_extensions(uint32_t* count) override;
  uint32_t select_device(vk::Instance instance,
                         const std::vector<vk::PhysicalDevice>& physical_devices) override;
  void get_framebuffer_size(uint32_t* width, uint32_t* height) override;
  void get_window_size(uint32_t* width, uint32_t* height) override;

  vk::SurfaceKHR create_surface(vk::PhysicalDevice physical_device, vk::Instance instance) override;

  bool should_close() override;
  bool is_minimized() override;

  void im_gui_new_frame() override;

  void begin() override;
  void end() override;

  float get_aspect_ratio() override;
  ///@}

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_GLFW_WINDOW_HPP */
