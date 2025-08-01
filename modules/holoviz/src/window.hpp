/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_VIZ_WINDOW_HPP
#define HOLOSCAN_VIZ_WINDOW_HPP

#include <nvmath/nvmath_types.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "holoviz/callbacks.hpp"

namespace holoscan::viz {

/**
 * Base class for all window backends.
 */
class Window {
 public:
  /**
   * Construct a new Window object.
   */
  Window();

  /**
   * Destroy the Window object.
   */
  virtual ~Window() {}

  /**
   * Initialize ImGui to be used with that window (mouse, keyboard, ...).
   */
  virtual void init_im_gui() = 0;

  /**
   * Callback handle, returned but the add_???_callback() functions, removes callback when
   * destroyed
   */
  typedef std::shared_ptr<void> CallbackHandle;

  /// key callback type
  typedef void (*KeyCallbackType)(Key key, KeyAndButtonAction action, KeyModifiers modifiers);
  /// key callback function type
  typedef std::function<std::remove_pointer_t<KeyCallbackType>> KeyCallbackFunction;

  /// unicode char callback type
  typedef void (*UnicodeCharCallbackType)(uint32_t code_point);
  /// unicode char callback function type
  typedef std::function<std::remove_pointer_t<UnicodeCharCallbackType>> UnicodeCharCallbackFunction;

  /// mouse button callback type
  typedef void (*MouseButtonCallbackType)(MouseButton button, KeyAndButtonAction action,
                                          KeyModifiers modifiers);
  /// mouse button callback function type
  typedef std::function<std::remove_pointer_t<MouseButtonCallbackType>> MouseButtonCallbackFunction;

  /// scroll callback type
  typedef void (*ScrollCallbackType)(double x_offset, double y_offset);
  /// scroll callback function type
  typedef std::function<std::remove_pointer_t<ScrollCallbackType>> ScrollCallbackFunction;

  /// cursor position callback type
  typedef void (*CursorPosCallbackType)(double x_pos, double y_pos);
  /// cursor position callback function type
  typedef std::function<std::remove_pointer_t<CursorPosCallbackType>> CursorPosCallbackFunction;

  /// framebuffer size callback type
  typedef void (*FramebufferSizeCallbackType)(int width, int height);
  /// framebuffer size callback function type
  typedef std::function<std::remove_pointer_t<FramebufferSizeCallbackType>>
      FramebufferSizeCallbackFunction;

  /// window size callback type
  typedef void (*WindowSizeCallbackType)(int width, int height);
  /// window size callback function type
  typedef std::function<std::remove_pointer_t<WindowSizeCallbackType>> WindowSizeCallbackFunction;

  /**
   * Add a key callback. The callback function is called when a key is pressed, released or
   * repeated.
   *
   * @param callback the key callback to add
   *
   * @return callback handle, callback is automatically removed when handle is deleted
   */
  virtual CallbackHandle add_key_callback(KeyCallbackFunction callback) { return CallbackHandle(); }

  /**
   * Add a Unicode character callback. The callback function is called when a Unicode character is
   * input.
   *
   * @param callback the new Unicode character callback to add
   *
   * @return callback handle, callback is automatically removed when handle is deleted
   */
  virtual CallbackHandle add_unicode_char_callback(UnicodeCharCallbackFunction callback) {
    return CallbackHandle();
  }

  /**
   * Add a mouse button callback. The callback function is called when a mouse button is pressed
   * or released.
   *
   * @param callback the new mouse button callback to add
   *
   * @return callback handle, callback is automatically removed when handle is deleted
   */
  virtual CallbackHandle add_mouse_button_callback(MouseButtonCallbackFunction callback) {
    return CallbackHandle();
  }

  /**
   * Add a scroll callback. The callback function is called when a scrolling device is used,
   * such as a mouse scroll wheel or the scroll area of a touch pad.
   *
   * @param callback the new cursor callback to add
   *
   * @return callback handle, callback is automatically removed when handle is deleted
   */
  virtual CallbackHandle add_scroll_callback(ScrollCallbackFunction callback) {
    return CallbackHandle();
  }

  /**
   * Add a cursor position callback. The callback function is called when the cursor position
   * changes. Coordinates are provided in screen coordinates, relative to the upper left edge of the
   * content area.
   *
   * @param callback the new cursor position callback to add
   *
   * @return callback handle, callback is automatically removed when handle is deleted
   */
  virtual CallbackHandle add_cursor_pos_callback(CursorPosCallbackFunction callback) {
    return CallbackHandle();
  }

  /**
   * Add a framebuffer size callback. The callback function is called when the framebuffer is
   * resized.
   *
   * @param callback the new framebuffer size callback to add
   *
   * @return callback handle, callback is automatically removed when handle is deleted
   */
  virtual CallbackHandle add_framebuffer_size_callback(FramebufferSizeCallbackFunction callback) {
    return CallbackHandle();
  }

  /**
   * Add a window size callback. The callback function is called when the window is resized.
   *
   * @param callback the new window size callback to add
   *
   * @return callback handle, callback is automatically removed when handle is deleted
   */
  virtual CallbackHandle add_window_size_callback(WindowSizeCallbackFunction callback) {
    return CallbackHandle();
  }

  /**
   * Get the required instance extensions for vulkan.
   *
   * @param [out] count   returns the extension count
   * @return array with required extensions
   */
  virtual const char** get_required_instance_extensions(uint32_t* count) = 0;

  /**
   * Get the required device extensions for vulkan.
   *
   * @param [out] count   returns the extension count
   * @return array with required extensions
   */
  virtual const char** get_required_device_extensions(uint32_t* count) = 0;

  /**
   * Select a device from the given list of supported physical devices.
   *
   * @param instance instance the devices belong to
   * @param physical_devices list of supported physical devices
   * @return index of the selected physical device
   */
  virtual uint32_t select_device(vk::Instance instance,
                                 const std::vector<vk::PhysicalDevice>& physical_devices) = 0;

  /**
   * Get the current framebuffer size.
   *
   * @param [out] width, height   framebuffer size in pixels
   */
  virtual void get_framebuffer_size(uint32_t* width, uint32_t* height) = 0;

  /**
   * Get the current window size.
   *
   * @param [out] width, height   window size in screen coordinates
   */
  virtual void get_window_size(uint32_t* width, uint32_t* height) = 0;

  /**
   * Create a Vulkan surface.
   *
   * @param physical_device    Vulkan device
   * @param instance          Vulkan instance
   * @return vulkan surface
   */
  virtual vk::SurfaceKHR create_surface(vk::PhysicalDevice physical_device,
                                        vk::Instance instance) = 0;

  /**
   * @return true if the window should be closed
   */
  virtual bool should_close() = 0;

  /**
   * @return true if the window is minimized
   */
  virtual bool is_minimized() = 0;

  /**
   * Start a new ImGui frame.
   */
  virtual void im_gui_new_frame() = 0;

  /**
   * Start a new frame.
   */
  virtual void begin() = 0;

  /**
   * End the current frame.
   */
  virtual void end() = 0;

  /**
   * Set the camera eye, look at and up vectors.
   *
   * @param eye_x, eye_y, eye_z               eye position
   * @param look_at_x, look_at_y, look_at_z   look at position
   * @param up_x, up_y, up_z                  up vector
   * @param anim                              animate transition
   */
  void set_camera(const nvmath::vec3f& eye, const nvmath::vec3f& look_at, const nvmath::vec3f& up,
                  bool anim);

  /**
   * Get the view matrix
   *
   * @param view_matrix
   */
  void get_view_matrix(nvmath::mat4f* view_matrix);

  /**
   * Get the view matrix
   *
   * @param view_matrix
   */
  void get_camera_matrix(nvmath::mat4f* camera_matrix);

  /**
   * @return the horizontal aspect ratio
   */
  virtual float get_aspect_ratio() = 0;
};

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_WINDOW_HPP */
