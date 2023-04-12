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

#ifndef HOLOSCAN_VIZ_WINDOW_HPP
#define HOLOSCAN_VIZ_WINDOW_HPP

#include <nvmath/nvmath_types.h>

#include <cstdint>
#include <functional>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace holoscan::viz {

/**
 * Base class for all window backends.
 */
class Window {
 public:
  /**
   * Construct a new Window object.
   */
  Window() {}

  /**
   * Destroy the Window object.
   */
  virtual ~Window() {}

  /**
   * Initialize ImGui to be used with that window (mouse, keyboard, ...).
   */
  virtual void init_im_gui() = 0;

  /**
   * Setup call backs.
   *
   * @param framebuffersize_cb    Called when the frame buffer size changes
   */
  virtual void setup_callbacks(std::function<void(int width, int height)> frame_buffer_size_cb) = 0;

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
   * Get the current frame buffer size.
   *
   * @param [out] width, height   framebuffer size
   */
  virtual void get_framebuffer_size(uint32_t* width, uint32_t* height) = 0;

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
   * @returns true if the window should be closed
   */
  virtual bool should_close() = 0;

  /**
   * @returns true if the window is minimized
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
   * Get the view matrix
   *
   * @param view_matrix
   */
  virtual void get_view_matrix(nvmath::mat4f* view_matrix) { *view_matrix = nvmath::mat4f(1); }

  /**
   * @returns the horizontal aspect ratio
   */
  virtual float get_aspect_ratio() = 0;
};

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_WINDOW_HPP */
