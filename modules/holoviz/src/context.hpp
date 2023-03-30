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

#ifndef HOLOSCAN_VIZ_CONTEXT_HPP
#define HOLOSCAN_VIZ_CONTEXT_HPP

#include <cuda.h>

#include <list>
#include <memory>

#include "holoviz/image_format.hpp"
#include "holoviz/init_flags.hpp"
#include "util/non_copyable.hpp"

struct ImGuiContext;
typedef struct GLFWwindow GLFWwindow;

namespace holoscan::viz {

class Window;
class ImageLayer;
class GeometryLayer;
class Layer;

/**
 * The Context object is the central object for Holoviz to store all the state. It's a singleton.
 */
class Context : public NonCopyable {
 public:
  /**
   * @returns the context instance
   */
  static Context& get();

  /**
   * Initialize the context with an existing GLFWwindow object
   *
   * @param window    existing GLFWwindow window
   * @param flags     init flags
   */
  void init(GLFWwindow* window, InitFlags flags);

  /**
   * Initialize the context and create a window with the given properties.
   *
   * @param width, height     window size
   * @param title             window title
   * @param flags             init flags
   */
  void init(uint32_t width, uint32_t height, const char* title, InitFlags flags);

  /**
   * Initialize the context and create a exclusive window with the given properties.
   *
   * @param display_name  name of the display, this can either be the EDID name as displayed
   *                      in the NVIDIA Settings, or the output name used by xrandr,
   *                      if nullptr then the first display is selected.
   * @param width         desired width, ignored if 0
   * @param height        desired height, ignored if 0
   * @param refresh_rate  desired refresh rate (number of times the display is refreshed
   *                      each second multiplied by 1000), ignored if 0
   * @param flags         init flags
   */
  void init(const char* display_name, uint32_t width, uint32_t height, uint32_t refresh_rate,
            InitFlags flags);

  /**
   * Shutdown and cleanup the global context.
   */
  void shutdown();

  /**
   * Get the window object.
   *
   * @return window object
   */
  Window* get_window() const;

  /**
   * If using ImGui, create a context and pass it to Holoviz, do this before calling viz::Init().
   *
   * Background: The ImGui context is a global variable and global variables are not shared
   * across so/DLL boundaries. Therefore the app needs to create the ImGui context first and
   *  then provides the pointer to Holoviz like this:
   * @code{.cpp}
   * ImGui::CreateContext();
   * holoscan::viz::ImGuiSetCurrentContext(ImGui::GetCurrentContext());
   * @endcode
   *
   * @param context   ImGui context
   */
  void im_gui_set_current_context(ImGuiContext* context);

  /**
   * Set the Cuda stream used by Holoviz for Cuda operations.
   *
   * The default stream is 0, i.e. non-concurrent mode. All Cuda commands issued by Holoviz
   * then provides the pointer  use that stream.
   * The stream can be changed any time.
   *
   * @param stream    Cuda stream to use
   */
  void set_cuda_stream(CUstream stream);

  /**
   * @returns the currently active cuda stream
   */
  CUstream get_cuda_stream() const;

  /**
   * Set the font used to render text, do this before calling viz::Init().
   *
   * The font is converted to bitmaps, these bitmaps are scaled to the final size when rendering.
   *
   * @param path path to TTF font file
   * @param size_in_pixels size of the font bitmaps
   */
  void set_font(const char* path, float size_in_pixels);

  /**
   * Start recording layer definitions.
   */
  void begin();

  /**
   * End recording and output the composited layers.
   */
  void end();

  /**
   * Begin an image layer definition.
   */
  void begin_image_layer();

  /**
   * Begin a geometry layer definition.
   */
  void begin_geometry_layer();

  /**
   * Begin an ImGui layer definition.
   */
  void begin_im_gui_layer();

  /**
   * End the current layer.
   */
  void end_layer();

  /**
   * Read the framebuffer and store it to cuda device memory.
   *
   * Can only be called outside of Begin()/End().
   *
   * @param fmt           image format, currently only R8G8B8A8_UNORM is supported.
   * @param width, height width and height of the region to read back, will be limited to the
   *                      framebuffer size if the framebuffer is smaller than that
   * @param buffer_size   size of the storage buffer in bytes
   * @param device_ptr    pointer to Cuda device memory to store the framebuffer into
   */
  void read_framebuffer(ImageFormat fmt, uint32_t width, uint32_t height, size_t buffer_size,
                        CUdeviceptr device_ptr);

  /**
   * @returns the active layer
   */
  Layer* get_active_layer() const;

  /**
   * @returns the active image layer
   */
  ImageLayer* get_active_image_layer() const;

  /**
   * @returns the active geometry layer
   */
  GeometryLayer* get_active_geometry_layer() const;

 private:
  Context();

  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_CONTEXT_HPP */
