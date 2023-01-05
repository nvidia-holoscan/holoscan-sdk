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

#ifndef HOLOSCAN_VIZ_HOLOVIZ_HOLOVIZ_HPP
#define HOLOSCAN_VIZ_HOLOVIZ_HOLOVIZ_HPP

/**
 * \file
 *
 * \mainpage Holoviz
 *
 * Holoviz composites real time streams of frames with multiple different other layers like
 * segmentation mask layers, geometry layers and GUI layers.
 *
 * For maximum performance Holoviz makes use of [Vulkan](https://www.vulkan.org/), which is
 * already installed as part of the Nvidia driver.
 *
 * \section Concepts
 *
 * Holoviz uses the concept of the immediate mode design pattern for its API, inspired by the
 * [Dear ImGui](https://github.com/ocornut/imgui) library. The difference to the retained mode, for
 * which most APIs are designed for, is, that there are no objects created and stored by the
 * application.
 * This makes it easy to quickly build and change an Holoviz app.
 *
 * \section Usage
 *
 * The code below creates a window and displays an image.
 * First Holoviz needs to be initialized. This is done by calling clara::viz::Init().
 *
 * The elements to display are defined in the render loop, termination of the loop is checked with
 * clara::viz::WindowShouldClose().
 *
 * The definition of the displayed content starts with clara::viz::Begin() and ends with
 * clara::viz::End(). clara::viz::End() starts the rendering and displays the rendered result.
 *
 * Finally Holoviz is shutdown with viz::Shutdown().
 *
 * ~~~{.c}
namespace viz = holoscan::viz;

viz::Init("Holoviz Example");

while (!viz::WindowShouldClose()) {
    viz::Begin();
    viz::BeginImageLayer();
    viz::ImageHost(width, height, viz::ImageFormat::R8G8B8A8_UNORM, image_data);
    viz::EndLayer();
    viz::End();
}

viz::Shutdown();
 * ~~~
 *
 */

#include <cuda.h>

#include <cstdint>

#include "holoviz/image_format.hpp"
#include "holoviz/init_flags.hpp"
#include "holoviz/primitive_topology.hpp"

// forward declaration of external types
typedef struct GLFWwindow GLFWwindow;
struct ImGuiContext;

namespace holoscan::viz
{

/**
 * Initialize Holoviz using an existing GLFW window.
 *
 * @param window    GLFW window
 * @param flags     init flags
 */
void Init(GLFWwindow *window, InitFlags flags = InitFlags::NONE);

/**
 * Initialize Holoviz.
 *
 * This creates a window using the given width and height and sets the title.
 *
 * @param width     desired width
 * @param height    desired height
 * @param title     window title
 * @param flags     init flags
 */
void Init(uint32_t width, uint32_t height, const char *title, InitFlags flags = InitFlags::NONE);

/**
 * Initialize Holoviz to use a display in exclusive mode.
 *
 * Setup:
 *  - when multiple displays are connected:
 *    The display to be used in exclusive mode needs to be disabled in the NVIDIA Settings. Open the `X Server Display Configuration`
 *    tab, select the display and under `Configuration` select `Disabled`. Press `Apply`.
 *  - when a single display is connected:
 *    SSH into the machine, stop the X server with `sudo systemctl stop display-manager`.
 *
 * @param displayName   name of the display, this can either be the EDID name as displayed in the
 *                      NVIDIA Settings, or the output name used by xrandr,
 *                      if nullptr then the first display is selected.
 * @param width         desired width, ignored if 0
 * @param height        desired height, ignored if 0
 * @param refreshRate   desired refresh rate (number of times the display is refreshed
 *                      each second multiplied by 1000), ignored if 0
 * @param flags         init flags
 */
void Init(const char *displayName, uint32_t width = 0, uint32_t height = 0, uint32_t refreshRate = 0,
          InitFlags flags = InitFlags::NONE);

/**
 * If using ImGui, create a context and pass it to Holoviz, do this before calling viz::Init().
 *
 * Background: The ImGui context is a global variable and global variables are not shared across
 * so/DLLboundaries. Therefore the app needs to create the ImGui context first and then provides
 * the pointer to Holoviz like this:
 * @code{.cpp}
 * ImGui::CreateContext();
 * holoscan::viz::ImGuiSetCurrentContext(ImGui::GetCurrentContext());
 * @endcode
 *
 * @param context  ImGui context
 */
void ImGuiSetCurrentContext(ImGuiContext *context);

/**
 * Set the Cuda stream used by Holoviz for Cuda operations.
 *
 * The default stream is 0, i.e. non-concurrent mode. All Cuda commands issued by Holoviz
 * use that stream.
 * The stream can be changed any time.
 *
 * @param stream    Cuda stream to use
 */
void SetCudaStream(CUstream stream);

/**
 * Check if the window close button had been pressed
 *
 * @returns true if the window close button had been pressed
 */
bool WindowShouldClose();

/**
 * Check if the window is minimized. This can be used to skip rendering on minimized windows.
 *
 * @returns true if the window is minimized
 */
bool WindowIsMinimized();

/**
 * Shutdown Holoviz. All resources are destroyed.
 */
void Shutdown();

/**
 * Start recording layer definitions.
 */
void Begin();

/**
 * End recording and output the composited layers.
 */
void End();

/**
 * Begin an image layer definition.
 *
 * Layer properties (priority and opacity are set to the defaults).
 */
void BeginImageLayer();

/**
 * Defines the image data for this layer, source is Cuda device memory.
 *
 * If the image has a alpha value it's multiplied with the layer opacity.
 *
 * @param width         width of the image
 * @param height        height of the image
 * @param fmt           image format
 * @param device_ptr    Cuda device memory pointer
 */
void ImageCudaDevice(uint32_t width, uint32_t height, ImageFormat fmt, CUdeviceptr device_ptr);

/**
 * Defines the image data for this layer, source is a Cuda array.
 *
 * If the image has a alpha value it's multiplied with the layer opacity.
 *
 * @param fmt       image format
 * @param array     Cuda array
 */
void ImageCudaArray(ImageFormat fmt, CUarray array);

/**
 * Defines the image data for this layer, source is host memory.
 *
 * If the image has a alpha value it's multiplied with the layer opacity.
 *
 * @param width     width of the image
 * @param height    height of the image
 * @param fmt       image format
 * @param data      host memory pointer
 */
void ImageHost(uint32_t width, uint32_t height, ImageFormat fmt, const void *data);

/**
 * Defines the lookup table for this image layer.
 *
 * If a lookup table is used the image format has to be a single channel integer or
 * float format (e.g. ::ImageFormat::R8_UINT, ::ImageFormat::R16_UINT, ::ImageFormat::R32_UINT,
 * ::ImageFormat::R32_SFLOAT).
 *
 * The function processed is as follow
 * @code{.cpp}
 *  out = lut[clamp(in, 0, size)]
 * @endcode
 * Input image values are clamped to the range of the lookup table size: `[0, size[`.
 *
 * @param size      size of the lookup table in elements
 * @param fmt       lookup table color format
 * @param data_size size of the lookup table data in bytes
 * @param data      host memory pointer to lookup table data
 */
void LUT(uint32_t size, ImageFormat fmt, size_t data_size, const void *data);

/**
 * Start a ImGUI layer.
 *
 * Layer properties (priority and opacity are set to the defaults).
 */
void BeginImGuiLayer();

/**
 * Start a geometry layer.
 *
 * Layer properties (priority and opacity are set to the defaults).
 * Coordinates start with (0, 0) in the top left and end with (1, 1) in the bottom right.
 */
void BeginGeometryLayer();

/**
 * Set the color for following geometry.
 *
 * @param r,g,b,a RGBA color. Default (1.0, 1.0, 1.0, 1.0).
 */
void Color(float r, float g, float b, float a);

/**
 * Set the line width for geometry made of lines.
 *
 * @param width line width in pixels. Default 1.0.
 */
void LineWidth(float width);

/**
 * Set the point size for geometry made of points.
 *
 * @param size  point size in pixels. Default 1.0.
 */
void PointSize(float size);

/**
 * Draw a geometric primitive.
 *
 * @param topology          primitive topology
 * @param primitive_count   primitive count
 * @param data_size         size of the data array in floats
 * @param data              pointer to data, the format and size of the array depends on the
 *                          primitive count and topology
 */
void Primitive(PrimitiveTopology topology, uint32_t primitive_count, size_t data_size, const float *data);
/**
 * Draw text.
 *
 * @param x     x coordinate
 * @param y     y coordinate
 * @param size  font size
 * @param text  text to draw
 */
void Text(float x, float y, float size, const char *text);

/**
 * Set the layer opacity.
 *
 * @param opacity   layer opacity. 1.0 is fully opaque, 0.0 is fully transparent. Default 1.0.
 */
void LayerOpacity(float opacity);

/**
 * Set the layer priority.
 *
 * Before rendering the layers they are sorted by priority, the layers with the lowest priority
 * are rendered first so that the layer with the highest priority is rendered on top of
 * all other layers.
 * If layers have the same priority then the render order of these layers is undefined.
 *
 * @param priority  layer priority. Default 0.
 */
void LayerPriority(int32_t priority);

/**
 * End the current layer.
 */
void EndLayer();

/**
 * Read the framebuffer and store it to cuda device memory.
 *
 * Can only be called outside of Begin()/End().
 *
 * @param fmt          image format, currently only R8G8B8A8_UNORM is supported.
 * @param buffer_size  size of the storage buffer in bytes
 * @param device_ptr   pointer to Cuda device memory to store the framebuffer into
 */
void ReadFramebuffer(ImageFormat fmt, size_t buffer_size, CUdeviceptr device_ptr);

} // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_HOLOVIZ_HOLOVIZ_HPP */