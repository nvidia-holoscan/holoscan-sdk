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

#ifndef MODULES_HOLOVIZ_SRC_HOLOVIZ_HOLOVIZ_HPP
#define MODULES_HOLOVIZ_SRC_HOLOVIZ_HOLOVIZ_HPP

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
 * \section Instance
 *
 * The Holoviz module uses a thread-local instance object to store its internal state. The instance
 * object is created when calling the Holoviz module is first called from a thread. All Holoviz
 * module functions called from that thread use this instance.
 *
 * When calling into the Holoviz module from other threads other than the thread from which the
 * Holoviz module functions were first called, make sure to call {func}`viz::GetCurrent()` and
 * {func}`viz::SetCurrent()` in the respective threads.
 *
 * There are usage cases where multiple instances are needed, for example, to open multiple windows.
 * Instances can be created by calling {func}`viz::Create()`. Call {func}`viz::SetCurrent()` to make
 * the instance current before calling the Holoviz module function to be executed for the window the
 * instance belongs to.
 *
 * \section Callbacks
 *
 * Callbacks are executed for certain events like key input ({func}`viz::SetKeyCallback()`,
 * {func}`viz::SetUnicodeCharCallback()`), mouse events ({func}`viz::SetMouseButtonCallback()`,
 * {func}`viz::SetCursorPosCallback()`), framebuffer events
({func}`viz::SetFramebufferSizeCallback()`
 * and window events ({func}`viz::SetWindowSizeCallback()`).
 *
 * \section Usage
 *
 * The code below creates a window and displays an image.
 * First Holoviz needs to be initialized. This is done by calling holoscan::viz::Init().
 *
 * The elements to display are defined in the render loop, termination of the loop is checked with
 * holoscan::viz::WindowShouldClose().
 *
 * The definition of the displayed content starts with holoscan::viz::Begin() and ends with
 * holoscan::viz::End(). holoscan::viz::End() starts the rendering and displays the rendered result.
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
 */

#include <cuda.h>

#include <cstdint>

#include "holoviz/callbacks.hpp"
#include "holoviz/depth_map_render_mode.hpp"
#include "holoviz/display_event_type.hpp"
#include "holoviz/image_format.hpp"
#include "holoviz/init_flags.hpp"
#include "holoviz/present_mode.hpp"
#include "holoviz/primitive_topology.hpp"
#include "holoviz/surface_format.hpp"

// forward declaration of external types
typedef struct GLFWwindow GLFWwindow;
struct ImGuiContext;

namespace holoscan::viz {

// forward declaration of internal types
typedef void* InstanceHandle;

/**
 * Create a new instance.
 *
 * Note: this does not make the instance current for the active thread.
 *
 * @return created instance
 */
InstanceHandle Create();

/**
 * Set the current instance for the active thread.
 *
 * @param instance instance to set
 */
void SetCurrent(InstanceHandle instance);

/**
 * @return the current instance for the active thread.
 */
InstanceHandle GetCurrent();

/**
 * Initialize Holoviz using an existing GLFW window.
 *
 * Note that this functionality is not supported when using the Wayland display server protocol and
 * statically linking GLFW.
 * Reason: GLFW maintains a global variable with state when using Wayland. When statically linking
 * the GLFW library binaries (which is the default for the Holoviz library) there are different
 * global for the app binary, which creates the GLFW windon and the Holobiz binary, which is trying
 * to use it. This results in a segfault.
 *
 * @param window    GLFW window
 * @param flags     init flags
 */
void Init(GLFWwindow* window, InitFlags flags = InitFlags::NONE);

/**
 * Initialize Holoviz.
 *
 * This creates a window using the given width and height and sets the title.
 *
 * @param width     desired width
 * @param height    desired height
 * @param title     window title
 * @param flags     init flags
 * @param display_name  if the `FULLSCREEN` init flag is specified, this is the name of the display
 *                      to use for full screen mode. Use the output name provided by `xrandr` or
 *                      `hwinfo --monitor`. If nullptr then the primary display is selected.
 */
void Init(uint32_t width, uint32_t height, const char* title, InitFlags flags = InitFlags::NONE,
          const char* display_name = nullptr);

/**
 * Initialize Holoviz to use a display in exclusive mode.
 *
 * Setup:
 *  - when multiple displays are connected:
 *    The display to be used in exclusive mode needs to be disabled in the NVIDIA Settings. Open the
 *    `X Server Display Configuration` tab, select the display and under `Configuration` select
 *    `Disabled`. Press `Apply`.
 *  - when a single display is connected:
 *    SSH into the machine, stop the X server with `sudo systemctl stop display-manager`.
 *
 * @param displayName   name of the display, this can either be the EDID name as displayed in the
 *                      NVIDIA Settings, or the output name provided by `xrandr` or
 *                      `hwinfo --monitor`.
 *                      if nullptr then the first display is selected.
 * @param width         desired width, ignored if 0
 * @param height        desired height, ignored if 0
 * @param refreshRate   desired refresh rate (number of times the display is refreshed
 *                      each second multiplied by 1000), ignored if 0
 * @param flags         init flags
 */
void Init(const char* displayName, uint32_t width = 0, uint32_t height = 0,
          uint32_t refreshRate = 0, InitFlags flags = InitFlags::NONE);

/**
 * Set the key callback. The callback function is called when a key is pressed, released or
 * repeated.
 *
 * @param user_pointer user pointer value to be passed to the callback
 * @param callback the new key callback or nullptr to remove the current callback
 */
void SetKeyCallback(void* user_pointer, KeyCallbackFunction callback);

/**
 * Set the Unicode character callback. The callback function is called when a Unicode character is
 * input.
 *
 * @param user_pointer user pointer value to be passed to the callback
 * @param callback the new Unicode character callback or nullptr to remove the current callback
 */
void SetUnicodeCharCallback(void* user_pointer, UnicodeCharCallbackFunction callback);

/**
 * Set the mouse button callback. The callback function is called when a mouse button is pressed
 * or released.
 *
 * @param user_pointer user pointer value to be passed to the callback
 * @param callback the new mouse button callback or nullptr to remove the current callback
 */
void SetMouseButtonCallback(void* user_pointer, MouseButtonCallbackFunction callback);

/**
 * Set the scroll callback. The callback function is called when a scrolling device is used,
 * such as a mouse scroll wheel or the scroll area of a touch pad.
 *
 * @param user_pointer user pointer value to be passed to the callback
 * @param callback the new cursor callback or nullptr to remove the current callback
 */
void SetScrollCallback(void* user_pointer, ScrollCallbackFunction callback);

/**
 * Set the cursor position callback. The callback function is called when the cursor position
 * changes. Coordinates are provided in screen coordinates, relative to the upper left edge of the
 * content area.
 *
 * @param user_pointer user pointer value to be passed to the callback
 * @param callback the new cursor callback or nullptr to remove the current callback
 */
void SetCursorPosCallback(void* user_pointer, CursorPosCallbackFunction callback);

/**
 * Set the framebuffer size callback. The callback function is called when the framebuffer is
 * resized.
 *
 * @param user_pointer user pointer value to be passed to the callback
 * @param callback the new framebuffer size callback or nullptr to remove the current callback
 */
void SetFramebufferSizeCallback(void* user_pointer, FramebufferSizeCallbackFunction callback);

/**
 * Set the window size callback. The callback function is called when the window is resized.
 *
 * @param user_pointer user pointer value to be passed to the callback
 * @param callback the new window size callback or nullptr to remove the current callback
 */
void SetWindowSizeCallback(void* user_pointer, WindowSizeCallbackFunction callback);

/**
 * Get the supported present modes.
 *
 * `viz::Init()` has to be called before since the present modes depend on the window.
 *
 * If `present_modes` is nullptr, then the number of present modes supported for the current window
 * is returned in `present_mode_count`. Otherwise, `present_mode_count` must point to a variable set
 * by the application to the number of elements in the `present_modes` array, and on return the
 * variable is overwritten with the number of values actually written to `present_modes`. If the
 * value of `present_mode_count` is less than the number of presentation modes supported, at most
 * `present_mode_count` values will be written,
 *
 * @param present_mode_count number of presentation modes available or queried
 * @param present_modes either nullptr or a pointer to an array of PresentMode values
 */
void GetPresentModes(uint32_t* present_mode_count, PresentMode* present_modes);

/**
 * Set the present mode.
 *
 * The present mode determines how the rendered result will be presented on the screen.
 *
 * Default is 'PresentMode::AUTO'.
 *
 * @param present_mode present mode
 */
void SetPresentMode(PresentMode present_mode);

/**
 * Get the supported surface formats.
 *
 * `viz::Init()` has to be called before since the surface formats depend on the window.
 *
 * If `surface_formats` is nullptr, then the number of surface formats supported for the current
 * window is returned in `surface_format_count`. Otherwise, `surface_format_count` must point to a
 * variable set by the application to the number of elements in the `surface_formats` array, and on
 * return the variable is overwritten with the number of values actually written to
 * `surface_formats`. If the value of `surface_format_count` is less than the number of surface
 * formats supported, at most `surface_format_count` values will be written,
 *
 * @param surface_format_count number of surface formats available or queried
 * @param surface_formats either nullptr or a pointer to an array of SurfaceFormat values
 */
void GetSurfaceFormats(uint32_t* surface_format_count, SurfaceFormat* surface_formats);

/**
 * Set the framebuffer surface format, do this before calling viz::Init().
 *
 * Default is 'B8G8R8A8_UNORM'.
 *
 * @param surface_format framebuffer surface format
 */
void SetSurfaceFormat(SurfaceFormat surface_format);

/**
 * Get the supported image formats.
 *
 * `viz::Init()` has to be called before since the image formats depend on the physical device.
 *
 * If `image_formats` is nullptr, then the number of image formats supported for the current device
 * is returned in `image_format_count`. Otherwise, `image_format_count` must point to a variable set
 * by the application to the number of elements in the `image_formats` array, and on return the
 * variable is overwritten with the number of values actually written to `image_formats`. If the
 * value of `image_format_count` is less than the number of image formats supported, at most
 * `image_format_count` values will be written,
 *
 * @param image_format_count number of image formats available or queried
 * @param image_formats either nullptr or a pointer to an array of ImageFormat values
 */
void GetImageFormats(uint32_t* image_format_count, ImageFormat* image_formats);

/**
 * Set the font used to render text, do this before calling viz::Init().
 *
 * The font is converted to bitmaps, these bitmaps are scaled to the final size when rendering.
 *
 * @param path path to TTF font file
 * @param size_in_pixels size of the font bitmaps
 */
void SetFont(const char* path, float size_in_pixels);

/**
 * Set the CUDA stream used by Holoviz for CUDA operations.
 *
 * The default stream is 0, i.e. non-concurrent mode. All CUDA commands issued by Holoviz
 * use that stream.
 * The stream can be changed any time.
 *
 * @param stream    CUDA stream to use
 */
void SetCudaStream(CUstream stream);

/**
 * Check if the window close button had been pressed
 *
 * @return true if the window close button had been pressed
 */
bool WindowShouldClose();

/**
 * Check if the window is minimized. This can be used to skip rendering on minimized windows.
 *
 * @return true if the window is minimized
 */
bool WindowIsMinimized();

/**
 * Shutdown Holoviz. All resources are destroyed.
 *
 * @param instance optional instance to shutdown, else shutdown active instance
 */
void Shutdown(InstanceHandle instance = nullptr);

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
 * Layer properties (priority and opacity) are set to the defaults.
 */
void BeginImageLayer();

/**
 * Defines the image data for this layer, source is CUDA device memory.
 *
 * If the image has a alpha value it's multiplied with the layer opacity.
 *
 * If fmt is a depth format, the image will be interpreted as a depth image, and will be written
 * to the depth buffer when rendering the color image from a separate invocation of Image*() for
 * the same layer. This enables depth-compositing image layers with other Holoviz layers.
 * Supported depth formats are: D16_UNORM, X8_D24_UNORM, D32_SFLOAT.
 *
 * Supports multi-planar images (e.g. YUV), `device_ptr` and `row_pitch` specify the parameters
 * for the first plane (plane 0), `device_ptr_n` and `row_pitch_n` for subsequent planes.
 *
 * @param width         width of the image
 * @param height        height of the image
 * @param fmt           image format
 * @param device_ptr    CUDA device memory pointer
 * @param row_pitch     the number of bytes between each row, if zero then data is
 * assumed to be contiguous in memory
 * @param device_ptr_plane_1    CUDA device memory pointer for plane 1
 * @param row_pitch_plane_1 the number of bytes between each row for plane 1, if zero then data is
 * assumed to be contiguous in memory
 * @param device_ptr_plane_2    CUDA device memory pointer for plane 2
 * @param row_pitch_plane_2 the number of bytes between each row for plane 2, if zero then data is
 * assumed to be contiguous in memory
 */
void ImageCudaDevice(uint32_t width, uint32_t height, ImageFormat fmt, CUdeviceptr device_ptr,
                     size_t row_pitch = 0, CUdeviceptr device_ptr_plane_1 = 0,
                     size_t row_pitch_plane_1 = 0, CUdeviceptr device_ptr_plane_2 = 0,
                     size_t row_pitch_plane_2 = 0);

/**
 * Defines the image data for this layer, source is a CUDA array.
 *
 * If the image has a alpha value it's multiplied with the layer opacity.
 *
 * If fmt is a depth format, the image will be interpreted as a depth image, and will be written
 * to the depth buffer when rendering the color image from a separate invocation of Image*() for
 * the same layer. This enables depth-compositing image layers with other Holoviz layers.
 * Supported depth formats are: D16_UNORM, X8_D24_UNORM, D32_SFLOAT.
 *
 * @param fmt       image format
 * @param array     CUDA array
 */
void ImageCudaArray(ImageFormat fmt, CUarray array);

/**
 * Defines the image data for this layer, source is host memory.
 *
 * If the image has a alpha value it's multiplied with the layer opacity.
 *
 * If fmt is a depth format, the image will be interpreted as a depth image, and will be written
 * to the depth buffer when rendering the color image from a separate invocation of Image*() for
 * the same layer. This enables depth-compositing image layers with other Holoviz layers.
 * Supported depth formats are: D16_UNORM, X8_D24_UNORM, D32_SFLOAT.
 *
 * Supports multi-planar images (e.g. YUV), `device_ptr` and `row_pitch` specify the parameters
 * for the first plane (plane 0), `device_ptr_n` and `row_pitch_n` for subsequent planes.
 *
 * @param width     width of the image
 * @param height    height of the image
 * @param fmt       image format
 * @param data      host memory pointer
 * @param row_pitch the number of bytes between each row, if zero then data is assumed to be
 * contiguous in memory
 * @param data_plane_1      host memory pointer for plane 1
 * @param row_pitch_plane_1 the number of bytes between each row for plane 1, if zero then data is
 * assumed to be contiguous in memory
 * @param data_plane_2      host memory pointer for plane 2
 * @param row_pitch_plane_2 the number of bytes between each row for plane 2, if zero then data is
 * assumed to be contiguous in memory
 */
void ImageHost(uint32_t width, uint32_t height, ImageFormat fmt, const void* data,
               size_t row_pitch = 0, const void* data_plane_1 = nullptr,
               size_t row_pitch_plane_1 = 0, const void* data_plane_2 = nullptr,
               size_t row_pitch_plane_2 = 0);

/**
 * Defines the lookup table for this image layer.
 *
 * If a lookup table is used the image format has to be a single channel integer or
 * float format (e.g. ::ImageFormat::R8_UINT, ::ImageFormat::R16_UINT, ::ImageFormat::R32_UINT,
 * ::ImageFormat::R8_UNORM, ::ImageFormat::R16_UNORM, ::ImageFormat::R32_SFLOAT).
 *
 * If normalized is 'true' the function processed is as follow
 * @code{.cpp}
 *  out = lut[clamp(in, 0.0, 1.0)]
 * @endcode
 * Input image values are clamped to the range of the lookup table size: `[0.0, 1.0[`.
 *
 * If normalized is 'false' the function processed is as follow
 * @code{.cpp}
 *  out = lut[clamp(in, 0, size)]
 * @endcode
 * Input image values are clamped to the range of the lookup table size: `[0.0, size[`.
 *
 * @param size      size of the lookup table in elements
 * @param fmt       lookup table color format
 * @param data_size size of the lookup table data in bytes
 * @param data      host memory pointer to lookup table data
 * @param normalized if true then the range of the lookup table is '[0.0, 1.0[', else it is
 * `[0.0, size[`
 */
void LUT(uint32_t size, ImageFormat fmt, size_t data_size, const void* data,
         bool normalized = false);

/**
 * Specifies how the color components of an image are mapped to the color components of the output.
 * Output components can be set to the R, G, B or A component of the input or fixed to zero or one
 * or just identical to the input.
 *
 * Default: all output components are identical to the input components
 * (ComponentSwizzle::IDENTITY).
 *
 * This can be used display an image in color formats which are not natively supported by Holoviz.
 * For example to display a BGRA image:
 * @code{.cpp}
 *  ImageComponentMapping(ComponentSwizzle::B, ComponentSwizzle::G, ComponentSwizzle::R,
 *    ComponentSwizzle::A);
 *  ImageHost(width, height, ImageFormat::R8G8B8A8_UNORM, bgra_data);
 * @endcode
 * or to display a single component image in gray scale:
 * @code{.cpp}
 *  ImageComponentMapping(ComponentSwizzle::R, ComponentSwizzle::R, ComponentSwizzle::R,
 *    ComponentSwizzle::ONE);
 *  ImageHost(width, height, ImageFormat::R8_UNORM, single_component_data);
 * @endcode
 *
 * @param r, g, b, a    sets how the component values are placed in each component of the output
 */
void ImageComponentMapping(ComponentSwizzle r, ComponentSwizzle g, ComponentSwizzle b,
                           ComponentSwizzle a);

/**
 * Specifies the YUV model conversion.
 *
 * @param yuv_model_conversion YUV model conversion. Default is `YUV_601`.
 */
void ImageYuvModelConversion(YuvModelConversion yuv_model_conversion);

/**
 * Specifies the YUV range.
 *
 * @param yuv_range YUV range. Default is `ITU_FULL`.
 */
void ImageYuvRange(YuvRange yuv_range);

/**
 * Defines the location of downsampled chroma component samples relative to the luma samples.
 *
 * @param x_chroma_location chroma location in x direction for formats which are chroma downsampled
 * in width (420 and 422). Default is `COSITED_EVEN`.
 * @param y_chroma_location chroma location in y direction for formats which are chroma downsampled
 * in height (420). Default is `COSITED_EVEN`.
 */
void ImageChromaLocation(ChromaLocation x_chroma_location, ChromaLocation y_chroma_location);

/**
 * Start a ImGUI layer.
 *
 * Layer properties (priority and opacity) are set to the defaults.
 */
void BeginImGuiLayer();

/**
 * Start a geometry layer.
 *
 * Layer properties (priority and opacity) are set to the defaults.
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
void Primitive(PrimitiveTopology topology, uint32_t primitive_count, size_t data_size,
               const float* data);

/**
 * Draw a geometric primitive, source is CUDA device memory.
 *
 * @param topology          primitive topology
 * @param primitive_count   primitive count
 * @param data_size         size of the data array in floats
 * @param data              CUDA device memory pointer to data, the format and size of the array
 *                          depends on the primitive count and topology
 */
void PrimitiveCudaDevice(PrimitiveTopology topology, uint32_t primitive_count, size_t data_size,
                         CUdeviceptr data);

/**
 * Draw text.
 *
 * @param x     x coordinate
 * @param y     y coordinate
 * @param size  font size
 * @param text  text to draw
 */
void Text(float x, float y, float size, const char* text);

/**
 * Render a depth map.
 *
 * Depth maps are rectangular 2D arrays where each element represents a depth value. The data is
 * rendered as a 3D object using points, lines or triangles.
 * Additionally a 2D array with a color value for each point in the grid can be specified.
 * Supported depth formats are: R8_UNORM, D32_SFLOAT.
 *
 * @param render_mode       depth map render mode
 * @param width             width of the depth map
 * @param height            height of the depth map
 * @param depth_fmt         format of the depth map data
 * @param depth_device_ptr  CUDA device memory pointer holding the depth data
 * @param color_fmt         format of the color data (has to be ImageFormat::R8G8B8A8_UNORM)
 * @param color_device_ptr  CUDA device memory pointer holding the color data (optional)
 */
void DepthMap(DepthMapRenderMode render_mode, uint32_t width, uint32_t height,
              ImageFormat depth_fmt, CUdeviceptr depth_device_ptr, ImageFormat color_fmt,
              CUdeviceptr color_device_ptr);

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
 * Add a layer view.
 *
 * By default a layer will fill the whole window. When using a view the layer can be placed
 * freely within the window.
 *
 * Layers can also be placed in 3D space by specifying a 3D transformation matrix.
 * Note that for geometry layers there is a default matrix which allows coordinates in the range
 * of [0 ... 1] instead of the Vulkan [-1 ... 1] range. When specifying a matrix for a geometry
 * layer, this default matrix is overwritten.
 *
 * When multiple views are specified the layer is drawn multiple times using the specified
 * layer views.
 *
 * It's possible to specify a negative term for height, which flips the image. When using a negative
 * height, one should also adjust the y value to point to the lower left corner of the viewport
 * instead of the upper left corner.
 *
 * @param offset_x, offset_y offset of top-left corner of the view. Top left coordinate of the
 * window area is (0, 0) bottom right coordinate is (1, 1)
 * @param width, height      width and height of the view in normalized range. 1.0 is full size.
 * @param matrix             row major 4x4 transform matrix (optional, can be nullptr)
 */
void LayerAddView(float offset_x, float offset_y, float width, float height,
                  const float* matrix = nullptr);

/**
 * End the current layer.
 */
void EndLayer();

/**
 * Read an image from the framebuffer and store it to CUDA device memory.
 *
 * If fmt is a depth format, the depth image attachment of the framebuffer will be copied to
 * device_ptr.
 *
 * Can only be called outside of Begin()/End().
 *
 * @param fmt           image format, currently only R8G8B8A8_UNORM, R8G8B8A8_SRGB and D32_SFLOAT
 *                      are supported
 * @param width, height width and height of the region to read back, will be limited to the
 *                      framebuffer size if the framebuffer is smaller than that
 * @param buffer_size   size of the storage buffer in bytes
 * @param device_ptr    pointer to CUDA device memory to store the framebuffer into
 * @param row_pitch     the number of bytes between each row, if zero then data is assumed to be
 * contiguous in memory
 */
void ReadFramebuffer(ImageFormat fmt, uint32_t width, uint32_t height, size_t buffer_size,
                     CUdeviceptr device_ptr, size_t row_pitch = 0);

/**
 * Set the camera eye, look at and up vectors.
 *
 * @param eye_x, eye_y, eye_z               eye position
 * @param look_at_x, look_at_y, look_at_z   look at position
 * @param up_x, up_y, up_z                  up vector
 * @param anim                              animate transition
 */
void SetCamera(float eye_x, float eye_y, float eye_z, float look_at_x, float look_at_y,
               float look_at_z, float up_x, float up_y, float up_z, bool anim = false);

/**
 * Get the camera pose.
 *
 * The camera parameters are returned in a 4x4 row major projection matrix.
 *
 * The camera is operated using the mouse.
 *  - Orbit        (LMB)
 *  - Pan          (LMB + CTRL  | MMB)
 *  - Dolly        (LMB + SHIFT | RMB | Mouse wheel)
 *  - Look Around  (LMB + ALT   | LMB + CTRL + SHIFT)
 *  - Zoom         (Mouse wheel + SHIFT)
 *
 * @param size   size of the memory `matrix` points to in floats
 * @param matrix pointer to a float array to store the row major projection matrix to
 */
void GetCameraPose(size_t size, float* matrix);

/**
 * Get the camera pose.
 *
 * The camera parameters are returned as camera extrinsics parameters.
 * The extrinsics camera matrix is defined as follow:
 * @code
 *  T = [R | t]
 * @endcode
 *
 * The camera is operated using the mouse.
 *  - Orbit        (LMB)
 *  - Pan          (LMB + CTRL  | MMB)
 *  - Dolly        (LMB + SHIFT | RMB | Mouse wheel)
 *  - Look Around  (LMB + ALT   | LMB + CTRL + SHIFT)
 *  - Zoom         (Mouse wheel + SHIFT)
 *
 * @param rotation row major rotation matrix
 * @param translation translation vector
 */
void GetCameraPose(float (&rotation)[9], float (&translation)[3]);

/**
 * Block until either the present_id is greater than or equal to the current present id, or
 * timeout_ns nanoseconds passes. The present ID is initially zero and increments after each
 * present.
 *
 * @param present_id the presentation presentId to wait for
 * @param timeout_ns timeout in nanoseconds
 * @return true if the present_id is greater than or equal to the current present id, false if
 * timeout_ns nanoseconds passes
 */
bool WaitForPresent(uint64_t present_id, uint64_t timeout_ns);

/**
 * Block until either the display_event_type is signaled, or timeout_ns nanoseconds passes.
 *
 * @param display_event_type display event type
 * @param timeout_ns timeout in nanoseconds
 * @return true if the display_event_type is signaled, false if timeout_ns nanoseconds passes
 */
bool WaitForDisplayEvent(DisplayEventType display_event_type, uint64_t timeout_ns);

/**
 * @returns the counter incrementing once every time a vertical blanking period occurs on the
 * display associated window or the display selected in full screen mode or exclusive display mode.
 */
uint64_t GetVBlankCounter();

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_HOLOVIZ_HOLOVIZ_HPP */
