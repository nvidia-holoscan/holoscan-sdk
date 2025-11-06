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

#ifndef HOLOSCAN_OPERATORS_HOLOVIZ_HOLOVIZ_HPP
#define HOLOSCAN_OPERATORS_HOLOVIZ_HOLOVIZ_HPP

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/file_fifo_mutex.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

#include <holoviz/callbacks.hpp>

namespace holoscan::viz {

typedef void* InstanceHandle;

}  // namespace holoscan::viz

namespace holoscan {
class FirstPixelOutCondition;
class PresentDoneCondition;
}  // namespace holoscan

namespace holoscan::ops {

// forward declaration
struct BufferInfo;

/**
 * @brief Operator class for data visualization.
 *
 * This high-speed viewer handles compositing, blending, and visualization of RGB or RGBA images,
 * masks, geometric primitives, text and depth maps. The operator can auto detect the format of the
 * input tensors acquired at the `receivers` port. Else the input specification can be set at
 * creation time using the `tensors` parameter or at runtime when passing input specifications to
 * the `input_specs` port.
 *
 * Depth maps and 3D geometry are rendered in 3D and support camera movement. The camera is
 * controlled using the mouse:
 *    - Orbit        (LMB)
 *    - Pan          (LMB + CTRL  | MMB)
 *    - Dolly        (LMB + SHIFT | RMB | Mouse wheel)
 *    - Look Around  (LMB + ALT   | LMB + CTRL + SHIFT)
 *    - Zoom         (Mouse wheel + SHIFT)
 * Or by providing new values at the `camera_eye_input`, `camera_look_at_input` or `camera_up_input`
 * input ports. The camera pose can be output at the `camera_pose_output` port when
 * `enable_camera_pose_output` is set to `true`.
 *
 * Callbacks can be used to receive updates on key presses, mouse position and buttons, and window
 * size.
 *
 * ==Named Inputs==
 *
 * - **receivers** : multi-receiver accepting `nvidia::gxf::Tensor` and/or
 *   `nvidia::gxf::VideoBuffer`
 *   - Any number of upstream ports may be connected to this `receivers` port. This port can
 *     accept either VideoBuffers or Tensors. These inputs can be in either host or device
 *     memory. Each tensor or video buffer will result in a layer. The operator autodetects the
 *     layer type for certain input types (e.g. a video buffer will result in an image layer).
 *     For other input types or more complex use cases, input specifications can be provided
 *     either at initialization time as a parameter or dynamically at run time (via
 *     `input_specs`). On each call to `compute`, tensors corresponding to all names specified
 *     in the `tensors` parameter must be found or an exception will be raised. Any extra,
 *     named tensors not present in the `tensors` parameter specification (or optional, dynamic
 *     `input_specs` input) will be ignored.
 * - **input_specs** : `std::vector<holoscan::ops::HolovizOp::InputSpec>` (optional)
 *   - A list of `InputSpec` objects. This port can be used to dynamically update the overlay
 *     specification at run time. No inputs are required on this port in order for the operator
 *     to `compute`.
 * - **render_buffer_input** : `nvidia::gxf::VideoBuffer` (optional)
 *   - An empty render buffer can optionally be provided. The video buffer must have format
 *     GXF_VIDEO_FORMAT_RGBA and be in device memory. This input port only exists if
 *     `enable_render_buffer_input` was set to true, in which case `compute` will only be
 *     called when a message arrives on this input.
 * - **depth_buffer_input** : `nvidia::gxf::VideoBuffer` (optional)
 *   - An empty depth buffer can optionally be provided. The video buffer must have format
 *     GXF_VIDEO_FORMAT_D32F and be in device memory. This input port only exists if
 *     `enable_depth_buffer_input` was set to true, in which case `compute` will only be
 *     called when a message arrives on this input.
 * - **camera_eye_input** : `std::array<float, 3>` (optional)
 *   - Camera eye position. The camera is animated to reach the new position.
 * - **camera_look_at_input** : `std::array<float, 3>` (optional)
 *   - Camera look at position. The camera is animated to reach the new position.
 * - **camera_up_input** :  : `std::array<float, 3>` (optional)
 *   - Camera up vector. The camera is animated to reach the new vector.
 *
 * ==Named Outputs==
 *
 * - **render_buffer_output** : `nvidia::gxf::VideoBuffer` (optional)
 *   - Output for a filled render buffer. If an input render buffer is specified, it is using
 *     that one, else it allocates a new buffer. The video buffer will have format
 *     GXF_VIDEO_FORMAT_RGBA and will be in device memory. This output is useful for offline
 *     rendering or headless mode. This output port only exists if `enable_render_buffer_output`
 *     was set to true.
 * - **depth_buffer_output** : `nvidia::gxf::VideoBuffer` (optional)
 *   - Output for a filled depth buffer. If an input depth buffer is specified, it is using
 *     that one, else it allocates a new buffer. The video buffer will have format
 *     GXF_VIDEO_FORMAT_D32F and will be in device memory. This output is useful for offline
 *     rendering or headless mode. This output port only exists if `enable_depth_buffer_output`
 *     was set to true.
 * - **camera_pose_output** : `std::array<float, 16>` or `nvidia::gxf::Pose3D` (optional)
 *   - Output the camera pose. Depending on the value of `camera_pose_output_type` this outputs a
 *     4x4 row major projection matrix (type `std::array<float, 16>`) or the camera extrinsics
 *     model (type `nvidia::gxf::Pose3D`). This output port only exists if
 *     `enable_camera_pose_output` was set to `True`.
 *
 * ==Parameters==
 *
 * - **receivers**: List of input queues to component accepting `gxf::Tensor` or
 *   `gxf::VideoBuffer`.
 *   - type: `std::vector<gxf::Handle<gxf::Receiver>>`
 * - **enable_render_buffer_input**: Enable `render_buffer_input` (default: `false`)
 *   - type: `bool`
 * - **enable_render_buffer_output**: Enable `render_buffer_output` (default: `false`)
 *   - type: `bool`
 * - **enable_depth_buffer_input**: Enable `depth_buffer_input` (default: `false`)
 *   - type: `bool`
 * - **enable_depth_buffer_output**: Enable `depth_buffer_output` (default: `false`)
 *   - type: `bool`
 * - **enable_camera_pose_output**: Enable `camera_pose_output` (default: `false`)
 *   - type: `bool`
 * - **tensors**: List of input tensor specifications (default: `[]`)
 *   - type: `std::vector<InputSpec>`
 *     - **name**: name of the tensor containing the input data to display
 *       - type: `std::string`
 *     - **type**: input type (default `"unknown"`)
 *       - type: `std::string`
 *       - possible values:
 *         - **unknown**: unknown type, the operator tries to guess the type by inspecting the
 *           tensor.
 *         - **color**: RGB or RGBA color 2d image.
 *         - **color_lut**: single channel 2d image, color is looked up.
 *         - **points**: point primitives, one coordinate (x, y) per primitive.
 *         - **lines**: line primitives, two coordinates (x0, y0) and (x1, y1) per primitive.
 *         - **line_strip**: line strip primitive, a line primitive i is defined by each
 *           coordinate (xi, yi) and the following (xi+1, yi+1).
 *         - **triangles**: triangle primitive, three coordinates (x0, y0), (x1, y1) and (x2, y2)
 *           per primitive.
 *         - **crosses**: cross primitive, a cross is defined by the center coordinate and the
 *           size (xi, yi, si).
 *         - **rectangles**: axis aligned rectangle primitive, each rectangle is defined by two
 *           coordinates (xi, yi) and (xi+1, yi+1).
 *         - **ovals**: oval primitive, an oval primitive is defined by the center coordinate and
 *           the axis sizes (xi, yi, sxi, syi).
 *         - **text**: text is defined by the top left coordinate and the size (x, y, s) per
 *           string, text strings are defined by InputSpec member **text**.
 *         - **depth_map**: single channel 2d array where each element represents a depth value.
 *           The data is rendered as a 3d object using points, lines or triangles. The color for
 *           the elements can be specified through `depth_map_color`.
 *           Supported formats for the depth map:
 *            - 8-bit unsigned normalized format that has a single 8-bit depth component
 *            - 32-bit signed float format that has a single 32-bit depth component
 *         - **depth_map_color**: RGBA 2d image, same size as the depth map. One color value for
 *           each element of the depth map grid. Supported format: 32-bit unsigned normalized
 *           format that has an 8-bit R component in byte 0, an 8-bit G component in byte 1, an
 *           8-bit B component in byte 2, and an 8-bit A component in byte 3.
 *     - **opacity**: layer opacity, 1.0 is fully opaque, 0.0 is fully transparent (default:
 *       `1.0`)
 *       - type: `float`
 *     - **priority**: layer priority, determines the render order, layers with higher priority
 *         values are rendered on top of layers with lower priority values (default: `0`)
 *       - type: `int32_t`
 *     - **image_format**: color image format, used if `type` is `color`, `color_lut` or
 *         `depth_map_color`. (default: `auto_detect`).
 *       - type: `std::string`
 *     - **color**: RGBA color of rendered geometry (default: `[1.F, 1.F, 1.F, 1.F]`)
 *       - type: `std::vector<float>`
 *     - **line_width**: line width for geometry made of lines (default: `1.0`)
 *       - type: `float`
 *     - **point_size**: point size for geometry made of points (default: `1.0`)
 *       - type: `float`
 *     - **text**: array of text strings, used when `type` is `text`. (default: `[]`)
 *       - type: `std::vector<std::string>`
 *     - **depth_map_render_mode**: depth map render mode (default: `points`)
 *       - type: `std::string`
 *       - possible values:
 *         - **points**: render as points
 *         - **lines**: render as lines
 *         - **triangles**: render as triangles
 * - **color_lut**: Color lookup table for tensors of type 'color_lut', vector of four float
 *   RGBA values
 *   - type: `std::vector<std::vector<float>>`
 * - **window_title**: Title on window canvas (default: `"Holoviz"`)
 *   - type: `std::string`
 * - **display_name**: In exclusive display or fullscreen mode, name of display to use as shown
 *   with `xrandr` or `hwinfo --monitor` (default: `""`)
 *   - type: `std::string`
 * - **width**: Window width or display resolution width if in exclusive display or fullscreen mode
 *   (default: `1920`)
 *   - type: `uint32_t`
 * - **height**: Window height or display resolution height if in exclusive display or fullscreen
 *   mode (default: `1080`)
 *   - type: `uint32_t`
 * - **framerate**: Display framerate if in exclusive display mode (default: `60`)
 *   - type: `uint32_t`
 * - **use_exclusive_display**: Enable exclusive display mode (default: `false`)
 *   - type: `bool`
 * - **fullscreen**: Enable fullscreen window (default: `false`)
 *   - type: `bool`
 * - **headless**: Enable headless mode. No window is opened, the render buffer can be output to
 *   `render_buffer_output` and/or `depth_buffer_output` if enabled. (default: `false`)
 *   - type: `bool`
 * - **framebuffer_srgb**: Enable sRGB framebuffer. If set to true, the operator will use an sRGB
 *   framebuffer for rendering. If set to false, the operator will use a linear framebuffer.
 *   (default: `false`)
 *   - type: `bool`
 * - **vsync**: Enable vertical sync. If set to true the operator waits for the next vertical
 *   blanking period of the display to update the current image. (default: `false`)
 *   - type: `bool`
 * - **display_color_space**: Set the display color space. Supported color spaces depend on
 *   the display setup. 'ColorSpace::SRGB_NONLINEAR' is always supported. In headless mode, only
 *   'ColorSpace::PASS_THROUGH' is supported since there is no display. For other color spaces the
 *   display needs to be configured for HDR (default: `ColorSpace::AUTO`)
 *   - type: `std::string`
 * - **window_close_condition.**: BooleanCondition on the operator that will cause it to stop
 *   executing if the display window is closed. By default, this condition is created
 *   automatically during HolovizOp::initialize. The user may want to provide it if, for example
 *   there are multiple HolovizOp operators and you want to share the same window close condition
 *   across both. By sharing the same condition, if one of the display windows is closed it would
 *   also close the other(s).
 * - **window_close_scheduling_term**: This is a deprecated parameter name for
 *   `window_close_condition`. Please use `window_close_condition` instead as
 *   `window_close_scheduling_term` will be removed in a future release.
 *   - type: `gxf::Handle<gxf::BooleanSchedulingTerm>`
 * - **allocator**: Allocator used to allocate memory for `render_buffer_output` and
 * `depth_buffer_output`
 *   - type: `gxf::Handle<gxf::Allocator>`
 * - **font_path**: File path for the font used for rendering text (default: `""`)
 *   - type: `std::string`
 * - **cuda_stream_pool**: Instance of gxf::CudaStreamPool
 *   - type: `gxf::Handle<gxf::CudaStreamPool>`
 * - **camera_pose_output_type**: Type of data output at `camera_pose_output`. Supported values are
 *   `projection_matrix` and `extrinsics_model`. Default value is `projection_matrix`.
 *   - type: `std::string`
 * - **camera_eye**: Initial camera eye position.
 *   - type: `std::array<float, 3>`
 * - **camera_look_at**: Initial camera look at position.
 *   - type: `std::array<float, 3>`
 * - **camera_up**: Initial camera up vector.
 *   - type: `std::array<float, 3>`
 * - **key_callback**: The callback function is called when a key is pressed, released or repeated.
 *   - type: `KeyCallbackFunction`
 * - **unicode_char_callback**: The callback function is called when a Unicode character is input.
 *   - type: `UnicodeCharCallbackFunction`
 * - **mouse_button_callback**: The callback function is called when a mouse button is pressed or
 *   released.
 *   - type: `MouseButtonCallbackFunction`
 * - **scroll_callback**: The callback function is called when a scrolling device is used, such as a
 *   mouse scroll wheel or the scroll area of a touch pad.
 *   - type: `ScrollCallbackFunction`
 * - **cursor_pos_callback**: The callback function is called when the cursor position changes.
 *   Coordinates are provided in screen coordinates, relative to the upper left edge of the content
 *   area.
 *   - type: `CursorPosCallbackFunction`
 * - **framebuffer_size_callback**: The callback function is called when the framebuffer is resized.
 *   - type: `FramebufferSizeCallbackFunction`
 * - **window_size_callback**: The callback function is called when the window is resized.
 *   - type: `:WindowSizeCallbackFunction`
 * - **layer_callback**: The callback function is called when HolovizOp processed all layers
 *   defined by the input specification. It can be used to add extra layers.
 *   - type: `LayerCallbackFunction`
 *
 * ==Device Memory Requirements==
 *
 * If `render_buffer_input` or `depth_buffer_input` is enabled, the provided buffer is used and no
 * memory block will be allocated. Otherwise, when using this operator with a `BlockMemoryPool`, a
 * single device memory block is needed (`storage_type` = 1). The size of this memory block can be
 * determined by rounding the width and height up to the nearest even size and then padding the rows
 * as needed so that the row stride is a multiple of 256 bytes. C++ code to calculate the block size
 * is as follows:
 *
 * ```cpp
 * #include <cstdint>
 *
 * int64_t get_block_size(int32_t height, int32_t width) {
 *   int32_t height_even = height + (height & 1);
 *   int32_t width_even = width + (width & 1);
 *   int64_t row_bytes = width_even * 4;  // 4 bytes per pixel for 8-bit RGBA or 32-bit depth
 *   int64_t row_stride = (row_bytes % 256 == 0) ? row_bytes : ((row_bytes / 256 + 1) * 256);
 *   return height_even * row_stride;
 * }
 * ```
 *
 * ==Notes==
 *
 * 1. Displaying Color Images
 *
 *    Image data can either be on host or device (GPU). Multiple image formats are supported
 *    - R 8 bit unsigned
 *    - R 16 bit unsigned
 *    - R 16 bit float
 *    - R 32 bit unsigned
 *    - R 32 bit float
 *    - RGB 8 bit unsigned
 *    - BGR 8 bit unsigned
 *    - RGBA 8 bit unsigned
 *    - BGRA 8 bit unsigned
 *    - RGBA 16 bit unsigned
 *    - RGBA 16 bit float
 *    - RGBA 32 bit float
 *
 *    When the `type` parameter is set to `color_lut` the final color is looked up using the values
 *    from the `color_lut` parameter. For color lookups these image formats are supported
 *    - R 8 bit unsigned
 *    - R 16 bit unsigned
 *    - R 32 bit unsigned
 *
 * 2. Drawing Geometry
 *
 *    In all cases, `x` and `y` are normalized coordinates in the range `[0, 1]`. The `x` and `y`
 *    correspond to the horizontal and vertical axes of the display, respectively. The origin `(0,
 *    0)` is at the top left of the display.
 *    Geometric primitives outside of the visible area are clipped.
 *    Coordinate arrays are expected to have the shape `(N, C)` where `N` is the coordinate count
 *    and `C` is the component count for each coordinate.
 *
 *    - Points are defined by a `(x, y)` coordinate pair.
 *    - Lines are defined by a set of two `(x, y)` coordinate pairs.
 *    - Lines strips are defined by a sequence of `(x, y)` coordinate pairs. The first two
 *      coordinates define the first line, each additional coordinate adds a line connecting to the
 *      previous coordinate.
 *    - Triangles are defined by a set of three `(x, y)` coordinate pairs.
 *    - Crosses are defined by `(x, y, size)` tuples. `size` specifies the size of the cross in the
 *      `x` direction and is optional, if omitted it's set to `0.05`. The size in the `y` direction
 *      is calculated using the aspect ratio of the window to make the crosses square.
 *    - Rectangles (bounding boxes) are defined by a pair of 2-tuples defining the upper-left and
 *      lower-right coordinates of a box: `(x1, y1), (x2, y2)`.
 *    - Ovals are defined by `(x, y, size_x, size_y)` tuples. `size_x` and `size_y` are optional, if
 *      omitted they are set to `0.05`.
 *    - Texts are defined by `(x, y, size)` tuples. `size` specifies the size of the text in `y`
 *      direction and is optional, if omitted it's set to `0.05`. The size in the `x` direction is
 *      calculated using the aspect ratio of the window. The index of each coordinate references a
 *      text string from the `text` parameter and the index is clamped to the size of the text
 *      array. For example, if there is one item set for the `text` parameter, e.g.
 *      `text=["my_text"]` and three coordinates, then `my_text` is rendered three times. If
 *      `text=["first text", "second text"]` and three coordinates are specified, then `first text`
 *      is rendered at the first coordinate, `second text` at the second coordinate and then `second
 *      text` again at the third coordinate. The `text` string array is fixed and can't be changed
 *      after initialization. To hide text which should not be displayed, specify coordinates
 *      greater than `(1.0, 1.0)` for the text item, the text is then clipped away.
 *    - 3D Points are defined by a `(x, y, z)` coordinate tuple.
 *    - 3D Lines are defined by a set of two `(x, y, z)` coordinate tuples.
 *    - 3D Lines strips are defined by a sequence of `(x, y, z)` coordinate tuples. The first two
 *      coordinates define the first line, each additional coordinate adds a line connecting to the
 *      previous coordinate.
 *    - 3D Triangles are defined by a set of three `(x, y, z)` coordinate tuples.
 *
 * 3. Displaying Depth Maps
 *
 *    When `type` is `depth_map` the provided data is interpreted as a rectangular array of depth
 *    values. Additionally a 2d array with a color value for each point in the grid can be specified
 *    by setting `type` to `depth_map_color`.
 *
 *    The type of geometry drawn can be selected by setting `depth_map_render_mode`.
 *
 *    Depth maps are rendered in 3D and support camera movement.
 *
 * 4. Output
 *
 *    By default a window is opened to display the rendering, but the extension can also be run in
 *    headless mode with the `headless` parameter.
 *
 *    Using a display in exclusive mode is also supported with the `use_exclusive_display`
 *    parameter. This reduces the latency by avoiding the desktop compositor.
 *
 *    The rendered framebuffer can be output to `render_buffer_output` or `depth_buffer_output` if
 * enabled.
 */
class HolovizOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HolovizOp)

  HolovizOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

  /**
   * Input type.
   *
   * All geometric primitives expect a 1d array of coordinates. Coordinates range from 0.0 (left,
   * top) to 1.0 (right, bottom).
   */
  enum class InputType {
    UNKNOWN,     ///< unknown type, the operator tries to guess the type by inspecting the tensor
    COLOR,       ///< GRAY, RGB or RGBA 2d color image
    COLOR_LUT,   ///< single channel 2d image, color is looked up
    POINTS,      ///< point primitives, one coordinate (x, y) per primitive
    LINES,       ///< line primitives, two coordinates (x0, y0) and (x1, y1) per primitive
    LINE_STRIP,  ///< line strip primitive, a line primitive i is defined by each coordinate (xi,
                 ///< yi) and the following (xi+1, yi+1)
    TRIANGLES,   ///< triangle primitive, three coordinates (x0, y0), (x1, y1) and (x2, y2) per
                 ///< primitive
    CROSSES,     ///< cross primitive, a cross is defined by the center coordinate and the size (xi,
                 ///< yi, si)
    RECTANGLES,  ///< axis aligned rectangle primitive, each rectangle is defined by two coordinates
                 ///< (xi, yi) and (xi+1, yi+1)
    OVALS,  ///< oval primitive, an oval primitive is defined by the center coordinate and the axis
            ///< sizes (xi, yi, sxi, syi)
    TEXT,   ///< text is defined by the top left coordinate and the size (x, y, s) per string, text
            ///< strings are define by InputSpec::text_
    DEPTH_MAP,  ///< single channel 2d array where each element represents a depth value. The data
                ///< is rendered as a 3d object using points, lines or triangles. The color for the
                ///< elements can be specified through `DEPTH_MAP_COLOR`.
                ///< Supported format:
                ///<   8-bit unsigned normalized format that has a single 8-bit depth component
    DEPTH_MAP_COLOR,  ///< RGBA 2d image, same size as the depth map. One color value for each
                      ///< element of the depth map grid.
                      ///< Supported format:
                      ///<   32-bit unsigned normalized format that has an 8-bit R component in byte
                      ///>   0, an 8-bit G component in byte 1, an 8-bit B component in byte 2,
                      ///<   and an 8-bit A component in byte 3
    POINTS_3D,        ///< 3D point primitives, one coordinate (x, y, z) per primitive
    LINES_3D,  ///< 3D line primitives, two coordinates (x0, y0, z0) and (x1, y1, z1) per primitive
    LINE_STRIP_3D,  ///< 3D line strip primitive, a line primitive i is defined by each coordinate
                    ///< (xi, yi, zi) and the following (xi+1, yi+1, zi+1)
    TRIANGLES_3D,  ///< 3D triangle primitive, three coordinates (x0, y0, z0), (x1, y1, z1) and (x2,
                   ///< y2, z2) per primitive
  };

  /**
   * Image formats.
   *
   * {component format}_{numeric format}
   *
   * - component format
   *   - indicates the size in bits of the R, G, B, A or Y, U, V components if present
   * - numeric format
   *   - UNORM - unsigned normalize values, range [0, 1]
   *   - SNORM - signed normalized values, range [-1,1]
   *   - UINT - unsigned integer values, range [0,2n-1]
   *   - SINT - signed integer values, range [-2n-1,2n-1-1]
   *   - SFLOAT - signed floating-point numbers
   *   - SRGB - the R, G, and B components are unsigned normalized values that
   *            represent values using sRGB nonlinear encoding, while the A
   *            component (if one exists) is a regular unsigned normalized value
   * - multi-planar formats
   *   - 2PLANE - data is stored in two separate memory planes
   *   - 3PLANE - data is stored in three separate memory planes
   * - YUV formats
   *   - 420 - the horizontal and vertical resolution of the chroma (UV) planes is halved
   *   - 422 - the horizontal of the chroma (UV) planes is halved
   *
   * Note: this needs to match the viz::ImageFormat enum (except the AUTO_DETECT value).
   */
  enum class ImageFormat {
    R8_UINT,   ///< specifies a one-component, 8-bit unsigned integer format that has
               ///  a single 8-bit R component
    R8_SINT,   ///< specifies a one-component, 8-bit signed integer format that has
               ///  a single 8-bit R component
    R8_UNORM,  ///< specifies a one-component, 8-bit unsigned normalized format that has
               ///  a single 8-bit R component
    R8_SNORM,  ///< specifies a one-component, 8-bit signed normalized format that has
               ///  a single 8-bit R component
    R8_SRGB,   ///< specifies a one-component, 8-bit unsigned normalized format that has
               ///  a single 8-bit R component stored with sRGB nonlinear encoding

    R16_UINT,    ///< specifies a one-component, 16-bit unsigned integer format that has
                 ///  a single 16-bit R component
    R16_SINT,    ///< specifies a one-component, 16-bit signed integer format that has
                 ///  a single 16-bit R component
    R16_UNORM,   ///< specifies a one-component, 16-bit unsigned normalized format that has
                 ///  a single 16-bit R component
    R16_SNORM,   ///< specifies a one-component, 16-bit signed normalized format that has
                 ///  a single 16-bit R component
    R16_SFLOAT,  ///< specifies a one-component, 16-bit signed floating-point format that has
                 ///  a single 16-bit R component

    R32_UINT,    ///< specifies a one-component, 16-bit unsigned integer format that has
                 ///  a single 16-bit R component
    R32_SINT,    ///< specifies a one-component, 16-bit signed integer format that has
                 ///  a single 16-bit R component
    R32_SFLOAT,  ///< specifies a one-component, 32-bit signed floating-point format that has
                 ///  a single 32-bit R component

    R8G8B8_UNORM,  ///< specifies a three-component, 24-bit unsigned normalized format that has
                   ///  a 8-bit R component in byte 0,
                   ///  a 8-bit G component in byte 1,
                   ///  and a 8-bit B component in byte 2
    R8G8B8_SNORM,  ///< specifies a three-component, 24-bit signed normalized format that has
                   ///  a 8-bit R component in byte 0,
                   ///  a 8-bit G component in byte 1,
                   ///  and a 8-bit B component in byte 2
    R8G8B8_SRGB,   ///< specifies a three-component, 24-bit unsigned normalized format that has
                   ///  a 8-bit R component stored with sRGB nonlinear encoding in byte 0,
                   ///  a 8-bit G component stored with sRGB nonlinear encoding in byte 1,
                   ///  and a 8-bit B component stored with sRGB nonlinear encoding in byte 2

    R8G8B8A8_UNORM,  ///< specifies a four-component, 32-bit unsigned normalized format that has
                     ///  a 8-bit R component in byte 0,
                     ///  a 8-bit G component in byte 1,
                     ///  a 8-bit B component in byte 2,
                     ///  and a 8-bit A component in byte 3
    R8G8B8A8_SNORM,  ///< specifies a four-component, 32-bit signed normalized format that has
                     ///  a 8-bit R component in byte 0,
                     ///  a 8-bit G component in byte 1,
                     ///  a 8-bit B component in byte 2,
                     ///  and a 8-bit A component in byte 3
    R8G8B8A8_SRGB,   ///< specifies a four-component, 32-bit unsigned normalized format that has
                     ///  a 8-bit R component stored with sRGB nonlinear encoding in byte 0,
                     ///  a 8-bit G component stored with sRGB nonlinear encoding in byte 1,
                     ///  a 8-bit B component stored with sRGB nonlinear encoding in byte 2,
                     ///  and a 8-bit A component in byte 3

    R16G16B16A16_UNORM,   ///< specifies a four-component,
                          ///  64-bit unsigned normalized format that has
                          ///  a 16-bit R component in bytes 0..1,
                          ///  a 16-bit G component in bytes 2..3,
                          ///  a 16-bit B component in bytes 4..5,
                          ///  and a 16-bit A component in bytes 6..7
    R16G16B16A16_SNORM,   ///< specifies a four-component,
                          ///  64-bit signed normalized format that has
                          ///  a 16-bit R component in bytes 0..1,
                          ///  a 16-bit G component in bytes 2..3,
                          ///  a 16-bit B component in bytes 4..5,
                          ///  and a 16-bit A component in bytes 6..7
    R16G16B16A16_SFLOAT,  ///< specifies a four-component,
                          ///  64-bit signed floating-point format that has
                          ///  a 16-bit R component in bytes 0..1,
                          ///  a 16-bit G component in bytes 2..3,
                          ///  a 16-bit B component in bytes 4..5,
                          ///  and a 16-bit A component in bytes 6..7
    R32G32B32A32_SFLOAT,  ///< specifies a four-component,
                          ///  128-bit signed floating-point format that has
                          ///  a 32-bit R component in bytes 0..3,
                          ///  a 32-bit G component in bytes 4..7,
                          ///  a 32-bit B component in bytes 8..11,
                          ///  and a 32-bit A component in bytes 12..15

    D16_UNORM,     ///< specifies a one-component, 16-bit unsigned normalized format that has
                   ///  a single 16-bit depth component
    X8_D24_UNORM,  ///< specifies a two-component, 32-bit format that has
                   ///  24 unsigned normalized bits in the depth component,
                   ///  and, optionally, 8 bits that are unused
    D32_SFLOAT,    ///< specifies a one-component, 32-bit signed floating-point format that has
                   ///  32 bits in the depth component

    A2B10G10R10_UNORM_PACK32,  ///< specifies a four-component, 32-bit packed unsigned normalized
                               ///  format that has
                               ///  a 2-bit A component in bits 30..31,
                               ///  a 10-bit B component in bits 20..29,
                               ///  a 10-bit G component in bits 10..19,
                               ///  and a 10-bit R component in bits 0..9.

    A2R10G10B10_UNORM_PACK32,  ///< specifies a four-component, 32-bit packed unsigned normalized
                               ///  format that has
                               ///  a 2-bit A component in bits 30..31,
                               ///  a 10-bit R component in bits 20..29,
                               ///  a 10-bit G component in bits 10..19,
                               ///  and a 10-bit B component in bits 0..9.

    B8G8R8A8_UNORM,  ///< specifies a four-component, 32-bit unsigned normalized format that has
                     ///  a 8-bit B component in byte 0,
                     ///  a 8-bit G component in byte 1,
                     ///  a 8-bit R component in byte 2,
                     ///  and a 8-bit A component in byte 3
    B8G8R8A8_SRGB,   ///< specifies a four-component, 32-bit unsigned normalized format that has
                     ///  a 8-bit B component stored with sRGB nonlinear encoding in byte 0,
                     ///  a 8-bit G component stored with sRGB nonlinear encoding in byte 1,
                     ///  a 8-bit R component stored with sRGB nonlinear encoding in byte 2,
                     ///  and a 8-bit A component in byte 3

    A8B8G8R8_UNORM_PACK32,  ///< specifies a four-component, 32-bit packed unsigned normalized
                            ///< format
                            ///  that has
                            ///  an 8-bit A component in bits 24..31,
                            ///  an 8-bit B component in bits 16..23,
                            ///  an 8-bit G component in bits 8..15,
                            ///  and an 8-bit R component in bits 0..7.
    A8B8G8R8_SRGB_PACK32,  ///< specifies a four-component, 32-bit packed unsigned normalized format
                           ///  that has
                           ///  an 8-bit A component in bits 24..31,
                           ///  an 8-bit B component stored with sRGB nonlinear encoding in
                           ///  bits 16..23,
                           ///  an 8-bit G component stored with sRGB nonlinear encoding
                           ///  in bits 8..15,
                           ///  and an 8-bit R component stored with sRGB nonlinear
                           ///  encoding in bits 0..7.

    Y8U8Y8V8_422_UNORM,  ///< specifies a four-component, 32-bit format containing a pair of Y
                         ///  components, a V component, and a U component, collectively encoding a
                         ///  2×1 rectangle of unsigned normalized RGB texel data. One Y value is
                         ///  present at each i coordinate, with the U and V values shared across
                         ///  both Y values and thus recorded at half the horizontal resolution of
                         ///  the image. This format has an 8-bit Y component for the even i
                         ///  coordinate in byte 0, an 8-bit U component in byte 1, an 8-bit Y
                         ///  component for the odd i coordinate in byte 2, and an 8-bit V component
                         ///  in byte 3. This format only supports images with a width that is a
                         ///  multiple of two.
    U8Y8V8Y8_422_UNORM,  ///< specifies a four-component, 32-bit format containing a pair of Y
                         ///  components, a V component, and a U component, collectively encoding a
                         ///  2×1 rectangle of unsigned normalized RGB texel data. One Y value is
                         ///  present at each i coordinate, with the U and V values shared across
                         ///  both Y values and thus recorded at half the horizontal resolution of
                         ///  the image. This format has an 8-bit U component in byte 0, an 8-bit Y
                         ///  component for the even i coordinate in byte 1, an 8-bit V component in
                         ///  byte 2, and an 8-bit Y component for the odd i coordinate in byte 3.
                         ///  This format only supports images with a width that is a multiple of
                         ///  two.
    Y8_U8V8_2PLANE_420_UNORM,  ///< specifies an unsigned normalized multi-planar format that has an
                               ///  8-bit Y component in plane 0, and a two-component, 16-bit UV
                               ///  plane 1 consisting of an 8-bit U component in byte 0 and an
                               ///  8-bit V component in byte 1. The horizontal and vertical
                               ///  dimensions of the UV plane are halved relative to the image
                               ///  dimensions. This format only supports images with a width and
                               ///  height that are a multiple of two.
    Y8_U8V8_2PLANE_422_UNORM,  ///< specifies an unsigned normalized multi-planar format that has an
                               ///  8-bit Y component in plane 0, and a two-component, 16-bit UV
                               ///  plane 1 consisting of an 8-bit U component in byte 0 and an
                               ///  8-bit V component in byte 1. The horizontal dimension of the UV
                               ///  plane is halved relative to the image dimensions. This format
                               ///  only supports images with a width that is a multiple of two.
    Y8_U8_V8_3PLANE_420_UNORM,  ///< specifies an unsigned normalized multi-planar format that has
                                ///< an
                                ///  8-bit Y component in plane 0, an 8-bit U component in plane 1,
                                ///  and an 8-bit V component in plane 2. The horizontal and
                                ///  vertical dimensions of the V and U planes are halved relative
                                ///  to the image dimensions. This format only supports images with
                                ///  a width and height that are a multiple of two.
    Y8_U8_V8_3PLANE_422_UNORM,  ///< specifies an unsigned normalized multi-planar format that has
                                ///< an
                                ///  8-bit Y component in plane 0, an 8-bit U component in plane 1,
                                ///  and an 8-bit V component in plane 2. The horizontal dimension
                                ///  of the V and U plane is halved relative to the image
                                ///  dimensions. This format only supports images with a width that
                                ///  is a multiple of two.
    Y16_U16V16_2PLANE_420_UNORM,  ///< specifies an unsigned normalized multi-planar format that has
                                  ///< a
                                  ///  16-bit Y component in each 16-bit word of plane 0, and a
                                  ///  two-component, 32-bit UV plane 1 consisting of a 16-bit U
                                  ///  component in the word in bytes 0..1, and a 16-bit V component
                                  ///  in the word in bytes 2..3. The horizontal and vertical
                                  ///  dimensions of the UV plane are halved relative to the image
                                  ///  dimensions. This format only supports images with a width and
                                  ///  height that are a multiple of two.
    Y16_U16V16_2PLANE_422_UNORM,  ///< specifies an unsigned normalized multi-planar format that has
                                  ///< a
                                  ///  16-bit Y component in each 16-bit word of plane 0, and a
                                  ///  two-component, 32-bit UV plane 1 consisting of a 16-bit U
                                  ///  component in the word in bytes 0..1, and a 16-bit V component
                                  ///  in the word in bytes 2..3. The horizontal dimension of the UV
                                  ///  plane is halved relative to the image dimensions. This format
                                  ///  only supports images with a width that is a multiple of two.
    Y16_U16_V16_3PLANE_420_UNORM,  ///< specifies an unsigned normalized multi-planar format that
                                   ///< has
                                   ///  a 16-bit Y component in each 16-bit word of plane 0, a
                                   ///  16-bit U component in each 16-bit word of plane 1, and a
                                   ///  16-bit V component in each 16-bit word of plane 2. The
                                   ///  horizontal and vertical dimensions of the V and U planes are
                                   ///  halved relative to the image dimensions. This format only
                                   ///  supports images with a width and height that are a multiple
                                   ///  of two.
    Y16_U16_V16_3PLANE_422_UNORM,  ///< specifies an unsigned normalized multi-planar format that
                                   ///< has
                                   ///  a 16-bit Y component in each 16-bit word of plane 0, a
                                   ///  16-bit U component in each 16-bit word of plane 1, and a
                                   ///  16-bit V component in each 16-bit word of plane 2. The
                                   ///  horizontal dimension of the V and U plane is halved relative
                                   ///  to the image dimensions. This format only supports images
                                   ///  with a width that is a multiple of two.

    AUTO_DETECT = -1  ///< Auto detect the image format. If the input is a video buffer the format
                      ///  of the video buffer is used, if the input is a tensor then the format
                      ///  depends on the component count
                      ///   - one component : gray level image
                      ///   - three components : RGB image
                      ///   - four components : RGBA image
                      ///  and the component type.
  };

  /**
   * Defines the conversion from the source color model to the shader color model.
   */
  enum class YuvModelConversion {
    YUV_601,   ///< specifies the color model conversion from YUV to RGB defined in BT.601
    YUV_709,   ///< specifies the color model conversion from YUV to RGB defined in BT.709
    YUV_2020,  ///< specifies the color model conversion from YUV to RGB defined in BT.2020
  };

  /**
   * Specifies the YUV range
   */
  enum class YuvRange {
    ITU_FULL,    ///< specifies that the full range of the encoded values are valid and
                 ///< interpreted according to the ITU "full range" quantization rules
    ITU_NARROW,  ///< specifies that headroom and foot room are reserved in the numerical range
                 ///< of encoded values, and the remaining values are expanded according to the
                 ///< ITU "narrow range" quantization rules
  };

  /**
   * Defines the location of downsampled chroma component samples relative to the luma samples.
   */
  enum class ChromaLocation {
    COSITED_EVEN,  ///< specifies that downsampled chroma samples are aligned with luma samples with
                   ///< even coordinates
    MIDPOINT,  ///< specifies that downsampled chroma samples are located half way between each even
               ///< luma sample and the nearest higher odd luma sample.
  };

  /**
   * Depth map render mode.
   */
  enum class DepthMapRenderMode {
    POINTS,    ///< render points
    LINES,     ///< render lines
    TRIANGLES  ///< render triangles
  };

  /**
   * The color space specifies how the surface data is interpreted when presented on screen.
   *
   * Note: this needs to match the viz::ColorSpace enum (except the AUTO value).
   */
  enum class ColorSpace {
    SRGB_NONLINEAR,        ///< sRGB color space
    EXTENDED_SRGB_LINEAR,  ///< extended sRGB color space to be displayed using a linear EOTF
    BT2020_LINEAR,         ///< BT2020 color space to be displayed using a linear EOTF
    HDR10_ST2084,  ///< HDR10 (BT2020 color) space to be displayed using the SMPTE ST2084 Perceptual
                   ///< Quantizer (PQ) EOTF
    PASS_THROUGH,  ///< color components are used "as is"
    BT709_LINEAR,  ///< BT709 color space to be displayed using a linear EOTF
    AUTO = -1,  ///< Auto select the color format. Is a display is connected then `SRGB_NONLINEAR`
                ///< is used, in headless mode `PASS_THROUGH` is used.
  };

  /**
   * Input specification
   */
  struct InputSpec {
    InputSpec() = default;
    InputSpec(const std::string& tensor_name, InputType type)
        : tensor_name_(tensor_name), type_(type) {}
    InputSpec(const std::string& tensor_name, const std::string& type_str);

    /**
     * @return an InputSpec from the YAML form output by description()
     */
    explicit InputSpec(const std::string& yaml_description);

    /**
     * @return true if the input spec is valid
     */
    explicit operator bool() const noexcept { return !tensor_name_.empty(); }

    /**
     * @return a YAML string representation of the InputSpec
     */
    std::string description() const;

    std::string tensor_name_;  ///< name of the tensor/video buffer containing the input data
    InputType type_ = InputType::UNKNOWN;  ///< input type
    float opacity_ = 1.F;  ///< layer opacity, 1.0 is fully opaque, 0.0 is fully transparent
    int32_t priority_ =
        0;  ///< layer priority, determines the render order, layers with higher priority values are
            ///< rendered on top of layers with lower priority values
    ImageFormat image_format_ = ImageFormat::AUTO_DETECT;  ///< image format
    YuvModelConversion yuv_model_conversion_ =
        YuvModelConversion::YUV_601;           ///< YUV model conversion
    YuvRange yuv_range_ = YuvRange::ITU_FULL;  ///< YUV range
    ChromaLocation x_chroma_location_ =
        ChromaLocation::COSITED_EVEN;  ///< chroma location in x direction for formats which are
                                       ///< chroma downsampled in width (420 and 422)
    ChromaLocation y_chroma_location_ =
        ChromaLocation::COSITED_EVEN;  ///< chroma location in y direction for formats which are
                                       ///< chroma downsampled in height (420)
    std::vector<float> color_{1.F, 1.F, 1.F, 1.F};  ///< color of rendered geometry
    float line_width_ = 1.F;                        ///< line width for geometry made of lines
    float point_size_ = 1.F;                        ///< point size for geometry made of points
    std::vector<std::string> text_;  ///< array of text strings, used when type_ is TEXT.
    DepthMapRenderMode depth_map_render_mode_ =
        DepthMapRenderMode::POINTS;  ///< depth map render mode, used if type_ is
                                     ///< DEPTH_MAP or DEPTH_MAP_COLOR.

    /**
     * Layer view.
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
     * It's possible to specify a negative term for height, which flips the image. When using a
     * negative height, one should also adjust the y value to point to the lower left corner of the
     * viewport instead of the upper left corner.
     */
    struct View {
      float offset_x_ = 0.F,
            offset_y_ = 0.F;  ///< offset of top-left corner of the view. Top left coordinate of
                              /// the window area is (0, 0) bottom right
                              /// coordinate is (1, 1).
      float width_ = 1.F,
            height_ = 1.F;  ///< width and height of the view in normalized range. 1.0 is full size.
      std::optional<std::array<float, 16>>
          matrix_;  ///< row major 4x4 transform matrix (optional, can be nullptr)
    };
    std::vector<View> views_;
  };

  /// export the types used by the callbacks directly from Holoviz module
  using Key = viz::Key;
  using KeyAndButtonAction = viz::KeyAndButtonAction;
  using KeyModifiers = viz::KeyModifiers;
  using MouseButton = viz::MouseButton;

  /**
   * Function pointer type for key callbacks.
   *
   * The callback function receives:
   * - key: the key that was pressed
   * - action: key action (PRESS, RELEASE, REPEAT)
   * - modifiers: bit field describing which modifiers were held down
   */
  using KeyCallbackFunction =
      std::function<void(Key key, KeyAndButtonAction action, KeyModifiers modifiers)>;

  /**
   * Function pointer type for Unicode character callbacks.
   *
   * The callback function receives:
   * - code_point: Unicode code point of the character
   */
  using UnicodeCharCallbackFunction = std::function<void(uint32_t code_point)>;

  /**
   * Function pointer type for mouse button callbacks.
   *
   * The callback function receives:
   * - button: the mouse button that was pressed
   * - action: button action (PRESS, RELEASE)
   * - modifiers: bit field describing which modifiers were held down
   */
  using MouseButtonCallbackFunction =
      std::function<void(MouseButton button, KeyAndButtonAction action, KeyModifiers modifiers)>;

  /**
   * Function pointer type for scroll callbacks.
   *
   * The callback function receives:
   * - x_offset: scroll offset along the x-axis
   * - y_offset: scroll offset along the y-axis
   */
  using ScrollCallbackFunction = std::function<void(double x_offset, double y_offset)>;

  /**
   * Function pointer type for cursor position callbacks.
   *
   * The callback function receives:
   * - x_pos: new cursor x-coordinate in screen coordinates, relative to the left edge of the
   *   content area
   * - y_pos: new cursor y-coordinate in screen coordinates, relative to the left edge of the
   *   content area
   */
  using CursorPosCallbackFunction = std::function<void(double x_pos, double y_pos)>;

  /**
   * Function pointer type for framebuffer size callbacks.
   *
   * The callback function receives:
   * - width: new width of the framebuffer in pixels
   * - height: new height of the framebuffer in pixels
   */
  using FramebufferSizeCallbackFunction = std::function<void(int width, int height)>;

  /**
   * Function pointer type for window size callbacks.
   *
   * The callback function receives:
   * - width: new width of the window in screen coordinates
   * - height: new height of the window in screen coordinates
   */
  using WindowSizeCallbackFunction = std::function<void(int width, int height)>;

  /**
   * Function pointer type for layer callbacks. This function is called when HolovizOp processed
   * all layers defined by the input specification. It can be used to add extra layers.
   *
   * The callback function receives:
   * - inputs: the entities received from the 'receivers' input port
   */
  using LayerCallbackFunction =
      std::function<void(const std::vector<holoscan::gxf::Entity>& inputs)>;

  /// table to convert input type to string
  static const std::array<std::pair<InputType, std::string>, 17> kInputTypeToStr;

  /**
   * Convert a string to a input type enum
   *
   * @param string input type string
   * @return input type enum
   */
  static nvidia::gxf::Expected<holoscan::ops::HolovizOp::InputType> inputTypeFromString(
      const std::string& string);

  /**
   * Convert a input type enum to a string
   *
   * @param input_type input type enum
   * @return input type string
   */
  static std::string inputTypeToString(holoscan::ops::HolovizOp::InputType input_type);

  /// table to convert image format to string
  static const std::array<std::pair<holoscan::ops::HolovizOp::ImageFormat, std::string>, 41>
      kImageFormatToStr;

  /**
   * Convert a string to a image format enum
   *
   * @param string image format string
   * @return image format enum
   */
  static nvidia::gxf::Expected<holoscan::ops::HolovizOp::ImageFormat> imageFormatFromString(
      const std::string& string);

  /**
   * Convert a image format enum to a string
   *
   * @param image_format image format enum
   * @return image format string
   */
  static std::string imageFormatToString(holoscan::ops::HolovizOp::ImageFormat image_format);

  /// table to convert depth map render mode to string
  static const std::array<std::pair<holoscan::ops::HolovizOp::DepthMapRenderMode, std::string>, 3>
      kDepthMapRenderModeToStr;

  /**
   * Convert a string to a depth map render mode enum
   *
   * @param string depth map render mode string
   * @return depth map render mode enum
   */
  static nvidia::gxf::Expected<holoscan::ops::HolovizOp::DepthMapRenderMode>
  depthMapRenderModeFromString(const std::string& string);

  /**
   * Convert a depth map render mode enum to a string
   *
   * @param depth_map_render_mode depth map render mode enum
   * @return depth map render mode string
   */
  static std::string depthMapRenderModeToString(
      holoscan::ops::HolovizOp::DepthMapRenderMode depth_map_render_mode);

  /// table to convert yuv model conversion enum to string
  static const std::array<std::pair<holoscan::ops::HolovizOp::YuvModelConversion, std::string>, 3>
      kYuvModelConversionToStr;

  /**
   * Convert a string to a yuv model conversion enum
   *
   * @param string yuv model conversion string
   * @return yuv model conversion enum
   */
  static nvidia::gxf::Expected<holoscan::ops::HolovizOp::YuvModelConversion>
  yuvModelConversionFromString(const std::string& string);

  /**
   * Convert a yuv model conversion enum to a string
   *
   * @param yuv_model_conversion yuv model conversion enum
   * @return depth map render mode string
   */
  static std::string yuvModelConversionToString(
      holoscan::ops::HolovizOp::YuvModelConversion yuv_model_conversion);

  /// table to convert yuv range enum to string
  static const std::array<std::pair<holoscan::ops::HolovizOp::YuvRange, std::string>, 2>
      kYuvRangeToStr;

  /**
   * Convert a string to a yuv range enum
   *
   * @param string yuv range string
   * @return yuv range enum
   */
  static nvidia::gxf::Expected<holoscan::ops::HolovizOp::YuvRange> yuvRangeFromString(
      const std::string& string);

  /**
   * Convert a yuv range enum to a string
   *
   * @param yuv_range yuv range enum
   * @return yuv range string
   */
  static std::string yuvRangeToString(holoscan::ops::HolovizOp::YuvRange yuv_range);

  /// table to convert chroma location enum to string
  static const std::array<std::pair<holoscan::ops::HolovizOp::ChromaLocation, std::string>, 2>
      kChromaLoactionToStr;

  /**
   * Convert a string to a chroma location enum
   *
   * @param string chroma location string
   * @return chroma location enum
   */
  static nvidia::gxf::Expected<holoscan::ops::HolovizOp::ChromaLocation> chromaLocationFromString(
      const std::string& string);

  /**
   * Convert a chroma location enum to a string
   *
   * @param chroma_location chroma location enum
   * @return chroma location string
   */
  static std::string chromaLocationToString(
      holoscan::ops::HolovizOp::ChromaLocation chroma_location);

  /// table to convert color space enum to string
  static const std::array<std::pair<holoscan::ops::HolovizOp::ColorSpace, std::string>, 7>
      kColorSpaceToStr;

  /**
   * Convert a string to a color space enum
   *
   * @param string color space string
   * @return color space enum
   */
  static nvidia::gxf::Expected<holoscan::ops::HolovizOp::ColorSpace> colorSpaceFromString(
      const std::string& string);

  /**
   * Convert a color space enum to a string
   *
   * @param color_space color space enum
   * @return color space string
   */
  static std::string colorSpaceToString(holoscan::ops::HolovizOp::ColorSpace color_space);

 protected:
  void disable_via_window_close();

 private:
  friend class ::holoscan::FirstPixelOutCondition;
  friend class ::holoscan::PresentDoneCondition;

  bool enable_conditional_port(const std::string& name,
                               bool set_none_condition_on_disabled = false);
  void set_input_spec(const InputSpec& input_spec);
  void set_input_spec_geometry(const InputSpec& input_spec);
  void read_frame_buffer(InputContext& op_input, OutputContext& op_output,
                         ExecutionContext& context, bool buffer_input_enabled,
                         const std::string& buffer_name, nvidia::gxf::VideoFormat video_format);
  void render_color_image(const InputSpec& input_spec, BufferInfo& buffer_info);
  void render_geometry(const InputSpec& input_spec, BufferInfo& buffer_info, cudaStream_t stream);
  void render_depth_map(InputSpec* const input_spec_depth_map,
                        const BufferInfo& buffer_info_depth_map,
                        InputSpec* const input_spec_depth_map_color,
                        const BufferInfo& buffer_info_depth_map_color);

  /**
   * Block until either the present_id is greater than or equal to the presentation index, or
   * timeout_ns nanoseconds passes. The present ID is initially zero and increments after each
   * present.
   *
   * This function is thread-safe and can be called asynchronously to the operator execution.
   *
   * @param present_id the presentation presentId to wait for
   * @param timeout_ns timeout in nanoseconds
   * @return true if the present_id is greater than or equal to the presentation index, false if
   * timeout_ns nanoseconds passes
   */
  bool wait_for_present(uint64_t present_id, uint64_t timeout_ns);

  /**
   * Block until either the first pixel of the next display refresh cycle leaves the display engine
   * for the display or timeout_ns nanoseconds passes.
   *
   * This function is thread-safe and can be called asynchronously to the operator execution.
   *
   * @param timeout_ns timeout in nanoseconds
   * @return true if the first pixel out event is detected, false if timeout_ns nanoseconds passes
   */
  bool wait_for_first_pixel_out(uint64_t timeout_ns);

  Parameter<holoscan::IOSpec*> render_buffer_input_;
  Parameter<holoscan::IOSpec*> render_buffer_output_;
  Parameter<holoscan::IOSpec*> depth_buffer_input_;
  Parameter<holoscan::IOSpec*> depth_buffer_output_;
  Parameter<holoscan::IOSpec*> camera_pose_output_;

  Parameter<std::vector<InputSpec>> tensors_;

  Parameter<std::vector<std::vector<float>>> color_lut_;

  Parameter<std::string> window_title_;
  Parameter<std::string> display_name_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<float> framerate_;
  Parameter<bool> use_exclusive_display_;
  Parameter<bool> fullscreen_;
  Parameter<bool> headless_;
  Parameter<bool> framebuffer_srgb_;
  Parameter<bool> vsync_;
  Parameter<uint32_t> multiprocess_framedrop_waittime_ms_;
  Parameter<std::string> holoviz_multiprocess_mutex_path_;
  Parameter<ColorSpace> display_color_space_;
  Parameter<std::shared_ptr<BooleanCondition>> window_close_condition_;
  Parameter<std::shared_ptr<BooleanCondition>> window_close_scheduling_term_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_;
  Parameter<std::string> font_path_;
  Parameter<std::string> camera_pose_output_type_;
  Parameter<std::array<float, 3>> camera_eye_;
  Parameter<std::array<float, 3>> camera_look_at_;
  Parameter<std::array<float, 3>> camera_up_;
  std::vector<ImageFormat> supported_image_formats_;

  holoscan::Parameter<KeyCallbackFunction> key_callback_;
  holoscan::Parameter<UnicodeCharCallbackFunction> unicode_char_callback_;
  holoscan::Parameter<MouseButtonCallbackFunction> mouse_button_callback_;
  holoscan::Parameter<ScrollCallbackFunction> scroll_callback_;
  holoscan::Parameter<CursorPosCallbackFunction> cursor_pos_callback_;
  holoscan::Parameter<FramebufferSizeCallbackFunction> framebuffer_size_callback_;
  holoscan::Parameter<WindowSizeCallbackFunction> window_size_callback_;
  holoscan::Parameter<LayerCallbackFunction> layer_callback_;

  // internal state
  viz::InstanceHandle instance_ = nullptr;

  // prevents that holoviz is shutdown while wait_for_first_pixel_out or wait_for_present is running
  std::shared_mutex running_state_mutex_;
  // operator running state
  enum class RunningState {
    NOT_STARTED,
    STARTED,
    STOPPED,
  };
  std::atomic<RunningState> running_state_ = RunningState::NOT_STARTED;

  std::vector<float> lut_;
  std::vector<InputSpec> initial_input_spec_;
  bool render_buffer_input_enabled_ = false;
  bool render_buffer_output_enabled_ = false;
  bool depth_buffer_input_enabled_ = false;
  bool depth_buffer_output_enabled_ = false;
  bool camera_pose_output_enabled_ = false;
  bool is_first_tick_ = true;
  bool is_holoviz_multiprocess_mutex_enabled_ = false;
  std::shared_ptr<holoscan::FileFIFOMutex> holoviz_multiprocess_mutex_;
  unsigned long long dropped_frame_count_ = 0;

  static std::remove_pointer_t<viz::KeyCallbackFunction> key_callback_handler;
  static std::remove_pointer_t<viz::UnicodeCharCallbackFunction> unicode_char_callback_handler;
  static std::remove_pointer_t<viz::MouseButtonCallbackFunction> mouse_button_callback_handler;
  static std::remove_pointer_t<viz::ScrollCallbackFunction> scroll_callback_handler;
  static std::remove_pointer_t<viz::CursorPosCallbackFunction> cursor_pos_callback_handler;
  static std::remove_pointer_t<viz::FramebufferSizeCallbackFunction>
      framebuffer_size_callback_handler;
  static std::remove_pointer_t<viz::WindowSizeCallbackFunction> window_size_callback_handler;

  std::array<float, 3> camera_eye_cur_;      //< current camera eye position
  std::array<float, 3> camera_look_at_cur_;  //< current camera look at position
  std::array<float, 3> camera_up_cur_;       //< current camera up vector
};
}  // namespace holoscan::ops

/**
 * Custom YAML parser for InputSpec class
 */
template <>
struct YAML::convert<holoscan::ops::HolovizOp::InputSpec> {
  /**
   * @brief Encodes an InputSpec object to a YAML Node.
   * @param input_spec The InputSpec object to encode.
   * @return YAML Node representation of the InputSpec.
   */
  static Node encode(const holoscan::ops::HolovizOp::InputSpec& input_spec) {
    Node node;
    node["type"] = holoscan::ops::HolovizOp::inputTypeToString(input_spec.type_);
    node["name"] = input_spec.tensor_name_;
    node["opacity"] = std::to_string(input_spec.opacity_);
    node["priority"] = std::to_string(input_spec.priority_);
    switch (input_spec.type_) {
      case holoscan::ops::HolovizOp::InputType::COLOR:
      case holoscan::ops::HolovizOp::InputType::COLOR_LUT:
      case holoscan::ops::HolovizOp::InputType::DEPTH_MAP_COLOR:
        node["image_format"] =
            holoscan::ops::HolovizOp::imageFormatToString(input_spec.image_format_);
        switch (input_spec.image_format_) {
          case holoscan::ops::HolovizOp::ImageFormat::Y8U8Y8V8_422_UNORM:
          case holoscan::ops::HolovizOp::ImageFormat::U8Y8V8Y8_422_UNORM:
          case holoscan::ops::HolovizOp::ImageFormat::Y8_U8V8_2PLANE_420_UNORM:
          case holoscan::ops::HolovizOp::ImageFormat::Y8_U8V8_2PLANE_422_UNORM:
          case holoscan::ops::HolovizOp::ImageFormat::Y8_U8_V8_3PLANE_420_UNORM:
          case holoscan::ops::HolovizOp::ImageFormat::Y8_U8_V8_3PLANE_422_UNORM:
          case holoscan::ops::HolovizOp::ImageFormat::Y16_U16V16_2PLANE_420_UNORM:
          case holoscan::ops::HolovizOp::ImageFormat::Y16_U16V16_2PLANE_422_UNORM:
          case holoscan::ops::HolovizOp::ImageFormat::Y16_U16_V16_3PLANE_420_UNORM:
          case holoscan::ops::HolovizOp::ImageFormat::Y16_U16_V16_3PLANE_422_UNORM:
            node["yuv_model_conversion"] = holoscan::ops::HolovizOp::yuvModelConversionToString(
                input_spec.yuv_model_conversion_);
            node["yuv_range"] = holoscan::ops::HolovizOp::yuvRangeToString(input_spec.yuv_range_);
            node["x_chroma_location"] =
                holoscan::ops::HolovizOp::chromaLocationToString(input_spec.x_chroma_location_);
            node["y_chroma_location"] =
                holoscan::ops::HolovizOp::chromaLocationToString(input_spec.y_chroma_location_);
            break;
          default:
            break;
        }
        break;
      case holoscan::ops::HolovizOp::InputType::POINTS:
      case holoscan::ops::HolovizOp::InputType::LINES:
      case holoscan::ops::HolovizOp::InputType::LINE_STRIP:
      case holoscan::ops::HolovizOp::InputType::TRIANGLES:
      case holoscan::ops::HolovizOp::InputType::CROSSES:
      case holoscan::ops::HolovizOp::InputType::RECTANGLES:
      case holoscan::ops::HolovizOp::InputType::OVALS:
      case holoscan::ops::HolovizOp::InputType::POINTS_3D:
      case holoscan::ops::HolovizOp::InputType::LINES_3D:
      case holoscan::ops::HolovizOp::InputType::LINE_STRIP_3D:
      case holoscan::ops::HolovizOp::InputType::TRIANGLES_3D:
        node["color"] = input_spec.color_;
        node["line_width"] = std::to_string(input_spec.line_width_);
        node["point_size"] = std::to_string(input_spec.point_size_);
        break;
      case holoscan::ops::HolovizOp::InputType::TEXT:
        node["color"] = input_spec.color_;
        node["text"] = input_spec.text_;
        break;
      case holoscan::ops::HolovizOp::InputType::DEPTH_MAP:
        node["depth_map_render_mode"] =
            holoscan::ops::HolovizOp::depthMapRenderModeToString(input_spec.depth_map_render_mode_);
        break;
      default:
        break;
    }
    for (auto&& view : input_spec.views_) {
      node["views"].push_back(view);
    }
    return node;
  }

  /**
   * @brief Decodes a YAML Node to an InputSpec object.
   * @param node The YAML Node to decode.
   * @param input_spec The InputSpec object to populate.
   * @return true if successful, false otherwise.
   */
  static bool decode(const Node& node, holoscan::ops::HolovizOp::InputSpec& input_spec) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("InputSpec: expected a map");
      return false;
    }

    // YAML is using exceptions, catch them
    try {
      const auto maybe_input_type =
          holoscan::ops::HolovizOp::inputTypeFromString(node["type"].as<std::string>());
      if (!maybe_input_type) {
        return false;
      }

      input_spec.tensor_name_ = node["name"].as<std::string>();
      input_spec.type_ = maybe_input_type.value();
      input_spec.opacity_ = node["opacity"].as<float>(input_spec.opacity_);
      input_spec.priority_ = node["priority"].as<int32_t>(input_spec.priority_);
      switch (input_spec.type_) {
        case holoscan::ops::HolovizOp::InputType::COLOR:
        case holoscan::ops::HolovizOp::InputType::COLOR_LUT:
        case holoscan::ops::HolovizOp::InputType::DEPTH_MAP_COLOR:
          if (node["image_format"]) {
            const auto maybe_image_format = holoscan::ops::HolovizOp::imageFormatFromString(
                node["image_format"].as<std::string>());
            if (maybe_image_format) {
              input_spec.image_format_ = maybe_image_format.value();
            }
          }
          if (node["yuv_model_conversion"]) {
            const auto maybe_yuv_model_conversion =
                holoscan::ops::HolovizOp::yuvModelConversionFromString(
                    node["yuv_model_conversion"].as<std::string>());
            if (maybe_yuv_model_conversion) {
              input_spec.yuv_model_conversion_ = maybe_yuv_model_conversion.value();
            }
          }
          if (node["yuv_range"]) {
            const auto maybe_yuv_range =
                holoscan::ops::HolovizOp::yuvRangeFromString(node["yuv_range"].as<std::string>());
            if (maybe_yuv_range) {
              input_spec.yuv_range_ = maybe_yuv_range.value();
            }
          }
          if (node["x_chroma_location"]) {
            const auto maybe_x_chroma_location = holoscan::ops::HolovizOp::chromaLocationFromString(
                node["x_chroma_location"].as<std::string>());
            if (maybe_x_chroma_location) {
              input_spec.x_chroma_location_ = maybe_x_chroma_location.value();
            }
          }
          if (node["chroma_y_location"]) {
            const auto maybe_y_chroma_location = holoscan::ops::HolovizOp::chromaLocationFromString(
                node["y_chroma_location"].as<std::string>());
            if (maybe_y_chroma_location) {
              input_spec.y_chroma_location_ = maybe_y_chroma_location.value();
            }
          }
          break;
        case holoscan::ops::HolovizOp::InputType::LINES:
        case holoscan::ops::HolovizOp::InputType::LINE_STRIP:
        case holoscan::ops::HolovizOp::InputType::TRIANGLES:
        case holoscan::ops::HolovizOp::InputType::CROSSES:
        case holoscan::ops::HolovizOp::InputType::RECTANGLES:
        case holoscan::ops::HolovizOp::InputType::OVALS:
        case holoscan::ops::HolovizOp::InputType::POINTS_3D:
        case holoscan::ops::HolovizOp::InputType::LINES_3D:
        case holoscan::ops::HolovizOp::InputType::LINE_STRIP_3D:
        case holoscan::ops::HolovizOp::InputType::TRIANGLES_3D:
          input_spec.color_ = node["color"].as<std::vector<float>>(input_spec.color_);
          input_spec.line_width_ = node["line_width"].as<float>(input_spec.line_width_);
          input_spec.point_size_ = node["point_size"].as<float>(input_spec.point_size_);
          break;
        case holoscan::ops::HolovizOp::InputType::TEXT:
          input_spec.color_ = node["color"].as<std::vector<float>>(input_spec.color_);
          input_spec.text_ = node["text"].as<std::vector<std::string>>(input_spec.text_);
          break;
        case holoscan::ops::HolovizOp::InputType::DEPTH_MAP:
          if (node["depth_map_render_mode"]) {
            const auto maybe_depth_map_render_mode =
                holoscan::ops::HolovizOp::depthMapRenderModeFromString(
                    node["depth_map_render_mode"].as<std::string>());
            if (maybe_depth_map_render_mode) {
              input_spec.depth_map_render_mode_ = maybe_depth_map_render_mode.value();
            }
          }
          break;
        default:
          break;
      }

      if (node["views"]) {
        input_spec.views_ =
            node["views"].as<std::vector<holoscan::ops::HolovizOp::InputSpec::View>>();
      }

      return true;
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
  }
};

/**
 * Custom YAML parser for InputSpec::View class
 */
template <>
struct YAML::convert<holoscan::ops::HolovizOp::InputSpec::View> {
  /**
   * @brief Encodes an InputSpec::View object to a YAML Node.
   * @param view The InputSpec::View object to encode.
   * @return YAML Node representation of the View.
   */
  static Node encode(const holoscan::ops::HolovizOp::InputSpec::View& view) {
    Node node;
    node["offset_x"] = view.offset_x_;
    node["offset_y"] = view.offset_y_;
    node["width"] = view.width_;
    node["height"] = view.height_;
    if (view.matrix_.has_value()) {
      node["matrix"] = view.matrix_.value();
    }
    return node;
  }

  /**
   * @brief Decodes a YAML Node to an InputSpec::View object.
   * @param node The YAML Node to decode.
   * @param view The InputSpec::View object to populate.
   * @return true if successful, false otherwise.
   */
  static bool decode(const Node& node, holoscan::ops::HolovizOp::InputSpec::View& view) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("InputSpec: expected a map");
      return false;
    }

    // YAML is using exceptions, catch them
    try {
      view.offset_x_ = node["offset_x"].as<float>(view.offset_x_);
      view.offset_y_ = node["offset_y"].as<float>(view.offset_y_);
      view.width_ = node["width"].as<float>(view.width_);
      view.height_ = node["height"].as<float>(view.height_);
      if (node["matrix"]) {
        view.matrix_ = node["matrix"].as<std::array<float, 16>>();
      }

      return true;
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
  }
};

/**
 * Custom YAML parser for ColorSpace enum
 */
template <>
struct YAML::convert<holoscan::ops::HolovizOp::ColorSpace> {
  /**
   * @brief Encodes a ColorSpace enum to a YAML Node.
   * @param color_space The ColorSpace enum to encode.
   * @return YAML Node representation of the ColorSpace.
   */
  static Node encode(const holoscan::ops::HolovizOp::ColorSpace& color_space) {
    Node node;
    node.push_back(holoscan::ops::HolovizOp::colorSpaceToString(color_space));
    return node;
  }

  /**
   * @brief Decodes a YAML Node to a ColorSpace enum.
   * @param node The YAML Node to decode.
   * @param color_space The ColorSpace enum to populate.
   * @return true if successful, false otherwise.
   */
  static bool decode(const Node& node, holoscan::ops::HolovizOp::ColorSpace& color_space) {
    if (!node.IsScalar()) {
      return false;
    }

    // YAML is using exceptions, catch them
    try {
      const auto maybe_color_space = holoscan::ops::HolovizOp::colorSpaceFromString(node.Scalar());
      if (maybe_color_space) {
        color_space = maybe_color_space.value();
        return true;
      }
      return false;
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
  }
};

/**
 * @brief This macro defining a YAML converter which throws for unsupported types.
 *
 * Background: Holoscan supports setting parameters through YAML files. But for some parameters
 * accepted by the receiver operators like callbacks it makes no sense to specify them in YAML
 * files. Therefore use a converter which throws for these types.
 *
 * @tparam TYPE
 */
#define HOLOVIZ_YAML_CONVERTER(TYPE)                                                         \
  template <>                                                                                \
  struct YAML::convert<TYPE> {                                                               \
    /** @brief Throws runtime error as encoding this type is unsupported in YAML. */         \
    static Node encode(TYPE&) { throw std::runtime_error(#TYPE " is unsupported in YAML"); } \
    /** @brief Throws runtime error as decoding this type is unsupported in YAML. */         \
    static bool decode(const Node&, TYPE&) {                                                 \
      throw std::runtime_error(#TYPE " is unsupported in YAML");                             \
    }                                                                                        \
  };

HOLOVIZ_YAML_CONVERTER(holoscan::ops::HolovizOp::KeyCallbackFunction);
HOLOVIZ_YAML_CONVERTER(holoscan::ops::HolovizOp::UnicodeCharCallbackFunction);
HOLOVIZ_YAML_CONVERTER(holoscan::ops::HolovizOp::MouseButtonCallbackFunction);
HOLOVIZ_YAML_CONVERTER(holoscan::ops::HolovizOp::ScrollCallbackFunction);
// don't need CursorPosCallbackFunction since it has the same signature as ScrollCallbackFunction
HOLOVIZ_YAML_CONVERTER(holoscan::ops::HolovizOp::FramebufferSizeCallbackFunction);
// don't need WindowSizeCallbackFunction since it has the same signature as
// FramebufferSizeCallbackFunction
HOLOVIZ_YAML_CONVERTER(holoscan::ops::HolovizOp::LayerCallbackFunction);

#endif /* HOLOSCAN_OPERATORS_HOLOVIZ_HOLOVIZ_HPP */
