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

#ifndef HOLOSCAN_OPERATORS_HOLOVIZ_HOLOVIZ_HPP
#define HOLOSCAN_OPERATORS_HOLOVIZ_HOLOVIZ_HPP

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

struct BufferInfo;

namespace holoscan::ops {

/**
 * @brief Operator class for data visualization.
 *
 * This high-speed viewer handles compositing, blending, and visualization of RGB or RGBA images,
 * masks, geometric primitives, text and depth maps. The operator can auto detect the format of the
 * input tensors when only the `receivers` parameter list is specified. Else the input specification
 * can be set at creation time using the `tensors` parameter or at runtime when passing input
 * specifications to the `input_specs` port.
 *
 * 1. Parameters
 *
 *    - **`receivers`**: List of input queues to component accepting `gxf::Tensor` or
 *      `gxf::VideoBuffer`
 *      - type: `std::vector<gxf::Handle<gxf::Receiver>>`
 *    - **`enable_render_buffer_input`**: Enable `render_buffer_input`, (default: `false`)
 *      - type: `bool`
 *    - **`render_buffer_input`**: Input for an empty render buffer, type `gxf::VideoBuffer`
 *      - type: `gxf::Handle<gxf::Receiver>`
 *    - **`enable_render_buffer_output`**: Enable `render_buffer_output`, (default: `false`)
 *      - type: `bool`
 *    - **`render_buffer_output`**: Output for a filled render buffer. If an input render buffer is
 *      specified at `render_buffer_input` it uses that one, otherwise it allocates a new buffer.
 *      - type: `gxf::Handle<gxf::Transmitter>`
 *    - **`enable_camera_pose_output`**: Enable `camera_pose_output`, (default: `false`)
 *      - type: `bool`
 *    - **`camera_pose_output`**: Output the camera pose. The camera parameters are returned in a
 *      4x4 row major projection matrix.
 *      - type: `std::array<float, 16>`
 *    - **`tensors`**: List of input tensor specifications (default: `[]`)
 *      - type: `std::vector<InputSpec>`
 *        - **`name`**: name of the tensor containing the input data to display
 *          - type: `std::string`
 *        - **`type`**: input type (default `"unknown"`)
 *          - type: `std::string`
 *          - possible values:
 *            **`unknown`**: unknown type, the operator tries to guess the type by inspecting the
 *            tensor
 *            **`color`**: RGB or RGBA color 2d image
 *            **`color_lut`**: single channel 2d image, color is looked up
 *            **`points`**: point primitives, one coordinate (x, y) per primitive
 *            **`lines`**: line primitives, two coordinates (x0, y0) and (x1, y1) per primitive
 *            **`line_strip`**: line strip primitive, a line primitive i is defined by each
 *            coordinate (xi, yi) and the following (xi+1, yi+1)
 *            **`triangles`**: triangle primitive, three coordinates (x0, y0), (x1, y1) and (x2, y2)
 *            per primitive
 *            **`crosses`**: cross primitive, a cross is defined by the center coordinate and the
 *            size (xi, yi, si)
 *            **`rectangles`**: axis aligned rectangle primitive, each rectangle is defined by two
 *            coordinates (xi, yi) and (xi+1, yi+1)
 *            **`ovals`**: oval primitive, an oval primitive is defined by the center coordinate and
 *            the axis sizes (xi, yi, sxi, syi)
 *            **`text`**: text is defined by the top left coordinate and the size (x, y, s) per
 *            string, text strings are defined by InputSpec member **`text`**
 *            **`depth_map`**: single channel 2d array where each element represents a depth value.
 *            The data is rendered as a 3d object using points, lines or triangles. The color for
 *            the elements can be specified through `depth_map_color`. Supported format: 8-bit
 *            unsigned normalized format that has a single 8-bit depth component
 *            **`depth_map_color`**: RGBA 2d image, same size as the depth map. One color value for
 *            each element of the depth map grid. Supported format: 32-bit unsigned normalized
 *            format that has an 8-bit R component in byte 0, an 8-bit G component in byte 1, an
 *            8-bit B component in byte 2, and an 8-bit A component in byte 3.
 *        - **`opacity`**: layer opacity, 1.0 is fully opaque, 0.0 is fully transparent (default:
 *          `1.0`)
 *          - type: `float`
 *        - **`priority`**: layer priority, determines the render order, layers with higher priority
 *            values are rendered on top of layers with lower priority values (default: `0`)
 *          - type: `int32_t`
 *        - **`color`**: RGBA color of rendered geometry (default: `[1.f, 1.f, 1.f, 1.f]`)
 *          - type: `std::vector<float>`
 *        - **`line_width`**: line width for geometry made of lines (default: `1.0`)
 *          - type: `float`
 *        - **`point_size`**: point size for geometry made of points (default: `1.0`)
 *          - type: `float`
 *        - **`text`**: array of text strings, used when `type` is text. (default: `[]`)
 *          - type: `std::vector<std::string>`
 *        - **`depth_map_render_mode`**: depth map render mode (default: `points`)
 *          -type `std::string`
 *          - possible values:
 *            **`points`**: render as points
 *            **`lines`**: render as lines
 *            **`triangles`**: render as triangles
 *    - **`color_lut`**: Color lookup table for tensors of type 'color_lut', vector of four float
 *      RGBA values
 *      - type: `std::vector<std::vector<float>>`
 *    - **`window_title`**: Title on window canvas (default: `Holoviz`)
 *      - type: `std::string`
 *    - **`display_name`**: In exclusive mode, name of display to use as shown with xrandr (default:
 *      `DP-0`)
 *      - type: `std::string`
 *    - **`width`**: Window width or display resolution width if in exclusive or fullscreen mode
 *      (default: `1920`)
 *      - type: `uint32_t`
 *    - **`height`**: Window height or display resolution height if in exclusive or fullscreen mode
 *      (default: `1080`)
 *      - type: `uint32_t`
 *    - **`framerate`**: Display framerate if in exclusive mode (default: `60`)
 *      - type: `uint32_t`
 *    - **`use_exclusive_display`**: Enable exclusive display (default: `false`)
 *      - type: `bool`
 *    - **`fullscreen`**: Enable fullscreen window (default: `false`)
 *      - type: `bool`
 *    - **`headless`**: Enable headless mode. No window is opened, the render buffer is output to
 *      `render_buffer_output`. (default: `false`)
 *      - type: `bool`
 *    - **`window_close_scheduling_term`**: BooleanSchedulingTerm to stop the codelet from ticking
 *      when the window is closed
 *      - type: `gxf::Handle<gxf::BooleanSchedulingTerm>`
 *    - **`allocator`**: Allocator used to allocate memory for `render_buffer_output`
 *      - type: `gxf::Handle<gxf::Allocator>`
 *    - **`font_path`**: File path for the font used for rendering text.
 *      - type: `std::string`
 *    - **`cuda_stream_pool`**: Instance of gxf::CudaStreamPool
 *      - type: `gxf::Handle<gxf::CudaStreamPool>`
 *
 * 2. Displaying Color Images
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
 * 3. Drawing Geometry
 *
 *    In all cases, `x` and `y` are normalized coordinates in the range `[0, 1]`. The `x` and `y`
 *    correspond to the horizontal and vertical axes of the display, respectively. The origin `(0,
 *    0)` is at the top left of the display. All coordinates should be defined using a single
 *    precision float data type. Geometric primitives outside of the visible area are clipped.
 *    Coordinate arrays are expected to have the shape `(1, N, C)` where `N` is the coordinate count
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
 *      `text=['my_text']` and three coordinates, then `my_text` is rendered three times. If
 *      `text=['first text', 'second text']` and three coordinates are specified, then `first text`
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
 * 4. Displaying Depth Maps
 *
 *    When `type` is `depth_map` the provided data is interpreted as a rectangular array of depth
 *    values. Additionally a 2d array with a color value for each point in the grid can be specified
 *    by setting `type` to `depth_map_color`.
 *
 *    The type of geometry drawn can be selected by setting `depth_map_render_mode`.
 *
 *    Depth maps are rendered in 3D and support camera movement. The camera is controlled using the
 *    mouse:
 *    - Orbit        (LMB)
 *    - Pan          (LMB + CTRL  | MMB)
 *    - Dolly        (LMB + SHIFT | RMB | Mouse wheel)
 *    - Look Around  (LMB + ALT   | LMB + CTRL + SHIFT)
 *    - Zoom         (Mouse wheel + SHIFT)
 *
 * 5. Output
 *
 *    By default a window is opened to display the rendering, but the extension can also be run in
 *    headless mode with the `headless` parameter.
 *
 *    Using a display in exclusive mode is also supported with the `use_exclusive_display`
 *    parameter. This reduces the latency by avoiding the desktop compositor.
 *
 *    The rendered framebuffer can be output to `render_buffer_output`.
 *
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
    COLOR,       ///< RGB or RGBA color 2d image
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
   * Depth map render mode.
   */
  enum class DepthMapRenderMode {
    POINTS,    ///< render points
    LINES,     ///< render lines
    TRIANGLES  ///< render triangles
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
     * @returns an InputSpec from the YAML form output by description()
     */
    explicit InputSpec(const std::string& yaml_description);

    /**
     * @returns true if the input spec is valid
     */
    explicit operator bool() const noexcept { return !tensor_name_.empty(); }

    /**
     * @returns a YAML string representation of the InputSpec
     */
    std::string description() const;

    std::string tensor_name_;              ///< name of the tensor containing the input data
    InputType type_ = InputType::UNKNOWN;  ///< input type
    float opacity_ = 1.f;  ///< layer opacity, 1.0 is fully opaque, 0.0 is fully transparent
    int32_t priority_ =
        0;  ///< layer priority, determines the render order, layers with higher priority values are
            ///< rendered on top of layers with lower priority values
    std::vector<float> color_{1.f, 1.f, 1.f, 1.f};  ///< color of rendered geometry
    float line_width_ = 1.f;                        ///< line width for geometry made of lines
    float point_size_ = 1.f;                        ///< point size for geometry made of points
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
      float offset_x_ = 0.f,
            offset_y_ = 0.f;  ///< offset of top-left corner of the view. Top left coordinate of
                              /// the window area is (0, 0) bottom right
                              /// coordinate is (1, 1).
      float width_ = 1.f,
            height_ = 1.f;  ///< width and height of the view in normalized range. 1.0 is full size.
      std::optional<std::array<float, 16>>
          matrix_;  ///< row major 4x4 transform matrix (optional, can be nullptr)
    };
    std::vector<View> views_;
  };

 private:
  void enable_conditional_port(const std::string& name);
  bool check_port_enabled(const std::string& name);
  void set_input_spec(const InputSpec& input_spec);
  void set_input_spec_geometry(const InputSpec& input_spec);
  void read_frame_buffer(InputContext& op_input, OutputContext& op_output,
                         ExecutionContext& context);
  void render_depth_map(InputSpec* const input_spec_depth_map,
                        const BufferInfo& buffer_info_depth_map,
                        InputSpec* const input_spec_depth_map_color,
                        const BufferInfo& buffer_info_depth_map_color);

  Parameter<std::vector<holoscan::IOSpec*>> receivers_;

  Parameter<holoscan::IOSpec*> render_buffer_input_;
  Parameter<holoscan::IOSpec*> render_buffer_output_;
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
  Parameter<std::shared_ptr<BooleanCondition>> window_close_scheduling_term_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::string> font_path_;

  // internal state
  std::vector<float> lut_;
  std::vector<InputSpec> initial_input_spec_;
  CudaStreamHandler cuda_stream_handler_;
  bool render_buffer_input_enabled_;
  bool render_buffer_output_enabled_;
  bool camera_pose_output_enabled_;
  bool is_first_tick_ = true;
};
}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_HOLOVIZ_HOLOVIZ_HPP */
