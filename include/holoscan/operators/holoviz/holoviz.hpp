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

#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class for data visualization.
 *
 * This high-speed viewer handles compositing, blending, and visualization of RGB or RGBA images,
 * masks, geometric primitives, text and depth maps. The operator can auto detect the format of the
 * input tensors when only the `receivers` parameter list is specified.
 *
 * 1. Displaying Color Images
 *
 * Image data can either be on host or device (GPU). Multiple image formats are supported
 * - R 8 bit unsigned
 * - R 16 bit unsigned
 * - R 16 bit float
 * - R 32 bit unsigned
 * - R 32 bit float
 * - RGB 8 bit unsigned
 * - BGR 8 bit unsigned
 * - RGBA 8 bit unsigned
 * - BGRA 8 bit unsigned
 * - RGBA 16 bit unsigned
 * - RGBA 16 bit float
 * - RGBA 32 bit float
 *
 * When the `type` parameter is set to `color_lut` the final color is looked up using the values
 * from the `color_lut` parameter. For color lookups these image formats are supported
 * - R 8 bit unsigned
 * - R 16 bit unsigned
 * - R 32 bit unsigned
 *
 * 2. Drawing Geometry
 *
 * In all cases, `x` and `y` are normalized coordinates in the range `[0, 1]`. The `x` and `y`
 * correspond to the horizontal and vertical axes of the display, respectively. The origin `(0, 0)`
 * is at the top left of the display. All coordinates should be defined using a single precision
 * float data type. Geometric primitives outside of the visible area are clipped. Coordinate arrays
 * are expected to have the shape `(1, N, C)` where `N` is the coordinate count and `C` is the
 * component count for each coordinate.
 *
 * - Points are defined by a `(x, y)` coordinate pair.
 * - Lines are defined by a set of two `(x, y)` coordinate pairs.
 * - Lines strips are defined by a sequence of `(x, y)` coordinate pairs. The first two coordinates
 * define the first line, each additional coordinate adds a line connecting to the previous
 * coordinate.
 * - Triangles are defined by a set of three `(x, y)` coordinate pairs.
 * - Crosses are defined by `(x, y, size)` tuples. `size` specifies the size of the cross in the `x`
 * direction and is optional, if omitted it's set to `0.05`. The size in the `y` direction is
 * calculated using the aspect ratio of the window to make the crosses square.
 * - Rectangles (bounding boxes) are defined by a pair of 2-tuples defining the upper-left and
 * lower-right coordinates of a box: `(x1, y1), (x2, y2)`.
 * - Ovals are defined by `(x, y, size_x, size_y)` tuples. `size_x` and `size_y` are optional, if
 * omitted they are set to `0.05`.
 * - Texts are defined by `(x, y, size)` tuples. `size` specifies the size of the text in `y`
 * direction and is optional, if omitted it's set to `0.05`. The size in the `x` direction is
 * calculated using the aspect ratio of the window. The index of each coordinate references a text
 * string from the `text` parameter and the index is clamped to the size of the text array. For
 * example, if there is one item set for the `text` parameter, e.g. `text=['my_text']` and three
 * coordinates, then `my_text` is rendered three times. If `text=['first text', 'second text']` and
 * three coordinates are specified, then `first text` is rendered at the first coordinate, `second
 * text` at the second coordinate and then `second text` again at the third coordinate. The `text`
 * string array is fixed and can't be changed after initialization. To hide text which should not be
 * displayed, specify coordinates greater than `(1.0, 1.0)` for the text item, the text is then
 * clipped away.
 *
 * 3. Displaying Depth Maps
 *
 * When `type` is `depth_map` the provided data is interpreted as a rectangular array of depth
 * values. Additionally a 2d array with a color value for each point in the grid can be specified by
 * setting `type` to `depth_map_color`.
 *
 * The type of geometry drawn can be selected by setting `depth_map_render_mode`.
 *
 * Depth maps are rendered in 3D and support camera movement. The camera is controlled using the
 * mouse:
 * - Orbit        (LMB)
 * - Pan          (LMB + CTRL  | MMB)
 * - Dolly        (LMB + SHIFT | RMB | Mouse wheel)
 * - Look Around  (LMB + ALT   | LMB + CTRL + SHIFT)
 * - Zoom         (Mouse wheel + SHIFT)
 *
 * 4. Output
 *
 * By default a window is opened to display the rendering, but the extension can also be run in
 * headless mode with the `headless` parameter.
 *
 * Using a display in exclusive mode is also supported with the `use_exclusive_display` parameter.
 * This reduces the latency by avoiding the desktop compositor.
 *
 * The rendered framebuffer can be output to `render_buffer_output`.
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
    DEPTH_MAP_COLOR  ///< RGBA 2d image, same size as the depth map. One color value for each
                     ///< element of the depth map grid.
                     ///< Supported format:
                     ///<   32-bit unsigned normalized format that has an 8-bit R component in byte
                     ///>   0, an 8-bit G component in byte 1, an 8-bit B component in byte 2,
                     ///<   and an 8-bit A component in byte 3unsigned 8-bit RGBA
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
    InputSpec(const std::string tensor_name, InputType type)
        : tensor_name_(tensor_name), type_(type) {}

    /**
     * @returns true if the input spec is valid
     */
    explicit operator bool() const noexcept { return !tensor_name_.empty(); }

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
  };

 private:
  void enable_conditional_port(const std::string& name);
  bool check_port_enabled(const std::string& name);

  Parameter<std::vector<holoscan::IOSpec*>> receivers_;

  Parameter<holoscan::IOSpec*> render_buffer_input_;
  Parameter<holoscan::IOSpec*> render_buffer_output_;

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
  std::vector<InputSpec> input_spec_;
  CudaStreamHandler cuda_stream_handler_;
  bool render_buffer_input_enabled_;
  bool render_buffer_output_enabled_;
  bool is_first_tick_ = true;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_HOLOVIZ_HOLOVIZ_HPP */
