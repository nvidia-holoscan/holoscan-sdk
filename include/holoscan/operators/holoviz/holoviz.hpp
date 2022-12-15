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

#ifndef HOLOSCAN_OPERATORS_HOLOVIZ_HOLOVIZ_HPP
#define HOLOSCAN_OPERATORS_HOLOVIZ_HOLOVIZ_HPP

#include <memory>
#include <string>
#include <vector>

#include "../../core/gxf/gxf_operator.hpp"

namespace holoscan::ops {

class HolovizOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(HolovizOp, holoscan::ops::GXFOperator)

  HolovizOp() = default;

  const char* gxf_typename() const override { return "nvidia::holoscan::Holoviz"; }

  void setup(OperatorSpec& spec) override;

  void initialize() override;

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
    TEXT    ///< text is defined by the top left coordinate and the size (x, y, s) per string, text
            ///< strings are define by InputSpec::text_
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
    std::vector<std::string> text_;  ///< array of text strings, used when type_ is text.
  };

 private:
  void enable_conditional_port(const std::string &name);

  Parameter<std::vector<holoscan::IOSpec*>> receivers_;

  Parameter<holoscan::IOSpec*> render_buffer_input_;
  Parameter<holoscan::IOSpec*> render_buffer_output_;

  Parameter<std::vector<InputSpec>> tensors_;

  Parameter<std::vector<std::vector<float>>> color_lut_;

  Parameter<std::string> window_title_;
  Parameter<std::string> display_name_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<uint32_t> framerate_;
  Parameter<bool> use_exclusive_display_;
  Parameter<bool> fullscreen_;
  Parameter<bool> headless_;
  Parameter<std::shared_ptr<Condition>> window_close_scheduling_term_;

  Parameter<std::shared_ptr<Allocator>> allocator_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_HOLOVIZ_HOLOVIZ_HPP */
