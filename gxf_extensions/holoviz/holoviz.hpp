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

#ifndef GXF_EXTENSIONS_HOLOVIZ_HOLOVIZ_HPP
#define GXF_EXTENSIONS_HOLOVIZ_HOLOVIZ_HPP

#include <array>
#include <list>
#include <string>
#include <vector>

#include "gxf/core/handle.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/scheduling_terms.hpp"

namespace nvidia::holoscan {

class Holoviz : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

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

    std::string tensor_name_;  ///< name of the tensor containing the input data
    InputType type_;           ///< input type
    float opacity_ = 1.f;      ///< layer opacity, 1.0 is fully opaque, 0.0 is fully transparent
    int32_t priority_ =
        0;  ///< layer priority, determines the render order, layers with higher priority values are
            ///< rendered on top of layers with lower priority values
    std::vector<float> color_{1.f, 1.f, 1.f, 1.f};  ///< color of rendered geometry
    float line_width_ = 1.f;                        ///< line width for geometry made of lines
    float point_size_ = 1.f;                        ///< point size for geometry made of points
    std::vector<std::string> text_;  ///< array of text strings, used when type_ is text.
  };

 private:
  // parameters
  gxf::Parameter<std::vector<gxf::Handle<gxf::Receiver>>> receivers_;

  gxf::Parameter<gxf::Handle<gxf::Receiver>> render_buffer_input_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> render_buffer_output_;

  gxf::Parameter<std::vector<InputSpec>> tensors_;

  gxf::Parameter<std::vector<std::vector<float>>> color_lut_;

  gxf::Parameter<std::string> window_title_;
  gxf::Parameter<std::string> display_name_;
  gxf::Parameter<uint32_t> width_;
  gxf::Parameter<uint32_t> height_;
  gxf::Parameter<uint32_t> framerate_;
  gxf::Parameter<bool> use_exclusive_display_;
  gxf::Parameter<bool> fullscreen_;
  gxf::Parameter<bool> headless_;
  gxf::Parameter<gxf::Handle<gxf::BooleanSchedulingTerm>> window_close_scheduling_term_;

  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;

  // internal state
  std::vector<float> lut_;
  std::vector<InputSpec> input_spec_;
};

}  // namespace nvidia::holoscan

#endif /* GXF_EXTENSIONS_HOLOVIZ_HOLOVIZ_HPP */
