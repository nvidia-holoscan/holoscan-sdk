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

#include "holoscan/operators/holoviz/holoviz.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

namespace {

/// table to convert input type to string
static const std::array<std::pair<holoscan::ops::HolovizOp::InputType, std::string>, 11>
    kInputTypeToStr{{{holoscan::ops::HolovizOp::InputType::UNKNOWN, "unknown"},
                     {holoscan::ops::HolovizOp::InputType::COLOR, "color"},
                     {holoscan::ops::HolovizOp::InputType::COLOR_LUT, "color_lut"},
                     {holoscan::ops::HolovizOp::InputType::POINTS, "points"},
                     {holoscan::ops::HolovizOp::InputType::LINES, "lines"},
                     {holoscan::ops::HolovizOp::InputType::LINE_STRIP, "line_strip"},
                     {holoscan::ops::HolovizOp::InputType::TRIANGLES, "triangles"},
                     {holoscan::ops::HolovizOp::InputType::CROSSES, "crosses"},
                     {holoscan::ops::HolovizOp::InputType::RECTANGLES, "rectangles"},
                     {holoscan::ops::HolovizOp::InputType::OVALS, "ovals"},
                     {holoscan::ops::HolovizOp::InputType::TEXT, "text"}}};

/**
 * Convert a string to a input type enum
 *
 * @param string input type string
 * @return input type enum
 */
static nvidia::gxf::Expected<holoscan::ops::HolovizOp::InputType> inputTypeFromString(
    const std::string& string) {
  const auto it = std::find_if(std::cbegin(kInputTypeToStr),
                               std::cend(kInputTypeToStr),
                               [&string](const auto& v) { return v.second == string; });
  if (it != std::cend(kInputTypeToStr)) { return it->first; }

  GXF_LOG_ERROR("Unsupported tensor type '%s'", string.c_str());
  return nvidia::gxf::Unexpected{GXF_FAILURE};
}

/**
 * Convert a input type enum to a string
 *
 * @param input_type input type enum
 * @return input type string
 */
static std::string inputTypeToString(holoscan::ops::HolovizOp::InputType input_type) {
  const auto it = std::find_if(std::cbegin(kInputTypeToStr),
                               std::cend(kInputTypeToStr),
                               [&input_type](const auto& v) { return v.first == input_type; });
  if (it != std::cend(kInputTypeToStr)) { return it->second; }

  return "invalid";
}

}  // namespace

/**
 * Custom YAML parser for InputSpec class
 */
template <>
struct YAML::convert<holoscan::ops::HolovizOp::InputSpec> {
  static Node encode(const holoscan::ops::HolovizOp::InputSpec& input_spec) {
    Node node;
    node["type"] = inputTypeToString(input_spec.type_);
    node["name"] = input_spec.tensor_name_;
    node["opacity"] = std::to_string(input_spec.opacity_);
    node["priority"] = std::to_string(input_spec.priority_);
    node["color"] = input_spec.color_;
    node["line_width"] = std::to_string(input_spec.line_width_);
    node["point_size"] = std::to_string(input_spec.point_size_);
    node["text"] = input_spec.text_;
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::HolovizOp::InputSpec& input_spec) {
    if (!node.IsMap()) {
      GXF_LOG_ERROR("InputSpec: expected a map");
      return false;
    }

    // YAML is using exceptions, catch them
    try {
      const auto maybe_input_type = inputTypeFromString(node["type"].as<std::string>());
      if (!maybe_input_type) { return false; }

      input_spec.tensor_name_ = node["name"].as<std::string>();
      input_spec.type_ = maybe_input_type.value();
      input_spec.opacity_ = node["opacity"].as<float>(input_spec.opacity_);
      input_spec.priority_ = node["priority"].as<int32_t>(input_spec.priority_);
      input_spec.color_ = node["color"].as<std::vector<float>>(input_spec.color_);
      input_spec.line_width_ = node["line_width"].as<float>(input_spec.line_width_);
      input_spec.point_size_ = node["point_size"].as<float>(input_spec.point_size_);
      input_spec.text_ = node["text"].as<std::vector<std::string>>(input_spec.text_);

      return true;
    } catch (const std::exception& e) {
      GXF_LOG_ERROR(e.what());
      return false;
    }
  }
};

namespace holoscan::ops {

void HolovizOp::setup(OperatorSpec& spec) {
  constexpr uint32_t DEFAULT_WIDTH = 1920;
  constexpr uint32_t DEFAULT_HEIGHT = 1080;
  constexpr uint32_t DEFAULT_FRAMERATE = 60;
  static const std::string DEFAULT_WINDOW_TITLE("Holoviz");
  static const std::string DEFAULT_DISPLAY_NAME("DP-0");
  constexpr bool DEFAULT_EXCLUSIVE_DISPLAY = false;
  constexpr bool DEFAULT_FULLSCREEN = false;
  constexpr bool DEFAULT_HEADLESS = false;

  spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});

  auto& render_buffer_input =
      spec.input<gxf::Entity>("render_buffer_input").condition(ConditionType::kNone);
  spec.param(render_buffer_input_,
             "render_buffer_input",
             "RenderBufferInput",
             "Input for an empty render buffer.",
             &render_buffer_input);
  auto& render_buffer_output =
      spec.output<gxf::Entity>("render_buffer_output").condition(ConditionType::kNone);
  spec.param(render_buffer_output_,
             "render_buffer_output",
             "RenderBufferOutput",
             "Output for a filled render buffer. If an input render buffer is specified it is "
             "using that one, else it allocates a new buffer.",
             &render_buffer_output);

  spec.param(
      tensors_,
      "tensors",
      "Input Tensors",
      "List of input tensors. 'name' is required, 'type' is optional (unknown, color, color_lut, "
      "points, lines, line_strip, triangles, crosses, rectangles, ovals, text).",
      std::vector<InputSpec>());

  spec.param(color_lut_,
             "color_lut",
             "ColorLUT",
             "Color lookup table for tensors of type 'color_lut'",
             {});

  spec.param(window_title_,
             "window_title",
             "Window title",
             "Title on window canvas",
             DEFAULT_WINDOW_TITLE);
  spec.param(display_name_,
             "display_name",
             "Display name",
             "In exclusive mode, name of display to use as shown with xrandr.",
             DEFAULT_DISPLAY_NAME);
  spec.param(width_,
             "width",
             "Width",
             "Window width or display resolution width if in exclusive or fullscreen mode.",
             DEFAULT_WIDTH);
  spec.param(height_,
             "height",
             "Height",
             "Window height or display resolution height if in exclusive or fullscreen mode.",
             DEFAULT_HEIGHT);
  spec.param(framerate_,
             "framerate",
             "Framerate",
             "Display framerate if in exclusive mode.",
             DEFAULT_FRAMERATE);
  spec.param(use_exclusive_display_,
             "use_exclusive_display",
             "Use exclusive display",
             "Enable exclusive display",
             DEFAULT_EXCLUSIVE_DISPLAY);
  spec.param(fullscreen_,
             "fullscreen",
             "Use fullscreen window",
             "Enable fullscreen window",
             DEFAULT_FULLSCREEN);
  spec.param(headless_,
             "headless",
             "Headless",
             "Enable headless mode. No window is opened, the render buffer is output to "
             "‘render_buffer_output’.",
             DEFAULT_HEADLESS);
  spec.param(window_close_scheduling_term_,
             "window_close_scheduling_term",
             "WindowCloseSchedulingTerm",
             "BooleanSchedulingTerm to stop the codelet from ticking when the window is closed.");

  spec.param(
      allocator_, "allocator", "Allocator", "Allocator used to allocate render buffer output.");
}

void HolovizOp::enable_conditional_port(const std::string& port_name) {
  const std::string enable_port_name = std::string("enable_") + port_name;

  // Check if the boolean argument with the name "enable_(port_name)" is present.
  auto enable_port =
      std::find_if(args().begin(), args().end(), [&enable_port_name](const auto& arg) {
        return (arg.name() == enable_port_name);
      });

  // If the argument is not present or it is set to false, we unset(nullify) the port parameter.
  const bool disable_port =
      (enable_port == args().end()) ||
      (enable_port->has_value() && (std::any_cast<bool>(enable_port->value()) == false));

  if (enable_port != args().end()) {
    // If the argument is present, we just remove it from the arguments.
    args().erase(enable_port);
  }

  if (disable_port) { add_arg(Arg(port_name) = static_cast<holoscan::IOSpec*>(nullptr)); }
}

void HolovizOp::initialize() {
  register_converter<std::vector<InputSpec>>();

  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();
  auto window_close_scheduling_term =
      frag->make_condition<holoscan::BooleanCondition>("window_close_scheduling_term");
  add_arg(Arg("window_close_scheduling_term") = window_close_scheduling_term);

  // Conditional inputs and outputs are enabled using a boolean argument
  enable_conditional_port("render_buffer_input");
  enable_conditional_port("render_buffer_output");

  GXFOperator::initialize();
}

}  // namespace holoscan::ops
