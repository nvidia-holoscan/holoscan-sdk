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

#include "holoscan/operators/holoviz/holoviz.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/codecs.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/operators/holoviz/codecs.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

#include "gxf/multimedia/video.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/tensor.hpp"
#include "holoviz/holoviz.hpp"  // holoviz module

#define CUDA_TRY(stmt)                                                                          \
  ({                                                                                            \
    cudaError_t _holoscan_cuda_err = stmt;                                                      \
    if (cudaSuccess != _holoscan_cuda_err) {                                                    \
      HOLOSCAN_LOG_ERROR("CUDA Runtime call {} in line {} of file {} failed with '{}' ({}).\n", \
                         #stmt,                                                                 \
                         __LINE__,                                                              \
                         __FILE__,                                                              \
                         cudaGetErrorString(_holoscan_cuda_err),                                \
                         _holoscan_cuda_err);                                                   \
    }                                                                                           \
    _holoscan_cuda_err;                                                                         \
  })

namespace viz = holoscan::viz;

/// Buffer information, can be initialized either with a tensor or a video buffer
struct BufferInfo {
  /**
   * Initialize with tensor
   *
   * @returns error code
   */
  gxf_result_t init(const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor) {
    rank = tensor->rank();
    shape = tensor->shape();
    element_type = tensor->element_type();
    name = tensor.name();
    buffer_ptr = tensor->pointer();
    storage_type = tensor->storage_type();
    bytes_size = tensor->bytes_size();
    for (uint32_t i = 0; i < rank; ++i) { stride[i] = tensor->stride(i); }

    return GXF_SUCCESS;
  }

  /**
   * Initialize with video buffer
   *
   * @returns error code
   */
  gxf_result_t init(const nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>& video) {
    // NOTE: VideoBuffer::moveToTensor() converts VideoBuffer instance to the Tensor instance
    // with an unexpected shape:  [width, height] or [width, height, num_planes].
    // And, if we use moveToTensor() to convert VideoBuffer to Tensor, we may lose the original
    // video buffer when the VideoBuffer instance is used in other places. For that reason, we
    // directly access internal data of VideoBuffer instance to access Tensor data.
    const auto& buffer_info = video->video_frame_info();

    int32_t channels;
    switch (buffer_info.color_format) {
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        channels = 1;
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY16:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned16;
        channels = 1;
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY32:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned32;
        channels = 1;
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        channels = 3;
        break;
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        channels = 4;
        break;
      default:
        HOLOSCAN_LOG_ERROR("Unsupported input format: {}\n",
                           static_cast<int64_t>(buffer_info.color_format));
        return GXF_FAILURE;
    }

    rank = 3;
    shape = nvidia::gxf::Shape{static_cast<int32_t>(buffer_info.height),
                               static_cast<int32_t>(buffer_info.width),
                               channels};
    name = video.name();
    buffer_ptr = video->pointer();
    storage_type = video->storage_type();
    bytes_size = video->size();
    stride[0] = buffer_info.color_planes[0].stride;
    stride[1] = channels;
    stride[2] = PrimitiveTypeSize(element_type);

    return GXF_SUCCESS;
  }

  uint32_t rank;
  nvidia::gxf::Shape shape;
  nvidia::gxf::PrimitiveType element_type;
  std::string name;
  const nvidia::byte* buffer_ptr;
  nvidia::gxf::MemoryStorageType storage_type;
  uint64_t bytes_size;
  nvidia::gxf::Tensor::stride_array_t stride;
};

namespace {

/**
 * Get the Holoviz image format for a given buffer.
 *
 * @param buffer_info buffer info
 * @return Holoviz image format
 */
nvidia::gxf::Expected<viz::ImageFormat> getImageFormatFromTensor(const BufferInfo& buffer_info) {
  if (buffer_info.rank != 3) {
    HOLOSCAN_LOG_ERROR("Invalid tensor rank count, expected 3, got {}", buffer_info.rank);
    return nvidia::gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }

  struct Format {
    nvidia::gxf::PrimitiveType type_;
    int32_t channels_;
    viz::ImageFormat format_;
  };
  constexpr Format kGFXToHolovizFormats[] = {
      {nvidia::gxf::PrimitiveType::kUnsigned8, 1, viz::ImageFormat::R8_UINT},
      {nvidia::gxf::PrimitiveType::kUnsigned16, 1, viz::ImageFormat::R16_UINT},
      {nvidia::gxf::PrimitiveType::kUnsigned32, 1, viz::ImageFormat::R32_UINT},
      {nvidia::gxf::PrimitiveType::kFloat32, 1, viz::ImageFormat::R32_SFLOAT},
      {nvidia::gxf::PrimitiveType::kUnsigned8, 3, viz::ImageFormat::R8G8B8_UNORM},
      {nvidia::gxf::PrimitiveType::kUnsigned8, 4, viz::ImageFormat::R8G8B8A8_UNORM},
      {nvidia::gxf::PrimitiveType::kUnsigned16, 4, viz::ImageFormat::R16G16B16A16_UNORM},
      {nvidia::gxf::PrimitiveType::kFloat32, 4, viz::ImageFormat::R32G32B32A32_SFLOAT}};

  viz::ImageFormat image_format = static_cast<viz::ImageFormat>(-1);
  for (auto&& format : kGFXToHolovizFormats) {
    if ((format.type_ == buffer_info.element_type) &&
        (format.channels_ == buffer_info.shape.dimension(2))) {
      image_format = format.format_;
      break;
    }
  }
  if (image_format == static_cast<viz::ImageFormat>(-1)) {
    HOLOSCAN_LOG_ERROR("Element type {} and channel count {} not supported",
                       static_cast<int>(buffer_info.element_type),
                       buffer_info.shape.dimension(3));
    return nvidia::gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }

  return image_format;
}

/// table to convert input type to string
static const std::array<std::pair<holoscan::ops::HolovizOp::InputType, std::string>, 17>
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
                     {holoscan::ops::HolovizOp::InputType::TEXT, "text"},
                     {holoscan::ops::HolovizOp::InputType::DEPTH_MAP, "depth_map"},
                     {holoscan::ops::HolovizOp::InputType::DEPTH_MAP_COLOR, "depth_map_color"},
                     {holoscan::ops::HolovizOp::InputType::POINTS_3D, "points_3d"},
                     {holoscan::ops::HolovizOp::InputType::LINES_3D, "lines_3d"},
                     {holoscan::ops::HolovizOp::InputType::LINE_STRIP_3D, "line_strip_3d"},
                     {holoscan::ops::HolovizOp::InputType::TRIANGLES_3D, "triangles_3d"}}};

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

  HOLOSCAN_LOG_ERROR("Unsupported input type '{}'", string);
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

/// table to convert depth map render mode to string
static const std::array<std::pair<holoscan::ops::HolovizOp::DepthMapRenderMode, std::string>, 3>
    kDepthMapRenderModeToStr{
        {{holoscan::ops::HolovizOp::DepthMapRenderMode::POINTS, "points"},
         {holoscan::ops::HolovizOp::DepthMapRenderMode::LINES, "lines"},
         {holoscan::ops::HolovizOp::DepthMapRenderMode::TRIANGLES, "triangles"}}};

/**
 * Convert a string to a depth map render mode enum
 *
 * @param string depth map render mode string
 * @return depth map render mode enum
 */
static nvidia::gxf::Expected<holoscan::ops::HolovizOp::DepthMapRenderMode>
depthMapRenderModeFromString(const std::string& string) {
  const auto it = std::find_if(std::cbegin(kDepthMapRenderModeToStr),
                               std::cend(kDepthMapRenderModeToStr),
                               [&string](const auto& v) { return v.second == string; });
  if (it != std::cend(kDepthMapRenderModeToStr)) { return it->first; }

  HOLOSCAN_LOG_ERROR("Unsupported depth map render mode '{}'", string);
  return nvidia::gxf::Unexpected{GXF_FAILURE};
}

/**
 * Convert a depth map render mode enum to a string
 *
 * @param depth_map_render_mode depth map render mode enum
 * @return depth map render mode string
 */
static std::string depthMapRenderModeToString(
    holoscan::ops::HolovizOp::DepthMapRenderMode depth_map_render_mode) {
  const auto it = std::find_if(
      std::cbegin(kDepthMapRenderModeToStr),
      std::cend(kDepthMapRenderModeToStr),
      [&depth_map_render_mode](const auto& v) { return v.first == depth_map_render_mode; });
  if (it != std::cend(kDepthMapRenderModeToStr)) { return it->second; }

  return "invalid";
}

/**
 * Try to detect the input type enum for given buffer properties.
 *
 * @param buffer_info buffer info
 * @param has_lut true if the user specified a LUT
 *
 *  @return input type enum
 */
nvidia::gxf::Expected<holoscan::ops::HolovizOp::InputType> detectInputType(
    const BufferInfo& buffer_info, bool has_lut) {
  // auto detect type
  if (buffer_info.rank == 3) {
    if ((buffer_info.shape.dimension(2) == 2) && (buffer_info.shape.dimension(0) == 1) &&
        (buffer_info.element_type == nvidia::gxf::PrimitiveType::kFloat32)) {
      // array of 2D coordinates, draw crosses
      return holoscan::ops::HolovizOp::InputType::CROSSES;
    } else if ((buffer_info.shape.dimension(2) == 1) && has_lut) {
      // color image with lookup table
      return holoscan::ops::HolovizOp::InputType::COLOR_LUT;
    } else if ((buffer_info.shape.dimension(2) == 3) || (buffer_info.shape.dimension(2) == 4)) {
      // color image (RGB or RGBA)
      return holoscan::ops::HolovizOp::InputType::COLOR;
    } else {
      HOLOSCAN_LOG_ERROR("Can't auto detect type of input '{}'", buffer_info.name);
    }
  }
  return nvidia::gxf::Unexpected{GXF_FAILURE};
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
    switch (input_spec.type_) {
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
            depthMapRenderModeToString(input_spec.depth_map_render_mode_);
        break;
      default:
        break;
    }
    for (auto&& view : input_spec.views_) { node["views"].push_back(view); }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::HolovizOp::InputSpec& input_spec) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("InputSpec: expected a map");
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
      switch (input_spec.type_) {
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
                depthMapRenderModeFromString(node["depth_map_render_mode"].as<std::string>());
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
  static Node encode(const holoscan::ops::HolovizOp::InputSpec::View& view) {
    Node node;
    node["offset_x"] = view.offset_x_;
    node["offset_y"] = view.offset_y_;
    node["width"] = view.width_;
    node["height"] = view.height_;
    if (view.matrix_.has_value()) { node["matrix"] = view.matrix_.value(); }
    return node;
  }

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
      if (node["matrix"]) { view.matrix_ = node["matrix"].as<std::array<float, 16>>(); }

      return true;
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
  }
};

namespace holoscan::ops {

HolovizOp::InputSpec::InputSpec(const std::string& tensor_name, const std::string& type_str)
    : tensor_name_(tensor_name) {
  const auto maybe_type = inputTypeFromString(type_str);
  if (!maybe_type) {
    throw std::runtime_error("");
    return;
  }
  type_ = maybe_type.value();
}

void HolovizOp::setup(OperatorSpec& spec) {
  constexpr uint32_t DEFAULT_WIDTH = 1920;
  constexpr uint32_t DEFAULT_HEIGHT = 1080;
  constexpr float DEFAULT_FRAMERATE = 60.f;
  static const std::string DEFAULT_WINDOW_TITLE("Holoviz");
  static const std::string DEFAULT_DISPLAY_NAME("DP-0");
  constexpr bool DEFAULT_EXCLUSIVE_DISPLAY = false;
  constexpr bool DEFAULT_FULLSCREEN = false;
  constexpr bool DEFAULT_HEADLESS = false;

  spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});

  spec.input<std::any>("input_specs").condition(ConditionType::kNone);

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

  auto& camera_pose_output =
      spec.output<std::array<float, 16>>("camera_pose_output").condition(ConditionType::kNone);
  spec.param(camera_pose_output_,
             "camera_pose_output",
             "CameraPoseOutput",
             "Output the camera pose. The camera parameters are returned in a 4x4 row major "
             "projection matrix.",
             &camera_pose_output);

  spec.param(
      tensors_,
      "tensors",
      "Input Tensors",
      "List of input tensors. 'name' is required, 'type' is optional (unknown, color, color_lut, "
      "points, lines, line_strip, triangles, crosses, rectangles, ovals, text, points_3d, "
      "lines_3d, line_strip_3d, triangles_3d).",
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
             "Display framerate in Hz if in exclusive mode.",
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

  spec.param(font_path_,
             "font_path",
             "FontPath",
             "File path for the font used for rendering text",
             std::string());

  cuda_stream_handler_.defineParams(spec);
}

bool HolovizOp::check_port_enabled(const std::string& port_name) {
  const std::string enable_port_name = std::string("enable_") + port_name;

  // Check if the boolean argument with the name "enable_(port_name)" is present.
  auto enable_port =
      std::find_if(args().begin(), args().end(), [&enable_port_name](const auto& arg) {
        return (arg.name() == enable_port_name);
      });
  const bool disable_port =
      (enable_port == args().end()) ||
      (enable_port->has_value() && (std::any_cast<bool>(enable_port->value()) == false));
  return !disable_port;
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

void HolovizOp::read_frame_buffer(InputContext& op_input, OutputContext& op_output,
                                  ExecutionContext& context) {
  auto entity = nvidia::gxf::Entity::New(context.context());
  if (!entity) {
    throw std::runtime_error("Failed to allocate message for the render buffer output.");
  }

  auto video_buffer = entity.value().add<nvidia::gxf::VideoBuffer>("render_buffer_output");
  if (!video_buffer) {
    throw std::runtime_error("Failed to allocate the video buffer for the render buffer output.");
  }

  if (render_buffer_input_enabled_) {
    // check if there is a input buffer given to copy the output into
    auto render_buffer_input = op_input.receive<gxf::Entity>("render_buffer_input").value();
    if (!render_buffer_input) {
      throw std::runtime_error("No message available at 'render_buffer_input'.");
    }

    // Get the empty input buffer
    const auto& video_buffer_in =
        static_cast<nvidia::gxf::Entity>(render_buffer_input).get<nvidia::gxf::VideoBuffer>();
    if (!video_buffer_in) {
      throw std::runtime_error("No video buffer attached to message on 'render_buffer_input'.");
    }

    const nvidia::gxf::VideoBufferInfo info = video_buffer_in.value()->video_frame_info();

    if ((info.color_format != nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA)) {
      throw std::runtime_error("Invalid render buffer input, expected an RGBA buffer.");
    }

    video_buffer.value()->wrapMemory(info,
                                     video_buffer_in.value()->size(),
                                     video_buffer_in.value()->storage_type(),
                                     video_buffer_in.value()->pointer(),
                                     nullptr);
  } else {
    // if there is no input buffer given, allocate one
    if (!allocator_.get()) {
      throw std::runtime_error("No render buffer input specified and no allocator set.");
    }

    // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());

    video_buffer.value()->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
        width_,
        height_,
        nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR,
        nvidia::gxf::MemoryStorageType::kDevice,
        allocator.value());
    if (!video_buffer.value()->pointer()) {
      throw std::runtime_error("Failed to allocate render output buffer.");
    }
  }

  // read the framebuffer
  viz::ReadFramebuffer(viz::ImageFormat::R8G8B8A8_UNORM,
                       width_,
                       height_,
                       video_buffer.value()->size(),
                       reinterpret_cast<CUdeviceptr>(video_buffer.value()->pointer()));

  // Output the filled render buffer object
  auto result = gxf::Entity(std::move(entity.value()));
  op_output.emit(result, "render_buffer_output");
}

void HolovizOp::set_input_spec(const InputSpec& input_spec) {
  viz::LayerPriority(input_spec.priority_);
  viz::LayerOpacity(input_spec.opacity_);
  for (auto&& view : input_spec.views_) {
    const float* matrix = nullptr;
    if (view.matrix_.has_value()) { matrix = view.matrix_.value().data(); }
    viz::LayerAddView(view.offset_x_, view.offset_y_, view.width_, view.height_, matrix);
  }
}

void HolovizOp::set_input_spec_geometry(const InputSpec& input_spec) {
  // first set common layer properties
  set_input_spec(input_spec);

  // now set geometry layer specific properties
  std::array<float, 4> color{1.f, 1.f, 1.f, 1.f};
  for (size_t index = 0; index < std::min(input_spec.color_.size(), color.size()); ++index) {
    color[index] = input_spec.color_[index];
  }
  viz::Color(color[0], color[1], color[2], color[3]);
  viz::PointSize(input_spec.point_size_);
  viz::LineWidth(input_spec.line_width_);
}

void HolovizOp::render_depth_map(InputSpec* const input_spec_depth_map,
                                 const BufferInfo& buffer_info_depth_map,
                                 InputSpec* const input_spec_depth_map_color,
                                 const BufferInfo& buffer_info_depth_map_color) {
  /// @todo this is assuming HWC, should either auto-detect (if possible) or make user
  /// configurable
  const auto height = buffer_info_depth_map.shape.dimension(0);
  const auto width = buffer_info_depth_map.shape.dimension(1);

  viz::ImageFormat depth_map_color_fmt = viz::ImageFormat::R8G8B8A8_UNORM;
  CUdeviceptr depth_map_color_device_ptr = 0;
  if (input_spec_depth_map_color) {
    // if there is a color buffer, the size has to match

    /// @todo this is assuming HWC, should either auto-detect (if possible) or make user
    /// configurable
    const auto color_height = buffer_info_depth_map_color.shape.dimension(0);
    const auto color_width = buffer_info_depth_map_color.shape.dimension(1);
    if ((width != color_width) || (height != color_height)) {
      throw std::runtime_error(
          fmt::format("The buffer dimensions {}x{} of the depth map color buffer '{}' need to "
                      "match the depth map '{}' dimensions {}x{}",
                      color_width,
                      color_height,
                      buffer_info_depth_map_color.name,
                      buffer_info_depth_map.name,
                      width,
                      height));
    }
    auto maybe_image_format = getImageFormatFromTensor(buffer_info_depth_map_color);
    if (!maybe_image_format) {
      auto code = nvidia::gxf::ToResultCode(maybe_image_format);
      throw std::runtime_error(fmt::format("failed setting input format with code {}", code));
    }
    depth_map_color_fmt = maybe_image_format.value();

    depth_map_color_device_ptr =
        reinterpret_cast<CUdeviceptr>(buffer_info_depth_map_color.buffer_ptr);
  }

  viz::DepthMapRenderMode depth_map_render_mode;
  switch (input_spec_depth_map->depth_map_render_mode_) {
    case DepthMapRenderMode::POINTS:
      depth_map_render_mode = viz::DepthMapRenderMode::POINTS;
      break;
    case DepthMapRenderMode::LINES:
      depth_map_render_mode = viz::DepthMapRenderMode::LINES;
      break;
    case DepthMapRenderMode::TRIANGLES:
      depth_map_render_mode = viz::DepthMapRenderMode::TRIANGLES;
      break;
    default:
      throw std::runtime_error(
          fmt::format("Unhandled depth map render mode {}",
                      depthMapRenderModeToString(input_spec_depth_map->depth_map_render_mode_)));
  }

  // start a geometry layer containing the depth map
  viz::BeginGeometryLayer();
  set_input_spec_geometry(*input_spec_depth_map);

  const auto cu_buffer_ptr = reinterpret_cast<CUdeviceptr>(buffer_info_depth_map.buffer_ptr);
  viz::DepthMap(depth_map_render_mode,
                width,
                height,
                viz::ImageFormat::R8_UNORM,
                cu_buffer_ptr,
                depth_map_color_fmt,
                depth_map_color_device_ptr);
  viz::EndLayer();
}

void HolovizOp::initialize() {
  register_converter<std::vector<InputSpec>>();
  register_codec<std::vector<InputSpec>>("std::vector<holoscan::ops::HolovizOp::InputSpec>", true);

  // Set up prerequisite parameters before calling Operator::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'window_close_scheduling_term'
  auto has_window_close_scheduling_term =
      std::find_if(args().begin(), args().end(), [](const auto& arg) {
        return (arg.name() == "window_close_scheduling_term");
      });
  // Create the BooleanCondition if there is no argument provided.
  if (has_window_close_scheduling_term == args().end()) {
    window_close_scheduling_term_ =
        frag->make_condition<holoscan::BooleanCondition>("window_close_scheduling_term");
    add_arg(window_close_scheduling_term_.get());
  }

  // TODO: avoid duplicate computations between check_port_enabled and enable_conditional_port
  render_buffer_input_enabled_ = check_port_enabled("render_buffer_input");
  render_buffer_output_enabled_ = check_port_enabled("render_buffer_output");
  camera_pose_output_enabled_ = check_port_enabled("camera_pose_output");

  // Conditional inputs and outputs are enabled using a boolean argument
  enable_conditional_port("render_buffer_input");
  enable_conditional_port("render_buffer_output");
  enable_conditional_port("camera_pose_output");

  // parent class initialize() call must be after the argument additions above
  Operator::initialize();
}

void HolovizOp::start() {
  // set the font to be used
  if (!font_path_.get().empty()) { viz::SetFont(font_path_.get().c_str(), 25.f); }

  // initialize Holoviz
  viz::InitFlags init_flags = viz::InitFlags::NONE;
  if (fullscreen_ && headless_) {
    throw std::runtime_error("Headless and fullscreen are mutually exclusive.");
  }
  if (fullscreen_) { init_flags = viz::InitFlags::FULLSCREEN; }
  if (headless_) { init_flags = viz::InitFlags::HEADLESS; }

  if (use_exclusive_display_) {
    viz::Init(
        display_name_.get().c_str(), width_, height_, uint32_t(framerate_ * 1000.f), init_flags);
  } else {
    viz::Init(width_, height_, window_title_.get().c_str(), init_flags);
  }

  // get the color lookup table
  const auto& color_lut = color_lut_.get();
  lut_.reserve(color_lut.size() * 4);
  for (auto&& color : color_lut) {
    if (color.size() != 4) {
      std::string msg = fmt::format(
          "Expected four components in color lookup table element, but got {}", color.size());
      throw std::runtime_error(msg);
    }
    lut_.insert(lut_.end(), color.begin(), color.end());
  }

  // cast Condition to BooleanCondition
  window_close_scheduling_term_->enable_tick();

  // Copy the user defined input spec list to the internal input spec list. If there is no user
  // defined input spec it will be generated from the first messages received.
  if (!tensors_.get().empty()) {
    initial_input_spec_.reserve(tensors_.get().size());
    initial_input_spec_.insert(
        initial_input_spec_.begin(), tensors_.get().begin(), tensors_.get().end());
  }
}

void HolovizOp::stop() {
  viz::Shutdown();
}

void HolovizOp::compute(InputContext& op_input, OutputContext& op_output,
                        ExecutionContext& context) {
  std::vector<gxf::Entity> messages_h =
      op_input.receive<std::vector<gxf::Entity>>("receivers").value();

  // create vector of nvidia::gxf::Entity as expected by the code below
  std::vector<nvidia::gxf::Entity> messages;
  messages.reserve(messages_h.size());
  for (auto& message_h : messages_h) {
    // cast each holoscan::gxf:Entity to its base class
    nvidia::gxf::Entity message = static_cast<nvidia::gxf::Entity>(message_h);
    messages.push_back(message);
  }

  // cast Condition to BooleanCondition
  if (viz::WindowShouldClose()) {
    window_close_scheduling_term_->disable_tick();
    return;
  }

  // nothing to do if minimized
  if (viz::WindowIsMinimized()) { return; }

  // build the input spec list
  std::vector<InputSpec> input_spec_list(initial_input_spec_);

  // check the messages for input specs, they are added to the list
  if (!op_input.empty("input_specs")) {
    auto msg_input_specs =
        op_input.receive<std::vector<holoscan::ops::HolovizOp::InputSpec>>("input_specs").value();
    input_spec_list.insert(input_spec_list.end(), msg_input_specs.begin(), msg_input_specs.end());
  }

  // then get all tensors and video buffers of all messages, check if an input spec for the tensor
  // is already there, if not try to detect the input spec from the tensor or video buffer
  // information
  for (auto&& message : messages) {
    const auto tensors = message.findAll<nvidia::gxf::Tensor>();
    for (auto&& tensor : tensors.value()) {
      // check if an input spec with the same tensor name already exist
      const std::string tensor_name(tensor->name());
      const auto it = std::find_if(
          std::begin(input_spec_list), std::end(input_spec_list), [&tensor_name](const auto& v) {
            return v.tensor_name_ == tensor_name;
          });

      if (it == std::end(input_spec_list)) {
        // no input spec found, try to detect
        BufferInfo buffer_info;
        if (buffer_info.init(tensor.value()) != GXF_FAILURE) {
          // try to detect the input type, if we can't detect it, ignore the tensor
          const auto maybe_input_type = detectInputType(buffer_info, !lut_.empty());
          if (maybe_input_type) {
            input_spec_list.emplace_back(tensor->name(), maybe_input_type.value());
          }
        }
      }
    }
    const auto video_buffers = message.findAll<nvidia::gxf::VideoBuffer>();
    for (auto&& video_buffer : video_buffers.value()) {
      // check if an input spec with the same tensor name already exist
      const std::string tensor_name(video_buffer->name());
      const auto it = std::find_if(
          std::begin(input_spec_list), std::end(input_spec_list), [&tensor_name](const auto& v) {
            return v.tensor_name_ == tensor_name;
          });

      if (it == std::end(input_spec_list)) {
        // no input spec found, try to detect
        BufferInfo buffer_info;
        if (buffer_info.init(video_buffer.value()) != GXF_FAILURE) {
          // try to detect the input type, if we can't detect it, ignore the tensor
          const auto maybe_input_type = detectInputType(buffer_info, !lut_.empty());
          if (maybe_input_type) {
            input_spec_list.emplace_back(video_buffer->name(), maybe_input_type.value());
          }
        }
      }
    }
  }

  // get the CUDA stream from the input message
  const gxf_result_t result = cuda_stream_handler_.fromMessages(context.context(), messages);
  if (result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }
  viz::SetCudaStream(cuda_stream_handler_.getCudaStream(context.context()));

  // Depth maps have two tensors, the depth map itself and the depth map color values. Therefore
  // collect the information in the first pass through the input specs and then render the depth
  // map.
  InputSpec* input_spec_depth_map = nullptr;
  BufferInfo buffer_info_depth_map;
  InputSpec* input_spec_depth_map_color = nullptr;
  BufferInfo buffer_info_depth_map_color;

  // begin visualization
  viz::Begin();

  // get the tensors attached to the messages by the tensor names defined by the input spec and
  // display them
  for (auto& input_spec : input_spec_list) {
    nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::Tensor>> maybe_input_tensor =
        nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE};
    nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>> maybe_input_video =
        nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE};
    auto message = messages.begin();
    while (message != messages.end()) {
      maybe_input_tensor = message->get<nvidia::gxf::Tensor>(input_spec.tensor_name_.c_str());
      if (maybe_input_tensor) {
        // pick the first one with that name
        break;
      }
      // check for video if no tensor found
      maybe_input_video = message->get<nvidia::gxf::VideoBuffer>(input_spec.tensor_name_.c_str());
      if (maybe_input_video) {  // pick the first one with that name
        break;
      }
      ++message;
    }
    if (message == messages.end()) {
      throw std::runtime_error(
          fmt::format("Failed to retrieve input '{}'", input_spec.tensor_name_));
    }

    BufferInfo buffer_info;
    gxf_result_t result;
    if (maybe_input_tensor) {
      result = buffer_info.init(maybe_input_tensor.value());
    } else {
      result = buffer_info.init(maybe_input_video.value());
    }
    if (result != GXF_SUCCESS) {
      throw std::runtime_error(fmt::format("Unsupported buffer format tensor/video buffer '{}'",
                                           input_spec.tensor_name_));
    }

    // If the buffer is empty, skip processing it
    if (buffer_info.bytes_size == 0) { continue; }

    // if the input type is unknown it now can be detected using the image properties
    if (input_spec.type_ == InputType::UNKNOWN) {
      const auto maybe_input_type = detectInputType(buffer_info, !lut_.empty());
      if (!maybe_input_type) {
        auto code = nvidia::gxf::ToResultCode(maybe_input_type);
        throw std::runtime_error(fmt::format("failed setting input type with code {}", code));
      }
      input_spec.type_ = maybe_input_type.value();
    }

    switch (input_spec.type_) {
      case InputType::COLOR:
      case InputType::COLOR_LUT: {
        // 2D color image

        // sanity checks
        if (buffer_info.rank != 3) {
          throw std::runtime_error(
              fmt::format("Expected rank 3 for tensor '{}', type '{}', but got {}",
                          buffer_info.name,
                          inputTypeToString(input_spec.type_),
                          buffer_info.rank));
        }

        /// @todo this is assuming HWC, should either auto-detect (if possible) or make user
        /// configurable
        const auto height = buffer_info.shape.dimension(0);
        const auto width = buffer_info.shape.dimension(1);
        const auto channels = buffer_info.shape.dimension(2);

        if (input_spec.type_ == InputType::COLOR_LUT) {
          if (channels != 1) {
            throw std::runtime_error(fmt::format(
                "Expected one channel for tensor '{}' when using lookup table, but got {}",
                buffer_info.name,
                channels));
          }
          if (lut_.empty()) {
            throw std::runtime_error(fmt::format(
                "Type of tensor '{}' is '{}', but a color lookup table has not been specified",
                buffer_info.name,
                inputTypeToString(input_spec.type_)));
          }
        }

        auto maybe_image_format = getImageFormatFromTensor(buffer_info);
        if (!maybe_image_format) {
          auto code = nvidia::gxf::ToResultCode(maybe_image_format);
          throw std::runtime_error(fmt::format("failed setting input format with code {}", code));
        }
        const viz::ImageFormat image_format = maybe_image_format.value();

        // start an image layer
        viz::BeginImageLayer();
        set_input_spec(input_spec);

        if (input_spec.type_ == InputType::COLOR_LUT) {
          viz::LUT(lut_.size() / 4,
                   viz::ImageFormat::R32G32B32A32_SFLOAT,
                   lut_.size() * sizeof(float),
                   lut_.data());
        }

        if (buffer_info.storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
          // if it's the device convert to `CUDeviceptr`
          const auto cu_buffer_ptr = reinterpret_cast<CUdeviceptr>(buffer_info.buffer_ptr);
          viz::ImageCudaDevice(width, height, image_format, cu_buffer_ptr, buffer_info.stride[0]);
        } else {
          // convert to void * if using the system/host
          const auto host_buffer_ptr = reinterpret_cast<const void*>(buffer_info.buffer_ptr);
          viz::ImageHost(width, height, image_format, host_buffer_ptr, buffer_info.stride[0]);
        }
        viz::EndLayer();
      } break;

      case InputType::POINTS:
      case InputType::LINES:
      case InputType::LINE_STRIP:
      case InputType::TRIANGLES:
      case InputType::CROSSES:
      case InputType::RECTANGLES:
      case InputType::OVALS:
      case InputType::TEXT:
      case InputType::POINTS_3D:
      case InputType::LINES_3D:
      case InputType::LINE_STRIP_3D:
      case InputType::TRIANGLES_3D: {
        // geometry layer
        if (buffer_info.element_type != nvidia::gxf::PrimitiveType::kFloat32) {
          throw std::runtime_error(fmt::format(
              "Expected gxf::PrimitiveType::kFloat32 element type for coordinates, but got "
              "element type {}",
              static_cast<int>(buffer_info.element_type)));
        }

        // get pointer to tensor buffer
        std::vector<nvidia::byte> host_buffer;
        if (buffer_info.storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
          host_buffer.resize(buffer_info.bytes_size);

          // copy from device to host
          CUDA_TRY(cudaMemcpyAsync(static_cast<void*>(host_buffer.data()),
                                   static_cast<const void*>(buffer_info.buffer_ptr),
                                   buffer_info.bytes_size,
                                   cudaMemcpyDeviceToHost,
                                   cuda_stream_handler_.getCudaStream(context.context())));
          // wait for the CUDA memory copy to finish
          CUDA_TRY(cudaStreamSynchronize(cuda_stream_handler_.getCudaStream(context.context())));

          buffer_info.buffer_ptr = host_buffer.data();
        }

        // start a geometry layer
        viz::BeginGeometryLayer();
        set_input_spec_geometry(input_spec);

        /// @todo this is assuming NHW, should either auto-detect (if possible) or make user
        /// configurable
        const auto coordinates = buffer_info.shape.dimension(1);
        const auto components = buffer_info.shape.dimension(2);

        if (input_spec.type_ == InputType::TEXT) {
          // text is defined by the top left coordinate and the size (x, y, s) per string, text
          // strings are define by InputSpec::text_
          if ((components < 2) || (components > 3)) {
            throw std::runtime_error(
                fmt::format("Expected two or three values per text, but got '{}'", components));
          }
          if (input_spec.text_.empty()) {
            throw std::runtime_error(fmt::format("No text has been specified by input spec '{}'.",
                                                 input_spec.tensor_name_));
          }
          const float* src_coord = reinterpret_cast<const float*>(buffer_info.buffer_ptr);
          for (int32_t index = 0; index < coordinates; ++index) {
            viz::Text(
                src_coord[0],
                src_coord[1],
                (components == 3) ? src_coord[2] : 0.05f,
                input_spec.text_[std::min(index, static_cast<int32_t>(input_spec.text_.size()) - 1)]
                    .c_str());
            src_coord += components;
          }
        } else {
          std::vector<float> coords;
          viz::PrimitiveTopology topology;
          uint32_t primitive_count;
          uint32_t coordinate_count;
          uint32_t values_per_coordinate;
          std::vector<float> default_coord;
          switch (input_spec.type_) {
            case InputType::POINTS:
              // point primitives, one coordinate (x, y) per primitive
              if (components != 2) {
                throw std::runtime_error(
                    fmt::format("Expected two values per point, but got '{}'", components));
              }
              topology = viz::PrimitiveTopology::POINT_LIST;
              primitive_count = coordinates;
              coordinate_count = primitive_count;
              values_per_coordinate = 2;
              default_coord = {0.f, 0.f};
              break;
            case InputType::LINES:
              // line primitives, two coordinates (x0, y0) and (x1, y1) per primitive
              if (components != 2) {
                throw std::runtime_error(
                    fmt::format("Expected two values per line vertex, but got '{}'", components));
              }
              topology = viz::PrimitiveTopology::LINE_LIST;
              primitive_count = coordinates / 2;
              coordinate_count = primitive_count * 2;
              values_per_coordinate = 2;
              default_coord = {0.f, 0.f};
              break;
            case InputType::LINE_STRIP:
              // line strip primitive, a line primitive i is defined by each coordinate (xi, yi) and
              // the following (xi+1, yi+1)
              if (components != 2) {
                throw std::runtime_error(fmt::format(
                    "Expected two values per line strip vertex, but got '{}'", components));
              }
              topology = viz::PrimitiveTopology::LINE_STRIP;
              primitive_count = coordinates - 1;
              coordinate_count = coordinates;
              values_per_coordinate = 2;
              default_coord = {0.f, 0.f};
              break;
            case InputType::TRIANGLES:
              // triangle primitive, three coordinates (x0, y0), (x1, y1) and (x2, y2) per primitive
              if (components != 2) {
                throw std::runtime_error(fmt::format(
                    "Expected two values per triangle vertex, but got '{}'", components));
              }
              topology = viz::PrimitiveTopology::TRIANGLE_LIST;
              primitive_count = coordinates / 3;
              coordinate_count = primitive_count * 3;
              values_per_coordinate = 2;
              default_coord = {0.f, 0.f};
              break;
            case InputType::CROSSES:
              // cross primitive, a cross is defined by the center coordinate and the size (xi, yi,
              // si)
              if ((components < 2) || (components > 3)) {
                throw std::runtime_error(fmt::format(
                    "Expected two or three values per cross, but got '{}'", components));
              }

              topology = viz::PrimitiveTopology::CROSS_LIST;
              primitive_count = coordinates;
              coordinate_count = primitive_count;
              values_per_coordinate = 3;
              default_coord = {0.f, 0.f, 0.05f};
              break;
            case InputType::RECTANGLES:
              // axis aligned rectangle primitive, each rectangle is defined by two coordinates (xi,
              // yi) and (xi+1, yi+1)
              if (components != 2) {
                throw std::runtime_error(fmt::format(
                    "Expected two values per rectangle vertex, but got '{}'", components));
              }
              topology = viz::PrimitiveTopology::RECTANGLE_LIST;
              primitive_count = coordinates / 2;
              coordinate_count = primitive_count * 2;
              values_per_coordinate = 2;
              default_coord = {0.f, 0.f};
              break;
            case InputType::OVALS:
              // oval primitive, an oval primitive is defined by the center coordinate and the axis
              // sizes (xi, yi, sxi, syi)
              if ((components < 2) || (components > 4)) {
                throw std::runtime_error(fmt::format(
                    "Expected two, three or four values per oval, but got '{}'", components));
              }
              topology = viz::PrimitiveTopology::OVAL_LIST;
              primitive_count = coordinates;
              coordinate_count = primitive_count;
              values_per_coordinate = 4;
              default_coord = {0.f, 0.f, 0.05f, 0.05f};
              break;
            case InputType::POINTS_3D:
              // point primitives, one coordinate (x, y, z) per primitive
              if (components != 3) {
                throw std::runtime_error(
                    fmt::format("Expected three values per 3D point, but got '{}'", components));
              }
              topology = viz::PrimitiveTopology::POINT_LIST_3D;
              primitive_count = coordinates;
              coordinate_count = primitive_count;
              values_per_coordinate = 3;
              default_coord = {0.f, 0.f, 0.f};
              break;
            case InputType::LINES_3D:
              // line primitives, two coordinates (x0, y0, z0) and (x1, y1, z1) per primitive
              if (components != 3) {
                throw std::runtime_error(fmt::format(
                    "Expected three values per 3D line vertex, but got '{}'", components));
              }
              topology = viz::PrimitiveTopology::LINE_LIST_3D;
              primitive_count = coordinates / 2;
              coordinate_count = primitive_count * 2;
              values_per_coordinate = 3;
              default_coord = {0.f, 0.f, 0.f};
              break;
            case InputType::LINE_STRIP_3D:
              // line primitives, two coordinates (x0, y0, z0) and (x1, y1, z1) per primitive
              if (components != 3) {
                throw std::runtime_error(fmt::format(
                    "Expected three values per 3D line strip vertex, but got '{}'", components));
              }
              topology = viz::PrimitiveTopology::LINE_STRIP_3D;
              primitive_count = coordinates - 1;
              coordinate_count = coordinates;
              values_per_coordinate = 3;
              default_coord = {0.f, 0.f};
              break;
            case InputType::TRIANGLES_3D:
              // triangle primitive, three coordinates (x0, y0, z0), (x1, y1, z1) and (x2, y2, z2)
              // per primitive
              if (components != 3) {
                throw std::runtime_error(fmt::format(
                    "Expected three values per 3D triangle vertex, but got '{}'", components));
              }
              topology = viz::PrimitiveTopology::TRIANGLE_LIST_3D;
              primitive_count = coordinates / 3;
              coordinate_count = primitive_count * 3;
              values_per_coordinate = 3;
              default_coord = {0.f, 0.f};
              break;
            default:
              throw std::runtime_error(
                  fmt::format("Unhandled tensor type '{}'", inputTypeToString(input_spec.type_)));
          }

          // copy coordinates
          const float* src_coord = reinterpret_cast<const float*>(buffer_info.buffer_ptr);
          coords.reserve(coordinate_count * values_per_coordinate);
          for (int32_t index = 0; index < static_cast<int32_t>(coordinate_count); ++index) {
            int32_t component_index = 0;
            // copy from source array
            while (component_index < std::min(components, int32_t(values_per_coordinate))) {
              coords.push_back(src_coord[component_index]);
              ++component_index;
            }
            // fill from default array
            while (component_index < static_cast<int32_t>(values_per_coordinate)) {
              coords.push_back(default_coord[component_index]);
              ++component_index;
            }
            src_coord += buffer_info.stride[1] / sizeof(float);
          }

          if (primitive_count) {
            viz::Primitive(topology, primitive_count, coords.size(), coords.data());
          }
        }

        viz::EndLayer();
      } break;
      case InputType::DEPTH_MAP: {
        // 2D depth map

        // sanity checks
        if (buffer_info.rank != 3) {
          throw std::runtime_error(
              fmt::format("Expected rank 3 for tensor '{}', type '{}', but got {}",
                          buffer_info.name,
                          inputTypeToString(input_spec.type_),
                          buffer_info.rank));
        }
        if (buffer_info.element_type != nvidia::gxf::PrimitiveType::kUnsigned8) {
          throw std::runtime_error(fmt::format(
              "Expected gxf::PrimitiveType::kUnsigned8 element type for tensor '{}', but got "
              "element type {}",
              buffer_info.name,
              static_cast<int>(buffer_info.element_type)));
        }
        if (buffer_info.storage_type != nvidia::gxf::MemoryStorageType::kDevice) {
          throw std::runtime_error(
              fmt::format("Only device storage is supported for tensor '{}'", buffer_info.name));
        }
        /// @todo this is assuming HWC, should either auto-detect (if possible) or make user
        /// configurable
        const auto channels = buffer_info.shape.dimension(2);
        if (buffer_info.shape.dimension(2) != 1) {
          throw std::runtime_error(fmt::format(
              "Expected one channel for tensor '{}', but got {}", buffer_info.name, channels));
        }

        // Store the depth map information, we render after the end of the input spec loop when
        // we also have the (optional) depth map color information.
        input_spec_depth_map = &input_spec;
        buffer_info_depth_map = buffer_info;
      } break;
      case InputType::DEPTH_MAP_COLOR: {
        // 2D depth map color

        // sanity checks
        if (buffer_info.rank != 3) {
          throw std::runtime_error(
              fmt::format("Expected rank 3 for tensor '{}', type '{}', but got {}",
                          buffer_info.name,
                          inputTypeToString(input_spec.type_),
                          buffer_info.rank));
        }
        if (buffer_info.element_type != nvidia::gxf::PrimitiveType::kUnsigned8) {
          throw std::runtime_error(fmt::format(
              "Expected gxf::PrimitiveType::kUnsigned8 element type for tensor '{}', but got "
              "element type {}",
              buffer_info.name,
              static_cast<int>(buffer_info.element_type)));
        }
        if (buffer_info.storage_type != nvidia::gxf::MemoryStorageType::kDevice) {
          throw std::runtime_error(
              fmt::format("Only device storage is supported for tensor '{}'", buffer_info.name));
        }
        /// @todo this is assuming HWC, should either auto-detect (if possible) or make user
        /// configurable
        const auto channels = buffer_info.shape.dimension(2);
        if (buffer_info.shape.dimension(2) != 4) {
          throw std::runtime_error(fmt::format(
              "Expected four channels for tensor '{}', but got {}", buffer_info.name, channels));
        }

        // Store the depth map color information, we render after the end of the input spec loop
        // when we have both the depth and color information
        input_spec_depth_map_color = &input_spec;
        buffer_info_depth_map_color = buffer_info;
      } break;
      default:
        throw std::runtime_error(
            fmt::format("Unhandled input type '{}'", inputTypeToString(input_spec.type_)));
    }
  }

  // we now have both tensors to render a depth map
  if (input_spec_depth_map) {
    render_depth_map(input_spec_depth_map,
                     buffer_info_depth_map,
                     input_spec_depth_map_color,
                     buffer_info_depth_map_color);
  }

  viz::End();

  // check if the render buffer should be output
  if (render_buffer_output_enabled_) { read_frame_buffer(op_input, op_output, context); }

  // check if the the camera pose should be output
  if (camera_pose_output_enabled_) {
    auto camera_pose_output = std::make_shared<std::array<float, 16>>();

    viz::GetCameraPose(camera_pose_output->size(), camera_pose_output->data());

    op_output.emit(camera_pose_output, "camera_pose_output");
  }

  if (is_first_tick_) {
    // log the input spec
    YAML::Emitter emitter;
    emitter << YAML::Node(input_spec_list);
    HOLOSCAN_LOG_INFO("Input spec:\n{}\n", emitter.c_str());
    is_first_tick_ = false;
  }
}

std::string HolovizOp::InputSpec::description() const {
  YAML::Emitter emitter;
  emitter << YAML::convert<holoscan::ops::HolovizOp::InputSpec>::encode(*this);
  return emitter.c_str();
}

HolovizOp::InputSpec::InputSpec(const std::string& yaml_description) {
  YAML::Node input_spec_node = YAML::Load(yaml_description);
  YAML::convert<holoscan::ops::HolovizOp::InputSpec>::decode(input_spec_node, *this);
}
}  // namespace holoscan::ops
