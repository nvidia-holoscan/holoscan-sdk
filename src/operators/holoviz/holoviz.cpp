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

#include "holoscan/operators/holoviz/holoviz.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <magic_enum.hpp>

#include "holoscan/core/application.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/file_fifo_mutex.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/operators/holoviz/buffer_info.hpp"
#include "holoscan/operators/holoviz/codecs.hpp"
#include "holoscan/utils/cuda_macros.hpp"

#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"
#include "holoviz/holoviz.hpp"  // holoviz module

namespace viz = holoscan::viz;

namespace {

/**
 * Try to detect the input type enum for given buffer properties.
 *
 * @param buffer_info buffer info
 * @param has_lut true if the user specified a LUT
 *
 *  @return input type enum
 */
nvidia::gxf::Expected<holoscan::ops::HolovizOp::InputType> detect_input_type(
    const holoscan::ops::BufferInfo& buffer_info, bool has_lut) {
  nvidia::gxf::Expected<holoscan::ops::HolovizOp::InputType> input_type =
      nvidia::gxf::Unexpected{GXF_FAILURE};

  // auto detect type
  if ((buffer_info.components == 1) && has_lut) {
    // color image with lookup table
    input_type = holoscan::ops::HolovizOp::InputType::COLOR_LUT;
  } else if ((buffer_info.components == 2) &&
             (buffer_info.element_type == nvidia::gxf::PrimitiveType::kFloat32)) {
    // array of 2D coordinates, draw crosses
    input_type = holoscan::ops::HolovizOp::InputType::CROSSES;
  } else if (buffer_info.image_format != holoscan::ops::HolovizOp::ImageFormat::AUTO_DETECT) {
    // color image
    input_type = holoscan::ops::HolovizOp::InputType::COLOR;
  }

  if (!input_type.has_value()) {
    HOLOSCAN_LOG_ERROR(
        "Can't auto detect type of input '{}'. Please provide a valid type value in the input "
        "specification.\n"
        "Auto detection recognizes\n"
        "- `COLOR_LUT` when the `color_lut` parameter is set and the input buffer components is 1\n"
        "- `CROSSES` when the input buffer components is 2 and element type is `float`\n"
        "- `COLOR` when the input buffer has a valid color format",
        buffer_info.name);
  }
  return input_type;
}

/**
 * @brief Get the format string of input tensor/video buffer
 */
std::string get_format_str(
    const std::string& tensor_name,
    const nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::Tensor>>& maybe_input_tensor,
    const nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>>& maybe_input_video) {
  std::string format_str;
  if (maybe_input_tensor) {
    format_str = fmt::format("Tensor '{}':\n{}", tensor_name, *maybe_input_tensor.value().get());

  } else {
    format_str =
        fmt::format("VideoBuffer '{}':\n{}", tensor_name, *maybe_input_video.value().get());
  }
  return format_str;
}

/**
 * RAII type class to push a Holoviz instance. Previous instance will be restored when class
 * instance goes out of scope.
 */
class ScopedPushInstance {
 public:
  /**
   * Push a Holoviz instance
   *
   * @param instance instance to push
   */
  explicit ScopedPushInstance(holoscan::viz::InstanceHandle instance)
      : prev_instance_(viz::GetCurrent()) {
    viz::SetCurrent(instance);
  }

  /**
   * Destructor, restore the previous instance.
   */
  ~ScopedPushInstance() { viz::SetCurrent(prev_instance_); }

 private:
  // hide default and copy constructors, copy assignment
  ScopedPushInstance() = delete;
  ScopedPushInstance(const ScopedPushInstance&) = delete;
  ScopedPushInstance& operator=(const ScopedPushInstance&) = delete;

  const holoscan::viz::InstanceHandle prev_instance_;
};

/**
 * @brief Holoviz specific RAII class of the FileFIFOMutex. It enables a
 * scoped FileFIFOMutex only if the is_mutex_enabled flag is true. This class
 * manages scoped lock and unlock of the FileFIFOMutex.
 *
 */
class ScopedHolovizFileMutex {
 public:
  // Deleting the default, copy and assignment constructors
  ScopedHolovizFileMutex() = delete;
  ScopedHolovizFileMutex(const ScopedHolovizFileMutex&) = delete;
  ScopedHolovizFileMutex& operator=(const ScopedHolovizFileMutex&) = delete;

  explicit ScopedHolovizFileMutex(bool is_mutex_enabled,
                                  std::shared_ptr<holoscan::FileFIFOMutex> mutex) {
    if (is_mutex_enabled && mutex) {
      mutex_ = std::move(mutex);
      mutex_->lock();
    }
  }
  bool locked() const { return mutex_ && mutex_->locked(); }
  ~ScopedHolovizFileMutex() {
    if (mutex_) {
      try {
        mutex_->unlock();
      } catch (const std::exception& e) {
        // Silently handle any exceptions during cleanup
        try {
          HOLOSCAN_LOG_ERROR(
              "FileFIFOMutex unlock failed during ScopedHolovizFileMutex destructor with {}",
              e.what());
        } catch (...) {}
      }
    }
  }

 private:
  std::shared_ptr<holoscan::FileFIFOMutex> mutex_;
};

}  // namespace

namespace holoscan::ops {

/*static*/ void HolovizOp::key_callback_handler(void* user_pointer, viz::Key key,
                                                viz::KeyAndButtonAction action,
                                                viz::KeyModifiers modifiers) {
  reinterpret_cast<HolovizOp*>(user_pointer)->key_callback_.get()(key, action, modifiers);
}

/*static*/ void HolovizOp::unicode_char_callback_handler(void* user_pointer, uint32_t code_point) {
  reinterpret_cast<HolovizOp*>(user_pointer)->unicode_char_callback_.get()(code_point);
}

/*static*/ void HolovizOp::mouse_button_callback_handler(void* user_pointer,
                                                         viz::MouseButton button,
                                                         viz::KeyAndButtonAction action,
                                                         viz::KeyModifiers modifiers) {
  reinterpret_cast<HolovizOp*>(user_pointer)
      ->mouse_button_callback_.get()(button, action, modifiers);
}

/*static*/ void HolovizOp::scroll_callback_handler(void* user_pointer, double x_offset,
                                                   double y_offset) {
  reinterpret_cast<HolovizOp*>(user_pointer)->scroll_callback_.get()(x_offset, y_offset);
}

/*static*/ void HolovizOp::cursor_pos_callback_handler(void* user_pointer, double x_pos,
                                                       double y_pos) {
  reinterpret_cast<HolovizOp*>(user_pointer)->cursor_pos_callback_.get()(x_pos, y_pos);
}

/*static*/ void HolovizOp::framebuffer_size_callback_handler(void* user_pointer, int width,
                                                             int height) {
  reinterpret_cast<HolovizOp*>(user_pointer)->framebuffer_size_callback_.get()(width, height);
}

/*static*/ void HolovizOp::window_size_callback_handler(void* user_pointer, int width, int height) {
  reinterpret_cast<HolovizOp*>(user_pointer)->window_size_callback_.get()(width, height);
}

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
  constexpr float DEFAULT_FRAMERATE = 60.F;
  static const std::string DEFAULT_WINDOW_TITLE("Holoviz");
  static const std::string DEFAULT_DISPLAY_NAME("");
  constexpr bool DEFAULT_EXCLUSIVE_DISPLAY = false;
  constexpr bool DEFAULT_FULLSCREEN = false;
  constexpr bool DEFAULT_HEADLESS = false;
  constexpr bool DEFAULT_FRAMEBUFFER_SRGB = false;
  constexpr bool DEFAULT_VSYNC = false;
  constexpr uint32_t DEFAULT_MULTIPROCESS_FRAMEDROP_WAITTIME_MS = 0;
  static const std::string DEFAULT_HOLOVIZ_MULTIPROCESS_MUTEX_PATH("/tmp/holoscan_holoviz_mutex");
  constexpr ColorSpace DEFAULT_DISPLAY_COLOR_SPACE = ColorSpace::AUTO;

  spec.input<std::vector<gxf::Entity>>("receivers", IOSpec::kAnySize);

  spec.input<std::any>("input_specs").condition(ConditionType::kNone);
  spec.input<std::array<float, 3>>("camera_eye_input").condition(ConditionType::kNone);
  spec.input<std::array<float, 3>>("camera_look_at_input").condition(ConditionType::kNone);
  spec.input<std::array<float, 3>>("camera_up_input").condition(ConditionType::kNone);

  auto& render_buffer_input =
      spec.input<gxf::Entity>("render_buffer_input").condition(ConditionType::kNone);
  spec.param(render_buffer_input_,
             "render_buffer_input",
             "RenderBufferInput",
             "Input for an empty color render buffer.",
             &render_buffer_input);
  auto& render_buffer_output = spec.output<gxf::Entity>("render_buffer_output");
  spec.param(
      render_buffer_output_,
      "render_buffer_output",
      "RenderBufferOutput",
      "Output for a filled color render buffer. If an input color render buffer is specified "
      "it is using that one, else it allocates a new buffer.",
      &render_buffer_output);

  auto& depth_buffer_input =
      spec.input<gxf::Entity>("depth_buffer_input").condition(ConditionType::kNone);
  spec.param(depth_buffer_input_,
             "depth_buffer_input",
             "DepthBufferInput",
             "Input for an empty depth render buffer.",
             &depth_buffer_input);
  auto& depth_buffer_output = spec.output<gxf::Entity>("depth_buffer_output");
  spec.param(
      depth_buffer_output_,
      "depth_buffer_output",
      "DepthBufferOutput",
      "Output for a filled depth render buffer. If an input depth render buffer is specified "
      "it is using that one, else it allocates a new buffer.",
      &depth_buffer_output);

  auto& camera_pose_output = spec.output<nvidia::gxf::Pose3D>("camera_pose_output");
  spec.param(camera_pose_output_,
             "camera_pose_output",
             "CameraPoseOutput",
             "Output the camera extrinsics model.",
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
  spec.param(
      display_name_,
      "display_name",
      "Display name",
      "In exclusive display or fullscreen mode, name of display to use as shown with `xrandr` or "
      "`hwinfo --monitor`.",
      DEFAULT_DISPLAY_NAME);
  spec.param(width_,
             "width",
             "Width",
             "Window width or display resolution width if in exclusive display or fullscreen mode.",
             DEFAULT_WIDTH);
  spec.param(height_,
             "height",
             "Height",
             "Window height or display resolution height if in exclusive display or fullscreen "
             "mode.",
             DEFAULT_HEIGHT);
  spec.param(framerate_,
             "framerate",
             "Framerate",
             "Display framerate in Hz if in exclusive display mode.",
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
  spec.param(
      headless_,
      "headless",
      "Headless",
      "Enable headless mode. No window is opened, the color render buffer is output to "
      "`render_buffer_output` and the depth render buffer is output to `depth_buffer_output`.",
      DEFAULT_HEADLESS);
  spec.param(framebuffer_srgb_,
             "framebuffer_srgb",
             "Framebuffer SRGB",
             "Enable SRGB framebuffer. If set to true, the operator will use an sRGB framebuffer "
             "for rendering. If set to false, the operator will use a linear framebuffer.",
             DEFAULT_FRAMEBUFFER_SRGB);
  spec.param(vsync_,
             "vsync",
             "Vertical Sync",
             "Enable vertical sync. If set to true the operator waits for the next vertical "
             "blanking period of the display to update the current image.",
             DEFAULT_VSYNC);
  spec.param(multiprocess_framedrop_waittime_ms_,
             "multiprocess_framedrop_waittime_ms",
             "Multiprocess Framedrop Wait Time in ms",
             "To enable frame dropping in multiprocess scenarios when Holoviz mutex is not "
             "available, set this parameter to a non-zero positive integer. Setting it to zero "
             "(default value) will not drop any frames. This option is only available when "
             "HOLOSCAN_HOLOVIZ_MUTEX environment variable is set. ",
             DEFAULT_MULTIPROCESS_FRAMEDROP_WAITTIME_MS);
  spec.param(holoviz_multiprocess_mutex_path_,
             "holoviz_multiprocess_mutex_path",
             "Holoviz Multiprocess Mutex Path",
             "Path to the mutex file for multiprocess scenarios."
             "This option is only available when "
             "HOLOSCAN_HOLOVIZ_MUTEX environment variable is set.",
             DEFAULT_HOLOVIZ_MULTIPROCESS_MUTEX_PATH);
  spec.param(display_color_space_,
             "display_color_space",
             "Display Color Space",
             "Set the display color space. Supported color spaces depend on the display setup. "
             "'ColorSpace::SRGB_NONLINEAR' is always supported. In headless mode, only "
             "'ColorSpace::PASS_THROUGH' is supported since there is no display. For other color "
             "spaces the display needs to be configured for HDR.",
             DEFAULT_DISPLAY_COLOR_SPACE);
  spec.param(window_close_scheduling_term_,
             "window_close_scheduling_term",
             "WindowCloseSchedulingTerm",
             "This is a deprecated parameter name for `window_close_condition`. Please use "
             "`window_close_condition` instead as `window_close_scheduling_term` will be removed "
             "in a future release.",
             ParameterFlag::kOptional);
  spec.param(window_close_condition_,
             "window_close_condition",
             "window close condition",
             "BooleanCondition on the operator that will cause it to stop executing if the "
             "display window is closed. By default, this condition is created automatically "
             "during HolovizOp::initialize. The user may want to provide it if, for example, "
             "there are multiple HolovizOp operators and you want to share the same window close "
             "condition across both. By sharing the same condition, if one of the display "
             "windows is closed it would also close the other(s).",
             ParameterFlag::kOptional);
  spec.param(allocator_,
             "allocator",
             "Allocator",
             "Allocator used to allocate color and depth render buffer outputs.");
  spec.param(font_path_,
             "font_path",
             "FontPath",
             "File path for the font used for rendering text",
             std::string());
  spec.param(camera_pose_output_type_,
             "camera_pose_output_type",
             "Camera Pose Output Type",
             "Type of data output at `camera_pose_output`. Supported values are "
             "`projection_matrix` and `extrinsics_model`.",
             std::string("projection_matrix"));
  spec.param(camera_eye_, "camera_eye", "Camera Eye", "Camera eye position", {{0.F, 0.F, 1.F}});
  spec.param(camera_look_at_,
             "camera_look_at",
             "Camera Look At",
             "Camera look at position",
             {{0.F, 0.F, 0.F}});
  spec.param(camera_up_, "camera_up", "Camera Up", "Camera up vector", {{0.F, 1.F, 0.F}});

  spec.param(key_callback_,
             "key_callback",
             "Key Callback",
             "The callback function is called when a key is pressed, released or repeated.");
  spec.param(unicode_char_callback_,
             "unicode_char_callback",
             "Unicode Char Callback",
             "The callback function is called when a Unicode character is input.");
  spec.param(mouse_button_callback_,
             "mouse_button_callback",
             "Mouse Button Callback",
             "The callback function is called when a mouse button is pressed or released.");
  spec.param(scroll_callback_,
             "scroll_callback",
             "Scroll Callback",
             "The callback function is called when a scrolling device is used, such as a mouse "
             "scroll wheel or the scroll area of a touch pad.");
  spec.param(
      cursor_pos_callback_,
      "cursor_pos_callback",
      "Cursor Pos Callback",
      "The callback function is called when the cursor position changes. Coordinates are provided "
      "in screen coordinates, relative to the upper left edge of the content area.");
  spec.param(framebuffer_size_callback_,
             "framebuffer_size_callback",
             "Framebuffer Size Callback",
             "The callback function is called when the framebuffer is resized.");
  spec.param(window_size_callback_,
             "window_size_callback",
             "Window Size Callback",
             "The callback function is called when the window is resized.");
  spec.param(layer_callback_,
             "layer_callback",
             "Layer Callback",
             "The callback function is called when HolovizOp processed all layers defined by the "
             "input specification. It can be used to add extra layers.");
  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "CUDA Stream Pool",
             "Instance of gxf::CudaStreamPool.");
}

/*static*/ const std::array<std::pair<holoscan::ops::HolovizOp::InputType, std::string>, 17>
    HolovizOp::kInputTypeToStr{
        {{holoscan::ops::HolovizOp::InputType::UNKNOWN, "unknown"},
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

/*static*/ nvidia::gxf::Expected<holoscan::ops::HolovizOp::InputType>
HolovizOp::inputTypeFromString(const std::string& string) {
  const auto it = std::find_if(std::cbegin(kInputTypeToStr),
                               std::cend(kInputTypeToStr),
                               [&string](const auto& v) { return v.second == string; });
  if (it != std::cend(kInputTypeToStr)) { return it->first; }

  HOLOSCAN_LOG_ERROR("Unsupported input type '{}'", string);
  return nvidia::gxf::Unexpected{GXF_FAILURE};
}

/*static*/ std::string HolovizOp::inputTypeToString(
    holoscan::ops::HolovizOp::InputType input_type) {
  const auto it = std::find_if(std::cbegin(kInputTypeToStr),
                               std::cend(kInputTypeToStr),
                               [&input_type](const auto& v) { return v.first == input_type; });
  if (it != std::cend(kInputTypeToStr)) { return it->second; }

  return "invalid";
}

/*static*/ const std::array<std::pair<holoscan::ops::HolovizOp::ImageFormat, std::string>, 41>
    HolovizOp::kImageFormatToStr{
        {{holoscan::ops::HolovizOp::ImageFormat::AUTO_DETECT, "auto_detect"},
         {holoscan::ops::HolovizOp::ImageFormat::R8_UINT, "r8_uint"},
         {holoscan::ops::HolovizOp::ImageFormat::R8_SINT, "r8_sint"},
         {holoscan::ops::HolovizOp::ImageFormat::R8_UNORM, "r8_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::R8_SNORM, "r8_snorm"},
         {holoscan::ops::HolovizOp::ImageFormat::R8_SRGB, "r8_srgb"},
         {holoscan::ops::HolovizOp::ImageFormat::R16_UINT, "r16_uint"},
         {holoscan::ops::HolovizOp::ImageFormat::R16_SINT, "r16_sint"},
         {holoscan::ops::HolovizOp::ImageFormat::R16_UNORM, "r16_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::R16_SNORM, "r16_snorm"},
         {holoscan::ops::HolovizOp::ImageFormat::R16_SFLOAT, "r16_sfloat"},
         {holoscan::ops::HolovizOp::ImageFormat::R32_UINT, "r32_uint"},
         {holoscan::ops::HolovizOp::ImageFormat::R32_SINT, "r32_sint"},
         {holoscan::ops::HolovizOp::ImageFormat::R32_SFLOAT, "r32_sfloat"},
         {holoscan::ops::HolovizOp::ImageFormat::R8G8B8_UNORM, "r8g8b8_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::R8G8B8_SNORM, "r8g8b8_snorm"},
         {holoscan::ops::HolovizOp::ImageFormat::R8G8B8_SRGB, "r8g8b8_srgb"},
         {holoscan::ops::HolovizOp::ImageFormat::R8G8B8A8_UNORM, "r8g8b8a8_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::R8G8B8A8_SNORM, "r8g8b8a8_snorm"},
         {holoscan::ops::HolovizOp::ImageFormat::R8G8B8A8_SRGB, "r8g8b8a8_srgb"},
         {holoscan::ops::HolovizOp::ImageFormat::R16G16B16A16_UNORM, "r16g16b16a16_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::R16G16B16A16_SNORM, "r16g16b16a16_snorm"},
         {holoscan::ops::HolovizOp::ImageFormat::R16G16B16A16_SFLOAT, "r16g16b16a16_sfloat"},
         {holoscan::ops::HolovizOp::ImageFormat::R32G32B32A32_SFLOAT, "r32g32b32a32_sfloat"},
         {holoscan::ops::HolovizOp::ImageFormat::A2B10G10R10_UNORM_PACK32,
          "a2b10g10r10_unorm_pack32"},
         {holoscan::ops::HolovizOp::ImageFormat::A2R10G10B10_UNORM_PACK32,
          "a2r10g10b10_unorm_pack32"},
         {holoscan::ops::HolovizOp::ImageFormat::B8G8R8A8_UNORM, "b8g8r8a8_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::B8G8R8A8_SRGB, "b8g8r8a8_srgb"},
         {holoscan::ops::HolovizOp::ImageFormat::A8B8G8R8_UNORM_PACK32, "a8b8g8r8_unorm_pack32"},
         {holoscan::ops::HolovizOp::ImageFormat::A8B8G8R8_SRGB_PACK32, "a8b8g8r8_srgb_pack32"},
         {holoscan::ops::HolovizOp::ImageFormat::Y8U8Y8V8_422_UNORM, "y8u8y8v8_422_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::U8Y8V8Y8_422_UNORM, "u8y8v8y8_422_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::Y8_U8V8_2PLANE_420_UNORM,
          "y8_u8v8_2plane_420_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::Y8_U8V8_2PLANE_422_UNORM,
          "y8_u8v8_2plane_422_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::Y8_U8_V8_3PLANE_420_UNORM,
          "y8_u8_v8_3plane_420_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::Y8_U8_V8_3PLANE_422_UNORM,
          "y8_u8_v8_3plane_422_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::Y16_U16V16_2PLANE_420_UNORM,
          "y16_u16v16_2plane_420_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::Y16_U16V16_2PLANE_422_UNORM,
          "y16_u16v16_2plane_422_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::Y16_U16_V16_3PLANE_420_UNORM,
          "y16_u16_v16_3plane_420_unorm"},
         {holoscan::ops::HolovizOp::ImageFormat::Y16_U16_V16_3PLANE_422_UNORM,
          "y16_u16_v16_3plane_422_unorm"}}};

/*static*/ nvidia::gxf::Expected<holoscan::ops::HolovizOp::ImageFormat>
HolovizOp::imageFormatFromString(const std::string& string) {
  const auto it = std::find_if(std::cbegin(kImageFormatToStr),
                               std::cend(kImageFormatToStr),
                               [&string](const auto& v) { return v.second == string; });
  if (it != std::cend(kImageFormatToStr)) { return it->first; }

  HOLOSCAN_LOG_ERROR("Unsupported image format '{}'", string);
  return nvidia::gxf::Unexpected{GXF_FAILURE};
}

/*static*/ std::string HolovizOp::imageFormatToString(
    holoscan::ops::HolovizOp::ImageFormat image_format) {
  const auto it = std::find_if(std::cbegin(kImageFormatToStr),
                               std::cend(kImageFormatToStr),
                               [&image_format](const auto& v) { return v.first == image_format; });
  if (it != std::cend(kImageFormatToStr)) { return it->second; }

  return "invalid";
}

/*static*/ const std::array<std::pair<holoscan::ops::HolovizOp::DepthMapRenderMode, std::string>, 3>
    HolovizOp::kDepthMapRenderModeToStr{
        {{holoscan::ops::HolovizOp::DepthMapRenderMode::POINTS, "points"},
         {holoscan::ops::HolovizOp::DepthMapRenderMode::LINES, "lines"},
         {holoscan::ops::HolovizOp::DepthMapRenderMode::TRIANGLES, "triangles"}}};

/*static*/ nvidia::gxf::Expected<holoscan::ops::HolovizOp::DepthMapRenderMode>
HolovizOp::depthMapRenderModeFromString(const std::string& string) {
  const auto it = std::find_if(std::cbegin(kDepthMapRenderModeToStr),
                               std::cend(kDepthMapRenderModeToStr),
                               [&string](const auto& v) { return v.second == string; });
  if (it != std::cend(kDepthMapRenderModeToStr)) { return it->first; }

  HOLOSCAN_LOG_ERROR("Unsupported depth map render mode '{}'", string);
  return nvidia::gxf::Unexpected{GXF_FAILURE};
}

/*static*/ std::string HolovizOp::depthMapRenderModeToString(
    holoscan::ops::HolovizOp::DepthMapRenderMode depth_map_render_mode) {
  const auto it = std::find_if(
      std::cbegin(kDepthMapRenderModeToStr),
      std::cend(kDepthMapRenderModeToStr),
      [&depth_map_render_mode](const auto& v) { return v.first == depth_map_render_mode; });
  if (it != std::cend(kDepthMapRenderModeToStr)) { return it->second; }

  return "invalid";
}

/*static*/ const std::array<std::pair<holoscan::ops::HolovizOp::YuvModelConversion, std::string>, 3>
    HolovizOp::kYuvModelConversionToStr{
        {{holoscan::ops::HolovizOp::YuvModelConversion::YUV_601, "yuv_601"},
         {holoscan::ops::HolovizOp::YuvModelConversion::YUV_709, "yuv_709"},
         {holoscan::ops::HolovizOp::YuvModelConversion::YUV_2020, "yuv_2020"}}};

/*static*/ nvidia::gxf::Expected<holoscan::ops::HolovizOp::YuvModelConversion>
HolovizOp::yuvModelConversionFromString(const std::string& string) {
  const auto it = std::find_if(std::cbegin(kYuvModelConversionToStr),
                               std::cend(kYuvModelConversionToStr),
                               [&string](const auto& v) { return v.second == string; });
  if (it != std::cend(kYuvModelConversionToStr)) { return it->first; }

  HOLOSCAN_LOG_ERROR("Unsupported yuv model conversion '{}'", string);
  return nvidia::gxf::Unexpected{GXF_FAILURE};
}

/*static*/ std::string HolovizOp::yuvModelConversionToString(
    holoscan::ops::HolovizOp::YuvModelConversion yuv_model_conversion) {
  const auto it = std::find_if(
      std::cbegin(kYuvModelConversionToStr),
      std::cend(kYuvModelConversionToStr),
      [&yuv_model_conversion](const auto& v) { return v.first == yuv_model_conversion; });
  if (it != std::cend(kYuvModelConversionToStr)) { return it->second; }

  return "invalid";
}

/*static*/ const std::array<std::pair<holoscan::ops::HolovizOp::YuvRange, std::string>, 2>
    HolovizOp::kYuvRangeToStr{{{holoscan::ops::HolovizOp::YuvRange::ITU_FULL, "itu_full"},
                               {holoscan::ops::HolovizOp::YuvRange::ITU_NARROW, "itu_narrow"}}};

/*static*/ nvidia::gxf::Expected<holoscan::ops::HolovizOp::YuvRange> HolovizOp::yuvRangeFromString(
    const std::string& string) {
  const auto it = std::find_if(std::cbegin(kYuvRangeToStr),
                               std::cend(kYuvRangeToStr),
                               [&string](const auto& v) { return v.second == string; });
  if (it != std::cend(kYuvRangeToStr)) { return it->first; }

  HOLOSCAN_LOG_ERROR("Unsupported yuv range '{}'", string);
  return nvidia::gxf::Unexpected{GXF_FAILURE};
}

/*static*/ std::string HolovizOp::yuvRangeToString(holoscan::ops::HolovizOp::YuvRange yuv_range) {
  const auto it = std::find_if(std::cbegin(kYuvRangeToStr),
                               std::cend(kYuvRangeToStr),
                               [&yuv_range](const auto& v) { return v.first == yuv_range; });
  if (it != std::cend(kYuvRangeToStr)) { return it->second; }

  return "invalid";
}

/*static*/ const std::array<std::pair<holoscan::ops::HolovizOp::ChromaLocation, std::string>, 2>
    HolovizOp::kChromaLoactionToStr{
        {{holoscan::ops::HolovizOp::ChromaLocation::COSITED_EVEN, "cosited_even"},
         {holoscan::ops::HolovizOp::ChromaLocation::MIDPOINT, "midpoint"}}};

/*static*/ nvidia::gxf::Expected<holoscan::ops::HolovizOp::ChromaLocation>
HolovizOp::chromaLocationFromString(const std::string& string) {
  const auto it = std::find_if(std::cbegin(kChromaLoactionToStr),
                               std::cend(kChromaLoactionToStr),
                               [&string](const auto& v) { return v.second == string; });
  if (it != std::cend(kChromaLoactionToStr)) { return it->first; }

  HOLOSCAN_LOG_ERROR("Unsupported chroma location '{}'", string);
  return nvidia::gxf::Unexpected{GXF_FAILURE};
}

/*static*/ std::string HolovizOp::chromaLocationToString(
    holoscan::ops::HolovizOp::ChromaLocation chroma_location) {
  const auto it =
      std::find_if(std::cbegin(kChromaLoactionToStr),
                   std::cend(kChromaLoactionToStr),
                   [&chroma_location](const auto& v) { return v.first == chroma_location; });
  if (it != std::cend(kChromaLoactionToStr)) { return it->second; }

  return "invalid";
}

/*static*/ const std::array<std::pair<holoscan::ops::HolovizOp::ColorSpace, std::string>, 7>
    HolovizOp::kColorSpaceToStr{
        {{holoscan::ops::HolovizOp::ColorSpace::SRGB_NONLINEAR, "srgb_nonlinear"},
         {holoscan::ops::HolovizOp::ColorSpace::EXTENDED_SRGB_LINEAR, "extended_srgb_linear"},
         {holoscan::ops::HolovizOp::ColorSpace::BT2020_LINEAR, "bt2020_linear"},
         {holoscan::ops::HolovizOp::ColorSpace::HDR10_ST2084, "hdr10_st2084"},
         {holoscan::ops::HolovizOp::ColorSpace::PASS_THROUGH, "pass_through"},
         {holoscan::ops::HolovizOp::ColorSpace::BT709_LINEAR, "bt709_linear"},
         {holoscan::ops::HolovizOp::ColorSpace::AUTO, "auto"}}};

/*static*/ nvidia::gxf::Expected<holoscan::ops::HolovizOp::ColorSpace>
HolovizOp::colorSpaceFromString(const std::string& string) {
  const auto it = std::find_if(std::cbegin(HolovizOp::kColorSpaceToStr),
                               std::cend(HolovizOp::kColorSpaceToStr),
                               [&string](const auto& v) { return v.second == string; });
  if (it != std::cend(HolovizOp::kColorSpaceToStr)) { return it->first; }

  HOLOSCAN_LOG_ERROR("Unsupported color space '{}'", string);
  return nvidia::gxf::Unexpected{GXF_FAILURE};
}

/*static*/ std::string HolovizOp::colorSpaceToString(
    holoscan::ops::HolovizOp::ColorSpace color_space) {
  const auto it = std::find_if(std::cbegin(HolovizOp::kColorSpaceToStr),
                               std::cend(HolovizOp::kColorSpaceToStr),
                               [&color_space](const auto& v) { return v.first == color_space; });
  if (it != std::cend(HolovizOp::kColorSpaceToStr)) { return it->second; }

  return "invalid";
}

bool HolovizOp::enable_conditional_port(const std::string& port_name,
                                        bool set_none_condition_on_disabled) {
  bool enable_port = false;

  // Check if the boolean argument with the name "enable_(port_name)" is present.
  const std::string enable_port_name = std::string("enable_") + port_name;
  auto enable_port_arg =
      std::find_if(args().begin(), args().end(), [&enable_port_name](const auto& arg) {
        return (arg.name() == enable_port_name);
      });

  // If present ...
  if (enable_port_arg != args().end()) {
    // ... and with a value ...
    if (enable_port_arg->has_value()) {
      // ...try extracting a boolean value through YAML::Node or generic bool cast
      std::any& any_arg = enable_port_arg->value();
      if (enable_port_arg->arg_type().element_type() == ArgElementType::kYAMLNode) {
        auto& arg_value = std::any_cast<YAML::Node&>(any_arg);
        bool parse_ok = YAML::convert<bool>::decode(arg_value, enable_port);
        if (!parse_ok) {
          HOLOSCAN_LOG_ERROR("Could not parse YAML parameter '{}' as a 'bool' type",
                             enable_port_name);
          enable_port = false;
        }
      } else {
        try {
          enable_port = std::any_cast<bool>(any_arg);
        } catch (const std::bad_any_cast& e) {
          HOLOSCAN_LOG_ERROR(
              "Could not cast parameter '{}' as 'bool': {}", enable_port_name, e.what());
        }
      }
    }
    // If the "enable_(port_name)" argument is present, we remove it so that it won't
    // be passed on further, since we only care about "(port_name)" afterwards.
    args().erase(enable_port_arg);
  }

  // Disable the '(port_name)' argument based on the value of "enable_(port_name)"
  if (!enable_port) { add_arg(Arg(port_name) = static_cast<holoscan::IOSpec*>(nullptr)); }

  // If 'set_none_condition_on_disabled' is true and the port (named by 'port_name') is disabled,
  // insert ConditionType::kNone condition so that its default condition
  // (DownstreamMessageAffordableCondition) is not added during Operator::initialize().
  if (!enable_port && set_none_condition_on_disabled) {
    spec()->outputs()[port_name]->condition(ConditionType::kNone);
  }

  return enable_port;
}

void HolovizOp::read_frame_buffer(InputContext& op_input, OutputContext& op_output,
                                  ExecutionContext& context, bool buffer_input_enabled,
                                  const std::string& buffer_name,
                                  nvidia::gxf::VideoFormat video_format) {
  nvidia::gxf::Expected<nvidia::gxf::Entity> entity = nvidia::gxf::Unexpected{GXF_FAILURE};
  nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>> video_buffer =
      nvidia::gxf::Unexpected{GXF_FAILURE};

  const std::string buffer_input_name = fmt::format("{}_buffer_input", buffer_name);
  const std::string buffer_output_name = fmt::format("{}_buffer_output", buffer_name);

  if (buffer_input_enabled) {
    // check if there is a input buffer given to copy the output into
    auto maybe_render_buffer_input = op_input.receive<gxf::Entity>(buffer_input_name.c_str());
    if (!maybe_render_buffer_input || maybe_render_buffer_input.value().is_null()) {
      std::string err_msg = fmt::format("Operator '{}': No message available at '{}': {}",
                                        name_,
                                        maybe_render_buffer_input.error().what());
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }
    entity = maybe_render_buffer_input.value();

    // Get the empty input buffer
    video_buffer = entity.value().get<nvidia::gxf::VideoBuffer>();
    if (!video_buffer) {
      throw std::runtime_error(
          fmt::format("No video buffer attached to message on '{}'.", buffer_input_name));
    }
  } else {
    entity = holoscan::gxf::Entity::New(&context);
    if (!entity) { throw std::runtime_error("Failed to allocate message for buffer output."); }

    video_buffer = entity.value().add<nvidia::gxf::VideoBuffer>(buffer_output_name.c_str());
    if (!video_buffer) {
      throw std::runtime_error("Failed to allocate the video buffer for buffer output.");
    }

    // if there is no input buffer given, allocate one
    if (!allocator_.get()) {
      throw std::runtime_error("No buffer input specified and no allocator set.");
    }

    // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());

    if (video_format == nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA) {
      video_buffer.value()->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
          width_,
          height_,
          nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
          nvidia::gxf::MemoryStorageType::kDevice,
          allocator.value());
    } else if (video_format == nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F) {
      video_buffer.value()->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F>(
          width_,
          height_,
          nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
          nvidia::gxf::MemoryStorageType::kDevice,
          allocator.value());
    } else {
      throw std::runtime_error(
          fmt::format("Unsupported video format: {}", magic_enum::enum_name(video_format)));
    }

    if (!video_buffer.value()->pointer()) {
      throw std::runtime_error("Failed to allocate render output buffer.");
    }
  }

  const nvidia::gxf::VideoBufferInfo info = video_buffer.value()->video_frame_info();

  if ((info.color_format != video_format)) {
    throw std::runtime_error(fmt::format("Invalid buffer input, expected a {} buffer.",
                                         magic_enum::enum_name(video_format)));
  }

  viz::ImageFormat image_format;
  if (video_format == nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA) {
    image_format = viz::ImageFormat::R8G8B8A8_UNORM;
  } else if (video_format == nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F) {
    image_format = viz::ImageFormat::D32_SFLOAT;
  } else {
    throw std::runtime_error(
        fmt::format("Unsupported video format: {}", magic_enum::enum_name(video_format)));
  }

  // read the framebuffer
  viz::ReadFramebuffer(image_format,
                       width_,
                       height_,
                       video_buffer.value()->size(),
                       reinterpret_cast<CUdeviceptr>(video_buffer.value()->pointer()),
                       size_t(info.color_planes[0].stride));

  // Output the filled render buffer object
  auto result = gxf::Entity(std::move(entity.value()));
  op_output.emit(result, buffer_output_name.c_str());
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
  std::array<float, 4> color{1.F, 1.F, 1.F, 1.F};
  for (size_t index = 0; index < std::min(input_spec.color_.size(), color.size()); ++index) {
    color[index] = input_spec.color_[index];
  }
  viz::Color(color[0], color[1], color[2], color[3]);
  viz::PointSize(input_spec.point_size_);
  viz::LineWidth(input_spec.line_width_);
}

void HolovizOp::render_color_image(const InputSpec& input_spec, BufferInfo& buffer_info) {
  viz::ImageFormat image_format;
  if (input_spec.type_ == InputType::COLOR_LUT) {
    if (buffer_info.components != 1) {
      throw std::runtime_error(
          fmt::format("Expected one channel for tensor '{}' when using lookup table, but got {}",
                      buffer_info.name,
                      buffer_info.components));
    }
    if (lut_.empty()) {
      throw std::runtime_error(fmt::format(
          "Type of tensor '{}' is '{}', but a color lookup table has not been specified",
          buffer_info.name,
          inputTypeToString(input_spec.type_)));
    }

    // when using a LUT, the unorm formats are handled as single channel int formats
    switch (buffer_info.image_format) {
      case ImageFormat::R8_UNORM:
        image_format = viz::ImageFormat::R8_UINT;
        break;
      case ImageFormat::R8_SNORM:
        image_format = viz::ImageFormat::R8_SINT;
        break;
      case ImageFormat::R16_UNORM:
        image_format = viz::ImageFormat::R16_UINT;
        break;
      case ImageFormat::R16_SNORM:
        image_format = viz::ImageFormat::R16_SINT;
        break;
      default:
        image_format = viz::ImageFormat(buffer_info.image_format);
        break;
    }
  } else {
    image_format = viz::ImageFormat(buffer_info.image_format);
  }

  // start an image layer
  viz::BeginImageLayer();
  set_input_spec(input_spec);

  if (input_spec.type_ == InputType::COLOR_LUT) {
    viz::LUT(lut_.size() / 4,
             viz::ImageFormat::R32G32B32A32_SFLOAT,
             lut_.size() * sizeof(float),
             lut_.data());
  }

  switch (buffer_info.yuv_model_conversion) {
    case YuvModelConversion::YUV_601:
      viz::ImageYuvModelConversion(viz::YuvModelConversion::YUV_601);
      break;
    case YuvModelConversion::YUV_709:
      viz::ImageYuvModelConversion(viz::YuvModelConversion::YUV_601);
      break;
    case YuvModelConversion::YUV_2020:
      viz::ImageYuvModelConversion(viz::YuvModelConversion::YUV_2020);
      break;
    default:
      throw std::runtime_error(
          fmt::format("Unhandled yuv model conversion {}",
                      yuvModelConversionToString(buffer_info.yuv_model_conversion)));
  }

  switch (buffer_info.yuv_range) {
    case YuvRange::ITU_FULL:
      viz::ImageYuvRange(viz::YuvRange::ITU_FULL);
      break;
    case YuvRange::ITU_NARROW:
      viz::ImageYuvRange(viz::YuvRange::ITU_NARROW);
      break;
    default:
      throw std::runtime_error(
          fmt::format("Unhandled yuv range {}", yuvRangeToString(buffer_info.yuv_range)));
  }

  viz::ChromaLocation x_chroma_location, y_chroma_location;
  switch (input_spec.x_chroma_location_) {
    case ChromaLocation::COSITED_EVEN:
      x_chroma_location = viz::ChromaLocation::COSITED_EVEN;
      break;
    case ChromaLocation::MIDPOINT:
      x_chroma_location = viz::ChromaLocation::MIDPOINT;
      break;
    default:
      throw std::runtime_error(fmt::format("Unhandled x chroma location {}",
                                           chromaLocationToString(input_spec.x_chroma_location_)));
  }
  switch (input_spec.y_chroma_location_) {
    case ChromaLocation::COSITED_EVEN:
      y_chroma_location = viz::ChromaLocation::COSITED_EVEN;
      break;
    case ChromaLocation::MIDPOINT:
      y_chroma_location = viz::ChromaLocation::MIDPOINT;
      break;
    default:
      throw std::runtime_error(fmt::format("Unhandled y chroma location {}",
                                           chromaLocationToString(input_spec.y_chroma_location_)));
  }
  viz::ImageChromaLocation(x_chroma_location, y_chroma_location);

  viz::ImageComponentMapping(buffer_info.component_swizzle[0],
                             buffer_info.component_swizzle[1],
                             buffer_info.component_swizzle[2],
                             buffer_info.component_swizzle[3]);
  if (buffer_info.storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
    // if it's the device convert to `CUDeviceptr`
    const auto cu_buffer_ptr = reinterpret_cast<CUdeviceptr>(buffer_info.buffer_ptr);
    CUdeviceptr cu_buffer_ptr_plane_1 = 0;
    size_t row_pitch_plane_1 = 0;
    CUdeviceptr cu_buffer_ptr_plane_2 = 0;
    size_t row_pitch_plane_2 = 0;
    if (buffer_info.color_planes.size() >= 2) {
      cu_buffer_ptr_plane_1 = cu_buffer_ptr + buffer_info.color_planes[1].offset;
      row_pitch_plane_1 = buffer_info.color_planes[1].stride;
    }
    if (buffer_info.color_planes.size() >= 3) {
      cu_buffer_ptr_plane_2 = cu_buffer_ptr + buffer_info.color_planes[2].offset;
      row_pitch_plane_2 = buffer_info.color_planes[2].stride;
    }
    viz::ImageCudaDevice(buffer_info.width,
                         buffer_info.height,
                         image_format,
                         cu_buffer_ptr,
                         buffer_info.stride[0],
                         cu_buffer_ptr_plane_1,
                         row_pitch_plane_1,
                         cu_buffer_ptr_plane_2,
                         row_pitch_plane_2);
  } else {
    // convert to void * if using the system/host
    const auto host_buffer_ptr = reinterpret_cast<const void*>(buffer_info.buffer_ptr);
    const void* host_buffer_ptr_plane_1 = nullptr;
    size_t row_pitch_plane_1 = 0;
    const void* host_buffer_ptr_plane_2 = nullptr;
    size_t row_pitch_plane_2 = 0;
    if (buffer_info.color_planes.size() >= 2) {
      host_buffer_ptr_plane_1 = reinterpret_cast<const void*>(uintptr_t(host_buffer_ptr) +
                                                              buffer_info.color_planes[1].offset);
      row_pitch_plane_1 = buffer_info.color_planes[1].stride;
    }
    if (buffer_info.color_planes.size() >= 3) {
      host_buffer_ptr_plane_2 = reinterpret_cast<const void*>(uintptr_t(host_buffer_ptr) +
                                                              buffer_info.color_planes[2].offset);
      row_pitch_plane_2 = buffer_info.color_planes[2].stride;
    }
    viz::ImageHost(buffer_info.width,
                   buffer_info.height,
                   image_format,
                   host_buffer_ptr,
                   buffer_info.stride[0],
                   host_buffer_ptr_plane_1,
                   row_pitch_plane_1,
                   host_buffer_ptr_plane_2,
                   row_pitch_plane_2);
  }
  viz::EndLayer();
}

void HolovizOp::render_geometry(const InputSpec& input_spec, BufferInfo& buffer_info,
                                cudaStream_t stream) {
  if ((buffer_info.element_type != nvidia::gxf::PrimitiveType::kFloat32) &&
      (buffer_info.element_type != nvidia::gxf::PrimitiveType::kFloat64)) {
    throw std::runtime_error(
        fmt::format("Expected gxf::PrimitiveType::kFloat32 or gxf::PrimitiveType::kFloat64 "
                    "element type for coordinates, but got element type {}",
                    static_cast<int>(buffer_info.element_type)));
  }

  // start a geometry layer
  viz::BeginGeometryLayer();
  set_input_spec_geometry(input_spec);

  const auto coordinates = buffer_info.width;

  if (input_spec.type_ == InputType::TEXT) {
    // text is defined by the top left coordinate and the size (x, y, s) per string, text
    // strings are define by InputSpec::text_
    if ((buffer_info.components < 2) || (buffer_info.components > 3)) {
      throw std::runtime_error(fmt::format("Expected two or three values per text, but got '{}'",
                                           buffer_info.components));
    }
    if (input_spec.text_.empty()) {
      throw std::runtime_error(
          fmt::format("No text has been specified by input spec '{}'.", input_spec.tensor_name_));
    }

    uintptr_t src_coord;
    std::vector<nvidia::byte> host_buffer;
    if (buffer_info.storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
      host_buffer.resize(buffer_info.bytes_size);

      // copy from device to host
      HOLOSCAN_CUDA_CALL_THROW_ERROR(
          cudaMemcpyAsync(static_cast<void*>(host_buffer.data()),
                          static_cast<const void*>(buffer_info.buffer_ptr),
                          buffer_info.bytes_size,
                          cudaMemcpyDeviceToHost,
                          stream),
          "Failed to copy coordinates to host");
      // When copying from device memory to pageable memory the call is synchronous with the host
      // execution. No need to synchronize here.

      src_coord = reinterpret_cast<uintptr_t>(host_buffer.data());
    } else {
      src_coord = reinterpret_cast<uintptr_t>(buffer_info.buffer_ptr);
    }
    constexpr uint32_t values_per_coordinate = 3;
    float coords[values_per_coordinate]{0.F, 0.F, 0.05F};
    for (uint32_t index = 0; index < coordinates; ++index) {
      uint32_t component_index = 0;
      // copy from source array
      while (component_index < buffer_info.components) {
        switch (buffer_info.element_type) {
          case nvidia::gxf::PrimitiveType::kFloat32:
            coords[component_index] = reinterpret_cast<const float*>(src_coord)[component_index];
            break;
          case nvidia::gxf::PrimitiveType::kFloat64:
            coords[component_index] = reinterpret_cast<const double*>(src_coord)[component_index];
            break;
          default:
            throw std::runtime_error("Unhandled element type");
        }
        ++component_index;
      }
      src_coord += buffer_info.stride[1];
      viz::Text(
          coords[0],
          coords[1],
          coords[2],
          input_spec.text_[std::min(index, static_cast<uint32_t>(input_spec.text_.size()) - 1)]
              .c_str());
    }
  } else {
    viz::PrimitiveTopology topology;
    uint32_t primitive_count;
    uint32_t coordinate_count;
    uint32_t values_per_coordinate;
    std::vector<float> default_coord;

    switch (input_spec.type_) {
      case InputType::POINTS:
        // point primitives, one coordinate (x, y) per primitive
        if (buffer_info.components != 2) {
          throw std::runtime_error(
              fmt::format("Expected two values per point, but got '{}'", buffer_info.components));
        }
        topology = viz::PrimitiveTopology::POINT_LIST;
        primitive_count = coordinates;
        coordinate_count = primitive_count;
        values_per_coordinate = 2;
        default_coord = {0.F, 0.F};
        break;
      case InputType::LINES:
        // line primitives, two coordinates (x0, y0) and (x1, y1) per primitive
        if (buffer_info.components != 2) {
          throw std::runtime_error(fmt::format("Expected two values per line vertex, but got '{}'",
                                               buffer_info.components));
        }
        topology = viz::PrimitiveTopology::LINE_LIST;
        primitive_count = coordinates / 2;
        coordinate_count = primitive_count * 2;
        values_per_coordinate = 2;
        default_coord = {0.F, 0.F};
        break;
      case InputType::LINE_STRIP:
        // line strip primitive, a line primitive i is defined by each coordinate (xi, yi) and
        // the following (xi+1, yi+1)
        if (buffer_info.components != 2) {
          throw std::runtime_error(fmt::format(
              "Expected two values per line strip vertex, but got '{}'", buffer_info.components));
        }
        topology = viz::PrimitiveTopology::LINE_STRIP;
        primitive_count = coordinates - 1;
        coordinate_count = coordinates;
        values_per_coordinate = 2;
        default_coord = {0.F, 0.F};
        break;
      case InputType::TRIANGLES:
        // triangle primitive, three coordinates (x0, y0), (x1, y1) and (x2, y2) per primitive
        if (buffer_info.components != 2) {
          throw std::runtime_error(fmt::format(
              "Expected two values per triangle vertex, but got '{}'", buffer_info.components));
        }
        topology = viz::PrimitiveTopology::TRIANGLE_LIST;
        primitive_count = coordinates / 3;
        coordinate_count = primitive_count * 3;
        values_per_coordinate = 2;
        default_coord = {0.F, 0.F};
        break;
      case InputType::CROSSES:
        // cross primitive, a cross is defined by the center coordinate and the size (xi, yi,
        // si)
        if ((buffer_info.components < 2) || (buffer_info.components > 3)) {
          throw std::runtime_error(fmt::format(
              "Expected two or three values per cross, but got '{}'", buffer_info.components));
        }

        topology = viz::PrimitiveTopology::CROSS_LIST;
        primitive_count = coordinates;
        coordinate_count = primitive_count;
        values_per_coordinate = 3;
        default_coord = {0.F, 0.F, 0.05F};
        break;
      case InputType::RECTANGLES:
        // axis aligned rectangle primitive, each rectangle is defined by two coordinates (xi,
        // yi) and (xi+1, yi+1)
        if (buffer_info.components != 2) {
          throw std::runtime_error(fmt::format(
              "Expected two values per rectangle vertex, but got '{}'", buffer_info.components));
        }
        topology = viz::PrimitiveTopology::RECTANGLE_LIST;
        primitive_count = coordinates / 2;
        coordinate_count = primitive_count * 2;
        values_per_coordinate = 2;
        default_coord = {0.F, 0.F};
        break;
      case InputType::OVALS:
        // oval primitive, an oval primitive is defined by the center coordinate and the axis
        // sizes (xi, yi, sxi, syi)
        if ((buffer_info.components < 2) || (buffer_info.components > 4)) {
          throw std::runtime_error(fmt::format(
              "Expected two, three or four values per oval, but got '{}'", buffer_info.components));
        }
        topology = viz::PrimitiveTopology::OVAL_LIST;
        primitive_count = coordinates;
        coordinate_count = primitive_count;
        values_per_coordinate = 4;
        default_coord = {0.F, 0.F, 0.05F, 0.05F};
        break;
      case InputType::POINTS_3D:
        // point primitives, one coordinate (x, y, z) per primitive
        if (buffer_info.components != 3) {
          throw std::runtime_error(fmt::format("Expected three values per 3D point, but got '{}'",
                                               buffer_info.components));
        }
        topology = viz::PrimitiveTopology::POINT_LIST_3D;
        primitive_count = coordinates;
        coordinate_count = primitive_count;
        values_per_coordinate = 3;
        default_coord = {0.F, 0.F, 0.F};
        break;
      case InputType::LINES_3D:
        // line primitives, two coordinates (x0, y0, z0) and (x1, y1, z1) per primitive
        if (buffer_info.components != 3) {
          throw std::runtime_error(fmt::format(
              "Expected three values per 3D line vertex, but got '{}'", buffer_info.components));
        }
        topology = viz::PrimitiveTopology::LINE_LIST_3D;
        primitive_count = coordinates / 2;
        coordinate_count = primitive_count * 2;
        values_per_coordinate = 3;
        default_coord = {0.F, 0.F, 0.F};
        break;
      case InputType::LINE_STRIP_3D:
        // line primitives, two coordinates (x0, y0, z0) and (x1, y1, z1) per primitive
        if (buffer_info.components != 3) {
          throw std::runtime_error(
              fmt::format("Expected three values per 3D line strip vertex, but got '{}'",
                          buffer_info.components));
        }
        topology = viz::PrimitiveTopology::LINE_STRIP_3D;
        primitive_count = coordinates - 1;
        coordinate_count = coordinates;
        values_per_coordinate = 3;
        default_coord = {0.F, 0.F};
        break;
      case InputType::TRIANGLES_3D:
        // triangle primitive, three coordinates (x0, y0, z0), (x1, y1, z1) and (x2, y2, z2)
        // per primitive
        if (buffer_info.components != 3) {
          throw std::runtime_error(
              fmt::format("Expected three values per 3D triangle vertex, but got '{}'",
                          buffer_info.components));
        }
        topology = viz::PrimitiveTopology::TRIANGLE_LIST_3D;
        primitive_count = coordinates / 3;
        coordinate_count = primitive_count * 3;
        values_per_coordinate = 3;
        default_coord = {0.F, 0.F};
        break;
      default:
        throw std::runtime_error(
            fmt::format("Unhandled tensor type '{}'", inputTypeToString(input_spec.type_)));
    }

    if (primitive_count) {
      if ((buffer_info.element_type == nvidia::gxf::PrimitiveType::kFloat32) &&
          (buffer_info.components == values_per_coordinate)) {
        // can use the buffer directly, no copy required
        if (buffer_info.storage_type == nvidia::gxf::MemoryStorageType::kSystem) {
          viz::Primitive(topology,
                         primitive_count,
                         coordinate_count * values_per_coordinate,
                         reinterpret_cast<const float*>(buffer_info.buffer_ptr));
        } else {
          viz::PrimitiveCudaDevice(topology,
                                   primitive_count,
                                   coordinate_count * values_per_coordinate,
                                   reinterpret_cast<CUdeviceptr>(buffer_info.buffer_ptr));
        }

      } else {
        // copy coordinates, convert from double to float if needed and add defaults
        uintptr_t src_coord;
        std::vector<nvidia::byte> host_buffer;
        if (buffer_info.storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
          host_buffer.resize(buffer_info.bytes_size);

          // copy from device to host
          HOLOSCAN_CUDA_CALL_THROW_ERROR(
              cudaMemcpyAsync(static_cast<void*>(host_buffer.data()),
                              static_cast<const void*>(buffer_info.buffer_ptr),
                              buffer_info.bytes_size,
                              cudaMemcpyDeviceToHost,
                              stream),
              "Failed to copy coordinates to host");
          // When copying from device memory to pageable memory the call is synchronous with the
          // host execution. No need to synchronize here.

          src_coord = reinterpret_cast<uintptr_t>(host_buffer.data());
        } else {
          src_coord = reinterpret_cast<uintptr_t>(buffer_info.buffer_ptr);
        }

        // copy coordinates
        std::vector<float> coords;
        coords.reserve(coordinate_count * values_per_coordinate);

        for (uint32_t index = 0; index < coordinate_count; ++index) {
          uint32_t component_index = 0;
          // copy from source array
          while (component_index < std::min(buffer_info.components, values_per_coordinate)) {
            switch (buffer_info.element_type) {
              case nvidia::gxf::PrimitiveType::kFloat32:
                coords.push_back(reinterpret_cast<const float*>(src_coord)[component_index]);
                break;
              case nvidia::gxf::PrimitiveType::kFloat64:
                coords.push_back(reinterpret_cast<const double*>(src_coord)[component_index]);
                break;
              default:
                throw std::runtime_error("Unhandled element type");
            }
            ++component_index;
          }
          // fill from default array
          while (component_index < values_per_coordinate) {
            coords.push_back(default_coord[component_index]);
            ++component_index;
          }
          src_coord += buffer_info.stride[1];
        }

        viz::Primitive(topology, primitive_count, coords.size(), coords.data());
      }
    }
  }

  viz::EndLayer();
}

void HolovizOp::render_depth_map(InputSpec* const input_spec_depth_map,
                                 const BufferInfo& buffer_info_depth_map,
                                 InputSpec* const input_spec_depth_map_color,
                                 const BufferInfo& buffer_info_depth_map_color) {
  viz::ImageFormat depth_map_color_fmt = viz::ImageFormat::R8G8B8A8_UNORM;
  CUdeviceptr depth_map_color_device_ptr = 0;
  if (input_spec_depth_map_color) {
    // if there is a color buffer, the size has to match
    if ((buffer_info_depth_map.width != buffer_info_depth_map_color.width) ||
        (buffer_info_depth_map.height != buffer_info_depth_map_color.height)) {
      throw std::runtime_error(
          fmt::format("The buffer dimensions {}x{} of the depth map color buffer '{}' need to "
                      "match the depth map '{}' dimensions {}x{}",
                      buffer_info_depth_map_color.width,
                      buffer_info_depth_map_color.height,
                      buffer_info_depth_map_color.name,
                      buffer_info_depth_map.name,
                      buffer_info_depth_map.width,
                      buffer_info_depth_map.height));
    }
    depth_map_color_fmt = viz::ImageFormat(buffer_info_depth_map_color.image_format);

    depth_map_color_device_ptr =
        reinterpret_cast<CUdeviceptr>(buffer_info_depth_map_color.buffer_ptr);
  }

  viz::ImageFormat depth_format;
  switch (buffer_info_depth_map.image_format) {
    case ImageFormat::R8_UNORM:
      depth_format = viz::ImageFormat::R8_UNORM;
      break;
    case ImageFormat::R32_SFLOAT:
      // if the input is a tensor the image format is auto detected as a single channel color
      // format, convert to a depth format
      depth_format = viz::ImageFormat::D32_SFLOAT;
      break;
    case ImageFormat::D32_SFLOAT:
      depth_format = viz::ImageFormat::D32_SFLOAT;
      break;
    default:
      throw std::runtime_error(fmt::format("Depth map: depth image format {} not supported",
                                           int(buffer_info_depth_map.image_format)));
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
                buffer_info_depth_map.width,
                buffer_info_depth_map.height,
                depth_format,
                cu_buffer_ptr,
                depth_map_color_fmt,
                depth_map_color_device_ptr);
  viz::EndLayer();
}

void HolovizOp::initialize() {
  register_converter<std::vector<InputSpec>>();
  register_converter<std::array<float, 3>>();
  register_converter<ColorSpace>();
  register_converter<KeyCallbackFunction>();
  register_converter<UnicodeCharCallbackFunction>();
  register_converter<MouseButtonCallbackFunction>();
  register_converter<ScrollCallbackFunction>();
  register_converter<CursorPosCallbackFunction>();
  register_converter<FramebufferSizeCallbackFunction>();
  register_converter<WindowSizeCallbackFunction>();
  register_converter<LayerCallbackFunction>();
  holoscan::gxf::GXFExecutor::register_codec<std::vector<InputSpec>>(
      "std::vector<holoscan::ops::HolovizOp::InputSpec>", true);

  // Set up prerequisite parameters before calling Operator::initialize()
  auto frag = fragment();

  // Find if there is an argument for the deprecated 'window_close_scheduling_term' name or
  // the newer 'window_close_condition' name.
  auto window_scheduling_term_iter =
      std::find_if(args().begin(), args().end(), [](const auto& arg) {
        return (arg.name() == "window_close_scheduling_term");
      });
  auto window_condition_iter = std::find_if(args().begin(), args().end(), [](const auto& arg) {
    return (arg.name() == "window_close_condition");
  });
  bool has_window_close_scheduling_term = window_scheduling_term_iter != args().end();
  bool has_window_close_condition = window_condition_iter != args().end();
  if (has_window_close_scheduling_term) {
    if (has_window_close_condition) {
      HOLOSCAN_LOG_WARN(
          "Both \"window_close_scheduling_term\" and \"window_close_condition\" arguments "
          "were provided. Please provide only \"window_close_condition\". Now discarding the "
          "duplicate \"window_close_scheduling_term\" argument.");
      // remove the duplicate argument using the deprecated name
      args().erase(window_scheduling_term_iter);
    } else {
      HOLOSCAN_LOG_WARN(
          "An argument named \"window_close_scheduling_term\" was provided, but this parameter "
          "name is deprecated. Please provide this argument via its new name, "
          "\"window_close_condition\", instead. Now renaming the argument to "
          "\"window_close_condition\".");

      // rename the existing argument in-place
      std::string new_name{"window_close_condition"};
      window_scheduling_term_iter->name(new_name);
    }
    // in either case above, we now have only "window_close_condition"
    has_window_close_condition = true;
  }

  // Create the BooleanCondition if there was no window close argument provided.
  if (!has_window_close_condition) {
    window_close_condition_ =
        frag->make_condition<holoscan::BooleanCondition>("window_close_condition");
    add_arg(window_close_condition_.get());
  }

  // Conditional inputs and outputs are enabled using a boolean argument
  render_buffer_input_enabled_ = enable_conditional_port("render_buffer_input");
  render_buffer_output_enabled_ = enable_conditional_port("render_buffer_output", true);
  depth_buffer_input_enabled_ = enable_conditional_port("depth_buffer_input");
  depth_buffer_output_enabled_ = enable_conditional_port("depth_buffer_output", true);
  camera_pose_output_enabled_ = enable_conditional_port("camera_pose_output", true);

  // parent class initialize() call must be after the argument additions above
  Operator::initialize();
}

void HolovizOp::start() {
  static const char* mutex_env_str = std::getenv("HOLOSCAN_HOLOVIZ_MUTEX");
  if (mutex_env_str) {
    // Check if HOLOSCAN_HOLOVIZ_MUTEX is a non-zero value or not
    this->is_holoviz_multiprocess_mutex_enabled_ = mutex_env_str[0] == '0' ? false : true;
  } else {
    this->is_holoviz_multiprocess_mutex_enabled_ = false;
  }

  if (is_holoviz_multiprocess_mutex_enabled_ && !holoviz_multiprocess_mutex_) {
    holoviz_multiprocess_mutex_ =
        std::make_shared<holoscan::FileFIFOMutex>(holoviz_multiprocess_mutex_path_.get());
    holoviz_multiprocess_mutex_->set_wait_time_ms(multiprocess_framedrop_waittime_ms_);
  }

  // set the font to be used
  if (!font_path_.get().empty()) { viz::SetFont(font_path_.get().c_str(), 25.F); }

  // create Holoviz instance
  instance_ = viz::Create();
  // make the instance current
  ScopedPushInstance scoped_instance(instance_);

  if (vsync_) { viz::SetPresentMode(viz::PresentMode::FIFO); }

  // initialize Holoviz
  viz::InitFlags init_flags = viz::InitFlags::NONE;
  if (fullscreen_ && headless_) {
    throw std::runtime_error("Headless and fullscreen are mutually exclusive.");
  }
  if (fullscreen_) { init_flags = viz::InitFlags::FULLSCREEN; }
  if (headless_) { init_flags = viz::InitFlags::HEADLESS; }

  if (use_exclusive_display_) {
    // In exclusive mode, we don't use Holoviz mutex for initialization.
    viz::Init(
        display_name_.get().c_str(), width_, height_, uint32_t(framerate_ * 1000.F), init_flags);
  } else {
    ScopedHolovizFileMutex scoped_mutex(is_holoviz_multiprocess_mutex_enabled_,
                                        holoviz_multiprocess_mutex_);
    // At initialization, we do not consider the framedrop option as
    // initialization must happen in every case. Therefore, we do our best to issue a
    // mutually exclusive initialization.
    viz::Init(width_,
              height_,
              window_title_.get().c_str(),
              init_flags,
              display_name_.get().empty() ? nullptr : display_name_.get().c_str());
  }

  if (framebuffer_srgb_.get() || (display_color_space_.get() != ColorSpace::AUTO)) {
    // If the SRGB framebuffer is enabled or a display color space is requested, get the supported
    // surface formats and look for a format supporting the requirements.
    uint32_t surface_format_count = 0;
    viz::GetSurfaceFormats(&surface_format_count, nullptr);
    std::vector<viz::SurfaceFormat> surface_formats(surface_format_count);
    viz::GetSurfaceFormats(&surface_format_count, surface_formats.data());

    viz::ColorSpace color_space = viz::ColorSpace(display_color_space_.get());
    bool found = false;
    for (auto surface_format_it = surface_formats.begin();
         surface_format_it != surface_formats.end();
         ++surface_format_it) {
      if (framebuffer_srgb_.get() ==
          ((surface_format_it->image_format_ == viz::ImageFormat::R8G8B8A8_SRGB) ||
           (surface_format_it->image_format_ == viz::ImageFormat::B8G8R8A8_SRGB) ||
           (surface_format_it->image_format_ == viz::ImageFormat::A8B8G8R8_SRGB_PACK32))) {
        if (display_color_space_.get() == ColorSpace::AUTO) {
          // Ignore the color space, it might be `SRGB_NONLINEAR` if a display is connected or
          // `PASS_THROUGH` in headless mode.
        } else if (surface_format_it->color_space_ != color_space) {
          // check the next format for the required color space
          continue;
        }

        viz::SetSurfaceFormat(*surface_format_it);
        found = true;
        break;
      }
    }

    if (!found) {
      throw std::runtime_error(fmt::format(
          "No framebuffer format found which supports the requirements {} {}.",
          framebuffer_srgb_ ? "srgb" : "",
          display_color_space_.has_value() ? colorSpaceToString(display_color_space_.get()) : ""));
    }
  }

  // get the supported image formats
  uint32_t image_format_count = 0;
  viz::GetImageFormats(&image_format_count, nullptr);
  supported_image_formats_.resize(image_format_count);
  viz::GetImageFormats(
      &image_format_count,
      reinterpret_cast<holoscan::viz::ImageFormat*>(supported_image_formats_.data()));

  // initialize the camera with the provided parameters
  camera_eye_cur_ = camera_eye_.get();
  camera_look_at_cur_ = camera_look_at_.get();
  camera_up_cur_ = camera_up_.get();
  viz::SetCamera(camera_eye_cur_[0],
                 camera_eye_cur_[1],
                 camera_eye_cur_[2],
                 camera_look_at_cur_[0],
                 camera_look_at_cur_[1],
                 camera_look_at_cur_[2],
                 camera_up_cur_[0],
                 camera_up_cur_[1],
                 camera_up_cur_[2],
                 false);

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
  window_close_condition_->enable_tick();

  // Copy the user defined input spec list to the internal input spec list. If there is no user
  // defined input spec it will be generated from the first messages received.
  if (!tensors_.get().empty()) {
    initial_input_spec_.reserve(tensors_.get().size());
    initial_input_spec_.insert(
        initial_input_spec_.begin(), tensors_.get().begin(), tensors_.get().end());
  }

  // setup callbacks
  if (key_callback_.has_value()) { viz::SetKeyCallback(this, key_callback_handler); }
  if (unicode_char_callback_.has_value()) {
    viz::SetUnicodeCharCallback(this, unicode_char_callback_handler);
  }
  if (mouse_button_callback_.has_value()) {
    viz::SetMouseButtonCallback(this, mouse_button_callback_handler);
  }
  if (scroll_callback_.has_value()) { viz::SetScrollCallback(this, scroll_callback_handler); }
  if (cursor_pos_callback_.has_value()) {
    viz::SetCursorPosCallback(this, cursor_pos_callback_handler);
  }
  if (framebuffer_size_callback_.has_value()) {
    viz::SetFramebufferSizeCallback(this, framebuffer_size_callback_handler);
  }
  if (window_size_callback_.has_value()) {
    viz::SetWindowSizeCallback(this, window_size_callback_handler);
  }
}

void HolovizOp::stop() {
  if (instance_) { viz::Shutdown(instance_); }
  if (is_holoviz_multiprocess_mutex_enabled_ && multiprocess_framedrop_waittime_ms_) {
    HOLOSCAN_LOG_INFO("Dropped {} frames due to unavailable mutex in multiprocess scenario.",
                      dropped_frame_count_);
  }
  holoviz_multiprocess_mutex_.reset();  // early cleanup
}

void HolovizOp::disable_via_window_close() {
  const auto& fragment_graph = fragment()->application()->fragment_graph();

  bool is_distributed = !fragment_graph.is_empty();
  if (is_distributed) {
    HOLOSCAN_LOG_WARN("Initiating distributed app shutdown from HolovizOp");
    // Initiate shutdown via RPC
    auto app = fragment()->application();
    app->initiate_distributed_app_shutdown(fragment()->name());
  }
  window_close_condition_->disable_tick();
  return;
}

void HolovizOp::compute(InputContext& op_input, OutputContext& op_output,
                        ExecutionContext& context) {
  // receive input messages
  auto maybe_receivers_messages = op_input.receive<std::vector<gxf::Entity>>("receivers");
  if (!maybe_receivers_messages || maybe_receivers_messages->empty()) {
    std::string err_msg = fmt::format(
        "No input messages received for op '{}' on port 'receivers': {}",
        name_,
        !maybe_receivers_messages.has_value() ? maybe_receivers_messages.error().what() : "empty");
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  const auto receivers_messages = maybe_receivers_messages.value();

  const auto input_specs_messages =
      op_input.receive<std::vector<holoscan::ops::HolovizOp::InputSpec>>("input_specs");

  const auto camera_eye_message = op_input.receive<std::array<float, 3>>("camera_eye_input");
  const auto camera_look_at_message =
      op_input.receive<std::array<float, 3>>("camera_look_at_input");
  const auto camera_up_message = op_input.receive<std::array<float, 3>>("camera_up_input");

  // make instance current
  ScopedPushInstance scoped_instance(instance_);

  // cast Condition to BooleanCondition
  if (viz::WindowShouldClose()) { disable_via_window_close(); }

  // nothing to do if minimized
  if (viz::WindowIsMinimized()) { return; }

  // Right near the window minimization option, check if we need to lock and whether
  // the mutex is available.
  ScopedHolovizFileMutex scoped_mutex(is_holoviz_multiprocess_mutex_enabled_,
                                      holoviz_multiprocess_mutex_);
  if (is_holoviz_multiprocess_mutex_enabled_ && multiprocess_framedrop_waittime_ms_ &&
      !scoped_mutex.locked()) {
    HOLOSCAN_LOG_DEBUG("Frame dropped due to unavailable mutex in multiprocess scenario.");
    dropped_frame_count_++;
    return;
  }

  // handle camera messages
  if (camera_eye_message || camera_eye_message || camera_up_message) {
    if (camera_eye_message) { camera_eye_cur_ = camera_eye_message.value(); }
    if (camera_look_at_message) { camera_look_at_cur_ = camera_look_at_message.value(); }
    if (camera_up_message) { camera_up_cur_ = camera_up_message.value(); }
    // set the camera
    viz::SetCamera(camera_eye_cur_[0],
                   camera_eye_cur_[1],
                   camera_eye_cur_[2],
                   camera_look_at_cur_[0],
                   camera_look_at_cur_[1],
                   camera_look_at_cur_[2],
                   camera_up_cur_[0],
                   camera_up_cur_[1],
                   camera_up_cur_[2],
                   true);
  }

  // build the input spec list
  std::vector<InputSpec> input_spec_list(initial_input_spec_);

  // check the messages for input specs, they are added to the list
  if (input_specs_messages) {
    input_spec_list.insert(
        input_spec_list.end(), input_specs_messages->begin(), input_specs_messages->end());
  }

  // then get all tensors and video buffers of all messages, check if an input spec for the tensor
  // is already there, if not try to detect the input spec from the tensor or video buffer
  // information
  for (auto&& message : receivers_messages) {
    const auto tensors = message.nvidia::gxf::Entity::findAll<nvidia::gxf::Tensor>();
    for (auto&& tensor : tensors.value()) {
      // check if an input spec with the same tensor name already exist
      const std::string tensor_name(tensor->name());
      const auto it = std::find_if(
          std::begin(input_spec_list), std::end(input_spec_list), [&tensor_name](const auto& v) {
            return v.tensor_name_ == tensor_name;
          });

      if (it == std::end(input_spec_list)) {
        // no input spec found, try to detect the input type. If we can't detect it, ignore the
        // tensor
        bool detected = false;
        BufferInfo buffer_info;
        if (buffer_info.init(tensor.value()) == GXF_SUCCESS) {
          const auto maybe_input_type = detect_input_type(buffer_info, !lut_.empty());
          if (maybe_input_type) {
            input_spec_list.emplace_back(tensor->name(), maybe_input_type.value());
            detected = true;
          }
        }
        if (!detected) {
          HOLOSCAN_LOG_ERROR(
              "Ignoring {}",
              get_format_str(
                  tensor->name(), *tensor, nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE}));
        }
      }
    }

    const auto video_buffers = message.findAllHeap<nvidia::gxf::VideoBuffer>();
    for (auto&& video_buffer : video_buffers.value()) {
      // check if an input spec with the same tensor name already exist
      const std::string tensor_name(video_buffer->name());
      const auto it = std::find_if(
          std::begin(input_spec_list), std::end(input_spec_list), [&tensor_name](const auto& v) {
            return v.tensor_name_ == tensor_name;
          });

      if (it == std::end(input_spec_list)) {
        // no input spec found, try to detect the input type. If we can't detect it, ignore the
        // video buffer
        bool detected = false;
        BufferInfo buffer_info;
        if (buffer_info.init(video_buffer.value()) == GXF_SUCCESS) {
          const auto maybe_input_type = detect_input_type(buffer_info, !lut_.empty());
          if (maybe_input_type) {
            input_spec_list.emplace_back(video_buffer->name(), maybe_input_type.value());
            detected = true;
          }
        }
        if (!detected) {
          HOLOSCAN_LOG_ERROR("Ignoring {}",
                             get_format_str(video_buffer->name(),
                                            nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                            *video_buffer));
        }
      }
    }
  }

  // get the CUDA stream from the input message
  cudaStream_t cuda_stream = cudaStreamDefault;
  // `receive_cuda_stream` returns a new stream if there is a cuda stream pool or the first input
  // stream. If there are multiple streams, the streams are synchronized to the returned stream.
  if (receivers_messages.size() > 0) { cuda_stream = op_input.receive_cuda_stream("receivers"); }

  viz::SetCudaStream(cuda_stream);

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
    auto message = receivers_messages.begin();
    while (message != receivers_messages.end()) {
      maybe_input_tensor =
          message->nvidia::gxf::Entity::get<nvidia::gxf::Tensor>(input_spec.tensor_name_.c_str());
      if (maybe_input_tensor) {
        // pick the first one with that name
        break;
      }

      // check for video if no tensor found
      maybe_input_video = message->nvidia::gxf::Entity::get<nvidia::gxf::VideoBuffer>(
          input_spec.tensor_name_.c_str());
      if (maybe_input_video) {  // pick the first one with that name
        break;
      }
      ++message;
    }
    if (message == receivers_messages.end()) {
      throw std::runtime_error(
          fmt::format("Failed to retrieve input '{}'", input_spec.tensor_name_));
    }

    // initialize buffer info
    BufferInfo buffer_info;
    gxf_result_t result = GXF_SUCCESS;
    if (maybe_input_tensor) {
      result = buffer_info.init(maybe_input_tensor.value(), input_spec.image_format_);
    } else {
      result = buffer_info.init(maybe_input_video.value(), input_spec.image_format_);
    }
    if (result == GXF_SUCCESS) {
      // update the input spec image format when auto detecting and we detected a format so the
      // user can see the detected format in the console
      if ((input_spec.image_format_ == ImageFormat::AUTO_DETECT) &&
          (buffer_info.image_format != ImageFormat::AUTO_DETECT)) {
        input_spec.image_format_ = buffer_info.image_format;
      }
    }

    // if the input type is unknown it now can be detected using the image format
    if (input_spec.type_ == InputType::UNKNOWN) {
      const auto maybe_input_type = detect_input_type(buffer_info, !lut_.empty());
      if (maybe_input_type) { input_spec.type_ = maybe_input_type.value(); }
    }

    // if
    // - building buffer info failed
    // - or input type is unknown
    // - or the image format is not set or not supported for input type requiring an image format
    // log the format and the supported formats and throw an error.
    if ((result != GXF_SUCCESS) || (input_spec.type_ == InputType::UNKNOWN) ||
        (((buffer_info.image_format == ImageFormat::AUTO_DETECT) ||
          (std::find(supported_image_formats_.begin(),
                     supported_image_formats_.end(),
                     buffer_info.image_format) == supported_image_formats_.end())) &&
         (((input_spec.type_ == InputType::COLOR) || (input_spec.type_ == InputType::COLOR_LUT) ||
           (input_spec.type_ == InputType::DEPTH_MAP) ||
           (input_spec.type_ == InputType::DEPTH_MAP_COLOR))))) {
      HOLOSCAN_LOG_ERROR(
          "Format is not supported:\n{}\n\nSupported formats:\n{}",
          get_format_str(input_spec.tensor_name_, maybe_input_tensor, maybe_input_video),
          maybe_input_tensor
              ? BufferInfo::get_supported_tensor_formats_str(supported_image_formats_)
              : BufferInfo::get_supported_video_buffer_formats_str(supported_image_formats_));
      throw std::runtime_error(fmt::format("Unsupported input `{}`", input_spec.tensor_name_));
    }

    // If the buffer is empty, skip processing it
    if (buffer_info.bytes_size == 0) { continue; }

    switch (input_spec.type_) {
      case InputType::COLOR:
      case InputType::COLOR_LUT:
        // 2D color image
        render_color_image(input_spec, buffer_info);
        break;

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
      case InputType::TRIANGLES_3D:
        // geometry layer
        render_geometry(input_spec, buffer_info, cuda_stream);
        break;

      case InputType::DEPTH_MAP: {
        // 2D depth map
        if ((buffer_info.element_type != nvidia::gxf::PrimitiveType::kFloat32) &&
            (buffer_info.element_type != nvidia::gxf::PrimitiveType::kUnsigned8)) {
          throw std::runtime_error(
              fmt::format("Expected gxf::PrimitiveType::kUnsigned8 or gxf::PrimitiveType::kFloat32 "
                          "element type for tensor '{}', but got "
                          "element type {}",
                          buffer_info.name,
                          static_cast<int>(buffer_info.element_type)));
        }
        if (buffer_info.storage_type != nvidia::gxf::MemoryStorageType::kDevice) {
          throw std::runtime_error(
              fmt::format("Only device storage is supported for tensor '{}'", buffer_info.name));
        }
        if (buffer_info.components != 1) {
          throw std::runtime_error(fmt::format("Expected one channel for tensor '{}', but got {}",
                                               buffer_info.name,
                                               buffer_info.components));
        }

        // Store the depth map information, we render after the end of the input spec loop when
        // we also have the (optional) depth map color information.
        input_spec_depth_map = &input_spec;
        buffer_info_depth_map = buffer_info;
      } break;
      case InputType::DEPTH_MAP_COLOR: {
        // 2D depth map color
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
        if (buffer_info.components != 4) {
          throw std::runtime_error(fmt::format("Expected four channels for tensor '{}', but got {}",
                                               buffer_info.name,
                                               buffer_info.components));
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

  // call the layer callback when there is one specified
  if (layer_callback_.has_value()) { layer_callback_.get()(receivers_messages); }

  viz::End();

  // check if the render buffer should be output
  if (render_buffer_output_enabled_) {
    read_frame_buffer(op_input,
                      op_output,
                      context,
                      render_buffer_input_enabled_,
                      "render",
                      nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA);
  }
  if (depth_buffer_output_enabled_) {
    read_frame_buffer(op_input,
                      op_output,
                      context,
                      depth_buffer_input_enabled_,
                      "depth",
                      nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_D32F);
  }

  // check if the the camera pose should be output
  if (camera_pose_output_enabled_) {
    if (camera_pose_output_type_.get() == "projection_matrix") {
      auto camera_pose_output = std::make_shared<std::array<float, 16>>();

      viz::GetCameraPose(camera_pose_output->size(), camera_pose_output->data());

      op_output.emit(camera_pose_output, "camera_pose_output");
    } else if (camera_pose_output_type_.get() == "extrinsics_model") {
      float rotation[9];
      float translation[3];

      viz::GetCameraPose(rotation, translation);

      auto pose = std::make_shared<nvidia::gxf::Pose3D>();
      std::copy(std::begin(rotation), std::end(rotation), std::begin(pose->rotation));
      std::copy(std::begin(translation), std::end(translation), std::begin(pose->translation));
      op_output.emit(pose, "camera_pose_output");
    } else {
      throw std::runtime_error(fmt::format("Unhandled camera pose output type type '{}'",
                                           camera_pose_output_type_.get()));
    }
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
