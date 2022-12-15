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


#include "holoviz.hpp"

#include <cuda_runtime.h>

#include <cinttypes>
#include <list>
#include <utility>

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"

#include "holoviz/holoviz.hpp"

#define CUDA_TRY(stmt)                                                                     \
  ({                                                                                       \
    cudaError_t _holoscan_cuda_err = stmt;                                                 \
    if (cudaSuccess != _holoscan_cuda_err) {                                               \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", \
                    #stmt,                                                                 \
                    __LINE__,                                                              \
                    __FILE__,                                                              \
                    cudaGetErrorString(_holoscan_cuda_err),                                \
                    _holoscan_cuda_err);                                                   \
    }                                                                                      \
    _holoscan_cuda_err;                                                                    \
  })

namespace viz = holoscan::viz;

namespace {

/**
 * Check is the message contains a cuda stream, if yes, set the Holoviz cuda stream with that
 * stream.
 *
 * @param context current GXF context
 * @param message GXF message to check for a cuda stream
 */
void setCudaStreamFromMessage(gxf_context_t context, const nvidia::gxf::Entity& message) {
  // check if the message contains a Cuda stream
  const auto maybe_cuda_stream_id = message.get<nvidia::gxf::CudaStreamId>();
  if (maybe_cuda_stream_id) {
    const auto maybe_cuda_stream_handle = nvidia::gxf::Handle<nvidia::gxf::CudaStream>::Create(
        context, maybe_cuda_stream_id.value()->stream_cid);
    if (maybe_cuda_stream_handle) {
      const auto cuda_stream = maybe_cuda_stream_handle.value();
      if (cuda_stream) { viz::SetCudaStream(cuda_stream->stream().value()); }
    }
  }
}

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
        GXF_LOG_ERROR("Unsupported input format: %" PRId64 "\n",
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

/**
 * Get the Holoviz image format for a given buffer.
 *
 * @param buffer_info buffer info
 * @return Holoviz image format
 */
nvidia::gxf::Expected<viz::ImageFormat> getImageFormatFromTensor(const BufferInfo& buffer_info) {
  if (buffer_info.rank != 3) {
    GXF_LOG_ERROR("Invalid tensor rank count, expected 3, got %u", buffer_info.rank);
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

  viz::ImageFormat image_format = (viz::ImageFormat)-1;
  for (auto&& format : kGFXToHolovizFormats) {
    if ((format.type_ == buffer_info.element_type) &&
        (format.channels_ == buffer_info.shape.dimension(2))) {
      image_format = format.format_;
      break;
    }
  }
  if (image_format == (viz::ImageFormat)-1) {
    GXF_LOG_ERROR("Element type %d and channel count %d not supported",
                  static_cast<int>(buffer_info.element_type),
                  buffer_info.shape.dimension(3));
    return nvidia::gxf::Unexpected{GXF_INVALID_DATA_FORMAT};
  }

  return image_format;
}

/// table to convert input type to string
static const std::array<std::pair<nvidia::holoscan::Holoviz::InputType, std::string>, 11>
    kInputTypeToStr{{{nvidia::holoscan::Holoviz::InputType::UNKNOWN, "unknown"},
                     {nvidia::holoscan::Holoviz::InputType::COLOR, "color"},
                     {nvidia::holoscan::Holoviz::InputType::COLOR_LUT, "color_lut"},
                     {nvidia::holoscan::Holoviz::InputType::POINTS, "points"},
                     {nvidia::holoscan::Holoviz::InputType::LINES, "lines"},
                     {nvidia::holoscan::Holoviz::InputType::LINE_STRIP, "line_strip"},
                     {nvidia::holoscan::Holoviz::InputType::TRIANGLES, "triangles"},
                     {nvidia::holoscan::Holoviz::InputType::CROSSES, "crosses"},
                     {nvidia::holoscan::Holoviz::InputType::RECTANGLES, "rectangles"},
                     {nvidia::holoscan::Holoviz::InputType::OVALS, "ovals"},
                     {nvidia::holoscan::Holoviz::InputType::TEXT, "text"}}};

/**
 * Convert a string to a input type enum
 *
 * @param string input type string
 * @return input type enum
 */
static nvidia::gxf::Expected<nvidia::holoscan::Holoviz::InputType> inputTypeFromString(
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
static std::string inputTypeToString(nvidia::holoscan::Holoviz::InputType input_type) {
  const auto it = std::find_if(std::cbegin(kInputTypeToStr),
                               std::cend(kInputTypeToStr),
                               [&input_type](const auto& v) { return v.first == input_type; });
  if (it != std::cend(kInputTypeToStr)) { return it->second; }

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
nvidia::gxf::Expected<nvidia::holoscan::Holoviz::InputType> detectInputType(
    const BufferInfo& buffer_info, bool has_lut) {
  // auto detect type
  if (buffer_info.rank == 3) {
    if ((buffer_info.shape.dimension(2) == 2) && (buffer_info.shape.dimension(0) == 1) &&
        (buffer_info.element_type == nvidia::gxf::PrimitiveType::kFloat32)) {
      // array of 2D coordinates, draw crosses
      return nvidia::holoscan::Holoviz::InputType::CROSSES;
    } else if ((buffer_info.shape.dimension(2) == 1) && has_lut) {
      // color image with lookup table
      return nvidia::holoscan::Holoviz::InputType::COLOR_LUT;
    } else if ((buffer_info.shape.dimension(2) == 3) || (buffer_info.shape.dimension(2) == 4)) {
      // color image (RGB or RGBA)
      return nvidia::holoscan::Holoviz::InputType::COLOR;
    } else {
      GXF_LOG_ERROR("Can't auto detect type of input %s", buffer_info.name.c_str());
    }
  }
  return nvidia::gxf::Unexpected{GXF_FAILURE};
}

/**
 * Log the input spec
 *
 * @param input_specs input spec to log
 */
void logInputSpec(const std::vector<nvidia::holoscan::Holoviz::InputSpec>& input_specs) {
  std::stringstream ss;
  ss << "Input spec:" << std::endl;
  for (auto&& input_spec : input_specs) {
    ss << "- name: '" << input_spec.tensor_name_ << "'" << std::endl;
    ss << "   type: '" << inputTypeToString(input_spec.type_) << "'" << std::endl;
    ss << "   opacity: " << input_spec.opacity_ << std::endl;
    ss << "   priority: " << input_spec.priority_ << std::endl;
    if ((input_spec.type_ == nvidia::holoscan::Holoviz::InputType::POINTS) ||
        (input_spec.type_ == nvidia::holoscan::Holoviz::InputType::LINES) ||
        (input_spec.type_ == nvidia::holoscan::Holoviz::InputType::LINE_STRIP) ||
        (input_spec.type_ == nvidia::holoscan::Holoviz::InputType::TRIANGLES) ||
        (input_spec.type_ == nvidia::holoscan::Holoviz::InputType::CROSSES) ||
        (input_spec.type_ == nvidia::holoscan::Holoviz::InputType::RECTANGLES) ||
        (input_spec.type_ == nvidia::holoscan::Holoviz::InputType::OVALS) ||
        (input_spec.type_ == nvidia::holoscan::Holoviz::InputType::TEXT)) {
      ss << "   color: [";
      for (auto it = input_spec.color_.cbegin(); it < input_spec.color_.cend(); ++it) {
        ss << *it;
        if (it + 1 != input_spec.color_.cend()) { ss << ", "; }
      }
      ss << "]" << std::endl;
      ss << "   line_width: " << input_spec.line_width_ << std::endl;
      ss << "   point_size: " << input_spec.point_size_ << std::endl;
      ss << "   text: [";
      for (auto it = input_spec.text_.cbegin(); it < input_spec.text_.cend(); ++it) {
        ss << *it;
        if (it + 1 != input_spec.text_.cend()) { ss << ", "; }
      }
      ss << "]" << std::endl;
    }
  }
  GXF_LOG_INFO(ss.str().c_str());
}

}  // namespace

/**
 * Custom YAML parser for InputSpec class
 */
template <>
struct YAML::convert<nvidia::holoscan::Holoviz::InputSpec> {
  static Node encode(const nvidia::holoscan::Holoviz::InputSpec& input_spec) {
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

  static bool decode(const Node& node, nvidia::holoscan::Holoviz::InputSpec& input_spec) {
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

namespace nvidia::holoscan {

constexpr uint32_t kDefaultWidth = 1920;
constexpr uint32_t kDefaultHeight = 1080;
constexpr uint32_t kDefaultFramerate = 60;
const std::string kDefaultWindowTitle = "Holoviz";  // NOLINT
const std::string kDefaultDisplayName = "DP-0";  // NOLINT
constexpr bool kDefaultExclusiveDisplay = false;
constexpr bool kDefaultFullscreen = false;
constexpr bool kDefaultHeadless = false;

gxf_result_t Holoviz::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &=
      registrar->parameter(receivers_, "receivers", "Input Receivers", "List of input receivers.");

  result &= registrar->parameter(render_buffer_input_,
                                 "render_buffer_input",
                                 "RenderBufferInput",
                                 "Input for an empty render buffer.",
                                 gxf::Registrar::NoDefaultParameter(),
                                 GXF_PARAMETER_FLAGS_OPTIONAL);
  result &=
      registrar->parameter(render_buffer_output_,
                           "render_buffer_output",
                           "RenderBufferOutput",
                           "Output for a filled render buffer. If an input render buffer is "
                           "specified it is using that one, otherwise it allocates a new buffer.",
                           gxf::Registrar::NoDefaultParameter(),
                           GXF_PARAMETER_FLAGS_OPTIONAL);

  result &= registrar->parameter(
      tensors_,
      "tensors",
      "Input Tensors",
      "List of input tensors. 'name' is required, 'type' is optional (unknown, color, color_lut, "
      "points, lines, line_strip, triangles, crosses, rectangles, ovals, text).",
      std::vector<InputSpec>());

  result &= registrar->parameter(color_lut_,
                                 "color_lut",
                                 "ColorLUT",
                                 "Color lookup table for tensors of type 'color_lut'",
                                 std::vector<std::vector<float>>());

  result &= registrar->parameter(
      window_title_, "window_title", "Window title", "Title on window canvas", kDefaultWindowTitle);
  result &= registrar->parameter(display_name_,
                                 "display_name",
                                 "Display name",
                                 "In exclusive mode, name of display to use as shown with xrandr.",
                                 kDefaultDisplayName);
  result &= registrar->parameter(
      width_,
      "width",
      "Width",
      "Window width or display resolution width if in exclusive or fullscreen mode.",
      kDefaultWidth);
  result &= registrar->parameter(
      height_,
      "height",
      "Height",
      "Window height or display resolution height if in exclusive or fullscreen mode.",
      kDefaultHeight);
  result &= registrar->parameter(framerate_,
                                 "framerate",
                                 "Framerate",
                                 "Display framerate if in exclusive mode.",
                                 kDefaultFramerate);
  result &= registrar->parameter(use_exclusive_display_,
                                 "use_exclusive_display",
                                 "Use exclusive display",
                                 "Enable exclusive display",
                                 kDefaultExclusiveDisplay);
  result &= registrar->parameter(fullscreen_,
                                 "fullscreen",
                                 "Use fullscreen window",
                                 "Enable fullscreen window",
                                 kDefaultFullscreen);
  result &= registrar->parameter(headless_,
                                 "headless",
                                 "Headless",
                                 "Enable headless mode. No window is opened, the render buffer is "
                                 "output to `render_buffer_output`",
                                 kDefaultHeadless);
  result &= registrar->parameter(
      window_close_scheduling_term_,
      "window_close_scheduling_term",
      "WindowCloseSchedulingTerm",
      "BooleanSchedulingTerm to stop the codelet from ticking when the window is closed.",
      gxf::Handle<gxf::BooleanSchedulingTerm>());

  result &= registrar->parameter(allocator_,
                                 "allocator",
                                 "Allocator",
                                 "Allocator used to allocate render buffer output.",
                                 gxf::Registrar::NoDefaultParameter(),
                                 GXF_PARAMETER_FLAGS_OPTIONAL);

  return gxf::ToResultCode(result);
}

gxf_result_t Holoviz::start() {
  try {
    // initialize Holoviz
    viz::InitFlags init_flags = viz::InitFlags::NONE;
    if (fullscreen_ && headless_) {
      GXF_LOG_ERROR("Headless and fullscreen are mutually exclusive.");
      return GXF_FAILURE;
    }
    if (fullscreen_) { init_flags = viz::InitFlags::FULLSCREEN; }
    if (headless_) { init_flags = viz::InitFlags::HEADLESS; }

    if (use_exclusive_display_) {
      viz::Init(display_name_.get().c_str(), width_, height_, framerate_, init_flags);
    } else {
      viz::Init(width_, height_, window_title_.get().c_str(), init_flags);
    }

    // get the color lookup table
    const auto& color_lut = color_lut_.get();
    lut_.reserve(color_lut.size() * 4);
    for (auto&& color : color_lut) {
      if (color.size() != 4) {
        GXF_LOG_ERROR("Expected four components in color lookup table element, but got %zu",
                      color.size());
        return GXF_FAILURE;
      }
      lut_.insert(lut_.end(), color.begin(), color.end());
    }

    if (window_close_scheduling_term_.get()) {
      const auto result = window_close_scheduling_term_->enable_tick();
      if (!result) {
        GXF_LOG_ERROR("Failed to enable entity execution using '%s'",
                      window_close_scheduling_term_->name());
        return gxf::ToResultCode(result);
      }
    }

    // Copy the user defined input spec list to the internal input spec list. If there is no user
    // defined input spec it will be generated from the first messages received.
    if (!tensors_.get().empty()) {
      input_spec_.reserve(tensors_.get().size());
      input_spec_.insert(input_spec_.begin(), tensors_.get().begin(), tensors_.get().end());
    }
  } catch (const std::exception& e) {
    GXF_LOG_ERROR(e.what());
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t Holoviz::stop() {
  try {
    viz::Shutdown();
  } catch (const std::exception& e) {
    GXF_LOG_ERROR(e.what());
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t Holoviz::tick() {
  try {
    // Grabs the messages from all receivers
    std::vector<gxf::Entity> messages;
    messages.reserve(receivers_.get().size());
    for (auto& rx : receivers_.get()) {
      gxf::Expected<gxf::Entity> maybe_message = rx->receive();
      if (maybe_message) { messages.push_back(std::move(maybe_message.value())); }
    }
    if (messages.empty()) {
      GXF_LOG_ERROR("No message available.");
      return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
    }

    if (window_close_scheduling_term_.get()) {
      // check if the window had been closed, if yes, stop the execution
      if (viz::WindowShouldClose()) {
        const auto result = window_close_scheduling_term_->disable_tick();
        if (!result) {
          GXF_LOG_ERROR("Failed to enable entity execution using '%s'",
                        window_close_scheduling_term_->name());
          return gxf::ToResultCode(result);
        }
        return GXF_SUCCESS;
      }
    }

    // nothing to do if minimized
    if (viz::WindowIsMinimized()) { return GXF_SUCCESS; }

    // if user provided it, we have a input spec which had been copied at start().
    // else we build the input spec automatically by inspecting the tensors/videobuffers of all
    // messages
    if (input_spec_.empty()) {
      // get all tensors and video buffers of all messages and build the input spec
      for (auto&& message : messages) {
        const auto tensors = message.findAll<gxf::Tensor>();
        for (auto&& tensor : tensors.value()) {
          BufferInfo buffer_info;
          if (buffer_info.init(tensor.value()) != GXF_FAILURE) {
            // try to detect the input type, if we can't detect it, ignore the tensor
            const auto maybe_input_type = detectInputType(buffer_info, !lut_.empty());
            if (maybe_input_type) {
              input_spec_.emplace_back(tensor->name(), maybe_input_type.value());
            }
          }
        }
        const auto video_buffers = message.findAll<gxf::VideoBuffer>();
        for (auto&& video_buffer : video_buffers.value()) {
          BufferInfo buffer_info;
          if (buffer_info.init(video_buffer.value()) != GXF_FAILURE) {
            // try to detect the input type, if we can't detect it, ignore the tensor
            const auto maybe_input_type = detectInputType(buffer_info, !lut_.empty());
            if (maybe_input_type) {
              input_spec_.emplace_back(video_buffer->name(), maybe_input_type.value());
            }
          }
        }
      }
    }

    // begin visualization
    viz::Begin();

    // get the tensors attached to the messages by the tensor names defined by the input spec and
    // display them
    for (auto& input_spec : input_spec_) {
      gxf::Expected<gxf::Handle<gxf::Tensor>> maybe_input_tensor =
          gxf::Unexpected{GXF_UNINITIALIZED_VALUE};
      gxf::Expected<gxf::Handle<gxf::VideoBuffer>> maybe_input_video =
          gxf::Unexpected{GXF_UNINITIALIZED_VALUE};
      auto message = messages.begin();
      while (message != messages.end()) {
        maybe_input_tensor = message->get<gxf::Tensor>(input_spec.tensor_name_.c_str());
        if (maybe_input_tensor) {
          // pick the first one with that name
          break;
        }
        // check for video if no tensor found
        maybe_input_video = message->get<gxf::VideoBuffer>(input_spec.tensor_name_.c_str());
        if (maybe_input_video) {  // pick the first one with that name
          break;
        }
        ++message;
      }
      if (message == messages.end()) {
        GXF_LOG_ERROR("Failed to retrieve input '%s'", input_spec.tensor_name_.c_str());
        return GXF_FAILURE;
      }

      BufferInfo buffer_info;
      gxf_result_t result;
      if (maybe_input_tensor) {
        result = buffer_info.init(maybe_input_tensor.value());
      } else {
        result = buffer_info.init(maybe_input_video.value());
      }
      if (result != GXF_SUCCESS) {
        GXF_LOG_ERROR("Unsupported buffer format tensor/video buffer '%s'",
                      input_spec.tensor_name_.c_str());
        return result;
      }

      // if the input type is unknown it now can be detected using the image properties
      if (input_spec.type_ == InputType::UNKNOWN) {
        const auto maybe_input_type = detectInputType(buffer_info, !lut_.empty());
        if (!maybe_input_type) { return gxf::ToResultCode(maybe_input_type); }
        input_spec.type_ = maybe_input_type.value();
      }

      switch (input_spec.type_) {
        case InputType::COLOR:
        case InputType::COLOR_LUT: {
          // 2D color image

          // sanity checks
          if (buffer_info.rank != 3) {
            GXF_LOG_ERROR("Expected rank 3 for tensor '%s', type '%s', but got %u",
                          buffer_info.name.c_str(),
                          inputTypeToString(input_spec.type_).c_str(),
                          buffer_info.rank);
            return GXF_FAILURE;
          }

          /// @todo this is assuming HWC, should either auto-detect (if possible) or make user
          /// configurable
          const auto height = buffer_info.shape.dimension(0);
          const auto width = buffer_info.shape.dimension(1);
          const auto channels = buffer_info.shape.dimension(2);

          if (input_spec.type_ == InputType::COLOR_LUT) {
            if (channels != 1) {
              GXF_LOG_ERROR(
                  "Expected one channel for tensor '%s' when using lookup table, but got %d",
                  buffer_info.name.c_str(),
                  channels);
              return GXF_FAILURE;
            }
            if (lut_.empty()) {
              GXF_LOG_ERROR(
                  "Type of tensor '%s' is '%s', but a color lookup table has not been specified",
                  buffer_info.name.c_str(),
                  inputTypeToString(input_spec.type_).c_str());
              return GXF_FAILURE;
            }
          }

          auto maybe_image_format = getImageFormatFromTensor(buffer_info);
          if (!maybe_image_format) { return gxf::ToResultCode(maybe_image_format); }
          const viz::ImageFormat image_format = maybe_image_format.value();

          // start a image layer
          viz::BeginImageLayer();
          viz::LayerPriority(input_spec.priority_);
          viz::LayerOpacity(input_spec.opacity_);

          if (input_spec.type_ == InputType::COLOR_LUT) {
            viz::LUT(lut_.size() / 4,
                     viz::ImageFormat::R32G32B32A32_SFLOAT,
                     lut_.size() * sizeof(float),
                     lut_.data());
          }

          if (buffer_info.storage_type == gxf::MemoryStorageType::kDevice) {
            setCudaStreamFromMessage(context(), *message);

            // if it's the device convert to `CUDeviceptr`
            const auto cu_buffer_ptr = reinterpret_cast<CUdeviceptr>(buffer_info.buffer_ptr);
            viz::ImageCudaDevice(width, height, image_format, cu_buffer_ptr);
          } else {
            // convert to void * if using the system/host
            const auto host_buffer_ptr = reinterpret_cast<const void*>(buffer_info.buffer_ptr);
            viz::ImageHost(width, height, image_format, host_buffer_ptr);
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
        case InputType::TEXT: {
          // geometry layer

          // get pointer to tensor buffer
          std::vector<nvidia::byte> host_buffer;
          if (buffer_info.storage_type == gxf::MemoryStorageType::kDevice) {
            host_buffer.resize(buffer_info.bytes_size);

            CUDA_TRY(cudaMemcpy(static_cast<void*>(host_buffer.data()),
                                static_cast<const void*>(buffer_info.buffer_ptr),
                                buffer_info.bytes_size,
                                cudaMemcpyDeviceToHost));

            buffer_info.buffer_ptr = host_buffer.data();
          }

          // start a geometry layer
          viz::BeginGeometryLayer();
          viz::LayerPriority(input_spec.priority_);
          viz::LayerOpacity(input_spec.opacity_);
          std::array<float, 4> color{1.f, 1.f, 1.f, 1.f};
          for (size_t index = 0; index < std::min(input_spec.color_.size(), color.size());
               ++index) {
            color[index] = input_spec.color_[index];
          }
          viz::Color(color[0], color[1], color[2], color[3]);

          /// @todo this is assuming NHW, should either auto-detect (if possible) or make user
          /// configurable
          const auto coordinates = buffer_info.shape.dimension(1);
          const auto components = buffer_info.shape.dimension(2);

          if (input_spec.type_ == InputType::TEXT) {
            // text is defined by the top left coordinate and the size (x, y, s) per string, text
            // strings are define by InputSpec::text_
            if ((components < 2) || (components > 3)) {
              GXF_LOG_ERROR("Expected two or three values per text, but got '%d'", components);
              return GXF_FAILURE;
            }
            const float* src_coord = reinterpret_cast<const float*>(buffer_info.buffer_ptr);
            for (int32_t index = 0; index < coordinates; ++index) {
              viz::Text(
                  src_coord[0],
                  src_coord[1],
                  (components == 3) ? src_coord[2] : 0.05f,
                  input_spec.text_[std::min(index, (int32_t)input_spec.text_.size() - 1)].c_str());
              src_coord += components;
            }
          } else {
            viz::LineWidth(input_spec.line_width_);

            std::vector<float> coords;
            viz::PrimitiveTopology topology;
            uint32_t primitive_count;
            uint32_t coordinate_count;
            uint32_t values_per_coordinate;
            std::vector<float> default_coord;
            if (input_spec.type_ == InputType::POINTS) {
              // point primitives, one coordinate (x, y) per primitive
              if (components != 2) {
                GXF_LOG_ERROR("Expected two values per point, but got '%d'", components);
                return GXF_FAILURE;
              }

              viz::PointSize(input_spec.point_size_);

              topology = viz::PrimitiveTopology::POINT_LIST;
              primitive_count = coordinates;
              coordinate_count = primitive_count;
              values_per_coordinate = 2;
              default_coord = {0.f, 0.f};
            } else if (input_spec.type_ == InputType::LINES) {
              // line primitives, two coordinates (x0, y0) and (x1, y1) per primitive
              if (components != 2) {
                GXF_LOG_ERROR("Expected two values per line vertex, but got '%d'", components);
                return GXF_FAILURE;
              }
              topology = viz::PrimitiveTopology::LINE_LIST;
              primitive_count = coordinates / 2;
              coordinate_count = primitive_count * 2;
              values_per_coordinate = 2;
              default_coord = {0.f, 0.f};
            } else if (input_spec.type_ == InputType::LINE_STRIP) {
              // line primitives, two coordinates (x0, y0) and (x1, y1) per primitive
              if (components != 2) {
                GXF_LOG_ERROR("Expected two values per line strip vertex, but got '%d'",
                              components);
                return GXF_FAILURE;
              }
              topology = viz::PrimitiveTopology::LINE_STRIP;
              primitive_count = coordinates - 1;
              coordinate_count = coordinates;
              values_per_coordinate = 2;
              default_coord = {0.f, 0.f};
            } else if (input_spec.type_ == InputType::TRIANGLES) {
              // triangle primitive, three coordinates (x0, y0), (x1, y1) and (x2, y2) per primitive
              if (components != 2) {
                GXF_LOG_ERROR("Expected two values per triangle vertex, but got '%d'", components);
                return GXF_FAILURE;
              }
              topology = viz::PrimitiveTopology::TRIANGLE_LIST;
              primitive_count = coordinates / 3;
              coordinate_count = primitive_count * 3;
              values_per_coordinate = 2;
              default_coord = {0.f, 0.f};
            } else if (input_spec.type_ == InputType::CROSSES) {
              // cross primitive, a cross is defined by the center coordinate and the size (xi, yi,
              // si)
              if ((components < 2) || (components > 3)) {
                GXF_LOG_ERROR("Expected two or three values per cross, but got '%d'", components);
                return GXF_FAILURE;
              }

              topology = viz::PrimitiveTopology::CROSS_LIST;
              primitive_count = coordinates;
              coordinate_count = primitive_count;
              values_per_coordinate = 3;
              default_coord = {0.f, 0.f, 0.05f};
            } else if (input_spec.type_ == InputType::RECTANGLES) {
              // axis aligned rectangle primitive, each rectangle is defined by two coordinates (xi,
              // yi) and (xi+1, yi+1)
              if (components != 2) {
                GXF_LOG_ERROR("Expected two values per rectangle vertex, but got '%d'", components);
                return GXF_FAILURE;
              }
              topology = viz::PrimitiveTopology::RECTANGLE_LIST;
              primitive_count = coordinates / 2;
              coordinate_count = primitive_count * 2;
              values_per_coordinate = 2;
              default_coord = {0.f, 0.f};
            } else if (input_spec.type_ == InputType::OVALS) {
              // oval primitive, an oval primitive is defined by the center coordinate and the axis
              // sizes (xi, yi, sxi, syi)
              if ((components < 2) || (components > 4)) {
                GXF_LOG_ERROR("Expected two, three or four values per oval, but got '%d'",
                              components);
                return GXF_FAILURE;
              }
              topology = viz::PrimitiveTopology::OVAL_LIST;
              primitive_count = coordinates;
              coordinate_count = primitive_count;
              values_per_coordinate = 4;
              default_coord = {0.f, 0.f, 0.05f, 0.05f};
            } else {
              GXF_LOG_ERROR("Unhandled tensor type '%s'",
                            inputTypeToString(input_spec.type_).c_str());
              return GXF_FAILURE;
            }

            // copy coordinates
            const float* src_coord = reinterpret_cast<const float*>(buffer_info.buffer_ptr);
            coords.reserve(coordinate_count * values_per_coordinate);
            for (int32_t index = 0; index < coordinate_count; ++index) {
              int32_t component_index = 0;
              // copy from source array
              while (component_index < std::min(components, int32_t(values_per_coordinate))) {
                coords.push_back(src_coord[component_index]);
                ++component_index;
              }
              // fill from default array
              while (component_index < values_per_coordinate) {
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
        default:
          GXF_LOG_ERROR("Unhandled input type '%s'", inputTypeToString(input_spec.type_).c_str());
          return GXF_FAILURE;
      }
    }

    viz::End();

    // check if the render buffer should be output
    if (render_buffer_output_.try_get()) {
      auto entity = gxf::Entity::New(context());
      if (!entity) {
        GXF_LOG_ERROR("Failed to allocate message for the render buffer output.");
        return GXF_FAILURE;
      }

      auto video_buffer = entity.value().add<gxf::VideoBuffer>("render_buffer_output");
      if (!video_buffer) {
        GXF_LOG_ERROR("Failed to allocate the video buffer for the render buffer output.");
        return GXF_FAILURE;
      }

      // check if there is a input buffer given to copy the output into
      if (render_buffer_input_.try_get()) {
        const auto& render_buffer_input = render_buffer_input_.try_get().value()->receive();
        if (!render_buffer_input) {
          GXF_LOG_ERROR("No message available at 'render_buffer_input'.");
          return GXF_FAILURE;
        }

        // Get the empty input buffer
        const auto& video_buffer_in = render_buffer_input.value().get<gxf::VideoBuffer>();
        if (!video_buffer_in) {
          GXF_LOG_ERROR("No video buffer attached to message on 'render_buffer_input'.");
          return GXF_FAILURE;
        }

        const gxf::VideoBufferInfo info = video_buffer_in.value()->video_frame_info();

        if ((info.color_format != gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA)) {
          GXF_LOG_ERROR("Invalid render buffer input, expected an RGBA buffer.");
          return GXF_FAILURE;
        }

        video_buffer.value()->wrapMemory(info,
                                         video_buffer_in.value()->size(),
                                         video_buffer_in.value()->storage_type(),
                                         video_buffer_in.value()->pointer(),
                                         nullptr);
      } else {
        // if there is no input buffer given, allocate one
        if (!allocator_.try_get()) {
          GXF_LOG_ERROR("No render buffer input specified and no allocator set.");
          return GXF_FAILURE;
        }

        video_buffer.value()->resize<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
            width_,
            height_,
            gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR,
            gxf::MemoryStorageType::kDevice,
            allocator_.try_get().value());
        if (!video_buffer.value()->pointer()) {
          GXF_LOG_ERROR("Failed to allocate render output buffer.");
          return GXF_FAILURE;
        }
      }

      // read the framebuffer
      viz::ReadFramebuffer(viz::ImageFormat::R8G8B8A8_UNORM,
                           video_buffer.value()->size(),
                           reinterpret_cast<CUdeviceptr>(video_buffer.value()->pointer()));

      // Output the filled render buffer object
      const auto result =
          render_buffer_output_.try_get().value()->publish(std::move(entity.value()));
      if (GXF_SUCCESS != gxf::ToResultCode(result)) {
        GXF_LOG_ERROR("Failed to publish render output buffer");
        return GXF_FAILURE;
      }
    }

    // print the input spec on first tick to let the user know what had been detected
    if (isFirstTick()) { logInputSpec(input_spec_);
    }
  } catch (const std::exception& e) {
    GXF_LOG_ERROR(e.what());
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

}  // namespace nvidia::holoscan
