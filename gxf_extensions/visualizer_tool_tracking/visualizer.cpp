/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "visualizer.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <npp.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"

#include "opengl_utils.hpp"

#define CUDA_TRY(stmt)                                                                            \
  ({                                                                                              \
    cudaError_t _holoscan_cuda_err = stmt;                                                        \
    if (cudaSuccess != _holoscan_cuda_err) {                                                      \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", #stmt, \
                    __LINE__, __FILE__, cudaGetErrorString(_holoscan_cuda_err),                   \
                    _holoscan_cuda_err);                                                          \
    }                                                                                             \
    _holoscan_cuda_err;                                                                           \
  })

namespace nvidia {
namespace holoscan {
namespace visualizer_tool_tracking {

constexpr int32_t DEFAULT_SRC_WIDTH = 640;
constexpr int32_t DEFAULT_SRC_HEIGHT = 480;
constexpr int16_t DEFAULT_SRC_CHANNELS = 3;
constexpr uint8_t DEFAULT_SRC_BYTES_PER_PIXEL = 1;
// 12 qualitative classes color scheme from colorbrewer2
static const std::vector<std::vector<float>> DEFAULT_COLORS = {
    {0.12f, 0.47f, 0.71f}, {0.20f, 0.63f, 0.17f}, {0.89f, 0.10f, 0.11f}, {1.00f, 0.50f, 0.00f},
    {0.42f, 0.24f, 0.60f}, {0.69f, 0.35f, 0.16f}, {0.65f, 0.81f, 0.89f}, {0.70f, 0.87f, 0.54f},
    {0.98f, 0.60f, 0.60f}, {0.99f, 0.75f, 0.44f}, {0.79f, 0.70f, 0.84f}, {1.00f, 1.00f, 0.60f}};

static const uint32_t NUM_POSITION_COMPONENTS = 2;
static const uint32_t NUM_TOOL_CLASSES = 7;
static const uint32_t POSITION_BUFFER_SIZE =
    sizeof(float) * NUM_TOOL_CLASSES * NUM_POSITION_COMPONENTS;
static const uint32_t CONFIDENCE_BUFFER_SIZE = sizeof(float) * NUM_TOOL_CLASSES;

static void glfwPrintErrorCallback(int error, const char* msg) {
  std::cerr << " [" << error << "] " << msg << "\n";
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react
// accordingly
// ---------------------------------------------------------------------------------------------
static void glfwProcessInput(GLFWwindow* window_) {
  if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window_, true);
}

// whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
static void glfwFramebufferSizeCallback(GLFWwindow* window_, int width, int height) {
  Sink* sink = static_cast<Sink*>(glfwGetWindowUserPointer(window_));
  if (sink) { sink->onFramebufferSizeCallback(width, height); }
}

static void glfwKeyCallback(GLFWwindow* window_, int key, int scancode, int action, int mods) {
  Sink* sink = static_cast<Sink*>(glfwGetWindowUserPointer(window_));
  if (sink) { sink->onKeyCallback(key, scancode, action, mods); }
}

Sink::Sink()
    : video_frame_vis_(video_frame_tex_),
      tooltip_vis_(frame_data_),
      label_vis_(frame_data_),
      overlay_img_vis(frame_data_, overlay_img_tex_) {}

gxf_result_t Sink::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(videoframe_vertex_shader_path_, "videoframe_vertex_shader_path",
                                 "Videoframe GLSL Vertex Shader File Path",
                                 "Path to vertex shader to be loaded");
  result &= registrar->parameter(
      videoframe_fragment_shader_path_, "videoframe_fragment_shader_path",
      "Videoframe GLSL Fragment Shader File Path", "Path to fragment shader to be loaded");

  result &= registrar->parameter(tooltip_vertex_shader_path_, "tooltip_vertex_shader_path",
                                 "Tool tip GLSL Vertex Shader File Path",
                                 "Path to vertex shader to be loaded");
  result &= registrar->parameter(tooltip_fragment_shader_path_, "tooltip_fragment_shader_path",
                                 "Tool tip GLSL Fragment Shader File Path",
                                 "Path to fragment shader to be loaded");
  result &= registrar->parameter(num_tool_classes_, "num_tool_classes", "Tool Classes",
                                 "Number of different tool classes");
  result &= registrar->parameter(num_tool_pos_components_, "num_tool_pos_components",
                                 "Position Components",
                                 "Number of components of the tool position vector", 2);
  result &= registrar->parameter(
      tool_tip_colors_, "tool_tip_colors", "Tool Tip Colors",
      "Color of the tool tips, a list of RGB values with components between 0 and 1",
      DEFAULT_COLORS);

  result &= registrar->parameter(overlay_img_vertex_shader_path_, "overlay_img_vertex_shader_path",
                                 "Overlay Image GLSL Vertex Shader File Path",
                                 "Path to vertex shader to be loaded");
  result &= registrar->parameter(
      overlay_img_fragment_shader_path_, "overlay_img_fragment_shader_path",
      "Overlay Image GLSL Fragment Shader File Path", "Path to fragment shader to be loaded");
  result &= registrar->parameter(overlay_img_width_, "overlay_img_width", "Overlay Image Width",
                                 "Width of overlay image");
  result &= registrar->parameter(overlay_img_height_, "overlay_img_height", "Overlay Image Height",
                                 "Height of overlay image");
  result &=
      registrar->parameter(overlay_img_channels_, "overlay_img_channels",
                           "Number of Overlay Image Channels", "Number of Overlay Image Channels");
  result &=
      registrar->parameter(overlay_img_layers_, "overlay_img_layers",
                           "Number of Overlay Image Layers", "Number of Overlay Image Layers");
  result &= registrar->parameter(
      overlay_img_colors_, "overlay_img_colors", "Overlay Image Layer Colors",
      "Color of the image overlays, a list of RGB values with components between 0 and 1",
      DEFAULT_COLORS);

  result &= registrar->parameter(
      tool_labels_, "tool_labels", "Tool Labels", "List of tool names.",
      {});  // Default handled in instrument_label to dynamically adjust for the number of tools
  result &= registrar->parameter(label_sans_font_path_, "label_sans_font_path",
                                 "File path for sans font for displaying tool name",
                                 "Path for sans font to be loaded");
  result &= registrar->parameter(label_sans_bold_font_path_, "label_sans_bold_font_path",
                                 "File path for sans bold font for displaying tool name",
                                 "Path for sans bold font to be loaded");

  result &= registrar->parameter(in_, "in", "Input", "List of input channels");
  result &= registrar->parameter(in_tensor_names_, "in_tensor_names", "Input Tensor Names",
                                 "Names of input tensors.", {std::string("")});
  result &= registrar->parameter(in_width_, "in_width", "InputWidth", "Width of the image.",
                                 DEFAULT_SRC_WIDTH);
  result &= registrar->parameter(in_height_, "in_height", "InputHeight", "Height of the image.",
                                 DEFAULT_SRC_HEIGHT);
  result &= registrar->parameter(in_channels_, "in_channels", "InputChannels",
                                 "Number of channels.", DEFAULT_SRC_CHANNELS);
  result &=
      registrar->parameter(in_bytes_per_pixel_, "in_bytes_per_pixel", "InputBytesPerPixel",
                           "Number of bytes per pixel of the image.", DEFAULT_SRC_BYTES_PER_PIXEL);
  result &= registrar->parameter(alpha_value_, "alpha_value", "Alpha value",
                                 "Alpha value that can be used when converting RGB888 to RGBA8888.",
                                 static_cast<uint8_t>(255));
  result &= registrar->parameter(pool_, "pool", "Pool", "Pool to allocate the output message.");
  result &= registrar->parameter(
      window_close_scheduling_term_, "window_close_scheduling_term", "WindowCloseSchedulingTerm",
      "BooleanSchedulingTerm to stop the codelet from ticking after all messages are published.");

  return gxf::ToResultCode(result);
}

gxf_result_t Sink::start() {
  glfwSetErrorCallback(glfwPrintErrorCallback);

  // Create window
  // -------------
  // initialize and configure
  if (!glfwInit()) {
    GXF_LOG_ERROR("Failed to initialize GLFW");
    glfwTerminate();
    return GXF_FAILURE;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  window_ = glfwCreateWindow(in_width_, in_height_, "GXF Video Stream", NULL, NULL);
  if (window_ == NULL) {
    GXF_LOG_ERROR("Failed to create GLFW window");
    glfwTerminate();
    return GXF_FAILURE;
  }
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, glfwFramebufferSizeCallback);
  glfwSetKeyCallback(window_, glfwKeyCallback);
  glfwMakeContextCurrent(window_);

  // propage width, height manually as first framebuffer resize callback is not triggered
  onFramebufferSizeCallback(in_width_, in_height_);

  // Load all OpenGL function pointers
  GLADloadproc gl_loader = reinterpret_cast<GLADloadproc>(glfwGetProcAddress);
  if (!gladLoadGLLoader(gl_loader)) {
    GXF_LOG_ERROR("Failed to initialize GLAD");
    return GXF_FAILURE;
  }

  glEnable(GL_DEBUG_OUTPUT);
  // disable frequent GL API notification messages, e.g. buffer usage info, to avoid spamming log
  glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DEBUG_SEVERITY_NOTIFICATION, 0,
                        0, GL_FALSE);
  glDebugMessageCallback(OpenGLDebugMessageCallback, 0);
  glDisable(GL_BLEND);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // CUDA GL Interop not support with 3 channel formats, e.g. GL_RGB
  use_cuda_opengl_interop_ = in_channels_.get() != 3;

  GXF_LOG_INFO("#%d channels, CUDA / OpenGL interop %s", in_channels_.get(),
               use_cuda_opengl_interop_ ? "enabled" : "disabled");

  // Allocate host memory and OpenGL buffers, textures for video frame and inference results
  // ----------------------------------------------------------------------------------

  video_frame_buffer_host_.resize(in_width_ * in_height_ * in_channels_.get(), 0);
  overlay_img_buffer_host_.resize(overlay_img_width_ * overlay_img_height_ * overlay_img_layers_,
                                  0.0f);
  overlay_img_layered_host_.resize(overlay_img_width_ * overlay_img_height_ * overlay_img_layers_,
                                   0.0f);
  frame_data_.confidence_host_.resize(NUM_TOOL_CLASSES, 0.0f);
  frame_data_.position_host_.resize(NUM_TOOL_CLASSES * NUM_POSITION_COMPONENTS, 0.0f);

  glCreateBuffers(1, &frame_data_.position_);
  glNamedBufferData(frame_data_.position_, POSITION_BUFFER_SIZE, NULL, GL_STREAM_DRAW);
  glCreateBuffers(1, &frame_data_.confidence_);
  glNamedBufferData(frame_data_.confidence_, CONFIDENCE_BUFFER_SIZE, NULL, GL_STREAM_DRAW);

  // register frame_data_.confidence_ and position_ with CUDA
  {
    cudaError_t cuda_status = CUDA_TRY(cudaGraphicsGLRegisterBuffer(
        &cuda_confidence_resource_, frame_data_.confidence_, cudaGraphicsMapFlagsWriteDiscard));
    if (cuda_status) {
      GXF_LOG_ERROR("Failed to register confidence buffer for CUDA / OpenGL Interop");
      return GXF_FAILURE;
    }
    cuda_status = CUDA_TRY(cudaGraphicsGLRegisterBuffer(
        &cuda_position_resource_, frame_data_.position_, cudaGraphicsMapFlagsWriteDiscard));
    if (cuda_status) {
      GXF_LOG_ERROR("Failed to register position buffer for CUDA / OpenGL Interop");
      return GXF_FAILURE;
    }
  }

  glActiveTexture(GL_TEXTURE0);
  glGenTextures(1, &video_frame_tex_);
  glBindTexture(GL_TEXTURE_2D, video_frame_tex_);
  // allocate immutable texture storage ( if resize need to re-create texture object)
  GLenum format = (in_channels_ == 4) ? GL_RGBA8 : GL_RGB8;
  glTexStorage2D(GL_TEXTURE_2D, 1, format, in_width_, in_height_);
  // set the texture wrapping parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

  // register this texture with CUDA
  if (use_cuda_opengl_interop_) {
    cudaError_t cuda_status =
        CUDA_TRY(cudaGraphicsGLRegisterImage(&cuda_video_frame_tex_resource_, video_frame_tex_,
                                             GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
    if (cuda_status) {
      GXF_LOG_ERROR("Failed to register video frame texture for CUDA / OpenGL Interop");
      return GXF_FAILURE;
    }
  }

  glGenTextures(1, &overlay_img_tex_);
  glBindTexture(GL_TEXTURE_2D_ARRAY, overlay_img_tex_);
  // allocate immutable texture storage ( if resize need to re-create texture object)
  glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_R32F, overlay_img_width_, overlay_img_height_,
                 overlay_img_layers_);
  // set the texture wrapping parameters
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

  // CUDA / GL interop for overlay image tex
  {
    cudaError_t cuda_status = CUDA_TRY(
        cudaGraphicsGLRegisterImage(&cuda_overlay_img_tex_resource_, overlay_img_tex_,
                                    GL_TEXTURE_2D_ARRAY, cudaGraphicsMapFlagsWriteDiscard));
    if (cuda_status) {
      GXF_LOG_ERROR("Failed to register overlay image texture for CUDA / OpenGL Interop");
      return GXF_FAILURE;
    }
  }

  // Initialize helper class instancces
  // ----------------------------------------------------------------------------------

  video_frame_vis_.vertex_shader_file_path_ = videoframe_vertex_shader_path_.get();
  video_frame_vis_.fragment_shader_file_path_ = videoframe_fragment_shader_path_.get();

  tooltip_vis_.vertex_shader_file_path_ = tooltip_vertex_shader_path_.get();
  tooltip_vis_.fragment_shader_file_path_ = tooltip_fragment_shader_path_.get();
  tooltip_vis_.num_tool_classes_ = num_tool_classes_.get();
  tooltip_vis_.num_tool_pos_components_ = num_tool_pos_components_.get();
  tooltip_vis_.tool_tip_colors_ = tool_tip_colors_.get();

  overlay_img_vis.vertex_shader_file_path_ = overlay_img_vertex_shader_path_.get();
  overlay_img_vis.fragment_shader_file_path_ = overlay_img_fragment_shader_path_.get();
  overlay_img_vis.num_layers_ = overlay_img_layers_.get();
  overlay_img_vis.layer_colors_ = overlay_img_colors_.get();

  label_vis_.label_sans_font_path_ = label_sans_font_path_.get();
  label_vis_.label_sans_bold_font_path_ = label_sans_bold_font_path_.get();
  label_vis_.num_tool_classes_ = num_tool_classes_.get();
  label_vis_.num_tool_pos_components_ = num_tool_pos_components_.get();
  label_vis_.tool_labels_ = tool_labels_.get();

  gxf_result_t res = video_frame_vis_.start();
  if (res != GXF_SUCCESS) { return res; }
  res = label_vis_.start();
  if (res != GXF_SUCCESS) { return res; }
  res = overlay_img_vis.start();
  if (res != GXF_SUCCESS) { return res; }
  res = tooltip_vis_.start();
  if (res != GXF_SUCCESS) { return res; }

  window_close_scheduling_term_->enable_tick();

  return GXF_SUCCESS;
}

gxf_result_t Sink::stop() {
  // Free mem allocated in utility classes.
  // ----------------------------------------------------------------------------------

  video_frame_vis_.stop();
  label_vis_.stop();
  overlay_img_vis.stop();
  tooltip_vis_.stop();

  // Free OpenGL buffer and texture memory
  // ----------------------------------------------------------------------------------

  // terminate, clearing all previously allocated GLFW resources.
  if (window_ != nullptr) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
  }
  glfwTerminate();

  return GXF_SUCCESS;
}

gxf_result_t Sink::tick() {
  // Grabs latest messages from all receivers
  std::vector<gxf::Entity> messages;
  messages.reserve(in_.get().size());
  for (auto& rx : in_.get()) {
    gxf::Expected<gxf::Entity> maybe_message = rx->receive();
    if (maybe_message) { messages.push_back(std::move(maybe_message.value())); }
  }
  if (messages.empty()) {
    GXF_LOG_ERROR("No message available.");
    return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
  }

  glfwProcessInput(window_);
  if (glfwWindowShouldClose(window_)) {
    window_close_scheduling_term_->disable_tick();
    return GXF_SUCCESS;
  }

  cudaError_t cuda_status = {};

  // Read message from receiver, update buffers / memory
  // ----------------------------------------------------------------------------------

  gxf::Expected<gxf::Handle<gxf::Tensor>> maybe_tensor = gxf::Unexpected{GXF_UNINITIALIZED_VALUE};
  gxf::Expected<gxf::Handle<gxf::VideoBuffer>> maybe_video =
      gxf::Unexpected{GXF_UNINITIALIZED_VALUE};

  // Pick one input tensor
  for (const auto& tensor_name : in_tensor_names_.get()) {
    for (auto& msg : messages) {
      maybe_tensor = msg.get<gxf::Tensor>(tensor_name.c_str());
      if (maybe_tensor) { break; }
      maybe_video = msg.get<gxf::VideoBuffer>();
      if (maybe_video) { break; }
    }
    if (!maybe_tensor && !maybe_video) {
      GXF_LOG_ERROR("Failed to retrieve Tensor(%s) or VideoBuffer", tensor_name.c_str());
      return GXF_FAILURE;
    }

    // Pick only the first tensor from multiple input channels/tensors
    break;
  }

  uint8_t* in_tensor_ptr = nullptr;
  uint64_t buffer_size = 0;
  int32_t columns = 0;
  int32_t rows = 0;
  int16_t channels = 0;
  if (maybe_video) {
    {
      auto frame = maybe_video.value();

      // NOTE: VideoBuffer::moveToTensor() converts VideoBuffer instance to the Tensor instance
      // with an unexpected shape:  [width, height] or [width, height, num_planes].
      // And, if we use moveToTensor() to convert VideoBuffer to Tensor, we may lose the the
      // original video buffer when the VideoBuffer instance is used in other places. For that
      // reason, we directly access internal data of VideoBuffer instance to access Tensor data.
      const auto& buffer_info = frame->video_frame_info();
      switch (buffer_info.color_format) {
        case gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
          break;
        default:
          GXF_LOG_ERROR("Unsupported input format: %d\n", buffer_info.color_format);
          return GXF_FAILURE;
      }

      columns = buffer_info.width;
      rows = buffer_info.height;
      channels = 4;  // RGBA

      in_tensor_ptr = frame->pointer();
    }
  } else {
    // Get tensor attached to the message
    auto in_tensor = maybe_tensor;
    if (!in_tensor) { return in_tensor.error(); }
    if (in_tensor.value()->storage_type() != gxf::MemoryStorageType::kDevice) {
      return GXF_MEMORY_INVALID_STORAGE_MODE;
    }
    auto maybe_in_tensor_ptr = in_tensor.value()->data<uint8_t>();
    if (!maybe_in_tensor_ptr) { return maybe_in_tensor_ptr.error(); }

    const gxf::Shape shape = in_tensor.value()->shape();

    columns = shape.dimension(1);
    rows = shape.dimension(0);
    channels = shape.dimension(2);

    in_tensor_ptr = maybe_in_tensor_ptr.value();
  }

  if (in_channels_ != channels) {
    GXF_LOG_ERROR(
        "Received VideoBuffer/Tensor has a different number of channels (%d. Expected %d)",
        channels, in_channels_.get());
    return GXF_FAILURE;
  }

  buffer_size = columns * rows * channels;

  const int32_t in_height = in_height_;
  const int32_t in_width = in_width_;
  const int32_t in_channels = in_channels_;

  const uint64_t buffer_items_size = in_width * in_height * in_channels;

  if (in_height != rows || in_width != columns || in_channels != channels) {
    GXF_LOG_ERROR("Received buffer size doesn't match.");
    return GXF_FAILURE;
  }

  // Set alignment requirement to 1 so that the tensor with any width can work.
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  if (in_tensor_ptr && buffer_size > 0) {
    //  Video Frame
    // --------------------------------------------------------------------------------------------
    if (use_cuda_opengl_interop_) {
      cuda_status = CUDA_TRY(cudaGraphicsMapResources(1, &cuda_video_frame_tex_resource_, 0));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to map video frame texture via CUDA / OpenGL interop");
        return GXF_FAILURE;
      }
      cudaArray* texture_ptr = nullptr;
      cuda_status = CUDA_TRY(cudaGraphicsSubResourceGetMappedArray(
          &texture_ptr, cuda_video_frame_tex_resource_, 0, 0));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to get mapped array for video frame texture");
        return GXF_FAILURE;
      }
      size_t spitch = 4 * in_width_ * sizeof(GLubyte);
      cuda_status = CUDA_TRY(cudaMemcpy2DToArray(texture_ptr, 0, 0, in_tensor_ptr, spitch, spitch,
                                                 in_height, cudaMemcpyDeviceToDevice));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to copy video frame to OpenGL texture via CUDA / OpenGL interop");
        return GXF_FAILURE;
      }
      cuda_status = CUDA_TRY(cudaGraphicsUnmapResources(1, &cuda_video_frame_tex_resource_, 0));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to unmap video frame texture via CUDA / OpenGL interop");
        return GXF_FAILURE;
      }
    } else {
      const uint64_t buffer_size = buffer_items_size * in_bytes_per_pixel_;
      cuda_status = CUDA_TRY(cudaMemcpy(video_frame_buffer_host_.data(), in_tensor_ptr, buffer_size,
                                        cudaMemcpyDeviceToHost));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to copy video frame texture from Device to Host");
        return GXF_FAILURE;
      }
      // update data
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, video_frame_tex_);
      GLenum format = (in_channels == 4) ? GL_RGBA : GL_RGB;
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, in_width_, in_height_, format, GL_UNSIGNED_BYTE,
                      video_frame_buffer_host_.data());
    }
  }  // if (in_tensor_ptr && buffer_size > 0)

  if (messages.size() >= 2) {
    const auto& inference_message = messages[1];
    //  Confidence
    // --------------------------------------------------------------------------------------------
    if (enable_tool_tip_vis_ || enable_tool_labels_ || enable_overlay_img_vis_) {
      auto src = inference_message.get<gxf::Tensor>("probs").value()->data<float>().value();

      // download for use on host for e.g. controlling rendering of instrument names
      cuda_status = CUDA_TRY(cudaMemcpy(frame_data_.confidence_host_.data(), src,
                                        CONFIDENCE_BUFFER_SIZE, cudaMemcpyDeviceToHost));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to copy confidence buffer from Device to Host.");
        return GXF_FAILURE;
      }
      // use CUDA / OpenGL interop to copy confidence tensor to confidence OpenGL buffer object
      // with device to device memcpy
      cuda_status = CUDA_TRY(cudaGraphicsMapResources(1, &cuda_confidence_resource_, 0));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to map confidence buffer via CUDA / OpenGL interop");
        return GXF_FAILURE;
      }
      float* dptr = nullptr;
      size_t num_bytes = 0;
      cuda_status = CUDA_TRY(cudaGraphicsResourceGetMappedPointer(
          reinterpret_cast<void**>(&dptr), &num_bytes, cuda_confidence_resource_));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to get mapped pointer for confidence buffer");
        return GXF_FAILURE;
      }
      cuda_status = CUDA_TRY(cudaMemcpy(dptr, src, num_bytes, cudaMemcpyDeviceToDevice));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to copy confidence buffer to OpenGL via CUDA / OpenGL interop");
        return GXF_FAILURE;
      }
      cuda_status = CUDA_TRY(cudaGraphicsUnmapResources(1, &cuda_confidence_resource_, 0));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to unmap confidence buffer via CUDA / OpenGL interop");
        return GXF_FAILURE;
      }
    }

    //  Position
    // --------------------------------------------------------------------------------------------
    if (enable_tool_tip_vis_ || enable_tool_labels_) {
      auto src = inference_message.get<gxf::Tensor>("scaled_coords").value()->data<float>().value();
      // Data still needs to be copied to host for drawing text labels
      cuda_status = CUDA_TRY(cudaMemcpy(frame_data_.position_host_.data(), src,
                                        POSITION_BUFFER_SIZE, cudaMemcpyDeviceToHost));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to copy position buffer from Device to Host.");
        return GXF_FAILURE;
      }

      // use CUDA / OpenGL interop to copy position tensor to position OpenGL buffer object
      // with device to device memcpy
      cuda_status = CUDA_TRY(cudaGraphicsMapResources(1, &cuda_position_resource_, 0));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to map position buffer via CUDA / OpenGL interop");
        return GXF_FAILURE;
      }
      float* dptr = nullptr;
      size_t num_bytes = 0;
      cuda_status = CUDA_TRY(cudaGraphicsResourceGetMappedPointer(
          reinterpret_cast<void**>(&dptr), &num_bytes, cuda_position_resource_));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to get mapped pointer for position buffer");
        return GXF_FAILURE;
      }
      cuda_status = CUDA_TRY(cudaMemcpy(dptr, src, num_bytes, cudaMemcpyDeviceToDevice));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to copy position buffer to OpenGL via CUDA / OpenGL interop");
        return GXF_FAILURE;
      }
      cuda_status = CUDA_TRY(cudaGraphicsUnmapResources(1, &cuda_position_resource_, 0));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to unmap position buffer via CUDA / OpenGL interop");
        return GXF_FAILURE;
      }
    }

    //  Overlay Image Texture
    // --------------------------------------------------------------------------------------------
    if (enable_overlay_img_vis_) {
      auto src = inference_message.get<gxf::Tensor>("binary_masks").value()->data<float>().value();
      cuda_status = CUDA_TRY(cudaGraphicsMapResources(1, &cuda_overlay_img_tex_resource_, 0));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to map overlay image texture via CUDA / OpenGL interop");
        return GXF_FAILURE;
      }

      cudaArray* texture_ptr = nullptr;
      cudaMemcpy3DParms copyParams = {0};
      // overlay image from inference
      copyParams.srcPtr = make_cudaPitchedPtr(src, overlay_img_width_ * sizeof(float),
                                              overlay_img_width_, overlay_img_height_);
      copyParams.extent = make_cudaExtent(overlay_img_width_, overlay_img_height_, 1);
      copyParams.kind = cudaMemcpyDeviceToDevice;
      // copy overlay image layer by layer with data staying on GPU
      for (int layer = 0; layer != overlay_img_layers_; ++layer) {
        cuda_status = CUDA_TRY(cudaGraphicsSubResourceGetMappedArray(
            &texture_ptr, cuda_overlay_img_tex_resource_, layer, 0));
        if (cuda_status) {
          GXF_LOG_ERROR("Failed to get mapped array for %d-th layer of overlay OpenGL texture",
                        layer);
          return GXF_FAILURE;
        }
        // set dst array for cudaMemcpy3D, configure offset for src pointer to copy current layer to
        // OpenGL texture
        copyParams.dstArray = texture_ptr;
        copyParams.srcPos = make_cudaPos(0, 0, layer);

        cuda_status = CUDA_TRY(cudaMemcpy3D(&copyParams));
        if (cuda_status) {
          GXF_LOG_ERROR(
              "Failed to copy %d-th layer of overlay image tensor to OpenGL texture via CUDA / "
              "OpenGL interop",
              layer);
          return GXF_FAILURE;
        }
      }
      // unmap the graphics resource again
      cuda_status = CUDA_TRY(cudaGraphicsUnmapResources(1, &cuda_overlay_img_tex_resource_, 0));
      if (cuda_status) {
        GXF_LOG_ERROR("Failed to unmap overlay image texture via CUDA / OpenGL interop");
        return GXF_FAILURE;
      }
    }
  }  // if (messages.size() >= 2)

  if (vp_changed_) {
    glViewport(0, 0, vp_width_, vp_height_);
    vp_changed_ = false;
  }

  // Draw Frame
  // ----------------------------------------------------------------------------------

  video_frame_vis_.tick();

  // render inference results: Overlay Image, Tooltip, Tool Names
  glEnable(GL_BLEND);

  if (enable_overlay_img_vis_) { overlay_img_vis.tick(); }

  if (enable_tool_tip_vis_) { tooltip_vis_.tick(); }

  glDisable(GL_BLEND);

  if (enable_tool_labels_) {
    label_vis_.vp_width_ = vp_width_;
    label_vis_.vp_height_ = vp_height_;
    label_vis_.vp_aspect_ratio_ = vp_aspect_ratio_;

    label_vis_.tick();
  }

  // swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
  // -------------------------------------------------------------------------------
  glfwSwapBuffers(window_);
  glfwPollEvents();

  return GXF_SUCCESS;
}

void Sink::onKeyCallback(int key, int scancode, int action, int mods) {
  if (action != GLFW_RELEASE) { return; }

  switch (key) {
    default:
      break;
    case GLFW_KEY_1:
      enable_tool_tip_vis_ = !enable_tool_tip_vis_;
      break;
    case GLFW_KEY_2:
      enable_tool_labels_ = !enable_tool_labels_;
      break;
    case GLFW_KEY_3:
      enable_overlay_img_vis_ = !enable_overlay_img_vis_;
      break;
  }
}

void Sink::onFramebufferSizeCallback(int width, int height) {
  if (width == vp_width_ && height == vp_height_) { return; }

  vp_width_ = width;
  vp_height_ = height;
  vp_aspect_ratio_ = static_cast<float>(vp_width_) / static_cast<float>(vp_height_);
  vp_changed_ = true;
}

}  // namespace visualizer_tool_tracking
}  // namespace holoscan
}  // namespace nvidia
