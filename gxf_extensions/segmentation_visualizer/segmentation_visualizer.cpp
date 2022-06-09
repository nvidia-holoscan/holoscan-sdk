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
#include "segmentation_visualizer.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"

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

#define CUDA_TRY_OR_RETURN_FAILURE(stmt)                                                          \
  ({                                                                                              \
    cudaError_t _holoscan_cuda_err = stmt;                                                        \
    if (cudaSuccess != _holoscan_cuda_err) {                                                      \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", #stmt, \
                    __LINE__, __FILE__, cudaGetErrorString(_holoscan_cuda_err),                   \
                    _holoscan_cuda_err);                                                          \
      return GXF_FAILURE;                                                                         \
    }                                                                                             \
  })

namespace nvidia {
namespace holoscan {
namespace segmentation_visualizer {

static const float kVertices[8] = {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f};

static void glfwPrintErrorCallback(int error, const char* msg) {
  std::cerr << " [" << error << "] " << msg << "\n";
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react
// accordingly
// ----------------------------------------------------------------------------------------------
static void glfwProcessInput(GLFWwindow* window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
}

// whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
static void glfwFramebufferSizeCallback(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
}

static gxf::Expected<std::string> readFile(const std::string& path) {
  std::ifstream istream(path);
  if (istream.fail()) {
    GXF_LOG_WARNING("Failed to find file: '%s'", path.c_str());
    return gxf::Unexpected{GXF_FAILURE};
  }
  std::stringstream sstream;
  sstream << istream.rdbuf();
  return sstream.str();
}

gxf_result_t Visualizer::start() {
  window_ = nullptr;

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
  window_ = glfwCreateWindow(image_width_, image_height_, "GXF Segmentation Visualizer", nullptr,
                             nullptr);
  if (window_ == nullptr) {
    GXF_LOG_ERROR("Failed to create GLFW window");
    glfwTerminate();
    return GXF_FAILURE;
  }
  glfwMakeContextCurrent(window_);
  glfwSetFramebufferSizeCallback(window_, glfwFramebufferSizeCallback);

  // Load all OpenGL function pointers
  GLADloadproc gl_loader = reinterpret_cast<GLADloadproc>(glfwGetProcAddress);
  if (!gladLoadGLLoader(gl_loader)) {
    GXF_LOG_ERROR("Failed to initialize GLAD");
    return GXF_FAILURE;
  }

  // Create shaders
  // --------------
  // Compile the vertex shader
  GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  // Attach the shader source code to the shader object and compile the shader:
  gxf::Expected<std::string> maybe_vertex_shader_source =
      readFile("gxf_extensions/segmentation_visualizer/glsl/segmentation_mask.vert");
  if (!maybe_vertex_shader_source) {
    GXF_LOG_ERROR("Could not open vertex shader source!");
    return GXF_FAILURE;
  }
  const char* vertex_shader_sources[] = {maybe_vertex_shader_source->c_str()};
  glShaderSource(vertex_shader, 1, vertex_shader_sources, nullptr);
  glCompileShader(vertex_shader);
  int success = -1;
  glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char info_log[512] = {0};
    glGetShaderInfoLog(vertex_shader, sizeof(info_log), nullptr, info_log);
    GXF_LOG_ERROR("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s", info_log);
    return GXF_FAILURE;
  }

  // Compile fragment shader
  gxf::Expected<std::string> maybe_fragment_shader_source =
      readFile("gxf_extensions/segmentation_visualizer/glsl/segmentation_mask.frag");
  if (!maybe_fragment_shader_source) {
    GXF_LOG_ERROR("Could not open fragment shader source!");
    return GXF_FAILURE;
  }
  GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  const char* fragment_shader_sources[] = {maybe_fragment_shader_source->c_str()};
  glShaderSource(fragment_shader, 1, fragment_shader_sources, nullptr);
  glCompileShader(fragment_shader);
  glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char info_log[512] = {0};
    glGetShaderInfoLog(fragment_shader, sizeof(info_log), nullptr, info_log);
    GXF_LOG_ERROR("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s", info_log);
    return GXF_FAILURE;
  }

  // Create shader program object to link multiple shaders
  // -----------------------------------------------------
  GLuint shader_program = glCreateProgram();

  // Attach the previously compiled shaders to the program object and then link them
  glAttachShader(shader_program, vertex_shader);
  glAttachShader(shader_program, fragment_shader);
  glLinkProgram(shader_program);
  glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
  if (!success) {
    char info_log[512] = {0};
    glGetProgramInfoLog(shader_program, 512, nullptr, info_log);
    GXF_LOG_ERROR("ERROR::SHADER::PROGRAM::LINK_FAILED\n%s", info_log);
    return GXF_FAILURE;
  }

  // Activate the program. Every shader and rendering call after the activation will now use this
  // program object (and thus the shaders)
  glUseProgram(shader_program);

  // This is the maximum number of lookup colors used in our GLSL shader. It should be consistent
  // with glsl/segmentation_mask.frag
  static const uint32_t MAX_LUT_COLORS = 64;
  const auto& class_color_lut = class_color_lut_.get();
  if (class_color_lut.size() >= MAX_LUT_COLORS) {
    GXF_LOG_ERROR("Too many colors in the class_color_lut %d > %d", class_color_lut.size(),
                  MAX_LUT_COLORS);
    return GXF_FAILURE;
  }
  glUniform1ui(0, class_color_lut.size());
  for (size_t i = 0; i < class_color_lut.size(); ++i) {
    if (class_color_lut[i].size() != 4) {
      GXF_LOG_ERROR("Not enough color components in class_color_lut[%d].size() %d != 4", i,
                    class_color_lut[i].size());
      return GXF_FAILURE;
    }
    glUniform4fv(1 + i, 1, class_color_lut[i].data());
  }

  // Delete the shader objects once we've linked them into the program object
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  // Setup the vertex array.
  GLuint vao, vbo;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(kVertices), kVertices, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(0);

  //  load and create a texture
  glGenTextures(2, textures_);
  cuda_resources_.resize(2, nullptr);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textures_[0]);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // Set alignment requirement to 1 so that the tensor with any width can work.
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  const size_t bytes_per_pixel = 4;
  image_buffer_.resize(image_width_ * image_height_ * bytes_per_pixel, 0);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image_width_, image_height_, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, image_buffer_.data());

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, textures_[1]);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  class_index_buffer_.resize(class_index_width_ * class_index_height_, 0);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, class_index_width_, class_index_height_, 0,
               GL_RED_INTEGER, GL_UNSIGNED_BYTE, class_index_buffer_.data());
  CUDA_TRY_OR_RETURN_FAILURE(cudaGraphicsGLRegisterImage(
      &cuda_resources_[0], textures_[0], GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
  CUDA_TRY_OR_RETURN_FAILURE(cudaGraphicsGLRegisterImage(
      &cuda_resources_[1], textures_[1], GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

  window_close_scheduling_term_->enable_tick();

  return GXF_SUCCESS;
}

gxf_result_t Visualizer::unregisterCudaResources() {
  // Unregister all cuda resources and empty the vector.
  bool success = true;
  for (cudaGraphicsResource* resource : cuda_resources_) {
    if (resource == nullptr) { continue; }
    auto result = CUDA_TRY(cudaGraphicsUnregisterResource(resource));
    if (result != cudaSuccess) { success = false; }
  }
  cuda_resources_.clear();
  return success ? GXF_SUCCESS : GXF_FAILURE;
}

gxf_result_t Visualizer::stop() {
  gxf_result_t cuda_result = unregisterCudaResources();

  // Terminate GLFW regardless of cuda result.
  if (window_ != nullptr) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
  }
  glfwTerminate();

  return cuda_result;
}

gxf_result_t Visualizer::tick() {
  glfwProcessInput(window_);
  if (glfwWindowShouldClose(window_)) {
    window_close_scheduling_term_->disable_tick();
    return GXF_SUCCESS;
  }

  // Grabs latest messages from all receivers
  const auto image_message = image_in_->receive();
  if (!image_message || image_message.value().is_null()) {
    return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
  }

  // Get tensor attached to the message
  gxf::Expected<gxf::Handle<gxf::Tensor>> image_tensor = image_message.value().get<gxf::Tensor>();
  if (!image_tensor) {
    GXF_LOG_ERROR("Could not get input tensor data from message");
    return GXF_FAILURE;
  }
  if (image_tensor.value()->storage_type() != gxf::MemoryStorageType::kDevice) {
    GXF_LOG_ERROR("Expecting image tensor to be allocated on the device");
    return GXF_MEMORY_INVALID_STORAGE_MODE;
  }

  gxf::Expected<const uint8_t*> image_tensor_data = image_tensor.value()->data<uint8_t>();
  if (!image_tensor_data) {
    GXF_LOG_ERROR("Could not get image tensor data");
    return GXF_FAILURE;
  }

  const gxf::Shape image_shape = image_tensor.value()->shape();
  const int32_t image_height = image_shape.dimension(0);
  const int32_t image_width = image_shape.dimension(1);

  if (image_height != image_height_ || image_width != image_width_) {
    GXF_LOG_ERROR("Received Tensor has a different shape (%d, %d). Expected (%d, %d)", image_height,
                  image_width, image_height_.get(), image_width_.get());
    return GXF_FAILURE;
  }

  const auto class_index_message = class_index_in_->receive();
  if (!class_index_message || class_index_message.value().is_null()) {
    return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
  }

  // Get tensor attached to the message
  gxf::Expected<gxf::Handle<gxf::Tensor>> class_index_tensor =
      class_index_message.value().get<gxf::Tensor>();
  if (!class_index_tensor) {
    GXF_LOG_ERROR("Could not get input class index data from message");
    return GXF_FAILURE;
  }
  if (class_index_tensor.value()->storage_type() != gxf::MemoryStorageType::kDevice) {
    return GXF_MEMORY_INVALID_STORAGE_MODE;
  }

  gxf::Expected<const uint8_t*> class_index_tensor_data =
      class_index_tensor.value()->data<uint8_t>();
  if (!class_index_tensor_data) {
    GXF_LOG_ERROR("Could not get input tensor data");
    return GXF_FAILURE;
  }

  const gxf::Shape class_index_shape = class_index_tensor.value()->shape();
  const int32_t class_index_height = class_index_shape.dimension(0);
  const int32_t class_index_width = class_index_shape.dimension(1);

  if (class_index_width != class_index_width_ || class_index_height != class_index_height_) {
    GXF_LOG_ERROR("Received Tensor has a different shape (%d, %d). Expected (%d, %d)",
                  class_index_height, class_index_width, class_index_height_.get(),
                  class_index_width_.get());
    return GXF_FAILURE;
  }

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textures_[0]);
  CUDA_TRY_OR_RETURN_FAILURE(cudaGraphicsMapResources(1, &cuda_resources_[0], 0));
  cudaArray* image_opengl_cuda_ptr = nullptr;
  CUDA_TRY_OR_RETURN_FAILURE(
      cudaGraphicsSubResourceGetMappedArray(&image_opengl_cuda_ptr, cuda_resources_[0], 0, 0));
  const size_t image_bytes_per_pixel = 4;
  const size_t image_pitch_width = image_width * image_bytes_per_pixel;
  CUDA_TRY_OR_RETURN_FAILURE(
      cudaMemcpy2DToArray(image_opengl_cuda_ptr, 0, 0, image_tensor_data.value(), image_pitch_width,
                          image_pitch_width, image_height, cudaMemcpyDeviceToDevice));
  CUDA_TRY_OR_RETURN_FAILURE(cudaGraphicsUnmapResources(1, &cuda_resources_[0], 0));

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, textures_[1]);
  CUDA_TRY_OR_RETURN_FAILURE(cudaGraphicsMapResources(1, &cuda_resources_[1], 0));
  cudaArray* class_index_opengl_cuda_ptr = nullptr;
  CUDA_TRY_OR_RETURN_FAILURE(cudaGraphicsSubResourceGetMappedArray(&class_index_opengl_cuda_ptr,
                                                                   cuda_resources_[1], 0, 0));
  const size_t class_index_bytes_per_pixel = 1;
  const size_t class_index_pitch_width = class_index_width * class_index_bytes_per_pixel;
  CUDA_TRY_OR_RETURN_FAILURE(cudaMemcpy2DToArray(
      class_index_opengl_cuda_ptr, 0, 0, class_index_tensor_data.value(), class_index_pitch_width,
      class_index_pitch_width, class_index_height, cudaMemcpyDeviceToDevice));
  CUDA_TRY_OR_RETURN_FAILURE(cudaGraphicsUnmapResources(1, &cuda_resources_[1], 0));

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
  // -------------------------------------------------------------------------------
  glfwSwapBuffers(window_);
  glfwPollEvents();

  return GXF_SUCCESS;
}

gxf_result_t Visualizer::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(image_in_, "image_in", "Input", "Tensor input");
  result &= registrar->parameter(image_width_, "image_width", "ImageWidth",
                                 "Width of the input image.", 1920);
  result &= registrar->parameter(image_height_, "image_height", "ImageHeight",
                                 "Height of the input image.", 1080);

  result &= registrar->parameter(class_index_in_, "class_index_in", "Input", "Tensor input");
  result &= registrar->parameter(class_index_width_, "class_index_width", "ClassIndexWidth",
                                 "Width of the segmentation class index tensor.", 1920);
  result &= registrar->parameter(class_index_height_, "class_index_height", "ClassIndexHeight",
                                 "Height of the segmentation class index tensor.", 1080);

  result &= registrar->parameter(class_color_lut_, "class_color_lut", "ClassColorLUT",
                                 "Overlay Image Segmentation Class Colormap");

  result &= registrar->parameter(
      window_close_scheduling_term_, "window_close_scheduling_term", "WindowCloseSchedulingTerm",
      "BooleanSchedulingTerm to stop the codelet from ticking after all messages are published.");

  return gxf::ToResultCode(result);
}

}  // namespace segmentation_visualizer
}  // namespace holoscan
}  // namespace nvidia
