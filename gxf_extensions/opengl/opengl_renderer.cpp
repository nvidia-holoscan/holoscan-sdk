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
#include "opengl_renderer.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <iostream>

#define CUDA_ERROR(stmt)                                                                    \
  {                                                                                         \
    cudaError_t err = stmt;                                                                 \
    if (cudaSuccess != err) {                                                               \
      GXF_LOG_ERROR("CUDA runtime call %s (%s:%d) failed with '%s' (%d).", #stmt, __FILE__, \
                    __LINE__, cudaGetErrorString(err), err);                                \
      return GXF_FAILURE;                                                                   \
    }                                                                                       \
  }

namespace nvidia {
namespace holoscan {

static const char* kVertexShaderSource =
    "#version 330 core\n"
    "layout (location = 0) in vec2 coord;\n"
    "out vec2 texCoord;\n"
    "void main() {\n"
    "  gl_Position = vec4((coord * 2.0) - 1.0, 0.0, 1.0);\n"
    "  texCoord = vec2(coord.x, 1.0 - coord.y);\n"
    "}\n";

static const char* kFragmentShaderSource =
    "#version 330 core\n"
    "out vec4 fragColor;\n"
    "in vec2 texCoord;\n"
    "uniform sampler2D inTexture;\n"
    "void main() {\n"
    "  fragColor = texture2D(inTexture, texCoord);\n"
    "}\n";

static const float kVertices[32] = {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f};

static void glfwPrintErrorCallback(int error, const char* msg) {
  std::cerr << " [" << error << "] " << msg << "\n";
}

static void glfwFramebufferSizeCallback(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
}

OpenGLRenderer::OpenGLRenderer() : texture_width_(0), texture_height_(0), cuda_resource_(nullptr) {}

gxf_result_t OpenGLRenderer::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(signal_, "signal", "Input", "Input Channel");
  result &= registrar->parameter(width_, "width", "Width", "Width of the rendering window");
  result &= registrar->parameter(height_, "height", "Height", "Height of the rendering window");
  result &= registrar->parameter(
      window_close_scheduling_term_, "window_close_scheduling_term", "WindowCloseSchedulingTerm",
      "BooleanSchedulingTerm to stop the codelet from ticking after all messages are published.");
  return gxf::ToResultCode(result);
}

void OpenGLRenderer::processInput() {
  if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window_, true);
  }
}

GLuint OpenGLRenderer::createShader(GLenum type, const char* src) {
  auto shader = glCreateShader(type);

  glShaderSource(shader, 1, &src, NULL);
  glCompileShader(shader);

  GLint compile_status = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
  if (!compile_status) {
    GLchar info_log[512];
    glGetShaderInfoLog(shader, sizeof(info_log), NULL, info_log);
    GXF_LOG_ERROR("%s shader compilation failed with log:\n%s\n",
                  (type == GL_VERTEX_SHADER ? "Vertex" : "Fragment"), info_log);
    return 0;
  }

  return shader;
}

gxf_result_t OpenGLRenderer::start() {
  // Initialize GLFW.
  glfwSetErrorCallback(glfwPrintErrorCallback);
  if (!glfwInit()) {
    GXF_LOG_ERROR("Failed to initialize GLFW");
    return GXF_FAILURE;
  }

  // Create GLFW Window.
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  window_ = glfwCreateWindow(width_.get(), height_.get(), "GXF OpenGL Renderer", NULL, NULL);
  if (!window_) {
    GXF_LOG_ERROR("Failed to create GLFW window");
    glfwTerminate();
    return GXF_FAILURE;
  }
  glfwMakeContextCurrent(window_);
  glfwSetFramebufferSizeCallback(window_, glfwFramebufferSizeCallback);

  // Initialize GL function pointers.
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    GXF_LOG_ERROR("Failed to initialize GLAD");
    return GXF_FAILURE;
  }

  // Compile the shaders.
  auto vert_shader = createShader(GL_VERTEX_SHADER, kVertexShaderSource);
  auto frag_shader = createShader(GL_FRAGMENT_SHADER, kFragmentShaderSource);
  if (vert_shader == 0 || frag_shader == 0) { return GXF_FAILURE; }

  // Create the shader program.
  GLuint program = glCreateProgram();
  glAttachShader(program, vert_shader);
  glAttachShader(program, frag_shader);
  glDeleteShader(vert_shader);
  glDeleteShader(frag_shader);

  glLinkProgram(program);
  GLint link_status = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &link_status);
  if (!link_status) {
    char info_log[512];
    glGetProgramInfoLog(program, sizeof(info_log), NULL, info_log);
    GXF_LOG_ERROR("Program linking failed with log:\n%s\n", info_log);
    return GXF_FAILURE;
  }
  glUseProgram(program);

  // Setup the vertex array.
  GLuint vao, vbo;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(kVertices), kVertices, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(0);

  // Create the texture.
  glGenTextures(1, &gl_texture_);
  glBindTexture(GL_TEXTURE_2D, gl_texture_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  window_close_scheduling_term_->enable_tick();

  return GXF_SUCCESS;
}

gxf_result_t OpenGLRenderer::stop() {
  if (cuda_resource_) {
    cudaGraphicsUnregisterResource(cuda_resource_);
    cuda_resource_ = nullptr;
  }

  if (window_ != nullptr) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
  }
  glfwTerminate();

  return GXF_SUCCESS;
}

gxf_result_t OpenGLRenderer::tick() {
  processInput();
  if (glfwWindowShouldClose(window_)) {
    window_close_scheduling_term_->disable_tick();
    return GXF_SUCCESS;
  }

  auto message = signal_->receive();
  if (!message || message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }

  auto buffer = message.value().get<gxf::VideoBuffer>();
  if (!buffer || !buffer.value()->pointer()) {
    GXF_LOG_ERROR("VideoBuffer not provided");
    return GXF_FAILURE;
  }

  bool is_device_memory = buffer.value()->storage_type() == gxf::MemoryStorageType::kDevice;

  // Update the texture using the given buffer.
  auto info = buffer.value()->video_frame_info();
  if (texture_width_ != info.width || texture_height_ != info.height ||
      texture_format_ != getTextureFormat(info.color_format)) {
    texture_width_ = info.width;
    texture_height_ = info.height;
    texture_format_ = getTextureFormat(info.color_format);
    glTexImage2D(GL_TEXTURE_2D, 0, texture_format_, texture_width_, texture_height_, 0,
                 texture_format_, GL_UNSIGNED_BYTE, NULL);

    if (cuda_resource_) {
      CUDA_ERROR(cudaGraphicsUnregisterResource(cuda_resource_));
      cuda_resource_ = nullptr;
    }
    if (is_device_memory) {
      CUDA_ERROR(cudaGraphicsGLRegisterImage(&cuda_resource_, gl_texture_, GL_TEXTURE_2D,
                                             cudaGraphicsMapFlagsWriteDiscard));
    }
  }

  if (is_device_memory) {
    CUDA_ERROR(cudaGraphicsMapResources(1, &cuda_resource_, 0));
    cudaArray* cuda_ptr = nullptr;
    CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cuda_ptr, cuda_resource_, 0, 0));
    CUDA_ERROR(
        cudaMemcpy2DToArray(cuda_ptr, 0, 0, buffer.value()->pointer(), info.color_planes[0].stride,
                            info.color_planes[0].width * info.color_planes[0].bytes_per_pixel,
                            info.color_planes[0].height, cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaGraphicsUnmapResources(1, &cuda_resource_, 0));
  } else {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture_width_, texture_height_, texture_format_,
                    GL_UNSIGNED_BYTE, buffer.value()->pointer());
  }

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  glfwSwapBuffers(window_);
  glfwPollEvents();

  return GXF_SUCCESS;
}

GLenum OpenGLRenderer::getTextureFormat(gxf::VideoFormat format) {
  switch (format) {
    case gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA:
      return GL_RGBA;
    default:
      GXF_LOG_WARNING("Unsupported video format: %d", format);
      return GL_NONE;
  }
}

}  // namespace holoscan
}  // namespace nvidia
