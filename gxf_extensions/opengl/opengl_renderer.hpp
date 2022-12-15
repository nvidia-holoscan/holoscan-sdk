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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_OPENGL_RENDERER_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_OPENGL_RENDERER_HPP_

// clang-format off
#define GLFW_INCLUDE_NONE 1
#include <glad/glad.h>
#include <GLFW/glfw3.h>  // NOLINT(build/include_order)
// clang-format on

#include "gxf/multimedia/video.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/scheduling_terms.hpp"

struct cudaGraphicsResource;

namespace nvidia {
namespace holoscan {

/// @brief Visualization codelet using OpenGL and GLFW for display.
///
/// Provides a codelet that renders a VideoBuffer to an OpenGL texture leveraging
/// OpenGL/CUDA interoperation.
/// The texture is mapped to a full screen quad with simple GLSL shaders to render the texture.
class OpenGLRenderer : public gxf::Codelet {
 public:
  OpenGLRenderer();

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  void processInput();
  GLuint createShader(GLenum type, const char* src);
  GLenum getTextureFormat(gxf::VideoFormat format);

  gxf::Parameter<gxf::Handle<gxf::Receiver>> signal_;
  gxf::Parameter<unsigned int> width_;
  gxf::Parameter<unsigned int> height_;
  gxf::Parameter<gxf::Handle<gxf::BooleanSchedulingTerm>> window_close_scheduling_term_;

  GLFWwindow* window_;
  GLuint gl_texture_;
  uint32_t texture_width_;
  uint32_t texture_height_;
  GLenum texture_format_;

  cudaGraphicsResource* cuda_resource_;
};

}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_OPENGL_RENDERER_HPP_
