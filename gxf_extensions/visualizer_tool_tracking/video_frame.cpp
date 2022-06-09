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
#include "video_frame.hpp"

#include <string>

#include "opengl_utils.hpp"

namespace nvidia {
namespace holoscan {
namespace visualizer_tool_tracking {

gxf_result_t VideoFrame::start() {
  glGenVertexArrays(1, &vao_);

  glCreateSamplers(1, &sampler_);
  glSamplerParameteri(sampler_, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glSamplerParameteri(sampler_, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glSamplerParameteri(sampler_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glSamplerParameteri(sampler_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  if (!createGLSLShaderFromFile(GL_VERTEX_SHADER, vertex_shader_, vertex_shader_file_path_)) {
    GXF_LOG_ERROR("Failed to create GLSLvertex shader");
    return GXF_FAILURE;
  }

  if (!createGLSLShaderFromFile(GL_FRAGMENT_SHADER, fragment_shader_, fragment_shader_file_path_)) {
    GXF_LOG_ERROR("Failed to create GLSL fragment shader");
    return GXF_FAILURE;
  }

  if (!linkGLSLProgram(vertex_shader_, fragment_shader_, program_)) {
    GXF_LOG_ERROR("Failed to link GLSL program.");
    return GXF_FAILURE;
  }

  GXF_LOG_INFO("Build GLSL shaders and program succesfully");
  return GXF_SUCCESS;
}

gxf_result_t VideoFrame::tick() {
  glActiveTexture(GL_TEXTURE0);
  glBindSampler(0, sampler_);
  glBindTexture(GL_TEXTURE_2D, frame_tex_);
  glUseProgram(program_);
  glBindVertexArray(vao_);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 3);
  glBindTexture(GL_TEXTURE_2D, 0);

  return GXF_SUCCESS;
}

gxf_result_t VideoFrame::stop() {
  return GXF_SUCCESS;
}

}  // namespace visualizer_tool_tracking
}  // namespace holoscan
}  // namespace nvidia