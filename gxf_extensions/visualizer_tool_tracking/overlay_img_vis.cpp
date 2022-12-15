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
#include "overlay_img_vis.hpp"

#include <string>

#include "opengl_utils.hpp"

namespace nvidia {
namespace holoscan {
namespace visualizer_tool_tracking {

gxf_result_t OverlayImageVis::start() {
  if (num_layers_ > MAX_LAYERS) {
    GXF_LOG_ERROR("Number of layers (%d) exceeds maximum number of layers (%d)", num_layers_,
                  MAX_LAYERS);
    return GXF_FAILURE;
  }

  if (num_layers_ > layer_colors_.size()) {
    GXF_LOG_ERROR("Number of layers (%d) exceeds number of colors provided (%d)", num_layers_,
                  layer_colors_.size());
    return GXF_FAILURE;
  }

  for (auto color : layer_colors_) {
    if (color.size() != 3) {
      GXF_LOG_ERROR("Layer colors must be 3 elements (RGB)");
      return GXF_FAILURE;
    }
  }

  glGenVertexArrays(1, &vao_);

  glCreateSamplers(1, &sampler);
  glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  if (!createGLSLShaderFromFile(GL_VERTEX_SHADER, vertex_shader_, vertex_shader_file_path_)) {
    GXF_LOG_ERROR("Failed to create GLSL vertex shader");
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

  // Initialize constant uniforms.
  glUseProgram(program_);
  glUniform1ui(0, num_layers_);
  for (size_t i = 0; i < num_layers_; ++i) { glUniform3fv(1 + i, 1, layer_colors_[i].data()); }
  glUseProgram(0);

  GXF_LOG_INFO("Build GLSL shaders and program successfully");

  return GXF_SUCCESS;
}

gxf_result_t OverlayImageVis::tick() {
  glActiveTexture(GL_TEXTURE0);
  glBindSampler(0, sampler);
  glBindTexture(GL_TEXTURE_2D_ARRAY, overlay_img_tex_);

  glUseProgram(program_);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, frame_data_.confidence_);

  glBindVertexArray(vao_);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 3);

  return GXF_SUCCESS;
}

gxf_result_t OverlayImageVis::stop() {
  return GXF_SUCCESS;
}

}  // namespace visualizer_tool_tracking
}  // namespace holoscan
}  // namespace nvidia
