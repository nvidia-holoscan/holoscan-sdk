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
#include "instrument_tip.hpp"

#include <string>

#include "opengl_utils.hpp"

namespace nvidia {
namespace holoscan {
namespace visualizer_tool_tracking {

gxf_result_t InstrumentTip::start() {
  if (num_tool_classes_ > MAX_TOOLS) {
    GXF_LOG_ERROR("Number of layers (%d) exceeds maximum number of layers (%d)", num_tool_classes_,
                  MAX_TOOLS);
    return GXF_FAILURE;
  }

  if (num_tool_classes_ > tool_tip_colors_.size()) {
    GXF_LOG_ERROR("Number of tools (%d) exceeds number of colors provided (%d)", num_tool_classes_,
                  tool_tip_colors_.size());
    return GXF_FAILURE;
  }

  for (auto color : tool_tip_colors_) {
    if (color.size() != 3) {
      GXF_LOG_ERROR("Tool colors must be 3 elements (RGB)");
      return GXF_FAILURE;
    }
  }

  // generate and setup vertex array object
  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);
  glBindBuffer(GL_ARRAY_BUFFER, frame_data_.position_);
  glVertexAttribPointer(0, num_tool_pos_components_, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ARRAY_BUFFER, frame_data_.confidence_);
  glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glBindVertexArray(0);

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
  for (size_t i = 0; i < num_tool_classes_; ++i) { glUniform3fv(i, 1, tool_tip_colors_[i].data()); }
  glUseProgram(0);

  GXF_LOG_INFO("Build GLSL shaders and program successfully");
  return GXF_SUCCESS;
}

gxf_result_t InstrumentTip::tick() {
  glUseProgram(program_);
  glBindVertexArray(vao_);
  glDrawArrays(GL_POINTS, 0, num_tool_classes_);

  return GXF_SUCCESS;
}

gxf_result_t InstrumentTip::stop() {
  return GXF_SUCCESS;
}

}  // namespace visualizer_tool_tracking
}  // namespace holoscan
}  // namespace nvidia
