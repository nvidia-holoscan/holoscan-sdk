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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_OVERLAY_IMG_VIS_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_OVERLAY_IMG_VIS_HPP_

#include <glad/glad.h>

#include <string>
#include <vector>

#include "gxf/std/codelet.hpp"

#include "frame_data.hpp"

namespace nvidia {
namespace holoscan {
namespace visualizer_tool_tracking {

// Note: MAX_LAYERS should match the value used in the fragment shader.
constexpr size_t MAX_LAYERS = 64;

struct OverlayImageVis {
  // owned externally
  const FrameData& frame_data_;
  const GLuint& overlay_img_tex_;
  // owned internally
  GLuint vao_ = 0;
  GLuint sampler = 0;
  GLuint vertex_shader_ = 0, fragment_shader_ = 0, program_ = 0;
  std::string vertex_shader_file_path_;
  std::string fragment_shader_file_path_;
  size_t num_layers_;
  std::vector<std::vector<float>> layer_colors_;

  OverlayImageVis(const FrameData& frame_data, const GLuint& overlay_img_tex)
      : frame_data_(frame_data), overlay_img_tex_(overlay_img_tex) {}

  OverlayImageVis(const OverlayImageVis&) = delete;
  OverlayImageVis& operator=(const OverlayImageVis&) = delete;

  gxf_result_t start();
  gxf_result_t tick();
  gxf_result_t stop();
};
}  // namespace visualizer_tool_tracking
}  // namespace holoscan
}  // namespace nvidia
#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_OVERLAY_IMG_VIS_HPP_
