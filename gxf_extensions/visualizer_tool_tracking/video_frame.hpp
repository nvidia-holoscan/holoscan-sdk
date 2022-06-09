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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_VIDEO_FRAME_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_VIDEO_FRAME_HPP_
#include <glad/glad.h>

#include <string>

#include "gxf/std/codelet.hpp"

namespace nvidia {
namespace holoscan {
namespace visualizer_tool_tracking {

struct VideoFrame {
  // owned externally
  const GLuint& frame_tex_;
  // owned internally
  GLuint vao_ = 0;
  GLuint vertex_shader_ = 0, fragment_shader_ = 0, program_ = 0;
  GLuint sampler_ = 0;
  std::string vertex_shader_file_path_;
  std::string fragment_shader_file_path_;

  VideoFrame(const GLuint& frame_tex) : frame_tex_(frame_tex) {}

  VideoFrame(const VideoFrame&) = delete;
  VideoFrame& operator=(const VideoFrame&) = delete;

  gxf_result_t start();
  gxf_result_t tick();
  gxf_result_t stop();
};

}  // namespace visualizer_tool_tracking
}  // namespace holoscan
}  // namespace nvidia
#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_VIDEO_FRAME_HPP_
