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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_INSTRUMENT_TIP_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_INSTRUMENT_TIP_HPP_

#include <glad/glad.h>

#include <string>
#include <vector>

#include "gxf/std/codelet.hpp"

#include "frame_data.hpp"

namespace nvidia {
namespace holoscan {
namespace visualizer_tool_tracking {

// Note: MAX_TOOLS should match the value used in the fragment shader.
constexpr size_t MAX_TOOLS = 64;

struct InstrumentTip {
  // owned externally
  const FrameData& frame_data_;
  // owned internally
  GLuint vao_ = 0;
  GLuint vertex_shader_ = 0, fragment_shader_ = 0, program_ = 0;
  std::string vertex_shader_file_path_;
  std::string fragment_shader_file_path_;
  uint32_t num_tool_classes_ = 0;
  uint32_t num_tool_pos_components_ = 2;
  std::vector<std::vector<float>> tool_tip_colors_;

  explicit InstrumentTip(const FrameData& frame_data) : frame_data_(frame_data) {}

  InstrumentTip(const InstrumentTip&) = delete;
  InstrumentTip& operator=(const InstrumentTip&) = delete;

  gxf_result_t start();
  gxf_result_t tick();
  gxf_result_t stop();
};

}  // namespace visualizer_tool_tracking
}  // namespace holoscan
}  // namespace nvidia
#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_INSTRUMENT_TIP_HPP_
