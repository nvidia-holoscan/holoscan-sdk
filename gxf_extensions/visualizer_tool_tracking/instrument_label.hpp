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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_INSTRUMENT_LABEL_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_INSTRUMENT_LABEL_HPP_

#include <glad/glad.h>

#include <string>
#include <vector>

#include "gxf/std/codelet.hpp"

#include "frame_data.hpp"

struct NVGcontext;

namespace nvidia {
namespace holoscan {
namespace visualizer_tool_tracking {

struct InstrumentLabel {
  // owned externally
  const FrameData& frame_data_;
  // owned internally
  NVGcontext* nvg_ctx_ = nullptr;
  int vp_width_ = -1;
  int vp_height_ = -1;
  float vp_aspect_ratio_ = 0.0f;
  std::string label_sans_font_path_ = "UNDEFINED";
  std::string label_sans_bold_font_path_ = "UNDEFINED";
  uint32_t num_tool_classes_ = 0;
  uint32_t num_tool_pos_components_ = 2;
  std::vector<std::string> tool_labels_ = {};

  InstrumentLabel(const FrameData& frame_data) : frame_data_(frame_data) {}

  // private
  std::vector<std::string> tool_labels_or_numbers_ = {};

  InstrumentLabel(const InstrumentLabel&) = delete;
  InstrumentLabel& operator=(const InstrumentLabel&) = delete;

  gxf_result_t start();
  gxf_result_t tick();
  gxf_result_t stop();
};

}  // namespace visualizer_tool_tracking
}  // namespace holoscan
}  // namespace nvidia
#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_INSTRUMENT_LABEL_HPP_
