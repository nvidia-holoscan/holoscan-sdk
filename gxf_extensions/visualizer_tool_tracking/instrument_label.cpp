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
#include "instrument_label.hpp"

#include <nanovg.h>
#define NANOVG_GL3_IMPLEMENTATION
#include <nanovg_gl.h>
#include <nanovg_gl_utils.h>

#include <string>

#include "opengl_utils.hpp"

namespace nvidia {
namespace holoscan {
namespace visualizer_tool_tracking {

gxf_result_t InstrumentLabel::start() {
  nvg_ctx_ = nvgCreateGL3(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
  if (nvg_ctx_ == nullptr) {
    GXF_LOG_ERROR("Could not init NANOVG.\n");
    return GXF_FAILURE;
  }

  // ----------------------------------------------------------------------

  int font;
  font = nvgCreateFont(nvg_ctx_, "sans", label_sans_font_path_.c_str());
  if (font == -1) {
    GXF_LOG_ERROR("Could not add font regular: %s\n", label_sans_font_path_.c_str());
    return GXF_FAILURE;
  }
  font = nvgCreateFont(nvg_ctx_, "sans-bold", label_sans_bold_font_path_.c_str());
  if (font == -1) {
    GXF_LOG_ERROR("Could not add font bold: %s \n", label_sans_bold_font_path_.c_str());
    return GXF_FAILURE;
  }

  // Add numbers as default values if labels are missing
  tool_labels_or_numbers_ = tool_labels_;
  for (uint32_t i = tool_labels_or_numbers_.size(); i < num_tool_classes_; ++i) {
    tool_labels_or_numbers_.push_back(std::to_string(i));
  }

  return GXF_SUCCESS;
}

gxf_result_t InstrumentLabel::tick(float width, float height) {
  nvgBeginFrame(nvg_ctx_, width, height, width / height);

  nvgFontSize(nvg_ctx_, 25.0f);
  nvgFontFace(nvg_ctx_, "sans-bold");
  nvgTextAlign(nvg_ctx_, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);

  for (uint32_t i = 0; i != num_tool_classes_; ++i) {
    // skip non-present tools
    if (frame_data_.confidence_host_[i] < 0.5) { continue; }

    const char* label_text = tool_labels_or_numbers_[i].c_str();
    float x = frame_data_.position_host_[i * num_tool_pos_components_] * width + 40;
    float y = frame_data_.position_host_[i * num_tool_pos_components_ + 1] * height + 40;
    nvgFillColor(nvg_ctx_, nvgRGBAf(1.0f, 1.0f, 1.0f, 0.9f));
    nvgText(nvg_ctx_, x, y, label_text, NULL);
  }

  nvgEndFrame(nvg_ctx_);

  return GXF_SUCCESS;
}

gxf_result_t InstrumentLabel::stop() {
  return GXF_SUCCESS;
}

}  // namespace visualizer_tool_tracking
}  // namespace holoscan
}  // namespace nvidia
