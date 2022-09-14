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

#ifndef HOLOSCAN_OPERATORS_VISUALIZER_TOOL_TRACKING_VISUALIZER_TOOL_TRACKING_HPP
#define HOLOSCAN_OPERATORS_VISUALIZER_TOOL_TRACKING_VISUALIZER_TOOL_TRACKING_HPP

#include "../../core/gxf/gxf_operator.hpp"

#include <memory>
#include <string>
#include <vector>

namespace holoscan::ops {

/**
 * @brief Operator class to visualize the tool tracking results.
 *
 * This wraps a GXF Codelet(`nvidia::holoscan::visualizer_tool_tracking::Sink`).
 */
class ToolTrackingVizOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ToolTrackingVizOp, holoscan::ops::GXFOperator)

  ToolTrackingVizOp() = default;

  const char* gxf_typename() const override {
    return "nvidia::holoscan::visualizer_tool_tracking::Sink";
  }

  void setup(OperatorSpec& spec) override;

  void initialize() override;

 private:
  Parameter<int32_t> in_width_;
  Parameter<int32_t> in_height_;
  Parameter<int16_t> in_channels_;
  Parameter<uint8_t> in_bytes_per_pixel_;
  Parameter<uint8_t> alpha_value_;

  Parameter<std::string> videoframe_vertex_shader_path_;
  Parameter<std::string> videoframe_fragment_shader_path_;

  Parameter<std::string> tooltip_vertex_shader_path_;
  Parameter<std::string> tooltip_fragment_shader_path_;
  Parameter<int32_t> num_tool_classes_;
  Parameter<int32_t> num_tool_pos_components_;
  Parameter<std::vector<std::vector<float>>> tool_tip_colors_;

  Parameter<std::vector<std::string>> tool_labels_;
  Parameter<std::string> label_sans_font_path_;
  Parameter<std::string> label_sans_bold_font_path_;

  Parameter<int32_t> overlay_img_width_;
  Parameter<int32_t> overlay_img_height_;
  Parameter<int32_t> overlay_img_layers_;
  Parameter<int32_t> overlay_img_channels_;
  Parameter<std::vector<std::vector<float>>> overlay_img_colors_;

  Parameter<std::string> overlay_img_vertex_shader_path_;
  Parameter<std::string> overlay_img_fragment_shader_path_;

  Parameter<std::vector<IOSpec*>> in_;
  Parameter<std::vector<std::string>> in_tensor_names_;
  Parameter<std::shared_ptr<Allocator>> pool_;
  Parameter<std::shared_ptr<Condition>> window_close_scheduling_term_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_VISUALIZER_TOOL_TRACKING_VISUALIZER_TOOL_TRACKING_HPP */
