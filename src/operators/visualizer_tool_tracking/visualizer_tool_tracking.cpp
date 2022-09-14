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

#include "holoscan/operators/visualizer_tool_tracking/visualizer_tool_tracking.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

namespace holoscan::ops {

void ToolTrackingVizOp::setup(OperatorSpec& spec) {
  constexpr int32_t DEFAULT_SRC_WIDTH = 640;
  constexpr int32_t DEFAULT_SRC_HEIGHT = 480;
  constexpr int16_t DEFAULT_SRC_CHANNELS = 3;
  constexpr uint8_t DEFAULT_SRC_BYTES_PER_PIXEL = 1;
  // 12 qualitative classes color scheme from colorbrewer2
  static const std::vector<std::vector<float>> DEFAULT_COLORS = {{0.12f, 0.47f, 0.71f},
                                                                 {0.20f, 0.63f, 0.17f},
                                                                 {0.89f, 0.10f, 0.11f},
                                                                 {1.00f, 0.50f, 0.00f},
                                                                 {0.42f, 0.24f, 0.60f},
                                                                 {0.69f, 0.35f, 0.16f},
                                                                 {0.65f, 0.81f, 0.89f},
                                                                 {0.70f, 0.87f, 0.54f},
                                                                 {0.98f, 0.60f, 0.60f},
                                                                 {0.99f, 0.75f, 0.44f},
                                                                 {0.79f, 0.70f, 0.84f},
                                                                 {1.00f, 1.00f, 0.60f}};

  auto& in_source_video = spec.input<::gxf::Entity>("source_video");
  auto& in_tensor = spec.input<::gxf::Entity>("tensor");

  spec.param(in_, "in", "Input", "List of input channels", {&in_source_video, &in_tensor});

  spec.param(videoframe_vertex_shader_path_,
             "videoframe_vertex_shader_path",
             "Videoframe GLSL Vertex Shader File Path",
             "Path to vertex shader to be loaded");
  spec.param(videoframe_fragment_shader_path_,
             "videoframe_fragment_shader_path",
             "Videoframe GLSL Fragment Shader File Path",
             "Path to fragment shader to be loaded");

  spec.param(tooltip_vertex_shader_path_,
             "tooltip_vertex_shader_path",
             "Tool tip GLSL Vertex Shader File Path",
             "Path to vertex shader to be loaded");
  spec.param(tooltip_fragment_shader_path_,
             "tooltip_fragment_shader_path",
             "Tool tip GLSL Fragment Shader File Path",
             "Path to fragment shader to be loaded");
  spec.param(
      num_tool_classes_, "num_tool_classes", "Tool Classes", "Number of different tool classes");
  spec.param(num_tool_pos_components_,
             "num_tool_pos_components",
             "Position Components",
             "Number of components of the tool position vector",
             2);
  spec.param(tool_tip_colors_,
             "tool_tip_colors",
             "Tool Tip Colors",
             "Color of the tool tips, a list of RGB values with components between 0 and 1",
             DEFAULT_COLORS);

  spec.param(overlay_img_vertex_shader_path_,
             "overlay_img_vertex_shader_path",
             "Overlay Image GLSL Vertex Shader File Path",
             "Path to vertex shader to be loaded");
  spec.param(overlay_img_fragment_shader_path_,
             "overlay_img_fragment_shader_path",
             "Overlay Image GLSL Fragment Shader File Path",
             "Path to fragment shader to be loaded");
  spec.param(
      overlay_img_width_, "overlay_img_width", "Overlay Image Width", "Width of overlay image");
  spec.param(
      overlay_img_height_, "overlay_img_height", "Overlay Image Height", "Height of overlay image");
  spec.param(overlay_img_channels_,
             "overlay_img_channels",
             "Number of Overlay Image Channels",
             "Number of Overlay Image Channels");
  spec.param(overlay_img_layers_,
             "overlay_img_layers",
             "Number of Overlay Image Layers",
             "Number of Overlay Image Layers");
  spec.param(overlay_img_colors_,
             "overlay_img_colors",
             "Overlay Image Layer Colors",
             "Color of the image overlays, a list of RGB values with components between 0 and 1",
             DEFAULT_COLORS);

  spec.param(
      tool_labels_,
      "tool_labels",
      "Tool Labels",
      "List of tool names.",
      {});  // Default handled in instrument_label to dynamically adjust for the number of tools
  spec.param(label_sans_font_path_,
             "label_sans_font_path",
             "File path for sans font for displaying tool name",
             "Path for sans font to be loaded");
  spec.param(label_sans_bold_font_path_,
             "label_sans_bold_font_path",
             "File path for sans bold font for displaying tool name",
             "Path for sans bold font to be loaded");

  spec.param(in_tensor_names_,
             "in_tensor_names",
             "Input Tensor Names",
             "Names of input tensors.",
             {std::string("")});
  spec.param(in_width_, "in_width", "InputWidth", "Width of the image.", DEFAULT_SRC_WIDTH);
  spec.param(in_height_, "in_height", "InputHeight", "Height of the image.", DEFAULT_SRC_HEIGHT);
  spec.param(
      in_channels_, "in_channels", "InputChannels", "Number of channels.", DEFAULT_SRC_CHANNELS);
  spec.param(in_bytes_per_pixel_,
             "in_bytes_per_pixel",
             "InputBytesPerPixel",
             "Number of bytes per pexel of the image.",
             DEFAULT_SRC_BYTES_PER_PIXEL);
  spec.param(alpha_value_,
             "alpha_value",
             "Alpha value",
             "Alpha value that can be used when converting RGB888 to RGBA8888.",
             static_cast<uint8_t>(255));
  spec.param(pool_, "pool", "Pool", "Pool to allocate the output message.");
  spec.param(
      window_close_scheduling_term_,
      "window_close_scheduling_term",
      "WindowCloseSchedulingTerm",
      "BooleanSchedulingTerm to stop the codelet from ticking after all messages are published.");
}

void ToolTrackingVizOp::initialize() {
  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();
  auto window_close_scheduling_term =
      frag->make_condition<holoscan::BooleanCondition>("window_close_scheduling_term");
  add_arg(Arg("window_close_scheduling_term") = window_close_scheduling_term);

  GXFOperator::initialize();
}

}  // namespace holoscan::ops
