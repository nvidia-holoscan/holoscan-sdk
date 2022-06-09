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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_VISUALIZER_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_VISUALIZER_HPP_
// clang-format off
#define GLFW_INCLUDE_NONE 1
#include <glad/glad.h>
#include <GLFW/glfw3.h>  // NOLINT(build/include_order)
// clang-format on

#include <string>
#include <vector>

#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/scheduling_terms.hpp"

#include "frame_data.hpp"
#include "instrument_label.hpp"
#include "instrument_tip.hpp"
#include "overlay_img_vis.hpp"
#include "video_frame.hpp"

struct cudaGraphicsResource;

namespace nvidia {
namespace holoscan {
namespace visualizer_tool_tracking {

/// @brief Visualization Codelet for Instrument Tracking and Overlay
///
/// This visualizer uses OpenGL/CUDA interopt for quickly passing data from the output of inference
/// to an OpenGL context for rendering.
/// The visualizer renders the location and text of an instrument and optionally displays the
/// model's confidence score.
class Sink : public gxf::Codelet {
 public:
  Sink();

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

  void onKeyCallback(int key, int scancode, int action, int mods);
  void onFramebufferSizeCallback(int width, int height);

 private:
  // GLFW members and callback funds
  GLFWwindow* window_ = nullptr;
  // GL viewport
  int vp_width_ = 0;
  int vp_height_ = 0;
  float vp_aspect_ratio_ = 0;
  bool vp_changed_ = true;
  FrameData frame_data_;

  cudaGraphicsResource* cuda_confidence_resource_ = nullptr;
  cudaGraphicsResource* cuda_position_resource_ = nullptr;

  // Videoframe Vis related members
  // --------------------------------------------------------------------

  bool use_cuda_opengl_interop_ = true;
  GLuint video_frame_tex_ = 0;
  VideoFrame video_frame_vis_;
  cudaGraphicsResource* cuda_video_frame_tex_resource_ = nullptr;
  std::vector<unsigned char> video_frame_buffer_host_;

  gxf::Parameter<int32_t> in_width_;
  gxf::Parameter<int32_t> in_height_;
  gxf::Parameter<int16_t> in_channels_;
  gxf::Parameter<uint8_t> in_bytes_per_pixel_;
  gxf::Parameter<uint8_t> alpha_value_;

  gxf::Parameter<std::string> videoframe_vertex_shader_path_;
  gxf::Parameter<std::string> videoframe_fragment_shader_path_;

  // Tooltip Vis related members
  // --------------------------------------------------------------------


  bool enable_tool_tip_vis_ = true;
  InstrumentTip tooltip_vis_;
  gxf::Parameter<std::string> tooltip_vertex_shader_path_;
  gxf::Parameter<std::string> tooltip_fragment_shader_path_;
  // Those two below apply to Tool Name (instrument labels) also
  gxf::Parameter<int32_t> num_tool_classes_;
  gxf::Parameter<int32_t> num_tool_pos_components_;
  gxf::Parameter<std::vector<std::vector<float>>> tool_tip_colors_;

  // Tool Name Vis related members
  // --------------------------------------------------------------------

  bool enable_tool_labels_ = true;
  InstrumentLabel label_vis_;
  gxf::Parameter<std::vector<std::string>> tool_labels_;
  gxf::Parameter<std::string> label_sans_font_path_;
  gxf::Parameter<std::string> label_sans_bold_font_path_;

  // Overlay Img Vis related members
  // --------------------------------------------------------------------

  bool enable_overlay_img_vis_ = true;

  gxf::Parameter<int32_t> overlay_img_width_;
  gxf::Parameter<int32_t> overlay_img_height_;
  gxf::Parameter<int32_t> overlay_img_layers_;
  gxf::Parameter<int32_t> overlay_img_channels_;
  gxf::Parameter<std::vector<std::vector<float>>> overlay_img_colors_;

  GLuint overlay_img_tex_ = 0;
  std::vector<float> overlay_img_buffer_host_;
  std::vector<float> overlay_img_layered_host_;
  OverlayImageVis overlay_img_vis;

  cudaGraphicsResource* cuda_overlay_img_tex_resource_ = nullptr;
  gxf::Parameter<std::string> overlay_img_vertex_shader_path_;
  gxf::Parameter<std::string> overlay_img_fragment_shader_path_;

  // GFX Parameters
  // --------------------------------------------------------------------


  gxf::Parameter<std::vector<gxf::Handle<gxf::Receiver>>> in_;
  gxf::Parameter<std::vector<std::string>> in_tensor_names_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  gxf::Parameter<gxf::Handle<gxf::BooleanSchedulingTerm>> window_close_scheduling_term_;
};

}  // namespace visualizer_tool_tracking
}  // namespace holoscan
}  // namespace nvidia
#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_VISUALIZER_HPP_
