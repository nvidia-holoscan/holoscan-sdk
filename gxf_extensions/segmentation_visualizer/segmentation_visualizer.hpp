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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_SEGMENTATION_VISUALIZER_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_SEGMENTATION_VISUALIZER_HPP_

// clang-format off
#define GLFW_INCLUDE_NONE 1
#include <glad/glad.h>
#include <GLFW/glfw3.h>  // NOLINT(build/include_order)
// clang-format on

#include <string>
#include <vector>

#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/scheduling_terms.hpp"

// Forward declare from cuda_gl_interopt.h
struct cudaGraphicsResource;

namespace nvidia {
namespace holoscan {
namespace segmentation_visualizer {

/// @brief OpenGL Visualization Codelet that combines segmentation output overlaid on video input.
///
/// This Codelet performs visualization of a segmentation model alpha blended with the original
// video input.
/// The window is displayed with GLFW and uses CUDA/OpenGL interopt propagate CUDA Tensors from the
/// CUDA context to OpenGL.
/// The final class color is rendered using a GLSL fragment shader to remap the class index to an
/// RGBA LUT.
/// The LUT can be configured inside the YAML for common class colors. In some cases the alpha
/// component may be desired to be 0 (fully transparent) to prevent drawing over segmentation
/// classes that are uninteresting.
class Visualizer : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  gxf_result_t unregisterCudaResources();

  GLFWwindow* window_;
  std::vector<uint8_t> image_buffer_;
  std::vector<uint8_t> class_index_buffer_;
  GLuint textures_[2];
  std::vector<cudaGraphicsResource*> cuda_resources_;

  gxf::Parameter<gxf::Handle<gxf::Receiver>> class_index_in_;
  gxf::Parameter<int32_t> class_index_width_;
  gxf::Parameter<int32_t> class_index_height_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> image_in_;
  gxf::Parameter<int32_t> image_width_;
  gxf::Parameter<int32_t> image_height_;
  gxf::Parameter<std::vector<std::vector<float>>> class_color_lut_;
  gxf::Parameter<gxf::Handle<gxf::BooleanSchedulingTerm>> window_close_scheduling_term_;
};

}  // namespace segmentation_visualizer
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_SEGMENTATION_VISUALIZER_HPP_
