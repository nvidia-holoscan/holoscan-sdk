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

#include "holoscan/operators/tool_tracking_postprocessor/tool_tracking_postprocessor.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

namespace holoscan::ops {

void ToolTrackingPostprocessorOp::setup(OperatorSpec& spec) {
  constexpr float DEFAULT_MIN_PROB = 0.5f;
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

  auto& in_tensor = spec.input<gxf::Entity>("in");
  auto& out_tensor = spec.output<gxf::Entity>("out");

  spec.param(in_, "in", "Input", "Input channel.", &in_tensor);
  spec.param(out_, "out", "Output", "Output channel.", &out_tensor);

  spec.param(min_prob_, "min_prob", "Minimum probability", "Minimum probability.",
             DEFAULT_MIN_PROB);

  spec.param(overlay_img_colors_,
             "overlay_img_colors",
             "Overlay Image Layer Colors",
             "Color of the image overlays, a list of RGB values with components between 0 and 1",
             DEFAULT_COLORS);

  spec.param(host_allocator_, "host_allocator", "Allocator", "Output Allocator");
  spec.param(device_allocator_, "device_allocator", "Allocator", "Output Allocator");
}

}  // namespace holoscan::ops
