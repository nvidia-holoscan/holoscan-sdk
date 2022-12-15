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

#include "holoscan/operators/visualizer_icardio/visualizer_icardio.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

namespace holoscan::ops {

void VisualizerICardioOp::setup(OperatorSpec& spec) {
  auto& out_tensor_1 = spec.output<gxf::Entity>("keypoints");
  auto& out_tensor_2 = spec.output<gxf::Entity>("keyarea_1");
  auto& out_tensor_3 = spec.output<gxf::Entity>("keyarea_2");
  auto& out_tensor_4 = spec.output<gxf::Entity>("keyarea_3");
  auto& out_tensor_5 = spec.output<gxf::Entity>("keyarea_4");
  auto& out_tensor_6 = spec.output<gxf::Entity>("keyarea_5");
  auto& out_tensor_7 = spec.output<gxf::Entity>("lines");
  auto& out_tensor_8 = spec.output<gxf::Entity>("logo");

  spec.param(
      in_tensor_names_, "in_tensor_names", "Input Tensors", "Input tensors", {std::string("")});
  spec.param(
      out_tensor_names_, "out_tensor_names", "Output Tensors", "Output tensors", {std::string("")});
  spec.param(input_on_cuda_, "input_on_cuda", "Input buffer on CUDA", "", false);
  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
  spec.param(receivers_, "receivers", "Receivers", "List of receivers", {});
  spec.param(transmitters_,
             "transmitters",
             "Transmitters",
             "List of transmitters",
             {&out_tensor_1,
              &out_tensor_2,
              &out_tensor_3,
              &out_tensor_4,
              &out_tensor_5,
              &out_tensor_6,
              &out_tensor_7,
              &out_tensor_8});
}

}  // namespace holoscan::ops
