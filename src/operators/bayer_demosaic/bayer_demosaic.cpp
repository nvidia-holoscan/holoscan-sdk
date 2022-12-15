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

#include "holoscan/operators/bayer_demosaic/bayer_demosaic.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

namespace holoscan::ops {

void BayerDemosaicOp::setup(OperatorSpec& spec) {
  auto& receiver = spec.input<gxf::Entity>("receiver");
  auto& transmitter = spec.output<gxf::Entity>("transmitter");

  spec.param(receiver_, "receiver", "Entity receiver", "Receiver channel", &receiver);
  spec.param(
      transmitter_, "transmitter", "Entity transmitter", "Transmitter channel", &transmitter);
  spec.param(in_tensor_name_,
             "in_tensor_name",
             "InputTensorName",
             "Name of the input tensor",
             std::string(""));
  spec.param(out_tensor_name_,
             "out_tensor_name",
             "OutputTensorName",
             "Name of the output tensor",
             std::string(""));
  spec.param(pool_, "pool", "Pool", "Pool to allocate the output message.");
  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "CUDA Stream Pool",
             "CUDA Stream pool to create CUDA streams.");
  spec.param(bayer_interp_mode_,
             "interpolation_mode",
             "Interpolation used for demosaicing",
             "The interpolation model to be used for demosaicing (default UNDEFINED). Values "
             "available at: "
             "https://docs.nvidia.com/cuda/npp/"
             "group__typedefs__npp.html#ga2b58ebd329141d560aa4367f1708f191",
             0);
  spec.param(bayer_grid_pos_,
             "bayer_grid_pos",
             "Bayer grid position",
             "The Bayer grid position (default GBRG). Values available at: "
             "https://docs.nvidia.com/cuda/npp/"
             "group__typedefs__npp.html#ga5597309d6766fb2dffe155990d915ecb",
             2);
  spec.param(generate_alpha_,
             "generate_alpha",
             "Generate alpha channel",
             "Generate alpha channel.",
             false);
  spec.param(alpha_value_,
             "alpha_value",
             "Alpha value to be generated",
             "Alpha value to be generated if `generate_alpha` is set to `true` (default `255`).",
             255);
}

void BayerDemosaicOp::initialize() {
  holoscan::ops::GXFOperator::initialize();
}

}  // namespace holoscan::ops
