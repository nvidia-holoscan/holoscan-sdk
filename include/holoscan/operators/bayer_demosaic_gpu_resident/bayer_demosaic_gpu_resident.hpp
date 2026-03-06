/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_BAYER_DEMOSAIC_GPU_RESIDENT_BAYER_DEMOSAIC_GPU_RESIDENT_HPP
#define HOLOSCAN_OPERATORS_BAYER_DEMOSAIC_GPU_RESIDENT_BAYER_DEMOSAIC_GPU_RESIDENT_HPP

#include <npp.h>

#include <cstdint>

#include "holoscan/core/gpu_resident_operator.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator_spec.hpp"

namespace holoscan::ops {

/**
 * @brief GPU-resident operator class to demosaic the input video stream.
 *
 * This operator is designed for GPU-resident execution mode where:
 * - All setup and configuration are done once during initialization
 * - The `compute` function is launched only once from the CPU
 * - Subsequent executions are run entirely on the GPU via CUDA graph capture
 *
 * The operator performs Bayer pattern demosaicing using NPP (NVIDIA Performance Primitives)
 * to convert a single-channel Bayer pattern image to a 3-channel RGB or 4-channel RGBA image.
 *
 * ==Device Inputs==
 *
 * - **in** : Device memory containing single-channel Bayer pattern data
 *   - The input must be 8-bit or 16-bit unsigned integer grayscale Bayer pattern data.
 *   - Memory size must be pre-configured based on image dimensions.
 *
 * ==Device Outputs==
 *
 * - **out** : Device memory containing demosaiced RGB/RGBA data
 *   - Output is 3-channel RGB if `generate_alpha` is false, 4-channel RGBA otherwise.
 *   - Data type matches input (8-bit or 16-bit unsigned integer).
 *
 * ==Configuration Parameters==
 *
 * - **width**: Image width in pixels. Must be an even number.
 * - **height**: Image height in pixels. Must be an even number.
 * - **pixel_type**: Pixel type (0 = uint8, 1 = uint16). Default: 0.
 * - **interpolation_mode**: The interpolation model for demosaicing. Default: 0 (NPPI_INTER_UNDEFINED).
 * - **bayer_grid_pos**: The Bayer grid position. Default: 2 (NPPI_BAYER_GBRG).
 *   - NPPI_BAYER_BGGR (0): Default registration position BGGR.
 *   - NPPI_BAYER_RGGB (1): Registration position RGGB.
 *   - NPPI_BAYER_GBRG (2): Registration position GBRG.
 *   - NPPI_BAYER_GRBG (3): Registration position GRBG.
 * - **generate_alpha**: Generate alpha channel. Default: false.
 * - **alpha_value**: Alpha value if `generate_alpha` is true. Default: 255.
 */
class BayerDemosaicGpuResidentOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(BayerDemosaicGpuResidentOp, holoscan::GPUResidentOperator)

  BayerDemosaicGpuResidentOp() = default;

  void setup(OperatorSpec& spec) override;

  void initialize() override;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  // Configuration parameters
  Parameter<int32_t> width_;
  Parameter<int32_t> height_;
  Parameter<int32_t> pixel_type_;  // 0 = uint8, 1 = uint16
  Parameter<int> bayer_interp_mode_;
  Parameter<int> bayer_grid_pos_;
  Parameter<bool> generate_alpha_;
  Parameter<int> alpha_value_;

  // Cached NPP parameters (set once during initialize)
  NppStreamContext npp_stream_ctx_{};
  NppiInterpolationMode npp_bayer_interp_mode_ = NPPI_INTER_UNDEFINED;
  NppiBayerGridPosition npp_bayer_grid_pos_ = NPPI_BAYER_GBRG;

  // Pre-computed values for compute
  int32_t in_line_step_ = 0;
  int32_t out_line_step_ = 0;
  NppiSize roi_size_{0, 0};
  NppiRect roi_rect_{0, 0, 0, 0};
  int16_t out_channels_ = 3;
  size_t element_size_ = 1;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_BAYER_DEMOSAIC_GPU_RESIDENT_BAYER_DEMOSAIC_GPU_RESIDENT_HPP */
