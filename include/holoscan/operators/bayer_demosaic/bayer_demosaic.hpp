/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_BAYER_DEMOSAIC_HPP
#define HOLOSCAN_OPERATORS_BAYER_DEMOSAIC_HPP

#include <npp.h>

#include <memory>
#include <string>

#include "holoscan/core/io_context.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to demosaic the input video stream.
 *
 * **Named inputs:**
 *     - *receiver*: `nvidia::gxf::Tensor` or `nvidia::gxf::VideoBuffer`
 *         - The input video frame to process. If the input is a VideoBuffer it must be an 8-bit
 *         unsigned grayscale video (nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY). The video
 *         buffer may be in either host or device memory (a host->device copy is performed if
 *         needed). If a video buffer is not found, the input port message is searched for a tensor
 *         with the name specified by `in_tensor_name`. This must be a device tensor in either
 *         8-bit or 16-bit unsigned integer format.
 *
 * **Named outputs:**
 *     - *transmitter*: `nvidia::gxf::Tensor`
 *         - The output video frame after demosaicing. This will be a 3-channel RGB image if
 *         `alpha_value` is true, otherwise it will be a 4-channel RGBA image. The data type
 *         will be either 8-bit or 16-bit unsigned integer (matching the bit depth of the input).
 *         The name of the tensor that is output is controlled by `out_tensor_name`.
 */
class BayerDemosaicOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BayerDemosaicOp)

  BayerDemosaicOp() = default;

  void setup(OperatorSpec& spec) override;

  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  Parameter<holoscan::IOSpec*> receiver_;
  Parameter<holoscan::IOSpec*> transmitter_;
  Parameter<std::string> in_tensor_name_;
  Parameter<std::string> out_tensor_name_;
  Parameter<std::shared_ptr<Allocator>> pool_;
  Parameter<int> bayer_interp_mode_;
  Parameter<int> bayer_grid_pos_;
  Parameter<bool> generate_alpha_;
  Parameter<int> alpha_value_;

  NppStreamContext npp_stream_ctx_{};

  NppiInterpolationMode npp_bayer_interp_mode_;
  NppiBayerGridPosition npp_bayer_grid_pos_;

  nvidia::gxf::MemoryBuffer device_scratch_buffer_;

  CudaStreamHandler cuda_stream_handler_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_BAYER_DEMOSAIC_HPP */
