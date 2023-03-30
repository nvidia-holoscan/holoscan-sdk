/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>

#include "holoscan/core/gxf/gxf_operator.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to demosaic the input video stream.
 *
 * This wraps a GXF Codelet(`nvidia::holoscan::BayerDemosaic`).
 */
class BayerDemosaicOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(BayerDemosaicOp, holoscan::ops::GXFOperator)

  BayerDemosaicOp() = default;

  const char* gxf_typename() const override { return "nvidia::holoscan::BayerDemosaic"; }

  void setup(OperatorSpec& spec) override;

  void initialize() override;

 private:
  Parameter<holoscan::IOSpec*> receiver_;
  Parameter<holoscan::IOSpec*> transmitter_;
  Parameter<std::string> in_tensor_name_;
  Parameter<std::string> out_tensor_name_;
  Parameter<std::shared_ptr<Allocator>> pool_;
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_;
  Parameter<int> bayer_interp_mode_;
  Parameter<int> bayer_grid_pos_;
  Parameter<bool> generate_alpha_;
  Parameter<int> alpha_value_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_BAYER_DEMOSAIC_HPP */
