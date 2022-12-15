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

#ifndef HOLOSCAN_OPERATORS_AJA_SOURCE_AJA_SOURCE_HPP
#define HOLOSCAN_OPERATORS_AJA_SOURCE_AJA_SOURCE_HPP

#include <string>
#include <utility>
#include <vector>

#include "../../core/gxf/gxf_operator.hpp"
#include "./ntv2channel.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to get the video stream from AJA capture card.
 *
 * This wraps a GXF Codelet(`nvidia::holoscan::AJASource`).
 */
class AJASourceOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(AJASourceOp, holoscan::ops::GXFOperator)

  AJASourceOp() = default;

  const char* gxf_typename() const override { return "nvidia::holoscan::AJASource"; }

  void setup(OperatorSpec& spec) override;

  void initialize() override;

 private:
  Parameter<holoscan::IOSpec*> video_buffer_output_;
  Parameter<std::string> device_specifier_;
  Parameter<NTV2Channel> channel_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<uint32_t> framerate_;
  Parameter<bool> use_rdma_;
  Parameter<bool> enable_overlay_;
  Parameter<NTV2Channel> overlay_channel_;
  Parameter<bool> overlay_rdma_;
  Parameter<holoscan::IOSpec*> overlay_buffer_input_;
  Parameter<holoscan::IOSpec*> overlay_buffer_output_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_AJA_SOURCE_AJA_SOURCE_HPP */
