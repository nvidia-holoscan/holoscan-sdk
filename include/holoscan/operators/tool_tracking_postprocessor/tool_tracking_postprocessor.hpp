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

#ifndef HOLOSCAN_OPERATORS_TOOL_TRACKING_POSTPROCESSOR_TOOL_TRACKING_POSTPROCESSOR_HPP
#define HOLOSCAN_OPERATORS_TOOL_TRACKING_POSTPROCESSOR_TOOL_TRACKING_POSTPROCESSOR_HPP

#include <memory>
#include <string>
#include <vector>

#include "../../core/gxf/gxf_operator.hpp"
namespace holoscan::ops {

class ToolTrackingPostprocessorOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ToolTrackingPostprocessorOp, holoscan::ops::GXFOperator)

  ToolTrackingPostprocessorOp() = default;

  const char* gxf_typename() const override {
    return "nvidia::holoscan::tool_tracking_postprocessor::Postprocessor";
  }

  void setup(OperatorSpec& spec) override;

 private:
  Parameter<holoscan::IOSpec*> in_;
  Parameter<holoscan::IOSpec*> out_;

  Parameter<float> min_prob_;
  Parameter<std::vector<std::vector<float>>> overlay_img_colors_;

  Parameter<std::shared_ptr<Allocator>> host_allocator_;
  Parameter<std::shared_ptr<Allocator>> device_allocator_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_TOOL_TRACKING_POSTPROCESSOR_TOOL_TRACKING_POSTPROCESSOR_HPP */
