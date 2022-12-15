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

#ifndef HOLOSCAN_OPERATORS_SEGMENTATION_POSTPROCESSOR_POSTPROCESSOR_HPP
#define HOLOSCAN_OPERATORS_SEGMENTATION_POSTPROCESSOR_POSTPROCESSOR_HPP

#include <memory>
#include <string>
#include <utility>

#include "../../core/gxf/gxf_operator.hpp"

namespace holoscan::ops {

class SegmentationPostprocessorOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(SegmentationPostprocessorOp, holoscan::ops::GXFOperator)

  SegmentationPostprocessorOp() = default;

  const char* gxf_typename() const override {
    return "nvidia::holoscan::segmentation_postprocessor::Postprocessor";
  }

  // TODO(gbae): use std::expected
  void setup(OperatorSpec& spec) override;

 private:
  Parameter<holoscan::IOSpec*> in_;
  Parameter<holoscan::IOSpec*> out_;

  Parameter<std::shared_ptr<Allocator>> allocator_;

  Parameter<std::string> in_tensor_name_;
  Parameter<std::string> network_output_type_;
  Parameter<std::string> data_format_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_SEGMENTATION_POSTPROCESSOR_POSTPROCESSOR_HPP */
