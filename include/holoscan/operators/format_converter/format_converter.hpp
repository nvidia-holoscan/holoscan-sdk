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

#ifndef HOLOSCAN_OPERATORS_FORMAT_CONVERTER_FORMAT_CONVERTER_HPP
#define HOLOSCAN_OPERATORS_FORMAT_CONVERTER_FORMAT_CONVERTER_HPP

#include "../../core/gxf/gxf_operator.hpp"

#include <memory>
#include <string>
#include <vector>

namespace holoscan::ops {

/**
 * @brief Operator class to convert the data format of the input data.
 *
 * This wraps a GXF Codelet(`nvidia::holoscan::formatconverter::FormatConverter`).
 */
class FormatConverterOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(FormatConverterOp, holoscan::ops::GXFOperator)

  FormatConverterOp() = default;

  const char* gxf_typename() const override {
    return "nvidia::holoscan::formatconverter::FormatConverter";
  }

  // TODO(gbae): use std::expected
  void setup(OperatorSpec& spec) override;

 private:
  Parameter<holoscan::IOSpec*> in_;
  Parameter<holoscan::IOSpec*> out_;

  Parameter<std::string> in_tensor_name_;
  Parameter<std::string> out_tensor_name_;
  Parameter<float> scale_min_;
  Parameter<float> scale_max_;
  Parameter<uint8_t> alpha_value_;
  Parameter<int32_t> resize_width_;
  Parameter<int32_t> resize_height_;
  Parameter<int32_t> resize_mode_;
  Parameter<std::vector<int>> out_channel_order_;

  Parameter<std::shared_ptr<Allocator>> pool_;

  Parameter<std::string> in_dtype_str_;
  Parameter<std::string> out_dtype_str_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_FORMAT_CONVERTER_FORMAT_CONVERTER_HPP */
