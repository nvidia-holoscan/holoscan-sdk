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

#include "holoscan/operators/format_converter/format_converter.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

namespace holoscan::ops {

void FormatConverterOp::setup(OperatorSpec& spec) {
  auto& in_tensor = spec.input<gxf::Entity>("source_video");
  auto& out_tensor = spec.output<gxf::Entity>("tensor");

  spec.param(in_, "in", "Input", "Input channel.", &in_tensor);
  spec.param(out_, "out", "Output", "Output channel.", &out_tensor);

  spec.param(in_tensor_name_,
             "in_tensor_name",
             "InputTensorName",
             "Name of the input tensor.",
             std::string(""));
  spec.param(in_dtype_str_, "in_dtype", "InputDataType", "Source data type.", std::string(""));
  spec.param(out_tensor_name_,
             "out_tensor_name",
             "OutputTensorName",
             "Name of the output tensor.",
             std::string(""));
  spec.param(out_dtype_str_, "out_dtype", "OutputDataType", "Destination data type.");
  spec.param(scale_min_, "scale_min", "Scale min", "Minimum value of the scale.", 0.f);
  spec.param(scale_max_, "scale_max", "Scale max", "Maximum value of the scale.", 1.f);
  spec.param(alpha_value_,
             "alpha_value",
             "Alpha value",
             "Alpha value that can be used to fill the alpha channel when "
             "converting RGB888 to RGBA8888.",
             static_cast<uint8_t>(255));
  spec.param(resize_width_,
             "resize_width",
             "Resize width",
             "Width for resize. No actions if this value is zero.",
             0);
  spec.param(resize_height_,
             "resize_height",
             "Resize height",
             "Height for resize. No actions if this value is zero.",
             0);
  spec.param(resize_mode_,
             "resize_mode",
             "Resize mode",
             "Mode for resize. 4 (NPPI_INTER_CUBIC) if this value is zero.",
             0);
  spec.param(out_channel_order_,
             "out_channel_order",
             "Output channel order",
             "Host memory integer array describing how channel values are permutated.",
             std::vector<int>{});

  spec.param(pool_, "pool", "Pool", "Pool to allocate the output message.");

  // TODO (gbae): spec object holds an information about errors
  // TODO (gbae): incorporate std::expected to not throw exceptions
}

}  // namespace holoscan::ops
