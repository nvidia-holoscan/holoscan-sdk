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

#include "holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "holoscan/core/resources/gxf/allocator.hpp"

namespace holoscan::ops {

void SegmentationPostprocessorOp::setup(OperatorSpec& spec) {
  auto& in_tensor = spec.input<gxf::Entity>("in_tensor");
  auto& out_tensor = spec.output<gxf::Entity>("out_tensor");

  spec.param(in_, "in", "Input", "Input channel.", &in_tensor);
  spec.param(out_, "out", "Output", "Output channel.", &out_tensor);

  spec.param(in_tensor_name_,
             "in_tensor_name",
             "InputTensorName",
             "Name of the input tensor.",
             std::string(""));
  spec.param(network_output_type_,
             "network_output_type",
             "NetworkOutputType",
             "Network output type.",
             std::string("softmax"));
  spec.param(data_format_,
             "data_format",
             "DataFormat",
             "Data format of network output.",
             std::string("hwc"));
  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");

  // TODO (gbae): spec object holds an information about errors
  // TODO (gbae): incorporate std::expected to not throw exceptions
}

}  // namespace holoscan::ops
