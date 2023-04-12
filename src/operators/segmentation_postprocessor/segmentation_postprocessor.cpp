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

#include "holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp"

#include <limits>
#include <string>
#include <utility>

#include "gxf/std/tensor.hpp"

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

using holoscan::ops::segmentation_postprocessor::cuda_postprocess;
using holoscan::ops::segmentation_postprocessor::DataFormat;
using holoscan::ops::segmentation_postprocessor::NetworkOutputType;
using holoscan::ops::segmentation_postprocessor::output_type_t;
using holoscan::ops::segmentation_postprocessor::Shape;

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

  cuda_stream_handler_.defineParams(spec);

  // TODO (gbae): spec object holds an information about errors
  // TODO (gbae): incorporate std::expected to not throw exceptions
}

void SegmentationPostprocessorOp::compute(InputContext& op_input, OutputContext& op_output,
                                          ExecutionContext& context) {
  constexpr size_t kMaxChannelCount = std::numeric_limits<output_type_t>::max();

  // Process input message
  // The type of `in_message` is 'holoscan::gxf::Entity'.
  auto in_message = op_input.receive<gxf::Entity>("in_tensor");

  // if (!in_message || in_message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }

  const std::string in_tensor_name = in_tensor_name_.get();

  // Get tensor attached to the message
  // The type of `maybe_tensor` is 'std::shared_ptr<holoscan::Tensor>'.
  auto maybe_tensor = in_message.get<Tensor>(in_tensor_name.c_str());
  if (!maybe_tensor) {
    maybe_tensor = in_message.get<Tensor>();
    if (!maybe_tensor) {
      throw std::runtime_error(fmt::format("Tensor '{}' not found in message", in_tensor_name));
    }
  }
  auto in_tensor = maybe_tensor;

  // get the CUDA stream from the input message
  gxf_result_t stream_handler_result =
      cuda_stream_handler_.fromMessage(context.context(), in_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }

  segmentation_postprocessor::Shape shape = {};
  switch (data_format_value_) {
    case DataFormat::kHWC: {
      shape.height = in_tensor->shape()[0];
      shape.width = in_tensor->shape()[1];
      shape.channels = in_tensor->shape()[2];
    } break;
    case DataFormat::kNCHW: {
      shape.channels = in_tensor->shape()[1];
      shape.height = in_tensor->shape()[2];
      shape.width = in_tensor->shape()[3];
    } break;
    case DataFormat::kNHWC: {
      shape.height = in_tensor->shape()[1];
      shape.width = in_tensor->shape()[2];
      shape.channels = in_tensor->shape()[3];
    } break;
  }

  if (static_cast<size_t>(shape.channels) > kMaxChannelCount) {
    throw std::runtime_error(fmt::format(
        "Input channel count larger than allowed: {} > {}", shape.channels, kMaxChannelCount));
  }

  // Create a new message (nvidia::gxf::Entity)
  auto out_message = nvidia::gxf::Entity::New(context.context());

  auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>("out_tensor");
  if (!out_tensor) { throw std::runtime_error("Failed to allocate output tensor"); }

  // Allocate and convert output buffer on the device.
  nvidia::gxf::Shape output_shape{shape.height, shape.width, 1};

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                       allocator_.get()->gxf_cid());
  out_tensor.value()->reshape<uint8_t>(
      output_shape, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
  if (!out_tensor.value()->pointer()) {
    throw std::runtime_error("Failed to allocate output tensor buffer.");
  }

  const float* in_tensor_data = static_cast<float*>(in_tensor->data());

  nvidia::gxf::Expected<uint8_t*> out_tensor_data = out_tensor.value()->data<uint8_t>();
  if (!out_tensor_data) { throw std::runtime_error("Failed to get out tensor data!"); }

  cuda_postprocess(network_output_type_value_,
                   data_format_value_,
                   shape,
                   in_tensor_data,
                   out_tensor_data.value(),
                   cuda_stream_handler_.getCudaStream(context.context()));

  // pass the CUDA stream to the output message
  stream_handler_result = cuda_stream_handler_.toMessage(out_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
  }

  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result);
}

void SegmentationPostprocessorOp::start() {
  const std::string network_output_type = network_output_type_.get();
  if (network_output_type == "sigmoid") {
    network_output_type_value_ = NetworkOutputType::kSigmoid;
  } else if (network_output_type == "softmax") {
    network_output_type_value_ = NetworkOutputType::kSoftmax;
  } else {
    throw std::runtime_error(
        fmt::format("Unsupported network output type {}", network_output_type));
  }

  const std::string data_format = data_format_.get();
  if (data_format == "nchw") {
    data_format_value_ = DataFormat::kNCHW;
  } else if (data_format == "hwc") {
    data_format_value_ = DataFormat::kHWC;
  } else if (data_format == "nhwc") {
    data_format_value_ = DataFormat::kNHWC;
  } else {
    throw std::runtime_error(fmt::format("Unsupported data format type {}", data_format));
  }
}

}  // namespace holoscan::ops
