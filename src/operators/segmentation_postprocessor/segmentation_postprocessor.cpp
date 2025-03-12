/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include <string>
#include <utility>

#include "gxf/std/tensor.hpp"

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

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
  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "CUDA Stream Pool",
             "Instance of gxf::CudaStreamPool.",
             ParameterFlag::kOptional);

  // TODO (gbae): spec object holds an information about errors
  // TODO (gbae): incorporate std::expected to not throw exceptions
}

void SegmentationPostprocessorOp::compute(InputContext& op_input, OutputContext& op_output,
                                          ExecutionContext& context) {
  constexpr size_t kMaxChannelCount = std::numeric_limits<output_type_t>::max();

  // Process input message
  // The type of `in_message` is 'holoscan::gxf::Entity'.
  auto maybe_tensormap = op_input.receive<TensorMap>("in_tensor");
  if (!maybe_tensormap) {
    std::string err_msg =
        fmt::format("Operator '{}' failed to receive input message on port 'in_tensor': {}",
                    name_,
                    maybe_tensormap.error().what());
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  auto& tensormap = maybe_tensormap.value();

  const std::string in_tensor_name = in_tensor_name_.get();

  // Get tensor attached to the message
  // The type of `maybe_tensor` is 'std::shared_ptr<holoscan::Tensor>'.
  if (tensormap.empty()) { throw std::runtime_error("No tensors found in received message"); }
  auto tensor_it = tensormap.find(in_tensor_name);
  std::shared_ptr<Tensor> in_tensor;
  if (tensor_it != tensormap.end()) {
    in_tensor = tensor_it->second;
  } else {
    HOLOSCAN_LOG_ERROR(
        fmt::format("Specified tensor with in_tensor_name='{}' not found in message. First tensor "
                    "found ('{}') will be used instead.",
                    in_tensor_name,
                    tensormap.begin()->first));
    in_tensor = tensormap.begin()->second;
  }

  // validate tensor format
  DLDevice dev = in_tensor->device();
  if (dev.device_type != kDLCUDA && dev.device_type != kDLCUDAHost) {
    throw std::runtime_error("Input tensor must be in CUDA device or pinned host memory.");
  }
  DLDataType dtype = in_tensor->dtype();
  if (dtype.code != kDLFloat || dtype.bits != 32) {
    throw std::runtime_error("Input tensor must be of type float32.");
  }
  if (!in_tensor->is_contiguous()) {
    throw std::runtime_error("Input tensor must have row-major memory layout.");
  }

  // Get the CUDA stream from the input message if present, otherwise generate one.
  // This stream will also be transmitted on the "tensor" output port.
  cudaStream_t cuda_stream = op_input.receive_cuda_stream("in_tensor", true);

  segmentation_postprocessor::Shape shape = {};
  switch (data_format_value_) {
    case DataFormat::kHWC: {
      shape.height = in_tensor->shape()[0];
      shape.width = in_tensor->shape()[1];
      shape.channels = in_tensor->shape()[2];
    } break;
    case DataFormat::kNCHW: {
      if (in_tensor->shape()[0] != 1) { throw std::runtime_error("Batch size must be 1"); }
      shape.channels = in_tensor->shape()[1];
      shape.height = in_tensor->shape()[2];
      shape.width = in_tensor->shape()[3];
    } break;
    case DataFormat::kNHWC: {
      if (in_tensor->shape()[0] != 1) { throw std::runtime_error("Batch size must be 1"); }
      shape.height = in_tensor->shape()[1];
      shape.width = in_tensor->shape()[2];
      shape.channels = in_tensor->shape()[3];
    } break;
  }

  if (static_cast<size_t>(shape.channels) > kMaxChannelCount) {
    throw std::runtime_error(fmt::format(
        "Input channel count larger than allowed: {} > {}", shape.channels, kMaxChannelCount));
  }

  if ((network_output_type_value_ == NetworkOutputType::kSigmoid) && (shape.channels > 1)) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      HOLOSCAN_LOG_WARN(
          "Multi-channel input provided, but network_output_type is 'sigmoid'. Only the first "
          "channel will be used.");
    }
  }

  // Create a new message (nvidia::gxf::Entity)
  auto out_message = nvidia::gxf::Entity::New(context.context());

  auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>("out_tensor");
  if (!out_tensor) { throw std::runtime_error("Failed to allocate output tensor"); }

  // Allocate and convert output buffer on the device.
  nvidia::gxf::Shape output_shape{shape.height, shape.width, 1};

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
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
                   cuda_stream);

  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result, "out_tensor");
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
