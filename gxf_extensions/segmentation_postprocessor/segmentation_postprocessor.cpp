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
#include "segmentation_postprocessor.hpp"

#include <string>
#include <utility>

#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"

#define CUDA_TRY(stmt)                                                                          \
  ({                                                                                            \
    cudaError_t _holoscan_cuda_err = stmt;                                                      \
    if (cudaSuccess != _holoscan_cuda_err) {                                                    \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).", #stmt, \
                    __LINE__, __FILE__, cudaGetErrorString(_holoscan_cuda_err),                 \
                    _holoscan_cuda_err);                                                        \
    }                                                                                           \
    _holoscan_cuda_err;                                                                         \
  })

namespace nvidia {
namespace holoscan {
namespace segmentation_postprocessor {

gxf_result_t Postprocessor::start() {
  const std::string& network_output_type_name = network_output_type_.get();
  if (network_output_type_name == "sigmoid") {
    network_output_type_value_ = NetworkOutputType::kSigmoid;
  } else if (network_output_type_name == "softmax") {
    network_output_type_value_ = NetworkOutputType::kSoftmax;
  } else {
    GXF_LOG_ERROR("Unsupported network type %s", network_output_type_name.c_str());
    return GXF_FAILURE;
  }
  const std::string& data_format_name = data_format_.get();
  if (data_format_name == "nchw") {
    data_format_value_ = DataFormat::kNCHW;
  } else if (data_format_name == "hwc") {
    data_format_value_ = DataFormat::kHWC;
  } else if (data_format_name == "nhwc") {
    data_format_value_ = DataFormat::kNHWC;
  } else {
    GXF_LOG_ERROR("Unsupported format type %s", data_format_name.c_str());
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t Postprocessor::tick() {
  // Process input message
  const auto in_message = in_->receive();
  if (!in_message || in_message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }

  // Get tensor attached to the message
  auto maybe_tensor = in_message.value().get<gxf::Tensor>(in_tensor_name_.get().c_str());
  if (!maybe_tensor) {
    maybe_tensor = in_message.value().get<gxf::Tensor>();
    if (!maybe_tensor) {
      GXF_LOG_ERROR("Tensor '%s' not found in message.", in_tensor_name_.get().c_str());
      return GXF_FAILURE;
    }
  }
  gxf::Handle<gxf::Tensor> in_tensor = maybe_tensor.value();

  Shape shape = {};
  switch (data_format_value_) {
    case DataFormat::kHWC: {
      shape.height = in_tensor->shape().dimension(0);
      shape.width = in_tensor->shape().dimension(1);
      shape.channels = in_tensor->shape().dimension(2);
    } break;
    case DataFormat::kNCHW: {
      shape.channels = in_tensor->shape().dimension(1);
      shape.height = in_tensor->shape().dimension(2);
      shape.width = in_tensor->shape().dimension(3);
    } break;
    case DataFormat::kNHWC: {
      shape.height = in_tensor->shape().dimension(1);
      shape.width = in_tensor->shape().dimension(2);
      shape.channels = in_tensor->shape().dimension(3);
    } break;
  }

  if (shape.channels > kMaxChannelCount) {
    GXF_LOG_ERROR("Input channel count larger than allowed: %d > %d", shape.channels,
                  kMaxChannelCount);
    return GXF_FAILURE;
  }

  auto out_message = gxf::Entity::New(context());
  if (!out_message) {
    GXF_LOG_ERROR("Failed to allocate message");
    return GXF_FAILURE;
  }

  auto out_tensor = out_message.value().add<gxf::Tensor>("out_tensor");
  if (!out_tensor) {
    GXF_LOG_ERROR("Failed to allocate output tensor");
    return GXF_FAILURE;
  }

  // Allocate and convert output buffer on the device.
  gxf::Shape output_shape{shape.height, shape.width, 1};
  out_tensor.value()->reshape<uint8_t>(output_shape, gxf::MemoryStorageType::kDevice, allocator_);
  if (!out_tensor.value()->pointer()) {
    GXF_LOG_ERROR("Failed to allocate output tensor buffer.");
    return GXF_FAILURE;
  }

  gxf::Expected<const float*> in_tensor_data = in_tensor->data<float>();
  if (!in_tensor_data) {
    GXF_LOG_ERROR("Failed to get in tensor data!");
    return GXF_FAILURE;
  }
  gxf::Expected<uint8_t*> out_tensor_data = out_tensor.value()->data<uint8_t>();
  if (!out_tensor_data) {
    GXF_LOG_ERROR("Failed to get out tensor data!");
    return GXF_FAILURE;
  }

  cuda_postprocess(network_output_type_value_, data_format_value_, shape, in_tensor_data.value(),
                   out_tensor_data.value());

  const auto result = out_->publish(std::move(out_message.value()));
  if (!result) {
    GXF_LOG_ERROR("Failed to publish output!");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t Postprocessor::stop() {
  return GXF_SUCCESS;
}

gxf_result_t Postprocessor::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(in_, "in", "Input", "Input channel.");
  result &= registrar->parameter(in_tensor_name_, "in_tensor_name", "InputTensorName",
                                 "Name of the input tensor.", std::string(""));
  result &= registrar->parameter(network_output_type_, "network_output_type", "NetworkOutputType",
                                 "Network output type.", std::string("softmax"));
  result &= registrar->parameter(out_, "out", "Output", "Output channel.");
  result &= registrar->parameter(data_format_, "data_format", "DataFormat",
                                 "Data format of network output", std::string("hwc"));
  result &= registrar->parameter(allocator_, "allocator", "Allocator", "Output Allocator");
  return gxf::ToResultCode(result);
}

}  // namespace segmentation_postprocessor
}  // namespace holoscan
}  // namespace nvidia
