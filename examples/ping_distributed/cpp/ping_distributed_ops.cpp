/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ping_distributed_ops.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "gxf/std/allocator.hpp"
#include "gxf/std/tensor.hpp"

namespace holoscan::ops {

void PingTensorTxOp::initialize() {
  // Set up prerequisite parameters before calling Operator::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'allocator'
  auto has_allocator = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "allocator"); });
  // Create the allocator if there is no argument provided.
  if (has_allocator == args().end()) {
    allocator_ = frag->make_resource<UnboundedAllocator>("allocator");
    add_arg(allocator_.get());
  }
  Operator::initialize();
}

void PingTensorTxOp::setup(OperatorSpec& spec) {
  spec.output<holoscan::TensorMap>("out");

  spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor output.");
  spec.param(tensor_on_gpu_,
             "tensor_on_gpu",
             "Tensor on GPU",
             "Flag indicating that the tensor transmitted should be on the device.",
             true);
  spec.param(batch_size_,
             "batch_size",
             "batch size",
             "Size of the batch dimension (default: 0). The tensor shape will be "
             "([batch], rows, [columns], [channels]) where [] around a dimension indicates that "
             "it is only present if the corresponding parameter has a size > 0."
             "If 0, no batch dimension will be present.",
             static_cast<int32_t>(0));
  spec.param(rows_,
             "rows",
             "number of rows",
             "Number of rows (default: 32), must be >= 1.",
             static_cast<int32_t>(32));
  spec.param(columns_,
             "columns",
             "number of columns",
             "Number of columns (default: 64). If 0, no column dimension will be present.",
             static_cast<int32_t>(64));
  spec.param(
      channels_,
      "channels",
      "channels",
      "Number of channels (default: 0). If 0, no channel dimension will be present. (default: 0)",
      static_cast<int32_t>(0));
  spec.param(data_type_,
             "data_type",
             "data type for the tensor elements",
             "must be one of {'int8_t', 'int16_t', 'int32_t', 'int64_t', 'uint8_t', 'uint16_t',"
             "'uint32_t', 'uint64_t', 'float', 'double', 'complex<float>', 'complex<double>'}",
             std::string{"uint8_t"});
  spec.param(tensor_name_,
             "tensor_name",
             "output tensor name",
             "output tensor name (default: out)",
             std::string{"out"});
}

nvidia::gxf::PrimitiveType PingTensorTxOp::primitive_type(const std::string& data_type) {
  HOLOSCAN_LOG_INFO("PingTensorTxOp data type = {}", data_type);
  if (data_type == "int8_t") {
    return nvidia::gxf::PrimitiveType::kInt8;
  } else if (data_type == "int16_t") {
    return nvidia::gxf::PrimitiveType::kInt16;
  } else if (data_type == "int32_t") {
    return nvidia::gxf::PrimitiveType::kInt32;
  } else if (data_type == "int64_t") {
    return nvidia::gxf::PrimitiveType::kInt64;
  } else if (data_type == "uint8_t") {
    return nvidia::gxf::PrimitiveType::kUnsigned8;
  } else if (data_type == "uint16_t") {
    return nvidia::gxf::PrimitiveType::kUnsigned16;
  } else if (data_type == "uint32_t") {
    return nvidia::gxf::PrimitiveType::kUnsigned32;
  } else if (data_type == "uint64_t") {
    return nvidia::gxf::PrimitiveType::kUnsigned64;
  } else if (data_type == "float") {
    return nvidia::gxf::PrimitiveType::kFloat32;
  } else if (data_type == "double") {
    return nvidia::gxf::PrimitiveType::kFloat64;
  } else if (data_type == "complex<float>") {
    return nvidia::gxf::PrimitiveType::kComplex64;
  } else if (data_type == "complex<double>") {
    return nvidia::gxf::PrimitiveType::kComplex128;
  }
  throw std::runtime_error(std::string("Unrecognized data_type: ") + data_type);
}

void PingTensorTxOp::compute(InputContext&, OutputContext& op_output, ExecutionContext& context) {
  // the type of out_message is TensorMap
  TensorMap out_message;

  auto gxf_context = context.context();
  auto frag = fragment();

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(gxf_context, allocator_->gxf_cid());

  auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>();

  // Define the dimensions for the CUDA memory (64 x 32, uint8).
  int batch_size = batch_size_.get();
  int rows = rows_.get();
  int columns = columns_.get();
  int channels = channels_.get();
  auto dtype = element_type();

  std::vector<int32_t> shape_vec;
  if (batch_size > 0) { shape_vec.push_back(batch_size); }
  shape_vec.push_back(rows);
  if (columns > 0) { shape_vec.push_back(columns); }
  if (channels > 0) { shape_vec.push_back(channels); }
  auto tensor_shape = nvidia::gxf::Shape{shape_vec};

  const uint64_t bytes_per_element = nvidia::gxf::PrimitiveTypeSize(dtype);
  auto strides = nvidia::gxf::ComputeTrivialStrides(tensor_shape, bytes_per_element);
  nvidia::gxf::MemoryStorageType storage_type;
  if (tensor_on_gpu_) {
    storage_type = nvidia::gxf::MemoryStorageType::kDevice;
  } else {
    storage_type = nvidia::gxf::MemoryStorageType::kSystem;
  }

  // allocate a tensor of the specified shape and data type
  auto result = gxf_tensor->reshapeCustom(
      tensor_shape, dtype, bytes_per_element, strides, storage_type, allocator.value());
  if (!result) { HOLOSCAN_LOG_ERROR("failed to generate tensor"); }

  // Create Holoscan tensor
  auto maybe_dl_ctx = (*gxf_tensor).toDLManagedTensorContext();
  if (!maybe_dl_ctx) {
    HOLOSCAN_LOG_ERROR(
        "failed to get std::shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
  }
  std::shared_ptr<Tensor> holoscan_tensor = std::make_shared<Tensor>(maybe_dl_ctx.value());

  // insert tensor into the TensorMap
  out_message.insert({tensor_name_.get().c_str(), holoscan_tensor});

  op_output.emit(out_message);
}

void PingTensorRxOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("in");
}

void PingTensorRxOp::compute(InputContext& op_input, OutputContext&, ExecutionContext&) {
  auto in_message = op_input.receive<holoscan::TensorMap>("in").value();
  TensorMap out_message;
  for (auto& [key, tensor] : in_message) {  // Process with 'tensor' here.
    HOLOSCAN_LOG_INFO("message {}: Tensor key: '{}', shape: ({})",
                      count_++,
                      key,
                      fmt::join(tensor->shape(), ", "));
  }
}
}  // namespace holoscan::ops
