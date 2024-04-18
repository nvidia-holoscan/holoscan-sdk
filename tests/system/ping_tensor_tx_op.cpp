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

#include "ping_tensor_tx_op.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#define CUDA_TRY(stmt)                                                                        \
  ({                                                                                          \
    cudaError_t _holoscan_cuda_err = stmt;                                                    \
    if (cudaSuccess != _holoscan_cuda_err) {                                                  \
      HOLOSCAN_LOG_ERROR("CUDA Runtime call {} in line {} of file {} failed with '{}' ({}).", \
                         #stmt,                                                               \
                         __LINE__,                                                            \
                         __FILE__,                                                            \
                         cudaGetErrorString(_holoscan_cuda_err),                              \
                         _holoscan_cuda_err);                                                 \
    }                                                                                         \
    _holoscan_cuda_err;                                                                       \
  })

namespace holoscan {
namespace ops {

void PingTensorTxOp::setup(OperatorSpec& spec) {
  spec.output<TensorMap>("out");

  spec.param(
      rows_, "rows", "number of rows", "number of rows (default: 64)", static_cast<int32_t>(64));
  spec.param(columns_,
             "columns",
             "number of columns",
             "number of columns (default: 32)",
             static_cast<int32_t>(32));
  spec.param(channels_,
             "channels",
             "channels",
             "Number of channels. If 0, no channel dimension will be present. (default: 0)",
             static_cast<int32_t>(0));
  spec.param(tensor_name_,
             "tensor_name",
             "output tensor name",
             "output tensor name (default: tensor)",
             std::string{"tensor"});
}

void PingTensorTxOp::compute(InputContext&, OutputContext& op_output, ExecutionContext& context) {
  // Define the dimensions for the CUDA memory (64 x 32, uint8).
  int rows = rows_.get();
  int columns = columns_.get();
  int channels = channels_.get();
  // keep element type as kUnsigned8 for use with BayerDemosaicOp
  nvidia::gxf::PrimitiveType element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
  int element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
  nvidia::gxf::Shape shape;
  size_t nbytes;
  if (channels == 0) {
    shape = nvidia::gxf::Shape{rows, columns};
    nbytes = rows * columns * element_size;
  } else {
    shape = nvidia::gxf::Shape{rows, columns, channels};
    nbytes = rows * columns * channels * element_size;
  }

  // Create a shared pointer for the CUDA memory with a custom deleter.
  auto pointer = std::shared_ptr<void*>(new void*, [](void** pointer) {
    if (pointer != nullptr) {
      if (*pointer != nullptr) { CUDA_TRY(cudaFree(*pointer)); }
      delete pointer;
    }
  });

  // Allocate and initialize the CUDA memory.
  CUDA_TRY(cudaMalloc(pointer.get(), nbytes));
  std::vector<uint8_t> data(nbytes);
  for (size_t index = 0; index < data.size(); ++index) { data[index] = (index_ + index) % 256; }
  CUDA_TRY(cudaMemcpy(*pointer, data.data(), nbytes, cudaMemcpyKind::cudaMemcpyHostToDevice));

  // Holoscan Tensor doesn't support direct memory allocation.
  // Thus, create an Entity and use GXF tensor to wrap the CUDA memory.
  auto out_message = nvidia::gxf::Entity::New(context.context());
  auto gxf_tensor = out_message.value().add<nvidia::gxf::Tensor>(tensor_name_.get().c_str());
  gxf_tensor.value()->wrapMemory(shape,
                                 element_type,
                                 element_size,
                                 nvidia::gxf::ComputeTrivialStrides(shape, element_size),
                                 nvidia::gxf::MemoryStorageType::kDevice,
                                 *pointer,
                                 [orig_pointer = pointer](void*) mutable {
                                   orig_pointer.reset();  // decrement ref count
                                   return nvidia::gxf::Success;
                                 });
  HOLOSCAN_LOG_INFO("Tx message rows:{}, columns:{}, channels:{}", rows, columns, channels);
  HOLOSCAN_LOG_INFO("Tx message value - index:{}, size:{} ", index_, gxf_tensor.value()->size());

  // Emit the tensor.
  op_output.emit(out_message.value(), "out");

  index_++;
}

}  // namespace ops
}  // namespace holoscan
